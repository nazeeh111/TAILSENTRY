"""Frozen known-truth suite and cost-accounted algorithm adapters."""
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtri,log_ndtr
from scipy.stats import norm,t as student_t,chi2,qmc
from .models import normal_max,balls,QuadraticModel
from .references import normal_max_survival,normal_paired_survival,ball_survival
from .baselines import run_baseline
from .engine import estimate
from .envelope import estimate_envelope,build_cover
from .conditional import probability
from .uncertainty import summarize_observations

METHODS=['mc','antithetic','sobol','fixed_is','ce_single','ce_mixture','subset',
         'conditional','conditional_sobol','raw_mixture','projected_mixture',
         'envelope_raw','envelope_projected_fixed','envelope_projected','envelope_unoptimized']


def student_ball_truth(threshold,centers,radii,nu):
    from scipy.stats import ncx2
    centers=np.array(centers);radii=np.array(radii);d=centers.shape[1]
    # Probability integral transform avoids truncating a heavy-tail scale.
    def integrand(u):
        s=chi2.ppf(u,nu)/nu
        return float(ncx2.cdf(np.maximum(radii*radii-threshold,0)*s,d,np.sum(centers*centers,axis=1)*s).sum())
    value,error=quad(integrand,0,1,epsabs=1e-12,epsrel=2e-9,limit=400,points=[1e-6,1e-4,.01,.5,.99])
    return value,error


def make_case(name):
    if name=='single_1e3':
        m=normal_max(1);threshold=norm.isf(1e-3);truth=1e-3
    elif name=='paired4_1e4':
        m=normal_max(4,paired=True)
        threshold=brentq(lambda t:normal_paired_survival(t,4)-1e-4,2,8);truth=normal_paired_survival(threshold,4)
    elif name=='imbalanced4_1e5':
        off=np.array([0.,-.6,-1.4,-2.2]);m=normal_max(4,offsets=off)
        threshold=brentq(lambda t:normal_max_survival(t,4,off)-1e-5,2,8);truth=normal_max_survival(threshold,4,off)
    elif name=='correlated4_1e5':
        rho=.8;a=np.zeros((4,5));a[:,0]=np.sqrt(rho);a[:,1:]=np.sqrt(1-rho)*np.eye(4)
        m=QuadraticModel(np.zeros((4,5,5)),a,np.zeros(4),name)
        def survival(t):
            return quad(lambda z:-np.expm1(4*log_ndtr((t-np.sqrt(rho)*z)/np.sqrt(1-rho)))*norm.pdf(z),-12,12,epsabs=1e-13,epsrel=1e-10,limit=200)[0]
        threshold=brentq(lambda t:survival(t)-1e-5,2,8);truth=survival(threshold)
    elif name in ('compact2_1e4','compact4_1e5'):
        d=2 if name.startswith('compact2') else 4;p=1e-4 if d==2 else 1e-5
        axes=np.array([[1.,0.],[-1.,0.]]) if d==2 else np.vstack((np.eye(d)[0],-np.eye(d)[0],np.eye(d)[1]))
        r=np.full(len(axes),.8 if d==2 else 1.1)
        length=brentq(lambda x:ball_survival(0,axes*x,r)-p,3,12)
        centers=axes*length;m=balls(centers,r);threshold=0.;truth=ball_survival(0,centers,r)
    elif name=='student_balls_1e4':
        nu=5.;axes=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]])
        r=np.ones(3)
        length=brentq(lambda x:student_ball_truth(0,axes*x,r,nu)[0]-1e-4,4,25,xtol=1e-9)
        centers=axes*length;m=balls(centers,r,scale_df=nu);threshold=0.;truth=student_ball_truth(0,centers,r,nu)[0]
    elif name=='delta_gamma_1e4':
        d=8;u=np.ones(d)/np.sqrt(d);v=np.r_[np.ones(d//2),-np.ones(d//2)]/np.sqrt(d)
        lam=.25;delta=.6;cor=.4
        a=delta*(cor*u+np.sqrt(1-cor*cor)*v)
        m=QuadraticModel((lam*np.outer(u,u))[None],a[None],np.zeros(1),name)
        def survival(t):
            return quad(lambda z:norm.sf((t-lam*z*z-delta*cor*z)/(delta*np.sqrt(1-cor*cor)))*norm.pdf(z),-14,14,epsabs=1e-13,epsrel=1e-10,limit=200)[0]
        threshold=brentq(lambda t:survival(t)-1e-4,1,15);truth=survival(threshold)
    else:raise ValueError('unknown case '+name)
    return m,float(threshold),float(truth)


CASES=['single_1e3','paired4_1e4','imbalanced4_1e5','correlated4_1e5','compact2_1e4','compact4_1e5','student_balls_1e4','delta_gamma_1e4']


def _conditional_sobol(model,t,budget,seed):
    start=time.perf_counter();cover=build_cover(model,t)
    metric=np.einsum('k,ki,kj->ij',cover['masses'],cover['normals'],cover['normals'])
    _,vectors=np.linalg.eigh(metric);e=vectors[:,-1]
    if model.scale_df is not None:e=np.r_[e,0.]
    reps=8;power=int(np.floor(np.log2(budget//reps)));means=[]
    for child in np.random.SeedSequence(seed).spawn(reps):
        uniform=qmc.Sobol(model.dimension,scramble=True,seed=np.random.default_rng(child)).random_base2(power)
        z=ndtri(np.clip(uniform,np.nextafter(0.,1.),np.nextafter(1.,0.)))
        v=z-(z@e)[:,None]*e
        a,b,c=model.conditional_coefficients(v,e)
        means.append(float(probability(a,b,c,t).mean()))
    means=np.array(means);point=float(means.mean());se=float(means.std(ddof=1)/np.sqrt(reps));r=student_t.ppf(.975,reps-1)*se
    guaranteed=summarize_observations(means,1.)['interval']
    return dict(estimate=point,standard_error=se,interval=guaranteed,asymptotic_interval=[max(0.,point-r),min(1.,point+r)],
                tail_ess=None,seconds=time.perf_counter()-start,
                point_evaluations=0,conditional_evaluations=reps*2**power,confirmation_samples=reps*2**power,
                interval_kind='Bernstein across independent scramble means; asymptotic Student also reported')


def run_method(model,threshold,budget,seed,method):
    model.reset_counts()
    if method.startswith('envelope'):
        r=estimate_envelope(model,threshold,budget,seed,project=method!='envelope_raw',
                            select=method not in ('envelope_raw','envelope_projected_fixed'),
                            optimize_cover=method!='envelope_unoptimized')
    elif method in ('conditional','raw_mixture','projected_mixture'):
        r=estimate(model,threshold,budget,seed,method=method)
    elif method=='conditional_sobol':r=_conditional_sobol(model,threshold,budget,seed)
    else:
        out=run_baseline(model,method,budget,seed,threshold)
        r=dict(estimate=out['estimate'],standard_error=out['standard_error'],
               interval=out['confidence_interval'] if method=='mc' else None,
               asymptotic_interval=out['confidence_interval'] if method!='mc' else out['confidence_interval'],
               interval_kind=out['diagnostics']['uncertainty'],tail_ess=out['diagnostics'].get('tail_ess'),
               diagnostics=out['diagnostics'],seconds=out['seconds'],point_evaluations=out['evaluations'],conditional_evaluations=0)
    return r

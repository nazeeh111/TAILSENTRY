"""Structure-certified union sampling and its exact projection conditionalization.

Established union importance sampling + conditional Monte Carlo. The explicit
quadratic cover construction is inspectable; research novelty is unestablished.
"""
import time
import numpy as np
from scipy.special import log_ndtr,logsumexp,ndtri,ndtr,ndtri_exp
from scipy.stats import t as student_t,chi2,norm
from scipy.optimize import minimize
from .conditional import probability,tail_moment
from .uncertainty import summarize_observations,precision_status


def build_cover(model,threshold,optimize=True):
    """Prove {L>t} subset of a finite union {u_j·X>cut_j}.

    Max-affine: exact. Spherical compact quadratic: tangent supporting cap.
    General: discard negative eigenvalues, then allocate positive thresholds
    across linear and positive-square terms; pigeonhole principle gives cover.
    """
    d=model.factor_dimension;directions=[];cutoffs=[];proof=[]
    sf=norm.sf if model.scale_df is None else lambda x:student_t.sf(x,model.scale_df)
    for B,a,c in zip(model.quadratic,model.linear,model.constant):
        rescale=max(float(np.abs(B).max()),float(np.abs(a).max()),abs(float(c)),abs(float(threshold)))
        rescale=rescale if rescale>0 else 1.
        B,a,c=B/rescale,a/rescale,c/rescale
        local_threshold=threshold/rescale
        lam=-B[0,0]
        if lam>0 and np.array_equal(B,-lam*np.eye(d)):
            center=a/(2*lam);top=c+(a@a)/(4*lam)
            if top<=local_threshold:proof.append('empty spherical component');continue
            radius=np.sqrt((top-local_threshold)/lam);length=np.linalg.norm(center)
            if length>radius:
                directions.append(center/length);cutoffs.append(length-radius)
                proof.append('ball implies projection beyond center norm minus radius');continue
        eigen,vectors=np.linalg.eigh(B)
        positive=eigen>0
        linear_norm=np.linalg.norm(a)
        if np.all(B==0) and linear_norm>0:
            directions.append(a/linear_norm);cutoffs.append((local_threshold-c)/linear_norm)
            proof.append('exact affine halfspace');continue
        if local_threshold<=c:
            directions.extend([np.eye(d)[0],-np.eye(d)[0]]);cutoffs.extend([0.,0.])
            proof.append('whole space except null hyperplane');continue
        terms=[]
        if linear_norm>0:terms.append(('linear',linear_norm,a/linear_norm))
        terms.extend(('square',float(value),vector) for value,vector in zip(eigen[positive],vectors[:,positive].T))
        if not terms:proof.append('nonpositive component below threshold');continue
        available=local_threshold-c
        def thresholds(fractions):
            return np.array([available*f/scale if kind=='linear' else np.sqrt(available*f/scale)
                             for f,(kind,scale,u) in zip(fractions,terms)])
        def objective(fractions):
            cuts=thresholds(fractions)
            return float(sum((1 if row[0]=='linear' else 2)*sf(cut) for row,cut in zip(terms,cuts)))
        fractions=np.ones(len(terms))/len(terms)
        if optimize and len(terms)>1:
            result=minimize(objective,fractions,method='SLSQP',bounds=[(1e-8,1)]*len(terms),
                            constraints={'type':'eq','fun':lambda f:f.sum()-1},
                            options={'maxiter':100,'ftol':1e-13})
            if result.success and objective(result.x)<objective(fractions):fractions=result.x/result.x.sum()
        for (kind,scale,u),cut in zip(terms,thresholds(fractions)):
            directions.append(u);cutoffs.append(cut)
            if kind=='square':directions.append(-u);cutoffs.append(cut)
        proof.append('negative eigenspaces discarded; positive term caps sum exactly to threshold minus constant')
    if not directions:raise ValueError('event provably empty: no sampling cover needed')
    normals=np.array(directions);cuts=np.minimum(np.array(cutoffs),12.)
    # Relax extremely remote constraints instead of deleting positive-mass
    # directions whose Gaussian probabilities could underflow. Containment is
    # preserved because reducing a cutoff enlarges its halfspace.
    proof.append('cutoffs above12 relaxed to12 to retain numerical support')
    # Remove exact duplicate constraints only. Redundancy does not affect validity.
    unique=[]
    for j in range(len(cuts)):
        if not any(np.array_equal(normals[j],normals[k]) and cuts[j]==cuts[k] for k in unique):unique.append(j)
    normals=normals[unique];cuts=cuts[unique]
    masses=sf(cuts);total=float(masses.sum())
    if total<=0 or not np.isfinite(total) or np.any(masses<=0) or not np.isfinite(masses).all():raise ValueError('cover probabilities outside supported numerical range')
    return dict(normals=normals,cutoffs=cuts,masses=masses,total=total,proof=proof)


def sample_cover(model,cover,n,rng):
    normals=cover['normals'];cuts=cover['cutoffs'];mass=cover['masses']
    choice=rng.choice(len(normals),size=n,p=mass/mass.sum())
    u=normals[choice];uniform=np.maximum(rng.random(n),np.nextafter(0.,1.))
    g=rng.standard_normal((n,model.factor_dimension));g-=np.sum(g*u,axis=1)[:,None]*u
    if model.scale_df is None:
        axial=-ndtri_exp(log_ndtr(-cuts[choice])+np.log(uniform))
        return g+axial[:,None]*u
    nu=model.scale_df
    axial_t=student_t.isf(mass[choice]*uniform,nu)
    s=rng.chisquare(nu+1,size=n)/(1+axial_t**2/nu)
    axial=axial_t*np.sqrt(s/nu)
    g+=axial[:,None]*u
    # Encode the auxiliary S in the standard-normal latent coordinate used by
    # QuadraticModel. Supported probabilities exclude machine CDF endpoints.
    cdf=chi2.cdf(s,nu)
    if np.any((cdf<=0)|(cdf>=1)):raise FloatingPointError('Student auxiliary transform reached float64 endpoint')
    return np.column_stack((g,ndtri(cdf)))


def projected_values(model,cover,z,e,threshold,severity=False):
    v=z-(z@e)[:,None]*e
    a,b,c=model.conditional_coefficients(v,e)
    unit=cover['normals'];s=model.scale(v)
    slopes=unit@e[:model.factor_dimension]
    residual=cover['cutoffs'][None,:]/s[:,None]-v[:,:model.factor_dimension]@unit.T
    log_r=np.full(residual.shape,-np.inf)
    pos=slopes>0;neg=slopes<0;zero=~(pos|neg)
    log_r[:,pos]=log_ndtr(-residual[:,pos]/slopes[pos])
    log_r[:,neg]=log_ndtr(residual[:,neg]/slopes[neg])
    log_r[:,zero]=np.where(residual[:,zero]<0,0.,-np.inf)
    log_d=logsumexp(log_r,axis=1)-np.log(cover['total'])
    if not np.isfinite(log_d).all():raise FloatingPointError('projected cover marginal numerically vanished')
    w=np.exp(-log_d)
    y=w*probability(a,b,c,threshold)
    if np.any(y>cover['total']*(1+1e-8)):
        raise FloatingPointError('cover or conditional arithmetic violated probability bound')
    y=np.minimum(y,cover['total'])
    moment=w*tail_moment(a,b,c,threshold) if severity else None
    return y,moment,v,w


def estimate_envelope(model,threshold,budget=16384,seed=42,project=True,select=True,
                      optimize_cover=True,severity=False,alpha=.05,relative_target=.1,return_samples=False):
    if type(budget) is not int or budget<256 or not np.isfinite(threshold):raise ValueError('invalid budget/threshold')
    if not 0<alpha<1 or not 0<relative_target<1:raise ValueError('invalid confidence or relative precision')
    severity_support=model.severity_support()
    if severity and not severity_support["mean_supported"]:raise ValueError("Student tail mean not supported at these degrees of freedom")
    start=time.perf_counter();p0=model.evaluations;c0=model.conditional_evaluations
    cover=build_cover(model,threshold,optimize_cover)
    pilot_seed,final_seed=np.random.SeedSequence(seed).spawn(2)
    d=model.dimension
    candidates=[]
    for u in cover['normals']:
        e=np.r_[u,0.] if model.scale_df is not None else u.copy()
        if not any(abs(e@v)>1-1e-10 for v in candidates):candidates.append(e)
    # Principal geometry direction is available to fixed conditional baseline.
    metric=np.einsum('k,ki,kj->ij',cover['masses'],cover['normals'],cover['normals'])
    _,vectors=np.linalg.eigh(metric)
    e=np.r_[vectors[:,-1],0.] if model.scale_df is not None else vectors[:,-1]
    if not any(abs(e@v)>1-1e-10 for v in candidates):candidates.append(e)
    candidates=candidates[:8];selection=[];pilot=0
    if project and select and len(candidates)>1:
        n=max(8,(budget//5)//len(candidates))
        z=sample_cover(model,cover,n,np.random.default_rng(pilot_seed))
        for direction in candidates:
            y,_,_,_=projected_values(model,cover,z,direction,threshold)
            selection.append(dict(direction=direction.tolist(),second_moment=float(y@y/n)))
        e=np.array(min(selection,key=lambda row:row['second_moment'])['direction'])
        pilot=n*len(candidates)
    n=budget-pilot
    z=sample_cover(model,cover,n,np.random.default_rng(final_seed))
    if project:
        y,moment,v,w=projected_values(model,cover,z,e,threshold,severity)
    else:
        x=z[:,:model.factor_dimension]*model.scale(z)[:,None]
        overlap=np.count_nonzero(x@cover['normals'].T>cover['cutoffs'],axis=1)
        if np.any(overlap<1):raise FloatingPointError('conditional sample left cover')
        losses=model.loss(z);w=cover['total']/overlap;y=w*(losses>threshold)
        moment=w*losses*(losses>threshold) if severity else None
        v=z
    out=summarize_observations(y,cover['total'],alpha)
    out.update(precision_status(out['interval'],relative_target))
    out.update(method='envelope_projected' if project else 'envelope_raw',seed=int(seed),threshold=float(threshold),
               budget=budget,confirmation_samples=n,point_evaluations=model.evaluations-p0,
               conditional_evaluations=model.conditional_evaluations-c0,selection_calls=pilot,
               direction=e.tolist(),cover={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cover.items()},
               projection_selection=selection,seconds=time.perf_counter()-start,model=model.specification(),
               estimator_assumptions=['Proven supported max-quadratic event cover','Exact Gaussian/Student halfspace conditionals',
                                     'Frozen independent confirmation; fixed-budget Bernstein interval',
                                     'Finite precision roots/transforms, not interval-certified numerics'],
               geometry_access='Analytic coefficients, spectral decomposition and conditional oracle; compare conditional baselines')
    if moment is not None:
        out.update(tail_first_moment=float(moment.mean()),tail_first_moment_se=float(moment.std(ddof=1)/np.sqrt(n)) if severity_support["variance_supported"] else None,
                   conditional_severity=float(moment.mean()/y.mean()) if y.mean()>0 else None,
                   severity_support=severity_support,severity_kind='finite-sample biased tail-mean ratio; asymptotic SE only when variance supported')
    if return_samples:out.update(_observations=y,_moments=moment,_projected_samples=v,_weights=w)
    return out

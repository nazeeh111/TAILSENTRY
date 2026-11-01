"""Supported VaR/ES plug-in estimation using a frozen conditional survival curve.

Probability observations are unbiased. Estimated quantiles and ES are finite-
sample biased. Simultaneous finite-sample quantile brackets use a predeclared
finite grid and Bonferroni bounds, not pointwise intervals at a selected root.
"""
import time
import numpy as np
from scipy.optimize import brentq
from .engine import estimate,log_ratio
from .envelope import estimate_envelope
from .conditional import probability,tail_moment
from .uncertainty import summarize_observations


def estimate_risk(model,alpha,threshold_grid,budget=16384,seed=42,method='envelope',epsilon=.15):
    started=time.perf_counter()
    support=model.severity_support()
    if not support["mean_supported"]:raise ValueError("Expected Shortfall moment not supported")
    grid=np.asarray(threshold_grid,float)
    if not 0<alpha<1 or grid.ndim!=1 or len(grid)<3 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0):
        raise ValueError("need alpha in (0,1) and increasing finite threshold grid")
    # Reserve one conditional call per confirmation scenario for coefficients.
    # Parent engine charges its own pilot/confirmation; reuse is explicit here.
    if method=='envelope':
        run=estimate_envelope(model,float(grid[0]),budget=budget//2,seed=seed,select=False,return_samples=True)
        w=run.pop('_weights');bound=run['observation_bound']
    else:
        if method=='raw_mixture':raise ValueError("risk curve requires a conditional method")
        run=estimate(model,float(grid[0]),budget=budget//2,seed=seed,method=method,epsilon=epsilon,return_samples=True)
        bound=1. if method.startswith('conditional') else 1/epsilon
    v=run.pop('_projected_samples');run.pop('_observations');run.pop('_moments')
    e=np.array(run['direction'])
    a,b,c=model.conditional_coefficients(v,e)
    if method!='envelope':w=np.exp(log_ratio(v,np.array(run['means']),epsilon,e))
    beta=1-alpha
    def observations(t):return w*probability(a,b,c,t)
    def survival(t):return float(observations(t).mean())
    curves=[]
    for t in grid:
        s=summarize_observations(observations(t),bound,alpha=.05/len(grid))
        curves.append(dict(threshold=float(t),probability=s['estimate'],interval=s['interval']))
    if survival(grid[0])<beta or survival(grid[-1])>beta:
        return dict(status='quantile-not-bracketed',alpha=alpha,curve=curves,run=run,var=None,es=None)
    q=float(brentq(lambda t:survival(t)-beta,grid[0],grid[-1],xtol=1e-10))
    y=observations(q)
    excess=w*tail_moment(a,b,c,q)-q*y
    es=float(q+excess.mean()/beta)
    lower=[row['threshold'] for row in curves if row['interval'][0]>beta]
    upper=[row['threshold'] for row in curves if row['interval'][1]<beta]
    bracket=[max(lower) if lower else None,min(upper) if upper else None]
    h=1e-4*max(1,abs(q));density=(survival(q-h)-survival(q+h))/(2*h)
    run['conditional_evaluations']+=len(v)
    run['budget_requested_for_risk']=budget
    return dict(seconds=time.perf_counter()-started,status='estimated' if all(t is not None for t in bracket) else 'quantile-precision-unsupported',
                alpha=alpha,var=q,es=es,tail_probability=survival(q),
                var_se_asymptotic=float(y.std(ddof=1)/np.sqrt(len(y))/density) if density>0 else None,
                es_se_asymptotic=float(excess.std(ddof=1)/np.sqrt(len(y))/beta) if support["variance_supported"] else None,
                var_interval_grid_95=bracket,curve=curves,run=run,
                interpretation='Plug-in VaR and ES, finite-sample biased; asymptotic SE requires regularity and adequate mode representation. Finite-grid quantile bracket is simultaneous; model uncertainty excluded.')

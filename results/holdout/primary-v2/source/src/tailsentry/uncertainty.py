"""Fixed-budget bounds and deliberately separate asymptotic diagnostics."""
import numpy as np
from scipy.stats import beta, norm


def binomial_interval(hits,n,alpha=.05):
    if not isinstance(n,(int,np.integer)) or n<1 or not 0<=hits<=n or not 0<alpha<1:
        raise ValueError("invalid binomial interval inputs")
    return [0. if hits==0 else float(beta.ppf(alpha/2,hits,n-hits+1)),
            1. if hits==n else float(beta.ppf(1-alpha/2,hits+1,n-hits))]


def zero_hit_upper(n,alpha=.05):
    if type(n) is not int or n<1 or not 0<alpha<1: raise ValueError("invalid audit inputs")
    return float(-np.expm1(np.log(alpha)/n))


def summarize_observations(y,bound,alpha=.05):
    if not 0<alpha<1:raise ValueError("alpha must lie strictly between zero and one")
    y=np.asarray(y,float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError("need at least two finite nonnegative independent observations")
    if not np.isfinite(bound) or bound<=0 or np.any(y>bound*(1+1e-10)):
        raise ValueError("invalid known observation bound")
    n=len(y);mean=float(y.mean());var=float(y.var(ddof=1));se=np.sqrt(var/n)
    # Maurer–Pontil sample-variance bound, union bound over both tails.
    log=np.log(4/alpha)
    radius=np.sqrt(2*var*log/n)+7*bound*log/(3*(n-1))
    normal_radius=float(norm.ppf(1-alpha/2)*se)
    return dict(estimate=mean,standard_error=float(se),
                interval=[max(0.,mean-radius),min(1.,mean+radius)],
                interval_kind="fixed-budget empirical Bernstein (finite-sample, known bound)",
                asymptotic_interval=[max(0.,mean-normal_radius),min(1.,mean+normal_radius)],
                observation_bound=float(bound),confidence=1-alpha,
                tail_ess=float(y.sum()**2/(y@y)) if y@y>0 else 0.,
                max_contribution_fraction=float(y.max()/y.sum()) if y.sum()>0 else None)


def precision_status(interval,relative_target):
    low,high=interval
    # This interval-based criterion implies relative error <=target if coverage
    # holds, using the interval midpoint as the reported certified center.
    center=(high+low)/2
    valid=0<=low<=high<=1
    met=valid and low>0 and (high-low)/(2*low)<=relative_target
    return dict(status="inconsistent-interval" if not valid else "precision-supported" if met else "insufficient-budget",
                requested_relative_error=relative_target,certified_center=center,
                note="Fixed-budget simulation uncertainty under the stated model; no model-validity guarantee")

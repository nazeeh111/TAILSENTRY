"""Projection-conditioned defensive mixtures with independent confirmation.

This implementation combines established importance sampling and conditional
Monte Carlo. Originality beyond the implementation is explicitly unestablished.
"""
import time
import numpy as np
from scipy.special import logsumexp
from .conditional import probability,tail_moment
from .uncertainty import summarize_observations,precision_status


def mixture_sample(rng,n,means,epsilon):
    alpha=np.r_[epsilon,np.full(len(means),(1-epsilon)/len(means))]
    locations=np.vstack((np.zeros(means.shape[1]),means))
    choice=rng.choice(len(alpha),size=n,p=alpha)
    return rng.standard_normal((n,means.shape[1]))+locations[choice]


def log_ratio(z,means,epsilon,direction=None):
    locations=np.vstack((np.zeros(z.shape[1]),means))
    if direction is not None:
        locations=locations-(locations@direction)[:,None]*direction
    alpha=np.r_[epsilon,np.full(len(means),(1-epsilon)/len(means))]
    logits=z@locations.T-.5*np.sum(locations**2,axis=1)+np.log(alpha)
    return -logsumexp(logits,axis=1)


def _cluster(points,k,rng):
    k=min(k,len(points))
    if k<1: return np.zeros((1,points.shape[1]))
    # Deterministic seeded farthest-point initialization, then Lloyd updates.
    centers=[points[rng.integers(len(points))]]
    for _ in range(1,k):
        dist=((points[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2).min(axis=1)
        centers.append(points[dist.argmax()])
    centers=np.array(centers)
    for _ in range(12):
        labels=((points[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        updated=np.array([points[labels==j].mean(0) if np.any(labels==j) else centers[j] for j in range(k)])
        if np.allclose(updated,centers): break
        centers=updated
    return centers


def train(model,threshold,rng,n,k=4,epsilon=.15):
    """Two elite-mixture passes, pilot only, no recycled inference samples.

    A deliberately simple exploration heuristic, NOT a faithful SAIS algorithm.
    Identity covariance stays fixed; raw elite clustering is not a CE optimum.
    """
    counts=(n//2,n-n//2);means=np.zeros((1,model.dimension))
    history=[]
    for count in counts:
        z=mixture_sample(rng,count,means,epsilon)
        losses=model.loss(z)
        cut=min(threshold,float(np.quantile(losses,.85)))
        elite=z[losses>=cut]
        if len(elite)<4: elite=z[np.argsort(losses)[-min(count,8):]]
        means=_cluster(elite,k,rng)
        history.append(dict(evaluations=count,elite_threshold=cut,elite_count=len(elite),means=means.tolist()))
    return means,history


def _directions(model,means):
    d=model.factor_dimension
    candidates=[np.eye(model.dimension)[0]]
    for mu in means:
        v=mu.copy()
        if model.scale_df is not None:v[-1]=0
        if np.linalg.norm(v)>1e-10:candidates.append(v/np.linalg.norm(v))
    # Exact supported-model structure is available to EVERY conditional baseline.
    metric=np.einsum('ki,kj->ij',model.linear,model.linear)+np.sum(
        np.einsum('kij,kjl->kil',model.quadratic,model.quadratic),axis=0)
    _,vectors=np.linalg.eigh(metric)
    for v in vectors.T[-min(3,d):]:
        v=np.r_[v,0.] if model.scale_df is not None else v
        candidates.append(v)
    unique=[]
    for e in candidates:
        if not any(abs(e@v)>1-1e-10 for v in unique): unique.append(e)
    return unique[:8]


def estimate(model,threshold,budget=16384,seed=42,epsilon=.15,relative_target=.1,
             method="projected_mixture",severity=False,alpha=.05,return_samples=False):
    """Estimate P(L>threshold), with an independently frozen final proposal.

    budget counts point loss calls + conditional oracle calls (not equivalent
    information). Wall time and separate counters must accompany comparisons.
    method: projected_mixture, raw_mixture, conditional, conditional_structure.
    Guaranteed probability interval only; severity SE is asymptotic.
    """
    if type(budget) is not int or budget<256: raise ValueError("budget must be an integer >=256")
    if not np.isfinite(threshold) or not 0<epsilon<=1 or not 0<relative_target<1 or not 0<alpha<1:
        raise ValueError("invalid threshold, defensive mass, precision or confidence")
    if method not in ("projected_mixture","raw_mixture","conditional","conditional_structure"):
        raise ValueError("unknown estimator")
    severity_support=model.severity_support()
    if severity and not severity_support["mean_supported"]:raise ValueError("Student tail mean not supported at these degrees of freedom")
    start=time.perf_counter();p0=model.evaluations;c0=model.conditional_evaluations
    seq=np.random.SeedSequence(seed);rtrain,rselect,rconfirm=[np.random.default_rng(s) for s in seq.spawn(3)]
    history=[];selection=[];training=0;selection_calls=0
    if method in ("projected_mixture","raw_mixture"):
        training=max(64,budget//5)
        means,history=train(model,threshold,rtrain,training,epsilon=epsilon)
    else:means=np.zeros((1,model.dimension))
    e=np.eye(model.dimension)[0]
    if method=="conditional_structure":
        e=_directions(model,means)[-1]
    if method=="projected_mixture":
        directions=_directions(model,means)
        nselect=max(8,(budget//5)//len(directions))
        z=mixture_sample(rselect,nselect,means,epsilon)
        for direction in directions:
            v=z-(z@direction)[:,None]*direction
            a,b,c=model.conditional_coefficients(v,direction)
            y=probability(a,b,c,threshold)*np.exp(log_ratio(v,means,epsilon,direction))
            # Unbiased pilot second moment; conditioning draws have q_perp law.
            score=float(y@y/len(y))
            selection.append(dict(direction=direction.tolist(),second_moment=score))
        e=np.array(min(selection,key=lambda row:row['second_moment'])['direction'])
        selection_calls=nselect*len(directions)
    n=budget-training-selection_calls
    z=mixture_sample(rconfirm,n,means,epsilon)
    if method=="raw_mixture":
        losses=model.loss(z);w=np.exp(log_ratio(z,means,epsilon));y=w*(losses>threshold)
        bound=1/epsilon;moment=w*losses*(losses>threshold) if severity else None
        v=z
    else:
        v=z-(z@e)[:,None]*e
        a,b,c=model.conditional_coefficients(v,e)
        w=np.exp(log_ratio(v,means,epsilon,e))
        y=w*probability(a,b,c,threshold)
        bound=1. if method.startswith('conditional') else 1/epsilon
        moment=w*tail_moment(a,b,c,threshold) if severity else None
    result=summarize_observations(y,bound,alpha)
    result.update(precision_status(result['interval'],relative_target))
    result.update(method=method,seed=int(seed),threshold=float(threshold),budget=budget,
                  confirmation_samples=n,training_calls=training,selection_calls=selection_calls,
                  point_evaluations=model.evaluations-p0,conditional_evaluations=model.conditional_evaluations-c0,
                  direction=e.tolist(),means=means.tolist(),epsilon=epsilon,training_history=history,
                  projection_selection=selection,seconds=time.perf_counter()-start,
                  proposal_ess=float(w.sum()**2/(w@w)),
                  assumptions=["Explicit Gaussian latent density and exact supported loss coefficients",
                               "Independent frozen-proposal confirmation; fixed budget, no optional stopping",
                               "Analytic conditional oracle is additional information beyond point simulation",
                               "Float64 tails; Student normal-to-scale transform clipped at machine endpoints"],
                  model=model.specification())
    if moment is not None:
        result['tail_first_moment']=float(moment.mean())
        result['tail_first_moment_se']=float(moment.std(ddof=1)/np.sqrt(n)) if severity_support['variance_supported'] else None
        result['severity_support']=severity_support
        result['conditional_severity']=float(moment.mean()/y.mean()) if y.mean()>0 else None
        # This ratio is finite-sample biased. It is not ES unless threshold is VaR.
        result['severity_kind']='conditional tail mean at fixed threshold; ratio, not unbiased ES'
    if return_samples:
        result['_observations']=y;result['_moments']=moment;result['_projected_samples']=v
    return result

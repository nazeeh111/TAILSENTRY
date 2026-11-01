"""Exact one-dimensional Gaussian integration of unions of quadratic events.

No sign/monotonicity assumption. Both roots and all disconnected intervals are
retained. Gaussian interval arithmetic avoids subtracting two rounded CDFs in
positive tails. Float64 arithmetic is not interval-certified arithmetic.
"""
import numpy as np
from scipy.special import ndtr

SQRT2PI=np.sqrt(2*np.pi)


def normal_interval(lo,hi):
    lo,hi=np.broadcast_arrays(np.asarray(lo,float),np.asarray(hi,float))
    return np.maximum(0,np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo)))


def quadratic_roots(a,b,c):
    """Real roots, with NaN padding; stable q formula for quadratic roots."""
    a,b,c=np.broadcast_arrays(a,b,c)
    scale=np.maximum.reduce([np.abs(a),np.abs(b),np.abs(c)])
    scale=np.where(scale>0,scale,1.)
    a,b,c=a/scale,b/scale,c/scale
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        disc=b*b-4*a*c
        root=np.sqrt(np.maximum(disc,0))
        q=-.5*(b+np.where(b>=0,1.,-1.)*root)
        r1=q/a
        r2=np.where(q!=0,c/q,-b/(2*a))
        valid=(a!=0)&(disc>=0)
        lo=np.where(valid,np.minimum(r1,r2),np.nan)
        hi=np.where(valid,np.maximum(r1,r2),np.nan)
        linear=(a==0)&(b!=0)
        lo=np.where(linear,-c/b,lo)
    return lo,hi


def event_intervals(a,b,c):
    """Two padded intervals per component where a*t²+b*t+c > 0."""
    scale=np.maximum.reduce([np.abs(a),np.abs(b),np.abs(c)])
    scale=np.where(scale>0,scale,1.)
    a,b,c=a/scale,b/scale,c/scale
    n,k=a.shape
    lo=np.full((n,k,2),np.inf); hi=lo.copy()
    r1,r2=quadratic_roots(a,b,c)
    with np.errstate(over='ignore',invalid='ignore'):
        disc=b*b-4*a*c
    quadratic=a!=0; two=quadratic&(disc>0)
    neg=two&(a<0);pos=two&(a>0)
    lo[:,:,0]=np.where(neg,r1,lo[:,:,0]);hi[:,:,0]=np.where(neg,r2,hi[:,:,0])
    lo[:,:,0]=np.where(pos,-np.inf,lo[:,:,0]);hi[:,:,0]=np.where(pos,r1,hi[:,:,0])
    lo[:,:,1]=np.where(pos,r2,lo[:,:,1]);hi[:,:,1]=np.where(pos,np.inf,hi[:,:,1])
    positive_line=(a==0)&(b>0);negative_line=(a==0)&(b<0)
    lo[:,:,0]=np.where(positive_line,r1,lo[:,:,0]);hi[:,:,0]=np.where(positive_line,np.inf,hi[:,:,0])
    lo[:,:,0]=np.where(negative_line,-np.inf,lo[:,:,0]);hi[:,:,0]=np.where(negative_line,r1,hi[:,:,0])
    whole=(quadratic&(disc<=0)&(a>0))|((a==0)&(b==0)&(c>0))
    lo[:,:,0]=np.where(whole,-np.inf,lo[:,:,0]);hi[:,:,0]=np.where(whole,np.inf,hi[:,:,0])
    return lo.reshape(n,-1),hi.reshape(n,-1)


def probability(a,b,c,threshold):
    lo,hi=event_intervals(a,b,c-float(threshold))
    order=np.argsort(lo,axis=1,kind='stable')
    lo=np.take_along_axis(lo,order,axis=1); hi=np.take_along_axis(hi,order,axis=1)
    previous=np.concatenate((np.full((len(lo),1),-np.inf),np.maximum.accumulate(hi,axis=1)[:,:-1]),axis=1)
    start=np.maximum(lo,previous)
    return np.minimum(1,normal_interval(start,hi).sum(axis=1))


def tail_moment(a,b,c,threshold):
    """E[L 1{L>threshold}] by the exact upper quadratic envelope.

    Pairwise crossings partition every envelope segment. O(n*k^3) work;
    expensive severity integration is reported separately from point loss calls.
    """
    n,k=a.shape; roots=[]
    r1,r2=quadratic_roots(a,b,c-threshold);roots.extend([r1,r2])
    for j in range(k):
        for h in range(j):
            r1,r2=quadratic_roots(a[:,j]-a[:,h],b[:,j]-b[:,h],c[:,j]-c[:,h])
            roots.extend([r1[:,None],r2[:,None]])
    edges=np.sort(np.concatenate([np.full((n,1),-np.inf),*roots,np.full((n,1),np.inf)],axis=1),axis=1)
    # NaN roots sort to the right. Map padding to +inf and keep last tail.
    edges=np.where(np.isnan(edges),np.inf,edges)
    left,right=edges[:,:-1],edges[:,1:]
    finite_l=np.isfinite(left);finite_r=np.isfinite(right)
    mid=np.where(finite_l&finite_r,(np.where(finite_l,left,0)+np.where(finite_r,right,0))/2,
                 np.where(finite_l,left+1,np.where(finite_r,right-1,0)))
    mid=np.where(left==right,0,mid)
    values=a[:,None,:]*mid[:,:,None]**2+b[:,None,:]*mid[:,:,None]+c[:,None,:]
    index=values.argmax(axis=2)
    aa=np.take_along_axis(a,index,axis=1);bb=np.take_along_axis(b,index,axis=1);cc=np.take_along_axis(c,index,axis=1)
    active=(values.max(axis=2)>threshold)&(right>left)
    p=normal_interval(left,right)
    pl=np.exp(-.5*left**2)/SQRT2PI;pr=np.exp(-.5*right**2)/SQRT2PI
    lp=np.where(finite_l,np.where(finite_l,left,0)*pl,0)
    rp=np.where(finite_r,np.where(finite_r,right,0)*pr,0)
    return np.sum(np.where(active,aa*(p+lp-rp)+bb*(pl-pr)+cc*p,0),axis=1)

"""Exact argmax-branch probability attribution for a quadratic envelope.

Ties go to the first branch (as numpy.argmax). This is a decomposition of the
assumed model event, not causal attribution or an empirical market frequency.
"""
import numpy as np
from .conditional import quadratic_roots,normal_interval


def branch_probabilities(a,b,c,threshold):
    n,k=a.shape;roots=[]
    r1,r2=quadratic_roots(a,b,c-threshold);roots.extend([r1,r2])
    for j in range(k):
        for h in range(j):
            r1,r2=quadratic_roots(a[:,j]-a[:,h],b[:,j]-b[:,h],c[:,j]-c[:,h])
            roots.extend([r1[:,None],r2[:,None]])
    edges=np.sort(np.concatenate([np.full((n,1),-np.inf),*roots,np.full((n,1),np.inf)],axis=1),axis=1)
    edges=np.where(np.isnan(edges),np.inf,edges);left,right=edges[:,:-1],edges[:,1:]
    fl=np.isfinite(left);fr=np.isfinite(right)
    mid=np.where(fl&fr,(np.where(fl,left,0)+np.where(fr,right,0))/2,np.where(fl,left+1,np.where(fr,right-1,0)))
    mid=np.where(left==right,0,mid)
    values=a[:,None,:]*mid[:,:,None]**2+b[:,None,:]*mid[:,:,None]+c[:,None,:]
    index=values.argmax(axis=2);active=(values.max(axis=2)>threshold)&(right>left)
    p=normal_interval(left,right)*active
    return np.column_stack([np.sum(np.where(index==j,p,0),axis=1) for j in range(k)])

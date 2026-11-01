"""Explicit max-of-quadratics models in Gaussian or Student-t latent space.

A model evaluation is one full vector loss. Conditional oracle calls are counted
separately: they reveal more information than black-box point evaluations.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2


@dataclass
class QuadraticModel:
    quadratic: np.ndarray
    linear: np.ndarray
    constant: np.ndarray
    name: str = "custom"
    scale_df: float | None = None
    evaluations: int = field(default=0, init=False)
    conditional_evaluations: int = field(default=0, init=False)

    def __post_init__(self):
        self.quadratic = np.array(self.quadratic, dtype=float, copy=True)
        self.linear = np.array(self.linear, dtype=float, copy=True)
        self.constant = np.array(self.constant, dtype=float, copy=True)
        if self.linear.ndim != 2 or self.linear.shape[0] < 1 or self.linear.shape[1] < 1:
            raise ValueError("linear must have shape (modes, factors)")
        k, d = self.linear.shape
        if self.quadratic.shape != (k,d,d) or self.constant.shape != (k,):
            raise ValueError("quadratic/linear/constant shapes disagree")
        if not all(np.isfinite(x).all() for x in (self.quadratic,self.linear,self.constant)):
            raise ValueError("model coefficients must be finite")
        if not np.allclose(self.quadratic,self.quadratic.swapaxes(1,2),atol=1e-13,rtol=0):
            raise ValueError("quadratic matrices must be symmetric")
        self.quadratic = .5*(self.quadratic+self.quadratic.swapaxes(1,2))
        if self.scale_df is not None and (not np.isfinite(self.scale_df) or self.scale_df <= 0):
            raise ValueError("Student degrees of freedom must be positive")
        self.factor_dimension = d
        self.dimension = d + int(self.scale_df is not None)

    def reset_counts(self):
        self.evaluations = self.conditional_evaluations = 0

    def scale(self, z):
        if self.scale_df is None:
            return np.ones(len(z))
        # Open-unit clipping protects ppf; endpoints beyond float64 precision
        # are unsupported and logged as this numerical transform limitation.
        u = np.clip(ndtr(z[:,-1]),np.finfo(float).tiny,1-np.finfo(float).eps)
        return np.sqrt(self.scale_df / chi2.ppf(u,self.scale_df))

    def _points(self,z):
        z = np.asarray(z,dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("scenarios must be a finite (n, dimension) matrix")
        return z

    def components(self,z):
        z = self._points(z)
        x = z[:,:self.factor_dimension] * self.scale(z)[:,None]
        return np.einsum('ni,kij,nj->nk',x,self.quadratic,x,optimize=True)+x@self.linear.T+self.constant

    def loss(self,z):
        z = self._points(z)
        self.evaluations += len(z)
        return self.components(z).max(axis=1)

    def mode(self,z):
        return self.components(z).argmax(axis=1)

    def conditional_coefficients(self,v,e):
        """L(v+te)=max_j(a_j t²+b_j t+c_j), exact for supported axes."""
        v = self._points(v)
        e = np.asarray(e,float)
        if e.shape != (self.dimension,) or not np.isclose(e@e,1,atol=1e-10,rtol=0):
            raise ValueError("conditioning direction must be a unit vector")
        if self.scale_df is not None and abs(e[-1]) > 1e-14:
            raise ValueError("Student scale coordinate cannot be integrated analytically")
        self.conditional_evaluations += len(v)
        s = self.scale(v)
        ef = e[:self.factor_dimension]
        x = v[:,:self.factor_dimension]*s[:,None]
        be = np.einsum('kij,j->ki',self.quadratic,ef)
        a = np.broadcast_to(np.einsum('ki,i->k',be,ef),(len(v),len(be)))*s[:,None]**2
        b = (2*x@be.T + self.linear@ef)*s[:,None]
        c = np.einsum('ni,kij,nj->nk',x,self.quadratic,x,optimize=True)+x@self.linear.T+self.constant
        return a,b,c

    def halfspaces(self,threshold):
        if self.scale_df is not None or np.any(self.quadratic):
            raise ValueError("ALOE halfspaces require Gaussian max-affine loss")
        return self.linear.copy(),float(threshold)-self.constant.copy()

    def severity_support(self):
        # Sufficient, conservative existence conditions, not a complete symbolic
        # classification of semidefinite quadratic forms.
        bounded=all(np.linalg.eigvalsh(B).max()<0 for B in self.quadratic)
        quadratic=bool(np.any(self.quadratic))
        order=2 if quadratic else 1
        nu=self.scale_df
        return dict(mean_supported=nu is None or bounded or nu>order,
                    variance_supported=nu is None or bounded or nu>2*order,
                    bounded_upper_tail=bounded,required_df_for_mean=order,
                    required_df_for_variance=2*order,
                    note="Sufficient moment conditions; Student scale matrix is not its covariance")

    def specification(self):
        return dict(name=self.name,quadratic=self.quadratic.tolist(),linear=self.linear.tolist(),
                    constant=self.constant.tolist(),scale_df=self.scale_df)


def normal_max(d=4,paired=False,rotation=None,offsets=None):
    a = np.eye(d) if rotation is None else np.asarray(rotation,float)
    if paired: a=np.concatenate((a,-a))
    c=np.zeros(len(a)) if offsets is None else np.asarray(offsets,float)
    return QuadraticModel(np.zeros((len(a),d,d)),a,c,"gaussian-paired" if paired else "gaussian-max")


def balls(centers,radii,scale_df=None):
    centers=np.asarray(centers,float); radii=np.asarray(radii,float)
    k,d=centers.shape
    return QuadraticModel(-np.broadcast_to(np.eye(d),(k,d,d)),2*centers,
                          radii**2-np.sum(centers**2,axis=1),"disconnected-balls",scale_df)


def delta_gamma(d=8,scale_df=None,rho=.45):
    """Illustrative correlated multi-asset short-convexity loss, in risk units.

    Equal dollar delta exposure plus two short-gamma factor exposures. Not a
    fitted derivative pricing model. B is a positive semidefinite gamma loss.
    """
    covariance=(1-rho)*np.eye(d)+rho*np.ones((d,d))
    chol=np.linalg.cholesky(covariance)
    w=np.ones(d)/d
    g1=np.r_[np.ones(d//2),np.zeros(d-d//2)];g1/=np.linalg.norm(g1)
    g2=np.r_[np.zeros(d//2),np.ones(d-d//2)];g2/=np.linalg.norm(g2)
    b=chol.T @ (.18*np.outer(g1,g1)+.12*np.outer(g2,g2)) @ chol
    a=-w@chol
    return QuadraticModel(b[None],a[None],np.zeros(1),"delta-gamma-fx-illustration",scale_df)

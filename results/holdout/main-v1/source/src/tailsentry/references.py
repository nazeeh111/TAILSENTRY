"""Known-truth calculations independent of the sampling engine.

The normal formulas use products of independent one-dimensional CDFs.  Ball
probabilities use the noncentral chi-square identity, not simulated samples.
Quadrature error estimates returned here are numerical diagnostics, not Monte
Carlo confidence intervals or rigorous integration enclosures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr
from scipy.stats import ncx2


def _normal_inputs(threshold, dimension: int, offsets):
    if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    dimension = int(dimension)
    off = np.zeros(dimension) if offsets is None else np.asarray(offsets, dtype=float)
    if off.shape != (dimension,) or not np.all(np.isfinite(off)):
        raise ValueError("offsets must be a finite vector with length dimension")
    threshold = np.asarray(threshold, dtype=float)
    if np.any(np.isnan(threshold)):
        raise ValueError("threshold must not contain NaN")
    return threshold, off


def _scalar_if_needed(result):
    result = np.asarray(result)
    return float(result) if result.ndim == 0 else result


def normal_max_survival(threshold, dimension: int, offsets=None):
    """P(max_i(Z_i + offsets_i) > threshold), for IID standard normal Z_i.

    Scalar or array thresholds are supported.  log-CDF and expm1 avoid the
    cancellation in ``1 - prod(cdf)`` at the rare probability scales tested.
    """
    t, off = _normal_inputs(threshold, dimension, offsets)
    return _scalar_if_needed(-np.expm1(np.sum(log_ndtr(t[..., None] - off), axis=-1)))


def normal_paired_survival(threshold, dimension: int, offsets=None):
    """P(max_i(abs(Z_i) + offsets_i) > threshold), independent standard normals."""
    t, off = _normal_inputs(threshold, dimension, offsets)
    cut = t[..., None] - off
    # For cut<=0 the event has probability one.  Evaluate log1p only at
    # positive cuts to avoid log(0) warnings at the origin.
    log_inside = np.full(cut.shape, -np.inf)
    positive = cut > 0
    with np.errstate(divide="ignore"):
        log_inside[positive] = np.log1p(-2.0 * ndtr(-cut[positive]))
    return _scalar_if_needed(-np.expm1(np.sum(log_inside, axis=-1)))


def ball_survival(threshold, centers: Sequence[Sequence[float]], radii: Sequence[float]):
    """P(max_j(r_j**2 - ||Z-c_j||**2) > threshold), Z~N(0,I).

    The original balls must have disjoint interiors, and thresholds must be
    nonnegative.  The resulting shrunken balls remain disjoint, so summing
    noncentral chi-square CDFs is exact up to special-function error.  Overlap
    is rejected rather than silently double counted.
    """
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    if c.ndim != 2 or min(c.shape) < 1 or not np.all(np.isfinite(c)):
        raise ValueError("centers must be a nonempty finite matrix")
    if r.shape != (c.shape[0],) or np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("radii must be a positive finite vector, one per center")
    for j in range(len(c)):
        for k in range(j):
            if np.linalg.norm(c[j] - c[k]) < r[j] + r[k]:
                raise ValueError("ball interiors overlap; additive truth is invalid")
    t = np.asarray(threshold, dtype=float)
    if np.any(np.isnan(t)) or np.any(t < 0):
        raise ValueError("ball reference requires nonnegative thresholds")
    cutoff = np.maximum(r * r - t[..., None], 0.0)
    p = np.sum(ncx2.cdf(cutoff, c.shape[1], np.sum(c * c, axis=1)), axis=-1)
    return _scalar_if_needed(p)


def solve_risk(
    survival: Callable[[float], float],
    alpha: float,
    bracket: tuple[float, float],
    *,
    upper_support: float = np.inf,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> dict[str, float]:
    """Deterministic VaR/ES reference for an integrable continuous loss.

    ``survival(t)`` is P(L>t), ``alpha`` is the CDF confidence level, and the
    bracket must straddle the quantile.  Uses ES=q+integral_q^infinity S(t)dt /
    (1-alpha).  ``integration_error_estimate`` is QUADPACK's error diagnostic
    after division by the tail probability; it is not a certified bound.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    lo, hi = map(float, bracket)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("bracket must be two increasing finite values")
    if not (atol > 0 and rtol >= 4 * np.finfo(float).eps):
        raise ValueError("invalid numerical tolerances")
    if np.isnan(upper_support) or upper_support < hi:
        raise ValueError("upper_support must be at least the bracket upper end")
    beta = 1.0 - alpha

    def checked_survival(x):
        value = float(survival(float(x)))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("survival must return finite probabilities in [0,1]")
        return value

    if checked_survival(lo) < beta or checked_survival(hi) > beta:
        raise ValueError("bracket does not straddle the requested quantile")
    q = float(brentq(lambda x: checked_survival(x) - beta, lo, hi, xtol=atol, rtol=rtol))
    excess, error = quad(checked_survival, q, upper_support, epsabs=atol, epsrel=rtol, limit=250)
    return {
        "var": q,
        "es": q + excess / beta,
        "tail_probability": checked_survival(q),
        "tail_excess": float(excess),
        "integration_error_estimate": float(error / beta),
    }

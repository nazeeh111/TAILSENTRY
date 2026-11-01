"""ALOE baseline for a *known complete* union of Gaussian half-spaces.

This implements Owen, Maximov and Chertkov (2019), not a TAILSENTRY
invention.  No pilot fitting is needed.  Inputs are in independent standard
normal coordinates, and ``model.halfspaces(threshold)`` must describe the
entire event, not merely a selected subset of its modes.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp


def aloe(model, threshold: float, n: int, seed: int = 0) -> dict:
    """Estimate ``P(model.loss(Z) > threshold)`` for Gaussian ``Z``.

    The model supplies ``dimension``, ``loss(z)`` and
    ``halfspaces(threshold) -> (directions, cutoffs)``.  The latter describes
    ``union_j {directions[j] @ Z > cutoffs[j]}``.  Directions need not be
    normalized.  Scale-mixture targets are deliberately rejected.

    Each sample chooses component j with probability P(H_j)/sum_k P(H_k),
    then samples the exact target conditional on H_j.  Its raw (unnormalized)
    likelihood weight is sum_k P(H_k) / number_of_events_occurring.
    The sample mean is unbiased in exact arithmetic.  ``se`` is an empirical
    standard error, not a finite-sample confidence guarantee.
    """
    start = time.perf_counter()
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer of at least two")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if getattr(model, "scale_df", None) is not None:
        raise ValueError("ALOE baseline supports Gaussian targets only")
    try:
        halfspaces = model.halfspaces(threshold)
    except AttributeError as exc:
        raise ValueError("model does not provide a complete half-space event") from exc
    if halfspaces is None:
        raise ValueError("model does not provide a complete half-space event")
    directions, cutoffs = (np.asarray(a, dtype=float) for a in halfspaces)
    dimension = int(model.dimension)
    if directions.ndim != 2 or directions.shape[1] != dimension or not len(directions):
        raise ValueError("directions must have shape (components, dimension)")
    if cutoffs.shape != (len(directions),):
        raise ValueError("one cutoff is required for each direction")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(cutoffs)):
        raise ValueError("half-space parameters must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("half-space directions must have positive finite norm")
    unit = directions / norms[:, None]
    cutoffs = cutoffs / norms
    log_probabilities = log_ndtr(-cutoffs)
    log_union_bound = float(logsumexp(log_probabilities))
    union_bound = float(np.exp(log_union_bound))
    if union_bound == 0 or not np.isfinite(union_bound):
        raise ValueError("union bound is outside the supported floating-point range")
    component_probabilities = np.exp(log_probabilities - log_union_bound)
    component_probabilities /= component_probabilities.sum()
    rng = np.random.default_rng(seed)
    component = rng.choice(len(unit), size=n, p=component_probabilities)
    selected = unit[component]
    z = rng.normal(size=(n, dimension))
    z -= np.einsum("ij,ij->i", z, selected)[:, None] * selected
    # Survival inversion stays stable even when sf(cutoff) is extremely small.
    # For U uniform on (0,1), log sf(T) = log sf(cutoff) + log U.
    uniforms = np.maximum(rng.random(n), np.nextafter(0.0, 1.0))
    axial = -ndtri_exp(log_probabilities[component] + np.log(uniforms))
    z += axial[:, None] * selected
    overlap = np.empty(n, dtype=np.int64)
    for first in range(0, n, 4096):
        last = min(first + 4096, n)
        overlap[first:last] = np.count_nonzero(z[first:last] @ unit.T > cutoffs, axis=1)
    if np.any(overlap == 0):
        raise FloatingPointError("sample left its selected half-space through rounding")
    weights = union_bound / overlap
    losses = np.asarray(model.loss(z), dtype=float)
    if losses.shape != (n,) or not np.all(np.isfinite(losses)):
        raise ValueError("model.loss must return one finite loss per sample")
    if np.any(losses <= threshold):
        raise ValueError("provided half-spaces do not lie inside the stated loss event")
    estimate = float(np.mean(weights))
    se = float(np.std(weights, ddof=1) / np.sqrt(n))
    return {
        "estimate": estimate,
        "se": se,
        "samples": z,
        "weights": weights,
        "losses": losses,
        "values": weights.copy(),
        "observations": weights.copy(),
        "bound": union_bound,
        "evaluations": n,
        "elapsed_seconds": time.perf_counter() - start,
        "diagnostics": {
            "algorithm": "ALOE (Owen, Maximov and Chertkov, 2019)",
            "components": len(unit),
            "component_counts": np.bincount(component, minlength=len(unit)).tolist(),
            "constraint_evaluations": n * len(unit),
            "union_bound": union_bound,
            "weight_upper_bound": union_bound,
            "minimum_overlap": int(np.min(overlap)),
            "maximum_overlap": int(np.max(overlap)),
            "uncertainty": "empirical standard error; complete event geometry assumed",
            "target": "independent standard Gaussian",
        },
    }


run_aloe = aloe


def run_projected_aloe(model, budget: int, seed: int, threshold: float,
                       direction=None) -> dict:
    """Rao–Blackwellize ALOE along a fixed direction, for complete affine events.

    This is a composition of established ALOE and conditional Monte Carlo,
    provided as a strong comparator.  It uses the full-event geometry oracle
    also needed by ALOE.  The direction must be fixed before these draws;
    selecting it using the returned samples invalidates the iid argument.

    If r_j(v)=P(H_j | V=v), ALOE's marginal proposal is
    q_V(v)=p_V(v)*sum_j r_j(v)/S, with S=sum_j P(H_j).
    The observation is S*P(union H_j | V=v)/sum_j r_j(v), in [0,S].
    This is E[ordinary_ALOE_weight | V], hence has no greater variance
    at the same sample count, independently of computational cost.
    """
    start = time.perf_counter()
    result = aloe(model, threshold, budget, seed)
    raw_directions, raw_cutoffs = model.halfspaces(threshold)
    normals = np.asarray(raw_directions, dtype=float)
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, None]
    cutoffs = np.asarray(raw_cutoffs, dtype=float) / norms
    if direction is None:
        # An explicit geometry-informed default. It is not a learned direction.
        e = normals[np.argmax(log_ndtr(-cutoffs))].copy()
    else:
        e = np.asarray(direction, dtype=float)
        if e.shape != (model.dimension,) or not np.all(np.isfinite(e)):
            raise ValueError("direction must be a finite dimension-vector")
        norm = np.linalg.norm(e)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("direction must have positive finite norm")
        e = e / norm
    z = result["samples"]
    v = z - np.outer(z @ e, e)
    slopes = normals @ e
    positive = slopes > 0
    negative = slopes < 0
    zero = slopes == 0
    observations = np.empty(budget)
    conditional_probability = np.empty(budget)
    log_density_ratio = np.empty(budget)
    union_bound = result["bound"]
    for first in range(0, budget, 4096):
        last = min(first + 4096, budget)
        block = v[first:last]
        residual = cutoffs[None, :] - block @ normals.T
        log_r = np.full(residual.shape, -np.inf)
        lower = np.full(len(block), -np.inf)
        upper = np.full(len(block), np.inf)
        if np.any(positive):
            roots = residual[:, positive] / slopes[positive]
            log_r[:, positive] = log_ndtr(-roots)
            upper = np.min(roots, axis=1)
        if np.any(negative):
            roots = residual[:, negative] / slopes[negative]
            log_r[:, negative] = log_ndtr(roots)
            lower = np.max(roots, axis=1)
        if np.any(zero):
            log_r[:, zero] = np.where(residual[:, zero] < 0, 0.0, -np.inf)
            certain = np.any(residual[:, zero] < 0, axis=1)
        else:
            certain = np.zeros(len(block), dtype=bool)
        log_union = np.logaddexp(log_ndtr(lower), log_ndtr(-upper))
        log_union = np.where(certain | (lower >= upper), 0.0, log_union)
        log_sum_r = logsumexp(log_r, axis=1)
        if np.any(~np.isfinite(log_sum_r)):
            raise FloatingPointError("ALOE marginal proposal vanished numerically")
        log_ratio = log_sum_r - np.log(union_bound)
        log_y = log_union - log_ratio
        observations[first:last] = np.exp(log_y)
        conditional_probability[first:last] = np.exp(log_union)
        log_density_ratio[first:last] = log_ratio
    # Only suppress roundoff above the proved bound, never hide a real violation.
    if np.any(observations > union_bound * (1 + 1e-12)):
        raise FloatingPointError("projected observation violated its analytic bound")
    observations = np.minimum(observations, union_bound)
    result.update({
        "estimate": float(np.mean(observations)),
        "se": float(np.std(observations, ddof=1) / np.sqrt(budget)),
        "observations": observations,
        "values": observations,
        "bound": union_bound,
        "direction": e,
        "projected_samples": v,
        "conditional_probability": conditional_probability,
        "log_marginal_density_ratio": log_density_ratio,
        "elapsed_seconds": time.perf_counter() - start,
    })
    result["diagnostics"].update({
        "algorithm": "projected ALOE (conditional Monte Carlo composition)",
        "additional_projection_constraint_evaluations": budget * len(normals),
        "direction_source": "supplied fixed direction" if direction is not None else
                            "normal of most probable known half-space",
        "variance_comparison": "no greater than raw ALOE at identical fixed proposal/sample count",
    })
    return result

"""Paper-informed SAIS reproduction with separately frozen confirmation.

Helal and Elvira (2025), arXiv:2508.00210v1, Algorithm 1 and Eqs. 4,
10--21. See docs/sais-fidelity.md for disambiguations and numerical guards.
This is independently implemented source, not author-released code or a claim
to reproduce the paper's numerical tables. Only standard-normal latent targets
are supported. No model geometry, gradients, or reference probabilities are read.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, logsumexp
from scipy.stats import norm


def _loss(model: Any, z: np.ndarray) -> np.ndarray:
    value = np.asarray(model.loss(z), dtype=float)
    if value.shape != (len(z),) or not np.isfinite(value).all():
        raise ValueError("model.loss must return one finite value per scenario")
    return value


def _target_logpdf(x: np.ndarray) -> np.ndarray:
    return -0.5 * (x.shape[1] * math.log(2 * math.pi) + np.sum(x * x, axis=1))


def _log_components(x: np.ndarray, means: np.ndarray,
                    covariances: np.ndarray) -> np.ndarray:
    """Full-covariance log densities at the SAME points for every proposal."""
    values = np.empty((len(x), len(means)))
    constant = x.shape[1] * math.log(2 * math.pi)
    for j, (mean, covariance) in enumerate(zip(means, covariances)):
        chol = np.linalg.cholesky(covariance)
        delta = solve_triangular(chol, (x - mean).T, lower=True, check_finite=False)
        values[:, j] = -0.5 * (constant + 2 * np.log(np.diag(chol)).sum()
                               + np.sum(delta * delta, axis=0))
    return values


def _dm_logweights(x: np.ndarray, means: np.ndarray,
                   covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    component_logpdf = _log_components(x, means, covariances)
    logq = logsumexp(component_logpdf, axis=1) - math.log(len(means))
    return _target_logpdf(x) - logq, component_logpdf.argmax(axis=1)


def _draw_stratified(rng: np.random.Generator, means: np.ndarray,
                     covariances: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Exactly count observations per component, with deterministic mixture weights."""
    blocks = []
    for mean, covariance in zip(means, covariances):
        chol = np.linalg.cholesky(covariance)
        blocks.append(mean + rng.standard_normal((count, means.shape[1])) @ chol.T)
    return np.concatenate(blocks), np.repeat(np.arange(len(means)), count)


def _threshold(scores: np.ndarray, sources: np.ndarray, previous: float,
                rho: float, components: int) -> tuple[float, dict[str, Any]]:
    """Algorithm 1 Step 1c--2c, retaining its printed descending pooled order.

    Local 'top' elites are lowest S, since failure is S<0. Zero order-statistic
    ranks are rounded to one; this guard is necessary when rho*count<1.
    """
    elites = []
    counts, rounded = [], 0
    for j in range(components):
        local = scores[(sources == j) & (scores <= previous)]
        raw = math.floor(rho * len(local))
        take = max(1, raw) if len(local) else 0
        rounded += int(len(local) > 0 and raw == 0)
        counts.append(take)
        if take:
            elites.append(np.sort(local)[:take])
    if not elites:
        return previous, {"local_elite_counts": counts, "rank_guards": rounded,
                          "pooled_elites": 0, "status": "no previous-subset samples"}
    pooled = np.sort(np.concatenate(elites))[::-1]
    raw_rank = math.floor(rho * len(pooled))
    rank = max(1, raw_rank)
    return float(pooled[rank - 1]), {
        "local_elite_counts": counts, "pooled_elites": len(pooled),
        "pooled_descending_rank": rank,
        "rank_guards": rounded + int(raw_rank == 0), "status": "updated",
    }


def _floor_covariance(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, bool]:
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    guarded = bool(values.min() < floor)
    if guarded:
        covariance = (vectors * np.maximum(values, floor)) @ vectors.T
        covariance = 0.5 * (covariance + covariance.T)
    return covariance, guarded


def _lw_coefficient(x: np.ndarray, covariance: np.ndarray) -> tuple[float, float | None]:
    """Eq. 18 as printed, with its uncentered xx' numerator.

    The printed coefficient is clipped to the range required by Eq. 17.
    The isotropic/zero denominator limit is unspecified; use beta=1 explicitly.
    """
    k, d = x.shape
    norm4 = np.sum(x * x, axis=1) ** 2
    quadratic = np.einsum("ni,ij,nj->n", x, covariance, x)
    frobenius2 = float(np.sum(covariance * covariance))
    numerator = max(0.0, float(np.sum(norm4 - 2 * quadratic + frobenius2)))
    anisotropy = max(0.0, frobenius2 - float(np.trace(covariance)) ** 2 / d)
    denominator = k * k * anisotropy
    if denominator <= np.finfo(float).eps * max(1.0, k * k * frobenius2):
        return 1.0, None
    raw = numerator / denominator
    return float(np.clip(raw, 0.0, 1.0)), float(raw)


def _update_component(x: np.ndarray, scores: np.ndarray, logweights: np.ndarray,
                      previous_mean: np.ndarray, previous_covariance: np.ndarray,
                      boundary: float, iteration: int,
                      minimum_eigenvalue: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Eqs. 13--18 for one hard-reassigned previous-subset sample set."""
    local_count = len(x)
    active = scores <= boundary
    if local_count == 0 or not np.any(active):
        return previous_mean.copy(), previous_covariance.copy(), {
            "status": "retained: no reassigned samples at new threshold",
            "reassigned_samples": local_count, "active_samples": int(active.sum()),
        }
    shifted = np.exp(logweights - logweights.max())
    normalized = shifted / shifted.sum()
    ess = float(1 / np.sum(normalized * normalized))
    tempered = ess < local_count / 2
    gamma = float(expit(iteration)) if tempered else 1.0
    selected = x[active]
    active_logs = gamma * logweights[active]
    weights = np.exp(active_logs - active_logs.max())
    weights /= weights.sum()
    new_mean = weights @ selected
    # Literal Eq.14 uses the old mean. Eq.16's tempered mean is interpreted
    # as the newly fitted tempered mean; both choices are disclosed.
    center = new_mean if tempered else previous_mean
    delta = selected - center
    empirical = (delta.T * weights) @ delta
    beta, raw_beta = _lw_coefficient(selected, empirical)
    isotropic = float(np.trace(empirical)) / x.shape[1]
    covariance = ((1 - beta) * previous_covariance + beta * empirical
                  + (0.1 / iteration) * isotropic * np.eye(x.shape[1]))
    covariance, floor_used = _floor_covariance(covariance, minimum_eigenvalue)
    return new_mean, covariance, {
        "status": "updated", "reassigned_samples": local_count,
        "active_samples": len(selected), "ess": ess,
        "ess_threshold": local_count / 2, "tempered": tempered, "gamma": gamma,
        "beta": beta, "unclipped_beta": raw_beta,
        "beta_zero_denominator_guard": raw_beta is None,
        "beta_clipped": raw_beta is not None and not 0 <= raw_beta <= 1,
        "covariance_floor_used": floor_used,
        "covariance_center": "new tempered mean" if tempered else "old mean, printed Eq.14",
        "empirical_covariance": empirical.tolist(),
    }


def _recycle(estimates: list[float], forgetting: float) -> float:
    exponents = np.arange(len(estimates) - 1, -1, -1)
    mass = forgetting ** exponents
    return float(np.dot(mass / mass.sum(), estimates))


def _draw_confirmation(rng: np.random.Generator, means: np.ndarray,
                        covariances: np.ndarray, n: int,
                        defense: float) -> tuple[np.ndarray, np.ndarray]:
    """IID random mixture from frozen SAIS proposals plus explicit defense."""
    components, d = means.shape
    masses = np.r_[defense, np.full(components, (1 - defense) / components)]
    labels = rng.choice(components + 1, size=n, p=masses)
    x = rng.standard_normal((n, d))
    for j in range(components):
        selected = labels == j + 1
        if np.any(selected):
            chol = np.linalg.cholesky(covariances[j])
            x[selected] = means[j] + x[selected] @ chol.T
    log_components = _log_components(x, means, covariances)
    logq = np.logaddexp(
        math.log(defense) + _target_logpdf(x),
        math.log1p(-defense) + logsumexp(log_components, axis=1) - math.log(components),
    )
    weights = np.exp(_target_logpdf(x) - logq)
    return x, weights


def estimate_sais(
    model: Any, threshold: float, budget: int = 16384, seed: int = 42, *,
    components: int = 4, samples_per_component: int | None = None,
    max_iterations: int = 10, pilot_fraction: float = 1 / 3,
    rho: float = 0.1, forgetting: float = 0.1,
    confirmation_defense: float = 0.1, initial_means: np.ndarray | None = None,
    initial_covariances: np.ndarray | None = None, stop_at_target: bool = True,
    minimum_eigenvalue: float = 1e-8, alpha: float = 0.05,
    return_samples: bool = False,
) -> dict[str, Any]:
    """Run SAIS adaptation, report paper-style recycling, then frozen confirmation.

    The primary estimate is from independent confirmation, not adaptively reused
    samples. The paper-style recycled value is separately diagnostic. The full
    budget includes all training loss evaluations and confirmation. A model
    needs only ``dimension`` and vectorized ``loss(z)``; an evaluation counter
    is checked if supplied. This module never reads quadratic coefficients.
    """
    if (not isinstance(budget, (int, np.integer)) or budget < 256
            or not isinstance(components, (int, np.integer)) or components < 1
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("budget>=256, components>=1 and max_iterations>=1 must be integers")
    scalars = (threshold, pilot_fraction, rho, forgetting, confirmation_defense, minimum_eigenvalue, alpha)
    if not np.isfinite(scalars).all():
        raise ValueError("all numeric options must be finite")
    if (not 0 < pilot_fraction < 1 or not 0 < rho < 1 or not 0 < forgetting < 1
            or not 0 < confirmation_defense < 1 or minimum_eigenvalue <= 0 or not 0 < alpha < 1):
        raise ValueError("invalid adaptation, defense, floor or confidence parameter")
    d = model.dimension
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("model.dimension must be a positive integer")
    pilot_budget = int(math.floor(budget * pilot_fraction))
    if pilot_budget < 2 * components or budget - pilot_budget < 2:
        raise ValueError("budget split cannot support training and independent confirmation")
    rounds = min(max_iterations, max(1, pilot_budget // (32 * components)))
    if samples_per_component is None:
        k = pilot_budget // (components * rounds)
    else:
        if not isinstance(samples_per_component, (int, np.integer)) or samples_per_component < 2:
            raise ValueError("samples_per_component must be an integer >=2")
        k = int(samples_per_component)
        rounds = min(max_iterations, pilot_budget // (components * k))
        if rounds < 1:
            raise ValueError("requested per-component sample count exceeds training budget")
    start = time.perf_counter()
    before = getattr(model, "evaluations", None)
    init_seed, pilot_seed, confirmation_seed = np.random.SeedSequence(seed).spawn(3)
    initial_rng = np.random.default_rng(init_seed)
    rng = np.random.default_rng(pilot_seed)
    means = (initial_rng.uniform(-1, 1, size=(components, d)) if initial_means is None
             else np.asarray(initial_means, dtype=float).copy())
    covariances = (np.tile(np.eye(d), (components, 1, 1)) if initial_covariances is None
                   else np.asarray(initial_covariances, dtype=float).copy())
    if (means.shape != (components, d) or covariances.shape != (components, d, d)
            or not np.isfinite(means).all() or not np.isfinite(covariances).all()):
        raise ValueError("initial proposal arrays have invalid shapes or nonfinite values")
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
            raise ValueError("initial covariances must be symmetric")
        np.linalg.cholesky(covariance)
    initial = {"means": means.tolist(), "covariances": covariances.tolist()}
    previous = math.inf
    levels, level_estimates = [], []
    used = 0
    reached = False
    for iteration in range(1, rounds + 1):
        x, sources = _draw_stratified(rng, means, covariances, k)
        scores = threshold - _loss(model, x)
        used += len(x)
        logweights, assigned = _dm_logweights(x, means, covariances)
        failed = scores < 0  # Strict task event, equal to paper's event for continuous S.
        log_estimate = (float(logsumexp(logweights[failed])) - math.log(len(x))) if np.any(failed) else -math.inf
        level_estimate = math.exp(log_estimate) if np.isfinite(log_estimate) else 0.0
        if not np.isfinite(level_estimate):
            raise FloatingPointError("nonfinite paper-style deterministic mixture estimate")
        level_estimates.append(level_estimate)
        boundary, threshold_diagnostic = _threshold(scores, sources, previous, rho, components)
        updates = []
        next_means, next_covariances = means.copy(), covariances.copy()
        for j in range(components):
            local = (scores <= previous) & (assigned == j)
            next_means[j], next_covariances[j], diagnostic = _update_component(
                x[local], scores[local], logweights[local], means[j], covariances[j],
                boundary, iteration, minimum_eigenvalue,
            )
            updates.append(diagnostic)
        levels.append({
            "iteration": iteration, "previous_boundary": previous if np.isfinite(previous) else None,
            "boundary": boundary if np.isfinite(boundary) else None,
            "samples_per_component": k, "evaluations": len(x),
            "paper_level_estimate": level_estimate, "failure_samples": int(failed.sum()),
            "threshold": threshold_diagnostic, "components": updates,
        })
        means, covariances = next_means, next_covariances
        previous = boundary
        reached = boundary <= 0
        if stop_at_target and reached:
            break
    confirmation_count = int(budget - used)
    x, weights = _draw_confirmation(np.random.default_rng(confirmation_seed), means,
                                    covariances, confirmation_count, confirmation_defense)
    losses = _loss(model, x)
    y = weights * (losses > threshold)
    estimate = float(y.mean())
    variance = float(y.var(ddof=1))
    se = math.sqrt(variance / confirmation_count)
    radius = float(norm.ppf(1 - alpha / 2) * se)
    # Known defense gives a genuine fixed-sample bound. This is often very
    # conservative at rare probabilities; no precision claim hides that cost.
    bound = 1 / confirmation_defense
    log = math.log(4 / alpha)
    bernstein = math.sqrt(2 * variance * log / confirmation_count) + 7 * bound * log / (3 * (confirmation_count - 1))
    count = int(np.count_nonzero(y))
    observed = getattr(model, "evaluations", None)
    if before is not None and observed is not None and observed - before != budget:
        raise RuntimeError("model evaluation counter disagrees with charged SAIS budget")
    warnings = []
    if not reached:
        warnings.append("Training budget exhausted before subset threshold reached failure")
    if count < 30:
        warnings.append("Too few observed tail contributions for reliable asymptotic confirmation interval")
    warnings.append("Independent reproduction with documented paper ambiguities; numerical tables not reproduced")
    result = {
        "method": "sais_frozen_confirmation", "seed": int(seed), "threshold": float(threshold),
        "budget": int(budget), "evaluations": int(budget), "pilot_evaluations": used,
        "confirmation_samples": confirmation_count, "estimate": estimate, "standard_error": se,
        "confidence_interval": [max(0.0, estimate - radius), min(1.0, estimate + radius)],
        "interval_kind": "asymptotic IID normal interval from independent frozen confirmation",
        "finite_sample_interval": [max(0.0, estimate - bernstein), min(1.0, estimate + bernstein)],
        "finite_sample_interval_kind": "fixed-budget empirical Bernstein with known defensive weight bound",
        "confidence": 1 - alpha, "observation_bound": bound,
        "paper_recycled_estimate": _recycle(level_estimates, forgetting),
        "paper_last_level_estimate": level_estimates[-1],
        "paper_recycled_uncertainty": "not asserted; adaptive threshold stopping/recycling requires complete-run replication",
        "paper_stopping_random": bool(stop_at_target), "target_threshold_reached": reached,
        "iterations": len(levels), "max_planned_iterations": rounds,
        "samples_per_component": k, "rho": rho, "forgetting": forgetting,
        "confirmation_defense": confirmation_defense, "initial_proposals": initial,
        "frozen_means": means.tolist(), "frozen_covariances": covariances.tolist(),
        "levels": levels, "seconds": time.perf_counter() - start,
        "diagnostics": {"tail_hits": count,
                        "tail_ess": float(y.sum() ** 2 / (y @ y)) if y @ y > 0 else 0.0,
                        "max_weight": float(weights.max()), "warnings": warnings},
        "fidelity": "SAIS Algorithm 1 reproduction with explicit disambiguations and independent confirmation extension",
    }
    if return_samples:
        result.update(losses=losses, weights=weights, samples=x, observations=y)
    return result

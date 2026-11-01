"""Established rare-event baselines in a standard-normal latent space.

All thresholds use the strict event ``loss > threshold``. Adaptive importance
samplers discard their pilot observations, charge them to the budget, freeze
the fitted proposal, and use independent confirmation observations. These are
deliberately described as CE variants, not as reproductions of SAIS.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import beta as beta_dist
from scipy.stats import qmc, t as student_t


def _loss(model: Any, z: np.ndarray) -> np.ndarray:
    value = np.asarray(model.loss(z), dtype=float)
    if value.shape != (len(z),) or not np.all(np.isfinite(value)):
        raise ValueError("model.loss must return one finite scalar per input row")
    return value


def _log_normal(z: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Return n-by-k diagonal Gaussian log densities without n-by-k-by-d storage."""
    out = np.empty((len(z), len(means)))
    constant = z.shape[1] * math.log(2 * math.pi)
    for j, (mean, variance) in enumerate(zip(means, variances)):
        out[:, j] = -0.5 * (
            constant + np.log(variance).sum()
            + np.sum((z - mean) ** 2 / variance, axis=1)
        )
    return out


def _target_logpdf(z: np.ndarray) -> np.ndarray:
    return -0.5 * (z.shape[1] * math.log(2 * math.pi) + np.sum(z * z, axis=1))


def _draw_mixture(
    rng: np.random.Generator, n: int, means: np.ndarray,
    variances: np.ndarray, masses: np.ndarray, defense: float,
) -> tuple[np.ndarray, np.ndarray]:
    """IID random-mixture draws, with full-mixture likelihood corrections."""
    d = means.shape[1]
    all_means = np.vstack((np.zeros((1, d)), means))
    all_vars = np.vstack((np.ones((1, d)), variances))
    all_mass = np.r_[defense, (1 - defense) * masses]
    labels = rng.choice(len(all_mass), size=n, p=all_mass)
    z = all_means[labels] + np.sqrt(all_vars[labels]) * rng.standard_normal((n, d))
    logq = logsumexp(_log_normal(z, all_means, all_vars) + np.log(all_mass), axis=1)
    return z, np.exp(_target_logpdf(z) - logq)


def _cp(k: int, n: int, alpha: float) -> list[float]:
    return [
        0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1)),
        1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)),
    ]


def _iid_result(losses: np.ndarray, weights: np.ndarray, threshold: float,
                alpha: float) -> dict[str, Any]:
    y = weights * (losses > threshold)
    n = len(y)
    estimate = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    hits = int(np.count_nonzero(y))
    ess = float(y.sum() ** 2 / np.dot(y, y)) if np.any(y) else 0.0
    radius = float(student_t.ppf(1 - alpha / 2, n - 1) * se) if n > 1 else math.inf
    return {
        "estimate": estimate, "standard_error": se,
        "confidence_interval": [max(0.0, estimate - radius), min(1.0, estimate + radius)],
        "losses": losses, "weights": weights,
        "diagnostics": {
            "confirmation_samples": n, "tail_hits": hits, "tail_ess": ess,
            "max_weight": float(weights.max()),
            "weight_ess": float(weights.sum() ** 2 / np.dot(weights, weights)),
            "uncertainty": "asymptotic IID Student interval; not a finite-sample guarantee",
            "warnings": ["Too few observed tail contributions for reliable asymptotic inference"]
            if hits < 30 else [],
        },
    }


def _weighted_gmm(
    rng: np.random.Generator, x: np.ndarray, weights: np.ndarray,
    k: int, floor: float, iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted EM diagonal mixture; weighted k-means++ initialization."""
    weights = weights / weights.sum()
    k = min(k, len(x))
    centers = [x[rng.choice(len(x), p=weights)]]
    distances = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        probability = weights * distances
        probability = probability / probability.sum() if probability.sum() > 0 else weights
        centers.append(x[rng.choice(len(x), p=probability)])
        distances = np.minimum(distances, np.sum((x - centers[-1]) ** 2, axis=1))
    means = np.asarray(centers)
    global_mean = weights @ x
    global_var = np.maximum(weights @ (x - global_mean) ** 2, floor)
    variances = np.tile(global_var, (k, 1))
    masses = np.full(k, 1 / k)
    previous = -math.inf
    for _ in range(iterations):
        logresp = _log_normal(x, means, variances) + np.log(masses)
        lognorm = logsumexp(logresp, axis=1)
        objective = float(weights @ lognorm)
        resp = np.exp(logresp - lognorm[:, None]) * weights[:, None]
        nk = resp.sum(axis=0)
        # A small positive mass prevents numerically disappearing components.
        masses = np.maximum(nk, 1e-8)
        masses /= masses.sum()
        for j in range(k):
            if nk[j] <= 1e-10:
                means[j] = x[rng.choice(len(x), p=weights)]
                variances[j] = global_var
            else:
                means[j] = resp[:, j] @ x / nk[j]
                variances[j] = np.maximum(resp[:, j] @ (x - means[j]) ** 2 / nk[j], floor)
        if abs(objective - previous) <= 1e-7 * (1 + abs(objective)):
            break
        previous = objective
    return means, variances, masses


def _cross_entropy(model: Any, method: str, budget: int, seed: int,
                   threshold: float, alpha: float, **options: Any) -> dict[str, Any]:
    pilot_seed, confirmation_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(pilot_seed)
    d = model.dimension
    pilot_budget = budget // 3
    rounds = min(int(options.get("rounds", 5)), max(1, pilot_budget // 32))
    sizes = np.full(rounds, pilot_budget // rounds, dtype=int)
    sizes[: pilot_budget % rounds] += 1
    defense = float(options.get("defense", 0.1))
    floor = float(options.get("variance_floor", 0.05))
    elite_fraction = float(options.get("elite_fraction", 0.1))
    k = 1 if method == "ce_single" else int(options.get("components", 4))
    if not 0 < defense < 1 or not 0 < elite_fraction < 1 or floor <= 0 or k < 1:
        raise ValueError("Invalid CE defense, elite fraction, covariance floor, or component count")
    means, variances, masses = np.zeros((1, d)), np.ones((1, d)), np.ones(1)
    history = []
    for iteration, size in enumerate(sizes):
        z, w = _draw_mixture(rng, int(size), means, variances, masses, defense)
        losses = _loss(model, z)
        cutoff = min(threshold, float(np.quantile(losses, 1 - elite_fraction)))
        elite = losses >= cutoff
        if np.count_nonzero(elite) < 2:
            history.append({"round": iteration, "cutoff": cutoff, "status": "insufficient elites"})
            continue
        x, ew = z[elite], w[elite]
        if k == 1:
            ew = ew / ew.sum()
            mean = ew @ x
            variance = np.maximum(ew @ (x - mean) ** 2, floor)
            means, variances, masses = mean[None, :], variance[None, :], np.ones(1)
        else:
            means, variances, masses = _weighted_gmm(rng, x, ew, k, floor)
        history.append({"round": iteration, "cutoff": cutoff,
                        "elites": int(elite.sum()), "components": len(means)})
    z, w = _draw_mixture(np.random.default_rng(confirmation_seed), budget - pilot_budget,
                          means, variances, masses, defense)
    out = _iid_result(_loss(model, z), w, threshold, alpha)
    out["diagnostics"].update({
        "pilot_evaluations": pilot_budget, "adaptation": history,
        "proposal_means": means.tolist(), "proposal_variances": variances.tolist(),
        "proposal_masses": masses.tolist(), "defense": defense,
        "variant": "weighted elite CE with diagonal covariance and independent frozen confirmation",
    })
    return out


def _subset(model: Any, budget: int, seed: int, threshold: float,
            **options: Any) -> dict[str, Any]:
    """Conventional adaptive-threshold subset simulation with pCN transitions.

    Its product estimator has finite-sample bias and correlated observations.
    No within-run standard error or confidence interval is asserted.
    """
    rng = np.random.default_rng(seed)
    d = model.dimension
    n = min(budget, int(options.get("level_samples", max(32, budget // 6))))
    rho = float(options.get("conditional_probability", 0.1))
    step = float(options.get("pcn_step", 0.7))
    if n < 4 or not 0 < rho < 1 or not 0 < step <= 1:
        raise ValueError("Invalid subset level size, conditional probability, or pCN step")
    z = rng.standard_normal((n, d))
    losses = _loss(model, z)
    used, product, levels = n, 1.0, []
    status = "budget exhausted before final threshold"
    previous = -math.inf
    while True:
        order = np.argsort(losses)
        desired = max(2, int(math.ceil(rho * n)))
        boundary = float(losses[order[-desired]])
        if boundary >= threshold:
            status = "final threshold reached"
            break
        selected = np.flatnonzero(losses >= boundary)
        if boundary <= previous or len(selected) == n:
            status = "no progress: tied intermediate losses"
            break
        required = n - len(selected)
        if used + required > budget:
            break
        factor = len(selected) / n
        product *= factor
        previous = boundary
        seeds = selected[rng.permutation(len(selected))]
        chain_count = len(seeds)
        current = z[seeds].copy()
        current_losses = losses[seeds].copy()
        # Round-robin chains retain every seed and every subsequent state.
        output_z = [current.copy()]
        output_losses = [current_losses.copy()]
        remaining = required
        accepted, attempted = 0, 0
        while remaining:
            batch = min(chain_count, remaining)
            proposal = np.sqrt(1 - step * step) * current[:batch] + step * rng.standard_normal((batch, d))
            proposal_losses = _loss(model, proposal)
            accept = proposal_losses >= boundary
            current[:batch][accept] = proposal[accept]
            current_losses[:batch][accept] = proposal_losses[accept]
            output_z.append(current[:batch].copy())
            output_losses.append(current_losses[:batch].copy())
            accepted += int(accept.sum())
            attempted += batch
            remaining -= batch
            used += batch
        z, losses = np.concatenate(output_z), np.concatenate(output_losses)
        levels.append({"boundary": boundary, "conditional_fraction": factor,
                       "seeds": chain_count, "pcn_acceptance": accepted / attempted})
    estimate = product * float(np.mean(losses > threshold))
    return {
        "estimate": estimate, "standard_error": None, "confidence_interval": None,
        "losses": np.empty(0), "weights": np.empty(0),
        "diagnostics": {
            "status": status, "levels": levels, "level_samples": n,
            "final_conditional_exceedance": float(np.mean(losses > threshold)),
            "unused_budget": budget - used,
            "uncertainty": "requires independent complete-run replication; no IID interval",
            "warnings": ["Adaptive subset product has finite-sample bias and correlated samples"]
            + ([] if status == "final threshold reached" else [status]),
        },
    }


def run_baseline(model: Any, method: str, budget: int, seed: int,
                 threshold: float, **options: Any) -> dict[str, Any]:
    """Estimate P(loss(Z)>threshold), Z~N(0,I), under a total evaluation budget.

    ``model`` supplies ``dimension`` and vectorized ``loss(z)``. Every call is
    charged through that method. Returned loss/weight arrays contain estimation
    observations only, never CE training samples. Subset simulation returns no
    weighted-loss sample because its conditional samples do not have IID target
    weights. ``alpha`` defaults to .05. No method silently rounds its budget up.
    """
    if not isinstance(budget, (int, np.integer)) or budget < 32:
        raise ValueError("budget must be an integer >=32")
    if model.dimension < 1 or not np.isfinite(threshold):
        raise ValueError("positive dimension and finite threshold required")
    alpha = float(options.pop("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    start = time.perf_counter()
    before = getattr(model, "evaluations", None)
    rng = np.random.default_rng(seed)
    d = model.dimension
    if method == "mc":
        losses = _loss(model, rng.standard_normal((budget, d)))
        out = _iid_result(losses, np.ones(budget), threshold, alpha)
        out["confidence_interval"] = _cp(int(np.sum(losses > threshold)), budget, alpha)
        out["diagnostics"]["uncertainty"] = "exact fixed-sample binomial Clopper-Pearson interval"
    elif method == "antithetic":
        pairs = budget // 2
        z = rng.standard_normal((pairs, d))
        losses = _loss(model, np.vstack((z, -z)))
        out = _iid_result(losses, np.ones(2 * pairs), threshold, alpha)
        pair_values = ((losses[:pairs] > threshold).astype(float)
                       + (losses[pairs:] > threshold).astype(float)) / 2
        se = float(pair_values.std(ddof=1) / np.sqrt(pairs))
        radius = float(student_t.ppf(1 - alpha / 2, pairs - 1) * se)
        out["standard_error"] = se
        out["confidence_interval"] = [max(0.0, out["estimate"] - radius), min(1.0, out["estimate"] + radius)]
        out["diagnostics"].update({"independent_pairs": pairs, "unused_budget": budget % 2,
                                   "uncertainty": "asymptotic Student interval across independent antithetic pairs"})
    elif method == "sobol":
        repetitions = int(options.get("scrambles", 8))
        if repetitions < 8 or budget < 2 * repetitions:
            raise ValueError("Sobol requires >=8 independent scrambles and >=2 draws per scramble")
        power = int(math.floor(math.log2(budget // repetitions)))
        seeds = np.random.SeedSequence(seed).spawn(repetitions)
        blocks = []
        for child in seeds:
            engine = qmc.Sobol(d=d, scramble=True, seed=np.random.default_rng(child))
            u = engine.random_base2(power)
            z = ndtri(np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
            blocks.append(_loss(model, z))
        estimates = np.array([np.mean(block > threshold) for block in blocks])
        losses = np.concatenate(blocks)
        out = _iid_result(losses, np.ones(len(losses)), threshold, alpha)
        se = float(estimates.std(ddof=1) / np.sqrt(repetitions))
        estimate = float(estimates.mean())
        radius = float(student_t.ppf(1 - alpha / 2, repetitions - 1) * se)
        out.update({"estimate": estimate, "standard_error": se,
                    "confidence_interval": [max(0.0, estimate - radius), min(1.0, estimate + radius)]})
        out["diagnostics"].update({
            "scrambles": repetitions, "points_per_scramble": 2 ** power,
            "replicate_estimates": estimates.tolist(), "unused_budget": budget - len(losses),
            "uncertainty": "approximate Student interval across independent scrambled Sobol estimates",
        })
    elif method == "fixed_is":
        means = np.asarray(options.get("means", np.vstack((4 * np.eye(d), -4 * np.eye(d)))), dtype=float)
        if means.ndim != 2 or means.shape[1] != d or len(means) < 1 or not np.all(np.isfinite(means)):
            raise ValueError("means must be a finite k-by-d array")
        defense = float(options.get("defense", 0.1))
        if not 0 < defense < 1:
            raise ValueError("defense must lie strictly between zero and one")
        z, w = _draw_mixture(rng, budget, means, np.ones_like(means), np.full(len(means), 1 / len(means)), defense)
        out = _iid_result(_loss(model, z), w, threshold, alpha)
        out["diagnostics"].update({"proposal_means": means.tolist(), "defense": defense,
                                   "variant": "fixed unit-covariance Gaussian mean mixture"})
    elif method in ("ce_single", "ce_mixture"):
        out = _cross_entropy(model, method, budget, seed, threshold, alpha, **options)
    elif method == "subset":
        out = _subset(model, budget, seed, threshold, **options)
    else:
        raise ValueError(f"Unknown baseline {method!r}")
    after = getattr(model, "evaluations", None)
    observed_count = after - before if before is not None and after is not None else None
    out.update({"method": method, "seed": int(seed), "budget": int(budget),
                "evaluations": int(observed_count) if observed_count is not None else None,
                "seconds": time.perf_counter() - start})
    if observed_count is not None and observed_count > budget:
        raise RuntimeError("baseline exceeded its evaluation budget")
    return out

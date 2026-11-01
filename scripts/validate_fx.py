#!/usr/bin/env python3
"""Frozen chronological FX model check, separate from simulation validation.

Run from the repository root: python scripts/validate_fx.py
No network, portfolio optimization, evaluation-period fitting, or rolling model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import optimize, stats


CONFIDENCES = (0.95, 0.975, 0.99)
TRAIN_END = "2014-12-31"
EVALUATION_START = "2015-01-01"
EVALUATION_END = "2025-12-31"
BOOTSTRAP_SEED = 613000
BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATIONS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified_returns(data_dir: Path):
    provenance_path = data_dir / "fx_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    path = data_dir / "fx_returns.csv"
    if sha256(path) != provenance["outputs"]["fx_returns.csv"]:
        raise ValueError("fx_returns.csv differs from its source provenance")
    for source in provenance["sources"].values():
        if sha256(data_dir / source["path"]) != source["sha256"]:
            raise ValueError("raw source differs from its recorded hash")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    dates = [r["date"] for r in rows]
    if dates != sorted(set(dates)):
        raise ValueError("return dates must be strictly ordered and unique")
    for row in rows:
        if not row["previous_date"] < row["date"]:
            raise ValueError("invalid return interval chronology")
        expected_split = "train" if row["date"] <= TRAIN_END else "evaluation"
        if row["split"] != expected_split:
            raise ValueError("saved split conflicts with frozen date rule")
        returns = [float(row[f"simple_{currency}"]) for currency in ("eur", "jpy", "gbp")]
        if not np.isclose(float(row["basket_simple_loss"]), -np.mean(returns), atol=1e-16, rtol=1e-12):
            raise ValueError("basket loss differs from equal-value exposure definition")
    train = [r for r in rows if "2000-01-01" <= r["date"] <= TRAIN_END]
    evaluation = [r for r in rows if EVALUATION_START <= r["date"] <= EVALUATION_END]
    if not train or not evaluation:
        raise ValueError("both chronological partitions are required")
    return train, evaluation, provenance


def fit_train_models(train_losses: np.ndarray) -> dict:
    """Fit both prespecified families using ONLY the supplied training array."""
    train_losses = np.asarray(train_losses, dtype=float)
    if train_losses.ndim != 1 or len(train_losses) < 30 or not np.all(np.isfinite(train_losses)):
        raise ValueError("finite one-dimensional training losses are required")
    center = float(np.mean(train_losses))
    scale = float(np.std(train_losses, ddof=0))
    if scale <= 0:
        raise ValueError("training losses must have positive scale")
    standardized = (train_losses - center) / scale

    def objective(theta):
        df = 2 + np.exp(theta[0])
        return -float(np.sum(stats.t.logpdf(standardized, df, loc=theta[1], scale=np.exp(theta[2]))))

    # Fixed training-only multistart MLE. Restrict df to [2.05,200] so both
    # mean and variance exist; this modelling choice is not learned on eval.
    runs = []
    for initial_df in (3.0, 6.0, 12.0, 30.0):
        initial_scale = math.sqrt((initial_df - 2) / initial_df)
        opt = optimize.minimize(
            objective, [math.log(initial_df - 2), 0.0, math.log(initial_scale)],
            method="L-BFGS-B",
            bounds=[(math.log(0.05), math.log(198)), (-5.0, 5.0), (-8.0, 4.0)],
            options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 2000},
        )
        runs.append(opt)
    successful = [opt for opt in runs if opt.success and np.isfinite(opt.fun)]
    if not successful:
        raise RuntimeError("all Student training fits failed")
    optimum = min(successful, key=lambda opt: opt.fun)
    theta = optimum.x
    student = {
        "family": "student_t", "df": float(2 + np.exp(theta[0])),
        "location": float(center + scale * theta[1]),
        "scale": float(scale * np.exp(theta[2])),
        "fit": "training-only maximum likelihood; df constrained [2.05,200]",
        "training_log_likelihood": float(-optimum.fun - len(train_losses) * np.log(scale)),
        "optimizer_success": bool(optimum.success),
        "successful_starts": len(successful),
        "starts": [
            {"initial_df": initial_df, "success": bool(opt.success),
             "standardized_negative_log_likelihood": float(opt.fun)}
            for initial_df, opt in zip((3.0, 6.0, 12.0, 30.0), runs)
        ],
    }
    return {
        "gaussian": {"family": "gaussian", "location": center, "scale": scale,
                     "fit": "training-only maximum likelihood; scale uses ddof=0",
                     "training_log_likelihood": float(np.sum(stats.norm.logpdf(train_losses, loc=center, scale=scale)))},
        "student_t": student,
    }


def analytic_risk(model: dict, confidence: float) -> tuple[float, float]:
    beta = 1 - confidence
    loc, scale = model["location"], model["scale"]
    if model["family"] == "gaussian":
        quantile = stats.norm.ppf(confidence)
        return float(loc + scale * quantile), float(loc + scale * stats.norm.pdf(quantile) / beta)
    df = model["df"]
    if df <= 1:
        raise ValueError("Student Expected Shortfall requires df > 1")
    quantile = stats.t.ppf(confidence, df)
    es = loc + scale * (df + quantile**2) / (df - 1) * stats.t.pdf(quantile, df) / beta
    return float(loc + scale * quantile), float(es)


def diagnostic_series(losses: np.ndarray, q: float, es: float, confidence: float):
    beta = 1 - confidence
    hits = (losses > q).astype(float)
    excess = np.maximum(losses - q, 0)
    var_residual = hits - beta
    es_residual = es - q - excess / beta
    if es <= 0:
        raise ValueError("upper-loss FZ0 convention requires positive ES")
    fz0 = excess / (beta * es) + q / es + np.log(es) - 1
    return hits, var_residual, es_residual, fz0


def circular_block_bootstrap_means(values: np.ndarray, block_length: int = BLOCK_LENGTH,
                                   replications: int = BOOTSTRAP_REPLICATIONS,
                                   seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Shared indices preserve paired comparisons and short-run dependence."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap observations must be a finite matrix")
    n, width = values.shape
    if not 1 <= block_length <= n or replications < 2:
        raise ValueError("invalid block bootstrap configuration")
    nblocks = math.ceil(n / block_length)
    rng = np.random.default_rng(seed)
    answer = np.empty((replications, width))
    offsets = np.arange(block_length)
    for first in range(0, replications, 25):
        last = min(first + 25, replications)
        starts = rng.integers(n, size=(last - first, nblocks))
        indices = ((starts[:, :, None] + offsets) % n).reshape(last - first, -1)[:, :n]
        answer[first:last] = np.mean(values[indices], axis=1)
    return answer


def run(data_dir: Path, output_dir: Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be new or empty; use a fresh --output")
    train_rows, eval_rows, provenance = read_verified_returns(data_dir)
    train_losses = np.asarray([float(row["basket_simple_loss"]) for row in train_rows])
    # This is the only array passed to the fitting function.
    models = fit_train_models(train_losses)
    eval_losses = np.asarray([float(row["basket_simple_loss"]) for row in eval_rows])
    columns, series, diagnostics, forecast_rows = [], [], [], []
    for name, model in models.items():
        for confidence in CONFIDENCES:
            q, es = analytic_risk(model, confidence)
            hits, var_residual, es_residual, fz0 = diagnostic_series(eval_losses, q, es, confidence)
            key = f"{name}:{confidence:.3f}"
            for label, vector in (("var_residual", var_residual), ("es_residual", es_residual), ("fz0", fz0)):
                columns.append(f"{key}:{label}")
                series.append(vector)
            diagnostics.append({
                "key": key, "model": name, "confidence": confidence, "tail_probability": 1 - confidence,
                "var": q, "es": es, "exceedances": int(np.sum(hits)),
                "evaluation_observations": len(eval_rows),
                "expected_exceedances_under_model": len(eval_rows) * (1 - confidence),
                "exceedance_rate": float(np.mean(hits)),
                "mean_var_residual": float(np.mean(var_residual)),
                "mean_es_residual": float(np.mean(es_residual)), "mean_fz0": float(np.mean(fz0)),
            })
            for row, loss, hit, vr, er, score in zip(eval_rows, eval_losses, hits, var_residual, es_residual, fz0):
                forecast_rows.append({
                    "date": row["date"], "previous_date": row["previous_date"],
                    "calendar_days": int(row["calendar_days"]), "model": name,
                    "confidence": confidence, "loss": float(loss), "var": q, "es": es,
                    "exceeded": int(hit), "var_residual": float(vr),
                    "es_residual": float(er), "fz0": float(score),
                })
    values = np.column_stack(series)
    bootstrap = circular_block_bootstrap_means(values)
    for diagnostic in diagnostics:
        for label in ("var_residual", "es_residual", "fz0"):
            index = columns.index(f"{diagnostic['key']}:{label}")
            interval = np.quantile(bootstrap[:, index], [0.025, 0.975])
            diagnostic[f"{label}_mean_blockbootstrap_95_interval"] = interval.tolist()
    paired_scores = []
    for confidence in CONFIDENCES:
        gi = columns.index(f"gaussian:{confidence:.3f}:fz0")
        ti = columns.index(f"student_t:{confidence:.3f}:fz0")
        difference = values[:, ti] - values[:, gi]
        boot_difference = bootstrap[:, ti] - bootstrap[:, gi]
        paired_scores.append({
            "confidence": confidence, "difference": "student_t minus gaussian FZ0; lower is better",
            "mean_difference": float(np.mean(difference)),
            "paired_blockbootstrap_95_interval": np.quantile(boot_difference, [0.025, 0.975]).tolist(),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts_path = output_dir / "forecasts.csv"
    forecast_rows.sort(key=lambda row: (row["date"], row["model"], row["confidence"]))
    with forecasts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(forecast_rows[0]))
        writer.writeheader()
        writer.writerows(forecast_rows)
    # Store the actual 1,000 means for all diagnostics, not only reported intervals.
    bootstrap_path = output_dir / "bootstrap_means.csv"
    with bootstrap_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replication", *columns])
        writer.writerows((i, *row) for i, row in enumerate(bootstrap))
    summary = {
        "schema_version": 1,
        "purpose": "chronological validation of frozen financial distribution assumptions; not rare-event engine validation",
        "portfolio": provenance["selection"]["basket"],
        "train": {"start": train_rows[0]["date"], "end": train_rows[-1]["date"], "observations": len(train_rows)},
        "evaluation": {"start": eval_rows[0]["date"], "end": eval_rows[-1]["date"], "observations": len(eval_rows)},
        "fit_policy": "fit both prespecified unconditional families on 2000-2014 only; freeze all parameters for 2015-2025; no evaluation tuning",
        "models": models,
        "diagnostics": diagnostics,
        "paired_fz0_comparisons": paired_scores,
        "bootstrap": {"type": "circular moving block percentile intervals for means",
                      "block_length_observations": BLOCK_LENGTH, "replications": BOOTSTRAP_REPLICATIONS,
                      "seed": BOOTSTRAP_SEED, "interval_level": 0.95,
                      "paired_indices": True,
                      "assumptions": "approximate weak-dependence/stationarity inference conditional on frozen fitted parameters; does not include training estimation uncertainty or guarantee performance under regime change",
                      "multiplicity": "pointwise intervals; no familywise correction or model-selection claim"},
        "formulas": {"var_residual": "1{L>q} - beta",
                     "es_residual": "e-q-max(L-q,0)/beta",
                     "fz0_upper_loss": "max(L-q,0)/(beta*e)+q/e+log(e)-1; e>0; lower is better"},
        "source_hashes": {"fx_returns.csv": sha256(data_dir / "fx_returns.csv"),
                          "fx_provenance.json": sha256(data_dir / "fx_provenance.json"),
                          "validate_fx.py": sha256(Path(__file__))},
        "artifact_hashes": {"forecasts.csv": sha256(forecasts_path), "bootstrap_means.csv": sha256(bootstrap_path)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "limits": [
            "current revised H.10 source vintage, not historical point-in-time releases",
            "observation intervals include weekends and holidays; no carry, financing or transaction costs",
            "frozen unconditional basket fits do not model conditional volatility or full cross-asset dynamics",
            "confidence intervals assess historical mean diagnostics approximately, not Monte Carlo error",
            "no empirical validation of 1e-5 or one-in-a-million loss probabilities",
            "neither a narrow simulation interval nor acceptable historical calibration establishes a correct future financial model",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results" / "historical")
    args = parser.parse_args()
    summary = run(args.data_dir, args.output)
    print(json.dumps({"output": str(args.output), "models": summary["models"],
                      "diagnostics": summary["diagnostics"],
                      "paired_fz0_comparisons": summary["paired_fz0_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()

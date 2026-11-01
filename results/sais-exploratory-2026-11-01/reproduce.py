"""Rerun isolated SAIS exploration into a NEW destination.

Example, from the repository root, with package installed or PYTHONPATH=src:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python results/sais-exploratory-2026-11-01/reproduce.py --output /tmp/sais-new

Use --suite paper-s4 for the paper's N=5, K=3000, rho=.2 high-dimensional
settings (dimension 10). Neither suite modifies the frozen primary evaluation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import tailsentry.sais as implementation


class Case:
    def __init__(self, name, dimension):
        self.name, self.dimension, self.evaluations = name, dimension, 0

    def loss(self, z):
        self.evaluations += len(z)
        if self.name == "paper_S1":
            a, b = z[:, 0], z[:, 1]
            return -np.minimum(2-b+np.exp(-a*a/10)+(a/5)**4, 4.5-a*b)
        if self.name == "paper_S2":
            a, b = z[:, 0], z[:, 1]
            return -np.minimum.reduce([
                4+(a-b)**2/10-(a+b)/np.sqrt(2),
                4+(a-b)**2/10+(a+b)/np.sqrt(2),
                (a-b)+7/np.sqrt(2)+1, (b-a)+7/np.sqrt(2)+1,
            ])
        return z.sum(axis=1)/np.sqrt(self.dimension)


def references():
    def conditional_s1(x):
        upper = 2+np.exp(-x*x/10)+(x/5)**4
        if x > 0:
            return norm.sf(min(upper, 4.5/x))*norm.pdf(x)
        if x < 0:
            return (norm.sf(upper)+norm.cdf(4.5/x))*norm.pdf(x)
        return norm.sf(upper)*norm.pdf(x)
    left, le = quad(conditional_s1, -12, 0, epsabs=2e-13, epsrel=2e-11, limit=500)
    right, re = quad(conditional_s1, 0, 12, epsabs=2e-13, epsrel=2e-11, limit=500)
    vcut = 3.5+1/np.sqrt(2)
    body, be = quad(lambda v: 2*norm.sf(4+.2*v*v)*norm.pdf(v), -vcut, vcut,
                    epsabs=1e-14, epsrel=1e-11)
    values = {"paper_S1": left+right, "paper_S2": 2*norm.sf(vcut)+body,
              "paper_S4_d8": norm.sf(3.5), "paper_S4_d10": norm.sf(3.5)}
    return {key: float(value) for key, value in values.items()}, {
        "S1_reported_quadrature_error": le+re,
        "S1_omitted_target_mass_bound": float(2*norm.sf(12)),
        "S2_reported_quadrature_error": be,
        "method": "independent scalar conditional quadrature for S1/S2; analytic normal survival for S4",
    }


def summarize(rows, truths):
    result = {}
    for name in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == name and "error" not in row]
        truth = truths[name]
        entry = {"truth": truth, "runs": len(selected),
                 "errors": [row for row in rows if row["case"] == name and "error" in row]}
        if not selected:
            result[name] = entry
            continue
        for label in ("estimate", "paper_recycled_estimate", "paper_last_level_estimate"):
            values = np.array([row[label] for row in selected])
            entry[label] = {"mean": float(values.mean()),
                            "rmse": float(np.sqrt(np.mean((values-truth)**2))),
                            "relative_rmse": float(np.sqrt(np.mean((values-truth)**2))/truth),
                            "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
        entry.update({
            "asymptotic_confirmation_coverage": float(np.mean([
                row["confidence_interval"][0] <= truth <= row["confidence_interval"][1]
                for row in selected])),
            "finite_confirmation_coverage": float(np.mean([
                row["finite_sample_interval"][0] <= truth <= row["finite_sample_interval"][1]
                for row in selected])),
            "threshold_reached_fraction": float(np.mean([row["target_threshold_reached"] for row in selected])),
            "mean_pilot_evaluations": float(np.mean([row["pilot_evaluations"] for row in selected])),
            "mean_seconds": float(np.mean([row["seconds"] for row in selected])),
        })
        result[name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["small", "paper-s4"], default="small")
    parser.add_argument("--replications", type=int, default=24)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("at least two independent repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    truths, reference = references()
    if args.suite == "small":
        cases = [("paper_S1", 2, 0.), ("paper_S2", 2, 0.), ("paper_S4_d8", 8, 3.5)]
        first_seed, budget, options = 410200, 16384, {}
    else:
        cases = [("paper_S4_d10", 10, 3.5)]
        first_seed, budget = 410300, 450000
        options = {"components": 5, "samples_per_component": 3000,
                   "max_iterations": 10, "rho": .2}
    source = Path(inspect.getsourcefile(implementation))
    protocol = {"suite": args.suite, "cases": cases, "replications": args.replications,
                "first_seed": first_seed, "budget": budget, "options": options,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "truths": truths, "reference": reference,
                "kind": "exploratory, separate from frozen primary evaluation",
                "process_thread_environment": {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False))
    rows = []
    for name, dimension, threshold in cases:
        for seed in range(first_seed, first_seed+args.replications):
            try:
                row = implementation.estimate_sais(Case(name, dimension), threshold,
                                                   budget=budget, seed=seed, **options)
            except Exception as error:
                row = {"seed": seed, "error": repr(error)}
            rows.append({"case": name, **row})
            (args.output/"runs.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        summary = summarize(rows, truths)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(name, json.dumps(summary[name]), flush=True)


if __name__ == "__main__":
    main()

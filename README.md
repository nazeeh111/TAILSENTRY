# TAILSENTRY

**Rare-loss estimation with explicit structural assumptions, inspectable probability bounds, and a failed breakthrough hypothesis.**

TAILSENTRY is a runnable numerical research package for Gaussian and Student-t maximum-of-quadratic losses. It builds a proved cover of all supported failure regions, samples exact conditional stress scenarios, and integrates a Gaussian risk direction. It returns probability estimates, finite-budget bounds, severity diagnostics, weighted branch contributions and reproducible records. A local terminal-style interface exposes the actual engine.

The project does **not** establish an original general estimator or superiority over the state of the art. Its strongest comparator, conditional scrambled Sobol, defeated the selected candidate on the predefined primary task.

## What the evidence says

| Evidence | Result |
|---|---|
| Frozen evaluation | 17,280 complete runs; eight known-truth problems; 15 methods/variants; two budgets; 64 or 128 independent replications per condition |
| Primary target | **Failed.** Candidate / conditional-Sobol MSE × elapsed-time ratio **38.79**, 95% bootstrap interval **[26.48, 57.32]**; lower is better |
| Separate sequential timing | Same conclusion: ratio **37.37**, interval **[21.67, 64.54]** |
| Useful within-proposal result | Fixed projection improves raw union sampling; primary elapsed-efficiency ratio **0.0252**. This is not the best-comparator result |
| Probability certificates | Primary selected finite bound: 128/128 coverage, mean width **4.865e-6** for probability **1e-4**. Exact coverage-rate interval **[0.9716, 1]** |
| Missing-mode attack | Ordinary intervals cover **3/512** times despite median observed tail ESS about 18,000; conservative bounds abstain |
| Historical model sensitivity | Training-fitted Student vs Gaussian changes 99% FX Expected Shortfall by **21.8%**; both fail 95% VaR calibration under the chosen block-bootstrap diagnostic |
| Novelty | Established union sampling + conditional Monte Carlo + concentration bounds. Original-estimator breakthrough remains unachieved |

Methods share seed labels across conditions, so the full set of runs is not jointly independent. Point and conditional calls are separate information interfaces. Initial timings were locally contended; the randomized-order confirmation is retained. [Protocol](docs/protocol.md), [amendment](docs/protocol-amendment.md), [raw results](results/final/), and [adversarial review](reviews/adversarial-review.md) document these limits.

## Run

Python 3.11+ with NumPy/SciPy. The exact recorded research environment uses Python 3.14.7, NumPy 2.5.4 and SciPy 1.18.1 on an ARM Mac CPU. No API keys, paid services or GPU are required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[research]'
.venv/bin/tailsentry estimate --case delta_gamma_1e4 --budget 16384 --seed 42
.venv/bin/tailsentry serve --port 8768
```

Open **http://127.0.0.1:8768**. The server binds to loopback, rejects cross-origin mutation, and evaluates only supported configurations. Records export as JSON. Changing the form does not update an estimate until you run it. The screen distinguishes finite probability bounds from approximate standard errors and model uncertainty.

An isolated environment is already installed in this working copy. From this directory:

```bash
.venv/bin/tailsentry serve
```

## Python API

```python
from tailsentry.benchmark import make_case
from tailsentry.envelope import estimate_envelope
from tailsentry.risk import estimate_risk
import numpy as np

model, threshold, reference = make_case("delta_gamma_1e4")
result = estimate_envelope(
    model, threshold, budget=16384, seed=42,
    project=True, select=False, severity=True,
)
print(result["estimate"], result["interval"], result["status"])

# User-supplied quantile bracket. This is a biased plug-in estimator;
# its finite-grid quantile bracket and asymptotic diagnostics are separate.
risk = estimate_risk(model, .99, np.linspace(1.8, 2.7, 17),
                     budget=16384, seed=42)
print(risk["var"], risk["es"], risk["var_interval_grid_95"])
```

Supply custom supported losses with `QuadraticModel(B, a, c, scale_df=...)`, where loss is `max_j(c[j] + a[j] @ X + X @ B[j] @ X)`. Correlated factor loadings can be absorbed into these coefficients. Arbitrary black-box simulators do not provide the conditional oracle required by this method. The conventional baselines accept a simpler `dimension` / `loss(z)` interface.

## Mathematical mechanism

If the event is contained, up to target-null sets, in halfspaces `H_k`, sample component `k` with probability `P(H_k)/B`, where `B = sum P(H_k)`. Correct overlap count `M` gives raw contribution `B * 1(event) / M`. This is established union importance sampling, including ALOE when the cover equals the event.

Integrating one latent Gaussian coordinate gives contribution `B * P(event | V,S) / sum P(H_k | V,S)`, bounded by `B`. It is the conditional expectation of the raw contribution, so variance cannot increase at identical proposal/sample count. Equal runtime efficiency is an empirical question, and the primary experiment failed it against conditional Sobol.

The estimator uses independent frozen confirmation and fixed-budget empirical Bernstein intervals. Prefix traces do not permit optional stopping. VaR, ES and tail-mean ratios are finite-sample biased. Severity needs separate moment conditions; an unbiased probability estimate does not imply reliable severity inference.

## Reproduce and inspect

- [Research manuscript, PDF](paper/tailsentry.pdf) and [LaTeX](paper/tailsentry.tex).
- [Reproduction instructions](docs/reproduction.md), [estimator assumptions](docs/mathematics.md), [baseline definitions](docs/baselines.md).
- [Novelty assessment](docs/novelty.md), [modern comparison scope](docs/modern-baseline-scope.md), [SAIS equation concordance](docs/sais-fidelity.md).
- [Original financial data and transformations](docs/data.md), [historical validation](docs/historical-validation.md).
- [Independent mathematical review](reviews/mathematical-review.md), [adversarial review](reviews/adversarial-review.md), [manuscript review](reviews/manuscript-review.md).
- [Evidence-supported résumé wording](docs/accomplishments.md).

Run tests with `.venv/bin/python -m pytest -q`. Run `scripts/verify_artifacts.py` to verify saved hashes. The full study is optional and takes substantially longer than tests; use fresh output directories. The software and research artifacts are public on GitHub. No paper publication, external submission or independent adoption is claimed.

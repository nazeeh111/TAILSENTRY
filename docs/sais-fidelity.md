# SAIS reproduction and confirmation extension

This is independent source, not an author release. The owning reference is
[Helal and Elvira, arXiv:2508.00210v1](https://arxiv.org/html/2508.00210v1).
No published performance table has yet been reproduced. This module is separate
from the frozen primary experiment and makes no superiority claim.

Run programmatically:

```python
from tailsentry.sais import estimate_sais
result = estimate_sais(model, threshold=3.5, budget=16384, seed=410000)
```

The model exposes only its standard-normal latent dimension and `loss(z)`.
Adaptation sees neither exact geometry nor reference answers. Non-Gaussian
physical models are supported only when already represented through this latent
interface, with its numerical limitations preserved.

## Equation concordance

| Source | Implementation | Interpretation |
|---|---|---|
| Algorithm 1, Step 1a | `_draw_stratified` | Equal, deterministic sample counts per full-covariance Gaussian |
| Eq. 4 | `_dm_logweights` | Target density divided by the complete equal-weight mixture |
| Step 1b–2c | `_threshold` | Previous-subset filtering, local lowest-score elites, pooled descending order statistic |
| Eqs. 10–11 | `_dm_logweights` | Hard maximum posterior assignment at the same sample |
| Eq. 13 | `_update_component` | Weighted local mean at the new threshold |
| Eq. 14 | `_update_component` | Untempered covariance centered on the previous mean, as printed |
| Eqs. 15–16 | `_update_component` | Low local ESS triggers `gamma=sigmoid(iteration)`; tempered covariance uses newly fitted mean |
| Eqs. 17–18 | `_lw_coefficient`, `_floor_covariance` | Full covariance update, printed uncentered outer-product numerator, isotropic regularization |
| Eqs. 19–21 | `_recycle` | Normalized geometric combination of per-level DM estimates |

## Necessary disambiguations and safeguards

These choices are explicit because the paper alone does not determine a unique
byte-for-byte implementation.

1. With `S=threshold-loss`, failures have negative scores. “Top” local elites are
   interpreted as the **smallest** scores. The pooled ordering is descending,
   matching the displayed ordering rather than silently substituting a global
   cross-entropy quantile. A zero rank from a floor operation is rounded to one.
2. The displayed posterior denominator has sample-index ambiguity. The prose
   requires evaluating every proposal at the **same point**; that is implemented.
3. Local ESS is computed on the hard-reassigned previous-subset observations.
   New-threshold filtering then selects observations entering the parameter fit.
4. The untempered covariance follows the old-mean center printed in Eq. 14.
   The tempered mean's time index in Eq. 16 is interpreted as the newly fitted
   tempered mean. This difference is exposed in every update diagnostic.
5. Eq. 18's actual local fitting count replaces its generic `K`. Its coefficient
   is clipped to the stated admissible interval `[0,1]`; a zero anisotropy
   denominator uses coefficient one. Every clipping/zero-denominator event is
   recorded. An absolute eigenvalue floor `1e-8` prevents singular proposals.
6. An empty component retains its previous proposal. No hidden resampling,
   respawning, gradient step, or reference-informed recovery is added.
7. Initial means are independent uniform draws on `[-1,1]^d`, with identity
   covariances, following the explicitly specified high-dimensional example.
   Callers may supply different initial arrays and all arrays are recorded.
8. The task's event is strictly `loss > threshold`. Intermediate subsets use
   non-strict inequalities. This matches the paper's failure event except for
   atoms exactly on the boundary, for which the task's convention takes priority.

## Statistical separation

The paper-style adaptation estimate is returned as `paper_recycled_estimate`,
with its final-level value separately available. Target-threshold stopping makes
the level count random. No exact unbiasedness or within-run confidence interval
is asserted for that stopped recycled diagnostic. With a fixed predetermined
level count and deterministic geometric weights, levelwise DM corrections have
the usual conditional-unbiasedness argument; this does not validate optional
stopping or the paper's published numerical results.

The primary `estimate` uses an explicitly identified **extension**: freeze the
learned proposals, independently sample an IID mixture with 10% target defense,
and apply its full mixture likelihood correction. Pilot data are excluded from
confirmation. Every pilot evaluation is charged to the total budget; unused
pilot capacity becomes confirmation capacity. Seeds for initialization,
adaptation, and confirmation are independent SeedSequence children.

The confirmation estimator is unbiased conditional on training and its
contributions are bounded by ten. Its ordinary normal interval is labeled
asymptotic. A separate fixed-sample empirical-Bernstein interval uses that known
bound and can be very wide in a rare tail. High observed effective sample size
does not certify discovery of every catastrophic mode. The module reports
training failures and guards, rather than treating threshold arrival as evidence
of complete coverage.

The confirmation defense is absent from the paper-style pilot estimator and
must not be attributed to the original algorithm. No performance comparison may
combine the recycled point estimate with the confirmation interval.

## Separate exploratory evidence

The source-specific protocol and raw runs are retained in
`results/sais-exploratory-2026-11-01/`. These runs are **not** part of the frozen
primary evaluation and do not establish comparative superiority.

First, 24 independent seeds `410200..410223` per case used total budget 16,384,
four proposals, approximately 136 samples per proposal per pilot level, and
`rho=.1`. Independent scalar conditional quadrature gives probabilities
`.003478946320931294` for the paper's first example and
`.00006419213708727797` for its four-branch example. The normalized linear
eight-dimensional example has analytic probability `.00023262907903552502`.

| Case | Confirmation relative RMSE | Approximate 95% coverage | Reached failure threshold |
|---|---:|---:|---:|
| S1 | .078 | 23/24 | 24/24 |
| S2 | .327 | 5/24 | 24/24 |
| S4, dimension 8 | 1.671 | 8/24 | 1/24 |

The known-bound finite-sample intervals covered truth in all these runs, but
were wide. These negative findings illustrate configuration sensitivity,
undiscovered probability mass, and unreliable asymptotic intervals. They do not
establish finite-sample bias of the independent confirmation estimator, which
remains conditionally unbiased under its stated assumptions.

A second, stronger configuration uses the paper's expressly specified linear
example settings: five proposals, 3,000 samples per proposal per level,
`rho=.2`, dimension 10, and identity initial covariances. Twenty-four fresh seeds
`410300..410323` used a total budget of 450,000, including all pilot calls and
independent confirmation. Every run reached the failure threshold. The
paper-style recycled estimate averaged `.00023265403892048573`, with relative
RMSE `.02576`. The confirmation extension averaged `.00023239758781863702`,
with relative RMSE `.00603` and approximate coverage 23/24. Pilot level counts
were 3–7. These are known-truth reproduction checks, not a reproduction of a
published table or a comparison against another method at the same budget.

Run either suite into a fresh destination from the repository root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python results/sais-exploratory-2026-11-01/reproduce.py \
  --suite small --output /tmp/sais-small-fresh

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python results/sais-exploratory-2026-11-01/reproduce.py \
  --suite paper-s4 --output /tmp/sais-paper-s4-fresh
```

The protocol records the implementation hash, all random seeds, reference
calculation tolerances and process thread limits. Runtime is machine-dependent.
Tests validate full covariance densities against SciPy, mixture likelihoods
against direct formulas and quadrature, update branches, deterministic counts,
sampling separation, exact accounting, invalid inputs, and a known-truth event.

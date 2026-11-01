# Frozen historical FX model check

The financial model assumption changes risk materially: the fitted Student
model's 99% Expected Shortfall is **1.519% of basket value**, versus **1.247%**
for the Gaussian model, a **21.8% increase**. These are analytic risks under
two fitted distributions. They are not Monte Carlo estimates or evidence that
the extreme-tail engine predicts actual future losses correctly.

## Data and protocol

The input is the public Federal Reserve H.10 dataset described in
[data.md](data.md), with raw HTML receipts, aligned currency observations,
explicit yen inversion and SHA-256 provenance. It contains equal USD-value
exposures to EUR, JPY and GBP, reset at each observation interval. Basket loss
is minus the mean of the three simple returns. All observations are used;
there is no crisis selection, weight optimization or calendar forward fill.

The date rule is fixed: train on 3,769 returns ending 4 January 2000 through
31 December 2014; evaluate on 2,748 returns ending 2 January 2015 through
31 December 2025. Input dates are checked for order, uniqueness and partition
consistency before fitting. Model fitting receives only the training array.
An evaluation interval may begin with its preceding training-period quote.

Two prespecified **unconditional univariate basket-loss** families are fitted
by training maximum likelihood and then frozen for the entire evaluation
period:

- Gaussian: location −0.001660% and scale 0.468508%.
- Student t: location −0.0000374%, scale 0.388739%, degrees of freedom 6.42802.
  The degrees of freedom are fitted on training data within the prespecified
  range [2.05,200]. Four fixed optimization starts converge to effectively
  identical likelihoods. The Student scale is a distribution parameter,
  rather than its standard deviation.

Both models and all three reporting levels remain in the results. Evaluation
scores do not select a winning family, change the degrees of freedom, alter
portfolio weights or tune thresholds. Frozen unconditional forecasts are an
intentionally narrow model assessment. They do not model volatility dynamics,
conditional dependence or a full multivariate asset distribution.

## Analytic forecasts and observed exceedances

VaR is the stated upper loss quantile. ES is the average loss beyond that
quantile for these continuous models. Values below are percentages of basket
value; the expected count is `2,748 * (1-confidence)`.

| Confidence | Family | VaR | ES | Observed exceedances | Expected count |
|---|---|---:|---:|---:|---:|
| 95% | Gaussian | 0.7690% | 0.9647% | 95 | 137.40 |
| 95% | Student t | 0.7464% | 1.0323% | 105 | 137.40 |
| 97.5% | Gaussian | 0.9166% | 1.0936% | 63 | 68.70 |
| 97.5% | Student t | 0.9360% | 1.2343% | 57 | 68.70 |
| 99% | Gaussian | 1.0883% | 1.2470% | 32 | 27.48 |
| 99% | Student t | 1.1949% | 1.5186% | 18 | 27.48 |

At 95%, both fixed models have fewer exceedances than their nominal 5% tail
probability. At 99%, Gaussian exceedances are 1.164% of observations and
Student exceedances are 0.655%. A larger ES forecast and fewer exceedances do
not automatically establish a better model: excessive conservatism also
miscalibrates a quantile forecast. The model sensitivity at 99% is +9.8% for
VaR and +21.8% for ES when moving from Gaussian to Student assumptions.

## Calibration and dependence-aware uncertainty

For loss `L`, frozen VaR `q`, frozen ES `e` and tail probability `beta`, the
saved diagnostic series are:

```
VaR residual = 1{L > q} - beta
ES residual  = e - q - max(L - q, 0) / beta
```

Both means are zero at the correct continuous-distribution VaR/ES pair.
A negative ES residual mean means the model's predicted excess above its
own VaR is smaller than the observed excess scaled by `beta`. It is a joint
calibration diagnostic, not an isolated test of ES when VaR is incorrect.

The procedure uses 1,000 circular moving-block bootstrap replications with
20 consecutive observation intervals per block, seed 613000. Blocks preserve
short-run dependence; indices are shared across all families and levels for
paired comparisons. The 95% intervals are percentile intervals for historical
mean diagnostics. They condition on the frozen fitted parameters and assume
enough stationarity and weak dependence for this approximation. They exclude
training estimation uncertainty, have no finite-sample coverage guarantee
under regime change and are pointwise, without multiplicity correction.
The block length was fixed, rather than selected for favorable results.

One basis point (bp) below is 0.01% of basket value.

| Confidence | Family | Mean ES residual (bp) | Approximate 95% block interval (bp) |
|---|---|---:|---:|
| 95% | Gaussian | −1.10 | [−9.23, 6.30] |
| 95% | Student t | 6.25 | [−2.06, 13.93] |
| 97.5% | Gaussian | −6.97 | [−19.53, 3.33] |
| 97.5% | Student t | 6.85 | [−5.10, 16.82] |
| 99% | Gaussian | −18.04 | [−39.17, −0.024] |
| 99% | Student t | 8.11 | [−8.73, 22.84] |

The Gaussian 99% ES residual interval ends just below zero. This is weak,
borderline evidence of too little predicted tail excess under the stated
bootstrap assumptions, not a robust universal rejection. At 95%, the VaR
residual intervals exclude zero for both models: Gaussian [−0.02416,−0.00597]
and Student [−0.02053,−0.00233]. These indicate too few exceedances for these
frozen quantiles in this evaluation period. The corresponding 97.5% and 99%
VaR residual intervals include zero. Inclusion of zero does not establish
correct conditional calibration or validate the distribution's far tail.

## Joint VaR/ES score

The upper-loss version of the FZ0 score is evaluated when `e > 0`:

```
score(L,q,e;beta) = max(L-q,0)/(beta*e) + q/e + log(e) - 1
```

This follows the return-tail convention in
[Patton, Ziegel and Chen (2019)](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)
after replacing return with minus loss and negative return VaR/ES with
`-q,-e`. Lower expected score is better; negative score values are valid and
depend on the loss units. The same units and observations are used for both
families. At the correct risk pair its expected value equals `log(e)`.

| Confidence | Mean Student-minus-Gaussian score | Paired approximate 95% block interval |
|---|---:|---:|
| 95% | −0.00433 | [−0.00729, −0.00128] |
| 97.5% | 0.00180 | [−0.01521, 0.01649] |
| 99% | −0.00101 | [−0.06026, 0.05114] |

Student t has the lower observed joint score at 95%, conditional on this
fixed basket and period. At 97.5% and 99% the paired intervals do not resolve
a score difference. This does not support a blanket statement that Student
is more accurate, or justify selecting it as the true market model.

## Reproduction and audit trail

From the repository root, with NumPy and SciPy installed:

```bash
python scripts/prepare_fx.py --offline
python scripts/validate_fx.py --output results/historical-reproduction
python -m pytest tests/test_historical.py -q
```

The validation script verifies the saved returns and raw-source hashes and
refuses to overwrite a nonempty result directory. The saved result is:

- `results/historical/summary.json`: exact parameters, forecast values,
  counts, diagnostics, bootstrap configuration, package versions and hashes
  of source inputs, the producing script and output artifacts.
- `results/historical/forecasts.csv`: 16,488 per-date/model/level forecasts,
  losses, exceedance indicators, residuals and scores, sorted by date.
- `results/historical/bootstrap_means.csv`: all 1,000 replicated means for
  every diagnostic, with shared replication numbers for paired comparisons.

Eight tests independently compare Gaussian/Student ES formulas with numerical
tail quadrature, check score/calibration identities, verify paired bootstrap
behavior and check that the source-backed chronological partitions do not
overlap. The interval tables are derived from the saved JSON, not manual
Monte Carlo draws. The archived `historical-initial-order` run differs in
forecast-row ordering and producing-script hash; it is retained only as an
intermediate implementation artifact and is not the reported result.

## Scope of the evidence

These 2,748 evaluation observations cannot verify a daily probability of
10^-5. This exercise does not use the rare-event algorithm to manufacture
historical truth. Its role is to show the practical scale of model-assumption
sensitivity and observable-level calibration failures. Analytic model risks
contain no Monte Carlo error; uncertainty about fitted parameters, future
regimes and model family remains. The H.10 source is a revised 2026 vintage,
not a reconstruction of information available on every historical date.
Observation intervals include holidays and weekends, and exclude carry,
financing, transaction costs and executable-price effects. No trading,
regulatory or future-loss guarantee follows from this result.

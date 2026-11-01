# Established comparison algorithms

All models use a standard-normal latent vector, and every vectorized loss call
is charged by its row count. The event is strictly `loss > threshold`. The public
entry point is `run_baseline(model, method, budget, seed, threshold, **options)`.
The model supplies `dimension`, `loss(z)`, and an `evaluations` counter.

* `mc`: independent target observations; exact fixed-sample Clopper-Pearson
  binomial intervals.
* `antithetic`: independent pairs `(Z,-Z)`; standard errors use pair averages,
  not individual dependent observations. An odd last budget unit is unused.
* `sobol`: at least eight independently scrambled Sobol nets, each with a power
  of two points generated with `random_base2`. The standard error and approximate
  Student interval use independent scramble estimates. A common power-of-two
  block size is rounded down; unused evaluations are reported. The pointwise
  effective sample size is descriptive and does not imply IID QMC samples.
* `fixed_is`: a fixed mixture of unit-covariance Gaussian shifts and 10% target
  defense. Shifts can be passed as `means`; default shifts are `+/-4` along each
  coordinate. The likelihood correction uses the full mixture density, never
  the density of the randomly selected generating component.
* `ce_single`, `ce_mixture`: weighted-elite cross-entropy adaptation with
  diagonal covariance. One third of all evaluations trains the proposal; the
  other two thirds are independent confirmation samples from the frozen
  proposal. Defaults are five pilot rounds, a 10% elite fraction, 10% target
  defense, and a variance floor of .05. The mixture variant uses four components
  fitted by weighted EM with weighted k-means++ initialization. These are
  explicitly specified practical CE variants, not a faithful SAIS reproduction.
* `subset`: adaptive empirical-quantile levels and pCN proposals
  `Z' = sqrt(1-beta^2) Z + beta E`, with `beta=.7`. pCN is reversible for the
  standard-normal target; accepting only proposals inside the intermediate
  event preserves its conditional target. All chain states and reused seeds are
  retained, while only new proposals cost evaluations. Tied boundaries use the
  actual surviving fraction. Failure to progress or reach the threshold is
  exposed. The adaptive product estimator can have finite-sample bias, and its
  observations are correlated. No within-run confidence interval is asserted:
  assess uncertainty using independent complete runs. Unused budget is reported.

For ordinary importance sampling, the estimator is

`mean(1{loss(Z)>threshold} * p(Z)/q(Z))`.

Conditional on the pilot, confirmation observations are IID from fixed `q`.
Because `q >= defense*p`, support is preserved and weights are at most
`1/defense`. The reported Student intervals remain asymptotic; the defense
does not certify that all modes have been observed. Fewer than 30 nonzero tail
contributions triggers a warning. No interval here converts Monte Carlo
uncertainty into a claim about financial-model correctness. The returned
loss/weight pairs exclude adaptation data. Finite-sample VaR and Expected
Shortfall estimators derived from them are not claimed unbiased.

Subset simulation cannot return IID target-weighted loss observations; its
`losses` and `weights` arrays are empty by design.

## References

* Rubinstein (1999), The Cross-Entropy Method for Combinatorial and Continuous
  Optimization, https://doi.org/10.1023/A:1010091220143
* Au and Beck (2001), Estimation of small failure probabilities in high
  dimensions by subset simulation, https://doi.org/10.1016/S0266-8920(01)00019-4
* Cotter et al. (2013), MCMC Methods for Functions: Modifying Old Algorithms to
  Make Them Faster, https://doi.org/10.1214/13-STS421
* SciPy QMC documentation,
  https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
* Helal and Elvira (2025), Subset Adaptive Importance Sampling,
  https://arxiv.org/abs/2508.00210. This is relevant prior art, not the label of
  the CE implementation provided here.

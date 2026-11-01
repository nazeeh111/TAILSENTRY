# Novelty assessment and mathematical boundaries

Assessment date: 1 November 2026. This is an independent specialist's
literature and estimator assessment, not external peer review.

## Current verdict

**A new general rare-event estimator has not been established.** Defensive
mixtures, adaptive mixture allocation, pilot/confirmation splitting,
conditional Monte Carlo, line sampling and randomized quasi-Monte Carlo
are established ideas. TAILSENTRY's candidate is a specialized combination:
learn a multimodal Gaussian proposal, select a projection using pilot
conditional second moments, then integrate *every* failure interval along
that projection for an explicitly supported maximum of quadratic losses.
Its value must be established by equal-cost comparisons and ablation.
Even a large measured improvement would not by itself establish originality.

The most defensible potential contribution is an inspectable implementation
and controlled evidence about when projection selection and exact interval
integration improve or degrade an adaptive proposal. A claim of algorithmic
novelty requires a closer comparison with the work below and a precise new
mathematical or algorithmic result that those works do not already imply.
A failed search for the exact implementation phrase does not prove novelty.

`run_projected_aloe` is an additional demanding comparator on eligible
Gaussian half-space problems. If `r_j(v) = P(H_j | V=v)` and
`S = sum_j P(H_j)`, its marginal proposal is
`q_V(v) = p_V(v) * sum_j r_j(v) / S`. Its observation is
`S * P(union H_j | V=v) / sum_j r_j(v)`, which lies in `[0,S]` and is the
conditional expectation of the ordinary ALOE weight. The variance reduction
is the standard conditional-variance identity. This composition is not
claimed as new. It uses complete known half-space geometry, so neither this
baseline nor an exact ALOE result establishes discovery of unknown modes.

## Close precedents that constrain the claim

1. Owen, Maximov and Chertkov (2019), **Importance sampling the union of rare
   events with an application to power systems analysis**, Electronic Journal
   of Statistics 13(1), 231–254. [Author-hosted institution record](https://statistics.stanford.edu/technical-reports/importance-sampling-union-rare-events-application-power-systems-analysis),
   [preprint](https://arxiv.org/abs/1710.06965). ALOE samples from the target
   conditional on one union event and corrects for overlaps. Its guarantees
   make it a strong comparator for known half-space events. An experiment
   on max-affine Gaussian losses that omits ALOE can exaggerate a new method's
   value. `src/tailsentry/aloe.py` implements this established baseline.

2. Elvira and Santamaría (2021), **Multiple Importance Sampling for Symbol
   Error Rate Estimation of Maximum-Likelihood Detectors in MIMO Channels**,
   IEEE Transactions on Signal Processing 69, 1200–1212,
   [DOI:10.1109/TSP.2021.3055961](https://doi.org/10.1109/TSP.2021.3055961),
   [author preprint](https://victorelvira.github.io/assets/papers/elvira2021multiple_pre.pdf).
   Defensive adaptive ALOE combines selected half-space components with a
   Gaussian target truncated outside a hypersphere. It addresses omitted
   half-spaces and jointly adjusts component count and defensive mass.
   Thus discovering modes, adding a defensive component and relating its
   mass to a residual variance bound are already closely precedented.
   Its geometric residual envelope can outperform a nominal-prior audit.

3. Ahn and Kim (2018), **Efficient Simulation for Expectations over the Union
   of Half-Spaces**, ACM Transactions on Modeling and Computer Simulation
   28(3), article 23,
   [DOI:10.1145/3167969](https://doi.org/10.1145/3167969),
   [primary institutional abstract](https://pure.kaist.ac.kr/en/publications/efficient-simulation-for-expectations-over-the-union-of-half-spac/).
   Radial conditional sampling for elliptical targets, efficiency analysis
   and financial-network applications predate this project. Integrating a
   radial variable for a union of affine stress losses is not a new method.
   The full 2018 algorithm has not been locally replicated here.

4. Ahn and Zheng (2023), **Conditional Importance Sampling for Convex
   Rare-Event Sets**, Winter Simulation Conference, 363–374,
   [primary proceedings](https://www.informs-sim.org/wsc23papers/030.pdf).
   Geometry-informed conditional IS extends to nonlinear convex failure
   sets and proves bounded relative error under stated assumptions.
   Conditional estimation of nonlinear events is therefore not itself new.

5. Uribe, Papaioannou, Marzouk and Straub (2021), **Cross-entropy-based
   importance sampling with failure-informed dimension reduction for rare
   event simulation**, SIAM/ASA Journal on Uncertainty Quantification 9,
   818–847, [DOI:10.1137/20M1344585](https://doi.org/10.1137/20M1344585),
   [preprint](https://arxiv.org/abs/2006.05496). Failure-informed subspaces
   make cross-entropy adaptation tractable in high dimensions. TAILSENTRY's
   one-dimensional preintegration is a different operation, but a broad
   claim to discover low-dimensional rare-event structure would overlap.

6. Tong and Stadler (2023), **Large Deviation Theory-based Adaptive Importance
   Sampling for Rare Events in High Dimensions**, SIAM/ASA Journal on
   Uncertainty Quantification 11, 788–813,
   [DOI:10.1137/22M1524758](https://doi.org/10.1137/22M1524758),
   [preprint](https://arxiv.org/abs/2209.06278). Large-deviation optimization
   initializes Gaussian proposals and identifies an informative adaptation
   subspace. Its sample reuse requires its own justification; independently
   held-out confirmation is a simpler correctness route, not a novel one.

7. Cheng, Papaioannou, Lu, Zhang and Wang (2023), **Rare event estimation with
   sequential directional importance sampling**, Structural Safety 100,
   102291, [DOI:10.1016/j.strusafe.2022.102291](https://doi.org/10.1016/j.strusafe.2022.102291).
   Kanjilal and Bect (2026), **Directional subset simulation method for
   reliability analysis**, [preprint](https://arxiv.org/abs/2605.23631).
   Adaptive directional sampling and preserving multiple modes through
   subset levels are active established research directions. A directional
   interpretation alone does not distinguish TAILSENTRY.

8. Sak and Başoğlu (2017), **Efficient randomized quasi-Monte Carlo methods
   for portfolio market risk**, Insurance: Mathematics and Economics 76,
   87–94, [DOI:10.1016/j.insmatheco.2017.07.001](https://doi.org/10.1016/j.insmatheco.2017.07.001),
   [preprint](https://arxiv.org/abs/1510.01593). The paper combines RQMC,
   importance sampling and stratification for t-copula portfolio loss
   probabilities and conditional excesses. RQMC plus financial IS is not a
   new contribution.

9. Gilbert, Kuo and Sloan, **Preintegration is not smoothing when monotonicity
   fails**, [preprint](https://arxiv.org/abs/2112.11621),
   [published chapter DOI](https://doi.org/10.1007/978-3-031-10193-9_9).
   Nonmonotone preintegration can retain square-root singularities. Exact
   quadratic root integration may reduce variance while failing to produce
   the smoothness needed for standard strong QMC convergence claims.

10. Li, Reimann, Schorlepp and Stadler (submitted 20 October 2026), **Beyond
    exponential tilting: sharp asymptotics and improved importance sampling
    for rare events**, [preprint](https://arxiv.org/abs/2610.27434).
    Geometry-based control variates and anisotropic proposals address the
    residual coefficient-of-variation growth of classical exponential
    tilting. The reported bounded or vanishing relative error includes
    truncation qualifications. This is recent adjacent work, not a local
    replication or proof of TAILSENTRY's performance.

## Estimator assessment

Let an independent pilot determine an orthonormal coordinate system
`Z = e T + B V` and a fixed proposal `q_Z`. The target is standard Gaussian.
Let an auxiliary scale `S` have density `h` and be sampled from that same
nominal density independently of the proposal. The supported Student model
is obtained by evaluating its loss at `Z / sqrt(S)` with
`S = chi_squared(df) / df`. Define

```
U(v,s; tau) = integral 1{L(e*t + B*v, s) > tau} phi(t) dt
q_V(v) = integral q_Z(e*t + B*v) dt
Y(v,s; tau) = U(v,s; tau) phi_(d-1)(v) / q_V(v).
```

Sampling `(V,S)` from `q_V(v) h(s)` gives `E[Y] = P(L > tau)` by integration
and nonnegative Fubini. Moreover this is the conditional expectation, under
`q_Z h`, of the ordinary importance-weighted event indicator given `(V,S)`.
The conditional-variance identity therefore gives variance no larger than
that ordinary estimator *at the same fixed proposal and sample count*,
provided the second moment exists. It says nothing about equal wall time,
training cost, a different proposal, or missed-mode discovery.

For a defensive proposal `q_Z >= epsilon * phi_d`, marginalization gives
`q_V >= epsilon * phi_(d-1)` and `0 <= Y <= 1/epsilon`. This supports bounded
observation confidence procedures. It also explains why an honest bound can
remain too wide at small budgets: the global range protects unobserved
regions. Large ESS or a narrow normal interval cannot replace this protection.

The Gaussian-mixture *marginal* density must be used in the denominator.
Evaluating the full density at `T=0`, reusing a sampled discarded `T`, or
weighting by a selected component alone changes the estimator. If proposals
are adapted during confirmation, iid standard-error formulas no longer
follow from the pilot argument. If the scale law is biased too, its likelihood
ratio must also appear; the current argument assumes nominal scale sampling.

## Exact interval and severity requirements

For maximum-of-quadratic losses each branch is a quadratic polynomial in
`T`. All real roots and intervals must be retained, including compact failure
islands, two-sided failures and a zero quadratic coefficient. A single
positive root or monotonicity assumption silently loses valid events.
Stable normal survival differences are required when an interval lies deep
in a tail. Nearly tangent or ill-conditioned polynomials require explicit
numerical handling; floating-point exactness is not an analytic guarantee.

Probability integration needs the union of branch exceedance intervals.
Severity integration is harder: integrate the *maximum* branch, not the sum
of all exceeding branches. For exact quadratic severity this requires branch
intersection points and choosing the dominant polynomial on each interval.
Truncated Gaussian polynomial moments then provide an analytic integral.
A clipped loss changes the target and must be named as clipped severity.
Quadratic loss under a Student scale mixture needs `df > 2` for its first
moment and stronger conditions for finite second-moment standard errors.

An unbiased tail-probability estimate does not imply unbiased finite-sample
VaR or Expected Shortfall. Quantile inversion and ratios are nonlinear.
Their uncertainty needs separate validation. Common confirmation draws and
a fixed proposal can produce a monotone estimated tail curve; independently
refitting at every threshold does not automatically preserve monotonicity.

## Missing-mode audit: useful distinction, limited novelty

A frozen discovered region `H` permits a separate residual event
`R = {L > tau} \\ H`. For `m` independent nominal audit samples, residual hit
counts are binomial. With zero hits the exact one-sided upper bound is
`P(R) <= 1 - delta**(1/m)` at confidence `1-delta`. This is the established
binomial zero-event bound, not a new guarantee. Certifying a small residual
at tolerance `r` requires roughly `log(1/delta)/r` nominal draws even when
none is observed. An arbitrary black-box missed mode cannot be ruled out
cheaply merely by improving the discovered-mode estimator.

If a *proved* envelope `R subset B` has known probability `P(B)` and exact
conditional sampling, then labelled audit draws from `p(.|B)` instead give
`P(R) <= P(B) * [1-delta**(1/m)]` after zero residual hits. A radial envelope
can be useful, as DA2LOE demonstrates; without a valid global enclosure it is
only a heuristic. Choosing a mixture fraction from a pilot upper bound is
reasonable engineering, but the general allocation principle is closely
precedented and does not establish novelty.

The truthful fallback is to report implementation correctness, replicated
efficiency gains where observed, the audit's abstention rate and its cost,
while leaving the original-breakthrough goal explicitly unachieved unless
the final theoretical and prior-art review supports it.

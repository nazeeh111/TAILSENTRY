# Independent adversarial quantitative review

Review date: 1 November 2026. This is an AI reviewer working separately from the implementation author in the same session, not external peer review. The reviewer did not edit implementation files. Findings were sent to the author with reproducible counterexamples. This report distinguishes the original frozen implementation from repairs made during review.

## Scope and evidence

Inspected raw `envelope.py`, `conditional.py`, `models.py`, `uncertainty.py`, `risk.py`, benchmark adapters/runner, `validate_fx.py`, protocol and novelty/baseline documentation, raw development runs, and incomplete main holdout runs. The initial main holdout summary did not exist. The reproducible reviewer script and numerical evidence are `reviews/adversarial-checks.py`, `reviews/adversarial-checks.json` and `reviews/adversarial-checks.log`; the JSON pins the source hashes actually imported for those checks. The holdout snapshot contains 12,807 raw records, not a final selected table.

The core union and projection weighting is coherent. If the event is contained in the union of halfspaces, B is their summed probability, and m(x) counts covered halfspaces, the proposal density is f(x)m(x)/B. The raw probability observation is B 1(event)/m. After integrating a factor coordinate, its correct observation is B P(event|v)/sum_j P(H_j|v). These are the implemented denominators, rather than a selected component's density. The bound by B follows from union containment, not observed ESS. Dropping negative spectral terms and allocating positive-term caps gives a valid enclosing union in exact arithmetic. The Student conditional auxiliary law, chi-square(nu+1)/(1+t²/nu), is also correct. No assertion here certifies float64 arithmetic for arbitrary coefficients.

Independent checks found:

- 50 random three-branch quadratic upper envelopes matched companion-matrix-root partitioning and independent adaptive quadrature, with maximum absolute errors 3.33e-16 for probability and 1.78e-15 for first moments in ordinary-scale examples.
- 32 random three-dimensional, three-branch models produced 4,308 event points among 8,192 test points, with no observed uncovered event. This is a falsification attempt, not a proof replacing the containment derivation.
- Saved historical risks passed independent tail integrals at all six family/confidence combinations, with maximum absolute ES discrepancy 7.64e-17. All six exceedance counts matched the 2,748 saved evaluation losses. Raw-source/prepared-return provenance hash validation passed.

## Findings requiring disposition

### P1: compact-island severity cancellation in frozen main-v1 source

For L(T)=1e-12-T², T standard normal, threshold zero, probability is approximately 7.9788456081e-7. The original `tail_moment` returns **-7.04924042079e-18**, despite the loss being strictly positive on the event. Independent integration in local coordinates gives **+5.31923040535e-19**. The original truncated-moment expression subtracts nearly equal quantities. At L(T)=1e-10-T² the probability is approximately 7.9788456079e-6, within the target rare-probability scale, and the first moment is underestimated by 7.69%.

This invalidates unrestricted numerical severity accuracy for the original source, while not disproving its probability estimator. Both tail-mean ratios and plug-in ES depend on this routine. This is more than an abstract absence of interval-certified arithmetic: it produces a wrong-sign observable with simple finite coefficients.

Disposition observed during review: the author added local Gauss-Legendre integration on short finite envelope segments. The unshifted adversaries then agreed with independent local quadrature to approximately 4.5e-16 relative error, as recorded in `adversarial-checks.json`. A wider shifted-island check still found polynomial evaluation cancellation: a=-1, b=16, c=-64+1e-12 gives about 0.2655% relative moment error against vertex-coordinate integration, although its probability is far below the main target scale. Recommended final disposition: use compensated/extended-precision local polynomial evaluation, retain shifted-island tests, and describe the fallback as numerical integration rather than exact algebraic moment evaluation. Recheck the final repaired bytes. Historical analytic ES does not use this routine and is unaffected.

### P2: the precision badge can certify a different center from the displayed estimate

`precision_status` explicitly certifies the interval midpoint, returned as `certified_center`. The UI displays the raw sample mean as its main probability and labels the requested relative error supported. Those values coincide for an unclipped symmetric interval, but can differ after upper truncation at one. For example, interval [0.8,1.0] supports midpoint relative error 0.1/0.8=12.5%, so a 13% request passes. A displayed raw mean of 0.99 can nevertheless differ by 23.75% from p=0.8, which is inside that interval.

Action: display the certified midpoint as the estimate to which the badge applies, or compute the worst relative deviation from the actual displayed mean over the interval. This mainly affects common events reached through editable thresholds, rather than the default rare events. The finite interval itself is not invalidated.

### P2: main wall-time efficiency needs isolated confirmation

Methods are timed in grouped sequential blocks, while independent review and other workers ran concurrently. These are honest recorded elapsed costs, but they cannot isolate an algorithmic speedup. A concrete symptom is `compact2_1e4`: fixed and pilot-selected envelope variants produce identical per-seed estimates because there is one direction and no selection pilot, yet at budget16,384 their mean times are about 0.1125s and 0.1704s. The difference is about 52% despite the same computational path apart from an inactive selection flag. At budget4,096 the direction of the timing difference reverses.

Action: preserve original measurements and the predeclared target, then report an explicitly separate isolated randomized-order timing sensitivity with pinned source, environment and workloads. Do not quietly replace unfavorable original timing evidence. A bootstrap over runs does not remove systematic workload/order effects.

### Source provenance discrepancy: addressed, retain the amendment

Current source initially differed from main-v1's envelope hash during its run. The author subsequently supplied `results/holdout/main-v1/source/src/tailsentry`, whose hashes can be checked against configuration, and `docs/protocol-amendment.md`. The change relaxes cutoffs above 12 to 12, which enlarges covering halfspaces and preserves containment rather than dropping tiny spectral modes. A fresh primary-v2 evaluation is separately identified. The old loaded process and new source must not be described as one unchanged experiment; retain executed source and failed/unfinished evidence.

## Results that constrain any efficiency or reliability claim

Complete cells in the raw main snapshot already contradict blanket superiority:

- Compact two-dimensional balls, budget16,384: conditional Sobol relative RMSE approximately 1.2746e-5; fixed projected envelope approximately 0.008147. The conditional Sobol error is about 639 times smaller at the same oracle-call budget. Both are allowed analytic line integration.
- Correlated affine four-mode case, budget16,384: raw union relative RMSE approximately 0.001978 versus fixed projection 0.001483; their elapsed-time means were approximately 0.0404s and 0.1056s. Lower error alone does not establish better MSE-times-time efficiency.
- Paired affine four-mode case, budget4,096: raw union normal/asymptotic intervals cover only 20/64 runs and fixed projected intervals only 12/64. Their finite-bound intervals cover 64/64. The near-zero variance and high ESS do not validate the ordinary normal intervals. Finite-bound widths must remain next to coverage: a very wide interval can cover without supplying useful precision.

The benchmark includes the established ALOE mechanism through raw union sampling on affine unions. Conditional projection is ordinary Rao-Blackwellization of that proposal. Coefficient access, eigendecompositions and exact event-line integration are materially richer information than black-box point evaluation, and the protocol says so. Nothing reviewed establishes superiority over unimplemented SAIS/FAMIS/learned proposals, mathematical priority, or a new general rare-event method. The package's restricted novelty language is appropriate; literature completeness was not independently re-searched by this reviewer.

The main holdout was incomplete at this report's first version. No predeclared primary efficiency target is endorsed before the complete fresh primary results, coverage intervals, bootstrap calculation and timing sensitivity are checked.

## Historical financial interpretation

The frozen 2000–2014 fit and 2015–2025 evaluation arrays are separated in source; fitting receives only training losses. No evaluation-period fit or portfolio selection was found. The current revised H.10 vintage is disclosed, so this is not a point-in-time trading backtest. The yen inversion/equal-value basket and non-daily holiday intervals are explicitly documented.

The Gaussian and Student analytic VaR/ES formulas and the upper-loss FZ0 transformation are correct. The ES residual is a joint VaR/ES calibration diagnostic, not isolated proof of ES validity. The historical output itself contains failures: both 95% VaR residual intervals exclude zero, and the Gaussian 99% ES residual interval barely excludes zero. The paired Student-minus-Gaussian FZ0 difference is resolved only at 95%, not at 97.5% or 99%. There is no basis to call Student uniformly better or validate 1e-5 financial probabilities.

The circular-block bootstrap keeps shared comparison indices and dependence within 20-observation blocks. Its intervals are approximate, conditional on training fits, pointwise and dependent on stationarity/weak dependence and block length. Those limitations are disclosed. A single block length and one historical basket cannot establish robust calibration or future-market validity. The narrowly written historical report respects that boundary.

## Review limits

No trading, external messaging, publication or paid service was used. I did not independently audit external literature metadata or authenticate the live Federal Reserve site beyond the frozen raw source/provenance. I did not establish certified floating-point enclosures, simultaneous historical multiplicity control, arbitrary black-box support, asymptotic rare-event efficiency, finite-sample unbiased VaR/ES, or future financial validity. Reproducible source, specific favorable synthetic cases and honest negative results are supportable engineering outcomes. An original-breakthrough or universal-efficiency goal remains unestablished.

## Final artifact follow-up

A separate final audit is retained in `adversarial-final-checks.py`, `.json` and `.log`. This follow-up independently checked all 17,280 aggregated records against exact original source rows, all 240 case/method/budget groups and seed sets, all final-receipt hashes, and both archived executed-source snapshots. The main-v1 seven secondary cases each contain 64 runs per budget/method. The amended primary-v2 contains 128 runs per budget/method. No selective secondary omission or duplicated seed was found. Every summary RMSE, bias, MSE-times-elapsed-time quantity and finite/asymptotic coverage interval was recomputed from raw rows and matched. All 4,000-resample primary paired, unpaired and separate-timing bootstrap intervals were independently reconstructed and matched.

**The declared primary efficiency target failed decisively.** At budget16,384, fixed projected union sampling versus conditional Sobol has MSE-times-elapsed-time ratio **38.789**, paired bootstrap interval **[26.475,57.322]**. The separate randomized-order timing sensitivity gives ratio **37.374**, interval **[21.671,64.537]**. Thus the selected method is substantially less efficient than this established conditional baseline on the declared primary workload. The failure persists despite reducing the original timing confound. Favorable comparison to raw union sampling does not rescue this target: its original ratio is 0.02518, interval [0.01734,0.03638], and separate-timing ratio is approximately0.04209. These support a narrower ablation result, not a breakthrough.

The selected primary finite-bound interval covered all128 runs, with exact binomial coverage interval [0.97159,1] and mean width4.86469e-6 for probability1e-4. These are useful fixed-model bounded-observation results, but not evidence that the efficiency target succeeded or that a future financial loss model is valid. Coverage rates and widths must remain attached to their estimator, budget and source version.

Final numerical disposition:

- The author added vertex-coordinate extended-precision polynomial evaluation inside the short-segment Gauss-Legendre fallback. All12 centered/shifted compact-island checks now returned positive moments and removed the original wrong-sign defect. The worst remaining relative discrepancy in these adversaries was about1.88e-5, on a remote center8 island with radius squared1e-12. Root conditioning and finite arithmetic still matter; the package correctly does not promise interval-certified numerical integration. The original P1 is addressed for the explicit counterexamples.
- `precision_status` now accepts the actual displayed point estimate. The [0.8,1.0],13% example rejects estimate0.99 and accepts midpoint0.9. The actual-mean calls in the estimators address the precision-badge defect.
- The source snapshots for main-v1 and primary-v2 match every recorded source-map hash. Their different versions and the amendment remain explicit. The separate timing study preserves rather than replaces original measurements.

Additional reporting limits identified in final validation:

- `scripts/validate_risk.py` chooses each finite threshold grid using the known reference VaR, q±0.35. This is an **oracle-centered accuracy study**, and its initial grid threshold also informs proposal adaptation. It does not validate autonomous bracketing/grid discovery without reference access. This must be stated next to VaR/ES results. At both normal-max confidence levels, ordinary VaR intervals cover26/32, approximately81.25%, so a blanket nominal95% VaR uncertainty claim is contradicted by the package's own replication results. The finite-grid brackets cover all runs, with their simultaneous-grid interpretation and widths retained.
- The missed-mode result is correctly adverse: ordinary asymptotic coverage3/512 while bounded intervals cover512/512 and remain broad. Its audit-cost entry uses alpha0.025, whereas the displayed65,536-draw audit upper bound uses alpha0.05 by default. The numbers are individually valid, but different confidence levels should be labeled explicitly rather than compared as if identical.
- `primary-analysis.json` calls the favorable raw-union and CE comparisons `original_target_met=true`; the actual original target was conditional Sobol only. This metadata label should be limited to the declared comparator or renamed for secondary comparisons. The declared target's own record correctly says false.
- A new separately identified SAIS-inspired reproduction and confirmation extension now exists, with source-specific exploratory examples and explicit fidelity gaps. This reviewer did not reproduce its published paper tables or independently review every update equation; its presence does not establish state-of-the-art superiority. Older documentation describing the entire implementation as unfinished should be reconciled with `docs/sais-fidelity.md`.

At the final numeric audit, no manuscript text existed under `paper/` beyond bibliography and figures. Therefore manuscript accuracy, completeness of generated figures and claims made in a later paper were not yet reviewed. The checked final numerical artifacts support leading with the failed primary target, repaired numerical defects, useful bounded inference under narrow assumptions, and explicit novelty/financial limits.

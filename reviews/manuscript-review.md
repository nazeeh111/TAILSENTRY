# Independent manuscript claims review

Reviewed `paper/tailsentry.tex`, `primary-table.tex`, `suite-table.tex`, `references.bib`, the figure-generation source, and `docs/historical-validation.md` against the previously independently checked raw final artifacts. This is an internal independent AI review, not external peer review. No implementation or manuscript source was edited by the reviewer. External bibliographic metadata and the live cited websites were not independently reauthenticated in this pass. This pass reviews claims and mathematical text, not final PDF pagination or visual rendering.

## Main conclusion

The manuscript correctly leads with the failed primary hypothesis and does not claim an original general estimator. Its primary ratios, bootstrap intervals, problem definition/threshold, run counts, omitted-mode results, historical VaR/ES values, and risk-validation error/coverage ranges agree with saved evidence. The stated conditional-union derivation is correct under its fixed-cover, independent-confirmation, continuous-target and exact-arithmetic assumptions. The exact Student auxiliary conditional law is correct. None of the text reviewed establishes state-of-the-art dominance, future financial calibration, regulatory validity, or unbiased finite-sample VaR/ES, and the manuscript expressly avoids these conclusions.

The following bounded corrections improve mathematical and evidential precision. They do not change the negative primary conclusion.

## P2: describe independence within conditions, not across all 17,280 records

The abstract calls these “17,280 independent algorithm runs”; the experimental-design section has similar language. There are 64 or128 independent complete repetitions **within each method/case/budget condition**. Shared seed labels create coupling across methods/conditions. Indeed fixed and inactive-pilot envelope variants produce identical estimates on some compact2 conditions. The paired bootstrap deliberately retains such coupling.

Requested wording: “17,280 complete runs, with64 or128 independent repetitions per method/case/budget condition; methods share seed labels.” This is compatible with the actual design, paired bootstrap and appendix caution. It avoids suggesting17,280 mutually independent outcomes.

## P2: show probability interval widths and coverage uncertainty

The main table displays finite coverage percentages, but not widths or binomial uncertainty. Coverage alone makes a vacuous interval and a useful narrow interval appear equally successful. In the primary budget16,384 results:

- Selected projected union:128/128 finite intervals covered, exact95% binomial interval [0.9715919466,1]; mean probability-interval width4.8646923e-6.
- Conditional Sobol:128/128 covered with the same binomial interval, but mean finite Bernstein interval width1.0, because it bounds only eight independent scramble means. Its useful small error bars are separately approximate.
- Crude MC:127/128 covered, exact95% binomial interval [0.9572409681,0.9998022242]; mean finite interval width0.0003910067.

Add at least the first two width/uncertainty comparisons in the main text or a compact table note. This makes the bounded-inference contribution interpretable and honors the protocol's requirement to retain interval width alongside coverage. The raw JSON already preserves the correct quantities.

## P2: qualify cover containment up to target-null sets

The mathematical text states literal A⊆C, while the implemented whole-space fallback is `{x1>0} union {-x1>0}`. This excludes the hyperplane x1=0. Gaussian and Student targets assign that hyperplane zero probability, so the estimator and conditional-variance proof remain valid. Literal set containment can fail there for, for example, a constant positive branch.

State that containment is up to target-null sets, or use a literal whole-space event in the theoretical exposition. Likewise define the projected contribution to be zero wherever D=0; those points have zero proposal marginal density and U=0 almost everywhere under valid containment. The proof then has a fully specified domain without an implicit0/0 convention. No implementation correction is needed for this null-set issue.

## P2: retain the observable historical calibration failure in the financial summary

The detailed historical report correctly says both95% VaR residual intervals exclude zero and show too few exceedances under the frozen quantiles. The manuscript highlights the improved Student95% FZ0 score and borderline Gaussian99% ES residual, but does not explicitly retain that95% quantile-calibration failure.

Add one sentence that both models'95% VaR diagnostics indicate too few exceedances despite the Student's better joint score. A lower comparative score does not establish calibration. The surrounding manuscript already disclaims future validity, and the detailed financial report is appropriately cautious; this request preserves a material negative result in the short summary.

## Claims checked without requested correction

- The primary efficiency target fails:38.789 versus conditional Sobol, paired interval [26.475,57.322]; isolated timing37.374, interval [21.671,64.537]. Favorable within-proposal comparison against raw union is correctly separated.
- The cover proof drops a nonnegative quadratic form and uses a valid cap/pigeonhole argument. Relaxing a cutoff enlarges a halfspace and cannot delete event support. No independence between halfspaces is assumed.
- The projected denominator is the marginal sum of conditional halfspace probabilities. The proof is Rao–Blackwellization of the same raw proposal, and explicitly does not imply equal-time dominance over a different proposal.
- The empirical-Bernstein expression, fixed-budget scope, separate ordinary intervals, and displayed-point precision criterion agree with the final implementation. Numerical Gauss–Legendre severity stabilization and remote/tangent limitations are disclosed.
- The ordinary interval undercoverage in the malicious omitted-mode experiment is correctly adverse. The95% zero-hit upper bound and97.5% audit-cost example now name their distinct confidence levels.
- The oracle-centered VaR grid, finite-sample biased plug-ins, risk-interval undercoverage, and conservative Student moment gates are expressly stated. The experiment does not establish automated grid discovery.
- The eight-case,15-method suite is shown without omitting negative cases. The selected small-error comparator is not hidden. The most favorable one-dimensional results are explicitly clipped only in figure display, with raw values preserved.
- Point and conditional evaluations, richer analytic information, full adaptation/preparation time and cumulative RSS limitations are described. Independent timing sensitivity is kept separate from initial measurements.
- The SAIS reproduction/confirmation extension is separate exploratory evidence, not an equal-budget primary comparison or published-table replication. FAMIS remains unbenchmarked and novelty remains unestablished.
- The chronological FX fit uses only training losses; saved current-vintage data, equal exposures, holiday intervals, model sensitivity, paired block assumptions and absence of1e-5 historical validation are correctly distinguished. The reported21.8% ES sensitivity is analytic model-family sensitivity, not rare-event Monte Carlo error.

## Scope of acceptance

Subject to the corrections above, the paper's central scientific claims are supported by the checked artifacts: a specialized established-method composition with bounded fixed-model probability inference, repaired numerical examples, useful raw-proposal ablations, and a clearly failed primary computational-efficiency target. This review does not convert engineering validation into algorithmic novelty or external peer review. Final PDF readability, citations rendered by BibTeX, publication packaging, or later source changes require their own checks.

# Evaluation protocol frozen before holdout

Version 1, 1 November 2026. Development seeds 1000–1011 and specialist smoke seeds 7000–7039 were used; final seeds start at 210000. Changes to algorithms after holdout begins require a new, explicitly labelled evaluation, not replacement of inconvenient results.

## Question and estimand

Does exact one-axis conditionalization of a structure-certified union proposal improve rare-loss probability estimation at fixed computational cost, with trustworthy fixed-budget probability bounds? The algorithm uses analytic maximum-of-quadratics coefficients. It is not a general black-box method. Novel algorithmic originality is unestablished.

The primary financial-like controlled workload is an eight-factor delta-gamma loss with Gaussian factors, one quadratic factor and a correlated linear factor, at exceedance probability 1e-4. An independent one-dimensional Gaussian quadrature determines truth. Secondary cases: exact normal single tail 1e-3; four two-sided normal factors 1e-4; imbalanced four-mode max 1e-5; correlated four-mode max 1e-5 (independent conditional-factor quadrature); disconnected two-factor balls 1e-4; three disconnected four-factor balls 1e-5 (noncentral chi-square); Student-t5 three-factor balls 1e-4 (independent scale quadrature). These prescribed definitions are in benchmark.make_case. No historical event can validate these extreme probabilities.

## Methods and information

Crude MC, antithetic pairs, 8 independent scrambled Sobol replicates, fixed defensive Gaussian mixture, weighted cross-entropy single Gaussian and diagonal four-component mixture, pCN subset simulation, conditional MC, conditional scrambled Sobol, raw/adaptively projected Gaussian mixture, certified union proposal raw, fixed-axis projected, pilot-selected projected, and unoptimized-cap projected.

On max-affine Gaussian problems the raw union proposal is precisely ALOE and projected union sampling is its Rao–Blackwellization. No intentionally weak omission of ALOE is permitted. The CE variants are not SAIS reproductions. No superiority over unimplemented SAIS, Deep-PrAE, neural flow methods or all state of the art may be claimed.

Point methods get loss evaluations. Conditional/union methods additionally see analytic coefficients, eigenvectors and exact conditional event integration. Report this information advantage prominently. Count point and conditional oracle calls separately; one conditional call reveals a whole line, so equal call counts are not equal information. Include full preparation, spectral decomposition, cap optimization, pilot sampling and projection selection in elapsed and process CPU time. Direct proposal constraint operations and root algebra cost time but do not count as black-box loss calls. Subset may finish with unused allowance; report actual costs. RQMC uses 8 independent scrambles, powers of two, no skipping or thinning. Training is charged, and confirmation samples are independent and frozen. RNG seeds are explicit and method streams deterministic.

## Fixed resources and replications

Main suite: 64 independent full runs for every method/case at budgets 4096 and 16384. Seeds 210000–210063. This permits approximate descriptive coverage resolution; report exact binomial uncertainty on coverage. Primary-case confirmation: an additional 64 runs seeds210064–210127 at both budgets, producing 128 primary replications when pooled. Secondary 1e-5 claims retain 64 runs and their wider statistical uncertainty. Optional scaling budget65536, 16 fresh runs seeds310000–310015, is descriptive only and must remain separate.

Run on the observed ARM64 Mac CPU using NumPy/SciPy float64; no CUDA, GPU, energy or host-total-memory claims. Sequential timing with BLAS thread limits avoids unrecorded parallel allocation. RSS is a cumulative process high-water mark, an upper bound rather than isolated per-method memory. Energy unavailable. Hardware memory telemetry was denied.

## Falsifiable target from development

The 12-run development slice at budget4096 showed that fixed-axis projected union sampling reduced relative RMSE from .07175 (raw cover) to .00807 on delta-gamma, while conditional Sobol reached .00288 at higher recorded runtime. Pilot projection selection (.01023) added work and did not improve on the fixed direction, so it is an ablation, not the selected production default. On compact2, conditional Sobol was vastly more accurate than either union sampler. On affine unions, established ALOE was already nearly exact.

The selected primary estimator is **fixed-axis projected certified union sampling**, with analytically optimized caps. The predeclared target is a 95% bootstrap upper confidence limit below 1 for its (MSE × mean elapsed time) ratio against conditional Sobol on the delta-gamma primary case at budget16384, alongside empirical finite-bound coverage compatible with at least90%. Also report equal-call RMSE and compare raw union, CE-single and every other baseline. The target is an improvement test, not an invented required multiplier. If it fails, say so. No blanket all-case speedup target is claimed.

Bootstrap resamples independent entire-run observations, 4000 resamples, seed901; because methods use separate algorithm-specific transformations of matching seed labels, pairing is a computational coupling, not a claim of identical scenarios. Report unpaired bootstrap as sensitivity if correlation affects the conclusion. Timing noise and pilot-sized development evidence limit extrapolation. Primary estimation accuracy, coverage, abstention, and fixed-time efficiency must all remain visible.

## Metrics and uncertainty

Raw per-run estimates, signed error, relative RMSE, bias, zero estimates, interval width, interval coverage with exact binomial interval, observed contribution ESS, degeneracy, point/conditional evaluations, elapsed/CPU seconds and process RSS. Fixed-budget empirical Bernstein intervals use the proven contribution range B=Σhalfspace probabilities. Generic defensive Gaussian samplers have range1/epsilon and may abstain. MC uses exact binomial intervals. RQMC and CE practical intervals are marked asymptotic; subset has no unsupported within-run confidence interval. No optional-stopping guarantee is asserted.

A separate malicious missed-mode experiment fixes probability1e-5, discovers only the positive of two opposite normal tails, uses 90% conditional-positive plus10% prior, N20000, and 512 independent replications. Compare ordinary normal intervals with bounded intervals and the analytic hidden-mode miss probability. High ESS cannot establish coverage. Also show the exact nominal-audit zero-hit upper bound and its sample cost.

VaR/ES: finite-sample biased plug-in estimators, tested separately on analytic normal and nonlinear cases; probability unbiasedness is not inherited. Heavy-tail moment existence is checked; severity SE omitted if finite second moment is unsupported. Historical FX: original public Fed sources, equal exposures, fixed 2000–2014 fit /2015–2025 evaluation, Gaussian and Student alternatives, observable95%,97.5%,99% levels; dependence-aware calibration residual uncertainty. Current source vintage, no look-ahead calibration on evaluation, no regulatory compliance claim.

## Reproducibility and failure handling

Each fresh run directory stores configuration, source-byte SHA256 map, protocol hash, raw JSONL and summary hashes. Never overwrite a prior run. Source fixes require retained earlier evidence and explicit invalidation reasons. No tuning on final seeds. All cases and negative results remain in tables and figures. Independent reviewers receive raw code, requirements and artifacts; agent review is not outside peer review or adoption.

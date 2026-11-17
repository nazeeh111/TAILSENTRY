# What the bound does, and what it does not

This is a source-bound worked example, not another benchmark study. The current local workbench ran the supported eight-factor Gaussian delta-gamma workload with budget 4,096, seed 42 and the fixed projected-union method. The displayed threshold is 4.64427487894; the UI rounds the model default to 12 significant digits. [Exported record](worked-record.json) retains the exact used threshold, model, source hashes, oracle counts, prefix trace and finite interval.

The estimate is approximately **1.00899e−4**, with 95% fixed-budget empirical-Bernstein bound **[9.40766e−5, 1.07721e−4]**. The workload's known reference is 1e−4. A single successful interval is not empirical coverage evidence. The band is the final bound repeated across the trace for comparison; it is not a sequential confidence band. Prefixes do not authorize optional stopping.

## Read the mechanism

For the supported maximum of quadratic losses under a Gaussian or declared Student-t scale mixture, a proved halfspace cover bounds the event. Correcting cover overlap gives a bounded contribution. Analytically integrating a Gaussian direction reduces variance within that proposal. It does not imply better elapsed-time efficiency than a different estimator. Severity, VaR and ES need separate conditions and are not certified by this probability bound. Arbitrary black-box loss functions are outside this conditional interface.

## Preserve the negative result

The middle panel uses the original [primary analysis](primary-analysis.json), not this newly executed record. Candidate versus conditional-Sobol MSE × time ratio is **38.789**, paired bootstrap 95% interval **[26.475, 57.322]**; lower is better. Versus raw union, the ratio is 0.02518. The within-proposal improvement does not overturn the failed primary target.

The bottom panel encodes every stored [missed-mode run](missed-mode-runs.json), in seed order. Its separate malicious two-sided Gaussian experiment has truth 1e−5, 20,000 draws per run and 10% defensive mass. Ordinary asymptotic intervals cover in only **3/512** runs. **509/512** runs miss the hidden mode, despite median observed tail ESS about 18,000. All finite bounds include truth, but are too broad to deliver the desired precision. This is abstention, not evidence of accurate useful tail prediction. [Original summary](missed-mode-summary.json).

## Inspect and reopen

[Wide evidence](evidence.html) and [phone evidence](evidence-mobile.html) are standalone reader pages with a real mobile viewport. [Current workspace](workspace.png) and [phone workspace](workspace-mobile.png) are actual Chrome captures, not designed product mockups. Figure fonts are Arial with portable Helvetica/sans-serif fallback. The figures preserve all 64 prefix points, both paired bootstrap intervals and all 512 outcome cells; the ratio axis is logarithmic and probability axis linear. The matrix has no physical-space meaning.

With Python 3.11+, run `python -B docs/presentation/check.py` from an extracted bundle root. It uses only the standard library for stored integrity and counts. Install the declared NumPy/SciPy dependencies and add `--replay` to recompute only the saved estimate. The recorded current runtime was Python 3.14.10, NumPy 2.5.4 and SciPy 1.18.1, distinct from the original research environment. Timing and digest changes in a fresh run are expected. `figure.py` regenerates SVG/HTML from stored records without sampling.

The ZIP contains this guide, runnable package source, license, exact records, figures and a complete SHA-256 manifest. It excludes the full research campaign, which remains in the source repository. The archive reader verifies byte identity, not mathematical validity or real-world distributional assumptions. The package workbench exports records, and recovers the prior completed result on reload in the same running service. It has no JSON-import GUI or restart-persistent record store; reopening here means reading/recomputing the native JSON with the documented reader.

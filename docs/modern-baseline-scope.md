# Modern adaptive baselines: comparison boundary

Checked 1 November 2026. This package does **not** establish superiority over
SAIS, FAMIS, modern learned proposals, or the rare-event literature generally.

## SAIS

Helal and Elvira, [Efficient rare event estimation for multimodal and
high-dimensional system reliability via subset adaptive importance
sampling](https://arxiv.org/html/2508.00210v1), arXiv:2508.00210v1, 2025.

Its Algorithm 1 and Sections 4–5 combine intermediate subsets, multiple
proposals, local reassignment, deterministic-mixture weighting, cross-entropy
updates, covariance adaptation, and recycling. The implemented `ce_mixture`
does not reproduce this algorithm. It must not be relabeled SAIS or used as
evidence that SAIS was beaten.

The paper's arXiv HTML and abstract, the author's
[homepage](https://victorelvira.github.io/), visible
[GitHub repository listing](https://github.com/victorelvira?tab=repositories),
and targeted searches for the title, identifier, authors, code, and repository
links did not identify an author-released implementation. The HTML GitHub links
are arXiv rendering-issue links. The publisher page could not be retrieved
(HTTP 403). These observations do not prove that no implementation exists.
No third-party source was copied and no authors were contacted.

A separate paper-informed NumPy/SciPy reproduction now exists in
`src/tailsentry/sais.py`, with23 tests, an equation concordance and all
ambiguities/guards documented in `docs/sais-fidelity.md`. Exploratory paper-case
runs, including the stated linear-example N5/K3000 settings, are retained under
`results/sais-exploratory-2026-11-01`. Paper-style recycling and the independent
confirmation extension are reported separately. Published tables have not been
reproduced, and SAIS was not inserted into the already frozen primary holdout.
This remains a comparison limitation, not evidence against SAIS.

## Newer flow-based work

Helal and Elvira, [Repulsive normalizing flow mixtures for adaptive importance
sampling: reliability analysis of complex
systems](https://arxiv.org/html/2609.21160v1), arXiv:2609.21160v1,
18 September 2026, proposes FAMIS. It learns a nonuniform flow mixture with
defensive exploration, mixture-weight adaptation and repulsive training.
Its results have not been reproduced here. The package cannot claim to beat it.

## What the current comparisons can establish

The package includes explicitly specified conventional baselines, analytic
conditional Monte Carlo, and conditional-union importance sampling. Comparisons
with those actual implementations support only the measured cases, budgets,
seeds, metrics, and machine environment. Independent complete-run repetitions
are essential; a single small standard error is insufficient evidence.

The envelope method reads exact quadratic coefficients, uses a spectral
decomposition, and integrates one-dimensional conditional events. Generic SAIS
and learned-proposal algorithms can operate through loss evaluations alone.
Equal numbers of calls do not equalize these information interfaces. Report
point evaluations, conditional-oracle evaluations, pilot selection,
preprocessing and wall time separately; use the analytic conditional baseline
to test whether gains merely come from extra structural information.

Conditional-union sampling and conditional Monte Carlo are established methods.
The current implementation's inspectable cover construction, exact Student-t
joint conditional sampler, selected projection, and bounded probability
diagnostics are engineering and numerical results. Their combination is not a
verified claim of mathematical priority. In particular,
[Ahn and Kim (2018)](https://doi.org/10.1145/3167969) already study conditional
Monte Carlo over halfspace unions under elliptical distributions, with radial
conditioning.

Before a state-of-the-art claim, complete the missing modern comparisons on
fixed holdout cases, document each method's available information, reproduce
published baseline behavior, and retain failures and all adaptation costs.

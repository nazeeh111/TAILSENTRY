# Reproduction

No credentials, external account, paid API, GPU or market feed is needed. Source data and evidence are retained. Work from the repository root. Use a fresh output directory for every experiment; the runners refuse overwrites.

## Environment and quick checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[research]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/tailsentry estimate --case delta_gamma_1e4 --budget 16384 --seed 42
.venv/bin/tailsentry serve
```

Recorded research environment: Python3.14.7, NumPy2.5.4, SciPy1.18.1; the research lock file records exact installed versions. CPU-only ARM64 macOS. Other operating systems and Python versions are supported by source configuration but are not claimed locally tested. GitHub CI is configured, not claimed executed before a push.

## Frozen probability studies

`results/final/receipt.json` defines the complete final dataset: seven secondary cases from main-v1 and the amended primary from primary-v2, with all partial original primary cells excluded. There are17,280 records and240 groups. Original protocol and amendment are separate. Every executed source file is retained beneath each run's `source/` folder and matches its configuration hash map.

For exact algorithm-version reproduction, put the corresponding source snapshot first on PYTHONPATH. Copy the retained runner and protocol into that snapshot if using an older artifact distribution without them; verify their hashes against configuration before execution. Runtimes will vary.

```bash
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/tailsentry-matplotlib

PYTHONPATH=results/holdout/main-v1/source/src .venv/bin/python \
  results/holdout/main-v1/source/scripts/benchmark.py \
  --output /tmp/tailsentry-secondary-reproduction --runs 64 --seed-start 210000 \
  --budgets 4096 16384 --cases single_1e3 paired4_1e4 imbalanced4_1e5 \
  correlated4_1e5 compact2_1e4 compact4_1e5 student_balls_1e4

PYTHONPATH=results/holdout/primary-v2/source/src .venv/bin/python \
  results/holdout/primary-v2/source/scripts/benchmark.py \
  --output /tmp/tailsentry-primary-reproduction --runs 128 --seed-start 220000 \
  --budgets 4096 16384 --cases delta_gamma_1e4
```

Main-v1's original all-case run explicitly refused a support-underflow case. Its log, configuration, raw partial groups and original source are retained. The amendment safely enlarges the cover instead of dropping positive eigendirections. No failure was silently replaced.

Current-source runners can reproduce the current engine into any fresh location. `scripts/aggregate.py` reconstructs final statistics from the recorded original input paths and refuses to overwrite its final directory; run it in a disposable copied checkout for full artifact regeneration. It uses all predefined groups, not selected winners. Seed901,4000 complete-run bootstrap resamples, paired and unpaired sensitivity. `scripts/timing.py` uses randomized sequential method order,32 repetitions, seed920; run it without other studies. Initial timing contention and separate process CPU observations are disclosed.

## Other validation

- `scripts/missed_mode.py`:512 malicious omitted-mode replications with independently specified truth.
- `scripts/validate_risk.py`:32 replications per model/level, Gaussian-max and quadratic VaR/ES. The quantile grid is deliberately oracle-centered, so this tests estimation on a supplied bracket, not automatic bracket discovery. Ordinary intervals undercover; finite grid brackets are reported separately.
- `scripts/prepare_fx.py --offline`: regenerate aligned FX data from saved original Fed HTML and verify hashes. A new online vintage must use a new data directory.
- `scripts/validate_fx.py`: training-only Gaussian/Student fits and chronological historical diagnostics. Outputs are deliberately immutable; consult `--help` for fresh output arguments.
- `results/sais-exploratory-2026-11-01/reproduce.py`: separate SAIS exploratory cases and paper-scale linear settings, with equation ambiguities disclosed in `docs/sais-fidelity.md`. Not part of the frozen primary comparison.
- `scripts/figures.py` and `scripts/paper_tables.py`: regenerate all figures/tables from saved actual results. SVG, PDF, PNG and plot data are retained.

## Manuscript

Use a free Tectonic installation or a standard LaTeX distribution. The local official Tectonic0.17.0 binary and receipt are under ignored `.tools/`; it is not a committed dependency. Its resource cache is project-local.

```bash
tectonic paper/tailsentry.tex --keep-logs --untrusted
```

The compiled PDF was rendered and visually inspected. Figure and text source remain editable. No arXiv submission or external publication is authorized or claimed.

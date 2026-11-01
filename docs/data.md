# Historical FX data and evidence limits

TAILSENTRY includes a small, reproducible financial-model check using public
Federal Reserve H.10 observations. Source: Board of Governors of the Federal
Reserve System, H.10 Foreign Exchange Rates. The observations are noon buying
rates in New York for cable transfers payable in foreign currencies, rather
than executable closing prices. The [Board's source description](https://www.federalreserve.gov/releases/h10/hist/)
establishes this convention. Its [copyright notice](https://www.federalreserve.gov/disclaimer.htm)
allows copying and distribution of Board information unless otherwise marked,
and requests attribution. These currency tables carry no contrary notice.
No Board logo or third-party photograph is included.

## Source records

| Currency | Original source | Original quote | Analysis quote |
|---|---|---|---|
| EUR | [Historical euro rates](https://www.federalreserve.gov/releases/h10/hist/dat00_eu.htm) | USD per EUR | Unchanged |
| JPY | [Historical yen rates](https://www.federalreserve.gov/releases/h10/hist/dat00_ja.htm) | JPY per USD | Reciprocal, USD per JPY |
| GBP | [Historical pound rates](https://www.federalreserve.gov/releases/h10/hist/dat00_uk.htm) | USD per GBP | Unchanged |

`data/raw/h10_*.html` contains the exact downloaded HTML bytes, with a SHA-256
digest, URL and UTC retrieval time in each corresponding receipt. This is a
revised source vintage retrieved on 1 November 2026, not a point-in-time record
of what a forecaster could have known on each historical date. Later published
observations in these raw pages are excluded from the analysis.

`python scripts/prepare_fx.py --offline` regenerates the prepared files from the
saved HTML without network access and checks the raw-source hashes. Running
without `--offline` downloads only missing raw files; it never silently replaces
an existing source vintage. To acquire a new vintage, use a fresh `--data-dir`.
The downloader uses the Python standard library and requires no credentials.

## Fixed preparation

The selected interval is 1 January 2000 through 31 December 2025. The procedure
intersects dates with valid observations for all three currencies before
computing returns. `ND` observations are missing, never zero; there is no
forward filling. It rejects nonpositive rates and conflicting duplicate dates.
The saved extraction contains 6,518 complete price observations and 6,517 return
observations. Of the 6,783 source dates in the selected interval, 265 are dropped
because at least one currency is missing. The first complete observation is
3 January 2000 and the last is 31 December 2025.

For consecutive available observations, the simple return is
`P_current / P_previous - 1`; the log return is `log(P_current / P_previous)`.
The recorded previous date and calendar-day count preserve the interval:
1,429 returns span more than one calendar day, including weekends and holidays.
No assumption of uninterrupted daily observations is hidden in the prepared
file. FX quotes require no corporate-action adjustment. Interest differentials,
financing, transaction costs and intraday variation are outside this dataset.

The basket loss is minus the mean of the three simple currency returns. This
represents equal USD-value exposures reset at the start of each observation
interval, rather than a buy-and-hold wealth series. The script does not optimize
weights or fit a distribution.

The chronological split is fixed:

- Training: returns ending through 31 December 2014, 3,769 observations.
- Evaluation: returns ending 1 January 2015 through 31 December 2025, 2,748
  observations.

An evaluation return can use its preceding training-period quote as the start
price. This is the observed interval's definition, not training on future
returns. Model parameters must be fitted on the training split alone unless a
separately specified past-only rolling rule is used. The evaluation period must
not choose distributions, portfolio weights or favorable crisis dates.

`data/fx_levels.csv` stores aligned USD quotes; `data/fx_returns.csv` stores
simple and log currency returns, the basket loss, interval and split.
`data/fx_provenance.json` records the transformations, counts and output hashes.
The first euro observation, 1.0155 USD per EUR on 3 January 2000, matches the
source table. Yen inversion is explicit in both code and metadata. The current
offline regeneration reproduced the same output hashes.

## What the data can establish

Observable-level historical VaR exceedances and ES calibration assess the
financial distributional assumptions. Gaussian and heavier-tailed fits may
produce materially different risk even when both numerical estimates have
narrow Monte Carlo intervals. A few thousand observations cannot empirically
validate a daily loss probability of 10^-5. Known-truth synthetic experiments
support the extreme-tail computational claims; historical checks support only
the stated historical model assessment. Neither implies regulatory approval,
predictive trading utility or correct future-market probabilities.

The original H.10 sources are mutable. Frozen raw bytes and receipts establish
the exact data used here. They do not remove historical revisions or establish
the absence of source errors.

# Commodity Cycle data inputs

`data/commodity_term_structure_seasonal_10y.xlsx` is the fixed historical
term-structure baseline. The loader reads only the `App_Export` sheet;
`Current_Snapshot` is deliberately ignored. Blank spread rows remain missing.

Current curves are fetched independently on normal app refresh: dynamically
generated Yahoo Finance individual futures contracts for NYMEX/CBOT contracts
(with the approved TradingView MCP fallback), and Westmetall LME Cash/3M tables
for Copper and Aluminum. Only completed synchronized EOD observations are
eligible. Front-contract rolls require a two-session volume crossover (OI only
when volume data is unavailable) or a five-business-day hard-roll buffer before
expiry / first notice. Contract quality and rejection reasons are retained in
the Diagnostics tab. Price-data quality and contract-selection quality are
reported separately; a `CALENDAR_FALLBACK` selection is LOW confidence even
when both observed prices are fresh. Current/daily observations, roll state and finalized
month-end observations are stored separately in `persistent/commodity_cycle/`
and mounted as a Docker persistent volume. Monthly spread is the arithmetic
mean of synchronized daily spreads; finalized months need at least five valid
daily observations. The historical workbook is never overwritten. Seasonal
percentiles combine same-month, same-pair baseline rows with subsequent stored
month-end observations, use only prior completed years, and remain N/A until
the configured 5Y/10Y comparable-history sample is sufficient. The current
curve spread is ranked immediately; no minimum number of current-month daily
observations is required. Historical observations are benchmark samples only
and are never substituted for a missing current signal. If the current spread
or sufficient same-season history is unavailable, the current percentile is
N/A. The provenance fields expose the current curve date, comparable-history
count, current/history vendors, and cross-vendor status. Yahoo and Westmetall
observations are marked stale after more than three trading/business days.

The Commodity Cycle XLSX download is generated from the running engine. It
includes current tables, normalized seasonal history, and the term-structure
audit. Reconstructed Agriculture history is included when available, even
though the immutable baseline workbook retains its original legacy notes.

Displayed commodity prices and drill-down daily charts use Yahoo Finance
continuous-futures tickers (`CL=F`, `NG=F`, `RB=F`, `HG=F`, `ALI=F`, `ZC=F`,
`ZW=F`, `ZS=F`). Performance and the derived Price State prefer weekly OHLCV
from TradingView MCP continuous futures: `NYMEX:CL1!`, `NYMEX:NG1!`,
`NYMEX:RB1!`, `COMEX:HG1!`, `COMEX:ALI1!`, `CBOT:ZC1!`, `CBOT:ZW1!`, and
`CBOT:ZS1!`. The loader requests 500 weekly bars and retries once; if
TradingView returns an empty/invalid response or an error, it uses the Yahoo
Finance daily close series already loaded for displayed prices and charts. The
1M/3M/6M/12M returns compare the selected source's latest close with the latest
available close at or before 4/13/26/52 weeks earlier. The Additional commodity
diagnostics table and XLSX export identify which provider supplied each asset's
performance series. CFTC positioning is read from the application's shared
`positioning.py` service; Commodity Cycle does not own a separate CFTC
refresh/parser.

The curve outputs intentionally keep two different concepts separate. `Raw
Curve State` is based only on the sign of the current spread (backwardation or
contango). `Seasonal Relative State` is the level of that spread within its
comparable seasonal distribution; it does not describe the direction of change.
The existing price/seasonal-curve interaction is displayed as `Price × Seasonal
Curve`, without changing its thresholds or weights.

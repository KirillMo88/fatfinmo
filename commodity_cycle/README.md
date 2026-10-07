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
the full 5Y/10Y sample is available. Current-month percentiles are provisional
until at least five synchronized daily spreads exist. Yahoo and Westmetall
observations are marked stale after more than three trading/business days.

If a current Energy or Metals percentile is unavailable, the model falls back
only to the latest completed observation from the same calendar month. The
provenance fields expose the percentile date, fallback status, comparable
history count, current/history vendors, and cross-vendor status. A 10Y fallback
requires ten prior same-month observations; a ten-year baseline can therefore
legitimately show `9/10` and N/A for the fallback observation.

The Commodity Cycle XLSX download is generated from the running engine. It
includes current tables, normalized seasonal history, and the term-structure
audit. Reconstructed Agriculture history is included when available, even
though the immutable baseline workbook retains its original legacy notes.

Daily commodity price history is loaded from Yahoo Finance continuous-futures
tickers (`CL=F`, `NG=F`, `RB=F`, `HG=F`, `ALI=F`, `ZC=F`, `ZW=F`, `ZS=F`). The
1M/3M/6M/12M returns compare the current price with the latest available price
at or before 4/13/26/52 weeks earlier. CFTC positioning is read from the
application's shared `positioning.py` service; Commodity Cycle does not own a
separate CFTC refresh/parser.

The curve outputs intentionally keep two different concepts separate. `Raw
Curve State` is based only on the sign of the current spread (backwardation or
contango). `Seasonal Relative State` is the level of that spread within its
comparable seasonal distribution; it does not describe the direction of change.
The existing price/seasonal-curve interaction is displayed as `Price × Seasonal
Curve`, without changing its thresholds or weights.

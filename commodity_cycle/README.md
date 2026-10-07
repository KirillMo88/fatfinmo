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
the Diagnostics tab. Current/daily observations, roll state and finalized
month-end observations are stored separately in `persistent/commodity_cycle/`
and mounted as a Docker persistent volume. Monthly spread is the arithmetic
mean of synchronized daily spreads; finalized months need at least five valid
daily observations. The historical workbook is never overwritten. Seasonal
percentiles combine same-month, same-pair baseline rows with subsequent stored
month-end observations, use only prior completed years, and remain N/A until
the full 5Y/10Y sample is available. Current-month percentiles are provisional
until at least five synchronized daily spreads exist. Yahoo and Westmetall
observations are marked stale after more than three trading/business days.

Monthly commodity price history is loaded from Yahoo Finance continuous-futures
tickers (`CL=F`, `NG=F`, `RB=F`, `HG=F`, `ALI=F`, `ZC=F`, `ZW=F`, `ZS=F`). The
current incomplete month is excluded. CFTC positioning is read from the
application's shared `positioning.py` service; Commodity Cycle does not own a
separate CFTC refresh/parser.

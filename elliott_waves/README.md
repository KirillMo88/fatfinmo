# Eliot waves integration

The visible tab name intentionally follows the product requirement: `Eliot waves`.
Python modules use the conventional `elliott_waves` spelling.

## Data contracts

| Asset | Provider symbol | Source | Analytical base |
|---|---|---|---|
| SPX | `^GSPC` | Yahoo Finance | closed `1D` bars |
| NDX | `^NDX` | Yahoo Finance | closed `1D` bars |
| GOLD | `TVC:GOLD` | TradingView MCP | closed `1W` bars |
| BTCUSD | `INDEX:BTCUSD` | TradingView MCP | closed `1W` bars |

`TVC:GOLD` is labelled in the UI and exports as a CFD/index proxy, not physical
spot. `INDEX:BTCUSD` is the TradingView Bitcoin all-time-history index, not a
BTCUSDT exchange pair.

GOLD and BTCUSD use the explicitly approved weekly V2 fallback because the
TradingView MCP OHLCV method exposes at most 5,000 bars and no historical
pagination. Their `1D` view is fetched directly from TradingView and may have a
shorter history; daily bars are never reconstructed from weekly candles.

## Runtime

Initial/manual refresh:

```powershell
python refresh_jobs.py elliott-waves
```

The normal scheduler recalculates the analytical snapshots in the nightly job.
The 10-minute market overlay updates only the displayed quote and does not
modify the wave tree.

Snapshots are written below `persistent/elliott_waves/`. The tab reads the
published manifest and snapshot files only; opening the tab, changing candle
aggregation, date range, scale, scenario, or visible degree does not invoke the
engine. JSON and XLSX exports are generated from those same published snapshot
identifiers.

"""Data retrieval and common-date return calculations for PPI-led ETF analysis."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 9, 12, 24, 36)
COUNTRY_ASSETS = (
    ("RTSI", "Russia"), ("ECH", "Chile"), ("ENOR", "Norway"), ("EWC", "Canada"),
    ("EWA", "Australia"), ("EZA", "South Africa"), ("EPU", "Peru"), ("EWZ", "Brazil"),
)
RESOURCE_ASSETS = (
    ("GUNR", "Global Natural Resources"), ("GNR", "Global Natural Resources"),
    ("PICK", "Global Metals & Mining"), ("MXI", "Global Materials"), ("IXC", "Global Energy"),
    ("VEGI", "Agriculture Producers"), ("WOOD", "Timber & Forestry"),
    ("COPX", "Copper Miners"), ("URNM", "Uranium Miners"),
)
YFINANCE_TICKERS = tuple(ticker for ticker, _ in (*COUNTRY_ASSETS[1:], *RESOURCE_ASSETS))
EQUITY_ASSETS = tuple(ticker for ticker, _ in (*COUNTRY_ASSETS, *RESOURCE_ASSETS))


def _clean_price_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").copy()
    dates = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None).normalize()
    values.index = dates
    values = values.loc[~values.index.isna()].dropna()
    return values.groupby(level=0).last().sort_index().astype(float)


def calculate_period_returns(
    series: pd.Series,
    calculation_date: pd.Timestamp | str,
    horizons: Iterable[int] = HORIZONS,
) -> dict[str, float]:
    """Calculate calendar-month cumulative returns ending on/before a common date."""
    clean = _clean_price_series(series)
    result: dict[str, float] = {f"{int(month)}M": np.nan for month in horizons}
    if clean.empty:
        return result
    end_date = pd.Timestamp(calculation_date).normalize()
    end_values = clean.loc[clean.index <= end_date]
    if end_values.empty:
        return result
    end_price = float(end_values.iloc[-1])
    if not np.isfinite(end_price) or end_price == 0:
        return result
    for month in result:
        target_date = end_date - pd.DateOffset(months=int(month[:-1]))
        start_values = clean.loc[clean.index <= target_date]
        if start_values.empty:
            continue
        start_price = float(start_values.iloc[-1])
        if np.isfinite(start_price) and start_price != 0:
            result[month] = end_price / start_price - 1.0
    return result


def build_performance_table(
    ppiaco: pd.Series,
    market_prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build rows in the prescribed order and compute aggregates over equities only."""
    ppi = _clean_price_series(ppiaco)
    if ppi.empty:
        return pd.DataFrame(columns=["Asset", *[f"{month}M" for month in HORIZONS]]), None
    calculation_date = ppi.index.max()
    rows: list[dict[str, Any]] = []

    def add_row(ticker: str, label: str, series: pd.Series) -> None:
        row = {"Asset": label}
        row.update(calculate_period_returns(series, calculation_date))
        rows.append(row)

    add_row("PPIACO", "PPIACO", ppi)
    for ticker, description in COUNTRY_ASSETS:
        add_row(ticker, f"{ticker} — {description}", market_prices.get(ticker, pd.Series(dtype=float)))
    for ticker, description in RESOURCE_ASSETS:
        add_row(ticker, f"{ticker} — {description}", market_prices.get(ticker, pd.Series(dtype=float)))

    equity_rows = pd.DataFrame(rows[1:])
    for label, operation in (("Average", "mean"), ("Median", "median")):
        summary = {"Asset": label}
        for month in HORIZONS:
            column = f"{month}M"
            values = pd.to_numeric(equity_rows[column], errors="coerce")
            summary[column] = getattr(values, operation)() if values.notna().any() else np.nan
        rows.append(summary)
    return pd.DataFrame(rows), calculation_date


def download_market_prices(calculation_date: pd.Timestamp | str) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Fetch adjusted ETF closes in one Yahoo batch plus RTSI from TradingView MCP."""
    as_of = pd.Timestamp(calculation_date).normalize()
    start = (as_of - pd.DateOffset(months=40)).date().isoformat()
    end = (as_of + pd.Timedelta(days=1)).date().isoformat()  # yfinance end is exclusive
    prices: dict[str, pd.Series] = {}
    status: dict[str, str] = {}

    try:
        import yfinance as yf

        frame = yf.download(
            tickers=list(YFINANCE_TICKERS), start=start, end=end, interval="1d",
            auto_adjust=True, group_by="ticker", progress=False, threads=False,
        )
        for ticker in YFINANCE_TICKERS:
            try:
                if frame is None or frame.empty:
                    raise ValueError("empty yfinance response")
                if isinstance(frame.columns, pd.MultiIndex):
                    if (ticker, "Close") in frame.columns:
                        close = frame[(ticker, "Close")]
                    elif ("Close", ticker) in frame.columns:
                        close = frame[("Close", ticker)]
                    else:
                        raise KeyError("adjusted Close column missing")
                elif ticker in YFINANCE_TICKERS and "Close" in frame:
                    close = frame["Close"]
                else:
                    raise KeyError("adjusted Close column missing")
                series = _clean_price_series(close)
                if series.empty:
                    raise ValueError("no valid adjusted closes")
                prices[ticker] = series
                status[ticker] = f"yfinance adjusted close; last {series.index[-1]:%Y-%m-%d}"
            except Exception as exc:
                status[ticker] = f"yfinance unavailable: {exc}"
    except Exception as exc:
        for ticker in YFINANCE_TICKERS:
            status[ticker] = f"yfinance unavailable: {exc}"

    try:
        from tradingview_mcp import get_ohlcv_data

        bars = get_ohlcv_data("RUS:RTSI", interval="1D", count=1500)
        if bars is None or bars.empty or not {"date", "close"}.issubset(bars.columns):
            raise ValueError("empty/invalid OHLCV response for RUS:RTSI")
        dates = pd.to_datetime(bars["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        series = pd.Series(pd.to_numeric(bars["close"], errors="coerce").to_numpy(), index=dates)
        series = _clean_price_series(series)
        if series.empty:
            raise ValueError("no valid closes for RUS:RTSI")
        prices["RTSI"] = series
        status["RTSI"] = f"TradingView MCP RUS:RTSI; last {series.index[-1]:%Y-%m-%d}"
    except Exception as exc:
        status["RTSI"] = f"TradingView MCP unavailable (RUS:RTSI): {exc}"
    return prices, status


def relative_median_frame(table: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Return sorted equity-only relative returns and ranks for the chosen horizon."""
    if table.empty or horizon not in table:
        return pd.DataFrame(columns=["Ticker", "Asset Return", "Median Return", "Relative", "Rank"])
    median_rows = table.loc[table["Asset"].eq("Median"), horizon]
    if median_rows.empty or pd.isna(median_rows.iloc[0]):
        return pd.DataFrame(columns=["Ticker", "Asset Return", "Median Return", "Relative", "Rank"])
    median = float(median_rows.iloc[0])
    equities = table.loc[table["Asset"].str.split(" — ").str[0].isin(EQUITY_ASSETS), ["Asset", horizon]].copy()
    equities[horizon] = pd.to_numeric(equities[horizon], errors="coerce")
    equities = equities.dropna(subset=[horizon]).sort_values(horizon, ascending=False).reset_index(drop=True)
    equities["Ticker"] = equities["Asset"].str.split(" — ").str[0]
    equities["Asset Return"] = equities[horizon]
    equities["Median Return"] = median
    equities["Relative"] = equities["Asset Return"] - median
    equities["Rank"] = np.arange(1, len(equities) + 1)
    return equities[["Ticker", "Asset Return", "Median Return", "Relative", "Rank"]]

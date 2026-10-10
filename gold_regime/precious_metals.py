from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from time_ranges import TIME_RANGE_YEARS


YFINANCE_TICKERS = ("GLD", "GDX", "SIL", "SLV")
TRADINGVIEW_SYMBOLS = {
    "SILVER": "TVC:SILVER",
    "PALLADIUM": "TVC:PALLADIUM",
    "PLATINUM": "TVC:PLATINUM",
    "XAUUSD": "OANDA:XAUUSD",
}
PRECIOUS_METALS_ROWS = (
    ("Silver / XAUUSD", "SILVER", "XAUUSD"),
    ("Palladium / XAUUSD", "PALLADIUM", "XAUUSD"),
    ("Platinum / XAUUSD", "PLATINUM", "XAUUSD"),
)
PRODUCERS_VS_BULLION_ROWS = (
    ("GDX / GLD", "GDX", "GLD"),
    ("SIL / SLV", "SIL", "SLV"),
)
CHANGE_HORIZONS = (
    ("1W Change", pd.DateOffset(weeks=1)),
    ("1M Change", pd.DateOffset(months=1)),
    ("3M Change", pd.DateOffset(months=3)),
    ("6M Change", pd.DateOffset(months=6)),
    ("12M Change", pd.DateOffset(months=12)),
    ("2Y Change", pd.DateOffset(years=2)),
    ("3Y Change", pd.DateOffset(years=3)),
    ("5Y Change", pd.DateOffset(years=5)),
    ("10Y Change", pd.DateOffset(years=10)),
)


def load_precious_metals_ratios() -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch weekly metal prices and build the relative-price series used by Gold Regime."""
    yahoo_prices, status = _download_yahoo_weekly()
    tradingview_prices: dict[str, pd.Series] = {}
    try:
        from tradingview_mcp import get_ohlcv_data
    except Exception as exc:
        get_ohlcv_data = None
        status["TradingView MCP"] = f"Unavailable: {exc}"

    for name, symbol in TRADINGVIEW_SYMBOLS.items():
        if get_ohlcv_data is None:
            tradingview_prices[name] = pd.Series(dtype="float64")
            continue
        try:
            bars = get_ohlcv_data(symbol, interval="1W", count=1500)
            if bars is None or bars.empty or not {"date", "close"}.issubset(bars.columns):
                raise ValueError("empty or invalid OHLCV response")
            dates = pd.to_datetime(bars["date"], errors="coerce", utc=True).dt.tz_localize(None)
            series = pd.Series(
                pd.to_numeric(bars["close"], errors="coerce").to_numpy(),
                index=dates,
                name=name,
            )
            series = _weekly_series(series)
            if series.empty:
                raise ValueError("no valid weekly closes")
            tradingview_prices[name] = series
            status[symbol] = f"TradingView MCP; latest weekly close {series.index[-1]:%Y-%m-%d}"
        except Exception as exc:
            tradingview_prices[name] = pd.Series(dtype="float64")
            status[symbol] = f"Unavailable: {exc}"

    prices = {**yahoo_prices, **tradingview_prices}
    ratios: dict[str, pd.Series] = {}
    for label, numerator, denominator in (*PRECIOUS_METALS_ROWS, *PRODUCERS_VS_BULLION_ROWS):
        ratios[label] = _price_ratio(prices.get(numerator), prices.get(denominator), label)

    history = pd.concat(ratios.values(), axis=1).sort_index() if ratios else pd.DataFrame()
    return history, status


def build_change_table(
    history: pd.DataFrame,
    rows: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for label, _, _ in rows:
        series = _history_series(history, label)
        row: dict[str, Any] = {"Ratio": label}
        row.update({horizon: _period_change(series, offset) for horizon, offset in CHANGE_HORIZONS})
        output.append(row)
    return pd.DataFrame(output, columns=["Ratio", *[horizon for horizon, _ in CHANGE_HORIZONS]])


def build_indexed_chart_history(
    history: pd.DataFrame,
    selected_range: str = "10Y",
    range_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["Date"])
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if frame.empty:
        return pd.DataFrame(columns=["Date"])
    end = pd.to_datetime(range_end, errors="coerce") if range_end is not None else pd.NaT
    end = frame.index.max() if pd.isna(end) else pd.Timestamp(end).normalize()
    frame = frame.loc[frame.index <= end]
    years = TIME_RANGE_YEARS.get(selected_range)
    if years is not None:
        frame = frame.loc[frame.index >= end - pd.DateOffset(years=years)]
    indexed = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty or series.iloc[0] == 0:
            continue
        indexed[column] = frame[column] / float(series.iloc[0]) * 100.0
    indexed.index.name = "Date"
    return indexed.reset_index()


def _download_yahoo_weekly() -> tuple[dict[str, pd.Series], dict[str, str]]:
    prices = {ticker: pd.Series(dtype="float64", name=ticker) for ticker in YFINANCE_TICKERS}
    status: dict[str, str] = {}
    try:
        import yfinance as yf

        data = yf.download(
            tickers=list(YFINANCE_TICKERS),
            period="max",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return prices, {ticker: f"yfinance unavailable: {exc}" for ticker in YFINANCE_TICKERS}

    for ticker in YFINANCE_TICKERS:
        try:
            close = _yahoo_close_column(data, ticker)
            series = _weekly_series(close)
            if series.empty:
                raise ValueError("no valid adjusted closes")
            prices[ticker] = series.rename(ticker)
            status[ticker] = f"yfinance adjusted close; latest weekly close {series.index[-1]:%Y-%m-%d}"
        except Exception as exc:
            status[ticker] = f"yfinance unavailable: {exc}"
    return prices, status


def _yahoo_close_column(frame: pd.DataFrame, ticker: str) -> pd.Series:
    if frame is None or frame.empty:
        raise ValueError("empty yfinance response")
    if isinstance(frame.columns, pd.MultiIndex):
        for key in ((ticker, "Close"), ("Close", ticker)):
            if key in frame.columns:
                return frame[key]
        raise KeyError("adjusted Close column missing")
    if "Close" not in frame.columns:
        raise KeyError("adjusted Close column missing")
    return frame["Close"]


def _weekly_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(series, errors="coerce").copy()
    dates = pd.to_datetime(values.index, errors="coerce", utc=True)
    values.index = dates.tz_localize(None).normalize()
    values = values.loc[~values.index.isna()].dropna().sort_index()
    if values.empty:
        return values
    return values.groupby(level=0).last().resample("W-FRI").last().dropna().astype(float)


def _price_ratio(numerator: pd.Series | None, denominator: pd.Series | None, name: str) -> pd.Series:
    if numerator is None or denominator is None or numerator.empty or denominator.empty:
        return pd.Series(dtype="float64", name=name)
    numerator, denominator = numerator.align(denominator, join="inner")
    ratio = numerator / denominator
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return ratio.rename(name)


def _history_series(history: pd.DataFrame, column: str) -> pd.Series:
    if history is None or history.empty or column not in history.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(history[column], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(history.index, errors="coerce")).dropna().sort_index()


def _period_change(series: pd.Series, offset: pd.DateOffset) -> float:
    if series is None or series.empty:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty:
        return np.nan
    current_date = values.index[-1]
    current_value = float(values.iloc[-1])
    baseline = values.loc[values.index <= current_date - offset]
    if baseline.empty or baseline.iloc[-1] == 0:
        return np.nan
    return current_value / float(baseline.iloc[-1]) - 1.0

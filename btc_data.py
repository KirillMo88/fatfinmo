from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from btc_cycle import BTC_ARCHIVE_END, BTC_ARCHIVE_START, merge_btc_mcp_weekly_history
from finance_core import download_completed_ohlcv
from fund_flows import FundFlowCache, default_fund_flow_cache_path


BTC_SPOT_ETF_FLOW_TICKERS = (
    "IBIT", "FBTC", "GBTC", "ARKB", "BITB", "BTCO", "EZBC", "HODL", "BRRR", "BTCW",
)

BTC_WEEKLY_ARCHIVE_PATH = Path(__file__).with_name("data") / "btc" / "btc_weekly_tradingview_2009_2025.csv"
def read_btc_weekly_archive() -> pd.DataFrame:
    """Read the immutable TradingView BTC weekly history through 2025."""
    if not BTC_WEEKLY_ARCHIVE_PATH.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "source"])
    try:
        frame = pd.read_csv(BTC_WEEKLY_ARCHIVE_PATH)
    except Exception:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "source"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.loc[(frame["date"] >= BTC_ARCHIVE_START) & (frame["date"] <= BTC_ARCHIVE_END)]
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=21600)
def load_btc_weekly_price() -> pd.DataFrame:
    """Return the canonical BTC weekly history used by BTC Cycle."""
    daily = download_completed_ohlcv("BTC-USD", period="max", allow_archive_fallback=False)
    archive = read_btc_weekly_archive()
    if archive.empty:
        # Emergency compatibility path for an installation missing the
        # bundled archive; normal releases always use the immutable archive.
        try:
            from tradingview_mcp import get_ohlcv_data

            tradingview_weekly = get_ohlcv_data("INDEX:BTCUSD", interval="1W", count=5000)
        except Exception:
            tradingview_weekly = pd.DataFrame()
        return merge_btc_mcp_weekly_history(daily, tradingview_weekly)
    # The frozen archive is authoritative through 2025-12-31.  Yahoo BTC-USD
    # is the only live source after that date; do not mix in a newer TradingView
    # response that could revise the archived history or duplicate live bars.
    return merge_btc_mcp_weekly_history(daily, pd.DataFrame(), archive_weekly=archive)


def load_btc_etf_flow_history() -> pd.DataFrame:
    """Build the weekly US spot-BTC ETF flow history used by BTC Cycle."""
    cache = FundFlowCache(default_fund_flow_cache_path())
    frames: list[pd.DataFrame] = []
    for ticker in BTC_SPOT_ETF_FLOW_TICKERS:
        observations = cache.load_observations(ticker, pd.Timestamp("2024-01-01").date())
        if not observations:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime([obs.date for obs in observations]),
                    "ticker": ticker,
                    "net_flow": [obs.net_flow for obs in observations],
                    "aum": [obs.aum for obs in observations],
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ETF_Flow_1W",
                "ETF_Flow_4W",
                "ETF_Flow_13W",
                "ETF_Flow_Intensity_4W",
                "ETF_Flow_3Y_Pctl",
                "ETF_Flow_13W_Pctl",
                "ETF_Total_AUM",
                "ETF_Coverage_Count",
            ]
        )
    daily = pd.concat(frames, ignore_index=True)
    daily["net_flow"] = pd.to_numeric(daily["net_flow"], errors="coerce")
    daily["aum"] = pd.to_numeric(daily["aum"], errors="coerce")
    weekly_by_ticker = (
        daily.dropna(subset=["date", "net_flow"])
        .set_index("date")
        .groupby("ticker")
        .resample("W-FRI")
        .agg(net_flow=("net_flow", "sum"), aum=("aum", "last"))
        .dropna(subset=["net_flow"])
        .reset_index()
    )
    weekly = (
        weekly_by_ticker.groupby("date")
        .agg(
            ETF_Flow_1W=("net_flow", "sum"),
            ETF_Total_AUM=("aum", "sum"),
            ETF_Coverage_Count=("ticker", "nunique"),
        )
        .sort_index()
    )
    weekly["ETF_Flow_4W"] = weekly["ETF_Flow_1W"].rolling(4, min_periods=1).sum()
    weekly["ETF_Flow_13W"] = weekly["ETF_Flow_1W"].rolling(13, min_periods=1).sum()
    aum_ref = pd.to_numeric(weekly["ETF_Total_AUM"], errors="coerce").replace(0.0, np.nan)
    weekly["ETF_Flow_Intensity_4W"] = (weekly["ETF_Flow_4W"] / aum_ref) * 100.0
    fallback_scale = weekly["ETF_Flow_1W"].abs().rolling(156, min_periods=52).median()
    fallback_intensity = weekly["ETF_Flow_4W"] / (4.0 * fallback_scale.replace(0.0, np.nan))
    weekly["ETF_Flow_Intensity_4W"] = weekly["ETF_Flow_Intensity_4W"].where(
        weekly["ETF_Flow_Intensity_4W"].notna(), fallback_intensity,
    )
    weekly["ETF_Flow_3Y_Pctl"] = _trailing_percentile(weekly["ETF_Flow_Intensity_4W"], 156, 52)
    weekly["ETF_Flow_13W_Pctl"] = _trailing_percentile(weekly["ETF_Flow_13W"], 156, 104)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return weekly[weekly.index <= today].reset_index()


def _trailing_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def rank_last(window_values: np.ndarray) -> float:
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods or not np.isfinite(window_values[-1]):
            return np.nan
        return float((clean <= window_values[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)

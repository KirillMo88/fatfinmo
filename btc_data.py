from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from btc_cycle import merge_btc_mcp_weekly_history
from finance_core import download_completed_ohlcv
from fund_flows import FundFlowCache, default_fund_flow_cache_path


BTC_SPOT_ETF_FLOW_TICKERS = (
    "IBIT", "FBTC", "GBTC", "ARKB", "BITB", "BTCO", "EZBC", "HODL", "BRRR", "BTCW",
)


@st.cache_data(show_spinner=False, ttl=21600)
def load_btc_weekly_price() -> pd.DataFrame:
    """Return the canonical BTC weekly history used by BTC Cycle."""
    daily = download_completed_ohlcv("BTC-USD", period="max")
    try:
        from tradingview_mcp import get_ohlcv_data

        tradingview_weekly = get_ohlcv_data("INDEX:BTCUSD", interval="1W", count=5000)
    except Exception:
        tradingview_weekly = pd.DataFrame()
    return merge_btc_mcp_weekly_history(daily, tradingview_weekly)


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

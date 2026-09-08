from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_market_regime(weekly_ohlcv: pd.DataFrame, config: dict) -> dict:
    cfg = config["market_regime"]
    if weekly_ohlcv is None or weekly_ohlcv.empty:
        return unknown_market_regime()

    close = pd.to_numeric(weekly_ohlcv["Close"], errors="coerce").dropna().sort_index()
    if close.empty:
        return unknown_market_regime()

    high_source = weekly_ohlcv["High"] if "High" in weekly_ohlcv.columns else close
    high = pd.to_numeric(high_source, errors="coerce").reindex(close.index).ffill()
    sma40 = close.rolling(int(cfg["spy_sma_weeks"]), min_periods=int(cfg["spy_sma_weeks"])).mean()
    high52w = high.rolling(52, min_periods=52).max()
    weekly_returns = close.pct_change()
    vol13w = weekly_returns.rolling(int(cfg["vol_window_weeks"]), min_periods=int(cfg["vol_window_weeks"])).std() * np.sqrt(52.0)

    current_close = float(close.iloc[-1])
    current_sma40 = safe_last(sma40)
    current_high52w = safe_last(high52w)
    current_vol13w = safe_last(vol13w)
    if not np.isfinite(current_sma40) or not np.isfinite(current_high52w) or not np.isfinite(current_vol13w):
        return unknown_market_regime()

    spy_vs_sma40 = current_close / current_sma40 - 1.0
    spy_drawdown = current_close / current_high52w - 1.0
    vol_percentile = percentile_rank(vol13w.dropna(), current_vol13w)

    structural_bull = current_close > current_sma40 and spy_drawdown > float(cfg["drawdown_threshold"])
    high_vol = vol_percentile >= float(cfg["high_vol_percentile"])
    if structural_bull and not high_vol:
        regime = "BULL"
    elif structural_bull and high_vol:
        regime = "BULL_HIGH_VOL"
    elif not structural_bull and not high_vol:
        regime = "CORRECTION"
    else:
        regime = "STRESS"

    confidence = float(cfg["alpha_confidence"].get(regime, np.nan))
    return {
        "Market_Regime": regime,
        "Alpha_Confidence": confidence,
        "SPY_vs_SMA40W_%": spy_vs_sma40 * 100.0,
        "SPY_Drawdown_52W_%": spy_drawdown * 100.0,
        "SPY_Volatility_13W_%": current_vol13w * 100.0,
        "SPY_Volatility_Percentile": vol_percentile,
    }


def safe_last(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def percentile_rank(history: pd.Series, current_value: float) -> float:
    values = pd.to_numeric(history, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or not np.isfinite(current_value):
        return np.nan
    return float((values <= current_value).sum() / len(values) * 100.0)


def unknown_market_regime() -> dict:
    return {
        "Market_Regime": "UNKNOWN",
        "Alpha_Confidence": np.nan,
        "SPY_vs_SMA40W_%": np.nan,
        "SPY_Drawdown_52W_%": np.nan,
        "SPY_Volatility_13W_%": np.nan,
        "SPY_Volatility_Percentile": np.nan,
    }

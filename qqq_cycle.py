from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def expanding_percentile(series: pd.Series, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result: list[float] = []
    for index, value in enumerate(values):
        history = values.iloc[: index + 1].dropna()
        if len(history) < min_periods or not np.isfinite(safe_float(value)):
            result.append(np.nan)
        else:
            result.append(float((history <= value).mean() * 100.0))
    return pd.Series(result, index=series.index)


def confirm_state(candidate: pd.Series, required: int = 2) -> pd.Series:
    result: list[str] = []
    active: str | None = None
    run_value: str | None = None
    run_length = 0
    for raw_value in candidate.astype("object"):
        value = str(raw_value) if pd.notna(raw_value) else "DATA INCOMPLETE"
        if value == "DATA INCOMPLETE":
            result.append(active or "DATA INCOMPLETE")
            continue
        if value == run_value:
            run_length += 1
        else:
            run_value = value
            run_length = 1
        if active is None:
            active = value
        elif value != active and run_length >= required:
            active = value
        result.append(active)
    return pd.Series(result, index=candidate.index)


def classify_momentum_phase(level: Any, momentum: Any) -> str:
    level_value = safe_float(level)
    momentum_value = safe_float(momentum)
    if not np.isfinite(level_value) or not np.isfinite(momentum_value):
        return "DATA INCOMPLETE"
    if level_value > 0 and momentum_value > 0:
        return "RISK EXPANSION"
    if level_value > 0 and momentum_value < 0:
        return "LATE RISK EXPANSION"
    if level_value < 0 and momentum_value < 0:
        return "RISK CONTRACTION"
    if level_value < 0 and momentum_value > 0:
        return "EARLY RISK RECOVERY"
    return "TRANSITION"


def classify_structural_roc_momentum(value: Any) -> str:
    momentum = safe_float(value)
    if not np.isfinite(momentum):
        return "DATA INCOMPLETE"
    if momentum > 0.03:
        return "ACCELERATING"
    if momentum < -0.03:
        return "DECELERATING"
    return "STABLE"


def classify_extension_zone(value: Any) -> str:
    percentile = safe_float(value)
    if not np.isfinite(percentile):
        return "DATA INCOMPLETE"
    if percentile < 10:
        return "EXTREME OVERSOLD"
    if percentile < 25:
        return "OVERSOLD"
    if percentile < 75:
        return "NORMAL"
    if percentile < 90:
        return "EXTENDED"
    if percentile < 97.5:
        return "OVEREXTENDED"
    return "EXTREME OVEREXTENSION"


def build_structural_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    result = monthly.copy()
    if result.empty:
        return result
    close = pd.to_numeric(result["SPX_Close"], errors="coerce")
    result["SPX_SMA200M"] = close.rolling(200, min_periods=120).mean()
    result["StructuralExtensionPct"] = close / result["SPX_SMA200M"] - 1.0
    result["StructuralExtensionPercentile"] = expanding_percentile(
        result["StructuralExtensionPct"], min_periods=120
    )
    result["StructuralExtensionZone"] = result["StructuralExtensionPercentile"].map(
        classify_extension_zone
    )
    result["SPX_ROC36M"] = close.pct_change(36, fill_method=None)
    result["SPX_ROC36M_3MMA"] = result["SPX_ROC36M"].rolling(3, min_periods=2).mean()
    result["StructuralROCMomentum"] = result["SPX_ROC36M_3MMA"] - result[
        "SPX_ROC36M_3MMA"
    ].shift(12)
    result["StructuralROCMomentumState"] = result["StructuralROCMomentum"].map(
        classify_structural_roc_momentum
    )
    candidate = pd.Series(
        [
            classify_momentum_phase(level, momentum)
            for level, momentum in zip(
                result["SPX_ROC36M_3MMA"], result["StructuralROCMomentum"]
            )
        ],
        index=result.index,
    )
    result["StructuralMomentumCyclePhase"] = confirm_state(candidate, required=2)
    result["StructuralMarketCyclePhase"] = "DATA INCOMPLETE"
    return result


def build_momentum_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    result = monthly[["Date", "SPX_Close"]].copy()
    if result.empty:
        return result
    close = pd.to_numeric(result["SPX_Close"], errors="coerce")
    result["SPX_ROC12M"] = close.pct_change(12, fill_method=None)
    result["SPX_ROC12M_3MMA"] = result["SPX_ROC12M"].rolling(3, min_periods=2).mean()
    result["SPX_ROC_Momentum_3M"] = result["SPX_ROC12M_3MMA"] - result[
        "SPX_ROC12M_3MMA"
    ].shift(3)
    candidate = pd.Series(
        [
            classify_momentum_phase(level, momentum)
            for level, momentum in zip(
                result["SPX_ROC12M_3MMA"], result["SPX_ROC_Momentum_3M"]
            )
        ],
        index=result.index,
    )
    result["MomentumCyclePhase"] = confirm_state(candidate, required=2)
    return result


def build_ndx_cycle_history(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty or "Close" not in prices:
        return pd.DataFrame()

    source = prices.copy()
    source.index = pd.to_datetime(source.index, errors="coerce").tz_localize(None)
    source = source.loc[~source.index.isna()].sort_index()
    close = pd.to_numeric(source["Close"], errors="coerce").dropna()
    if close.empty:
        return pd.DataFrame()

    last_daily = pd.Timestamp(close.index.max()).normalize()
    monthly_close = close.resample("ME").last().dropna()
    monthly_close = monthly_close.loc[monthly_close.index <= last_daily]
    monthly = pd.DataFrame({"Date": monthly_close.index, "SPX_Close": monthly_close.to_numpy()})
    structural = build_structural_monthly(monthly)
    momentum = build_momentum_monthly(monthly)

    weekly_close = close.resample("W-FRI").last().dropna()
    if not weekly_close.empty and weekly_close.index[-1] > last_daily:
        weekly_close = weekly_close.iloc[:-1]
    weekly = pd.DataFrame({"Date": weekly_close.index, "SPX_Close": weekly_close.to_numpy()})
    if weekly.empty:
        return weekly

    for context in (structural, momentum):
        if context.empty:
            continue
        context_columns = context.drop(columns=["SPX_Close"], errors="ignore").copy()
        context_columns["Date"] = pd.to_datetime(context_columns["Date"], errors="coerce")
        weekly = pd.merge_asof(
            weekly.sort_values("Date"),
            context_columns.dropna(subset=["Date"]).sort_values("Date"),
            on="Date",
            direction="backward",
        )

    ndx_close = pd.to_numeric(weekly["SPX_Close"], errors="coerce")
    weekly["SPX_SMA200W"] = ndx_close.rolling(200, min_periods=200).mean()
    weekly["SMA200WExtensionPct"] = ndx_close / weekly["SPX_SMA200W"] - 1.0
    weekly["SMA200WExtensionPercentile"] = expanding_percentile(
        weekly["SMA200WExtensionPct"], min_periods=200
    )
    weekly["SMA200WZone"] = weekly["SMA200WExtensionPercentile"].map(
        classify_extension_zone
    )
    weekly["NDX_Close"] = ndx_close
    return weekly.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

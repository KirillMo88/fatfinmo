from bisect import bisect_right, insort

import numpy as np
import pandas as pd


def percentile_rank(values: pd.Series, current_value: float) -> float:
    """Inclusive percentile rank of current_value inside its own historical distribution."""
    if not np.isfinite(current_value):
        return np.nan

    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return np.nan

    return float((numeric <= current_value).mean() * 100.0)


def expanding_percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=numeric.index, dtype="float64")
    sorted_values: list[float] = []

    for i, value in enumerate(numeric):
        if not np.isfinite(value):
            continue
        value = float(value)
        insort(sorted_values, value)
        result.iloc[i] = (bisect_right(sorted_values, value) / len(sorted_values)) * 100.0

    return result


def calendar_performance_series(close: pd.Series, days: int) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    result = pd.Series(np.nan, index=prices.index, dtype="float64")
    if prices.empty:
        return result

    dates = prices.index
    offset = pd.DateOffset(days=int(days))
    for i, dt in enumerate(dates):
        start_dt = dt - offset
        start_pos = dates.searchsorted(start_dt, side="right") - 1
        if start_pos < 0:
            continue
        v0 = float(prices.iloc[start_pos])
        v1 = float(prices.iloc[i])
        if v0 == 0.0 or not np.isfinite(v0) or not np.isfinite(v1):
            continue
        result.iloc[i] = (v1 / v0 - 1.0) * 100.0

    return result


def forward_calendar_return_series(close: pd.Series, days: int) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    result = pd.Series(np.nan, index=prices.index, dtype="float64")
    if prices.empty:
        return result

    dates = prices.index
    last_dt = dates[-1]
    offset = pd.DateOffset(days=int(days))
    for i, dt in enumerate(dates):
        forward_dt = dt + offset
        if forward_dt > last_dt:
            continue
        end_pos = dates.searchsorted(forward_dt, side="right") - 1
        if end_pos <= i:
            continue
        v0 = float(prices.iloc[i])
        v1 = float(prices.iloc[end_pos])
        if v0 == 0.0 or not np.isfinite(v0) or not np.isfinite(v1):
            continue
        result.iloc[i] = (v1 / v0 - 1.0) * 100.0

    return result


def sma200w_distance_percentile(weekly_close: pd.Series) -> float:
    prices = pd.to_numeric(weekly_close, errors="coerce").dropna().sort_index()
    if prices.empty:
        return np.nan

    sma200w = prices.rolling(window=200, min_periods=200).mean()
    distance = (prices / sma200w - 1.0) * 100.0
    distance = distance.replace([np.inf, -np.inf], np.nan).dropna()
    if distance.empty:
        return np.nan

    return percentile_rank(distance, float(distance.iloc[-1]))


def historical_momentum_52w_metrics(
    close: pd.Series,
    analog_count: int = 75,
) -> tuple[float, float]:
    momentum_52w = calendar_performance_series(close, days=365)
    momentum_valid = momentum_52w.replace([np.inf, -np.inf], np.nan).dropna()
    if momentum_valid.empty:
        return np.nan, np.nan

    current_momentum = float(momentum_valid.iloc[-1])
    momentum_percentile = percentile_rank(momentum_valid, current_momentum)

    forward_6m = forward_calendar_return_series(close, days=182)
    analogs = pd.DataFrame(
        {
            "momentum": momentum_52w,
            "forward_6m": forward_6m,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if analogs.empty:
        return momentum_percentile, np.nan

    analogs["distance"] = (analogs["momentum"] - current_momentum).abs()
    nearest = analogs.nsmallest(min(analog_count, len(analogs)), "distance")
    return momentum_percentile, float(nearest["forward_6m"].mean())


def correction_risk_from_percentile_analogs(
    close: pd.Series,
    weekly_close: pd.Series,
    analog_count: int = 75,
    drawdown_threshold_pct: float = -7.0,
) -> tuple[float, float, float]:
    daily_prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    weekly_prices = pd.to_numeric(weekly_close, errors="coerce").dropna().sort_index()
    if daily_prices.empty or weekly_prices.empty:
        return np.nan, np.nan, np.nan

    sma200w = weekly_prices.rolling(window=200, min_periods=200).mean()
    sma200w_distance = (weekly_prices / sma200w - 1.0) * 100.0
    sma200w_percentile = expanding_percentile_rank(sma200w_distance)

    momentum_52w = calendar_performance_series(daily_prices, days=365)
    momentum_percentile_daily = expanding_percentile_rank(momentum_52w)
    momentum_percentile_weekly = momentum_percentile_daily.reindex(weekly_prices.index, method="ffill")

    current_sma_pct = float(sma200w_percentile.dropna().iloc[-1]) if not sma200w_percentile.dropna().empty else np.nan
    current_mom_pct = float(momentum_percentile_weekly.dropna().iloc[-1]) if not momentum_percentile_weekly.dropna().empty else np.nan
    if not np.isfinite(current_sma_pct) or not np.isfinite(current_mom_pct):
        return np.nan, np.nan, np.nan

    rows = []
    for i, dt in enumerate(weekly_prices.index):
        if i + 26 >= len(weekly_prices):
            continue

        sma_pct = float(sma200w_percentile.iloc[i])
        mom_pct = float(momentum_percentile_weekly.iloc[i])
        base_price = float(weekly_prices.iloc[i])
        if not np.isfinite(sma_pct) or not np.isfinite(mom_pct) or base_price == 0.0:
            continue

        next_13w = weekly_prices.iloc[i + 1 : i + 14]
        next_26w = weekly_prices.iloc[i + 1 : i + 27]
        if len(next_13w) < 13 or len(next_26w) < 26:
            continue

        drawdown_13w = (next_13w.min() / base_price - 1.0) * 100.0
        drawdown_26w = (next_26w.min() / base_price - 1.0) * 100.0
        rows.append(
            {
                "sma_pct": sma_pct,
                "mom_pct": mom_pct,
                "correction_3m": float(drawdown_13w <= drawdown_threshold_pct),
                "correction_6m": float(drawdown_26w <= drawdown_threshold_pct),
            }
        )

    analogs = pd.DataFrame(rows)
    if analogs.empty:
        return np.nan, np.nan, np.nan

    analogs["distance"] = np.sqrt(
        ((analogs["sma_pct"] - current_sma_pct) ** 2)
        + ((analogs["mom_pct"] - current_mom_pct) ** 2)
    )
    nearest = analogs.nsmallest(min(analog_count, len(analogs)), "distance")
    correction_3m = float(nearest["correction_3m"].mean() * 100.0)
    correction_6m = float(nearest["correction_6m"].mean() * 100.0)
    correction_risk = (0.7 * correction_3m) + (0.3 * correction_6m)
    return correction_risk, correction_3m, correction_6m

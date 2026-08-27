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

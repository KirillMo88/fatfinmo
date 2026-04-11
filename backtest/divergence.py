from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DivergenceConfig:
    pivot_window: int = 3
    lookback: int = 60


def _pivot_low(series: pd.Series, window: int) -> pd.Series:
    vals = series.values
    out = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        segment = vals[i - window : i + window + 1]
        center = vals[i]
        if np.isnan(center):
            continue
        if np.nanmin(segment) == center and np.sum(segment == center) == 1:
            out[i] = True
    return pd.Series(out, index=series.index)


def _pivot_high(series: pd.Series, window: int) -> pd.Series:
    vals = series.values
    out = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series) - window):
        segment = vals[i - window : i + window + 1]
        center = vals[i]
        if np.isnan(center):
            continue
        if np.nanmax(segment) == center and np.sum(segment == center) == 1:
            out[i] = True
    return pd.Series(out, index=series.index)


def _find_previous_pivot_idx(pivot_mask: pd.Series, current_idx: int, lookback: int) -> int | None:
    start = max(0, current_idx - lookback)
    candidates = np.where(pivot_mask.iloc[start:current_idx].values)[0]
    if len(candidates) == 0:
        return None
    return start + int(candidates[-1])


def bull_divergence_signal(
    price: pd.Series,
    indicator: pd.Series,
    config: DivergenceConfig,
) -> pd.Series:
    """
    Bull divergence at t:
    - price makes lower low than previous pivot low
    - indicator makes higher low than previous corresponding pivot low
    """
    if len(price) != len(indicator):
        raise ValueError("Price and indicator series must have the same length.")

    pl = _pivot_low(price, config.pivot_window)
    il = _pivot_low(indicator, config.pivot_window)
    signal = np.zeros(len(price), dtype=bool)

    for i in np.where(pl.values)[0]:
        prev_p = _find_previous_pivot_idx(pl, i, config.lookback)
        prev_i = _find_previous_pivot_idx(il, i, config.lookback)
        if prev_p is None or prev_i is None:
            continue
        if price.iloc[i] < price.iloc[prev_p] and indicator.iloc[i] > indicator.iloc[prev_i]:
            signal[i] = True

    return pd.Series(signal, index=price.index)


def bear_divergence_signal(
    price: pd.Series,
    indicator: pd.Series,
    config: DivergenceConfig,
) -> pd.Series:
    """
    Bear divergence at t:
    - price makes higher high than previous pivot high
    - indicator makes lower high than previous corresponding pivot high
    """
    if len(price) != len(indicator):
        raise ValueError("Price and indicator series must have the same length.")

    ph = _pivot_high(price, config.pivot_window)
    ih = _pivot_high(indicator, config.pivot_window)
    signal = np.zeros(len(price), dtype=bool)

    for i in np.where(ph.values)[0]:
        prev_p = _find_previous_pivot_idx(ph, i, config.lookback)
        prev_i = _find_previous_pivot_idx(ih, i, config.lookback)
        if prev_p is None or prev_i is None:
            continue
        if price.iloc[i] > price.iloc[prev_p] and indicator.iloc[i] < indicator.iloc[prev_i]:
            signal[i] = True

    return pd.Series(signal, index=price.index)

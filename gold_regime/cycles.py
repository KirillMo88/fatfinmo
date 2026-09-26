from __future__ import annotations

import numpy as np
import pandas as pd

from market_cycle import add_cycle_trough_metadata, cycle_direction


GOLD_SHORT_CYCLE_MIN_MONTHS = 60.0
GOLD_SHORT_CYCLE_MAX_MONTHS = 80.0
GOLD_LONG_CYCLE_MIN_MONTHS = 195.0
GOLD_LONG_CYCLE_MAX_MONTHS = 245.0


def build_gold_cycle_history(gold_close: pd.Series) -> pd.DataFrame:
    """Build monthly GOLD price and band-pass cycles using Market Cycle conventions."""
    close = pd.to_numeric(gold_close, errors="coerce").dropna()
    if close.empty:
        return pd.DataFrame()
    close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None)
    close = close[~close.index.isna()].sort_index()
    if close.empty:
        return pd.DataFrame()

    monthly = close.to_frame("GOLD_Close")
    monthly["_Month"] = monthly.index.to_period("M")
    monthly = monthly.groupby("_Month", sort=True).tail(1).copy()
    monthly["Date"] = monthly["_Month"].dt.to_timestamp("M")
    monthly = monthly.drop(columns=["_Month"]).reset_index(drop=True)
    monthly["GOLD_Log"] = np.log(pd.to_numeric(monthly["GOLD_Close"], errors="coerce"))

    monthly["GoldShortCycle"] = _standard_zscore(
        _fft_bandpass_cycle(
            monthly["GOLD_Log"],
            GOLD_SHORT_CYCLE_MIN_MONTHS,
            GOLD_SHORT_CYCLE_MAX_MONTHS,
        )
    )
    add_cycle_trough_metadata(
        monthly,
        cycle_col="GoldShortCycle",
        prefix="GoldShort",
        min_spacing_months=48.0,
        window_months=6,
        full_cycle_months=70.0,
    )
    monthly["GoldShortDirection"] = cycle_direction(monthly["GoldShortCycle"])

    monthly["GoldLongCycle"] = _standard_zscore(
        _fft_bandpass_cycle(
            monthly["GOLD_Log"],
            GOLD_LONG_CYCLE_MIN_MONTHS,
            GOLD_LONG_CYCLE_MAX_MONTHS,
        )
    )
    add_cycle_trough_metadata(
        monthly,
        cycle_col="GoldLongCycle",
        prefix="GoldLong",
        min_spacing_months=120.0,
        window_months=12,
        full_cycle_months=220.0,
    )
    monthly["GoldLongDirection"] = cycle_direction(monthly["GoldLongCycle"])
    return monthly


def _fft_bandpass_cycle(series: pd.Series, min_period: float, max_period: float) -> pd.Series:
    """Extract a cycle with zero-padded FFT resolution for the shorter GLD history."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < int(min_period):
        return pd.Series(np.nan, index=series.index, dtype="float64")

    segment = values.loc[valid.index[0] : valid.index[-1]].interpolate(limit_direction="both")
    n = len(segment)
    padded_n = 1
    while padded_n < n * 8:
        padded_n *= 2
    demeaned = segment.to_numpy(dtype="float64") - segment.mean()
    padded = np.pad(demeaned, (0, padded_n - n))
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(padded_n, d=1.0)
    mask = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    filtered = np.fft.irfft(spectrum * mask, n=padded_n)[:n]
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[segment.index] = filtered
    return out


def _standard_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < 24:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    std = valid.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (values - valid.mean()) / std

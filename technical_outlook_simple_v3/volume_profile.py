from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG


def build_volume_profile(frame: pd.DataFrame, *, timeframe: str) -> dict[str, Any]:
    """Build the deterministic displayed-window horizontal volume profile."""
    timeframe = str(timeframe).upper()
    configured = CONFIG["volume_profile"]
    requested = int(configured["weekly_lookback_bars" if timeframe == "WEEKLY" else "daily_lookback_bars"])
    values = frame.tail(requested).copy()
    bins = int(configured["profile_bins"])
    metadata: dict[str, Any] = {
        "timeframe": timeframe,
        "lookback_bars": int(len(values)),
        "requested_lookback_bars": requested,
        "reduced_history": bool(len(values) < requested),
        "profile_start": _date(values.iloc[0]["timestamp"]) if len(values) else None,
        "profile_end": _date(values.iloc[-1]["timestamp"]) if len(values) else None,
        "number_of_bins": bins,
        "smoothing_method": "none",
        "minimum_peak_separation_bins": 2,
        "local_peak_min_poc_fraction": 0.25,
        "max_local_peaks": 3,
        "methodology": "Uniform OHLC range allocation across equal-width price bins; not exchange trade-by-price VRVP.",
    }
    if values.empty:
        return {**metadata, "status": "NOT_AVAILABLE", "poc": None, "poc_zone": None, "local_peaks": [], "hvns": [], "bins": [], "range": None}

    lows = pd.to_numeric(values["low"], errors="coerce")
    highs = pd.to_numeric(values["high"], errors="coerce")
    closes = pd.to_numeric(values["close"], errors="coerce")
    # Some index/commodity feeds do not publish a volume column.  Keep the
    # profile deterministic in that case by creating an all-missing series;
    # callers receive NOT_AVAILABLE instead of a scalar/attribute error.
    raw_volume = values["volume"] if "volume" in values.columns else pd.Series(np.nan, index=values.index)
    volumes = pd.to_numeric(raw_volume, errors="coerce")
    valid = lows.notna() & highs.notna() & closes.notna() & volumes.notna() & volumes.gt(0)
    if not bool(valid.any()):
        return {**metadata, "status": "NOT_AVAILABLE", "poc": None, "poc_zone": None, "local_peaks": [], "hvns": [], "bins": [], "range": None}

    low_value = float(lows.loc[valid].min())
    high_value = float(highs.loc[valid].max())
    if high_value <= low_value:
        high_value = low_value + max(abs(low_value) * 1e-6, 1e-6)
    edges = np.linspace(low_value, high_value, bins + 1)
    accumulated = np.zeros(bins, dtype=float)
    for low, high, close, volume in zip(lows.loc[valid], highs.loc[valid], closes.loc[valid], volumes.loc[valid]):
        bar_low, bar_high, bar_volume = float(low), float(high), float(volume)
        if bar_high < bar_low:
            bar_low, bar_high = bar_high, bar_low
        intersected = np.where((edges[:-1] <= bar_high) & (edges[1:] >= bar_low))[0]
        if len(intersected) == 0:
            intersected = np.array([int(np.clip(np.searchsorted(edges, float(close), side="right") - 1, 0, bins - 1))])
        accumulated[intersected] += bar_volume / float(len(intersected))

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_index = int(np.argmax(accumulated))
    poc_volume = float(accumulated[poc_index])
    peaks: list[dict[str, Any]] = []
    threshold = poc_volume * float(configured["local_peak_min_poc_fraction"])
    for index in range(1, bins - 1):
        value = float(accumulated[index])
        if value <= 0 or value < threshold:
            continue
        if value > float(accumulated[index - 1]) and value >= float(accumulated[index + 1]) and index != poc_index:
            peaks.append({
                "bin_index": index,
                "center": float(centers[index]),
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "volume": value,
                "relative_volume": value / poc_volume if poc_volume else 0.0,
            })
    retained: list[dict[str, Any]] = []
    for peak in sorted(peaks, key=lambda item: (-float(item["volume"]), int(item["bin_index"]))):
        if any(abs(int(peak["bin_index"]) - int(other["bin_index"])) < 2 for other in retained):
            continue
        retained.append(peak)
        if len(retained) >= int(configured["max_local_peaks"]):
            break
    retained.sort(key=lambda item: item["center"])
    bin_rows = [
        {"index": index, "low": float(edges[index]), "high": float(edges[index + 1]), "center": float(centers[index]), "volume": float(accumulated[index])}
        for index in range(bins)
    ]
    return {
        **metadata,
        "status": "AVAILABLE",
        "poc": float(centers[poc_index]),
        "poc_volume": poc_volume,
        "poc_zone": {"low": float(edges[poc_index]), "high": float(edges[poc_index + 1]), "center": float(centers[poc_index])},
        "local_peaks": retained,
        "hvns": retained,
        "bins": bin_rows,
        "range": {"low": low_value, "high": high_value},
    }


def _date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed.isoformat() if pd.notna(parsed) else None

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG


def build_volume_profile(frame: pd.DataFrame, *, timeframe: str) -> dict[str, Any]:
    cfg = CONFIG["volume_profile"]
    weekly = str(timeframe).upper() == "WEEKLY"
    lookback = int(cfg["weekly_lookback_bars"] if weekly else cfg["daily_lookback_bars"])
    values = frame.tail(lookback).copy()
    volume = pd.to_numeric(values.get("volume"), errors="coerce")
    price = (
        pd.to_numeric(values["high"], errors="coerce")
        + pd.to_numeric(values["low"], errors="coerce")
        + pd.to_numeric(values["close"], errors="coerce")
    ) / 3.0
    valid = price.notna() & volume.notna() & volume.gt(0)
    metadata = {
        "timeframe": "WEEKLY" if weekly else "DAILY",
        "lookback_bars": int(len(values)),
        "requested_lookback_bars": lookback,
        "reduced_history": bool(len(values) < lookback),
        "profile_start": _date(values.iloc[0]["timestamp"]) if len(values) else None,
        "profile_end": _date(values.iloc[-1]["timestamp"]) if len(values) else None,
        "number_of_bins": int(cfg["profile_bins"]),
        "smoothing_method": cfg["smoothing_method"],
        "smoothing_parameters": {"gaussian_sigma": float(cfg["gaussian_sigma"])},
        "minimum_peak_separation_bins": int(cfg["minimum_peak_separation_bins"]),
        "hvn_min_prominence_poc_fraction": float(cfg["hvn_min_prominence_poc_fraction"]),
        "hvn_min_height_percentile": float(cfg["hvn_min_height_percentile"]),
        "max_hvn": int(cfg["max_hvn"]),
        "methodology": "Estimated horizontal distribution from OHLCV typical-price bars; not exchange trade-by-price VRVP.",
    }
    if int(valid.sum()) < 20:
        return {**metadata, "status": "NOT_AVAILABLE", "poc": None, "hvns": [], "value_area": None, "range": None}

    histogram, edges = np.histogram(
        price.loc[valid].to_numpy(dtype=float),
        bins=int(cfg["profile_bins"]),
        weights=volume.loc[valid].to_numpy(dtype=float),
    )
    smoothed = _gaussian_smooth(histogram.astype(float), float(cfg["gaussian_sigma"]))
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_index = int(np.argmax(smoothed))
    poc_volume = float(smoothed[poc_index])
    positive = smoothed[smoothed > 0]
    height_floor = float(np.quantile(positive, float(cfg["hvn_min_height_percentile"]))) if len(positive) else np.inf
    prominence_floor = poc_volume * float(cfg["hvn_min_prominence_poc_fraction"])
    separation = int(cfg["minimum_peak_separation_bins"])

    peaks: list[dict[str, Any]] = []
    for index in range(1, len(smoothed) - 1):
        height = float(smoothed[index])
        if height < height_floor or not (height >= smoothed[index - 1] and height > smoothed[index + 1]):
            continue
        left = max(0, index - max(2, separation * 2))
        right = min(len(smoothed), index + max(2, separation * 2) + 1)
        left_min = float(np.min(smoothed[left:index + 1]))
        right_min = float(np.min(smoothed[index:right]))
        prominence = height - max(left_min, right_min)
        if prominence + 1e-12 < prominence_floor:
            continue
        node_left = index
        node_right = index
        node_floor = max(height_floor, height - prominence)
        while node_left > 0 and smoothed[node_left - 1] >= node_floor:
            node_left -= 1
        while node_right < len(smoothed) - 1 and smoothed[node_right + 1] >= node_floor:
            node_right += 1
        peaks.append({
            "peak_index": index,
            "center": float(centers[index]),
            "low": float(edges[node_left]),
            "high": float(edges[node_right + 1]),
            "relative_prominence": float(prominence / poc_volume) if poc_volume else 0.0,
            "relative_height": float(height / poc_volume) if poc_volume else 0.0,
            "contains_poc": bool(node_left <= poc_index <= node_right),
        })

    retained: list[dict[str, Any]] = []
    for peak in sorted(peaks, key=lambda item: (-item["relative_prominence"], -item["relative_height"], item["center"])):
        if any(abs(int(peak["peak_index"]) - int(other["peak_index"])) < separation for other in retained):
            continue
        retained.append(peak)
        if len(retained) >= int(cfg["max_hvn"]):
            break
    retained.sort(key=lambda item: item["center"])
    for item in retained:
        item.pop("peak_index", None)

    order = list(np.argsort(smoothed)[::-1])
    selected: list[int] = []
    running = 0.0
    total = float(smoothed.sum())
    for index in order:
        selected.append(int(index))
        running += float(smoothed[index])
        if total > 0 and running / total >= float(cfg["value_area"]):
            break
    return {
        **metadata,
        "status": "AVAILABLE",
        "poc": float(centers[poc_index]),
        "poc_zone": {"low": float(edges[poc_index]), "high": float(edges[poc_index + 1]), "center": float(centers[poc_index])},
        "hvns": retained,
        "value_area": {"low": float(edges[min(selected)]), "high": float(edges[max(selected) + 1])},
        "range": {"low": float(edges[0]), "high": float(edges[-1])},
    }


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(float)
    radius = max(1, int(round(sigma * 3)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed.isoformat() if pd.notna(parsed) else None

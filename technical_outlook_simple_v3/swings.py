from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_causal_swings(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    atr_multiplier: float,
    min_reversal_pct: float,
) -> list[dict[str, Any]]:
    """Causal ATR/percentage ZigZag.

    The extreme receives its market timestamp, but cannot be used as confirmed
    evidence until the later confirmation bar has crossed the reversal threshold.
    """
    if frame.empty:
        return []
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(frame.get("atr14"), errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).tolist()
    timeframe = str(timeframe).upper()
    source = "weekly_swing" if timeframe == "WEEKLY" else "daily_swing"

    def threshold(index: int, price: float) -> float:
        atr = atrs[index] if 0 <= index < len(atrs) and np.isfinite(atrs[index]) else abs(price) * min_reversal_pct
        return max(float(atr) * float(atr_multiplier), abs(float(price)) * float(min_reversal_pct))

    def make_pivot(kind: str, index: int, status: str, confirmation_index: int | None, reversal: float | None) -> dict[str, Any]:
        price = float(highs[index] if kind == "HIGH" else lows[index])
        atr = float(atrs[index]) if np.isfinite(atrs[index]) else None
        reversal_pct = None if reversal is None or price == 0 else float(reversal) / abs(price) * 100.0
        confirmation_time = pd.Timestamp(timestamps[confirmation_index]).isoformat() if confirmation_index is not None else None
        return {
            "pivot_id": f"{timeframe}:{kind}:{pd.Timestamp(timestamps[index]).isoformat()}",
            "pivot_time": pd.Timestamp(timestamps[index]).isoformat(),
            "confirmation_time": confirmation_time,
            "confirmed_at": confirmation_time,
            "price": price,
            "kind": kind,
            "high_or_low": kind,
            "bar_index": int(index),
            "timeframe": timeframe,
            "degree": "STRUCTURAL" if timeframe == "WEEKLY" else "INTERMEDIATE",
            "source": source,
            "status": status,
            "reversal_magnitude": _finite(reversal),
            "reversal_magnitude_pct": _finite(reversal_pct),
            "ATR_at_pivot": atr,
            "atr_reference": atr,
            "threshold_used": threshold(index, price),
            "config_used": {
                "atr_multiplier": float(atr_multiplier),
                "min_reversal_pct": float(min_reversal_pct),
            },
        }

    pivots: list[dict[str, Any]] = []
    high_index = low_index = 0
    direction = 0
    for index in range(1, len(frame)):
        if direction == 0:
            if highs[index] >= highs[high_index]:
                high_index = index
            if lows[index] <= lows[low_index]:
                low_index = index
            if low_index < high_index and highs[high_index] - lows[low_index] >= threshold(low_index, lows[low_index]):
                pivots.append(make_pivot("LOW", low_index, "CONFIRMED", high_index, highs[high_index] - lows[low_index]))
                direction = 1
            elif high_index < low_index and highs[high_index] - lows[low_index] >= threshold(high_index, highs[high_index]):
                pivots.append(make_pivot("HIGH", high_index, "CONFIRMED", low_index, highs[high_index] - lows[low_index]))
                direction = -1
            continue
        if direction == 1:
            if highs[index] >= highs[high_index]:
                high_index = index
                # OHLC bars do not reveal whether the high preceded the low;
                # never confirm an extreme on the same bar that created it.
                continue
            reversal = highs[high_index] - lows[index]
            if reversal >= threshold(high_index, highs[high_index]):
                pivots.append(make_pivot("HIGH", high_index, "CONFIRMED", index, reversal))
                low_index = index
                direction = -1
        else:
            if lows[index] <= lows[low_index]:
                low_index = index
                continue
            reversal = highs[index] - lows[low_index]
            if reversal >= threshold(low_index, lows[low_index]):
                pivots.append(make_pivot("LOW", low_index, "CONFIRMED", index, reversal))
                high_index = index
                direction = 1

    potential_kind = "HIGH" if direction == 1 else "LOW" if direction == -1 else ("HIGH" if high_index > low_index else "LOW")
    potential_index = high_index if potential_kind == "HIGH" else low_index
    if not pivots or pivots[-1]["bar_index"] != potential_index:
        pivots.append(make_pivot(potential_kind, potential_index, "POTENTIAL", None, None))
    return pivots


def confirmed_as_of(pivots: list[dict[str, Any]], as_of: Any) -> list[dict[str, Any]]:
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    result = []
    for pivot in pivots:
        confirmation = pd.to_datetime(pivot.get("confirmation_time"), errors="coerce", utc=True)
        if pivot.get("status") == "CONFIRMED" and pd.notna(confirmation) and confirmation <= cutoff:
            result.append(pivot)
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

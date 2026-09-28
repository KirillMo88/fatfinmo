from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import Pivot


def wilder_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = pd.Series(np.nan, index=bars.index, dtype="float64")
    if len(true_range) < period:
        return atr
    first = true_range.iloc[:period].mean()
    atr.iloc[period - 1] = first
    for idx in range(period, len(true_range)):
        value = true_range.iloc[idx]
        previous = atr.iloc[idx - 1]
        if not np.isfinite(value) or not np.isfinite(previous):
            atr.iloc[idx] = np.nan
        else:
            atr.iloc[idx] = ((period - 1) * previous + value) / period
    return atr


@dataclass
class PivotStream:
    k: float
    branch: str
    pivots: list[Pivot]
    ambiguous_bar_ids: list[str]
    warmup_bars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "branch": self.branch,
            "pivots": [pivot.to_dict() for pivot in self.pivots],
            "ambiguous_bar_ids": self.ambiguous_bar_ids,
            "warmup_bars": self.warmup_bars,
        }


def build_causal_pivot_streams(
    bars: pd.DataFrame,
    *,
    asset_id: str,
    source_timeframe: str,
    atr_period: int = 14,
    multipliers: list[float] | tuple[float, ...] = (1.5, 3.0, 6.0),
    max_pivots: int = 500,
) -> list[PivotStream]:
    if bars.empty:
        return []
    values = bars.reset_index(drop=True).copy()
    values["atr"] = wilder_atr(values, atr_period)
    streams: list[PivotStream] = []
    for k in multipliers:
        candidates = [
            _run_branch(values, asset_id, source_timeframe, float(k), "HIGH", atr_period),
            _run_branch(values, asset_id, source_timeframe, float(k), "LOW", atr_period),
        ]
        seen: set[tuple] = set()
        for stream in candidates:
            signature = tuple(
                (pivot.kind, pivot.pivot_time, round(pivot.price, 10), pivot.confirmed_at)
                for pivot in stream.pivots
                if pivot.status == "PIVOT_CONFIRMED"
            )
            if signature in seen:
                continue
            seen.add(signature)
            if len(stream.pivots) > max_pivots:
                stream.pivots = stream.pivots[-max_pivots:]
            streams.append(stream)
    return streams


def _run_branch(
    bars: pd.DataFrame,
    asset_id: str,
    source_timeframe: str,
    k: float,
    initial_kind: str,
    atr_period: int,
) -> PivotStream:
    start_idx = next(
        (idx for idx in range(1, len(bars)) if np.isfinite(bars.loc[idx - 1, "atr"])),
        len(bars),
    )
    if start_idx >= len(bars):
        return PivotStream(k, f"START_{initial_kind}", [], [], len(bars))

    seek_kind = initial_kind
    candidate_idx = start_idx
    candidate_price = float(bars.loc[start_idx, "high" if seek_kind == "HIGH" else "low"])
    pivots: list[Pivot] = []
    ambiguous: list[str] = []
    confirmation_block_idx = -1

    for idx in range(start_idx, len(bars)):
        row = bars.loc[idx]
        old_candidate_price = candidate_price
        old_candidate_idx = candidate_idx
        extreme = float(row["high" if seek_kind == "HIGH" else "low"])
        improved = extreme > candidate_price if seek_kind == "HIGH" else extreme < candidate_price
        old_threshold = _threshold_at(bars, old_candidate_idx, k)
        crossed_old = False
        if old_threshold is not None and idx > old_candidate_idx:
            crossed_old = (
                float(row["low"]) <= old_candidate_price - old_threshold
                if seek_kind == "HIGH"
                else float(row["high"]) >= old_candidate_price + old_threshold
            )
        if improved:
            candidate_idx = idx
            candidate_price = extreme
            if crossed_old:
                ambiguous.append(str(row["bar_id"]))

        threshold = _threshold_at(bars, candidate_idx, k)
        if threshold is None or threshold <= 0 or idx <= candidate_idx or idx <= confirmation_block_idx:
            continue
        confirmed = (
            float(row["close"]) <= candidate_price - threshold
            if seek_kind == "HIGH"
            else float(row["close"]) >= candidate_price + threshold
        )
        if not confirmed:
            continue

        pivots.append(
            _make_pivot(
                bars,
                asset_id,
                source_timeframe,
                k,
                candidate_idx,
                seek_kind,
                candidate_price,
                confirmed_idx=idx,
                status="PIVOT_CONFIRMED",
            )
        )
        previous_extreme_idx = candidate_idx
        seek_kind = "LOW" if seek_kind == "HIGH" else "HIGH"
        candidate_idx, candidate_price = _opposite_candidate(
            bars,
            previous_extreme_idx + 1,
            idx,
            seek_kind,
        )
        confirmation_block_idx = idx

    if candidate_idx < len(bars):
        forming = _make_pivot(
            bars,
            asset_id,
            source_timeframe,
            k,
            candidate_idx,
            seek_kind,
            candidate_price,
            confirmed_idx=None,
            status="FORMING",
        )
        if not pivots or (pivots[-1].pivot_time, pivots[-1].kind) != (forming.pivot_time, forming.kind):
            pivots.append(forming)
    return PivotStream(k, f"START_{initial_kind}", pivots, sorted(set(ambiguous)), start_idx)


def _threshold_at(bars: pd.DataFrame, candidate_idx: int, k: float) -> float | None:
    reference_idx = candidate_idx - 1
    if reference_idx < 0:
        return None
    atr = bars.loc[reference_idx, "atr"]
    if not np.isfinite(atr) or float(atr) <= 0:
        return None
    return float(k * float(atr))


def _opposite_candidate(
    bars: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    kind: str,
) -> tuple[int, float]:
    if start_idx > end_idx:
        start_idx = end_idx
    column = "high" if kind == "HIGH" else "low"
    subset = pd.to_numeric(bars.loc[start_idx:end_idx, column], errors="coerce")
    if subset.empty:
        idx = end_idx
    elif kind == "HIGH":
        max_value = subset.max()
        idx = int(subset[subset.eq(max_value)].index[0])
    else:
        min_value = subset.min()
        idx = int(subset[subset.eq(min_value)].index[0])
    return idx, float(bars.loc[idx, column])


def _make_pivot(
    bars: pd.DataFrame,
    asset_id: str,
    source_timeframe: str,
    k: float,
    pivot_idx: int,
    kind: str,
    price: float,
    *,
    confirmed_idx: int | None,
    status: str,
) -> Pivot:
    pivot_time = pd.Timestamp(bars.loc[pivot_idx, "timestamp"]).isoformat()
    confirmed_at = (
        pd.Timestamp(bars.loc[confirmed_idx, "timestamp"]).isoformat()
        if confirmed_idx is not None
        else None
    )
    known_at = confirmed_at or pd.Timestamp(bars.iloc[-1]["timestamp"]).isoformat()
    raw_id = f"{asset_id}|{source_timeframe}|{k}|{kind}|{pivot_time}|{price:.10f}"
    reference = bars.loc[pivot_idx - 1, "atr"] if pivot_idx > 0 else np.nan
    evidence_ids = [str(bars.loc[pivot_idx, "bar_id"])]
    if confirmed_idx is not None:
        evidence_ids.append(str(bars.loc[confirmed_idx, "bar_id"]))
    return Pivot(
        pivot_id=hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20],
        pivot_time=pivot_time,
        confirmed_at=confirmed_at,
        known_at=known_at,
        price=float(price),
        kind=kind,
        status=status,
        source_timeframe=source_timeframe,
        k=float(k),
        atr_reference=float(reference) if np.isfinite(reference) else None,
        evidence_bar_ids=evidence_ids,
        bar_index=int(pivot_idx),
    )

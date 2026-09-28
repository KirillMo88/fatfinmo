from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import pandas as pd

from .models import WaveNode


def evaluate_lifecycle(nodes: Iterable[WaveNode], bars: pd.DataFrame) -> list[dict[str, Any]]:
    """Evaluate target and invalidation events without inventing intrabar order.

    The function deliberately uses OHLC containment only.  If a target and an
    invalidation can both occur in one bar, the result is AMBIGUOUS_BAR unless a
    lower-timeframe sequence is supplied by a future adapter.
    """
    if bars is None or bars.empty:
        return []
    values = bars.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce", utc=True)
    values = values.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        invalidation = node.invalidation or {}
        boundary = invalidation.get("level")
        action_direction = _action_direction(node)
        for target in node.targets:
            if target.get("status") == "EXCLUDED":
                continue
            issued_at = pd.to_datetime(target.get("issued_at"), errors="coerce", utc=True)
            if pd.isna(issued_at):
                continue
            low = float(target["price_low"])
            high = float(target["price_high"])
            before = values.loc[values["timestamp"] < issued_at]
            if _any_zone_intersection(before, low, high):
                target["status"] = "RETROSPECTIVE_LEVEL"
                target["status_known_at"] = str(target.get("issued_at"))
                event = _event(
                    "RETROSPECTIVE_LEVEL",
                    node,
                    target,
                    str(target.get("issued_at")),
                    str(target.get("issued_at")),
                    note="The price zone was traded before the target was issued; this is not a forecast hit.",
                )
                _append_once(events, seen, event)
                continue

            eligible = values.loc[values["timestamp"] >= issued_at]
            previous_close: float | None = None
            earlier = values.loc[values["timestamp"] < issued_at, "close"]
            if not earlier.empty:
                previous_close = float(earlier.iloc[-1])
            for _, row in eligible.iterrows():
                timestamp = pd.Timestamp(row["timestamp"]).isoformat()
                target_touch = float(row["low"]) <= high and float(row["high"]) >= low
                boundary_breach = _boundary_breached(row, boundary, action_direction)
                if target_touch and boundary_breach:
                    target["status"] = "AMBIGUOUS_BAR"
                    target["status_known_at"] = timestamp
                    event = _event(
                        "AMBIGUOUS_BAR",
                        node,
                        target,
                        timestamp,
                        timestamp,
                        boundary=boundary,
                        actual_extreme={"high": float(row["high"]), "low": float(row["low"])},
                        note="Target and invalidation are both possible inside the same OHLC bar; order is unknown.",
                    )
                    _append_once(events, seen, event)
                    break
                if _gap_passed(previous_close, float(row["open"]), low, high, action_direction):
                    target["status"] = "TARGET_PASSED_BY_GAP"
                    target["status_known_at"] = timestamp
                    event = _event(
                        "TARGET_PASSED_BY_GAP",
                        node,
                        target,
                        timestamp,
                        timestamp,
                        actual_extreme={"open": float(row["open"]), "previous_close": previous_close},
                        note="The opening gap crossed the full target zone; an actual trade inside the zone is not asserted.",
                    )
                    _append_once(events, seen, event)
                    break
                if target_touch:
                    target["status"] = "TARGET_ZONE_ENTERED"
                    target["status_known_at"] = timestamp
                    event = _event(
                        "TARGET_ZONE_ENTERED",
                        node,
                        target,
                        timestamp,
                        timestamp,
                        actual_extreme={"high": float(row["high"]), "low": float(row["low"])},
                    )
                    _append_once(events, seen, event)
                    break
                if boundary_breach:
                    target["status"] = "INVALIDATED"
                    target["status_known_at"] = timestamp
                    event = _event(
                        "RULE_BREACH",
                        node,
                        target,
                        timestamp,
                        timestamp,
                        boundary=boundary,
                        actual_extreme={"high": float(row["high"]), "low": float(row["low"])},
                        rule_id="ACTIVE_INVALIDATION_BOUNDARY",
                    )
                    _append_once(events, seen, event)
                    break
                previous_close = float(row["close"])
    return sorted(events, key=lambda item: (item["known_at"], item["event_id"]))


def _action_direction(node: WaveNode) -> int:
    if node.pattern_type in {
        "ZIGZAG",
        "FLAT_REGULAR",
        "FLAT_EXPANDED",
        "DOUBLE_ZIGZAG",
        "DOUBLE_THREE",
        "TRIANGLE_CONTRACTING",
        "TRIANGLE_BARRIER",
    }:
        return -int(node.direction)
    return int(node.direction)


def _any_zone_intersection(frame: pd.DataFrame, low: float, high: float) -> bool:
    if frame.empty:
        return False
    return bool(((pd.to_numeric(frame["low"], errors="coerce") <= high) & (pd.to_numeric(frame["high"], errors="coerce") >= low)).any())


def _boundary_breached(row: pd.Series, boundary: Any, action_direction: int) -> bool:
    if boundary is None:
        return False
    value = float(boundary)
    return float(row["low"]) <= value if action_direction > 0 else float(row["high"]) >= value


def _gap_passed(previous_close: float | None, opening: float, low: float, high: float, direction: int) -> bool:
    if previous_close is None:
        return False
    if direction > 0:
        return previous_close < low and opening > high
    return previous_close > high and opening < low


def _event(
    event_type: str,
    node: WaveNode,
    target: dict[str, Any],
    observed_at: str,
    known_at: str,
    **extra: Any,
) -> dict[str, Any]:
    identity = {
        "event_type": event_type,
        "node_id": node.node_id,
        "target_id": target.get("target_id"),
        "observed_at": observed_at,
    }
    raw = json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "event_id": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18],
        **identity,
        "known_at": known_at,
        "affected_scenarios": [],
        **extra,
    }


def _append_once(events: list[dict[str, Any]], seen: set[str], event: dict[str, Any]) -> None:
    event_id = str(event["event_id"])
    if event_id not in seen:
        seen.add(event_id)
        events.append(event)

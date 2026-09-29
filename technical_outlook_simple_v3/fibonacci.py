from __future__ import annotations

from typing import Any

from .config import CONFIG


def active_fibonacci_framework(pivots: list[dict[str, Any]], *, timeframe: str) -> dict[str, Any]:
    confirmed = [item for item in pivots if item.get("status") == "CONFIRMED"]
    potential = next((item for item in reversed(pivots) if item.get("status") == "POTENTIAL"), None)
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    status = "NOT_AVAILABLE"

    if potential is not None:
        start = next((item for item in reversed(confirmed) if item.get("kind") != potential.get("kind")), None)
        if start is not None and int(potential.get("bar_index", -1)) > int(start.get("bar_index", -1)):
            end = potential
            status = "DEVELOPING"
    if start is None or end is None:
        if len(confirmed) >= 2:
            candidate_end = confirmed[-1]
            candidate_start = next((item for item in reversed(confirmed[:-1]) if item.get("kind") != candidate_end.get("kind")), None)
            if candidate_start is not None:
                start, end, status = candidate_start, candidate_end, "CONFIRMED"

    if start is None or end is None:
        return {"status": "NOT_AVAILABLE", "timeframe": str(timeframe).upper(), "retracements": [], "extensions": []}

    start_price = float(start["price"])
    end_price = float(end["price"])
    span = abs(end_price - start_price)
    if span <= 0:
        return {"status": "NOT_AVAILABLE", "timeframe": str(timeframe).upper(), "retracements": [], "extensions": []}
    direction = "UP" if end_price > start_price else "DOWN"
    retracements = []
    multiplier = float(CONFIG["fibonacci"]["developing_quality_multiplier"]) if status == "DEVELOPING" else 1.0
    for ratio in CONFIG["fibonacci"]["retracements"]:
        ratio = float(ratio)
        price = end_price - span * ratio if direction == "UP" else end_price + span * ratio
        retracements.append({
            "ratio": ratio,
            "price": float(price),
            "source": f"fib_{ratio:.3f}",
            "family": "FIBONACCI",
            "quality_multiplier": multiplier,
            "framework_status": status,
        })
    extensions = []
    for ratio in CONFIG["fibonacci"]["extensions"]:
        ratio = float(ratio)
        price = start_price + span * ratio if direction == "UP" else start_price - span * ratio
        extensions.append({"ratio": ratio, "price": float(price), "direction": direction})
    return {
        "status": status,
        "timeframe": str(timeframe).upper(),
        "direction": direction,
        "start_anchor": _anchor(start),
        "end_anchor": _anchor(end),
        "retracements": retracements,
        "extensions": extensions,
        "quality_multiplier": multiplier,
    }


def _anchor(pivot: dict[str, Any]) -> dict[str, Any]:
    return {
        "pivot_id": pivot.get("pivot_id"),
        "pivot_time": pivot.get("pivot_time"),
        "confirmation_time": pivot.get("confirmation_time"),
        "price": float(pivot["price"]),
        "kind": pivot.get("kind"),
        "status": pivot.get("status"),
    }

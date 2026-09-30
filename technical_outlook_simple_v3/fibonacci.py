from __future__ import annotations

from typing import Any

import pandas as pd

from .config import CONFIG


def active_fibonacci_framework(
    pivots: list[dict[str, Any]],
    *,
    timeframe: str,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return direction-aware strategic/tactical Fibonacci frameworks.

    When ``frame`` is supplied, all ATH and anchor decisions are restricted to
    that caller-provided analytical window.  The legacy no-frame path remains
    for compatibility with older unit fixtures only.
    """
    if frame is None:
        return _legacy_framework(pivots, timeframe=timeframe)
    timeframe = str(timeframe).upper()
    confirmed = [item for item in pivots if item.get("status") == "CONFIRMED" and item.get("kind") in {"LOW", "HIGH"}]
    if frame.empty or not confirmed:
        return _empty(timeframe)
    high_row = pd.to_numeric(frame["high"], errors="coerce")
    if high_row.dropna().empty:
        return _empty(timeframe)
    ath_index = int(high_row.idxmax())
    ath_price = float(high_row.loc[ath_index])
    ath_time = pd.Timestamp(frame.loc[ath_index, "timestamp"]).isoformat()
    ath = {"price": ath_price, "date": ath_time, "bar_index": ath_index}
    qualifying_before = [
        item for item in confirmed
        if int(item.get("bar_index", -1)) <= ath_index and item.get("kind") == "LOW"
    ]
    post_ath = [
        item for item in confirmed
        if int(item.get("bar_index", -1)) > ath_index and item.get("kind") == "LOW"
    ]
    post_ath_15 = [item for item in post_ath if _drawdown(ath_price, float(item["price"])) >= 0.15]
    direction = "BEARISH" if post_ath_15 else "BULLISH"
    if direction == "BEARISH":
        tactical = _latest_qualifying(post_ath_15, ath_price, 0.15)
        strategic = _latest_qualifying(post_ath, ath_price, 0.30)
    else:
        tactical = _nearest_backward(qualifying_before, ath_price, 0.15)
        strategic = _nearest_backward(qualifying_before, ath_price, 0.30)
    strategic_framework = _build_framework("STRATEGIC", strategic, ath, direction, timeframe)
    tactical_framework = _build_framework("TACTICAL", tactical, ath, direction, timeframe)
    tactical_suppressed = bool(strategic and tactical and _pivot_key(strategic) == _pivot_key(tactical))
    if tactical_suppressed:
        tactical_framework = None
    return {
        "status": "AVAILABLE" if strategic_framework or tactical_framework else "NOT_AVAILABLE",
        "timeframe": timeframe,
        "direction": direction,
        "ath": ath,
        "ath_price": ath_price,
        "ath_date": ath_time,
        "strategic": strategic_framework,
        "tactical": tactical_framework,
        "tactical_suppressed": tactical_suppressed,
        "strategic_anchor": _anchor(strategic) if strategic else None,
        "tactical_anchor": _anchor(tactical) if tactical and not tactical_suppressed else None,
        "retracements": _flatten_levels(strategic_framework, tactical_framework),
        "extensions": _extensions(ath, strategic or tactical, direction),
    }


def _build_framework(role: str, anchor: dict[str, Any] | None, ath: dict[str, Any], direction: str, timeframe: str) -> dict[str, Any] | None:
    if anchor is None:
        return None
    low = float(anchor["price"])
    span = abs(float(ath["price"]) - low)
    if span <= 0:
        return None
    levels: list[dict[str, Any]] = []
    for ratio in (0.382, 0.500, 0.618):
        price = float(ath["price"] - ratio * span) if direction == "BULLISH" else float(low + ratio * span)
        level_type = "0.382" if ratio == 0.382 else "0.500_0.618"
        levels.append({
            "ratio": ratio,
            "price": price,
            "low": price if ratio == 0.382 else None,
            "high": price if ratio == 0.382 else None,
            "role": role,
            "type": level_type,
            "family": "FIBONACCI",
            "source": f"{role.lower()}_fib_{level_type}",
            "timeframe": timeframe,
        })
    band_prices = [item["price"] for item in levels if item["type"] == "0.500_0.618"]
    band = {"low": min(band_prices), "high": max(band_prices)}
    for item in levels:
        if item["type"] == "0.500_0.618":
            item["low"], item["high"] = band["low"], band["high"]
    return {
        "role": role,
        "direction": direction,
        "timeframe": timeframe,
        "anchor": _anchor(anchor),
        "ath": ath,
        "levels": levels,
        "fib_0382": next(item for item in levels if item["type"] == "0.382"),
        "fib_0500_0618": {"low": band["low"], "high": band["high"], "type": "0.500_0.618", "role": role, "family": "FIBONACCI", "source": f"{role.lower()}_fib_0.500_0.618"},
    }


def _nearest_backward(items: list[dict[str, Any]], ath_price: float, threshold: float) -> dict[str, Any] | None:
    candidates = [item for item in items if _drawdown(ath_price, float(item["price"])) >= threshold]
    return max(candidates, key=lambda item: int(item.get("bar_index", -1)), default=None)


def _latest_qualifying(items: list[dict[str, Any]], ath_price: float, threshold: float) -> dict[str, Any] | None:
    candidates = [item for item in items if _drawdown(ath_price, float(item["price"])) >= threshold]
    return max(candidates, key=lambda item: int(item.get("bar_index", -1)), default=None)


def _drawdown(ath: float, low: float) -> float:
    return 1.0 - low / ath if ath else 0.0


def _pivot_key(item: dict[str, Any]) -> str:
    return str(item.get("pivot_id") or item.get("pivot_time") or item.get("bar_index"))


def _flatten_levels(*frameworks: dict[str, Any] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for framework in frameworks:
        if framework:
            output.extend(framework.get("levels") or [])
    return output


def _extensions(ath: dict[str, Any], anchor: dict[str, Any] | None, direction: str) -> list[dict[str, Any]]:
    if not anchor:
        return []
    span = abs(float(ath["price"]) - float(anchor["price"]))
    return [
        {"ratio": ratio, "price": float(ath["price"] + span * (ratio - 1.0)) if direction == "BULLISH" else float(anchor["price"] - span * (ratio - 1.0)), "direction": direction}
        for ratio in CONFIG["fibonacci"]["extensions"]
    ]


def _empty(timeframe: str) -> dict[str, Any]:
    return {"status": "NOT_AVAILABLE", "timeframe": timeframe, "direction": None, "ath": None, "strategic": None, "tactical": None, "tactical_suppressed": False, "strategic_anchor": None, "tactical_anchor": None, "retracements": [], "extensions": []}


def _anchor(pivot: dict[str, Any]) -> dict[str, Any]:
    return {
        "pivot_id": pivot.get("pivot_id"),
        "pivot_time": pivot.get("pivot_time"),
        "confirmation_time": pivot.get("confirmation_time"),
        "price": float(pivot["price"]),
        "kind": pivot.get("kind"),
        "status": pivot.get("status"),
    }


def _legacy_framework(pivots: list[dict[str, Any]], *, timeframe: str) -> dict[str, Any]:
    """Compatibility path for callers that have no analytical frame."""
    confirmed = [item for item in pivots if item.get("status") == "CONFIRMED"]
    potential = next((item for item in reversed(pivots) if item.get("status") == "POTENTIAL"), None)
    start = next((item for item in reversed(confirmed[:-1]) if confirmed and item.get("kind") != confirmed[-1].get("kind")), None) if confirmed else None
    end = confirmed[-1] if confirmed else None
    if potential is not None and start is not None:
        end, status = potential, "DEVELOPING"
    elif start is not None and end is not None:
        status = "CONFIRMED"
    else:
        return _empty(str(timeframe).upper())
    span = abs(float(end["price"]) - float(start["price"]))
    direction = "UP" if float(end["price"]) > float(start["price"]) else "DOWN"
    retracements = [{"ratio": ratio, "price": float(end["price"] - span * ratio if direction == "UP" else end["price"] + span * ratio), "source": f"fib_{ratio:.3f}", "family": "FIBONACCI", "quality_multiplier": 0.75 if status == "DEVELOPING" else 1.0} for ratio in (0.236, 0.382, 0.5, 0.618, 0.786)]
    return {"status": status, "timeframe": str(timeframe).upper(), "direction": direction, "start_anchor": _anchor(start), "end_anchor": _anchor(end), "retracements": retracements, "extensions": []}

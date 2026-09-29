"""Frozen Technical Outlook v0 Support/Resistance and scenario stack.

Do not refactor this module to share calculations with SR/Scenario V1.  Its
purpose is regression comparison and historical backtesting of the original
pre-SR-V1 behaviour.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG


def build_support_resistance_v0(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    timeframe: str = "DAILY",
    max_pivots: int | None = 10,
    max_zones: int = 10,
) -> list[dict[str, Any]]:
    row = frame.iloc[-1]
    current = float(row["close"])
    atr = _finite(row.get("atr14")) or current * 0.02
    weights = CONFIG["support_resistance"]["weights"]
    levels: list[dict[str, Any]] = []
    timeframe = str(timeframe).upper()
    selected_pivots = pivots if max_pivots is None else pivots[-max_pivots:]
    for pivot in selected_pivots:
        levels.append({"price": float(pivot["price"]), "source": "major_swing", "weight": weights["major_swing"], "timeframe": timeframe})
    for window in CONFIG["ma"]["windows"]:
        value = _finite(row.get(f"sma{window}"))
        if value is not None:
            levels.append({"price": value, "source": f"sma{window}", "weight": weights[f"sma{window}"], "timeframe": timeframe})
    if profile.get("status") == "AVAILABLE":
        levels.append({"price": float(profile["poc"]), "source": "poc", "weight": weights["poc"], "timeframe": timeframe})
        for value in profile.get("hvns", []):
            levels.append({"price": float(value), "source": "hvn", "weight": weights["hvn"], "timeframe": timeframe})
        for value in profile.get("lvns", []):
            levels.append({"price": float(value), "source": "lvn", "weight": weights["lvn"], "timeframe": timeframe})
    if len(pivots) >= 2:
        first, second = pivots[-2], pivots[-1]
        low, high = sorted([float(first["price"]), float(second["price"])])
        for ratio in (0.382, 0.5, 0.618, 1.0, 1.618):
            price = high - (high - low) * ratio if ratio <= 1 else high + (high - low) * (ratio - 1)
            levels.append({"price": price, "source": "fibonacci", "weight": weights["fibonacci"], "timeframe": timeframe})
    magnitude = 10 ** max(0, int(math.floor(math.log10(max(abs(current), 1.0)))) - 1)
    for multiple in range(-2, 3):
        rounded = round(current / magnitude + multiple) * magnitude
        levels.append({"price": rounded, "source": "round_number", "weight": weights["round_number"], "timeframe": timeframe})
    threshold = max(current * float(CONFIG["support_resistance"]["cluster_price_fraction"]), atr * float(CONFIG["support_resistance"]["cluster_atr_multiplier"]))
    clusters: list[list[dict[str, Any]]] = []
    for level in sorted(levels, key=lambda item: item["price"]):
        if not clusters or abs(level["price"] - np.mean([item["price"] for item in clusters[-1]])) > threshold:
            clusters.append([level])
        else:
            clusters[-1].append(level)
    zones = []
    for cluster in clusters:
        center = float(np.average([item["price"] for item in cluster], weights=[item["weight"] for item in cluster]))
        score = float(sum(item["weight"] for item in cluster))
        label = "VERY_HIGH" if score >= 8 else "HIGH" if score >= 5 else "MEDIUM" if score >= 2.5 else "LOW"
        zones.append({
            "low": float(min(item["price"] for item in cluster) - threshold * 0.20),
            "high": float(max(item["price"] for item in cluster) + threshold * 0.20),
            "center": center,
            "role": "SUPPORT" if center <= current else "RESISTANCE",
            "sources": sorted({item["source"] for item in cluster}),
            "timeframes": sorted({item["timeframe"] for item in cluster}),
            "confluence_score": round(score, 2),
            "confluence": label,
            "active": True,
            "broken": False,
        })
    nearest = sorted(zones, key=lambda zone: (abs(zone["center"] - current), -zone["confluence_score"]))
    if len(nearest) <= max_zones:
        return nearest
    nearest_count = max(1, max_zones // 2)
    selected = nearest[:nearest_count]
    selected_ids = {id(zone) for zone in selected}
    strongest = sorted(zones, key=lambda zone: (-zone["confluence_score"], abs(zone["center"] - current)))
    for zone in strongest:
        if id(zone) in selected_ids:
            continue
        selected.append(zone)
        selected_ids.add(id(zone))
        if len(selected) >= max_zones:
            break
    return sorted(selected, key=lambda zone: (abs(zone["center"] - current), -zone["confluence_score"]))


def build_scenarios_v0(
    current: float,
    zones: list[dict[str, Any]],
    probabilities: dict[str, int],
    momentum: dict[str, Any],
    *,
    timeframe_label: str = "Weekly",
) -> list[dict[str, Any]]:
    supports = sorted((zone for zone in zones if zone["role"] == "SUPPORT"), key=lambda zone: abs(zone["center"] - current))
    resistances = sorted((zone for zone in zones if zone["role"] == "RESISTANCE"), key=lambda zone: abs(zone["center"] - current))
    support1 = supports[0] if supports else _synthetic_zone(current * 0.92, "SUPPORT")
    support2 = supports[1] if len(supports) > 1 else _synthetic_zone(current * 0.82, "SUPPORT")
    resistance1 = resistances[0] if resistances else _synthetic_zone(current * 1.08, "RESISTANCE")
    resistance2 = resistances[1] if len(resistances) > 1 else _synthetic_zone(current * 1.18, "RESISTANCE")
    return [
        {
            "scenario": "BULLISH", "probability": probabilities["BULLISH"],
            "trigger": f"{timeframe_label} close > {resistance1['high']:.2f}",
            "confirmation": f"Momentum {momentum['classification']} with ROC acceleration or improving breadth",
            "expected_path": [current, resistance1["center"], resistance2["center"]],
            "target_zone": [resistance2["low"], resistance2["high"]],
            "invalidation": f"{timeframe_label} close < {support1['low']:.2f}",
        },
        {
            "scenario": "NEUTRAL", "probability": probabilities["NEUTRAL"],
            "trigger": f"Price remains between {support1['low']:.2f} and {resistance1['high']:.2f}",
            "confirmation": f"Momentum remains mixed and {timeframe_label.lower()} structure does not resolve",
            "expected_path": [current, support1["center"], resistance1["center"]],
            "target_zone": [support1["low"], resistance1["high"]],
            "invalidation": f"{timeframe_label} close outside {support1['low']:.2f}–{resistance1['high']:.2f}",
        },
        {
            "scenario": "BEARISH", "probability": probabilities["BEARISH"],
            "trigger": f"{timeframe_label} close < {support1['low']:.2f}",
            "confirmation": "Negative momentum expands and support fails on volume",
            "expected_path": [current, support1["center"], support2["center"]],
            "target_zone": [support2["low"], support2["high"]],
            "invalidation": f"{timeframe_label} close > {resistance1['high']:.2f}",
        },
    ]


def build_confirmation_matrix_v0(
    scenarios: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    current: float,
    *,
    timeframe_label: str = "Weekly",
) -> list[dict[str, str]]:
    bullish = next(item for item in scenarios if item["scenario"] == "BULLISH")
    bearish = next(item for item in scenarios if item["scenario"] == "BEARISH")
    supports = sorted((zone for zone in zones if zone["role"] == "SUPPORT"), key=lambda zone: abs(zone["center"] - current))
    matrix = [
        {"event": bullish["trigger"], "interpretation": "Bull scenario confirmed"},
        {"event": bearish["trigger"], "interpretation": "Bear scenario activated"},
    ]
    if len(supports) > 1:
        matrix.append({"event": f"{timeframe_label} close < {supports[1]['low']:.2f}", "interpretation": "Structural trend deterioration"})
    matrix.append({"event": bullish["invalidation"], "interpretation": "Bull scenario invalidated"})
    return matrix


def _synthetic_zone(center: float, role: str) -> dict[str, Any]:
    return {"low": center * 0.99, "high": center * 1.01, "center": center, "role": role}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

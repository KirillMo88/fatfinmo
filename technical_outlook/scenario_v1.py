from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, SCENARIO_ENGINE_VERSION


def apply_relevance(
    zones: list[dict[str, Any]],
    current: float,
    daily_frame: pd.DataFrame,
    *,
    is_crypto: bool,
) -> list[dict[str, Any]]:
    """Attach transparent scenario/horizon Relevance V1 breakdowns."""
    values = [dict(zone) for zone in zones]
    for zone in values:
        by_scenario: dict[str, dict[str, Any]] = {}
        by_horizon: dict[str, float] = {}
        for scenario in ("BULLISH", "BEARISH"):
            horizon_values: dict[str, Any] = {}
            for horizon in ("SHORT", "MEDIUM", "6M"):
                breakdown = relevance_breakdown(zone, current, daily_frame, scenario=scenario, horizon=horizon, is_crypto=is_crypto)
                horizon_values[horizon] = breakdown
                by_horizon[horizon] = max(by_horizon.get(horizon, -999.0), float(breakdown["relevance_score"]))
            by_scenario[scenario] = horizon_values
        zone["relevance_by_scenario"] = by_scenario
        zone["relevance_by_horizon"] = by_horizon
    return values


def relevance_breakdown(
    zone: dict[str, Any],
    current: float,
    daily_frame: pd.DataFrame,
    *,
    scenario: str,
    horizon: str,
    is_crypto: bool,
) -> dict[str, Any]:
    cfg = CONFIG["scenario_v1"]
    role = str(zone.get("role"))
    expected_role = "RESISTANCE" if scenario == "BULLISH" else "SUPPORT"
    direction_component = 1.0 if role == expected_role else 0.0
    if scenario == "BULLISH":
        directional_distance = max(float(zone["low"]) - current, 0.0)
    else:
        directional_distance = max(current - float(zone["high"]), 0.0)
    bars_key = "crypto" if is_crypto else "market"
    horizon_bars = int(cfg["horizon_bars"][bars_key][horizon])
    close = pd.to_numeric(daily_frame.get("close"), errors="coerce")
    sigma = float(np.log(close / close.shift(1)).tail(63).std()) if len(close) else 0.0
    atr = _finite(daily_frame.iloc[-1].get("atr14")) if not daily_frame.empty else None
    volatility_move = current * max(sigma, 0.0) * math.sqrt(horizon_bars)
    atr_move = (atr or current * 0.01) * math.sqrt(horizon_bars)
    expected_move = max(volatility_move, atr_move, current * 0.005)
    reachability_ratio = directional_distance / expected_move
    reachability_component = 2.0 if reachability_ratio <= 0.75 else 1.5 if reachability_ratio <= 1.25 else 0.75 if reachability_ratio <= 2.0 else 0.0
    distance_penalty = -min(2.0, 0.5 * reachability_ratio)
    confluence_component = float(cfg["component_weights"]["confluence"].get(str(zone.get("confluence_class") or zone.get("confluence") or "LOW"), 0.5))
    strength_component = float(cfg["component_weights"]["strength"].get(str(zone.get("strength_class") or "WEAK"), 0.0))
    timeframe = str(zone.get("timeframe") or "DAILY")
    timeframe_component = float(cfg["timeframe_component"][horizon].get(timeframe, 0.0))
    cross_component = float(cfg["cross_timeframe_component"]) if bool(zone.get("cross_timeframe")) else 0.0
    alignment_component = float(zone.get("elliott_fib_alignment") or 0.0)
    score = (
        confluence_component
        + strength_component
        + timeframe_component
        + cross_component
        + direction_component
        + reachability_component
        + alignment_component
        + distance_penalty
    )
    if direction_component == 0.0:
        score = -99.0
    return {
        "relevance_score": round(float(score), 4),
        "confluence_component": confluence_component,
        "strength_component": strength_component,
        "timeframe_component": timeframe_component,
        "cross_timeframe_component": cross_component,
        "direction_component": direction_component,
        "horizon_reachability_component": reachability_component,
        "elliott_fib_alignment_component": alignment_component,
        "distance_penalty": round(distance_penalty, 4),
        "directional_distance": round(directional_distance, 6),
        "directional_distance_pct": round(directional_distance / max(current, 1e-12), 6),
        "expected_move": round(expected_move, 6),
        "reachability_ratio": round(reachability_ratio, 6),
        "horizon": horizon,
        "scenario": scenario,
    }


def support_resistance_component_v1(zones: list[dict[str, Any]]) -> float:
    canonical = _canonical_zones(zones)
    pressure: dict[str, float] = {"SUPPORT": 0.0, "RESISTANCE": 0.0}
    for role, scenario in (("SUPPORT", "BEARISH"), ("RESISTANCE", "BULLISH")):
        candidates = []
        for zone in canonical:
            if zone.get("role") != role:
                continue
            relevance = (((zone.get("relevance_by_scenario") or {}).get(scenario) or {}).get("MEDIUM") or {})
            ratio = _finite(relevance.get("reachability_ratio"))
            if ratio is None or ratio > 2.0:
                continue
            confluence = float(CONFIG["scenario_v1"]["component_weights"]["confluence"].get(str(zone.get("confluence_class") or zone.get("confluence") or "LOW"), 0.5)) / 3.0
            strength = float(CONFIG["scenario_v1"]["component_weights"]["strength"].get(str(zone.get("strength_class") or "WEAK"), 0.0)) / 2.0
            evidence = (0.60 * confluence + 0.40 * strength) * math.exp(-ratio)
            candidates.append((evidence, str(zone.get("zone_id"))))
        pressure[role] = sum(value for value, _ in sorted(candidates, key=lambda item: (-item[0], item[1]))[:3])
    denominator = pressure["SUPPORT"] + pressure["RESISTANCE"]
    if denominator <= 0:
        return 0.0
    return round(float(np.clip(100.0 * (pressure["SUPPORT"] - pressure["RESISTANCE"]) / denominator, -100.0, 100.0)), 2)


def build_scenarios_v1(
    current: float,
    zones: list[dict[str, Any]],
    probabilities: dict[str, int],
    momentum: dict[str, Any],
    *,
    daily_atr: float,
    weekly_atr: float,
) -> list[dict[str, Any]]:
    canonical = _canonical_zones(zones)
    bullish = _directional_scenario("BULLISH", current, canonical, probabilities["BULLISH"], momentum, daily_atr, weekly_atr)
    bearish = _directional_scenario("BEARISH", current, canonical, probabilities["BEARISH"], momentum, daily_atr, weekly_atr)
    supports = _sorted_candidates(canonical, "BEARISH", "SHORT", current)
    resistances = _sorted_candidates(canonical, "BULLISH", "SHORT", current)
    support = supports[0] if supports else None
    resistance = resistances[0] if resistances else None
    neutral_low = float(support["low"]) if support else current * 0.95
    neutral_high = float(resistance["high"]) if resistance else current * 1.05
    neutral = {
        "scenario": "NEUTRAL",
        "probability": probabilities["NEUTRAL"],
        "trigger": f"Price remains between {neutral_low:.2f} and {neutral_high:.2f}",
        "trigger_zone_id": None,
        "trigger_basis": "Primary tactical support/resistance bracket",
        "confirmation": f"Momentum remains mixed ({momentum.get('classification')}) and neither directional trigger closes beyond its ATR buffer",
        "expected_path": [current, neutral_low, neutral_high],
        "primary_target": _target_object(support) if support else None,
        "extended_target": _target_object(resistance) if resistance else None,
        "structural_target": None,
        "target_zone": [neutral_low, neutral_high],
        "target_basis": "Range between nearest meaningful support and resistance",
        "invalidation": f"Completed close outside {neutral_low:.2f}–{neutral_high:.2f}",
        "invalidation_zone_id": None,
        "invalidation_basis": "Confirmed range resolution",
        "scenario_engine_version": SCENARIO_ENGINE_VERSION,
    }
    return [bullish, neutral, bearish]


def confirmation_matrix_v1(scenarios: list[dict[str, Any]]) -> list[dict[str, str]]:
    matrix: list[dict[str, str]] = []
    for scenario in scenarios:
        name = str(scenario.get("scenario"))
        matrix.append({"event": str(scenario.get("trigger")), "interpretation": f"{name.title()} scenario activated", "basis": str(scenario.get("trigger_basis") or "")})
        matrix.append({"event": str(scenario.get("invalidation")), "interpretation": f"{name.title()} scenario invalidated", "basis": str(scenario.get("invalidation_basis") or "")})
    return matrix


def _directional_scenario(
    scenario: str,
    current: float,
    zones: list[dict[str, Any]],
    probability: int,
    momentum: dict[str, Any],
    daily_atr: float,
    weekly_atr: float,
) -> dict[str, Any]:
    cfg = CONFIG["scenario_v1"]
    candidates = _sorted_candidates(zones, scenario, "6M", current)
    primary = _first_eligible(candidates, float(cfg["primary_min_relevance"]), float(cfg["primary_max_reachability"]))
    fallback = False
    if primary is None:
        primary = _first_eligible(candidates, -99.0, 2.0)
        fallback = primary is not None
    remaining = [zone for zone in candidates if primary is None or not _material_overlap(zone, primary)]
    extended = _first_eligible(remaining, float(cfg["extended_min_relevance"]), float(cfg["extended_max_reachability"]))
    remaining_structural = [zone for zone in remaining if extended is None or not _material_overlap(zone, extended)]
    structural = _first_eligible(
        [zone for zone in remaining_structural if zone.get("timeframe") in {"WEEKLY", "CROSS_TIMEFRAME"}],
        float(cfg["structural_min_relevance"]),
        float(cfg["structural_max_reachability"]),
    )
    trigger_candidates = _sorted_candidates(zones, scenario, "SHORT", current)
    trigger_zone = _first_eligible(trigger_candidates, float(cfg["primary_min_relevance"]), float(cfg["primary_max_reachability"])) or primary
    opposite = "BEARISH" if scenario == "BULLISH" else "BULLISH"
    invalidation_candidates = [zone for zone in _sorted_candidates(zones, opposite, "MEDIUM", current) if zone.get("timeframe") in {"WEEKLY", "CROSS_TIMEFRAME"} and (zone.get("confluence_class") in {"HIGH", "VERY_HIGH"} or zone.get("strength_class") in {"STRONG", "VERY_STRONG"})]
    invalidation = invalidation_candidates[0] if invalidation_candidates else (_sorted_candidates(zones, opposite, "MEDIUM", current) or [None])[0]
    direction = "above" if scenario == "BULLISH" else "below"
    boundary_name = "high" if scenario == "BULLISH" else "low"
    atr = weekly_atr if trigger_zone and trigger_zone.get("timeframe") in {"WEEKLY", "CROSS_TIMEFRAME"} else daily_atr
    trigger_value = float(trigger_zone[boundary_name]) + (1 if scenario == "BULLISH" else -1) * float(cfg["break_buffer_atr"]) * atr if trigger_zone else current
    trigger = f"Completed {'Weekly' if trigger_zone and trigger_zone.get('timeframe') in {'WEEKLY', 'CROSS_TIMEFRAME'} else 'Daily'} close {direction} {trigger_value:.2f}"
    if invalidation:
        inv_boundary = "low" if scenario == "BULLISH" else "high"
        inv_direction = "below" if scenario == "BULLISH" else "above"
        inv_value = float(invalidation[inv_boundary]) + (-1 if scenario == "BULLISH" else 1) * float(cfg["break_buffer_atr"]) * weekly_atr
        invalidation_text = f"Completed Weekly close {inv_direction} {inv_value:.2f}"
    else:
        invalidation_text = "No reliable structural invalidation zone available"
    targets = [zone for zone in (primary, extended, structural) if zone is not None]
    expected_path = [current] + [float(zone["center"]) for zone in targets]
    primary_object = _target_object(primary)
    target_zone = [float(primary["low"]), float(primary["high"])] if primary else [current, current]
    return {
        "scenario": scenario,
        "probability": probability,
        "trigger": trigger,
        "trigger_zone_id": trigger_zone.get("zone_id") if trigger_zone else None,
        "trigger_basis": _basis(trigger_zone, "Nearest meaningful directional decision zone") if trigger_zone else "Synthetic current-price fallback",
        "confirmation": f"Momentum is {momentum.get('classification')} and price confirms the buffered directional close",
        "expected_path": expected_path,
        "primary_target": primary_object,
        "extended_target": _target_object(extended),
        "structural_target": _target_object(structural),
        "target_zone": target_zone,
        "target_basis": _basis(primary, "First reachable high-Relevance destination") if primary else "No reliable target",
        "target_confidence": "LOW" if fallback else "NORMAL",
        "fallback_target": fallback,
        "invalidation": invalidation_text,
        "invalidation_zone_id": invalidation.get("zone_id") if invalidation else None,
        "invalidation_basis": _basis(invalidation, "Structural opposite-role zone") if invalidation else "Unavailable",
        "scenario_engine_version": SCENARIO_ENGINE_VERSION,
    }


def _sorted_candidates(zones: list[dict[str, Any]], scenario: str, horizon: str, current: float) -> list[dict[str, Any]]:
    expected_role = "RESISTANCE" if scenario == "BULLISH" else "SUPPORT"
    candidates = []
    for zone in zones:
        if zone.get("role") != expected_role:
            continue
        breakdown = (((zone.get("relevance_by_scenario") or {}).get(scenario) or {}).get(horizon) or {})
        item = dict(zone)
        item["_relevance"] = float(breakdown.get("relevance_score", -99.0))
        item["_reachability"] = float(breakdown.get("reachability_ratio", 999.0))
        item["_distance"] = float(breakdown.get("directional_distance", abs(float(zone["center"]) - current)))
        candidates.append(item)
    priority = {"CROSS_TIMEFRAME": 0, "WEEKLY": 1, "DAILY": 2}
    return sorted(candidates, key=lambda item: (item["_distance"], -item["_relevance"], priority.get(str(item.get("timeframe")), 3), -float(item.get("strength_score") or 0), -float(item.get("final_confluence_score") or 0), str(item.get("zone_id"))))


def _first_eligible(candidates: list[dict[str, Any]], minimum_relevance: float, maximum_reachability: float) -> dict[str, Any] | None:
    for zone in candidates:
        if float(zone.get("_relevance", -99.0)) >= minimum_relevance and float(zone.get("_reachability", 999.0)) <= maximum_reachability:
            return zone
    return None


def _canonical_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_ids = {
        parent
        for zone in zones if zone.get("cross_timeframe")
        for parent in list(zone.get("parent_daily_zone_ids") or []) + list(zone.get("parent_weekly_zone_ids") or [])
    }
    return [zone for zone in zones if zone.get("cross_timeframe") or zone.get("zone_id") not in parent_ids]


def _target_object(zone: dict[str, Any] | None) -> dict[str, Any] | None:
    if zone is None:
        return None
    return {
        "zone_id": zone.get("zone_id"),
        "range": [float(zone["low"]), float(zone["high"])],
        "center": float(zone["center"]),
        "timeframe": zone.get("timeframe"),
        "relevance_score": zone.get("_relevance"),
        "reachability_ratio": zone.get("_reachability"),
        "basis": _basis(zone, "Ranked Relevance V1 zone"),
    }


def _basis(zone: dict[str, Any] | None, prefix: str) -> str:
    if not zone:
        return prefix
    families = ", ".join(zone.get("source_families") or []) or "no family metadata"
    return f"{prefix}; {zone.get('timeframe')} {zone.get('role')}; {families}; Confluence {zone.get('confluence_class')}; Strength {zone.get('strength_class')}"


def _material_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    overlap = max(0.0, min(float(first["high"]), float(second["high"])) - max(float(first["low"]), float(second["low"])))
    denominator = max(min(float(first["high"]) - float(first["low"]), float(second["high"]) - float(second["low"])), 1e-12)
    return overlap / denominator >= 0.30


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CONFIG, SCENARIO_ENGINE_VERSION


def build_weekly_scenario_matrix(
    *,
    current_price: float,
    weekly_atr: float,
    weekly_bars_count: int,
    structure: dict[str, Any],
    momentum: dict[str, Any],
    ma_structure: dict[str, Any],
    volume_context: dict[str, Any],
    analogs: dict[str, Any],
    extension_pct: float | None,
    weekly_zones: list[dict[str, Any]],
    fibonacci: dict[str, Any],
) -> dict[str, Any]:
    eligible = [zone for zone in weekly_zones if zone.get("key_point_class", zone.get("confluence_class")) in {"HIGH", "MID"} and not zone.get("hidden_by_60pct_filter")]
    components = _scenario_components(
        current_price=current_price,
        structure=structure,
        momentum=momentum,
        ma_structure=ma_structure,
        volume_context=volume_context,
        analogs=analogs,
        extension_pct=extension_pct,
        zones=eligible,
    )
    probabilities = scenario_probabilities(components)
    testing = sorted((zone for zone in eligible if zone.get("role") == "TESTING"), key=lambda zone: (-_class_rank(zone), -float(zone.get("quality_score") or 0.0)))[0:1]
    testing_zone = testing[0] if testing else None
    supports = _ordered_zones([zone for zone in eligible if zone.get("role") == "SUPPORT"], current_price, "DOWN")
    resistances = _ordered_zones([zone for zone in eligible if zone.get("role") == "RESISTANCE"], current_price, "UP")

    bull_trigger_zone = testing_zone or (resistances[0] if resistances else None)
    bear_trigger_zone = testing_zone or (supports[0] if supports else None)
    bull_trigger_price = float(bull_trigger_zone["high"]) if bull_trigger_zone else None
    bear_trigger_price = float(bear_trigger_zone["low"]) if bear_trigger_zone else None
    bull_destinations = [zone for zone in resistances if bull_trigger_price is None or float(zone["center"]) > bull_trigger_price]
    bear_destinations = [zone for zone in supports if bear_trigger_price is None or float(zone["center"]) < bear_trigger_price]
    bull_targets = _target_hierarchy(
        bull_destinations, current_price=current_price, weekly_atr=weekly_atr, direction="UP",
        extensions=fibonacci.get("extensions") or [],
    )
    bear_targets = _target_hierarchy(
        bear_destinations, current_price=current_price, weekly_atr=weekly_atr, direction="DOWN",
        extensions=fibonacci.get("extensions") or [],
    )
    bull_invalidation = testing_zone or (supports[0] if supports else None)
    bear_invalidation = testing_zone or (resistances[0] if resistances else None)
    history_ok = weekly_bars_count >= int(CONFIG["scenario"]["minimum_history_weeks"])

    bullish = _scenario(
        name="BULLISH", probability=probabilities["BULLISH"],
        trigger=_trigger("above", bull_trigger_price), targets=bull_targets,
        invalidation=_trigger("below", float(bull_invalidation["low"])) if bull_invalidation else None,
        structure=structure, momentum=momentum, history_ok=history_ok,
        target_uses_extension=bool(bull_targets.get("primary") and bull_targets["primary"].get("source") == "FIBONACCI_EXTENSION"),
    )
    neutral_support = testing_zone or (supports[0] if supports else None)
    neutral_resistance = testing_zone or (resistances[0] if resistances else None)
    neutral_target = None
    if neutral_support and neutral_resistance:
        neutral_target = {
            "range": [float(neutral_support["low"]), float(neutral_resistance["high"])],
            "source": "WEEKLY_STRUCTURAL_RANGE",
        }
    neutral = _scenario(
        name="NEUTRAL", probability=probabilities["NEUTRAL"],
        trigger=(
            f"Weekly balance remains between {neutral_support['low']:.2f} and {neutral_resistance['high']:.2f}"
            if neutral_support and neutral_resistance else None
        ),
        targets={"primary": neutral_target, "extended": None, "structural": None},
        invalidation=(
            f"Confirmed Weekly close outside {neutral_support['low']:.2f}–{neutral_resistance['high']:.2f}"
            if neutral_support and neutral_resistance else None
        ),
        structure=structure, momentum=momentum, history_ok=history_ok,
        target_uses_extension=False,
    )
    bearish = _scenario(
        name="BEARISH", probability=probabilities["BEARISH"],
        trigger=_trigger("below", bear_trigger_price), targets=bear_targets,
        invalidation=_trigger("above", float(bear_invalidation["high"])) if bear_invalidation else None,
        structure=structure, momentum=momentum, history_ok=history_ok,
        target_uses_extension=bool(bear_targets.get("primary") and bear_targets["primary"].get("source") == "FIBONACCI_EXTENSION"),
    )
    scenarios = [bullish, neutral, bearish]
    dominant = max(probabilities, key=probabilities.get)
    dominant_scenario = next(item for item in scenarios if item["scenario"] == dominant)
    expected_path: list[Any] = [current_price]
    for key in ("primary_target", "extended_target", "structural_target"):
        target = dominant_scenario.get(key)
        if isinstance(target, dict) and isinstance(target.get("range"), list):
            expected_path.append(target["range"])
    return {
        "scenario_engine_version": SCENARIO_ENGINE_VERSION,
        "components": components,
        "probabilities": probabilities,
        "scenarios": scenarios,
        "expected_path": expected_path,
        "dominant_scenario": dominant,
        "eligible_weekly_zone_ids": [zone.get("zone_id") for zone in eligible],
    }


def scenario_probabilities(components: dict[str, Any]) -> dict[str, int]:
    """Frozen v3 mapping: weighted Weekly direction, then bounded maturity shift."""
    directional = components["directional_components"]
    available = {name: item for name, item in directional.items() if item.get("value") is not None}
    weight_sum = sum(float(item["configured_weight"]) for item in available.values())
    direction = 0.0
    if weight_sum > 0:
        direction = sum(float(item["value"]) * float(item["configured_weight"]) / weight_sum for item in available.values())
    uncertainty = 1.0 + max(0.0, 0.75 - weight_sum) * 1.5
    logits = np.array([direction / 32.0, uncertainty - abs(direction) / 75.0, -direction / 32.0], dtype=float)
    logits -= logits.max()
    raw = np.exp(logits)
    probabilities = raw / raw.sum() * 100.0
    base = _integer_percentages(probabilities)
    adjusted = _apply_extension_modifier(base, components.get("extension_risk") or {}, direction)
    components["direction_score"] = round(direction, 4)
    components["available_weight"] = round(weight_sum, 4)
    components["effective_weights"] = {
        name: round(float(item["configured_weight"]) / weight_sum, 6) if weight_sum else 0.0
        for name, item in available.items()
    }
    components["base_probabilities_before_extension"] = base
    return adjusted


def _scenario_components(
    *, current_price: float, structure: dict[str, Any], momentum: dict[str, Any],
    ma_structure: dict[str, Any], volume_context: dict[str, Any], analogs: dict[str, Any],
    extension_pct: float | None, zones: list[dict[str, Any]],
) -> dict[str, Any]:
    weights = CONFIG["scenario"]["weights"]
    structure_value = {"BULL": 100.0, "BEAR": -100.0, "RANGE": 0.0, "TRANSITION": 0.0}.get(str(structure.get("state")))
    momentum_value = _finite(momentum.get("score"))
    ma_value = {
        "STRONG_BULL": 100.0, "BULL": 65.0, "BULL_CORRECTION": 25.0,
        "TRANSITION": 0.0, "BEAR": -65.0, "STRONG_BEAR": -100.0,
    }.get(str(ma_structure.get("state")))
    volume_value = None if volume_context.get("status") != "AVAILABLE" else (_finite(volume_context.get("score")) or 50.0) * 2.0 - 100.0
    analog_median = (((analogs.get("returns") or {}).get("6M") or {}).get("median"))
    analog_value = None if analog_median is None else float(np.clip(float(analog_median) * 6.0, -100.0, 100.0))
    sr_value = _sr_context(zones, current_price)
    extension = _extension_risk(extension_pct)
    directional = {
        "market_structure": {"value": structure_value, "configured_weight": weights["market_structure"]},
        "momentum": {"value": momentum_value, "configured_weight": weights["momentum"]},
        "ma_structure": {"value": ma_value, "configured_weight": weights["ma_structure"]},
        "support_resistance": {"value": sr_value, "configured_weight": weights["support_resistance"]},
        "volume_context": {"value": volume_value, "configured_weight": weights["volume_context"]},
        "historical_analogs": {"value": analog_value, "configured_weight": weights["historical_analogs"]},
    }
    return {
        "configured_weights": dict(weights),
        "directional_components": directional,
        "extension_risk": {**extension, "configured_weight": weights["extension_risk"]},
    }


def _extension_risk(value: float | None) -> dict[str, Any]:
    cfg = CONFIG["scenario"]["extension_risk"]
    if value is None or not np.isfinite(float(value)):
        return {"available": False, "extension_pct": None, "risk_fraction": 0.0, "direction": "NONE"}
    value = float(value)
    normal = float(cfg["normal_threshold_pct"])
    extreme = float(cfg["extreme_threshold_pct"])
    risk = float(np.clip((abs(value) - normal) / max(extreme - normal, 1e-9), 0.0, 1.0))
    return {"available": True, "extension_pct": value, "risk_fraction": risk, "direction": "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "NONE"}


def _apply_extension_modifier(base: dict[str, int], risk: dict[str, Any], direction_score: float) -> dict[str, int]:
    result = dict(base)
    fraction = float(risk.get("risk_fraction") or 0.0)
    if fraction <= 0:
        return result
    cfg = CONFIG["scenario"]["extension_risk"]
    neutral_shift = min(int(round(float(cfg["max_neutral_shift_pp"]) * fraction)), result["BULLISH"] if risk.get("direction") == "POSITIVE" else result["BEARISH"])
    if risk.get("direction") == "POSITIVE":
        result["BULLISH"] -= neutral_shift
        result["NEUTRAL"] += neutral_shift
        if direction_score < 0:
            opposite = min(int(round(float(cfg["max_opposite_shift_pp"]) * fraction)), result["BULLISH"])
            result["BULLISH"] -= opposite
            result["BEARISH"] += opposite
    elif risk.get("direction") == "NEGATIVE":
        result["BEARISH"] -= neutral_shift
        result["NEUTRAL"] += neutral_shift
        if direction_score > 0:
            opposite = min(int(round(float(cfg["max_opposite_shift_pp"]) * fraction)), result["BEARISH"])
            result["BEARISH"] -= opposite
            result["BULLISH"] += opposite
    return result


def _scenario(
    *, name: str, probability: int, trigger: str | None, targets: dict[str, Any],
    invalidation: str | None, structure: dict[str, Any], momentum: dict[str, Any],
    history_ok: bool, target_uses_extension: bool,
) -> dict[str, Any]:
    primary = targets.get("primary")
    complete = bool(trigger and primary and invalidation)
    structure_state = str(structure.get("state"))
    momentum_score = _finite(momentum.get("score")) or 0.0
    agrees = (
        (name == "BULLISH" and structure_state == "BULL" and momentum_score >= 20)
        or (name == "BEARISH" and structure_state == "BEAR" and momentum_score <= -20)
        or (name == "NEUTRAL" and structure_state in {"RANGE", "TRANSITION"} and abs(momentum_score) < 20)
    )
    conflict = (
        (name == "BULLISH" and (structure_state == "BEAR" or momentum_score <= -20))
        or (name == "BEARISH" and (structure_state == "BULL" or momentum_score >= 20))
        or (name == "NEUTRAL" and abs(momentum_score) >= 60)
    )
    if not complete or not history_ok or structure_state == "TRANSITION":
        confidence = "LOW"
    elif agrees and not conflict and not target_uses_extension:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"
    no_target = "No qualifying Weekly target"
    return {
        "scenario": name,
        "probability": int(probability),
        "trigger": trigger or "No qualifying Weekly trigger",
        "confirmation": _confirmation_text(name),
        "primary_target": primary or {"label": no_target, "range": None, "source": None},
        "extended_target": targets.get("extended"),
        "structural_target": targets.get("structural"),
        "target_zone": primary.get("range") if isinstance(primary, dict) else None,
        "invalidation": invalidation or "No meaningful Weekly invalidation",
        "confidence": confidence,
        "expected_path": [target.get("range") for target in (primary, targets.get("extended"), targets.get("structural")) if isinstance(target, dict) and target.get("range")],
    }


def _target_hierarchy(
    zones: list[dict[str, Any]], *, current_price: float, weekly_atr: float,
    direction: str, extensions: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_limit = float(CONFIG["scenario"]["reachability"]["primary_atr"])
    ordered = _ordered_zones(zones, current_price, direction)
    reachable = [zone for zone in ordered if abs(float(zone["center"]) - current_price) / max(weekly_atr, 1e-9) <= primary_limit]
    targets: list[dict[str, Any]] = []
    primary: dict[str, Any] | None = None
    extended: dict[str, Any] | None = None
    structural: dict[str, Any] | None = None
    if reachable:
        first = reachable[0]
        targets.append(_zone_target(first))
        for zone in ordered:
            if zone.get("zone_id") != first.get("zone_id"):
                targets.append(_zone_target(zone))
        primary = targets[0]
        extended = targets[1] if len(targets) > 1 else None
        structural = targets[2] if len(targets) > 2 else None
    elif ordered:
        # A zone outside the Primary reachability band remains a valid
        # Extended/Structural destination, never an invented Primary target.
        targets = [_zone_target(zone) for zone in ordered]
        extended = targets[0] if targets else None
        structural = targets[1] if len(targets) > 1 else None
    else:
        # Deterministic fallback order is the configured Weekly extension
        # order: use 1.272 first, then 1.618 when each is directionally valid.
        configured_order = {float(ratio): index for index, ratio in enumerate(CONFIG["fibonacci"]["extensions"])}
        valid_extensions = sorted(
            (
                item for item in extensions
                if (direction == "UP" and float(item["price"]) > current_price)
                or (direction == "DOWN" and float(item["price"]) < current_price)
            ),
            key=lambda item: (_extension_order(item.get("ratio"), configured_order), abs(float(item["price"]) - current_price)),
        )
        targets.extend({
            "range": [float(item["price"]), float(item["price"])],
            "source": "FIBONACCI_EXTENSION",
            "ratio": item.get("ratio"),
        } for item in valid_extensions)
        primary = targets[0] if targets else None
        extended = targets[1] if len(targets) > 1 else None
        structural = targets[2] if len(targets) > 2 else None
    return {
        "primary": primary,
        "extended": extended,
        "structural": structural,
    }


def _extension_order(ratio: Any, configured_order: dict[float, int]) -> int:
    try:
        return configured_order.get(float(ratio), 999)
    except (TypeError, ValueError):
        return 999


def _ordered_zones(zones: list[dict[str, Any]], current: float, direction: str) -> list[dict[str, Any]]:
    valid = [zone for zone in zones if (direction == "UP" and float(zone["center"]) > current) or (direction == "DOWN" and float(zone["center"]) < current)]
    return sorted(valid, key=lambda zone: (-_class_rank(zone), abs(float(zone["center"]) - current), -float(zone.get("quality_score") or 0.0), -_strength_rank(zone)))


def _class_rank(zone: dict[str, Any]) -> int:
    return {"LOW": 1, "MID": 2, "MEDIUM": 2, "HIGH": 3, "VERY_HIGH": 4}.get(str(zone.get("key_point_class", zone.get("confluence_class"))), 0)


def _strength_rank(zone: dict[str, Any]) -> int:
    return {"WEAK": 1, "MODERATE": 2, "STRONG": 3, "VERY_STRONG": 4}.get(str(zone.get("strength_class")), 0)


def _zone_target(zone: dict[str, Any]) -> dict[str, Any]:
    return {
        "range": [float(zone["low"]), float(zone["high"])],
        "source": "WEEKLY_ZONE",
        "zone_id": zone.get("zone_id"),
        "confluence": zone.get("key_point_class", zone.get("confluence_class")),
    }


def _sr_context(zones: list[dict[str, Any]], current: float) -> float | None:
    if not zones:
        return None
    testing = [zone for zone in zones if zone.get("role") == "TESTING"]
    if testing:
        return 0.0
    supports = [zone for zone in zones if zone.get("role") == "SUPPORT"]
    resistances = [zone for zone in zones if zone.get("role") == "RESISTANCE"]
    support_distance = min((abs(float(zone["center"]) / current - 1.0) for zone in supports), default=None)
    resistance_distance = min((abs(float(zone["center"]) / current - 1.0) for zone in resistances), default=None)
    if support_distance is None and resistance_distance is None:
        return None
    if support_distance is None:
        return -35.0
    if resistance_distance is None:
        return 35.0
    return float(np.clip((resistance_distance - support_distance) * 500.0, -50.0, 50.0))


def _trigger(direction: str, price: float | None) -> str | None:
    return f"Confirmed Weekly close {direction} {price:.2f}" if price is not None else None


def _confirmation_text(name: str) -> str:
    if name == "BULLISH":
        return "Weekly structure remains/improves bullish with RSI, MACD and ROC confirmation"
    if name == "BEARISH":
        return "Weekly structure deteriorates with weakening RSI, MACD and ROC"
    return "Weekly structure and momentum remain balanced"


def _integer_percentages(values: np.ndarray) -> dict[str, int]:
    floors = np.floor(values).astype(int)
    for index in np.argsort(values - floors)[::-1][: 100 - int(floors.sum())]:
        floors[index] += 1
    return {"BULLISH": int(floors[0]), "NEUTRAL": int(floors[1]), "BEARISH": int(floors[2])}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

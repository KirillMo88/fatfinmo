from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd

from .config import CONFIG


LLM_SCHEMA_VERSION = "TECHNICAL_OUTLOOK_SIMPLE_V3_LLM_V2"

LLM_TEXT_FIELDS = (
    "summary", "market_structure_summary", "elliott_summary", "momentum_summary",
    "volume_profile_summary", "key_levels_summary", "scenario_summary",
    "forecast_horizons_summary", "confirmation_invalidation_summary",
    "historical_analogs_summary", "interpretation_difference",
)
LLM_FIELDS = LLM_TEXT_FIELDS + ("risk_factors", "key_confirmation_points", "elliott_structure")
ELLIOTT_CANDIDATE_FIELDS = (
    "label", "pattern", "direction", "current_wave", "wave_state", "completion_state",
    "targets", "invalidation", "rationale", "waves",
)
ELLIOTT_WAVE_FIELDS = ("pivot_time", "price", "wave_label", "wave_status")


def call_llm_interpretation(snapshot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from openai import OpenAI
    from cio_view import get_openai_api_key, parse_json_text

    model = os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
    response = OpenAI(api_key=get_openai_api_key()).responses.create(
        model=model,
        reasoning={"effort": "medium"},
        instructions=(
            "You are the optional interpretation layer for Technical Outlook SIMPLE v3. Quant output is immutable and authoritative. "
            "Never change or invent prices, zones, probability, confidence, triggers, targets, invalidations, Fibonacci anchors, or expected paths. "
            "Provide concise block summaries and an Elliott primary/alternative interpretation using only exact supplied pivot_time/price points. "
            "If evidence is insufficient, return UNRESOLVED with an empty waves array. "
            "Return the exact Technical Outlook JSON schema with these top-level fields: " + ", ".join(LLM_FIELDS) + ". "
            "risk_factors and key_confirmation_points are arrays of strings; all *_summary fields and interpretation_difference are strings. "
            "elliott_structure contains exactly primary, alternative, confidence, current_wave_state. "
            "Each primary/alternative object contains exactly: " + ", ".join(ELLIOTT_CANDIDATE_FIELDS) + ". "
            "waves is an array of objects containing exactly pivot_time, price, wave_label, wave_status; confidence is HIGH, MEDIUM, or LOW."
        ),
        input=serialize_llm_input(snapshot),
    )
    value = parse_json_text(getattr(response, "output_text", "") or "")
    return validate_llm_output(value, snapshot), model


def structured_llm_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    weekly_limit = int(CONFIG["llm"]["weekly_zone_limit"])
    daily_limit = int(CONFIG["llm"]["daily_zone_limit"])
    return {
        "schema_version": LLM_SCHEMA_VERSION,
        "ticker": snapshot.get("ticker"),
        "as_of": snapshot.get("as_of_timestamp"),
        "quant_output_is_immutable": True,
        "weekly": {
            "structure": snapshot.get("weekly_structure"),
            "momentum": snapshot.get("weekly_momentum_summary"),
            "moving_averages": snapshot.get("weekly_moving_averages"),
            "pivots": _pivots(snapshot.get("weekly_pivots"), 18),
            "zones": _zones(snapshot.get("weekly_zones"), weekly_limit),
            "volume_profile": _profile(snapshot.get("weekly_volume_profile_summary")),
            "fibonacci": snapshot.get("weekly_fibonacci_framework"),
        },
        "daily": {
            "structure": snapshot.get("daily_structure"),
            "momentum": snapshot.get("daily_momentum_summary"),
            "moving_averages": snapshot.get("daily_moving_averages"),
            "pivots": _pivots(snapshot.get("daily_pivots"), 24),
            "zones": _zones(snapshot.get("daily_zones"), daily_limit),
            "volume_profile": _profile(snapshot.get("daily_volume_profile_summary")),
            "fibonacci": snapshot.get("daily_fibonacci_framework"),
        },
        "weekly_scenario_matrix": snapshot.get("weekly_scenario_matrix"),
        "weekly_expected_path": snapshot.get("weekly_expected_path"),
        "scenario_probabilities": snapshot.get("scenario_probabilities"),
        "historical_analogs": _analogs(snapshot.get("historical_analogs")),
        "deterministic_summary": snapshot.get("deterministic_narrative"),
    }


def serialize_llm_input(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(structured_llm_input(snapshot), ensure_ascii=False, separators=(",", ":"))
    limit = int(CONFIG["llm"]["max_chars"])
    if len(encoded) > limit:
        raise ValueError(f"SIMPLE v3 LLM input is too large ({len(encoded):,} characters; limit {limit:,})")
    return encoded


def _pivots(value: Any, limit: int) -> list[dict[str, Any]]:
    fields = ("pivot_time", "confirmation_time", "price", "kind", "status", "timeframe", "degree")
    items = value if isinstance(value, list) else []
    return [{field: item.get(field) for field in fields} for item in items[-limit:] if isinstance(item, dict)]


def _zones(value: Any, limit: int) -> list[dict[str, Any]]:
    fields = (
        "zone_id", "low", "high", "center", "role", "key_point_class", "key_point_score",
        "pivot_score", "sma_score", "volume_score", "fibonacci_score", "strength_class",
        "source_families", "distance_pct",
    )
    items = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        item_class = str(item.get("key_point_class", item.get("confluence_class", "LOW")))
        if item_class not in {"HIGH", "MID"}:
            continue
        items.append(item)
    rank = {"MID": 1, "HIGH": 2}
    items.sort(key=lambda item: (
        -rank.get(str(item.get("key_point_class", item.get("confluence_class"))), 0),
        -float(item.get("key_point_score", item.get("quality_score") or 0.0) or 0.0),
        abs(float(item.get("distance_pct") or 0.0)),
    ))
    return [{field: item.get(field) for field in fields} for item in items[:limit]]


def _profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value.get(field) for field in ("status", "poc", "local_peaks", "lookback_bars", "methodology")}


def _analogs(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value.get(field) for field in ("sample_size", "warning", "returns", "drawdown_probabilities")}


def validate_llm_output(payload: dict[str, Any], snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the v3 commentary contract without importing the retired tab/model."""
    if not isinstance(payload, dict):
        raise ValueError("SIMPLE v3 LLM response must be an object")
    extra = set(payload).difference(LLM_FIELDS)
    missing = set(LLM_FIELDS).difference(payload)
    if extra or missing:
        raise ValueError(f"Invalid SIMPLE v3 LLM schema; missing={sorted(missing)}, extra={sorted(extra)}")
    result: dict[str, Any] = {}
    for field in LLM_TEXT_FIELDS:
        value = payload[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        result[field] = value.strip()
    for field in ("risk_factors", "key_confirmation_points"):
        value = payload[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be an array of strings")
        result[field] = [item.strip() for item in value]
    result["elliott_structure"] = _validate_elliott_structure(payload["elliott_structure"], snapshot)
    return result


def _validate_elliott_structure(value: Any, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    expected = {"primary", "alternative", "confidence", "current_wave_state"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("elliott_structure must contain exactly primary, alternative, confidence, current_wave_state")
    confidence = value["confidence"]
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("elliott_structure.confidence must be HIGH, MEDIUM, or LOW")
    if not isinstance(value["current_wave_state"], str):
        raise ValueError("elliott_structure.current_wave_state must be a string")
    return {
        "primary": _validate_elliott_candidate(value["primary"], snapshot),
        "alternative": _validate_elliott_candidate(value["alternative"], snapshot),
        "confidence": confidence,
        "current_wave_state": value["current_wave_state"].strip(),
    }


def _validate_elliott_candidate(value: Any, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    expected = set(ELLIOTT_CANDIDATE_FIELDS)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Elliott candidate must contain exactly {sorted(expected)}")
    result: dict[str, Any] = {}
    for field in ELLIOTT_CANDIDATE_FIELDS:
        if field == "waves":
            continue
        if not isinstance(value[field], str):
            raise ValueError(f"Elliott candidate {field} must be a string")
        result[field] = value[field].strip()
    if result["direction"] not in {"UP", "DOWN", "NEUTRAL"}:
        raise ValueError("Elliott candidate direction must be UP, DOWN, or NEUTRAL")
    if result["wave_state"] not in {"CONFIRMED", "DEVELOPING", "POTENTIAL", "UNRESOLVED"}:
        raise ValueError("Invalid Elliott candidate wave_state")
    waves = value["waves"]
    if not isinstance(waves, list) or len(waves) > 12:
        raise ValueError("Elliott candidate waves must be an array with at most 12 points")
    result["waves"] = [_validate_wave_point(point, snapshot) for point in waves]
    return result


def _validate_wave_point(value: Any, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(ELLIOTT_WAVE_FIELDS):
        raise ValueError(f"Elliott wave point must contain exactly {list(ELLIOTT_WAVE_FIELDS)}")
    if not all(isinstance(value[field], str) for field in ("pivot_time", "wave_label", "wave_status")):
        raise ValueError("Elliott wave point labels and timestamp must be strings")
    try:
        price = float(value["price"])
    except Exception as exc:
        raise ValueError("Elliott wave point price must be numeric") from exc
    if not math.isfinite(price):
        raise ValueError("Elliott wave point price must be finite")
    point = {
        "pivot_time": value["pivot_time"].strip(),
        "price": price,
        "wave_label": value["wave_label"].strip(),
        "wave_status": value["wave_status"].strip(),
    }
    if snapshot is not None and not _matches_supplied_pivot(point, snapshot):
        raise ValueError("Every Elliott wave point must exactly match a supplied weekly/daily pivot")
    return point


def _matches_supplied_pivot(point: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    target_time = pd.to_datetime(point["pivot_time"], errors="coerce", utc=True)
    if pd.isna(target_time):
        return False
    for key in ("weekly_pivots", "daily_pivots"):
        for pivot in snapshot.get(key) or []:
            pivot_time = pd.to_datetime(pivot.get("pivot_time"), errors="coerce", utc=True)
            try:
                pivot_price = float(pivot.get("price"))
            except Exception:
                continue
            if pd.notna(pivot_time) and pivot_time == target_time and math.isclose(pivot_price, point["price"], rel_tol=1e-6, abs_tol=1e-6):
                return True
    return False

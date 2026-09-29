from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd


LLM_SCHEMA_VERSION = "TECHNICAL_OUTLOOK_LLM_V2"

LLM_TEXT_FIELDS = (
    "summary",
    "market_structure_summary",
    "elliott_summary",
    "momentum_summary",
    "volume_profile_summary",
    "key_levels_summary",
    "scenario_summary",
    "forecast_horizons_summary",
    "confirmation_invalidation_summary",
    "historical_analogs_summary",
    "interpretation_difference",
)

LLM_FIELDS = LLM_TEXT_FIELDS + (
    "risk_factors",
    "key_confirmation_points",
    "elliott_structure",
)

ELLIOTT_CANDIDATE_FIELDS = (
    "label",
    "pattern",
    "direction",
    "current_wave",
    "wave_state",
    "completion_state",
    "targets",
    "invalidation",
    "rationale",
    "waves",
)

ELLIOTT_WAVE_FIELDS = ("pivot_time", "price", "wave_label", "wave_status")


def call_llm_interpretation(snapshot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from openai import OpenAI
    from cio_view import get_openai_api_key, parse_json_text

    model = os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
    client = OpenAI(api_key=get_openai_api_key())
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        instructions=(
            "You are the Elliott Structure authority for the Technical Outlook module. "
            "Analyze only the supplied WEEKLY and DAILY evidence. The quantitative engine remains authoritative for prices, indicators, "
            "levels, probabilities, returns, and drawdowns; never replace, recompute, or invent those values. "
            "You must independently assign the Elliott primary and alternative structures. Every Elliott wave point must copy an exact "
            "pivot_time and price from one of the supplied weekly/daily/minor pivots. If evidence is insufficient, use UNRESOLVED and an empty waves array. "
            "Provide a general summary and a separate concise summary for every analysis block. "
            "Return one JSON object with exactly these top-level fields: " + ", ".join(LLM_FIELDS) + ". "
            "risk_factors and key_confirmation_points are arrays of strings. All *_summary fields and interpretation_difference are strings. "
            "elliott_structure must contain exactly primary, alternative, confidence, current_wave_state. "
            "Each primary/alternative object must contain exactly: " + ", ".join(ELLIOTT_CANDIDATE_FIELDS) + ". "
            "All candidate values are strings except waves. waves is an array of objects containing exactly pivot_time, price, wave_label, wave_status. "
            "confidence is HIGH, MEDIUM, or LOW. direction is UP, DOWN, or NEUTRAL."
        ),
        input=json.dumps(structured_llm_input(snapshot), ensure_ascii=False, indent=2),
    )
    parsed = parse_json_text(getattr(response, "output_text", "") or "")
    return validate_llm_output(parsed, snapshot), model


def structured_llm_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LLM_SCHEMA_VERSION,
        "ticker": snapshot.get("ticker"),
        "analysis_horizon": "6M",
        "as_of": snapshot.get("as_of_timestamp"),
        "weekly": {
            "market_structure": snapshot.get("weekly_structure"),
            "moving_averages": snapshot.get("weekly_moving_averages"),
            "momentum": snapshot.get("weekly_indicators"),
            "pivots": snapshot.get("weekly_pivots"),
        },
        "daily": {
            "market_structure": snapshot.get("daily_structure"),
            "moving_averages": snapshot.get("daily_moving_averages"),
            "momentum": snapshot.get("daily_indicators"),
            "volume": snapshot.get("volume_state"),
            "pivots": snapshot.get("daily_pivots"),
            "minor_pivots": snapshot.get("minor_pivots"),
        },
        "support_resistance": snapshot.get("support_resistance"),
        "volume_profile": snapshot.get("volume_profile"),
        "divergences": snapshot.get("divergences"),
        "scenario_engine": {
            "components": snapshot.get("scenario_components"),
            "probabilities": snapshot.get("scenario_probabilities"),
            "scenarios": snapshot.get("scenarios"),
            "confirmation_matrix": snapshot.get("confirmation_matrix"),
        },
        "forecast_horizons": snapshot.get("horizons"),
        "historical_analogs": snapshot.get("historical_analogs"),
        "deterministic_summary": snapshot.get("deterministic_narrative"),
        "final_quant_state": snapshot.get("final_state"),
    }


def validate_llm_output(payload: dict[str, Any], snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Technical Outlook LLM response must be an object")
    extra = set(payload).difference(LLM_FIELDS)
    missing = set(LLM_FIELDS).difference(payload)
    if extra or missing:
        raise ValueError(f"Invalid Technical Outlook LLM schema; missing={sorted(missing)}, extra={sorted(extra)}")
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
    current_state = value["current_wave_state"]
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("elliott_structure.confidence must be HIGH, MEDIUM, or LOW")
    if not isinstance(current_state, str):
        raise ValueError("elliott_structure.current_wave_state must be a string")
    return {
        "primary": _validate_elliott_candidate(value["primary"], snapshot),
        "alternative": _validate_elliott_candidate(value["alternative"], snapshot),
        "confidence": confidence,
        "current_wave_state": current_state.strip(),
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
        raise ValueError("Every Elliott wave point must exactly match a supplied weekly/daily/minor pivot")
    return point


def _matches_supplied_pivot(point: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    target_time = pd.to_datetime(point["pivot_time"], errors="coerce", utc=True)
    if pd.isna(target_time):
        return False
    for key in ("weekly_pivots", "daily_pivots", "minor_pivots"):
        for pivot in snapshot.get(key) or []:
            pivot_time = pd.to_datetime(pivot.get("pivot_time"), errors="coerce", utc=True)
            try:
                pivot_price = float(pivot.get("price"))
            except Exception:
                continue
            if pd.notna(pivot_time) and pivot_time == target_time and math.isclose(pivot_price, point["price"], rel_tol=1e-6, abs_tol=1e-6):
                return True
    return False

from __future__ import annotations

import json
import os
from typing import Any

from technical_outlook.llm import ELLIOTT_CANDIDATE_FIELDS, LLM_FIELDS, validate_llm_output

from .config import CONFIG


LLM_SCHEMA_VERSION = "TECHNICAL_OUTLOOK_SIMPLE_V3_LLM_V1"


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
        "zone_id", "low", "high", "center", "role", "confluence_class", "quality_score",
        "strength_class", "source_families", "distance_pct",
    )
    items = [item for item in (value if isinstance(value, list) else []) if isinstance(item, dict) and item.get("confluence_class") in {"HIGH", "VERY_HIGH"}]
    rank = {"HIGH": 1, "VERY_HIGH": 2}
    items.sort(key=lambda item: (-rank.get(str(item.get("confluence_class")), 0), -float(item.get("quality_score") or 0.0), abs(float(item.get("distance_pct") or 0.0))))
    return [{field: item.get(field) for field in fields} for item in items[:limit]]


def _profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value.get(field) for field in ("status", "poc", "hvns", "value_area", "lookback_bars", "methodology")}


def _analogs(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value.get(field) for field in ("sample_size", "warning", "returns", "drawdown_probabilities")}

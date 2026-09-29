from __future__ import annotations

import json
import os
from typing import Any

LLM_FIELDS = (
    "summary",
    "primary_elliott_interpretation",
    "alternative_elliott_interpretation",
    "short_term_commentary",
    "medium_term_commentary",
    "six_month_commentary",
    "bull_case_explanation",
    "neutral_case_explanation",
    "bear_case_explanation",
    "risk_factors",
    "key_confirmation_points",
    "interpretation_difference",
)


def call_llm_interpretation(snapshot: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from openai import OpenAI
    from cio_view import get_openai_api_key, parse_json_text

    model = os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
    client = OpenAI(api_key=get_openai_api_key())
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        instructions=(
            "You interpret a deterministic Technical Outlook state. Use only supplied evidence. "
            "Never alter or invent prices, indicators, pivots, Elliott scores, levels, probabilities, returns, drawdowns, or confirmations. "
            "If your interpretation differs from Quant, explain it only in interpretation_difference. "
            "Return one JSON object with exactly these fields: " + ", ".join(LLM_FIELDS) + ". "
            "risk_factors and key_confirmation_points are arrays of strings; every other field is a string."
        ),
        input=json.dumps(structured_llm_input(snapshot), ensure_ascii=False, indent=2),
    )
    parsed = parse_json_text(getattr(response, "output_text", "") or "")
    return validate_llm_output(parsed), model


def structured_llm_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": snapshot.get("ticker"),
        "analysis_horizon": "6M",
        "as_of": snapshot.get("as_of_timestamp"),
        "monthly": {
            "market_structure": snapshot.get("monthly_structure"),
            "moving_averages": snapshot.get("monthly_moving_averages"),
            "momentum": snapshot.get("monthly_indicators"),
            "pivots": snapshot.get("monthly_pivots"),
            "elliott_parent_state": snapshot.get("elliott_major_primary"),
        },
        "weekly": {
            "market_structure": snapshot.get("weekly_structure"),
            "moving_averages": snapshot.get("weekly_moving_averages"),
            "momentum": snapshot.get("weekly_indicators"),
            "volume": snapshot.get("volume_state"),
            "pivots": snapshot.get("weekly_pivots"),
            "elliott_candidates": snapshot.get("elliott_candidate_universe"),
        },
        "elliott": {
            "primary": snapshot.get("elliott_primary"),
            "alternative": snapshot.get("elliott_alternative"),
            "confidence": snapshot.get("elliott_confidence"),
            "parent_child_consistency": snapshot.get("elliott_parent_child_map"),
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
        "historical_analogs": snapshot.get("historical_analogs"),
        "final_technical_state": snapshot.get("final_state"),
    }


def validate_llm_output(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Technical Outlook LLM response must be an object")
    extra = set(payload).difference(LLM_FIELDS)
    missing = set(LLM_FIELDS).difference(payload)
    if extra or missing:
        raise ValueError(f"Invalid Technical Outlook LLM schema; missing={sorted(missing)}, extra={sorted(extra)}")
    result: dict[str, Any] = {}
    for field in LLM_FIELDS:
        value = payload[field]
        if field in {"risk_factors", "key_confirmation_points"}:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field} must be an array of strings")
            result[field] = [item.strip() for item in value]
        else:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            result[field] = value.strip()
    return result

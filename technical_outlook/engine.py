from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from elliott_waves.config import AssetSpec

from .analytics import (
    build_scenarios,
    build_support_resistance,
    build_timeframe_bars,
    calculate_indicators,
    classify_structure,
    confirmation_matrix,
    data_version,
    detect_divergences,
    detect_pivots,
    deterministic_narrative,
    extension_state,
    final_confidence,
    finite,
    historical_analogs,
    momentum_state,
    moving_average_state,
    scenario_probabilities,
    snapshot_id,
    volume_profile,
    volume_state,
)
from .config import CONFIG, CONFIG_VERSION, MODEL_VERSION
from .elliott import analyze_elliott_hierarchy


class TechnicalOutlookEngine:
    def analyze(
        self,
        daily_bars: pd.DataFrame,
        spec: AssetSpec,
        *,
        previous: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        created = created_at or datetime.now(timezone.utc)
        daily = daily_bars.loc[daily_bars["is_closed"].fillna(False)].copy().reset_index(drop=True)
        if daily.empty:
            raise ValueError(f"{spec.display_name}: no closed daily bars")
        monthly = calculate_indicators(build_timeframe_bars(daily, "1M", spec))
        weekly = calculate_indicators(build_timeframe_bars(daily, "1W", spec))
        if monthly.empty or weekly.empty:
            raise ValueError(f"{spec.display_name}: insufficient aggregated history")

        major_pivots = detect_pivots(monthly, "MONTHLY", "MAJOR")
        intermediate_pivots = detect_pivots(weekly, "WEEKLY", "INTERMEDIATE")
        minor_pivots = detect_pivots(weekly, "WEEKLY", "MINOR")
        monthly_structure = classify_structure(monthly, major_pivots)
        weekly_structure = classify_structure(weekly, intermediate_pivots)
        monthly_ma = moving_average_state(monthly)
        weekly_ma = moving_average_state(weekly)
        monthly_momentum = momentum_state(monthly)
        weekly_momentum = momentum_state(weekly)
        divergences = detect_divergences(monthly, major_pivots, "MONTHLY") + detect_divergences(weekly, intermediate_pivots, "WEEKLY")
        volume = volume_state(weekly)
        profile = volume_profile(weekly)
        zones = build_support_resistance(weekly, intermediate_pivots, profile)
        elliott = analyze_elliott_hierarchy(
            major_pivots,
            intermediate_pivots,
            minor_pivots,
            monthly_momentum,
            weekly_momentum,
            divergences,
            volume,
            previous,
        )
        primary = elliott["primary"]
        alternative = elliott["alternative"]
        elliott_confidence = elliott["confidence"]
        analogs = historical_analogs(weekly)
        confidence = final_confidence(monthly_structure, weekly_structure, weekly_momentum, elliott_confidence, volume, analogs)
        price = float(weekly.iloc[-1]["close"])
        extension = weekly_momentum["extension200"]
        components = _scenario_components(
            monthly_structure,
            weekly_structure,
            monthly_momentum,
            weekly_momentum,
            primary,
            volume,
            divergences,
            zones,
            analogs,
            extension,
            price,
        )
        probabilities = scenario_probabilities(components, confidence)
        scenarios = build_scenarios(price, zones, probabilities, weekly_momentum)
        horizons = _horizon_states(monthly_structure, weekly_structure, weekly_momentum, probabilities, zones, price)
        multi_timeframe = _multi_timeframe_state(monthly_structure["state"], weekly_structure["state"])
        final_state = {
            "structural_trend": _direction(monthly_structure["state"]),
            "medium_term_trend": _direction(weekly_structure["state"]),
            "momentum": _compact_momentum(weekly_momentum["classification"]),
            "momentum_trajectory": weekly_momentum["trajectory"],
            "elliott_phase": primary["label"],
            "elliott_wave_state": elliott["current_wave_state"],
            "extension": extension_state(extension),
            "six_month_bias": max(probabilities, key=probabilities.get),
            "confidence": confidence,
            "multi_timeframe_regime": multi_timeframe,
        }
        as_of = pd.Timestamp(daily.iloc[-1]["timestamp"]).isoformat()
        created_iso = pd.Timestamp(created).isoformat()
        version = data_version(daily)
        payload: dict[str, Any] = {
            "snapshot_id": snapshot_id(spec.display_name, as_of, created_iso, version),
            "as_of_date": pd.Timestamp(as_of).date().isoformat(),
            "as_of_timestamp": as_of,
            "ticker": spec.display_name,
            "provider_symbol": spec.provider_symbol,
            "source_id": spec.source_id,
            "source_metadata": spec.to_dict(),
            "price": price,
            "monthly_structure": monthly_structure,
            "weekly_structure": weekly_structure,
            "monthly_moving_averages": monthly_ma,
            "weekly_moving_averages": weekly_ma,
            "monthly_indicators": monthly_momentum,
            "weekly_indicators": weekly_momentum,
            "pivots": {"monthly": major_pivots, "weekly": intermediate_pivots, "minor": minor_pivots},
            "monthly_pivots": major_pivots,
            "weekly_pivots": intermediate_pivots,
            "minor_pivots": minor_pivots,
            "divergences": divergences,
            "volume_state": volume,
            "volume_profile": profile,
            "support_resistance": zones,
            "elliott_primary": primary,
            "elliott_primary_score": primary["score"],
            "elliott_alternative": alternative,
            "elliott_alternative_score": alternative["score"],
            "elliott_confidence": elliott_confidence,
            "elliott_current_wave_state": elliott["current_wave_state"],
            "elliott_candidate_universe_summary": elliott["candidate_universe_summary"],
            "elliott_candidate_universe": elliott["candidate_universe"],
            "elliott_major_primary": elliott["major_primary"],
            "elliott_major_alternative": elliott["major_alternative"],
            "elliott_major_confidence": elliott["major_confidence"],
            "elliott_parent_child_map": elliott["parent_child_map"],
            "elliott_complexity_penalties": elliott["complexity_penalties"],
            "elliott_change_from_prior_snapshot": elliott["change_from_prior_snapshot"],
            "scenario_components": components,
            "scenario_probabilities": probabilities,
            "bull_probability": probabilities["BULLISH"],
            "neutral_probability": probabilities["NEUTRAL"],
            "bear_probability": probabilities["BEARISH"],
            "scenarios": scenarios,
            "short_term_bias": horizons["short_term"]["state"],
            "medium_term_bias": horizons["medium_term"]["state"],
            "six_month_bias": horizons["six_month"]["state"],
            "horizons": horizons,
            "expected_path": next(item["expected_path"] for item in scenarios if item["scenario"] == final_state["six_month_bias"]),
            "confirmation_matrix": confirmation_matrix(scenarios, zones, price),
            "historical_analogs": analogs,
            "final_state": final_state,
            "confidence": confidence,
            "extension": extension_state(extension),
            "quant_updated_at": created_iso,
            "llm_enabled": False,
            "llm_interpretation": None,
            "llm_model": None,
            "llm_updated_at": None,
            "llm_difference": None,
            "llm_status": "DISABLED",
            "deterministic_narrative": "",
            "model_version": MODEL_VERSION,
            "config_version": CONFIG_VERSION,
            "model_parameters": CONFIG,
            "data_version": version,
            "data_status": "CURRENT",
            "stale_reason": None,
            "backtest_outcomes": None,
        }
        payload["deterministic_narrative"] = deterministic_narrative(payload)
        charts = {"1M": _chart_frame(monthly), "1W": _chart_frame(weekly)}
        return payload, charts


def _chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timestamp", "open", "high", "low", "close", "volume", "sma50", "sma100", "sma200",
        "rsi14", "macd", "macd_signal", "macd_hist", "roc12", "extension200",
    ]
    return frame[[column for column in columns if column in frame]].copy()


def _scenario_components(
    monthly_structure: dict[str, Any],
    weekly_structure: dict[str, Any],
    monthly_momentum: dict[str, Any],
    weekly_momentum: dict[str, Any],
    primary: dict[str, Any],
    volume: dict[str, Any],
    divergences: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    analogs: dict[str, Any],
    extension: float | None,
    price: float,
) -> dict[str, float | None]:
    direction = {"BULL": 100.0, "BEAR": -100.0, "RANGE": 0.0, "TRANSITION": 0.0}
    trend = direction.get(monthly_structure["state"], 0.0) * 0.65 + direction.get(weekly_structure["state"], 0.0) * 0.35
    momentum = float(monthly_momentum["score"]) * 0.35 + float(weekly_momentum["score"]) * 0.65
    elliott_direction = 1.0 if primary.get("direction") == "UP" else -1.0 if primary.get("direction") == "DOWN" else 0.0
    elliott = elliott_direction * float(primary.get("score") or 0.0)
    volume_component = None if volume.get("status") != "AVAILABLE" else (float(volume.get("score") or 50.0) - 50.0) * 2.0
    extension_risk = None if extension is None else float(np.clip(-np.sign(extension) * max(0.0, abs(extension) - 10.0) * 3.0, -100.0, 100.0))
    divergence = 0.0
    for item in divergences:
        if item.get("active"):
            divergence += 35.0 if item.get("direction") == "BULLISH" else -35.0
    divergence = float(np.clip(divergence, -100.0, 100.0))
    nearest = zones[0] if zones else None
    sr = 0.0
    if nearest and nearest.get("confluence") in {"HIGH", "VERY_HIGH"}:
        sr = 35.0 if nearest["role"] == "SUPPORT" else -35.0
        if abs(float(nearest["center"]) / price - 1.0) > 0.08:
            sr *= 0.5
    analog_return = ((analogs.get("returns") or {}).get("6M") or {}).get("median")
    analog = None if analog_return is None else float(np.clip(float(analog_return) * 6.0, -100.0, 100.0))
    return {
        "trend": round(trend, 2), "momentum": round(momentum, 2), "elliott": round(elliott, 2),
        "volume": finite(volume_component), "extension_risk": finite(extension_risk),
        "divergence": round(divergence, 2), "support_resistance": round(sr, 2),
        "historical_analog": finite(analog),
    }


def _horizon_states(
    monthly: dict[str, Any],
    weekly: dict[str, Any],
    momentum: dict[str, Any],
    probabilities: dict[str, int],
    zones: list[dict[str, Any]],
    price: float,
) -> dict[str, dict[str, str]]:
    short = "BULLISH" if momentum["score"] >= 20 else "BEARISH" if momentum["score"] <= -20 else "NEUTRAL"
    if momentum["trajectory"] == "DECELERATING" and short == "BEARISH" and weekly["state"] != "BEAR":
        short = "BOTTOMING"
    medium = "BULLISH" if weekly["state"] == "BULL" else "BEARISH" if weekly["state"] == "BEAR" else "NEUTRAL"
    six_month = max(probabilities, key=probabilities.get)
    nearest = zones[0] if zones else None
    zone_text = f"Nearest {nearest['role'].lower()} is {nearest['low']:.2f}–{nearest['high']:.2f}." if nearest else "No reliable zone is available."
    return {
        "short_term": {"range": "1–4 weeks", "state": short, "explanation": f"Weekly momentum is {momentum['classification']} and {momentum['trajectory']}. {zone_text}"},
        "medium_term": {"range": "1–3 months", "state": medium, "explanation": f"Weekly market structure is {weekly['state']} ({weekly['sequence']})."},
        "six_month": {"range": "3–6 months", "state": six_month, "explanation": f"Monthly priority state is {monthly['state']}; quantitative scenario probability is {probabilities[six_month]}%."},
    }


def _multi_timeframe_state(monthly: str, weekly: str) -> str:
    mapping = {
        ("BULL", "BULL"): "STRONG_BULL",
        ("BULL", "BEAR"): "CORRECTION_WITHIN_BULL_TREND",
        ("BEAR", "BULL"): "COUNTERTREND_BEAR_MARKET_RALLY",
        ("BEAR", "BEAR"): "STRONG_BEAR",
    }
    return mapping.get((monthly, weekly), "TRANSITION")


def _direction(state: str) -> str:
    return state if state in {"BULL", "BEAR"} else "NEUTRAL"


def _compact_momentum(state: str) -> str:
    if "POSITIVE" in state:
        return "POSITIVE"
    if "NEGATIVE" in state:
        return "NEGATIVE"
    return "NEUTRAL"

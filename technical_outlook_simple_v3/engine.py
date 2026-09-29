from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from elliott_waves.config import AssetSpec
from technical_outlook.analytics import (
    build_timeframe_bars,
    calculate_indicators,
    classify_structure,
    data_version,
    historical_analogs,
    momentum_state,
    moving_average_state,
    volume_state,
)

from .config import (
    CONFIG,
    CONFIG_VERSION,
    MODEL_VERSION,
    SCENARIO_ENGINE_VERSION,
    SR_ENGINE_VERSION,
    swing_config,
)
from .fibonacci import active_fibonacci_framework
from .scenario import build_weekly_scenario_matrix
from .support_resistance import build_support_resistance
from .swings import detect_causal_swings
from .volume_profile import build_volume_profile


class TechnicalOutlookSimpleV3Engine:
    def analyze(
        self,
        daily_bars: pd.DataFrame,
        spec: AssetSpec,
        *,
        previous: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        del previous  # v3 Quant output is a pure function of its current closed-bar input.
        created = created_at or datetime.now(timezone.utc)
        daily = daily_bars.loc[daily_bars["is_closed"].fillna(False)].copy().reset_index(drop=True)
        if daily.empty:
            raise ValueError(f"{spec.display_name}: no closed daily bars")
        daily = calculate_indicators(daily)
        weekly = calculate_indicators(build_timeframe_bars(daily, "1W", spec))
        if daily.empty or weekly.empty:
            raise ValueError(f"{spec.display_name}: insufficient Weekly/Daily history")

        configured = swing_config(spec.display_name)
        weekly_cfg = configured["weekly"]
        daily_cfg = configured["daily"]
        weekly_pivots = detect_causal_swings(
            weekly,
            timeframe="WEEKLY",
            atr_multiplier=float(weekly_cfg["atr_multiplier"]),
            min_reversal_pct=float(weekly_cfg["min_reversal_pct"]),
        )
        daily_pivots = detect_causal_swings(
            daily,
            timeframe="DAILY",
            atr_multiplier=float(daily_cfg["atr_multiplier"]),
            min_reversal_pct=float(daily_cfg["min_reversal_pct"]),
        )
        weekly_structure = classify_structure(weekly, weekly_pivots)
        daily_structure = classify_structure(daily, daily_pivots)
        weekly_ma = moving_average_state(weekly)
        daily_ma = moving_average_state(daily)
        weekly_momentum = momentum_state(weekly)
        daily_momentum = momentum_state(daily)
        weekly_volume_state = volume_state(weekly)
        daily_volume_state = volume_state(daily)
        weekly_profile = build_volume_profile(weekly, timeframe="WEEKLY")
        daily_profile = build_volume_profile(daily, timeframe="DAILY")
        weekly_fibonacci = active_fibonacci_framework(weekly_pivots, timeframe="WEEKLY")
        daily_fibonacci = active_fibonacci_framework(daily_pivots, timeframe="DAILY")
        as_of = pd.Timestamp(daily.iloc[-1]["timestamp"]).isoformat()
        price = float(daily.iloc[-1]["close"])
        weekly_zones = build_support_resistance(
            weekly, weekly_pivots, weekly_profile, weekly_fibonacci,
            timeframe="WEEKLY", as_of=as_of,
        )
        daily_zones = build_support_resistance(
            daily, daily_pivots, daily_profile, daily_fibonacci,
            timeframe="DAILY", as_of=as_of,
        )
        analogs = historical_analogs(weekly)
        weekly_atr = _finite(weekly.iloc[-1].get("atr14")) or price * 0.04
        scenario = build_weekly_scenario_matrix(
            current_price=price,
            weekly_atr=weekly_atr,
            weekly_bars_count=len(weekly),
            structure=weekly_structure,
            momentum=weekly_momentum,
            ma_structure=weekly_ma,
            volume_context=weekly_volume_state,
            analogs=analogs,
            extension_pct=_finite(weekly.iloc[-1].get("extension200")),
            weekly_zones=weekly_zones,
            fibonacci=weekly_fibonacci,
        )
        dominant = str(scenario["dominant_scenario"])
        dominant_row = next(item for item in scenario["scenarios"] if item["scenario"] == dominant)
        created_iso = pd.Timestamp(created).isoformat()
        version = data_version(daily)
        payload: dict[str, Any] = {
            "snapshot_id": _snapshot_id(spec.display_name, as_of, created_iso, version),
            "as_of": as_of,
            "as_of_date": pd.Timestamp(as_of).date().isoformat(),
            "as_of_timestamp": as_of,
            "ticker": spec.display_name,
            "provider_symbol": spec.provider_symbol,
            "source_id": spec.source_id,
            "source_metadata": spec.to_dict(),
            "price": price,
            "model_version": MODEL_VERSION,
            "config_version": CONFIG_VERSION,
            "sr_engine_version": SR_ENGINE_VERSION,
            "scenario_engine_version": SCENARIO_ENGINE_VERSION,
            "model_parameters": CONFIG,
            "swing_config_source": configured["source"],
            "weekly_swing_config": weekly_cfg,
            "daily_swing_config": daily_cfg,
            "btc_calibration_start": configured.get("calibration_start"),
            "weekly_pivots": weekly_pivots,
            "daily_pivots": daily_pivots,
            "weekly_structure": weekly_structure,
            "daily_structure": daily_structure,
            "weekly_moving_averages": weekly_ma,
            "daily_moving_averages": daily_ma,
            "weekly_indicators": weekly_momentum,
            "daily_indicators": daily_momentum,
            "weekly_momentum_summary": weekly_momentum,
            "daily_momentum_summary": daily_momentum,
            "weekly_volume_state": weekly_volume_state,
            "daily_volume_state": daily_volume_state,
            "weekly_volume_profile": weekly_profile,
            "daily_volume_profile": daily_profile,
            "weekly_volume_profile_summary": weekly_profile,
            "daily_volume_profile_summary": daily_profile,
            "weekly_fibonacci_framework": weekly_fibonacci,
            "daily_fibonacci_framework": daily_fibonacci,
            "weekly_zones": weekly_zones,
            "daily_zones": daily_zones,
            "weekly_scenario_matrix": scenario["scenarios"],
            "weekly_expected_path": scenario["expected_path"],
            "scenario_components": scenario["components"],
            "scenario_probabilities": scenario["probabilities"],
            "bull_probability": scenario["probabilities"]["BULLISH"],
            "neutral_probability": scenario["probabilities"]["NEUTRAL"],
            "bear_probability": scenario["probabilities"]["BEARISH"],
            "historical_analogs": analogs,
            "final_state": {
                "structural_trend": _direction(weekly_structure.get("state")),
                "weekly_momentum": weekly_momentum.get("classification"),
                "six_month_bias": dominant,
                "confidence": dominant_row.get("confidence"),
            },
            "deterministic_narrative": _narrative(
                spec.display_name, weekly_structure, weekly_momentum, dominant,
                scenario["probabilities"], dominant_row,
            ),
            "quant_updated_at": created_iso,
            "llm_enabled": False,
            "llm_interpretation": None,
            "llm_model": None,
            "llm_updated_at": None,
            "llm_status": "DISABLED",
            "data_version": version,
            "data_status": "CURRENT",
            "stale_reason": None,
        }
        chart_bars = int(CONFIG["chart"]["bars"])
        charts = {"1W": _chart_frame(weekly, chart_bars), "1D": _chart_frame(daily, chart_bars)}
        return payload, charts


def _chart_frame(frame: pd.DataFrame, bars: int) -> pd.DataFrame:
    columns = [
        "timestamp", "open", "high", "low", "close", "volume", "sma50", "sma100", "sma200",
        "rsi14", "macd", "macd_signal", "macd_hist", "roc12", "extension200", "atr14",
    ]
    return frame[[column for column in columns if column in frame]].tail(bars).copy()


def _direction(value: Any) -> str:
    return str(value) if str(value) in {"BULL", "BEAR"} else "NEUTRAL"


def _narrative(
    ticker: str,
    structure: dict[str, Any],
    momentum: dict[str, Any],
    bias: str,
    probabilities: dict[str, int],
    scenario: dict[str, Any],
) -> str:
    target = scenario.get("primary_target") or {}
    target_text = target.get("range") or target.get("label") or "No qualifying Weekly target"
    return (
        f"{ticker}: Weekly structure is {structure.get('state', 'N/A')} ({structure.get('sequence', 'N/A')}); "
        f"Weekly momentum is {momentum.get('classification', 'N/A')} and {momentum.get('trajectory', 'N/A')}. "
        f"The deterministic 6M bias is {bias} ({probabilities.get(bias, 0)}%) with {scenario.get('confidence', 'LOW')} confidence. "
        f"Primary destination: {target_text}."
    )


def _snapshot_id(ticker: str, as_of: str, created_at: str, data_version_value: str) -> str:
    value = {
        "ticker": ticker,
        "as_of": as_of,
        "created_at": created_at,
        "data_version": data_version_value,
        "model_version": MODEL_VERSION,
        "config_version": CONFIG_VERSION,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

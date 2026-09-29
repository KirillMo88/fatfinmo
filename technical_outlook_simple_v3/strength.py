from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG


def classify_strength(score: float) -> str:
    thresholds = CONFIG["strength"]["class_thresholds"]
    if score < float(thresholds["moderate"]):
        return "WEAK"
    if score < float(thresholds["strong"]):
        return "MODERATE"
    if score < float(thresholds["very_strong"]):
        return "STRONG"
    return "VERY_STRONG"


def interaction_strength(
    frame: pd.DataFrame,
    low: float,
    high: float,
    current_role: str,
    *,
    valid_from: Any,
    timeframe: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = CONFIG["strength"]["weekly" if str(timeframe).upper() == "WEEKLY" else "daily"]
    separation = int(cfg["minimum_touch_separation"])
    reaction_window = int(cfg["reaction_window"])
    minimum_reaction = float(CONFIG["strength"]["minimum_reaction_atr"])
    exit_atr = float(CONFIG["strength"]["episode_exit_atr"])
    values = frame.reset_index(drop=True)
    timestamps = pd.to_datetime(values["timestamp"], errors="coerce", utc=True)
    valid_timestamp = pd.to_datetime(valid_from, errors="coerce", utc=True)
    eligible_after = 0
    interactions: list[dict[str, Any]] = []
    for index, row in values.iterrows():
        if index < eligible_after or pd.isna(timestamps.iloc[index]):
            continue
        if pd.notna(valid_timestamp) and timestamps.iloc[index] < valid_timestamp:
            continue
        bar_low = _finite(row.get("low"))
        bar_high = _finite(row.get("high"))
        close = _finite(row.get("close"))
        atr = _finite(row.get("atr14"))
        if None in {bar_low, bar_high, close}:
            continue
        if not (float(bar_low) <= high and float(bar_high) >= low):
            continue
        atr = atr or max(abs(float(close)) * 0.02, 1e-12)
        interaction_role = "SUPPORT" if float(close) >= (low + high) / 2 else "RESISTANCE"
        end = min(len(values) - 1, index + reaction_window)
        future = values.iloc[index + 1 : end + 1]
        if interaction_role == "SUPPORT":
            reaction = max(0.0, float(pd.to_numeric(future.get("high"), errors="coerce").max()) - high) if len(future) else 0.0
            failure = max(0.0, low - float(pd.to_numeric(future.get("low"), errors="coerce").min())) if len(future) else 0.0
        else:
            reaction = max(0.0, low - float(pd.to_numeric(future.get("low"), errors="coerce").min())) if len(future) else 0.0
            failure = max(0.0, float(pd.to_numeric(future.get("high"), errors="coerce").max()) - high) if len(future) else 0.0
        reaction_atr = reaction / max(float(atr), 1e-12)
        if index + reaction_window >= len(values):
            state = "PENDING_TOUCH"
            confirmation_time = None
        elif reaction_atr >= minimum_reaction:
            state = "CONFIRMED_TOUCH"
            confirmation_time = timestamps.iloc[end].isoformat()
        else:
            state = "FAILED_TOUCH"
            confirmation_time = timestamps.iloc[end].isoformat()
        interactions.append({
            "interaction_time": timestamps.iloc[index].isoformat(),
            "confirmation_time": confirmation_time,
            "interaction_role": interaction_role,
            "state": state,
            "reaction_magnitude": reaction,
            "reaction_magnitude_ATR": reaction_atr,
            "failure_magnitude_ATR": failure / max(float(atr), 1e-12),
        })
        exit_index = index + 1
        while exit_index < len(values):
            exit_close = _finite(values.iloc[exit_index].get("close"))
            exit_row_atr = _finite(values.iloc[exit_index].get("atr14")) or atr
            if exit_close is not None and (exit_close < low - exit_atr * exit_row_atr or exit_close > high + exit_atr * exit_row_atr):
                break
            exit_index += 1
        eligible_after = max(index + separation, exit_index)

    active_role = current_role if current_role in {"SUPPORT", "RESISTANCE"} else (interactions[-1]["interaction_role"] if interactions else "UNKNOWN")
    relevant = [item for item in interactions if item["interaction_role"] == active_role]
    confirmed = [item for item in relevant if item["state"] == "CONFIRMED_TOUCH"]
    pending = [item for item in relevant if item["state"] == "PENDING_TOUCH"]
    failed = [item for item in relevant if item["state"] == "FAILED_TOUCH"]
    count = len(confirmed)
    touch_component = 0.0 if count == 0 else 1.0 if count == 1 else 1.8 if count == 2 else 2.4 if count == 3 else 3.0
    reactions = [float(item["reaction_magnitude_ATR"]) for item in confirmed]
    median_reaction = float(np.median(reactions)) if reactions else 0.0
    reaction_component = 0.0 if median_reaction < 0.5 else 1.0 if median_reaction < 1.0 else 2.0 if median_reaction < 1.5 else 3.0
    completed = len(confirmed) + len(failed)
    hold_rate = len(confirmed) / completed if completed else 0.0
    raw_hold = 0.0 if hold_rate < 0.4 else 1.0 if hold_rate < 0.6 else 1.5 if hold_rate < 0.8 else 2.0
    sample_factor = min(completed / max(int(CONFIG["strength"]["minimum_hold_episodes"]), 1), 1.0)
    hold_component = raw_hold * sample_factor
    recency_component = 0.0
    confirmations = [pd.Timestamp(item["confirmation_time"]) for item in confirmed if item.get("confirmation_time")]
    if confirmations:
        age = int((timestamps > max(confirmations)).sum())
        first, second, third, fourth = [int(value) for value in cfg["recency_bars"]]
        recency_component = 1.0 if age <= first else 0.75 if age <= second else 0.5 if age <= third else 0.25 if age <= fourth else 0.0
    reaction_mad = float(np.median(np.abs(np.asarray(reactions) - median_reaction))) if len(reactions) >= 2 else 0.0
    consistency_component = float(np.clip(1.0 - reaction_mad / median_reaction, 0.0, 1.0)) if len(reactions) >= 2 and median_reaction > 0 else 0.0
    components = {
        "touch": round(touch_component, 4),
        "reaction": round(reaction_component, 4),
        "hold": round(hold_component, 4),
        "recency": round(recency_component, 4),
        "consistency": round(consistency_component, 4),
    }
    score = float(np.clip(sum(components.values()), 0.0, 10.0))
    return interactions, {
        "confirmed_touch_count": len(confirmed),
        "pending_touch_count": len(pending),
        "failed_touch_count": len(failed),
        "reaction_statistics": {
            "median_ATR": round(median_reaction, 4),
            "MAD_ATR": round(reaction_mad, 4),
            "hold_rate": round(hold_rate, 4),
            "completed_episodes": completed,
            "active_role": active_role,
        },
        "strength_components": components,
        "strength_score": round(score, 4),
        "strength_class": classify_strength(score),
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

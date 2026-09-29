from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import CONFIG, CONFIG_VERSION, MODEL_VERSION, SR_ENGINE_VERSION


FAMILY_BY_SOURCE = {
    "structural_swing": "SWING_STRUCTURE",
    "weekly_swing": "SWING_STRUCTURE",
    "daily_swing": "SWING_STRUCTURE",
    "minor_swing": "SWING_STRUCTURE",
    "sma50": "MOVING_AVERAGE",
    "sma100": "MOVING_AVERAGE",
    "sma200": "MOVING_AVERAGE",
    "poc": "VOLUME_ACCEPTANCE",
    "hvn": "VOLUME_ACCEPTANCE",
    "fibonacci": "FIBONACCI",
    "round_number": "ROUND_NUMBER",
    "channel_boundary": "CHANNEL_STRUCTURE",
    "lvn": "LOW_ACCEPTANCE_CONTEXT",
}
POSITIVE_FAMILIES = {
    "SWING_STRUCTURE",
    "MOVING_AVERAGE",
    "VOLUME_ACCEPTANCE",
    "FIBONACCI",
    "ROUND_NUMBER",
    "CHANNEL_STRUCTURE",
}


def build_support_resistance_v1(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    timeframe: str,
    max_pivots: int | None,
    max_zones: int,
) -> list[dict[str, Any]]:
    """Build deterministic S/R V1 zones with independent family scoring."""
    if frame.empty:
        return []
    timeframe = str(timeframe).upper()
    cfg_key = "weekly" if timeframe == "WEEKLY" else "daily"
    row = frame.iloc[-1]
    current = float(row["close"])
    atr = _finite(row.get("atr14")) or current * 0.02
    as_of = pd.Timestamp(row["timestamp"]).isoformat()
    members = _build_members(frame, pivots, profile, timeframe=timeframe, max_pivots=max_pivots)
    clusters = _deterministic_clusters(members, atr, cfg_key)
    zones = [
        _zone_from_members(cluster, frame, current=current, atr=atr, timeframe=timeframe, as_of=as_of)
        for cluster in clusters
    ]
    zones = [zone for zone in zones if zone is not None]
    zones.sort(key=lambda zone: (abs(float(zone["center"]) - current), -float(zone["final_confluence_score"]), zone["zone_id"]))
    if len(zones) <= max_zones:
        return zones
    nearest_count = max(1, max_zones // 2)
    selected = zones[:nearest_count]
    seen = {zone["zone_id"] for zone in selected}
    for zone in sorted(
        zones,
        key=lambda item: (
            -_class_rank(item["confluence_class"]),
            -_class_rank(item["strength_class"]),
            -float(item["final_confluence_score"]),
            abs(float(item["center"]) - current),
            item["zone_id"],
        ),
    ):
        if zone["zone_id"] in seen:
            continue
        selected.append(zone)
        seen.add(zone["zone_id"])
        if len(selected) >= max_zones:
            break
    return sorted(selected, key=lambda zone: (abs(float(zone["center"]) - current), zone["zone_id"]))


def build_cross_timeframe_zones(
    daily_zones: list[dict[str, Any]],
    weekly_zones: list[dict[str, Any]],
    daily_frame: pd.DataFrame,
    *,
    max_zones: int = 12,
) -> list[dict[str, Any]]:
    """Create immutable derived same-role Daily/Weekly intersections."""
    cfg = CONFIG["support_resistance_v1"]
    minimum_overlap = float(cfg["cross_timeframe_overlap"])
    candidates: list[dict[str, Any]] = []
    if daily_frame.empty:
        return candidates
    current = float(daily_frame.iloc[-1]["close"])
    atr = _finite(daily_frame.iloc[-1].get("atr14")) or current * 0.02
    as_of = pd.Timestamp(daily_frame.iloc[-1]["timestamp"]).isoformat()
    for daily in daily_zones:
        for weekly in weekly_zones:
            if daily.get("role") != weekly.get("role") or daily.get("role") not in {"SUPPORT", "RESISTANCE"}:
                continue
            low = max(float(daily["low"]), float(weekly["low"]))
            high = min(float(daily["high"]), float(weekly["high"]))
            if high <= low:
                continue
            denominator = min(float(daily["high"]) - float(daily["low"]), float(weekly["high"]) - float(weekly["low"]))
            overlap_ratio = (high - low) / max(denominator, 1e-12)
            if overlap_ratio < minimum_overlap:
                continue
            members = _dedupe_members(list(daily.get("source_members") or []) + list(weekly.get("source_members") or []))
            family_scores, family_sources = _family_scores(members)
            base = float(sum(family_scores.values()))
            bonus = min(float(cfg["cross_timeframe_bonus"]), float(cfg["cross_timeframe_bonus_cap"]))
            final = base + bonus
            family_count = len([family for family in family_scores if family in POSITIVE_FAMILIES])
            label = _confluence_class(final, family_count)
            center = (low + high) / 2.0
            role = str(daily["role"])
            valid_from = max(str(daily.get("valid_from") or as_of), str(weekly.get("valid_from") or as_of))
            interactions, strength = _interaction_strength(
                daily_frame,
                low,
                high,
                role,
                valid_from=valid_from,
                timeframe="DAILY",
            )
            parents = sorted({str(daily["zone_id"]), str(weekly["zone_id"])})
            zone_id = _stable_id({"as_of": as_of, "timeframe": "CROSS_TIMEFRAME", "role": role, "bounds": [round(low, 8), round(high, 8)], "parents": parents})
            lineage_id = _stable_id({"timeframe": "CROSS_TIMEFRAME", "role": role, "parents": parents})
            for interaction in interactions:
                interaction["zone_id"] = zone_id
            candidates.append({
                "zone_id": zone_id,
                "zone_lineage_id": lineage_id,
                "zone_revision_id": zone_id,
                "as_of": as_of,
                "timeframe": "CROSS_TIMEFRAME",
                "structural_degree": "MULTI_TIMEFRAME",
                "role": role,
                "low": low,
                "high": high,
                "lower_bound": low,
                "upper_bound": high,
                "center": center,
                "width": high - low,
                "sources": sorted({member["source"] for member in members}),
                "source_members": members,
                "source_families": sorted(family_scores),
                "family_scores": family_scores,
                "family_sources": family_sources,
                "independent_family_count": family_count,
                "base_confluence_score": round(base, 4),
                "cross_timeframe_bonus": round(bonus, 4),
                "final_confluence_score": round(final, 4),
                "confluence_score": round(final, 4),
                "confluence_class": label,
                "confluence": label,
                "confirmed_touch_count": strength["confirmed_touch_count"],
                "pending_touch_count": strength["pending_touch_count"],
                "failed_touch_count": strength["failed_touch_count"],
                "reaction_statistics": strength["reaction_statistics"],
                "strength_components": strength["strength_components"],
                "strength_score": strength["strength_score"],
                "strength_class": strength["strength_class"],
                "interactions": interactions,
                "distance_pct": current / center - 1.0 if center else None,
                "distance_daily_ATR": abs(current - center) / max(atr, 1e-12),
                "distance_weekly_ATR": None,
                "cross_timeframe": True,
                "parent_daily_zone_ids": [str(daily["zone_id"])],
                "parent_weekly_zone_ids": [str(weekly["zone_id"])],
                "overlap_ratio": round(overlap_ratio, 4),
                "valid_from": valid_from,
                "valid_to": None,
                "observation_time": as_of,
                "confirmation_time": valid_from,
                "timeframes": ["DAILY", "WEEKLY"],
                "active": True,
                "broken": False,
                "model_version": MODEL_VERSION,
                "config_version": CONFIG_VERSION,
                "sr_engine_version": SR_ENGINE_VERSION,
                "relevance_by_horizon": {},
                "relevance_by_scenario": {},
            })
    deduped = _dedupe_cross_zones(candidates)
    deduped.sort(key=lambda zone: (abs(float(zone["center"]) - current), -float(zone["final_confluence_score"]), zone["zone_id"]))
    return deduped[:max_zones]


def _build_members(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    timeframe: str,
    max_pivots: int | None,
) -> list[dict[str, Any]]:
    cfg = CONFIG["support_resistance_v1"]
    weights = cfg["member_weights"]
    row = frame.iloc[-1]
    as_of = pd.Timestamp(row["timestamp"]).isoformat()
    selected = sorted(
        (pivot for pivot in pivots if pivot.get("status") == "CONFIRMED"),
        key=lambda pivot: (str(pivot.get("confirmation_time") or pivot.get("pivot_time") or ""), str(pivot.get("pivot_id") or "")),
    )
    if max_pivots is not None:
        selected = selected[-max_pivots:]
    members: list[dict[str, Any]] = []
    for item in selected:
        source = _pivot_source(item, timeframe)
        members.append(_member(float(item["price"]), source, weights[source], timeframe, str(item.get("pivot_id") or "pivot"), item.get("pivot_time"), item.get("confirmation_time")))
    for window in CONFIG["ma"]["windows"]:
        value = _finite(row.get(f"sma{window}"))
        if value is not None:
            source = f"sma{window}"
            members.append(_member(value, source, weights[source], timeframe, f"{timeframe}:{source}:{as_of}", as_of, as_of))
    if profile.get("status") == "AVAILABLE":
        if profile.get("poc") is not None:
            members.append(_member(float(profile["poc"]), "poc", weights["poc"], timeframe, f"{timeframe}:poc:{as_of}", as_of, as_of))
        for index, value in enumerate(profile.get("hvns") or []):
            members.append(_member(float(value), "hvn", weights["hvn"], timeframe, f"{timeframe}:hvn:{index}:{as_of}", as_of, as_of))
        for index, value in enumerate(profile.get("lvns") or []):
            members.append(_member(float(value), "lvn", weights["lvn"], timeframe, f"{timeframe}:lvn:{index}:{as_of}", as_of, as_of))
    if len(selected) >= 2:
        first, second = selected[-2], selected[-1]
        low, high = sorted((float(first["price"]), float(second["price"])))
        confirmation = max(str(first.get("confirmation_time") or as_of), str(second.get("confirmation_time") or as_of))
        for ratio in (0.382, 0.5, 0.618, 1.0, 1.618):
            price = high - (high - low) * ratio if ratio <= 1 else high + (high - low) * (ratio - 1)
            members.append(_member(price, "fibonacci", weights["fibonacci"], timeframe, f"{timeframe}:fib:{ratio}:{confirmation}", confirmation, confirmation))
    current = float(row["close"])
    magnitude = 10 ** max(0, int(math.floor(math.log10(max(abs(current), 1.0)))) - 1)
    for multiple in range(-2, 3):
        price = round(current / magnitude + multiple) * magnitude
        members.append(_member(price, "round_number", weights["round_number"], timeframe, f"{timeframe}:round:{price}", as_of, as_of))
    return _dedupe_members(members)


def _member(price: float, source: str, weight: float, timeframe: str, member_id: str, observation: Any, confirmation: Any) -> dict[str, Any]:
    return {
        "member_id": str(member_id),
        "price": float(price),
        "source": source,
        "family": FAMILY_BY_SOURCE[source],
        "weight": float(weight),
        "timeframe": timeframe,
        "observation_time": pd.Timestamp(observation).isoformat() if observation is not None else None,
        "confirmation_time": pd.Timestamp(confirmation).isoformat() if confirmation is not None else None,
    }


def _pivot_source(pivot: dict[str, Any], timeframe: str) -> str:
    degree = str(pivot.get("degree") or "").upper()
    if degree in {"STRUCTURAL", "PRIMARY"}:
        return "structural_swing"
    if timeframe == "WEEKLY" or degree == "MAJOR":
        return "weekly_swing"
    if degree == "MINOR":
        return "minor_swing"
    return "daily_swing"


def _deterministic_clusters(members: list[dict[str, Any]], atr: float, cfg_key: str) -> list[list[dict[str, Any]]]:
    ordered = sorted(members, key=lambda item: (float(item["price"]), item["family"], item["source"], str(item.get("confirmation_time")), item["member_id"]))
    clusters: list[list[dict[str, Any]]] = []
    for item in ordered:
        eligible: list[tuple[float, int, list[dict[str, Any]]]] = []
        for index, cluster in enumerate(clusters):
            trial = cluster + [item]
            center = _weighted_median([member["price"] for member in trial], [_center_weight(member) for member in trial])
            radius = _cluster_radius(center, atr, cfg_key)
            if all(abs(float(member["price"]) - center) <= radius + 1e-12 for member in trial):
                eligible.append((abs(float(item["price"]) - center), index, trial))
        if not eligible:
            clusters.append([item])
            continue
        _, index, trial = min(eligible, key=lambda value: (value[0], value[1]))
        clusters[index] = _stabilize_cluster(trial, atr, cfg_key)
    changed = True
    while changed:
        changed = False
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                trial = _stabilize_cluster(clusters[left] + clusters[right], atr, cfg_key)
                if len(trial) != len(clusters[left]) + len(clusters[right]):
                    continue
                center = _weighted_median([item["price"] for item in trial], [_center_weight(item) for item in trial])
                radius = _cluster_radius(center, atr, cfg_key)
                if all(abs(float(item["price"]) - center) <= radius + 1e-12 for item in trial):
                    clusters[left] = trial
                    del clusters[right]
                    changed = True
                    break
            if changed:
                break
    return [sorted(cluster, key=lambda item: (item["price"], item["member_id"])) for cluster in clusters if cluster]


def _stabilize_cluster(cluster: list[dict[str, Any]], atr: float, cfg_key: str) -> list[dict[str, Any]]:
    current = list(cluster)
    while current:
        center = _weighted_median([item["price"] for item in current], [_center_weight(item) for item in current])
        radius = _cluster_radius(center, atr, cfg_key)
        retained = [item for item in current if abs(float(item["price"]) - center) <= radius + 1e-12]
        if len(retained) == len(current):
            return retained
        current = retained
    return []


def _zone_from_members(
    members: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    current: float,
    atr: float,
    timeframe: str,
    as_of: str,
) -> dict[str, Any] | None:
    if not members:
        return None
    cfg_key = "weekly" if timeframe == "WEEKLY" else "daily"
    center = _weighted_median([item["price"] for item in members], [_center_weight(item) for item in members])
    radius = _cluster_radius(center, atr, cfg_key)
    members = [item for item in members if abs(float(item["price"]) - center) <= radius + 1e-12]
    if not members:
        return None
    center = _weighted_median([item["price"] for item in members], [_center_weight(item) for item in members])
    deviations = [abs(float(item["price"]) - center) for item in members]
    mad = _weighted_median(deviations, [_center_weight(item) for item in members])
    cfg = CONFIG["support_resistance_v1"][cfg_key]
    min_half = max(center * float(cfg["min_width_fraction"]), atr * float(cfg["min_width_atr_multiplier"])) / 2.0
    max_half = center * float(cfg["max_total_width_fraction"]) / 2.0
    raw_half = max(float(CONFIG["support_resistance_v1"]["mad_multiplier"]) * mad, max(deviations, default=0.0))
    half_width = min(max(raw_half, min_half), max_half)
    low, high = center - half_width, center + half_width
    role = "SUPPORT" if current > high else "RESISTANCE" if current < low else "TESTING"
    family_scores, family_sources = _family_scores(members)
    family_count = len([family for family in family_scores if family in POSITIVE_FAMILIES])
    base = float(sum(family_scores.values()))
    label = _confluence_class(base, family_count)
    structural_confirmations = [
        item.get("confirmation_time") for item in members
        if item.get("family") in {"SWING_STRUCTURE", "FIBONACCI"} and item.get("confirmation_time")
    ]
    valid_from = max(structural_confirmations) if structural_confirmations else as_of
    interactions, strength = _interaction_strength(frame, low, high, role, valid_from=valid_from, timeframe=timeframe)
    member_ids = sorted(item["member_id"] for item in members)
    zone_id = _stable_id({"as_of": as_of, "timeframe": timeframe, "role": role, "bounds": [round(low, 8), round(high, 8)], "members": member_ids})
    lineage_id = _stable_id({"timeframe": timeframe, "role": role, "center_bucket": round(center / max(radius, 1e-12), 0), "families": sorted(family_scores)})
    for interaction in interactions:
        interaction["zone_id"] = zone_id
    weekly_atr = atr if timeframe == "WEEKLY" else None
    daily_atr = atr if timeframe == "DAILY" else None
    return {
        "zone_id": zone_id,
        "zone_lineage_id": lineage_id,
        "zone_revision_id": zone_id,
        "as_of": as_of,
        "timeframe": timeframe,
        "structural_degree": "STRUCTURAL" if timeframe == "WEEKLY" else "TACTICAL",
        "role": role,
        "low": low,
        "high": high,
        "lower_bound": low,
        "upper_bound": high,
        "center": center,
        "width": high - low,
        "cluster_radius": radius,
        "weighted_mad": mad,
        "sources": sorted({item["source"] for item in members}),
        "source_members": members,
        "source_families": sorted(family_scores),
        "family_scores": family_scores,
        "family_sources": family_sources,
        "independent_family_count": family_count,
        "base_confluence_score": round(base, 4),
        "cross_timeframe_bonus": 0.0,
        "final_confluence_score": round(base, 4),
        "confluence_score": round(base, 4),
        "confluence_class": label,
        "confluence": label,
        "confirmed_touch_count": strength["confirmed_touch_count"],
        "pending_touch_count": strength["pending_touch_count"],
        "failed_touch_count": strength["failed_touch_count"],
        "reaction_statistics": strength["reaction_statistics"],
        "strength_components": strength["strength_components"],
        "strength_score": strength["strength_score"],
        "strength_class": strength["strength_class"],
        "interactions": interactions,
        "distance_pct": current / center - 1.0 if center else None,
        "distance_daily_ATR": abs(current - center) / max(daily_atr, 1e-12) if daily_atr else None,
        "distance_weekly_ATR": abs(current - center) / max(weekly_atr, 1e-12) if weekly_atr else None,
        "cross_timeframe": False,
        "parent_daily_zone_ids": [],
        "parent_weekly_zone_ids": [],
        "overlap_ratio": None,
        "valid_from": valid_from,
        "valid_to": None,
        "observation_time": as_of,
        "confirmation_time": valid_from,
        "timeframes": [timeframe],
        "active": True,
        "broken": False,
        "model_version": MODEL_VERSION,
        "config_version": CONFIG_VERSION,
        "sr_engine_version": SR_ENGINE_VERSION,
        "relevance_by_horizon": {},
        "relevance_by_scenario": {},
    }


def _family_scores(members: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[str]]]:
    cfg = CONFIG["support_resistance_v1"]
    grouped: dict[str, dict[str, float]] = {}
    for member in members:
        family = str(member["family"])
        if family not in POSITIVE_FAMILIES:
            continue
        source = str(member["source"])
        grouped.setdefault(family, {})[source] = max(float(member["weight"]), grouped.setdefault(family, {}).get(source, 0.0))
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for family, source_weights in sorted(grouped.items()):
        unique_sources = sorted(source_weights)
        breadth = min(
            float(cfg["family_breadth_increment"]) * max(0, len(unique_sources) - 1),
            float(cfg["family_breadth_cap"]),
        )
        scores[family] = round(max(source_weights.values()) + breadth, 4)
        sources[family] = unique_sources
    return scores, sources


def _interaction_strength(
    frame: pd.DataFrame,
    low: float,
    high: float,
    current_role: str,
    *,
    valid_from: str,
    timeframe: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg_key = "weekly" if timeframe == "WEEKLY" else "daily"
    cfg = CONFIG["support_resistance_v1"][cfg_key]
    values = frame.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(values["timestamp"], errors="coerce")
    start_time = pd.Timestamp(valid_from)
    separation = int(cfg["minimum_touch_separation"])
    reaction_window = int(cfg["reaction_window"])
    minimum_reaction = float(CONFIG["support_resistance_v1"]["minimum_reaction_atr"])
    interactions: list[dict[str, Any]] = []
    last_touch = -10_000
    eligible_after = 0
    for index in range(1, len(values)):
        if pd.isna(timestamps.iloc[index]) or timestamps.iloc[index] < start_time or index - last_touch < separation or index < eligible_after:
            continue
        bar_low = _finite(values.iloc[index].get("low"))
        bar_high = _finite(values.iloc[index].get("high"))
        if bar_low is None or bar_high is None or bar_low > high or bar_high < low:
            continue
        previous_close = _finite(values.iloc[index - 1].get("close"))
        if previous_close is None:
            continue
        role = "SUPPORT" if previous_close > high else "RESISTANCE" if previous_close < low else "UNKNOWN"
        if role == "UNKNOWN":
            continue
        atr = _finite(values.iloc[index].get("atr14")) or max((high - low) * 2.0, abs((low + high) / 2.0) * 0.01)
        end = min(len(values) - 1, index + reaction_window)
        future = values.iloc[index : end + 1]
        if role == "SUPPORT":
            reaction = max(0.0, float(pd.to_numeric(future["high"], errors="coerce").max()) - high)
            direction = "UP"
        else:
            reaction = max(0.0, low - float(pd.to_numeric(future["low"], errors="coerce").min()))
            direction = "DOWN"
        reaction_atr = reaction / max(atr, 1e-12)
        confirmed_offset = None
        for offset in range(0, end - index + 1):
            window = values.iloc[index : index + offset + 1]
            achieved = (
                float(pd.to_numeric(window["high"], errors="coerce").max()) - high
                if role == "SUPPORT"
                else low - float(pd.to_numeric(window["low"], errors="coerce").min())
            )
            if achieved / max(atr, 1e-12) >= minimum_reaction:
                confirmed_offset = offset
                break
        if confirmed_offset is not None:
            state = "CONFIRMED_TOUCH"
            confirmation_time = pd.Timestamp(timestamps.iloc[index + confirmed_offset]).isoformat()
            failure_time = None
        elif index + reaction_window >= len(values):
            state = "PENDING_TOUCH"
            confirmation_time = None
            failure_time = None
        else:
            state = "FAILED_TOUCH"
            confirmation_time = None
            failure_time = pd.Timestamp(timestamps.iloc[index + reaction_window]).isoformat()
        touch_time = pd.Timestamp(timestamps.iloc[index]).isoformat()
        interaction_id = _stable_id({"touch": touch_time, "role": role, "range": [round(low, 8), round(high, 8)], "timeframe": timeframe})
        interactions.append({
            "interaction_id": interaction_id,
            "touch_time": touch_time,
            "confirmation_time": confirmation_time,
            "failure_time": failure_time,
            "interaction_role": role,
            "direction": direction,
            "ATR_at_touch": atr,
            "reaction_magnitude": reaction,
            "reaction_magnitude_ATR": reaction_atr,
            "state": state,
        })
        last_touch = index
        exit_distance = float(CONFIG["support_resistance_v1"]["episode_exit_atr"]) * atr
        exit_index = len(values)
        for candidate_index in range(index + 1, len(values)):
            candidate_close = _finite(values.iloc[candidate_index].get("close"))
            if candidate_close is not None and (candidate_close > high + exit_distance or candidate_close < low - exit_distance):
                exit_index = candidate_index
                break
        eligible_after = max(index + separation, exit_index)
    active_role = current_role if current_role in {"SUPPORT", "RESISTANCE"} else (interactions[-1]["interaction_role"] if interactions else "UNKNOWN")
    relevant = [item for item in interactions if item["interaction_role"] == active_role]
    confirmed = [item for item in relevant if item["state"] == "CONFIRMED_TOUCH"]
    pending = [item for item in relevant if item["state"] == "PENDING_TOUCH"]
    failed = [item for item in relevant if item["state"] == "FAILED_TOUCH"]
    touch_count = len(confirmed)
    touch_component = 0.0 if touch_count == 0 else 1.0 if touch_count == 1 else 1.8 if touch_count == 2 else 2.4 if touch_count == 3 else 3.0
    reactions = [float(item["reaction_magnitude_ATR"]) for item in confirmed]
    median_reaction = float(np.median(reactions)) if reactions else 0.0
    reaction_component = 0.0 if median_reaction < 0.5 else 1.0 if median_reaction < 1.0 else 2.0 if median_reaction < 1.5 else 3.0
    completed = len(confirmed) + len(failed)
    hold_rate = len(confirmed) / completed if completed else 0.0
    raw_hold = 0.0 if hold_rate < 0.4 else 1.0 if hold_rate < 0.6 else 1.5 if hold_rate < 0.8 else 2.0
    sample_factor = min(completed / max(int(CONFIG["support_resistance_v1"]["minimum_hold_episodes"]), 1), 1.0)
    hold_component = raw_hold * sample_factor
    recency_component = 0.0
    if confirmed:
        most_recent = max(pd.Timestamp(item["confirmation_time"]) for item in confirmed if item.get("confirmation_time"))
        age_bars = int((timestamps > most_recent).sum())
        first, second, third, fourth = [int(value) for value in cfg["recency_bars"]]
        recency_component = 1.0 if age_bars <= first else 0.75 if age_bars <= second else 0.5 if age_bars <= third else 0.25 if age_bars <= fourth else 0.0
    consistency_component = 0.0
    reaction_mad = 0.0
    if len(reactions) >= 2 and median_reaction > 0:
        reaction_mad = float(np.median(np.abs(np.asarray(reactions) - median_reaction)))
        consistency_component = float(np.clip(1.0 - reaction_mad / median_reaction, 0.0, 1.0))
    components = {
        "touch": round(touch_component, 4),
        "reaction": round(reaction_component, 4),
        "hold": round(hold_component, 4),
        "recency": round(recency_component, 4),
        "consistency": round(consistency_component, 4),
    }
    score = float(np.clip(sum(components.values()), 0.0, 10.0))
    strength_class = "WEAK" if score < 2.5 else "MODERATE" if score < 5.0 else "STRONG" if score < 7.5 else "VERY_STRONG"
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
        "strength_class": strength_class,
    }


def _dedupe_cross_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    threshold = float(CONFIG["support_resistance_v1"]["cross_timeframe_duplicate_overlap"])
    groups: list[list[dict[str, Any]]] = []
    for zone in sorted(zones, key=lambda item: (item["role"], item["low"], item["high"], item["zone_id"])):
        matched = None
        for group in groups:
            representative = group[0]
            if representative["role"] != zone["role"]:
                continue
            overlap = max(0.0, min(representative["high"], zone["high"]) - max(representative["low"], zone["low"]))
            ratio = overlap / max(min(representative["width"], zone["width"]), 1e-12)
            if ratio >= threshold:
                matched = group
                break
        if matched is None:
            groups.append([zone])
        else:
            matched.append(zone)
    results = []
    for group in groups:
        representative = sorted(group, key=lambda item: (-item["overlap_ratio"], -item["final_confluence_score"], -item["strength_score"], item["width"], item["zone_id"]))[0]
        representative = dict(representative)
        representative["parent_daily_zone_ids"] = sorted({value for item in group for value in item["parent_daily_zone_ids"]})
        representative["parent_weekly_zone_ids"] = sorted({value for item in group for value in item["parent_weekly_zone_ids"]})
        results.append(representative)
    return results


def _dedupe_members(members: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item in sorted(members, key=lambda value: (value["price"], value["source"], value["member_id"])):
        key = str(item["member_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _cluster_radius(center: float, atr: float, cfg_key: str) -> float:
    cfg = CONFIG["support_resistance_v1"][cfg_key]
    return min(max(center * float(cfg["base_price_fraction"]), atr * float(cfg["atr_multiplier"])), center * float(cfg["radius_cap_fraction"]))


def _center_weight(member: dict[str, Any]) -> float:
    return max(float(member.get("weight") or 0.0), 0.1)


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = sorted((float(value), max(float(weight), 0.0)) for value, weight in zip(values, weights))
    if not pairs:
        return 0.0
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return float(np.median([value for value, _ in pairs]))
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= total / 2.0:
            return value
    return pairs[-1][0]


def _confluence_class(score: float, family_count: int) -> str:
    thresholds = CONFIG["support_resistance_v1"]["confluence_thresholds"]
    if score >= float(thresholds["very_high"]) and family_count >= 3:
        return "VERY_HIGH"
    if score >= float(thresholds["high"]) and family_count >= 2:
        return "HIGH"
    if score >= float(thresholds["medium"]):
        return "MEDIUM"
    return "LOW"


def _class_rank(value: str) -> int:
    return {"LOW": 0, "WEAK": 0, "MEDIUM": 1, "MODERATE": 1, "HIGH": 2, "STRONG": 2, "VERY_HIGH": 3, "VERY_STRONG": 3}.get(str(value), 0)


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

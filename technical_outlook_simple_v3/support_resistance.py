from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, CONFIG_VERSION, MODEL_VERSION, SR_ENGINE_VERSION
from .strength import interaction_strength


FAMILIES = ("SWING_STRUCTURE", "VOLUME_ACCEPTANCE", "FIBONACCI", "MOVING_AVERAGE")


def key_point_class(score: float) -> str:
    value = float(score)
    if value > 150:
        return "HIGH"
    if value >= 75:
        return "MID"
    return "LOW"


def confluence_class(family_count: int) -> str:
    """Compatibility alias; SIMPLE v3 now exposes score classes instead."""
    return {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "VERY_HIGH"}.get(int(family_count), "LOW")


def assign_role(low: float, high: float, current_price: float) -> str:
    if high < current_price:
        return "SUPPORT"
    if low > current_price:
        return "RESISTANCE"
    return "TESTING"


def build_support_resistance(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
    fibonacci: dict[str, Any],
    *,
    timeframe: str,
    as_of: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    timeframe = str(timeframe).upper()
    current = float(frame.iloc[-1]["close"])
    atr = _finite(frame.iloc[-1].get("atr14")) or current * (0.04 if timeframe == "WEEKLY" else 0.02)
    members = _candidate_members(frame, pivots, profile, fibonacci, timeframe=timeframe, as_of=as_of)
    clusters = deterministic_clusters(members, current=current, atr=atr, timeframe=timeframe)
    zones = [_zone_from_members(cluster, frame, current=current, atr=atr, timeframe=timeframe, as_of=as_of) for cluster in clusters]
    result = [zone for zone in zones if zone is not None]
    result.sort(key=lambda zone: (-float(zone["key_point_score"]), abs(float(zone["distance_pct"])), str(zone["zone_id"])))
    return result


def visible_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [zone for zone in zones if not zone.get("hidden_by_60pct_filter")]


def deterministic_clusters(
    members: list[dict[str, Any]],
    *,
    current: float,
    atr: float,
    timeframe: str,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(members, key=_member_sort_key)
    points = [item for item in ordered if not _is_interval(item)]
    intervals = [item for item in ordered if _is_interval(item)]
    clusters: list[list[dict[str, Any]]] = []
    for item in points:
        eligible: list[tuple[float, int, list[dict[str, Any]]]] = []
        for index, cluster in enumerate(clusters):
            trial = cluster + [item]
            center = _cluster_center(trial)
            radius = _cluster_radius(current, atr, timeframe)
            if all(_distance_to_member(center, member) <= radius + 1e-12 for member in trial):
                eligible.append((_distance_to_member(center, item), index, trial))
        if not eligible:
            clusters.append([item])
        else:
            _, index, trial = min(eligible, key=lambda value: (value[0], value[1]))
            clusters[index] = _stabilize(trial, current, atr, timeframe)
    for item in intervals:
        eligible: list[tuple[int, int, float, int, list[dict[str, Any]]]] = []
        for index, cluster in enumerate(clusters):
            trial = cluster + [item]
            center = _cluster_center(trial)
            radius = _cluster_radius(current, atr, timeframe)
            if all(_distance_to_member(center, member) <= radius + 1e-12 for member in trial):
                eligible.append((-len({str(member["family"]) for member in trial}), -len({str(member["family"]) for member in cluster}), _distance_to_member(_cluster_center(cluster), item), index, trial))
        if not eligible:
            clusters.append([item])
        else:
            _, _, _, index, trial = min(eligible)
            clusters[index] = trial
    changed = True
    while changed:
        changed = False
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                joined = clusters[left] + clusters[right]
                trial = _stabilize(joined, current, atr, timeframe)
                if len(trial) != len(joined):
                    continue
                center = _cluster_center(trial)
                radius = _cluster_radius(current, atr, timeframe)
                if all(_distance_to_member(center, item) <= radius + 1e-12 for item in trial):
                    clusters[left] = trial
                    del clusters[right]
                    changed = True
                    break
            if changed:
                break
    return [sorted(cluster, key=lambda item: (item["price"], item["member_id"])) for cluster in clusters if cluster]


def _candidate_members(frame: pd.DataFrame, pivots: list[dict[str, Any]], profile: dict[str, Any], fibonacci: dict[str, Any], *, timeframe: str, as_of: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    swing_source = "weekly_swing" if timeframe == "WEEKLY" else "daily_swing"
    swing_member_score = float(CONFIG["key_point_scores"]["swing_structure"]["one"])
    for pivot in pivots:
        if pivot.get("status") != "CONFIRMED":
            continue
        members.append(_member(
            price=float(pivot["price"]), family="SWING_STRUCTURE", source=swing_source,
            score=swing_member_score, confirmation_time=pivot.get("confirmation_time"),
            metadata={"pivot_id": pivot.get("pivot_id"), "kind": pivot.get("kind"), "pivot": pivot},
        ))
    if profile.get("status") == "AVAILABLE":
        if profile.get("poc") is not None:
            poc_zone = profile.get("poc_zone") or {}
            members.append(_member(
                price=float(profile["poc"]), family="VOLUME_ACCEPTANCE", source="poc",
                score=float(CONFIG["key_point_scores"]["volume_acceptance"]["poc"]),
                confirmation_time=as_of, low=_finite(poc_zone.get("low")), high=_finite(poc_zone.get("high")),
                metadata={"profile_start": profile.get("profile_start"), "profile_end": profile.get("profile_end"), "volume_type": "POC"},
            ))
        for peak in profile.get("local_peaks") or profile.get("hvns") or []:
            members.append(_member(
                price=float(peak["center"]), family="VOLUME_ACCEPTANCE", source="local_volume_peak",
                score=float(CONFIG["key_point_scores"]["volume_acceptance"]["local_peak"]),
                confirmation_time=as_of, low=_finite(peak.get("low")), high=_finite(peak.get("high")), metadata=peak,
            ))
    for framework_key in ("strategic", "tactical"):
        framework = fibonacci.get(framework_key) or {}
        seen_types: set[str] = set()
        for item in framework.get("levels") or []:
            ratio = str(item.get("type"))
            if ratio in seen_types:
                continue
            seen_types.add(ratio)
            source = f"{framework_key}_{ratio}"
            score = _fib_score(framework_key, ratio)
            members.append(_member(
                price=float(item.get("price") or ((float(item.get("low")) + float(item.get("high"))) / 2.0)),
                family="FIBONACCI", source=source, score=score,
                confirmation_time=(framework.get("anchor") or {}).get("confirmation_time"),
                low=_finite(item.get("low")), high=_finite(item.get("high")),
                metadata={"fib_direction": fibonacci.get("direction"), "fib_type": ratio, "fib_role": framework_key.upper()},
            ))
    row = frame.iloc[-1]
    moving_average_scores = CONFIG["key_point_scores"]["moving_average"]
    for window, score in ((100, moving_average_scores["sma100"]), (200, moving_average_scores["sma200"])):
        value = _finite(row.get(f"sma{window}"))
        if value is not None:
            members.append(_member(price=value, family="MOVING_AVERAGE", source=f"sma{window}", score=score, confirmation_time=as_of, metadata={"window": window}))
    return members


def _fib_score(framework: str, ratio: str) -> float:
    key = f"{framework}_{ratio}"
    return float(CONFIG["key_point_scores"]["fibonacci"].get(key, 0.0))


def _member(*, price: float, family: str, source: str, score: float, confirmation_time: Any, metadata: dict[str, Any], low: float | None = None, high: float | None = None) -> dict[str, Any]:
    lower = float(price if low is None else low)
    upper = float(price if high is None else high)
    if lower > upper:
        lower, upper = upper, lower
    identity = {"price": round(price, 8), "low": round(lower, 8), "high": round(upper, 8), "family": family, "source": source, "confirmation_time": confirmation_time, "metadata": metadata}
    return {"member_id": _stable_id(identity), "price": float(price), "family": family, "source": source, "score": float(score), "weight": float(score), "low": lower, "high": upper, "confirmation_time": confirmation_time, "metadata": metadata}


def _zone_from_members(members: list[dict[str, Any]], frame: pd.DataFrame, *, current: float, atr: float, timeframe: str, as_of: str) -> dict[str, Any] | None:
    if not members:
        return None
    center = _cluster_center(members)
    radius = _cluster_radius(current, atr, timeframe)
    members = [item for item in members if _distance_to_member(center, item) <= radius + 1e-12 or _is_interval(item)]
    if not members:
        return None
    center = _cluster_center(members)
    deviations = [_distance_to_member(center, item) for item in members]
    mad = _weighted_median(deviations, [_center_weight(item) for item in members])
    cfg = CONFIG["clustering"]["weekly" if timeframe == "WEEKLY" else "daily"]
    min_half = max(center * float(cfg["min_width_fraction"]), atr * float(cfg["min_width_atr_multiplier"])) / 2.0
    max_half = current * float(cfg["max_total_width_fraction"]) / 2.0
    raw_half = max(float(CONFIG["clustering"]["mad_multiplier"]) * mad, max(deviations, default=0.0), min_half)
    half = min(raw_half, max_half) if max_half > 0 else raw_half
    native_low = min(_member_bounds(item)[0] for item in members)
    native_high = max(_member_bounds(item)[1] for item in members)
    low, high = min(center - half, native_low), max(center + half, native_high)
    role = assign_role(low, high, current)
    scores = _family_max_scores(members)
    total = float(sum(scores.values()))
    structural_times = [item.get("confirmation_time") for item in members if item.get("confirmation_time")]
    valid_from = max(structural_times) if structural_times else as_of
    interactions, strength = interaction_strength(frame, low, high, role, valid_from=valid_from, timeframe=timeframe)
    hidden = bool(high < current * float(CONFIG["display"]["lower_cutoff_fraction"]))
    pivot_count = sum(1 for item in members if item["family"] == "SWING_STRUCTURE")
    key_class = key_point_class(total)
    family_sources = {family: sorted({str(item["source"]) for item in members if item["family"] == family}) for family in scores}
    fib_metadata = [item.get("metadata") or {} for item in members if item["family"] == "FIBONACCI"]
    member_ids = sorted(item["member_id"] for item in members)
    zone_id = _stable_id({"as_of": as_of, "timeframe": timeframe, "bounds": [round(low, 8), round(high, 8)], "members": member_ids})
    return {
        "zone_id": zone_id, "as_of": as_of, "timeframe": timeframe, "role": role,
        "low": low, "high": high, "lower_bound": low, "upper_bound": high, "center": center, "width": high - low,
        "cluster_radius": radius, "weighted_mad": mad, "source_families": sorted(scores), "family_count": len(scores),
        "independent_family_count": len(scores), "family_scores": scores, "family_sources": family_sources,
        "member_sources": sorted({item["source"] for item in members}), "source_members": members, "sources": sorted(scores),
        "pivot_score": scores.get("SWING_STRUCTURE", 0.0), "sma_score": scores.get("MOVING_AVERAGE", 0.0),
        "volume_score": scores.get("VOLUME_ACCEPTANCE", 0.0), "fibonacci_score": scores.get("FIBONACCI", 0.0),
        "key_point_score": total, "total_score": total, "quality_score": total,
        "key_point_class": key_class, "class": key_class, "confluence_class": key_class, "confluence": key_class,
        "pivot_count": pivot_count, "drivers": _drivers(members),
        "source_prices": [{"source": item["source"], "price": item["price"]} for item in members],
        "fib_direction": next((meta.get("fib_direction") for meta in fib_metadata if meta.get("fib_direction")), None),
        "fib_types": sorted({meta.get("fib_type") for meta in fib_metadata if meta.get("fib_type")}),
        "confirmed_touch_count": strength["confirmed_touch_count"], "pending_touch_count": strength["pending_touch_count"],
        "failed_touch_count": strength["failed_touch_count"], "reaction_statistics": strength["reaction_statistics"],
        "strength_components": strength["strength_components"], "strength_score": strength["strength_score"], "strength_class": strength["strength_class"],
        "interactions": interactions, "distance_pct": center / current - 1.0 if current else None,
        "visible_on_chart": not hidden, "hidden_by_60pct_filter": hidden, "valid_from": valid_from,
        "model_version": MODEL_VERSION, "config_version": CONFIG_VERSION, "sr_engine_version": SR_ENGINE_VERSION,
    }


def _family_max_scores(members: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for family in FAMILIES:
        values = [float(item.get("score") or 0.0) for item in members if item.get("family") == family]
        if values:
            if family == "SWING_STRUCTURE":
                pivot_count = len(values)
                swing = CONFIG["key_point_scores"]["swing_structure"]
                scores[family] = 0.0 if pivot_count == 0 else float(swing["one"]) if pivot_count == 1 else float(swing["two"]) if pivot_count == 2 else float(swing["three_plus"])
            else:
                scores[family] = max(values)
    return {key: value for key, value in scores.items() if value > 0}


def _drivers(members: list[dict[str, Any]]) -> list[str]:
    drivers: list[str] = []
    pivot_count = sum(1 for item in members if item["family"] == "SWING_STRUCTURE")
    if pivot_count:
        drivers.append(f"{pivot_count} confirmed pivots")
    for family in FAMILIES:
        if family == "SWING_STRUCTURE":
            continue
        sources = sorted({str(item["source"]) for item in members if item["family"] == family})
        drivers.extend(sources)
    return drivers


def _family_scores(members: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Backwards-compatible diagnostic helper using member weights."""
    grouped: dict[str, dict[str, float]] = {}
    for item in members:
        family, source = str(item["family"]), str(item["source"])
        grouped.setdefault(family, {})[source] = max(float(item.get("weight", item.get("score", 0.0))), grouped.setdefault(family, {}).get(source, 0.0))
    scores = {family: round(max(values.values()) + min(0.25 * max(0, len(values) - 1), 0.5), 4) for family, values in grouped.items()}
    return scores, {family: sorted(values) for family, values in grouped.items()}


def _stabilize(members: list[dict[str, Any]], current: float, atr: float, timeframe: str) -> list[dict[str, Any]]:
    result = list(members)
    while result:
        center = _cluster_center(result)
        retained = [item for item in result if _distance_to_member(center, item) <= _cluster_radius(current, atr, timeframe) + 1e-12 or _is_interval(item)]
        if len(retained) == len(result):
            return retained
        result = retained
    return []


def _cluster_radius(current: float, atr: float, timeframe: str) -> float:
    cfg = CONFIG["clustering"]["weekly" if str(timeframe).upper() == "WEEKLY" else "daily"]
    return min(max(current * float(cfg["base_price_fraction"]), atr * float(cfg["atr_multiplier"])), current * float(cfg["radius_cap_fraction"]))


def _member_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (*_member_bounds(item), float(item["price"]), str(item["family"]), str(item["source"]), str(item["member_id"]))


def _is_interval(item: dict[str, Any]) -> bool:
    low, high = _member_bounds(item)
    return high - low > 1e-12


def _member_bounds(item: dict[str, Any]) -> tuple[float, float]:
    price = float(item["price"])
    low = _finite(item.get("low"))
    high = _finite(item.get("high"))
    return (min(price if low is None else low, price if high is None else high), max(price if low is None else low, price if high is None else high))


def _distance_to_member(value: float, item: dict[str, Any]) -> float:
    low, high = _member_bounds(item)
    return low - value if value < low else value - high if value > high else 0.0


def _cluster_center(members: list[dict[str, Any]]) -> float:
    prices = [float(item["price"]) for item in members]
    weights = [_center_weight(item) for item in members]
    reference = _weighted_median(prices, weights)
    candidates = {reference}
    for item in members:
        low, high = _member_bounds(item)
        candidates.update((low, float(item["price"]), high))
    return float(min(candidates, key=lambda value: (sum(_center_weight(item) * _distance_to_member(value, item) for item in members), abs(value - reference), value)))


def _center_weight(item: dict[str, Any]) -> float:
    return max(float(item.get("weight") or item.get("score") or 0.0), 1e-9)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    cutoff, running = sum(weights) / 2.0, 0.0
    for value, weight in ordered:
        running += weight
        if running >= cutoff:
            return float(value)
    return float(ordered[-1][0])


def _stable_id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()[:24]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

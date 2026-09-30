from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, CONFIG_VERSION, MODEL_VERSION, SR_ENGINE_VERSION
from .strength import interaction_strength


FAMILIES = ("SWING_STRUCTURE", "VOLUME_ACCEPTANCE", "FIBONACCI", "MOVING_AVERAGE")


def confluence_class(family_count: int) -> str:
    return "VERY_HIGH" if family_count >= 4 else "HIGH" if family_count == 3 else "MEDIUM" if family_count == 2 else "LOW"


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
    zones = [
        _zone_from_members(cluster, frame, current=current, atr=atr, timeframe=timeframe, as_of=as_of)
        for cluster in clusters
    ]
    result = [zone for zone in zones if zone is not None]
    result.sort(key=lambda zone: (-int(zone["family_count"]), -float(zone["quality_score"]), abs(float(zone["distance_pct"])), str(zone["zone_id"])))
    return result


def visible_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [zone for zone in zones if bool(zone.get("visible_on_chart"))]


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
    # Establish point-based technical clusters first. A broad volume node may
    # confirm an existing price area, but must never pull a point away from an
    # otherwise valid Swing/Fibonacci/MA confluence cluster.
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

    # Attach POC/HVN intervals to the strongest compatible point cluster. The
    # interval participates across its full price node while remaining one
    # independent Volume Acceptance observation.
    for item in intervals:
        eligible_intervals: list[tuple[int, int, float, int, list[dict[str, Any]]]] = []
        for index, cluster in enumerate(clusters):
            trial = cluster + [item]
            center = _cluster_center(trial)
            radius = _cluster_radius(current, atr, timeframe)
            if not all(_distance_to_member(center, member) <= radius + 1e-12 for member in trial):
                continue
            trial_families = len({str(member["family"]) for member in trial})
            cluster_families = len({str(member["family"]) for member in cluster})
            eligible_intervals.append(
                (-trial_families, -cluster_families, _distance_to_member(_cluster_center(cluster), item), index, trial)
            )
        if not eligible_intervals:
            clusters.append([item])
        else:
            _, _, _, index, trial = min(eligible_intervals)
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


def _member_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (*_member_bounds(item), float(item["price"]), item["family"], item["source"], item["member_id"])


def _is_interval(item: dict[str, Any]) -> bool:
    low, high = _member_bounds(item)
    return high - low > 1e-12


def _candidate_members(
    frame: pd.DataFrame,
    pivots: list[dict[str, Any]],
    profile: dict[str, Any],
    fibonacci: dict[str, Any],
    *,
    timeframe: str,
    as_of: str,
) -> list[dict[str, Any]]:
    weights = CONFIG["family_quality"]["weights"]
    members: list[dict[str, Any]] = []
    swing_source = "weekly_swing" if timeframe == "WEEKLY" else "daily_swing"
    for pivot in pivots:
        if pivot.get("status") != "CONFIRMED":
            continue
        members.append(_member(
            price=float(pivot["price"]), family="SWING_STRUCTURE", source=swing_source,
            weight=float(weights[swing_source]), confirmation_time=pivot.get("confirmation_time"),
            metadata={"pivot_id": pivot.get("pivot_id"), "kind": pivot.get("kind")},
        ))
    if profile.get("status") == "AVAILABLE":
        if profile.get("poc") is not None:
            poc_zone = profile.get("poc_zone") or {}
            members.append(_member(
                price=float(profile["poc"]), family="VOLUME_ACCEPTANCE", source="poc",
                weight=float(weights["poc"]), confirmation_time=as_of,
                metadata={"profile_start": profile.get("profile_start"), "profile_end": profile.get("profile_end")},
                low=_finite(poc_zone.get("low")), high=_finite(poc_zone.get("high")),
            ))
        for index, hvn in enumerate(profile.get("hvns") or []):
            members.append(_member(
                price=float(hvn["center"]), family="VOLUME_ACCEPTANCE", source="hvn",
                weight=float(weights["hvn"]), confirmation_time=as_of,
                metadata={"hvn_index": index, **hvn},
                low=_finite(hvn.get("low")), high=_finite(hvn.get("high")),
            ))
    for item in fibonacci.get("retracements") or []:
        source = str(item["source"])
        base_weight = float(weights.get(source, 1.0))
        members.append(_member(
            price=float(item["price"]), family="FIBONACCI", source=source,
            weight=base_weight * float(item.get("quality_multiplier", 1.0)),
            confirmation_time=(fibonacci.get("end_anchor") or {}).get("confirmation_time") or (fibonacci.get("start_anchor") or {}).get("confirmation_time"),
            metadata={"ratio": item.get("ratio"), "framework_status": fibonacci.get("status")},
        ))
    row = frame.iloc[-1]
    for window in (50, 100, 200):
        value = _finite(row.get(f"sma{window}"))
        if value is not None:
            source = f"sma{window}"
            members.append(_member(
                price=value, family="MOVING_AVERAGE", source=source,
                weight=float(weights[source]), confirmation_time=as_of, metadata={"window": window},
            ))
    return members


def _member(
    *, price: float, family: str, source: str, weight: float,
    confirmation_time: Any, metadata: dict[str, Any],
    low: float | None = None, high: float | None = None,
) -> dict[str, Any]:
    lower = float(price if low is None else low)
    upper = float(price if high is None else high)
    if lower > upper:
        lower, upper = upper, lower
    identity = {
        "price": round(price, 8), "low": round(lower, 8), "high": round(upper, 8),
        "family": family, "source": source, "confirmation_time": confirmation_time, "metadata": metadata,
    }
    return {
        "member_id": _stable_id(identity), "price": float(price), "family": family,
        "low": lower, "high": upper,
        "source": source, "weight": float(weight), "confirmation_time": confirmation_time,
        "metadata": metadata,
    }


def _zone_from_members(
    members: list[dict[str, Any]],
    frame: pd.DataFrame,
    *, current: float, atr: float, timeframe: str, as_of: str,
) -> dict[str, Any] | None:
    if not members:
        return None
    center = _cluster_center(members)
    radius = _cluster_radius(current, atr, timeframe)
    members = [item for item in members if _distance_to_member(center, item) <= radius + 1e-12]
    if not members:
        return None
    center = _cluster_center(members)
    deviations = [_distance_to_member(center, item) for item in members]
    mad = _weighted_median(deviations, [_center_weight(item) for item in members])
    cfg = CONFIG["clustering"]["weekly" if timeframe == "WEEKLY" else "daily"]
    min_half = max(center * float(cfg["min_width_fraction"]), atr * float(cfg["min_width_atr_multiplier"])) / 2.0
    max_half = current * float(cfg["max_total_width_fraction"]) / 2.0
    raw_half = max(float(CONFIG["clustering"]["mad_multiplier"]) * mad, max(deviations, default=0.0))
    half = min(max(raw_half, min_half), max_half)
    low, high = center - half, center + half
    role = assign_role(low, high, current)
    family_scores, family_sources = _family_scores(members)
    family_count = len(family_scores)
    quality = float(sum(family_scores.values()))
    structural_times = [
        item.get("confirmation_time") for item in members
        if item.get("family") in {"SWING_STRUCTURE", "FIBONACCI"} and item.get("confirmation_time")
    ]
    valid_from = max(structural_times) if structural_times else as_of
    interactions, strength = interaction_strength(frame, low, high, role, valid_from=valid_from, timeframe=timeframe)
    cutoff = current * float(CONFIG["display"]["lower_cutoff_fraction"])
    hidden = bool(high < cutoff)
    confluence = confluence_class(family_count)
    visible = bool(confluence in CONFIG["display"]["classes"] and not hidden)
    member_ids = sorted(item["member_id"] for item in members)
    zone_id = _stable_id({"as_of": as_of, "timeframe": timeframe, "bounds": [round(low, 8), round(high, 8)], "members": member_ids})
    return {
        "zone_id": zone_id,
        "as_of": as_of,
        "timeframe": timeframe,
        "role": role,
        "low": low,
        "high": high,
        "lower_bound": low,
        "upper_bound": high,
        "center": center,
        "width": high - low,
        "cluster_radius": radius,
        "weighted_mad": mad,
        "family_count": family_count,
        "independent_family_count": family_count,
        "confluence_class": confluence,
        "confluence": confluence,
        "quality_score": round(quality, 4),
        "family_scores": family_scores,
        "source_families": sorted(family_scores),
        "family_sources": family_sources,
        "member_sources": sorted({item["source"] for item in members}),
        "source_members": members,
        "sources": sorted(family_scores),
        "confirmed_touch_count": strength["confirmed_touch_count"],
        "pending_touch_count": strength["pending_touch_count"],
        "failed_touch_count": strength["failed_touch_count"],
        "reaction_statistics": strength["reaction_statistics"],
        "strength_components": strength["strength_components"],
        "strength_score": strength["strength_score"],
        "strength_class": strength["strength_class"],
        "interactions": interactions,
        "distance_pct": center / current - 1.0 if current else None,
        "visible_on_chart": visible,
        "hidden_by_60pct_filter": hidden,
        "valid_from": valid_from,
        "model_version": MODEL_VERSION,
        "config_version": CONFIG_VERSION,
        "sr_engine_version": SR_ENGINE_VERSION,
    }


def _family_scores(members: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[str]]]:
    grouped: dict[str, dict[str, float]] = {}
    for item in members:
        family = str(item["family"])
        source = str(item["source"])
        grouped.setdefault(family, {})[source] = max(float(item["weight"]), grouped.setdefault(family, {}).get(source, 0.0))
    increment = float(CONFIG["family_quality"]["breadth_increment"])
    cap = float(CONFIG["family_quality"]["breadth_cap"])
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for family, source_weights in grouped.items():
        ordered = sorted(source_weights.items(), key=lambda value: (-value[1], value[0]))
        scores[family] = round(ordered[0][1] + min(increment * max(0, len(ordered) - 1), cap), 4)
        sources[family] = sorted(source_weights)
    return scores, sources


def _stabilize(members: list[dict[str, Any]], current: float, atr: float, timeframe: str) -> list[dict[str, Any]]:
    result = list(members)
    while result:
        center = _cluster_center(result)
        radius = _cluster_radius(current, atr, timeframe)
        retained = [item for item in result if _distance_to_member(center, item) <= radius + 1e-12]
        if len(retained) == len(result):
            return retained
        result = retained
    return []


def _cluster_radius(current: float, atr: float, timeframe: str) -> float:
    cfg = CONFIG["clustering"]["weekly" if timeframe == "WEEKLY" else "daily"]
    return min(
        max(current * float(cfg["base_price_fraction"]), atr * float(cfg["atr_multiplier"])),
        current * float(cfg["radius_cap_fraction"]),
    )


def _center_weight(item: dict[str, Any]) -> float:
    return max(float(item.get("weight") or 0.0), 1e-9)


def _member_bounds(item: dict[str, Any]) -> tuple[float, float]:
    price = float(item["price"])
    low = _finite(item.get("low"))
    high = _finite(item.get("high"))
    lower = price if low is None else low
    upper = price if high is None else high
    return (min(lower, upper), max(lower, upper))


def _distance_to_member(value: float, item: dict[str, Any]) -> float:
    low, high = _member_bounds(item)
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _cluster_center(members: list[dict[str, Any]]) -> float:
    """Return the deterministic weighted 1-D median of points and price intervals."""
    prices = [float(item["price"]) for item in members]
    weights = [_center_weight(item) for item in members]
    reference = _weighted_median(prices, weights)
    candidates = {reference}
    for item in members:
        low, high = _member_bounds(item)
        candidates.update((low, float(item["price"]), high))

    def objective(value: float) -> float:
        return sum(_center_weight(item) * _distance_to_member(value, item) for item in members)

    return float(min(candidates, key=lambda value: (objective(value), abs(value - reference), value)))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    cutoff = sum(weight for _, weight in ordered) / 2.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= cutoff:
            return float(value)
    return float(ordered[-1][0])


def _stable_id(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

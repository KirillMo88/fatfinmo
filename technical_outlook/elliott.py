from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG


PATTERN_SPECS: dict[str, dict[str, Any]] = {
    "IMPULSE": {"points": 6, "labels": ["0", "1", "2", "3", "4", "5"], "internal": [5, 3, 5, 3, 5]},
    "LEADING_DIAGONAL": {"points": 6, "labels": ["0", "1", "2", "3", "4", "5"], "internal": [5, 3, 5, 3, 5]},
    "ENDING_DIAGONAL": {"points": 6, "labels": ["0", "1", "2", "3", "4", "5"], "internal": [3, 3, 3, 3, 3]},
    "ZIGZAG": {"points": 4, "labels": ["S", "A", "B", "C"], "internal": [5, 3, 5]},
    "FLAT": {"points": 4, "labels": ["S", "A", "B", "C"], "internal": [3, 3, 5]},
    "EXPANDED_FLAT": {"points": 4, "labels": ["S", "A", "B", "C"], "internal": [3, 3, 5]},
    "TRIANGLE": {"points": 6, "labels": ["S", "A", "B", "C", "D", "E"], "internal": [3, 3, 3, 3, 3]},
    "WXY": {"points": 7, "labels": ["S", "W1", "W2", "X", "Y1", "Y2", "Y"], "internal": [3, 3, 3, 3, 3, 3]},
    "WXYXZ": {"points": 10, "labels": ["S", "W1", "W2", "X1", "Y1", "Y2", "X2", "Z1", "Z2", "Z"], "internal": [3] * 9},
}


def analyze_elliott_hierarchy(
    major_pivots: list[dict[str, Any]],
    intermediate_pivots: list[dict[str, Any]],
    minor_pivots: list[dict[str, Any]],
    monthly_momentum: dict[str, Any],
    weekly_momentum: dict[str, Any],
    divergences: list[dict[str, Any]],
    volume: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    major_universe = generate_candidates(
        major_pivots,
        "MAJOR",
        intermediate_pivots,
        monthly_momentum,
        divergences,
        volume,
        parent=None,
    )
    major_primary, major_alternative, major_confidence = select_primary_alternative(major_universe)
    intermediate_universe = generate_candidates(
        intermediate_pivots,
        "INTERMEDIATE",
        minor_pivots,
        weekly_momentum,
        divergences,
        volume,
        parent=major_primary if major_primary.get("credible") else None,
    )
    primary, alternative, confidence = select_primary_alternative(intermediate_universe)
    parent_map = {
        "major_primary_id": major_primary.get("candidate_id"),
        "major_alternative_id": major_alternative.get("candidate_id"),
        "intermediate_primary_id": primary.get("candidate_id"),
        "intermediate_alternative_id": alternative.get("candidate_id"),
        "primary_compatibility": primary.get("parent_consistency"),
        "alternative_compatibility": alternative.get("parent_consistency"),
    }
    change = _change_from_previous(previous, primary, alternative)
    return {
        "candidate_universe_summary": {
            "major_valid_candidates": len(major_universe),
            "intermediate_valid_candidates": len(intermediate_universe),
            "major_pattern_counts": _pattern_counts(major_universe),
            "intermediate_pattern_counts": _pattern_counts(intermediate_universe),
            "search_capped_at": int(CONFIG["elliott_search"]["max_candidates_per_degree"]),
        },
        "major_primary": major_primary,
        "major_alternative": major_alternative,
        "major_confidence": major_confidence,
        "primary": primary,
        "alternative": alternative,
        "confidence": confidence,
        "current_wave_state": primary.get("wave_state", "POTENTIAL"),
        "parent_child_map": parent_map,
        "complexity_penalties": CONFIG["elliott_complexity_penalties"],
        "change_from_prior_snapshot": change,
        "candidate_universe": {
            "major": [_compact_candidate(item) for item in major_universe[:10]],
            "intermediate": [_compact_candidate(item) for item in intermediate_universe[:15]],
        },
    }


def generate_candidates(
    pivots: list[dict[str, Any]],
    degree: str,
    lower_degree_pivots: list[dict[str, Any]],
    momentum: dict[str, Any],
    divergences: list[dict[str, Any]],
    volume: dict[str, Any],
    *,
    parent: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if len(pivots) < 3:
        return []
    search_limit = int(CONFIG["elliott_search"]["max_start_pivots"])
    values = pivots[-search_limit:]
    candidates: list[dict[str, Any]] = []
    for pattern, spec in PATTERN_SPECS.items():
        point_count = int(spec["points"])
        if len(values) >= point_count:
            for start in range(max(0, len(values) - point_count - 4), len(values) - point_count + 1):
                points = values[start : start + point_count]
                valid, checks = validate_pattern(pattern, points)
                if valid:
                    candidates.append(
                        score_candidate(pattern, degree, points, checks, lower_degree_pivots, momentum, divergences, volume, parent)
                    )
        confirmed = [item for item in values if item.get("status") == "CONFIRMED"]
        if len(confirmed) >= point_count - 1:
            points = confirmed[-(point_count - 1) :]
            valid, checks = validate_developing_prefix(pattern, points)
            if valid:
                candidates.append(
                    score_candidate(
                        pattern,
                        degree,
                        points,
                        checks,
                        lower_degree_pivots,
                        momentum,
                        divergences,
                        volume,
                        parent,
                        open_wave=True,
                    )
                )
    deduped: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    latest_bar_index = max(int(item["bar_index"]) for item in values)
    for candidate in candidates:
        bars_old = max(0, latest_bar_index - int(candidate["end_bar_index"]))
        recency_penalty = min(
            float(CONFIG["elliott_search"]["recency_penalty_cap"]),
            bars_old * float(CONFIG["elliott_search"]["recency_penalty_per_bar"]),
        )
        candidate["recency_penalty"] = recency_penalty
        candidate["score"] = round(max(0.0, float(candidate["score"]) - recency_penalty), 2)
        candidate["credible"] = candidate["score"] >= 35.0
        key = (candidate["pattern"], candidate["start_time"], candidate["end_time"], candidate["open_wave"])
        current = deduped.get(key)
        if current is None or candidate["score"] > current["score"]:
            deduped[key] = candidate
    ranked = sorted(deduped.values(), key=lambda item: (item["score"], item["end_time"]), reverse=True)
    return ranked[: int(CONFIG["elliott_search"]["max_candidates_per_degree"])]


def validate_pattern(pattern: str, points: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    if len(points) != int(PATTERN_SPECS[pattern]["points"]):
        return False, [_check("point_count", False, len(points))]
    if not _alternating(points):
        return False, [_check("alternating_pivots", False, None)]
    if pattern == "IMPULSE":
        return validate_impulse(points)
    if pattern in {"LEADING_DIAGONAL", "ENDING_DIAGONAL"}:
        return validate_diagonal(points, ending=pattern == "ENDING_DIAGONAL")
    if pattern == "ZIGZAG":
        return validate_zigzag(points)
    if pattern == "FLAT":
        return validate_flat(points, expanded=False)
    if pattern == "EXPANDED_FLAT":
        return validate_flat(points, expanded=True)
    if pattern == "TRIANGLE":
        return validate_triangle(points)
    if pattern == "WXY":
        return validate_compound(points, triple=False)
    if pattern == "WXYXZ":
        return validate_compound(points, triple=True)
    return False, [_check("supported_pattern", False, pattern)]


def validate_developing_prefix(pattern: str, points: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    expected = int(PATTERN_SPECS[pattern]["points"]) - 1
    if len(points) != expected or not _alternating(points):
        return False, [_check("developing_prefix", False, len(points))]
    prices, direction = _oriented_prices(points)
    checks = [_check("known_legs_logical", all((prices[index + 1] - prices[index]) * (1 if index % 2 == 0 else -1) > 0 for index in range(len(prices) - 1)), prices)]
    if pattern == "IMPULSE" and len(points) >= 5:
        checks.extend([
            _check("wave2_not_beyond_origin", prices[2] > prices[0], prices[2]),
            _check("wave4_no_wave1_overlap", prices[4] > prices[1], [prices[4], prices[1]]),
        ])
    if pattern in {"LEADING_DIAGONAL", "ENDING_DIAGONAL"} and len(points) >= 5:
        checks.append(_check("diagonal_overlap", prices[4] <= prices[1], [prices[4], prices[1]]))
    return all(item["passed"] for item in checks), checks


def validate_impulse(points: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    prices, _ = _oriented_prices(points)
    motive = [prices[1] - prices[0], prices[3] - prices[2], prices[5] - prices[4]]
    counter = [prices[2] - prices[1], prices[4] - prices[3]]
    checks = [
        _check("wave2_not_beyond_origin", prices[2] > prices[0], [prices[2], prices[0]]),
        _check("wave3_not_shortest", motive[1] >= min(motive[0], motive[2]), motive),
        _check("wave4_no_wave1_overlap", prices[4] > prices[1], [prices[4], prices[1]]),
        _check("motive_waves_direction", all(value > 0 for value in motive), motive),
        _check("corrective_waves_direction", all(value < 0 for value in counter), counter),
    ]
    return all(item["passed"] for item in checks), checks


def validate_diagonal(points: list[dict[str, Any]], *, ending: bool) -> tuple[bool, list[dict[str, Any]]]:
    prices, _ = _oriented_prices(points)
    motive = [prices[1] - prices[0], prices[3] - prices[2], prices[5] - prices[4]]
    counter = [prices[2] - prices[1], prices[4] - prices[3]]
    amplitudes = [abs(prices[index + 1] - prices[index]) for index in range(5)]
    convergence = amplitudes[-1] < amplitudes[0]
    checks = [
        _check("wave2_not_beyond_origin", prices[2] > prices[0], [prices[2], prices[0]]),
        _check("diagonal_motive_direction", all(value > 0 for value in motive), motive),
        _check("diagonal_corrective_direction", all(value < 0 for value in counter), counter),
        _check("diagonal_wave4_overlap", prices[4] <= prices[1], [prices[4], prices[1]]),
        _check("diagonal_geometry", convergence if ending else True, amplitudes),
    ]
    return all(item["passed"] for item in checks), checks


def validate_zigzag(points: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    prices, _ = _oriented_prices(points)
    a, b, c = prices[1] - prices[0], prices[2] - prices[1], prices[3] - prices[2]
    retracement = abs(b / max(abs(a), 1e-9))
    checks = [
        _check("a_c_same_direction", a > 0 and c > 0, [a, c]),
        _check("b_countertrend", b < 0, b),
        _check("b_below_origin", prices[2] > prices[0], retracement),
    ]
    return all(item["passed"] for item in checks), checks


def validate_flat(points: list[dict[str, Any]], *, expanded: bool) -> tuple[bool, list[dict[str, Any]]]:
    prices, _ = _oriented_prices(points)
    a = prices[1] - prices[0]
    b = prices[2] - prices[1]
    c = prices[3] - prices[2]
    b_ratio = abs(b / max(abs(a), 1e-9))
    if expanded:
        geometry = b_ratio > 1.0 and prices[3] > prices[1]
    else:
        geometry = 0.80 <= b_ratio <= 1.05 and c > 0
    checks = [
        _check("flat_b_deep_retracement", b_ratio >= 0.80, b_ratio),
        _check("expanded_geometry" if expanded else "regular_flat_geometry", geometry, [b_ratio, prices[3], prices[1]]),
    ]
    return all(item["passed"] for item in checks), checks


def validate_triangle(points: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    prices = [float(item["price"]) for item in points]
    amplitudes = [abs(prices[index + 1] - prices[index]) for index in range(5)]
    declines = sum(later < earlier for earlier, later in zip(amplitudes, amplitudes[1:]))
    highs = [prices[index] for index, item in enumerate(points) if item["kind"] == "HIGH"]
    lows = [prices[index] for index, item in enumerate(points) if item["kind"] == "LOW"]
    converging = declines >= 3 and (len(highs) < 2 or highs[-1] <= highs[0]) and (len(lows) < 2 or lows[-1] >= lows[0])
    checks = [_check("triangle_five_legs", len(amplitudes) == 5, len(amplitudes)), _check("triangle_convergence", converging, amplitudes)]
    return all(item["passed"] for item in checks), checks


def validate_compound(points: list[dict[str, Any]], *, triple: bool) -> tuple[bool, list[dict[str, Any]]]:
    prices = [float(item["price"]) for item in points]
    amplitudes = [abs(prices[index + 1] - prices[index]) for index in range(len(prices) - 1)]
    non_degenerate = all(value > 0 for value in amplitudes)
    connective_moves = amplitudes[2::3]
    main_moves = [value for index, value in enumerate(amplitudes) if index not in range(2, len(amplitudes), 3)]
    connectors_smaller = not connective_moves or np.median(connective_moves) <= np.median(main_moves)
    checks = [
        _check("compound_alternation", _alternating(points), None),
        _check("compound_non_degenerate", non_degenerate, amplitudes),
        _check("connectors_corrective_scale", connectors_smaller, [connective_moves, main_moves]),
        _check("triple_required_points" if triple else "double_required_points", len(points) == (10 if triple else 7), len(points)),
    ]
    return all(item["passed"] for item in checks), checks


def score_candidate(
    pattern: str,
    degree: str,
    points: list[dict[str, Any]],
    hard_checks: list[dict[str, Any]],
    lower_degree_pivots: list[dict[str, Any]],
    momentum: dict[str, Any],
    divergences: list[dict[str, Any]],
    volume: dict[str, Any],
    parent: dict[str, Any] | None,
    *,
    open_wave: bool = False,
) -> dict[str, Any]:
    labels = PATTERN_SPECS[pattern]["labels"]
    mapped = []
    for index, point in enumerate(points):
        mapped.append({**point, "wave_label": labels[index], "wave_status": "CONFIRMED" if point.get("status") == "CONFIRMED" else "POTENTIAL"})
    if open_wave:
        mapped.append({"wave_label": labels[len(points)], "pivot_id": None, "pivot_time": None, "price": None, "wave_status": "DEVELOPING"})
    wave_state, completion_state = _current_wave_state(mapped, open_wave)
    components = {
        "fibonacci": fibonacci_score(pattern, points),
        "momentum": momentum_score(pattern, points, momentum),
        "wave3": wave3_score(pattern, points, momentum),
        "divergence": divergence_score(pattern, divergences),
        "volume": volume_score(volume),
        "channel": channel_score(pattern, points),
        "time": time_score(points),
        "alternation": alternation_score(pattern, points),
        "internal_structure": internal_structure_score(pattern, points, lower_degree_pivots),
        "parent_consistency": parent_consistency_score(points, parent),
    }
    weighted = sum(float(components[key]) * float(CONFIG["elliott_weights"][key]) for key in components)
    complexity_penalty = float(CONFIG["elliott_complexity_penalties"][pattern])
    parent_penalty = 0.0
    if parent and components["parent_consistency"] < 40.0:
        parent_penalty = float(CONFIG["elliott_parent_inconsistency_penalty"])
    open_penalty = 4.0 if open_wave else 2.0 if wave_state == "POTENTIAL" else 0.0
    score = float(np.clip(weighted - complexity_penalty - parent_penalty - open_penalty, 0.0, 100.0))
    direction = _direction(points)
    last_price = float(points[-1]["price"])
    last_move = abs(float(points[-1]["price"]) - float(points[-2]["price"])) if len(points) >= 2 else last_price * 0.1
    invalidation = float(points[-2]["price"]) if len(points) >= 2 else None
    candidate_id = f"{degree}:{pattern}:{points[0]['pivot_id']}:{points[-1]['pivot_id']}:{'OPEN' if open_wave else 'CLOSED'}"
    return {
        "candidate_id": candidate_id,
        "pattern": pattern,
        "label": _display_label(pattern, labels[len(points)] if open_wave else labels[-1]),
        "degree": degree,
        "direction": direction,
        "credible": score >= 35.0,
        "open_wave": open_wave,
        "current_wave": labels[len(points)] if open_wave else labels[-1],
        "wave_state": wave_state,
        "completion_state": completion_state,
        "waves": mapped,
        "wave_to_pivot": {item["wave_label"]: item.get("pivot_id") for item in mapped},
        "hard_rules_passed": True,
        "hard_rule_checks": hard_checks,
        "score_components": {key: round(float(value), 2) for key, value in components.items()},
        "complexity_penalty": complexity_penalty,
        "parent_consistency_penalty": parent_penalty,
        "open_wave_penalty": open_penalty,
        "parent_consistency": "COMPATIBLE" if components["parent_consistency"] >= 60 else "AMBIGUOUS" if components["parent_consistency"] >= 40 else "INCOMPATIBLE",
        "parent_candidate_id": parent.get("candidate_id") if parent else None,
        "score": round(score, 2),
        "start_time": points[0]["pivot_time"],
        "end_time": points[-1]["pivot_time"],
        "fib_targets": [round(last_price + (1 if direction == "UP" else -1) * last_move * ratio, 4) for ratio in (0.618, 1.0, 1.618)],
        "invalidation": invalidation,
        "duration_bars": int(points[-1]["bar_index"]) - int(points[0]["bar_index"]),
        "end_bar_index": int(points[-1]["bar_index"]),
    }


def select_primary_alternative(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    credible = [item for item in candidates if item.get("credible")]
    if not credible:
        unresolved = unresolved_candidate()
        return unresolved, {**unresolved, "candidate_id": "UNRESOLVED_ALTERNATIVE"}, "LOW"
    primary = credible[0]
    alternative = next((item for item in credible[1:] if _family(item["pattern"]) != _family(primary["pattern"])), None)
    if alternative is None:
        alternative = credible[1] if len(credible) > 1 else unresolved_candidate()
    gap = float(primary["score"] - alternative.get("score", 0.0))
    cfg = CONFIG["elliott_confidence"]
    if primary["score"] >= cfg["high_min_score"] and gap >= cfg["high_gap"]:
        confidence = "HIGH"
    elif primary["score"] >= cfg["medium_min_score"] and gap >= cfg["medium_gap"]:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    primary = {**primary, "confidence": confidence, "score_gap_to_alternative": round(gap, 2)}
    alternative = {**alternative, "confidence": confidence}
    return primary, alternative, confidence


def fibonacci_score(pattern: str, points: list[dict[str, Any]]) -> float:
    prices = [float(item["price"]) for item in points]
    moves = [abs(prices[index + 1] - prices[index]) for index in range(len(prices) - 1)]
    if len(moves) < 2 or moves[0] == 0:
        return 40.0
    ratios = [move / moves[0] for move in moves[1:]]
    targets = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
    return float(np.mean([100.0 * math.exp(-4.0 * min(abs(ratio - target) for target in targets)) for ratio in ratios]))


def momentum_score(pattern: str, points: list[dict[str, Any]], momentum: dict[str, Any]) -> float:
    direction = 1.0 if _direction(points) == "UP" else -1.0
    score = float(momentum.get("score") or 0.0) * direction
    if pattern in {"FLAT", "TRIANGLE", "WXY", "WXYXZ"}:
        return float(np.clip(75.0 - abs(score) * 0.35, 20.0, 90.0))
    return float(np.clip(50.0 + score * 0.45, 10.0, 95.0))


def wave3_score(pattern: str, points: list[dict[str, Any]], momentum: dict[str, Any]) -> float:
    if pattern not in {"IMPULSE", "LEADING_DIAGONAL", "ENDING_DIAGONAL"} or len(points) < 4:
        return 50.0
    prices, _ = _oriented_prices(points)
    lengths = [prices[1] - prices[0], prices[3] - prices[2]]
    extension = lengths[1] / max(abs(lengths[0]), 1e-9)
    momentum_value = abs(float(momentum.get("score") or 0.0))
    return float(np.clip(35.0 + min(extension, 2.618) / 2.618 * 40.0 + momentum_value * 0.25, 0.0, 100.0))


def divergence_score(pattern: str, divergences: list[dict[str, Any]]) -> float:
    active = [item for item in divergences if item.get("active")]
    if not active:
        return 45.0
    if pattern in {"IMPULSE", "ENDING_DIAGONAL", "ZIGZAG", "FLAT", "EXPANDED_FLAT"}:
        return 80.0
    return 55.0


def volume_score(volume: dict[str, Any]) -> float:
    if volume.get("status") != "AVAILABLE":
        return 50.0
    return float(np.clip(volume.get("score") or 50.0, 0.0, 100.0))


def channel_score(pattern: str, points: list[dict[str, Any]]) -> float:
    if len(points) < 4:
        return 40.0
    prices = np.array([float(item["price"]) for item in points], dtype=float)
    x = np.arange(len(prices), dtype=float)
    residual = prices - np.polyval(np.polyfit(x, prices, 1), x)
    normalized = float(np.std(residual) / max(np.ptp(prices), 1e-9))
    base = 100.0 - normalized * 220.0
    if pattern == "TRIANGLE":
        base += 10.0
    return float(np.clip(base, 10.0, 95.0))


def time_score(points: list[dict[str, Any]]) -> float:
    durations = [max(1, int(second["bar_index"]) - int(first["bar_index"])) for first, second in zip(points, points[1:])]
    if not durations:
        return 40.0
    dispersion = float(np.std(durations) / max(np.mean(durations), 1.0))
    return float(np.clip(100.0 - dispersion * 75.0, 15.0, 95.0))


def alternation_score(pattern: str, points: list[dict[str, Any]]) -> float:
    if pattern != "IMPULSE" or len(points) < 5:
        return 55.0
    prices = [float(item["price"]) for item in points]
    wave1 = abs(prices[1] - prices[0])
    wave3 = abs(prices[3] - prices[2])
    wave2 = abs(prices[2] - prices[1]) / max(wave1, 1e-9)
    wave4 = abs(prices[4] - prices[3]) / max(wave3, 1e-9)
    return float(np.clip(55.0 + abs(wave2 - wave4) * 70.0, 30.0, 90.0))


def internal_structure_score(pattern: str, points: list[dict[str, Any]], lower_pivots: list[dict[str, Any]]) -> float:
    expected = PATTERN_SPECS[pattern]["internal"][: len(points) - 1]
    if not lower_pivots or not expected:
        return 35.0
    scores = []
    for index, (first, second) in enumerate(zip(points, points[1:])):
        start = pd.Timestamp(first["pivot_time"])
        end = pd.Timestamp(second["pivot_time"])
        internal = [item for item in lower_pivots if item.get("status") == "CONFIRMED" and start < pd.Timestamp(item["pivot_time"]) < end]
        observed_legs = len(internal) + 1
        scores.append(max(0.0, 100.0 - abs(observed_legs - expected[index]) * 18.0))
    return float(np.mean(scores)) if scores else 35.0


def parent_consistency_score(points: list[dict[str, Any]], parent: dict[str, Any] | None) -> float:
    if not parent or not parent.get("credible"):
        return 50.0
    parent_direction = parent.get("direction")
    child_direction = _direction(points)
    parent_waves = parent.get("waves") or []
    if not parent_waves:
        return 50.0
    start = pd.Timestamp(parent["start_time"])
    end = pd.Timestamp(parent["end_time"])
    child_start = pd.Timestamp(points[0]["pivot_time"])
    child_end = pd.Timestamp(points[-1]["pivot_time"])
    contained = start <= child_start <= child_end
    direction_match = parent_direction == child_direction
    return 90.0 if contained and direction_match else 60.0 if contained else 25.0


def unresolved_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "UNRESOLVED", "pattern": "UNRESOLVED", "label": "No high-confidence Elliott structure",
        "degree": "INTERMEDIATE", "direction": "UNKNOWN", "credible": False, "open_wave": True,
        "current_wave": "UNRESOLVED", "wave_state": "POTENTIAL", "completion_state": "DEVELOPING",
        "waves": [], "wave_to_pivot": {}, "hard_rules_passed": False, "hard_rule_checks": [],
        "score_components": {}, "complexity_penalty": 0.0, "parent_consistency_penalty": 0.0,
        "open_wave_penalty": 0.0, "parent_consistency": "AMBIGUOUS", "parent_candidate_id": None,
        "score": 0.0, "start_time": "", "end_time": "", "fib_targets": [], "invalidation": None,
        "duration_bars": 0, "end_bar_index": 0, "recency_penalty": 0.0, "confidence": "LOW",
    }


def _current_wave_state(waves: list[dict[str, Any]], open_wave: bool) -> tuple[str, str]:
    if open_wave:
        return "DEVELOPING", "DEVELOPING"
    last = waves[-1]
    if last.get("wave_status") == "CONFIRMED":
        return "CONFIRMED", "CONFIRMED_COMPLETE"
    return "POTENTIAL", "POTENTIALLY_COMPLETE"


def _oriented_prices(points: list[dict[str, Any]]) -> tuple[list[float], int]:
    raw = [float(item["price"]) for item in points]
    direction = 1 if raw[1] > raw[0] else -1
    return [value * direction for value in raw], direction


def _direction(points: list[dict[str, Any]]) -> str:
    if len(points) < 2:
        return "UNKNOWN"
    return "UP" if float(points[1]["price"]) > float(points[0]["price"]) else "DOWN"


def _alternating(points: list[dict[str, Any]]) -> bool:
    return all(first.get("kind") != second.get("kind") for first, second in zip(points, points[1:]))


def _check(rule: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {"rule": rule, "passed": bool(passed), "observed": observed}


def _display_label(pattern: str, current_wave: str) -> str:
    if pattern in {"IMPULSE", "LEADING_DIAGONAL", "ENDING_DIAGONAL"}:
        return f"{pattern.replace('_', ' ')} — Wave {current_wave}"
    return f"{pattern.replace('_', ' ')} — {current_wave}"


def _family(pattern: str) -> str:
    if pattern in {"IMPULSE", "LEADING_DIAGONAL", "ENDING_DIAGONAL"}:
        return "MOTIVE"
    if pattern in {"ZIGZAG", "FLAT", "EXPANDED_FLAT"}:
        return "ABC"
    return pattern


def _pattern_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item["pattern"]] = counts.get(item["pattern"], 0) + 1
    return counts


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate.get(key) for key in (
        "candidate_id", "pattern", "label", "degree", "direction", "score", "score_components",
        "complexity_penalty", "parent_consistency_penalty", "wave_state", "completion_state",
        "current_wave", "start_time", "end_time", "invalidation",
    )}


def _change_from_previous(previous: dict[str, Any] | None, primary: dict[str, Any], alternative: dict[str, Any]) -> dict[str, Any]:
    prior = (previous or {}).get("elliott_primary") or {}
    changed = bool(prior) and prior.get("candidate_id") != primary.get("candidate_id")
    reason = []
    if changed:
        if prior.get("pattern") != primary.get("pattern"):
            reason.append("PATTERN_FAMILY_CHANGED")
        if prior.get("wave_state") != primary.get("wave_state"):
            reason.append("WAVE_STATE_CHANGED")
        reason.append("NEW_PRICE_EVIDENCE_RESCORING")
    return {
        "previous_primary_label": prior.get("label"),
        "previous_primary_score": prior.get("score"),
        "new_primary_label": primary.get("label"),
        "new_primary_score": primary.get("score"),
        "new_alternative_label": alternative.get("label"),
        "primary_changed": changed,
        "reasons": reason,
    }

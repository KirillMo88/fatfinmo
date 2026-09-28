from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import Pivot, RuleCheck


@dataclass
class ValidationResult:
    pattern_type: str
    subtype: str | None
    direction: int
    checks: list[RuleCheck] = field(default_factory=list)
    ratios: list[dict[str, Any]] = field(default_factory=list)
    unknown_requirements: list[str] = field(default_factory=list)
    invalidation: dict[str, Any] | None = None
    profile_status: str = "CORE"

    @property
    def valid(self) -> bool:
        return not any(
            check.result == "FAIL" and check.rule_class in {"HARD_RULE", "PROFILE_RULE"}
            for check in self.checks
        )

    @property
    def geometry_status(self) -> str:
        return "VALID" if self.valid else "INVALID"


def validate_impulse(
    points: list[Pivot],
    bars: pd.DataFrame | None = None,
    *,
    truncated: bool = False,
) -> ValidationResult:
    _require_points(points, 6)
    d = 1 if points[1].price > points[0].price else -1
    q = [d * point.price for point in points]
    result = ValidationResult(
        "IMPULSE_TRUNCATED_5" if truncated else "IMPULSE",
        "TRUNCATED_5" if truncated else "REGULAR",
        d,
    )
    ids = [point.pivot_id for point in points]
    _add(result, "IMPULSE_ENDPOINT_0_2_1_3", "HARD_RULE", q[0] < q[2] < q[1] < q[3], q, "q0 < q2 < q1 < q3", ids[:4])
    _add(result, "IMPULSE_ENDPOINT_1_4_3", "HARD_RULE", q[1] < q[4] < q[3], q[1:5], "q1 < q4 < q3", ids[1:5])
    if truncated:
        _add(result, "TRUNCATED_5_ENDPOINT", "PROFILE_RULE", q[4] < q[5] <= q[3], q[3:6], "q4 < q5 <= q3", ids[3:6])
    else:
        _add(result, "IMPULSE_5_EXCEEDS_3", "HARD_RULE", q[5] > q[3], q[3:6], "q5 > q3", ids[3:6])

    l1, l3, l5 = q[1] - q[0], q[3] - q[2], q[5] - q[4]
    _add(result, "IMPULSE_3_NOT_SHORTEST", "HARD_RULE", l3 >= min(l1, l5), {"L1": l1, "L3": l3, "L5": l5}, "L3 >= min(L1,L5)", ids)
    result.ratios.extend(
        [
            _ratio("L3/L1", l3, l1),
            _ratio("L5/L1", l5, l1),
            _ratio("L5/L3", l5, l3),
            _ratio("R2/L1", q[1] - q[2], l1),
            _ratio("R4/L3", q[3] - q[4], l3),
        ]
    )

    wave1_max = _interval_q_extreme(bars, points[0], points[1], d, "max", fallback=max(q[0], q[1]))
    wave2_min = _interval_q_extreme(bars, points[1], points[2], d, "min", fallback=min(q[1], q[2]))
    wave4_min = _interval_q_extreme(bars, points[3], points[4], d, "min", fallback=min(q[3], q[4]))
    _add(result, "IMPULSE_2_NO_FULL_RETRACE", "HARD_RULE", wave2_min > q[0], wave2_min, f"> {q[0]}", ids[0:3])
    _add(result, "IMPULSE_4_NO_WAVE1_RANGE_OVERLAP", "HARD_RULE", wave4_min > wave1_max, {"wave4_min_q": wave4_min, "wave1_max_q": wave1_max}, "wave4_min_q > wave1_max_q", ids[0:5])
    if l3 < l1:
        _add(result, "IMPULSE_5_LENGTH_CAP", "HARD_RULE", l5 <= l3, {"L3": l3, "L5": l5}, "L5 <= L3 when L3 < L1", ids)

    if truncated:
        wave3_max = _interval_q_extreme(bars, points[2], points[3], d, "max", fallback=q[3])
        wave5_max = _interval_q_extreme(bars, points[4], points[5], d, "max", fallback=q[5])
        _add(result, "TRUNCATED_5_INTERNAL_EXTREME", "HARD_RULE", wave5_max <= wave3_max, {"wave3_max_q": wave3_max, "wave5_max_q": wave5_max}, "wave5 internal extreme <= wave3 internal extreme", ids[2:6])
        result.unknown_requirements.extend(["wave5_five_part_subdivision", "alternative_wave4_continuation"])
    else:
        result.unknown_requirements.extend(["wave1_subdivision", "wave2_subdivision", "wave3_subdivision", "wave4_subdivision", "wave5_subdivision"])

    result.invalidation = {
        "scope": "ordinary_impulse",
        "level": float(points[0].price),
        "basis": "High/Low",
        "meaning": "A full retracement through wave 0 invalidates this impulse count.",
    }
    return result


def validate_zigzag(points: list[Pivot], bars: pd.DataFrame | None = None) -> ValidationResult:
    _require_points(points, 4)
    d = 1 if points[1].price < points[0].price else -1
    q = [d * point.price for point in points]
    result = ValidationResult("ZIGZAG", "REGULAR", d)
    ids = [point.pivot_id for point in points]
    _add(result, "ZIGZAG_A_BELOW_START", "HARD_RULE", q[1] < q[0], q[:2], "qA < qS", ids[:2])
    _add(result, "ZIGZAG_B_BETWEEN_A_START", "PROFILE_RULE", q[1] < q[2] < q[0], q[:3], "qA < qB < qS", ids[:3])
    _add(result, "ZIGZAG_C_EXCEEDS_A", "PROFILE_RULE", q[3] < q[1], q[1:4], "qC < qA", ids[1:4])
    b_max = _interval_q_extreme(bars, points[1], points[2], d, "max", fallback=max(q[1], q[2]))
    _add(result, "ZIGZAG_B_NO_START_CROSS", "PROFILE_RULE", b_max < q[0], b_max, f"< {q[0]}", ids[:3])
    la, lb, lc = q[0] - q[1], q[2] - q[1], q[2] - q[3]
    result.ratios.extend([_ratio("B/A", lb, la), _ratio("C/A", lc, la)])
    result.unknown_requirements.extend(["A_motive_subdivision", "B_corrective_subdivision", "C_motive_subdivision"])
    result.invalidation = {
        "scope": "regular_zigzag",
        "level": float(points[0].price),
        "basis": "High/Low",
        "meaning": "Wave B crossing the start places the ordinary Zigzag outside the supported profile.",
    }
    return result


def validate_flat(
    points: list[Pivot],
    bars: pd.DataFrame | None = None,
    *,
    flat_min_b: float = 0.90,
    running_enabled: bool = False,
) -> list[ValidationResult]:
    _require_points(points, 4)
    d = 1 if points[1].price < points[0].price else -1
    q = [d * point.price for point in points]
    la = q[0] - q[1]
    lb = q[2] - q[1]
    rb = lb / la if la > 0 else np.nan
    outputs: list[ValidationResult] = []
    for pattern, subtype, endpoint_ok, profile_status in [
        ("FLAT_REGULAR", "REGULAR", flat_min_b <= rb <= 1.0 and q[3] < q[1], "CORE"),
        ("FLAT_EXPANDED", "EXPANDED", rb > 1.0 and q[3] < q[1], "CORE"),
        ("FLAT_RUNNING", "RUNNING", rb > 1.0 and q[1] < q[3] < q[0], "EXPERIMENTAL"),
    ]:
        if pattern == "FLAT_RUNNING" and not running_enabled:
            continue
        result = ValidationResult(pattern, subtype, d, profile_status=profile_status)
        ids = [point.pivot_id for point in points]
        _add(result, "FLAT_A_ACTION", "HARD_RULE", q[1] < q[0], q[:2], "qA < qS", ids[:2])
        _add(result, "FLAT_B_MINIMUM", "PROFILE_RULE", rb >= flat_min_b, rb, f">= {flat_min_b}", ids[:3])
        _add(result, f"{pattern}_ENDPOINT_PROFILE", "PROFILE_RULE", endpoint_ok, {"RB": rb, "qC": q[3]}, subtype, ids)
        result.ratios.extend([_ratio("B/A", lb, la), _ratio("C/A", q[2] - q[3], la)])
        result.unknown_requirements.extend(["A_corrective_subdivision", "B_corrective_subdivision", "C_motive_subdivision"])
        result.invalidation = {
            "scope": pattern,
            "level": float(points[0].price),
            "basis": "parent High/Low rule",
            "meaning": "The parent wave boundary remains active even when Flat B exceeds its own start.",
        }
        outputs.append(result)
    if q[3] == q[1]:
        boundary = ValidationResult("FLAT_BOUNDARY_UNRESOLVED", "C_EQUALS_A", d, profile_status="OUTSIDE_PROFILE")
        ids = [point.pivot_id for point in points]
        _add(boundary, "FLAT_C_EQUALS_A_BOUNDARY", "PROFILE_RULE", None, q[3], q[1], ids)
        boundary.unknown_requirements.extend(
            ["flat_subtype_at_exact_C_A_boundary", "A_corrective_subdivision", "B_corrective_subdivision", "C_motive_subdivision"]
        )
        outputs.append(boundary)
    return outputs


def validate_triangle(
    points: list[Pivot],
    bars: pd.DataFrame | None = None,
    *,
    barrier_tolerance_fraction: float = 0.05,
) -> list[ValidationResult]:
    _require_points(points, 6)
    d = 1 if points[1].price < points[0].price else -1
    q = [d * point.price for point in points]
    ids = [point.pivot_id for point in points]
    base_alternation = q[0] > q[1] and q[2] > q[1] and q[3] < q[2] and q[4] > q[3] and q[5] < q[4] and q[5] < q[0]
    href = q[2] - q[1]
    delta = barrier_tolerance_fraction * href
    width_d = _line_value(points[1], points[3], points[4])
    width_e_lower = _line_value(points[1], points[3], points[5])
    width_e_upper = _line_value(points[2], points[4], points[5])
    gap_d = d * points[2].price - d * points[3].price
    gap_e = d * width_e_upper - d * width_e_lower
    contracting_width = gap_d > 0 and gap_e > 0 and gap_e < gap_d
    outputs: list[ValidationResult] = []

    barrier = ValidationResult("TRIANGLE_BARRIER", "BARRIER", d)
    _add(barrier, "TRIANGLE_ALTERNATION", "HARD_RULE", base_alternation, q, "S-A-B-C-D-E alternating", ids)
    _add(barrier, "TRIANGLE_AC_CONTRACTION", "PROFILE_RULE", q[1] < q[3] < q[5], [q[1], q[3], q[5]], "qA < qC < qE", ids)
    _add(barrier, "TRIANGLE_BARRIER_SIDE", "PROFILE_RULE", abs(q[4] - q[2]) <= delta, {"difference": abs(q[4] - q[2]), "delta": delta}, "abs(qD-qB)<=delta", ids)
    _add(barrier, "TRIANGLE_POSITIVE_NARROWING_WIDTH", "PROFILE_RULE", contracting_width, {"gap_d": gap_d, "gap_e": gap_e}, "0 < gapE < gapD", ids)
    barrier.unknown_requirements.extend(["A_to_E_corrective_subdivisions", "allowed_parent_position"])
    outputs.append(barrier)

    contracting = ValidationResult("TRIANGLE_CONTRACTING", "CONTRACTING", d)
    _add(contracting, "TRIANGLE_ALTERNATION", "HARD_RULE", base_alternation, q, "S-A-B-C-D-E alternating", ids)
    _add(contracting, "TRIANGLE_AC_CONTRACTION", "PROFILE_RULE", q[1] < q[3] < q[5], [q[1], q[3], q[5]], "qA < qC < qE", ids)
    _add(contracting, "TRIANGLE_BD_CONTRACTION", "PROFILE_RULE", q[4] < q[2] - delta, {"qD": q[4], "qB-delta": q[2] - delta}, "qD < qB-delta", ids)
    _add(contracting, "TRIANGLE_POSITIVE_NARROWING_WIDTH", "PROFILE_RULE", contracting_width, {"gap_d": gap_d, "gap_e": gap_e}, "0 < gapE < gapD", ids)
    contracting.unknown_requirements.extend(["A_to_E_corrective_subdivisions", "allowed_parent_position"])
    outputs.append(contracting)
    # Barrier is checked first by caller. An E throw-through of A-C is not a failure.
    return outputs


def validate_ending_diagonal(points: list[Pivot], bars: pd.DataFrame | None = None) -> ValidationResult:
    _require_points(points, 6)
    d = 1 if points[1].price > points[0].price else -1
    q = [d * point.price for point in points]
    ids = [point.pivot_id for point in points]
    result = ValidationResult("ENDING_DIAGONAL_CONTRACTING_33333", "CONTRACTING_33333", d)
    endpoint_ok = q[0] < q[2] < q[4] < q[1] < q[3] < q[5]
    _add(result, "ENDING_DIAGONAL_ENDPOINTS", "HARD_RULE", endpoint_ok, q, "q0<q2<q4<q1<q3<q5", ids)
    l1, l2, l3, l4, l5 = q[1] - q[0], q[1] - q[2], q[3] - q[2], q[3] - q[4], q[5] - q[4]
    _add(result, "ENDING_DIAGONAL_CONTRACTING_MOTIVE", "PROFILE_RULE", l3 < l1 and l5 < l3, {"L1": l1, "L3": l3, "L5": l5}, "L5<L3<L1", ids)
    _add(result, "ENDING_DIAGONAL_CONTRACTING_RETRACE", "PROFILE_RULE", l4 < l2, {"L2": l2, "L4": l4}, "L4<L2", ids)
    _add(result, "ENDING_DIAGONAL_3_NOT_SHORTEST", "HARD_RULE", l3 >= min(l1, l5), {"L1": l1, "L3": l3, "L5": l5}, "L3>=min(L1,L5)", ids)
    wave2_min = _interval_q_extreme(bars, points[1], points[2], d, "min", fallback=q[2])
    wave4_min = _interval_q_extreme(bars, points[3], points[4], d, "min", fallback=q[4])
    _add(result, "ENDING_DIAGONAL_2_BOUNDARY", "HARD_RULE", wave2_min > q[0], wave2_min, f">{q[0]}", ids[:3])
    _add(result, "ENDING_DIAGONAL_4_BOUNDARY", "HARD_RULE", wave4_min > q[2], wave4_min, f">{q[2]}", ids[2:5])
    _add(result, "ENDING_DIAGONAL_1_4_OVERLAP", "PROFILE_RULE", wave4_min < q[1], wave4_min, f"<{q[1]}", ids[1:5])
    upper_at_4 = d * _line_value(points[1], points[3], points[4])
    upper_at_5 = d * _line_value(points[1], points[3], points[5])
    lower_at_5 = d * _line_value(points[2], points[4], points[5])
    width_at_4 = upper_at_4 - q[4]
    width_at_5 = upper_at_5 - lower_at_5
    _add(
        result,
        "ENDING_DIAGONAL_POSITIVE_NARROWING_WIDTH",
        "PROFILE_RULE",
        width_at_4 > 0 and width_at_5 > 0 and width_at_5 < width_at_4,
        {"width_at_4": width_at_4, "width_at_5": width_at_5},
        "0 < width_at_5 < width_at_4",
        ids,
    )
    result.ratios.extend([_ratio("L3/L1", l3, l1), _ratio("L5/L3", l5, l3), _ratio("L4/L2", l4, l2)])
    result.unknown_requirements.extend(["five_zigzag_components", "terminal_5_or_C_position"])
    return result


def validate_double_correction(
    points: list[Pivot],
    child_types: list[str],
    *,
    family: str,
) -> ValidationResult:
    _require_points(points, 4)
    if len(child_types) != 3:
        raise ValueError("Double correction requires W, X and Y child types")
    d = 1 if points[1].price < points[0].price else -1
    q = [d * point.price for point in points]
    ids = [point.pivot_id for point in points]
    result = ValidationResult(family, family, d)
    correction_types = {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED", "TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER", "DOUBLE_ZIGZAG", "DOUBLE_THREE"}
    _add(result, "DOUBLE_X_CORRECTIVE", "HARD_RULE", child_types[1] in correction_types, child_types[1], "corrective X", ids)
    _add(result, "DOUBLE_ENDPOINT_DIRECTION", "PROFILE_RULE", q[1] < q[0] and q[2] > q[1] and q[3] < q[0], q, "W action, X recovery, Y corrective action", ids)
    if family == "DOUBLE_ZIGZAG":
        _add(result, "DOUBLE_ZIGZAG_WY_TYPES", "HARD_RULE", child_types[0] == "ZIGZAG" and child_types[2] == "ZIGZAG", child_types, "W=ZIGZAG and Y=ZIGZAG", ids)
        _add(result, "DOUBLE_ZIGZAG_Y_EXCEEDS_W", "PROFILE_RULE", q[3] < q[1], q, "qY<qW", ids)
    elif family == "DOUBLE_THREE":
        allowed_pairs = {
            ("FLAT_REGULAR", "FLAT_REGULAR"),
            ("FLAT_REGULAR", "FLAT_EXPANDED"),
            ("FLAT_EXPANDED", "FLAT_REGULAR"),
            ("FLAT_EXPANDED", "FLAT_EXPANDED"),
            ("FLAT_REGULAR", "ZIGZAG"),
            ("FLAT_EXPANDED", "ZIGZAG"),
            ("ZIGZAG", "FLAT_REGULAR"),
            ("ZIGZAG", "FLAT_EXPANDED"),
            ("FLAT_REGULAR", "TRIANGLE_CONTRACTING"),
            ("FLAT_EXPANDED", "TRIANGLE_CONTRACTING"),
            ("ZIGZAG", "TRIANGLE_CONTRACTING"),
            ("FLAT_REGULAR", "TRIANGLE_BARRIER"),
            ("FLAT_EXPANDED", "TRIANGLE_BARRIER"),
            ("ZIGZAG", "TRIANGLE_BARRIER"),
        }
        _add(result, "DOUBLE_THREE_WY_TYPES", "PROFILE_RULE", (child_types[0], child_types[2]) in allowed_pairs, child_types, "supported W/Y pair", ids)
        triangle_count = sum("TRIANGLE" in value for value in child_types)
        _add(result, "DOUBLE_THREE_MAX_ONE_TRIANGLE", "PROFILE_RULE", triangle_count <= 1, triangle_count, "<=1", ids)
    else:
        raise ValueError(f"Unsupported double family={family}")
    return result


def _require_points(points: list[Pivot], expected: int) -> None:
    if len(points) != expected:
        raise ValueError(f"Expected {expected} points, got {len(points)}")


def _add(
    result: ValidationResult,
    rule_id: str,
    rule_class: str,
    passed: bool | None,
    observed: Any,
    expected: Any,
    point_ids: list[str],
) -> None:
    status = "UNKNOWN" if passed is None else "PASS" if bool(passed) else "FAIL"
    result.checks.append(RuleCheck(rule_id, rule_class, status, observed, expected, point_ids))


def _ratio(name: str, numerator: float, denominator: float) -> dict[str, Any]:
    value = numerator / denominator if denominator and np.isfinite(denominator) else None
    return {"ratio_id": name, "numerator": float(numerator), "denominator": float(denominator), "value": float(value) if value is not None and np.isfinite(value) else None}


def _interval_q_extreme(
    bars: pd.DataFrame | None,
    start: Pivot,
    end: Pivot,
    d: int,
    mode: str,
    *,
    fallback: float,
) -> float:
    if bars is None or bars.empty or start.bar_index is None or end.bar_index is None:
        return float(fallback)
    low_idx, high_idx = sorted((int(start.bar_index), int(end.bar_index)))
    interval = bars.iloc[low_idx : high_idx + 1]
    if interval.empty:
        return float(fallback)
    lows = pd.to_numeric(interval["low"], errors="coerce")
    highs = pd.to_numeric(interval["high"], errors="coerce")
    if mode == "min":
        value = lows.min() if d == 1 else -highs.max()
    else:
        value = highs.max() if d == 1 else -lows.min()
    return float(value) if np.isfinite(value) else float(fallback)


def _line_value(first: Pivot, second: Pivot, target: Pivot) -> float:
    x1 = int(first.bar_index or 0)
    x2 = int(second.bar_index or x1 + 1)
    xt = int(target.bar_index or x2)
    if x2 == x1:
        return float(second.price)
    return float(first.price + (second.price - first.price) * (xt - x1) / (x2 - x1))

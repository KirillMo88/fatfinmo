from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_WINDOWS,
    DEFAULT_ANALYSIS_WINDOW,
    ENGINE_PARAMETERS,
    ENGINE_VERSION,
    PATTERN_LABELS,
    RULE_PROFILE,
    AssetSpec,
)
from .data import data_version as calculate_data_version, validate_bars
from .lifecycle import evaluate_lifecycle
from .models import Pivot, RuleCheck, Scenario, Target, WaveNode
from .pivots import PivotStream, build_causal_pivot_streams
from .validators import (
    ValidationResult,
    validate_double_correction,
    validate_ending_diagonal,
    validate_flat,
    validate_impulse,
    validate_triangle,
    validate_zigzag,
)
from .wave_map import WaveMapBuilder


class ElliottWaveEngine:
    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        self.parameters = dict(ENGINE_PARAMETERS)
        if parameters:
            self.parameters.update(parameters)
        self.parameter_hash = _stable_hash(self.parameters, 20)

    def analyze(
        self,
        bars: pd.DataFrame,
        spec: AssetSpec,
        *,
        analysis_window: str = DEFAULT_ANALYSIS_WINDOW,
    ) -> dict[str, Any]:
        if analysis_window not in ANALYSIS_WINDOWS:
            raise ValueError(
                f"Unsupported analysis window {analysis_window!r}; expected one of {tuple(ANALYSIS_WINDOWS)}"
            )
        values = validate_bars(bars, spec).reset_index(drop=True)
        all_closed = values.loc[values["is_closed"]].reset_index(drop=True)
        if all_closed.empty:
            raise ValueError(f"{spec.canonical_asset_id}: no closed bars for Elliott analysis")
        as_of_timestamp = pd.to_datetime(all_closed.iloc[-1]["timestamp"], utc=True)
        requested_analysis_start = as_of_timestamp - pd.DateOffset(years=ANALYSIS_WINDOWS[analysis_window])
        first_timestamp = pd.to_datetime(all_closed.iloc[0]["timestamp"], utc=True)
        effective_analysis_start = max(requested_analysis_start, first_timestamp)
        timestamps = pd.to_datetime(all_closed["timestamp"], utc=True)
        analysis_start_index = int(timestamps.searchsorted(effective_analysis_start, side="left"))
        warmup_bars = int(self.parameters.get("analysis_warmup_bars", 260))
        warmup_start_index = max(0, analysis_start_index - warmup_bars)
        closed = all_closed.iloc[warmup_start_index:].reset_index(drop=True)
        analysis_start = pd.Timestamp(effective_analysis_start).isoformat()
        analysis_warmup_start = pd.Timestamp(closed.iloc[0]["timestamp"]).isoformat()
        source_version = calculate_data_version(all_closed)
        version = calculate_data_version(closed)
        streams = build_causal_pivot_streams(
            closed,
            asset_id=spec.canonical_asset_id,
            source_timeframe=spec.base_timeframe,
            atr_period=int(self.parameters["atr_period"]),
            multipliers=list(self.parameters["pivot_atr_multipliers"]),
            max_pivots=int(self.parameters["max_pivots_per_stream"]),
        )
        nodes, active_node_ids, search_statistics = self._build_nodes(closed, streams, spec)
        scenarios = self._rank_scenarios(nodes, active_node_ids, len(closed))
        self._attach_scenarios(nodes, scenarios)
        wave_map = WaveMapBuilder().build(
            nodes,
            closed,
            analysis_window=analysis_window,
            analysis_start=analysis_start,
        )
        main_scenario_id = wave_map["main_root_scenario_id"]
        created_at = pd.Timestamp.now(tz="UTC").isoformat()
        as_of = as_of_timestamp.isoformat()
        snapshot_id = _stable_hash(
            {
                "asset": spec.canonical_asset_id,
                "as_of": as_of,
                "data_version": version,
                "analysis_window": analysis_window,
                "analysis_start": analysis_start,
                "parameters": self.parameter_hash,
                "engine": ENGINE_VERSION,
            },
            24,
        )
        lifecycle_events = evaluate_lifecycle(nodes.values(), closed)
        events = self._events(streams, scenarios, nodes, as_of, search_statistics)
        events.extend(lifecycle_events)
        events.sort(key=lambda item: (item["known_at"], item["event_id"]))
        unresolved = (
            []
            if wave_map["wave_nodes"]
            else ["UNRESOLVED: no CORE candidate passed the available geometry checks"]
        )
        quality_flags = ["HISTORICAL_RECEIVED_AT_UNAVAILABLE"]
        if any(stream.ambiguous_bar_ids for stream in streams):
            quality_flags.append("AMBIGUOUS_BARS_PRESENT")
        if search_statistics["search_truncated"]:
            quality_flags.append("SEARCH_LIMITED")
        if len(closed) < int(self.parameters["atr_period"]) + 10:
            quality_flags.append("INSUFFICIENT_WARMUP")

        node_dicts = [node.to_dict() for node in nodes.values()]
        scenario_dicts = [scenario.to_dict() for scenario in scenarios]
        return {
            "snapshot_id": snapshot_id,
            "canonical_asset_id": spec.canonical_asset_id,
            "symbol": spec.display_name,
            "source_id": spec.source_id,
            "provider_symbol": spec.provider_symbol,
            "instrument_type": spec.instrument_type,
            "currency": spec.currency,
            "price_unit": spec.price_unit,
            "session_calendar": spec.session_calendar,
            "source_timezone": spec.source_timezone,
            "adjustment_mode": spec.adjustment_mode,
            "provider_label": spec.provider_label,
            "provenance_note": spec.provenance_note,
            "base_timeframe": spec.base_timeframe,
            "as_of": as_of,
            "created_at": created_at,
            "data_cutoff": as_of,
            "availability_mode": "CAUSAL_REPLAY_CURRENT_VINTAGE",
            "engine_version": ENGINE_VERSION,
            "rule_profile": RULE_PROFILE,
            "parameter_hash": self.parameter_hash,
            "parameters": self.parameters,
            "data_version": version,
            "source_data_version": source_version,
            "history_start": first_timestamp.isoformat(),
            "history_end": as_of,
            "bar_count": int(len(all_closed)),
            "analysis_window": analysis_window,
            "requested_analysis_start": pd.Timestamp(requested_analysis_start).isoformat(),
            "analysis_start": analysis_start,
            "analysis_warmup_start": analysis_warmup_start,
            "analysis_warmup_bars": int(analysis_start_index - warmup_start_index),
            "analysis_bar_count": int((timestamps >= effective_analysis_start).sum()),
            "main_scenario_id": main_scenario_id,
            "main_root_scenario_id": main_scenario_id,
            "alternatives": [scenario["scenario_id"] for scenario in wave_map["root_scenarios"][1:]],
            "unresolved_reasons": unresolved,
            "pivot_streams": [stream.to_dict() for stream in streams],
            "nodes": node_dicts,
            "scenarios": scenario_dicts,
            "root_scenarios": wave_map["root_scenarios"],
            "wave_nodes": wave_map["wave_nodes"],
            "structural_edges": wave_map["structural_edges"],
            "active_major_node_id": wave_map["active_major_node_id"],
            "active_intermediate_node_id": wave_map["active_intermediate_node_id"],
            "active_minor_node_id": wave_map["active_minor_node_id"],
            "historical_completed_node_ids": wave_map["historical_completed_node_ids"],
            "unresolved_intervals": wave_map["unresolved_intervals"],
            "rule_checks": [check for node in node_dicts for check in node["rule_checks"]],
            "ratios": [ratio for node in node_dicts for ratio in node["ratios"]],
            "targets": [target for node in node_dicts for target in node["targets"]],
            "channels": [channel for node in node_dicts for channel in node["channels"]],
            "events": events,
            "search_statistics": {
                **search_statistics,
                "wave_map": wave_map["wave_map_statistics"],
            },
            "quality_flags": quality_flags,
        }

    def review_anchored(
        self,
        bars: pd.DataFrame,
        spec: AssetSpec,
        pattern_type: str,
        points: list[Pivot],
    ) -> dict[str, Any]:
        """Validate user-selected existing anchors without weakening any rule.

        This is intentionally separate from ``analyze`` and never mutates or
        publishes the AUTO history.  A failed mandatory rule remains visible as
        MODEL_INVALID in the review result.
        """
        values = validate_bars(bars, spec).reset_index(drop=True)
        if pattern_type == "IMPULSE":
            results = [validate_impulse(points, values)]
        elif pattern_type == "IMPULSE_TRUNCATED_5":
            results = [validate_impulse(points, values, truncated=True)]
        elif pattern_type == "ZIGZAG":
            results = [validate_zigzag(points, values)]
        elif pattern_type in {"FLAT_REGULAR", "FLAT_EXPANDED", "FLAT_RUNNING"}:
            results = [
                result
                for result in validate_flat(
                    points,
                    values,
                    flat_min_b=float(self.parameters["flat_min_b"]),
                    running_enabled=bool(self.parameters["experimental_enabled"]),
                )
                if result.pattern_type == pattern_type
            ]
        elif pattern_type in {"TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}:
            results = [
                result
                for result in validate_triangle(
                    points,
                    values,
                    barrier_tolerance_fraction=float(self.parameters["barrier_tolerance_fraction"]),
                )
                if result.pattern_type == pattern_type
            ]
        elif pattern_type == "ENDING_DIAGONAL_CONTRACTING_33333":
            results = [validate_ending_diagonal(points, values)]
        else:
            return {
                "mode": "ANCHORED_REVIEW",
                "pattern_type": pattern_type,
                "status": "UNSUPPORTED_PATTERN",
                "auto_history_modified": False,
                "rule_checks": [],
            }
        if not results:
            return {
                "mode": "ANCHORED_REVIEW",
                "pattern_type": pattern_type,
                "status": "OUTSIDE_PROFILE",
                "auto_history_modified": False,
                "rule_checks": [],
            }
        result = results[0]
        return {
            "mode": "ANCHORED_REVIEW",
            "pattern_type": result.pattern_type,
            "status": "GEOMETRY_VALID" if result.valid else "MODEL_INVALID",
            "auto_history_modified": False,
            "anchor_ids": [point.pivot_id for point in points],
            "rule_checks": [check.to_dict() for check in result.checks],
            "unknown_requirements": list(result.unknown_requirements),
        }

    def _build_nodes(
        self,
        bars: pd.DataFrame,
        streams: list[PivotStream],
        spec: AssetSpec,
    ) -> tuple[dict[str, WaveNode], set[str], dict[str, Any]]:
        nodes: dict[str, WaveNode] = {}
        active_node_ids: set[str] = set()
        rejected = 0
        examined = 0
        hard_limit = int(self.parameters["max_active_scenarios"]) * 80
        truncated = False

        for stream in streams:
            points = stream.pivots
            if len(points) < 4:
                continue
            for size in (4, 6):
                if len(points) < size:
                    continue
                for start in range(0, len(points) - size + 1):
                    if examined >= hard_limit:
                        truncated = True
                        break
                    window = points[start : start + size]
                    if not _alternating(window):
                        continue
                    results: list[ValidationResult] = []
                    if size == 4:
                        results.append(validate_zigzag(window, bars))
                        results.extend(
                            validate_flat(
                                window,
                                bars,
                                flat_min_b=float(self.parameters["flat_min_b"]),
                                running_enabled=bool(self.parameters["experimental_enabled"]),
                            )
                        )
                    else:
                        results.append(validate_impulse(window, bars))
                        results.append(validate_impulse(window, bars, truncated=True))
                        results.extend(
                            validate_triangle(
                                window,
                                bars,
                                barrier_tolerance_fraction=float(self.parameters["barrier_tolerance_fraction"]),
                            )
                        )
                        results.append(validate_ending_diagonal(window, bars))
                    for result in results:
                        examined += 1
                        supported_patterns = set(self.parameters["core_patterns"]) | set(self.parameters["conditional_patterns"])
                        if bool(self.parameters["experimental_enabled"]):
                            supported_patterns |= set(self.parameters["experimental_patterns_disabled"])
                        if (
                            not result.valid
                            or result.pattern_type not in supported_patterns
                            or (
                                result.pattern_type in self.parameters["experimental_patterns_disabled"]
                                and not bool(self.parameters["experimental_enabled"])
                            )
                        ):
                            rejected += 1
                            continue
                        node = self._node_from_validation(
                            result,
                            window,
                            bars,
                            spec,
                            stream,
                            nodes,
                        )
                        existing = nodes.get(node.node_id)
                        if existing is None or node.verified_depth > existing.verified_depth:
                            nodes[node.node_id] = node
                        if start + size == len(points):
                            active_node_ids.add(node.node_id)
                if truncated:
                    break
            if truncated:
                break

        prefix_nodes, prefix_active, prefix_examined, prefix_rejected = self._build_forming_prefixes(
            bars,
            streams,
            spec,
            nodes,
        )
        nodes.update(prefix_nodes)
        active_node_ids.update(prefix_active)
        examined += prefix_examined
        rejected += prefix_rejected

        double_nodes, double_active, double_examined, double_rejected, double_truncated = self._build_double_nodes(
            nodes,
            bars,
            spec,
            hard_limit=max(200, hard_limit - examined),
        )
        nodes.update(double_nodes)
        active_node_ids.update(double_active)
        examined += double_examined
        rejected += double_rejected
        truncated = truncated or double_truncated
        if not truncated:
            recursive_nodes, recursive_active, recursive_examined, recursive_rejected, recursive_truncated = self._build_recursive_nodes(
                nodes,
                bars,
                spec,
                hard_limit=max(300, hard_limit - examined),
            )
            nodes.update(recursive_nodes)
            active_node_ids.update(recursive_active)
            examined += recursive_examined
            rejected += recursive_rejected
            truncated = truncated or recursive_truncated
        return nodes, active_node_ids, {
            "candidate_checks": examined,
            "valid_nodes": len(nodes),
            "rejected_candidates": rejected,
            "active_candidate_count": len(active_node_ids),
            "search_truncated": bool(truncated),
            "max_depth": int(self.parameters["max_depth"]),
            "beam_per_interval_pattern": int(self.parameters["beam_per_interval_pattern"]),
            "max_active_scenarios": int(self.parameters["max_active_scenarios"]),
        }

    def _build_forming_prefixes(
        self,
        bars: pd.DataFrame,
        streams: list[PivotStream],
        spec: AssetSpec,
        nodes: dict[str, WaveNode],
    ) -> tuple[dict[str, WaveNode], set[str], int, int]:
        created: dict[str, WaveNode] = {}
        active: set[str] = set()
        examined = rejected = 0
        for stream in streams:
            points = stream.pivots
            if len(points) >= 4:
                window = points[-4:]
                examined += 1
                d = 1 if window[1].price > window[0].price else -1
                q = [d * point.price for point in window]
                low_idx, high_idx = sorted((int(window[1].bar_index or 0), int(window[2].bar_index or 0)))
                interval = bars.iloc[low_idx : high_idx + 1]
                wave2_min = float(interval["low"].min()) if d == 1 else float(-interval["high"].max())
                result = ValidationResult("IMPULSE", "FORMING_1_2_3", d)
                result.checks.extend(
                    [
                        RuleCheck(
                            "IMPULSE_PREFIX_ENDPOINT_0_2_1_3",
                            "HARD_RULE",
                            "PASS" if q[0] < q[2] < q[1] < q[3] else "FAIL",
                            q,
                            "q0 < q2 < q1 < q3",
                            [point.pivot_id for point in window],
                        ),
                        RuleCheck(
                            "IMPULSE_2_NO_FULL_RETRACE",
                            "HARD_RULE",
                            "PASS" if wave2_min > q[0] else "FAIL",
                            wave2_min,
                            f"> {q[0]}",
                            [point.pivot_id for point in window[:3]],
                        ),
                    ]
                )
                result.unknown_requirements.extend(
                    ["wave1_subdivision", "wave2_subdivision", "wave3_subdivision", "wave4_not_observed", "wave5_not_observed"]
                )
                result.invalidation = {
                    "scope": "forming_impulse",
                    "level": float(window[0].price),
                    "basis": "High/Low",
                    "meaning": "A full retracement through wave 0 invalidates this forming impulse count.",
                }
                if result.valid:
                    node = self._node_from_validation(result, window, bars, spec, stream, {**nodes, **created})
                    node.geometry_status = "PENDING"
                    node.endpoint_status = "FORMING"
                    node.subdivision_status = "PARTIAL"
                    created[node.node_id] = node
                    active.add(node.node_id)
                else:
                    rejected += 1

            if len(points) >= 3:
                window = points[-3:]
                d = 1 if window[1].price < window[0].price else -1
                q = [d * point.price for point in window]
                la = q[0] - q[1]
                rb = (q[2] - q[1]) / la if la > 0 else math.nan
                candidates: list[tuple[str, str, bool, list[str]]] = [
                    (
                        "ZIGZAG",
                        "AWAITING_C",
                        q[1] < q[2] < q[0],
                        ["A_motive_subdivision", "B_corrective_subdivision", "C_not_observed"],
                    )
                ]
                if np.isfinite(rb) and rb >= float(self.parameters["flat_min_b"]):
                    candidates.append(
                        (
                            "FLAT_REGULAR" if rb <= 1.0 else "FLAT_EXPANDED",
                            "AWAITING_C",
                            q[1] < q[0],
                            ["A_corrective_subdivision", "B_corrective_subdivision", "C_not_observed"],
                        )
                    )
                for pattern, subtype, endpoint_ok, unknowns in candidates:
                    examined += 1
                    result = ValidationResult(pattern, subtype, d)
                    result.checks.append(
                        RuleCheck(
                            f"{pattern}_PREFIX_S_A_B",
                            "PROFILE_RULE",
                            "PASS" if endpoint_ok else "FAIL",
                            {"q": q, "B/A": rb},
                            "Valid S-A-B prefix; C is not observed",
                            [point.pivot_id for point in window],
                        )
                    )
                    result.unknown_requirements.extend(unknowns)
                    result.invalidation = {
                        "scope": pattern,
                        "level": float(window[0].price),
                        "basis": "High/Low",
                        "meaning": "The active parent boundary remains in force while C is awaited.",
                    }
                    if not result.valid:
                        rejected += 1
                        continue
                    node = self._node_from_validation(result, window, bars, spec, stream, {**nodes, **created})
                    node.geometry_status = "PENDING"
                    node.endpoint_status = "FORMING"
                    node.subdivision_status = "PARTIAL"
                    created[node.node_id] = node
                    active.add(node.node_id)
        return created, active, examined, rejected

    def _node_from_validation(
        self,
        result: ValidationResult,
        points: list[Pivot],
        bars: pd.DataFrame,
        spec: AssetSpec,
        stream: PivotStream,
        node_store: dict[str, WaveNode],
        *,
        child_node_ids: list[str] | None = None,
    ) -> WaveNode:
        if child_node_ids is None:
            child_node_ids = []
            for first, second in zip(points, points[1:]):
                leaf = self._leaf_node(first, second, bars, spec, stream)
                node_store.setdefault(leaf.node_id, leaf)
                child_node_ids.append(leaf.node_id)
            verified_depth = 0
            subdivision_status = "UNVERIFIED"
            coverage = 0.0
        else:
            verified_depth = 1 + min((node_store[child].verified_depth for child in child_node_ids), default=0)
            subdivision_status = "VERIFIED_TO_DEPTH"
            coverage = 1.0

        point_signature = [(point.pivot_time, round(point.price, 8), point.kind) for point in points]
        node_id = _stable_hash({"pattern": result.pattern_type, "points": point_signature, "children": child_node_ids}, 22)
        low_idx = min(int(point.bar_index or 0) for point in points)
        high_idx = max(int(point.bar_index or 0) for point in points)
        interval = bars.iloc[low_idx : high_idx + 1]
        known_at = max(point.known_at for point in points)
        labels = []
        label_names = PATTERN_LABELS.get(result.pattern_type, [str(idx) for idx in range(len(points))])
        for label, point in zip(label_names, points):
            payload = point.to_dict()
            payload["label"] = label
            payload["degree"] = "D0" if verified_depth == 0 else f"D{min(verified_depth, 3)}"
            labels.append(payload)
        targets = self._targets(result, points)
        for target in targets:
            target.node_id = node_id
        channels = self._channels(result, points)
        invalidation = dict(result.invalidation) if result.invalidation else None
        if invalidation is not None:
            invalidation.update(
                {
                    "scope_node_id": node_id,
                    "valid_from": points[1].known_at if len(points) > 1 else known_at,
                    "valid_until": None,
                    "active_stage": "WHILE_SCENARIO_ACTIVE",
                    "check_type": invalidation.get("basis", "High/Low"),
                }
            )
        return WaveNode(
            node_id=node_id,
            pattern_type=result.pattern_type,
            subtype=result.subtype,
            profile_id=RULE_PROFILE,
            direction=result.direction,
            orientation_direction=result.direction,
            relative_degree="D0" if verified_depth == 0 else f"D{min(verified_depth, 3)}",
            start_point=points[0].to_dict(),
            end_point=points[-1].to_dict(),
            internal_high=float(pd.to_numeric(interval["high"], errors="coerce").max()),
            internal_low=float(pd.to_numeric(interval["low"], errors="coerce").min()),
            extreme_times={
                "high": pd.Timestamp(interval.loc[pd.to_numeric(interval["high"], errors="coerce").idxmax(), "timestamp"]).isoformat(),
                "low": pd.Timestamp(interval.loc[pd.to_numeric(interval["low"], errors="coerce").idxmin(), "timestamp"]).isoformat(),
            },
            source_bar_range=[str(bars.iloc[low_idx]["bar_id"]), str(bars.iloc[high_idx]["bar_id"])],
            source_timeframes=[spec.base_timeframe],
            duration_bars=max(0, high_idx - low_idx),
            duration_calendar=max(0, (pd.Timestamp(points[-1].pivot_time) - pd.Timestamp(points[0].pivot_time)).days),
            endpoint_status=points[-1].status,
            geometry_status=result.geometry_status,
            context_status="UNRESOLVED",
            subdivision_status=subdivision_status,
            verified_depth=verified_depth,
            verification_coverage=coverage,
            known_at=known_at,
            first_observed_at=known_at,
            last_updated_at=known_at,
            segmentation_evidence=[
                {
                    "pivot_stream_k": stream.k,
                    "branch": stream.branch,
                    "point_ids": [point.pivot_id for point in points],
                }
            ],
            children=child_node_ids,
            labels=labels,
            ratios=result.ratios,
            rule_checks=[check.to_dict() for check in result.checks],
            unknown_requirements=_remaining_unknowns(result.unknown_requirements, child_node_ids, node_store),
            invalidation=invalidation,
            targets=[target.to_dict() for target in targets],
            channels=channels,
            pivot_stream_k=stream.k,
            start_pivot_index=low_idx,
            end_pivot_index=high_idx,
        )

    def _leaf_node(
        self,
        first: Pivot,
        second: Pivot,
        bars: pd.DataFrame,
        spec: AssetSpec,
        stream: PivotStream,
    ) -> WaveNode:
        node_id = _stable_hash({"leaf": [(first.pivot_time, first.price), (second.pivot_time, second.price)]}, 22)
        low_idx, high_idx = sorted((int(first.bar_index or 0), int(second.bar_index or 0)))
        interval = bars.iloc[low_idx : high_idx + 1]
        direction = 1 if second.price > first.price else -1
        return WaveNode(
            node_id=node_id,
            pattern_type="OBSERVED_LEAF",
            subtype=None,
            profile_id=RULE_PROFILE,
            direction=direction,
            orientation_direction=direction,
            relative_degree="D-1",
            start_point=first.to_dict(),
            end_point=second.to_dict(),
            internal_high=float(interval["high"].max()),
            internal_low=float(interval["low"].min()),
            extreme_times={},
            source_bar_range=[str(bars.iloc[low_idx]["bar_id"]), str(bars.iloc[high_idx]["bar_id"])],
            source_timeframes=[spec.base_timeframe],
            duration_bars=high_idx - low_idx,
            duration_calendar=max(0, (pd.Timestamp(second.pivot_time) - pd.Timestamp(first.pivot_time)).days),
            endpoint_status=second.status,
            geometry_status="VALID",
            context_status="UNRESOLVED",
            subdivision_status="INTERNAL_STRUCTURE_UNRESOLVED",
            verified_depth=0,
            verification_coverage=0.0,
            known_at=max(first.known_at, second.known_at),
            first_observed_at=max(first.known_at, second.known_at),
            last_updated_at=max(first.known_at, second.known_at),
            segmentation_evidence=[
                {
                    "pivot_stream_k": stream.k,
                    "branch": stream.branch,
                    "point_ids": [first.pivot_id, second.pivot_id],
                }
            ],
            children=[],
            labels=[],
            rule_checks=[],
            unknown_requirements=["internal_structure"],
            pivot_stream_k=stream.k,
            start_pivot_index=low_idx,
            end_pivot_index=high_idx,
        )

    def _build_double_nodes(
        self,
        nodes: dict[str, WaveNode],
        bars: pd.DataFrame,
        spec: AssetSpec,
        *,
        hard_limit: int,
    ) -> tuple[dict[str, WaveNode], set[str], int, int, bool]:
        corrections = [
            node
            for node in nodes.values()
            if node.pattern_type in {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED", "TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}
            and node.geometry_status == "VALID"
        ]
        by_start: dict[tuple[float | None, int | None], list[WaveNode]] = {}
        for node in corrections:
            by_start.setdefault((node.pivot_stream_k, node.start_pivot_index), []).append(node)
        created: dict[str, WaveNode] = {}
        active: set[str] = set()
        examined = rejected = 0
        latest_idx = len(bars) - 1
        truncated = False
        for w in corrections:
            for x in by_start.get((w.pivot_stream_k, w.end_pivot_index), [])[:20]:
                for y in by_start.get((x.pivot_stream_k, x.end_pivot_index), [])[:20]:
                    if examined >= hard_limit:
                        truncated = True
                        return created, active, examined, rejected, truncated
                    points = [
                        _pivot_from_payload(w.start_point),
                        _pivot_from_payload(w.end_point),
                        _pivot_from_payload(x.end_point),
                        _pivot_from_payload(y.end_point),
                    ]
                    child_types = [w.pattern_type, x.pattern_type, y.pattern_type]
                    stream = PivotStream(float(w.pivot_stream_k or 0.0), "COMBINED", points, [], 0)
                    for family in ("DOUBLE_ZIGZAG", "DOUBLE_THREE"):
                        examined += 1
                        result = validate_double_correction(points, child_types, family=family)
                        if not result.valid:
                            rejected += 1
                            continue
                        node = self._node_from_validation(
                            result,
                            points,
                            bars,
                            spec,
                            stream,
                            {**nodes, **created},
                            child_node_ids=[w.node_id, x.node_id, y.node_id],
                        )
                        created[node.node_id] = node
                        if node.end_pivot_index is not None and latest_idx - node.end_pivot_index <= 20:
                            active.add(node.node_id)
        return created, active, examined, rejected, truncated

    def _attach_scenarios(self, nodes: dict[str, WaveNode], scenarios: list[Scenario]) -> None:
        for scenario in scenarios:
            stack = [scenario.root_node_id]
            seen: set[str] = set()
            while stack:
                node_id = stack.pop()
                if node_id in seen or node_id not in nodes:
                    continue
                seen.add(node_id)
                node = nodes[node_id]
                if scenario.scenario_id not in node.scenario_ids:
                    node.scenario_ids.append(scenario.scenario_id)
                    node.scenario_ids.sort()
                stack.extend(node.children)

    def _build_recursive_nodes(
        self,
        nodes: dict[str, WaveNode],
        bars: pd.DataFrame,
        spec: AssetSpec,
        *,
        hard_limit: int,
    ) -> tuple[dict[str, WaveNode], set[str], int, int, bool]:
        created: dict[str, WaveNode] = {}
        active: set[str] = set()
        examined = rejected = 0
        truncated = False
        latest_idx = len(bars) - 1
        correction = {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED", "DOUBLE_ZIGZAG", "DOUBLE_THREE", "TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}
        motive = {"IMPULSE", "IMPULSE_TRUNCATED_5"}
        grammars: list[tuple[str, list[set[str]]]] = [
            ("IMPULSE", [motive, correction - {"TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}, {"IMPULSE"}, correction, motive | {"ENDING_DIAGONAL_CONTRACTING_33333"}]),
            ("ZIGZAG", [motive, correction, motive | {"ENDING_DIAGONAL_CONTRACTING_33333"}]),
            ("FLAT", [correction - {"TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}, correction, motive | {"ENDING_DIAGONAL_CONTRACTING_33333"}]),
            ("TRIANGLE", [{"ZIGZAG", "DOUBLE_ZIGZAG"}] * 5),
            ("ENDING_DIAGONAL_CONTRACTING_33333", [{"ZIGZAG"}] * 5),
        ]
        aggregate = {**nodes}
        max_depth = int(self.parameters["max_depth"])
        beam = int(self.parameters["beam_per_interval_pattern"])
        for depth in range(1, max_depth):
            eligible = [
                node
                for node in aggregate.values()
                if node.pattern_type != "OBSERVED_LEAF"
                and node.geometry_status == "VALID"
                and node.verified_depth >= depth - 1
            ]
            if not eligible:
                break
            by_start: dict[tuple[float | None, int | None], list[WaveNode]] = {}
            for node in eligible:
                by_start.setdefault((node.pivot_stream_k, node.start_pivot_index), []).append(node)
            for choices in by_start.values():
                choices.sort(key=lambda item: (-item.verified_depth, item.end_pivot_index or 0, item.node_id))

            pass_created = 0
            for family, roles in grammars:
                first_candidates = [node for node in eligible if node.pattern_type in roles[0]]
                for first in first_candidates:
                    sequences = self._child_sequences(first, roles, by_start, beam)
                    for sequence in sequences:
                        if examined >= hard_limit:
                            truncated = True
                            return created, active, examined, rejected, truncated
                        child_types = [child.pattern_type for child in sequence]
                        if family == "TRIANGLE" and sum(value == "DOUBLE_ZIGZAG" for value in child_types) > 1:
                            rejected += 1
                            continue
                        points = [_pivot_from_payload(sequence[0].start_point)] + [
                            _pivot_from_payload(child.end_point) for child in sequence
                        ]
                        results = self._validate_parent_family(family, points, bars)
                        for result in results:
                            examined += 1
                            if not result.valid:
                                rejected += 1
                                continue
                            result.checks.append(
                                RuleCheck(
                                    rule_id="PARENT_CHILD_GRAMMAR",
                                    rule_class="HARD_RULE",
                                    result="PASS",
                                    observed=child_types,
                                    expected=family,
                                    point_ids=[point.pivot_id for point in points],
                                )
                            )
                            stream = PivotStream(float(first.pivot_stream_k or 0.0), "RECURSIVE", points, [], 0)
                            store = {**aggregate, **created}
                            node = self._node_from_validation(
                                result,
                                points,
                                bars,
                                spec,
                                stream,
                                store,
                                child_node_ids=[child.node_id for child in sequence],
                            )
                            if node.verified_depth < depth:
                                continue
                            if node.node_id not in aggregate and node.node_id not in created:
                                created[node.node_id] = node
                                pass_created += 1
                            active_window = max(20, int(len(bars) * 0.05))
                            if node.end_pivot_index is not None and latest_idx - node.end_pivot_index <= active_window:
                                active.add(node.node_id)
            if pass_created == 0:
                break
            aggregate.update(created)
        return created, active, examined, rejected, truncated

    def _child_sequences(
        self,
        first: WaveNode,
        roles: list[set[str]],
        by_start: dict[tuple[float | None, int | None], list[WaveNode]],
        beam: int,
    ) -> list[list[WaveNode]]:
        sequences: list[list[WaveNode]] = []

        def visit(current: list[WaveNode], role_index: int) -> None:
            if len(sequences) >= beam:
                return
            if role_index == len(roles):
                sequences.append(list(current))
                return
            previous = current[-1]
            options = by_start.get((previous.pivot_stream_k, previous.end_pivot_index), [])
            for candidate in options:
                if candidate.pattern_type not in roles[role_index]:
                    continue
                if candidate.end_pivot_index is None or previous.end_pivot_index is None or candidate.end_pivot_index <= previous.end_pivot_index:
                    continue
                visit(current + [candidate], role_index + 1)
                if len(sequences) >= beam:
                    break

        visit([first], 1)
        return sequences

    def _validate_parent_family(
        self,
        family: str,
        points: list[Pivot],
        bars: pd.DataFrame,
    ) -> list[ValidationResult]:
        if family == "IMPULSE":
            return [validate_impulse(points, bars), validate_impulse(points, bars, truncated=True)]
        if family == "ZIGZAG":
            return [validate_zigzag(points, bars)]
        if family == "FLAT":
            return validate_flat(points, bars, flat_min_b=float(self.parameters["flat_min_b"]), running_enabled=False)
        if family == "TRIANGLE":
            return validate_triangle(points, bars, barrier_tolerance_fraction=float(self.parameters["barrier_tolerance_fraction"]))
        if family == "ENDING_DIAGONAL_CONTRACTING_33333":
            return [validate_ending_diagonal(points, bars)]
        return []

    def _rank_scenarios(
        self,
        nodes: dict[str, WaveNode],
        active_ids: set[str],
        bar_count: int,
    ) -> list[Scenario]:
        candidates = [nodes[node_id] for node_id in active_ids if node_id in nodes and nodes[node_id].pattern_type != "OBSERVED_LEAF"]
        candidates.sort(
            key=lambda node: (
                -node.verification_coverage,
                len(node.unknown_requirements),
                1 if node.pattern_type == "IMPULSE_TRUNCATED_5" else 0,
                -node.duration_bars,
                node.node_id,
            )
        )
        selected: list[WaveNode] = []
        seen_hypotheses: set[tuple] = set()
        for node in candidates:
            hypothesis = (
                node.pattern_type,
                pd.Timestamp(node.start_point["pivot_time"]).date().isoformat(),
                pd.Timestamp(node.end_point["pivot_time"]).date().isoformat(),
            )
            if hypothesis in seen_hypotheses:
                continue
            seen_hypotheses.add(hypothesis)
            selected.append(node)
            if len(selected) >= int(self.parameters["display_scenarios"]):
                break
        scenarios: list[Scenario] = []
        for idx, node in enumerate(selected, start=1):
            interval_fraction = min(1.0, node.duration_bars / max(1, bar_count - 1))
            coverage = float(node.verification_coverage) * interval_fraction
            ratio_values = [ratio.get("value") for ratio in node.ratios if ratio.get("value") is not None]
            fib_fit = _fib_fit(ratio_values) if ratio_values else None
            scenario_id = _stable_hash({"root": node.node_id, "rank_group": self.parameter_hash}, 20)
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    root_node_id=node.node_id,
                    status="ACTIVE",
                    rank=idx,
                    selection_label="Основной по правилам отбора" if idx == 1 else f"Альтернатива {idx - 1}",
                    coverage=coverage,
                    forming_coverage=interval_fraction if node.endpoint_status == "FORMING" else 0.0,
                    unknown_check_count=len(node.unknown_requirements),
                    complexity_units=1 if node.pattern_type in {"DOUBLE_ZIGZAG", "DOUBLE_THREE"} else 0,
                    exception_units=1 if node.pattern_type == "IMPULSE_TRUNCATED_5" else 0,
                    fib_fit=fib_fit,
                    fib_fit_mask=[str(ratio.get("ratio_id")) for ratio in node.ratios if ratio.get("value") is not None],
                    reason_selected="Best structural coverage with fewer unresolved mandatory checks; stable hash breaks exact ties.",
                    last_change_reason="Initial snapshot selection",
                )
            )
        return scenarios

    def _targets(self, result: ValidationResult, points: list[Pivot]) -> list[Target]:
        if result.pattern_type in {"IMPULSE", "IMPULSE_TRUNCATED_5"} and len(points) < 5:
            return []
        if result.pattern_type in {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED"} and len(points) < 3:
            return []
        d = result.direction
        q = [d * point.price for point in points]
        tolerance = float(self.parameters["extension_ratio_tolerance"])
        targets: list[Target] = []
        definitions: list[tuple[str, int, float, list[float], int]] = []
        if result.pattern_type in {"IMPULSE", "IMPULSE_TRUNCATED_5"}:
            definitions.append(("wave5_from_wave1", 4, q[1] - q[0], [0.618, 1.0, 1.618], 1))
            if bool(self.parameters["legacy_wave5_targets"]):
                definitions.append(("wave5_from_wave3_legacy", 4, q[3] - q[2], [0.382, 0.5, 0.618], 3))
        elif result.pattern_type == "ZIGZAG":
            definitions.append(("zigzag_C", 2, q[0] - q[1], [0.618, 1.0, 1.618], -1))
        elif result.pattern_type == "FLAT_REGULAR":
            definitions.append(("flat_C", 2, q[0] - q[1], [1.0, 1.236], -1))
        elif result.pattern_type == "FLAT_EXPANDED":
            definitions.append(("flat_C", 2, q[0] - q[1], [1.618, 2.0, 2.618], -1))
        for role, anchor_idx, reference, coefficients, direction_q in definitions:
            if reference <= 0:
                continue
            for coefficient in coefficients:
                center_q = q[anchor_idx] + direction_q * coefficient * reference
                half_width = tolerance * reference
                low_q, high_q = sorted((center_q - half_width, center_q + half_width))
                exclusion_reason: str | None = None
                if result.pattern_type == "IMPULSE":
                    if center_q <= q[3]:
                        exclusion_reason = "ordinary_wave5_center_must_exceed_wave3"
                    else:
                        low_q = max(low_q, float(np.nextafter(q[3], math.inf)))
                    l1 = q[1] - q[0]
                    l3 = q[3] - q[2]
                    if l3 < l1:
                        cap = q[4] + l3
                        if center_q > cap:
                            exclusion_reason = "wave5_center_exceeds_length_cap_when_wave3_is_shorter_than_wave1"
                        high_q = min(high_q, cap)
                elif result.pattern_type in {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED"}:
                    if center_q >= q[1]:
                        exclusion_reason = "correction_center_does_not_exceed_wave_A"
                    else:
                        high_q = min(high_q, float(np.nextafter(q[1], -math.inf)))
                if low_q > high_q:
                    exclusion_reason = exclusion_reason or "target_zone_empty_after_structural_clipping"
                low_price, high_price = sorted((d * low_q, d * high_q))
                target_id = _stable_hash({"role": role, "anchors": [p.pivot_id for p in points[: anchor_idx + 1]], "coefficient": coefficient}, 20)
                targets.append(
                    Target(
                        target_id=target_id,
                        node_id="pending",
                        role=role,
                        coefficient=coefficient,
                        price_low=float(low_price),
                        price_high=float(high_price),
                        center=float(d * center_q),
                        issued_at=points[anchor_idx].known_at,
                        known_at=points[anchor_idx].known_at,
                        anchor_ids=[point.pivot_id for point in points[: anchor_idx + 1]],
                        reference_wave_id=points[1].pivot_id if len(points) > 1 else None,
                        status="EXCLUDED" if exclusion_reason else "ACTIVE",
                        invalidation_scope=result.pattern_type,
                        exclusion_reason=exclusion_reason,
                    )
                )
        return targets

    def _channels(self, result: ValidationResult, points: list[Pivot]) -> list[dict[str, Any]]:
        if result.pattern_type not in {"IMPULSE", "IMPULSE_TRUNCATED_5", "ENDING_DIAGONAL_CONTRACTING_33333"} or len(points) < 5:
            return []
        return [
            {
                "channel_id": _stable_hash({"node_points": [p.pivot_id for p in points], "line": "1-3"}, 18),
                "kind": "LINE_1_3",
                "anchor_ids": [points[1].pivot_id, points[3].pivot_id],
                "anchors": [points[1].to_dict(), points[3].to_dict()],
                "parallel_through": points[2].to_dict(),
                "measurement_mode": "arithmetic",
            },
            {
                "channel_id": _stable_hash({"node_points": [p.pivot_id for p in points], "line": "2-4"}, 18),
                "kind": "LINE_2_4",
                "anchor_ids": [points[2].pivot_id, points[4].pivot_id],
                "anchors": [points[2].to_dict(), points[4].to_dict()],
                "parallel_through": points[3].to_dict(),
                "measurement_mode": "arithmetic",
            },
        ]

    def _events(
        self,
        streams: list[PivotStream],
        scenarios: list[Scenario],
        nodes: dict[str, WaveNode],
        as_of: str,
        search_statistics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for stream in streams:
            for pivot in stream.pivots:
                if pivot.status != "PIVOT_CONFIRMED":
                    continue
                key = (pivot.pivot_time, pivot.kind, round(pivot.price, 8), pivot.confirmed_at)
                if key in seen:
                    continue
                seen.add(key)
                events.append(
                    {
                        "event_id": _stable_hash(key, 18),
                        "event_type": "PIVOT_CONFIRMED",
                        "observed_at": pivot.pivot_time,
                        "known_at": pivot.known_at,
                        "pivot_id": pivot.pivot_id,
                    }
                )
        for scenario in scenarios:
            node = nodes.get(scenario.root_node_id)
            if node is None:
                continue
            if node.endpoint_status == "FORMING":
                events.append(
                    {
                        "event_id": _stable_hash({"wave_end_candidate": node.node_id, "known_at": node.known_at}, 18),
                        "event_type": "WAVE_END_CANDIDATE",
                        "observed_at": node.end_point.get("pivot_time"),
                        "known_at": node.known_at,
                        "node_id": node.node_id,
                        "affected_scenarios": [scenario.scenario_id],
                    }
                )
            if node.verified_depth > 0:
                events.append(
                    {
                        "event_id": _stable_hash({"verified": node.node_id, "depth": node.verified_depth}, 18),
                        "event_type": "STRUCTURE_VERIFIED_TO_DEPTH",
                        "observed_at": node.end_point.get("pivot_time"),
                        "known_at": node.known_at,
                        "node_id": node.node_id,
                        "verified_depth": node.verified_depth,
                        "affected_scenarios": [scenario.scenario_id],
                    }
                )
        if search_statistics["search_truncated"]:
            events.append(
                {
                    "event_id": _stable_hash({"search_limit": as_of}, 18),
                    "event_type": "SEARCH_LIMIT_REACHED",
                    "observed_at": as_of,
                    "known_at": as_of,
                }
            )
        if not scenarios:
            events.append(
                {
                    "event_id": _stable_hash({"unresolved": as_of}, 18),
                    "event_type": "UNRESOLVED",
                    "observed_at": as_of,
                    "known_at": as_of,
                }
            )
        return sorted(events, key=lambda item: (item["known_at"], item["event_id"]))


def _alternating(points: list[Pivot]) -> bool:
    return all(first.kind != second.kind for first, second in zip(points, points[1:]))


def _pivot_from_payload(payload: dict[str, Any]) -> Pivot:
    fields = {name: payload.get(name) for name in Pivot.__dataclass_fields__}
    fields["evidence_bar_ids"] = list(fields.get("evidence_bar_ids") or [])
    return Pivot(**fields)


def _remaining_unknowns(
    requirements: list[str],
    child_node_ids: list[str],
    node_store: dict[str, WaveNode],
) -> list[str]:
    """Resolve only requirements proven by structural child nodes.

    Direct geometry candidates are backed by OBSERVED_LEAF nodes, so they must
    retain every subdivision caveat.  A recursively assembled parent may clear
    subdivision-related caveats, but positional/context requirements still stay
    explicit until an ancestor proves them.
    """
    children = [node_store.get(node_id) for node_id in child_node_ids]
    structurally_verified = bool(children) and all(
        child is not None and child.pattern_type != "OBSERVED_LEAF" for child in children
    )
    if not structurally_verified:
        return list(requirements)

    structural_markers = (
        "subdivision",
        "components",
        "a_to_e",
        "five_zigzag",
        "five-wave",
        "three-wave",
        "internal_structure",
    )
    return [
        requirement
        for requirement in requirements
        if not any(marker in requirement.lower() for marker in structural_markers)
    ]


def _stable_hash(value: Any, length: int) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _fib_fit(values: list[float], tolerance: float = 0.10) -> float | None:
    if not values:
        return None
    coefficients = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618]
    scores = []
    for value in values:
        distance = min(abs(value - coefficient) for coefficient in coefficients)
        scores.append(math.exp(-0.5 * (distance / tolerance) ** 2))
    return float(100.0 * np.mean(scores))

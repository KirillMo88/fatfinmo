from __future__ import annotations

import copy
import hashlib
import json
from bisect import bisect_right
from typing import Any, Iterable

import pandas as pd

from .models import WaveNode


DEGREES = ("Major", "Intermediate", "Minor")
MOTIVE = {"IMPULSE", "IMPULSE_TRUNCATED_5", "ENDING_DIAGONAL_CONTRACTING_33333"}
CORRECTIVE = {
    "ZIGZAG",
    "FLAT_REGULAR",
    "FLAT_EXPANDED",
    "TRIANGLE_CONTRACTING",
    "TRIANGLE_BARRIER",
    "DOUBLE_ZIGZAG",
    "DOUBLE_THREE",
}
TRIANGLES = {"TRIANGLE_CONTRACTING", "TRIANGLE_BARRIER"}


class WaveMapBuilder:
    """Assemble local V2 candidates into one historical three-degree map."""

    def build(
        self,
        nodes: dict[str, WaveNode] | Iterable[WaveNode],
        bars: pd.DataFrame,
        *,
        analysis_window: str,
        analysis_start: str,
    ) -> dict[str, Any]:
        node_values = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        start_time = pd.to_datetime(analysis_start, utc=True)
        candidates = [self._map_payload(node) for node in node_values if self._eligible(node, start_time)]
        by_degree = {degree: [node for node in candidates if node["degree"] == degree] for degree in DEGREES}
        selected: dict[str, list[dict[str, Any]]] = {}
        selected["Major"] = self._resolve_degree(by_degree["Major"], len(bars))
        selected["Intermediate"] = self._resolve_degree(
            by_degree["Intermediate"], len(bars), parents=selected["Major"]
        )
        selected["Minor"] = self._resolve_degree(
            by_degree["Minor"], len(bars), parents=selected["Intermediate"]
        )

        edges: list[dict[str, Any]] = []
        for degree in DEGREES:
            ordered = selected[degree]
            for first, second in zip(ordered, ordered[1:]):
                edges.append(_edge("NEXT_SIBLING", first["node_id"], second["node_id"], degree=degree))

        selected_lookup = {node["node_id"]: node for degree in DEGREES for node in selected[degree]}
        self._attach_degree(selected["Major"], selected["Intermediate"], edges)
        self._attach_degree(selected["Intermediate"], selected["Minor"], edges)

        alternative_edges: list[dict[str, Any]] = []
        alternatives_by_selected: dict[str, list[dict[str, Any]]] = {}
        for degree in DEGREES:
            chosen_ids = {node["node_id"] for node in selected[degree]}
            for chosen in selected[degree]:
                conflicts = [
                    node
                    for node in by_degree[degree]
                    if node["node_id"] not in chosen_ids and _overlap(node, chosen)
                ]
                conflicts.sort(key=lambda node: (-self._candidate_score(node), node["node_id"]))
                alternatives_by_selected[chosen["node_id"]] = conflicts[:3]
                for alternative in conflicts[:3]:
                    alternative_edges.append(
                        _edge("ALTERNATIVE_TO", alternative["node_id"], chosen["node_id"], degree=degree)
                    )
        edges.extend(alternative_edges)
        alternative_intermediate = _unique_nodes(
            node
            for alternatives in alternatives_by_selected.values()
            for node in alternatives
            if node.get("degree") == "Intermediate"
        )
        alternative_minor = _unique_nodes(
            node
            for alternatives in alternatives_by_selected.values()
            for node in alternatives
            if node.get("degree") == "Minor"
        )
        self._attach_degree(selected["Major"], alternative_intermediate, edges)
        self._attach_degree(selected["Intermediate"], alternative_minor, edges)

        active_major = self._latest(selected["Major"])
        active_intermediate = (
            self._latest_child(selected["Intermediate"], active_major) or self._latest(selected["Intermediate"])
            if active_major is not None
            else self._latest(selected["Intermediate"])
        )
        active_minor = (
            self._latest_child(selected["Minor"], active_intermediate) or self._latest(selected["Minor"])
            if active_intermediate is not None
            else self._latest(selected["Minor"])
        )
        active_ids = {
            node["node_id"]
            for node in (active_major, active_intermediate, active_minor)
            if node is not None
        }
        for degree in DEGREES:
            for node in selected[degree]:
                if node["node_id"] in active_ids:
                    node["map_status"] = "ACTIVE"
                elif node.get("endpoint_status") == "PIVOT_CONFIRMED":
                    node["map_status"] = "HISTORICAL"
                else:
                    node["map_status"] = "COMPLETED"

        selected_ids = [node["node_id"] for degree in DEGREES for node in selected[degree]]
        main_scenario = (
            {
                "scenario_id": _stable_hash({"window": analysis_window, "nodes": selected_ids}, 20),
                "selection_label": "Основная Wave Map",
                "status": "ACTIVE",
                "node_ids": selected_ids,
                "active_path": [node_id for node_id in (
                    active_major and active_major["node_id"],
                    active_intermediate and active_intermediate["node_id"],
                    active_minor and active_minor["node_id"],
                ) if node_id],
            }
            if selected_ids
            else None
        )
        scenarios = [main_scenario] if main_scenario else []
        for active in (active_minor, active_intermediate, active_major):
            if active is None:
                continue
            for alternative in alternatives_by_selected.get(active["node_id"], [])[: 3 - len(scenarios)]:
                if active.get("parent_wave_id") != alternative.get("parent_wave_id"):
                    continue
                alternative_ids = [alternative["node_id"] if node_id == active["node_id"] else node_id for node_id in selected_ids]
                scenarios.append(
                    {
                        "scenario_id": _stable_hash({"window": analysis_window, "nodes": alternative_ids}, 20),
                        "selection_label": f"Альтернатива {len(scenarios)}",
                        "status": "ACTIVE",
                        "node_ids": alternative_ids,
                        "active_path": [
                            alternative["node_id"] if node_id == active["node_id"] else node_id
                            for node_id in (main_scenario or {}).get("active_path", [])
                        ],
                    }
                )
            if len(scenarios) >= 3:
                break

        output_nodes: dict[str, dict[str, Any]] = {node_id: node for node_id, node in selected_lookup.items()}
        for alternatives in alternatives_by_selected.values():
            for node in alternatives:
                node.setdefault("map_status", "ALTERNATIVE")
                output_nodes.setdefault(node["node_id"], node)

        unresolved = self._unresolved_intervals(selected["Major"], bars, start_time)
        historical = [node_id for node_id in selected_ids if node_id not in active_ids]
        return {
            "root_scenarios": scenarios,
            "main_root_scenario_id": main_scenario and main_scenario["scenario_id"],
            "wave_nodes": list(output_nodes.values()),
            "structural_edges": edges,
            "active_major_node_id": active_major and active_major["node_id"],
            "active_intermediate_node_id": active_intermediate and active_intermediate["node_id"],
            "active_minor_node_id": active_minor and active_minor["node_id"],
            "historical_completed_node_ids": historical,
            "unresolved_intervals": unresolved,
            "wave_map_statistics": {
                "candidate_count_by_degree": {degree: len(by_degree[degree]) for degree in DEGREES},
                "selected_count_by_degree": {degree: len(selected[degree]) for degree in DEGREES},
                "conflict_alternative_count": len(alternative_edges),
                "analysis_window": analysis_window,
                "analysis_start": pd.Timestamp(start_time).isoformat(),
            },
        }

    def _eligible(self, node: WaveNode, analysis_start: pd.Timestamp) -> bool:
        if node.pattern_type == "OBSERVED_LEAF" or node.geometry_status not in {"VALID", "PENDING"}:
            return False
        start = pd.to_datetime(node.start_point.get("pivot_time"), errors="coerce", utc=True)
        end = pd.to_datetime(node.end_point.get("pivot_time"), errors="coerce", utc=True)
        return pd.notna(start) and pd.notna(end) and start >= analysis_start and end >= start

    def _map_payload(self, node: WaveNode) -> dict[str, Any]:
        payload = copy.deepcopy(node.to_dict())
        payload["degree"] = _degree_for_k(node.pivot_stream_k)
        payload["relative_degree"] = payload["degree"]
        if payload.get("invalidation"):
            payload["invalidation"]["model_scope"] = payload["invalidation"].get("scope")
            payload["invalidation"]["scope"] = payload["degree"]
        for target in payload.get("targets", []):
            target["invalidation_scope"] = payload["degree"]
            target["target_scope"] = payload["degree"]
        payload["map_status"] = "CANDIDATE"
        payload["parent_wave_id"] = None
        payload["child_wave_ids"] = []
        payload["wave_role"] = None
        payload["context_status"] = "PARENT_UNRESOLVED"
        return payload

    def _resolve_degree(
        self,
        candidates: list[dict[str, Any]],
        bar_count: int,
        *,
        parents: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for node in candidates:
            key = (
                node.get("pattern_type"),
                node.get("start_pivot_index"),
                node.get("end_pivot_index"),
                node.get("subtype"),
            )
            current = deduped.get(key)
            if current is None or self._candidate_score(node, parents) > self._candidate_score(current, parents):
                deduped[key] = node
        ordered = sorted(deduped.values(), key=lambda node: (_end(node), _start(node), node["node_id"]))
        ends = [_end(node) for node in ordered]
        predecessors = [bisect_right(ends, _start(node), hi=index) - 1 for index, node in enumerate(ordered)]
        scores = [self._candidate_score(node, parents) for node in ordered]
        best = [0.0] * (len(ordered) + 1)
        take = [False] * len(ordered)
        for index, score in enumerate(scores, start=1):
            include = score + best[predecessors[index - 1] + 1]
            exclude = best[index - 1]
            if include > exclude:
                best[index] = include
                take[index - 1] = True
            else:
                best[index] = exclude
        selected: list[dict[str, Any]] = []
        index = len(ordered) - 1
        while index >= 0:
            include = scores[index] + best[predecessors[index] + 1]
            if take[index] and include >= best[index]:
                selected.append(ordered[index])
                index = predecessors[index]
            else:
                index -= 1
        selected.reverse()

        right_edge = max(0, bar_count - 1)
        right_candidates = [node for node in ordered if _end(node) >= right_edge - max(10, int(bar_count * 0.03))]
        if right_candidates and not any(node in selected for node in right_candidates):
            active = max(right_candidates, key=lambda node: (self._candidate_score(node, parents), node["node_id"]))
            selected = [node for node in selected if not _overlap(node, active)] + [active]
            selected.sort(key=lambda node: (_start(node), _end(node), node["node_id"]))
        return selected

    def _candidate_score(
        self,
        node: dict[str, Any],
        parents: list[dict[str, Any]] | None = None,
    ) -> float:
        duration = max(1, _end(node) - _start(node))
        verified = int(node.get("verified_depth") or 0)
        coverage = float(node.get("verification_coverage") or 0.0)
        unknown = len(node.get("unknown_requirements") or [])
        complexity = 1 if node.get("pattern_type") in {"DOUBLE_ZIGZAG", "DOUBLE_THREE"} else 0
        forming_penalty = 2 if node.get("endpoint_status") == "FORMING" else 0
        parent_compatible = any(
            _contains(parent, node)
            and (role := _role_for_child(parent, node)) is not None
            and _grammar_allows(parent.get("pattern_type"), role, node.get("pattern_type"))
            for parent in (parents or [])
        )
        context_bonus = 10_000.0 if parent_compatible else 0.0
        return (
            context_bonus
            + verified * 100.0
            + coverage * 100.0
            - unknown * 10.0
            + min(duration, 100) * 0.5
            - complexity * 3.0
            - forming_penalty
        )

    def _attach_degree(
        self,
        parents: list[dict[str, Any]],
        children: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        for child in children:
            containing = [parent for parent in parents if _contains(parent, child)]
            containing.sort(key=lambda node: (_end(node) - _start(node), node["node_id"]))
            for parent in containing:
                role = _role_for_child(parent, child)
                if role is None or not _grammar_allows(parent.get("pattern_type"), role, child.get("pattern_type")):
                    continue
                child["parent_wave_id"] = parent["node_id"]
                child["wave_role"] = role
                child["context_status"] = "RESOLVED"
                parent.setdefault("child_wave_ids", []).append(child["node_id"])
                parent["context_status"] = "RESOLVED"
                edges.append(_edge("PARENT_CHILD", parent["node_id"], child["node_id"], wave_role=role))
                break

    def _latest(self, nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
        return max(nodes, key=lambda node: (_end(node), _start(node), node["node_id"])) if nodes else None

    def _latest_child(
        self,
        nodes: list[dict[str, Any]],
        parent: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if parent is None:
            return None
        children = [node for node in nodes if node.get("parent_wave_id") == parent["node_id"]]
        return self._latest(children)

    def _unresolved_intervals(
        self,
        major_nodes: list[dict[str, Any]],
        bars: pd.DataFrame,
        analysis_start: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        if bars.empty:
            return []
        timestamps = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
        last_index = len(bars) - 1
        intervals: list[tuple[int, int]] = []
        cursor = int(timestamps.searchsorted(analysis_start, side="left"))
        for node in sorted(major_nodes, key=lambda item: (_start(item), _end(item))):
            if _start(node) > cursor:
                intervals.append((cursor, _start(node)))
            cursor = max(cursor, _end(node))
        if cursor < last_index:
            intervals.append((cursor, last_index))
        return [
            {
                "status": "UNRESOLVED",
                "start_bar_index": start,
                "end_bar_index": end,
                "start_time": pd.Timestamp(timestamps.iloc[start]).isoformat(),
                "end_time": pd.Timestamp(timestamps.iloc[end]).isoformat(),
            }
            for start, end in intervals
            if 0 <= start < len(timestamps) and 0 <= end < len(timestamps) and end > start
        ]


def _degree_for_k(value: float | None) -> str:
    k = float(value or 0.0)
    if k >= 5.0:
        return "Major"
    if k >= 2.5:
        return "Intermediate"
    return "Minor"


def _start(node: dict[str, Any]) -> int:
    return int(node.get("start_pivot_index") or 0)


def _end(node: dict[str, Any]) -> int:
    return int(node.get("end_pivot_index") or 0)


def _overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return max(_start(first), _start(second)) < min(_end(first), _end(second))


def _contains(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    return _start(parent) <= _start(child) and _end(child) <= _end(parent) and parent["node_id"] != child["node_id"]


def _role_for_child(parent: dict[str, Any], child: dict[str, Any]) -> str | None:
    labels = sorted(parent.get("labels") or [], key=lambda label: int(label.get("bar_index") or 0))
    for first, second in zip(labels, labels[1:]):
        first_index = int(first.get("bar_index") or 0)
        second_index = int(second.get("bar_index") or 0)
        if first_index <= _start(child) and _end(child) <= second_index:
            return str(second.get("label"))
    return None


def _grammar_allows(parent_pattern: str | None, role: str, child_pattern: str | None) -> bool:
    child = str(child_pattern)
    parent = str(parent_pattern)
    if parent in {"IMPULSE", "IMPULSE_TRUNCATED_5"}:
        if role in {"1", "3"}:
            return child == "IMPULSE"
        if role == "5":
            return child in MOTIVE
        if role == "2":
            return child in CORRECTIVE - TRIANGLES
        if role == "4":
            return child in CORRECTIVE
    if parent == "ZIGZAG":
        return child in MOTIVE if role in {"A", "C"} else child in CORRECTIVE
    if parent in {"FLAT_REGULAR", "FLAT_EXPANDED"}:
        if role == "A":
            return child in CORRECTIVE - TRIANGLES
        return child in MOTIVE if role == "C" else child in CORRECTIVE
    if parent in TRIANGLES:
        return child in {"ZIGZAG", "DOUBLE_ZIGZAG"}
    if parent == "ENDING_DIAGONAL_CONTRACTING_33333":
        return child == "ZIGZAG"
    if parent == "DOUBLE_ZIGZAG":
        return child == "ZIGZAG" if role in {"W", "Y"} else child in CORRECTIVE
    if parent == "DOUBLE_THREE":
        if role == "W":
            return child in {"ZIGZAG", "FLAT_REGULAR", "FLAT_EXPANDED"}
        return child in CORRECTIVE
    return False


def _edge(edge_type: str, source: str, target: str, **metadata: Any) -> dict[str, Any]:
    payload = {"edge_type": edge_type, "source_node_id": source, "target_node_id": target, **metadata}
    payload["edge_id"] = _stable_hash(payload, 18)
    return payload


def _unique_nodes(nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for node in nodes:
        unique.setdefault(str(node.get("node_id")), node)
    return list(unique.values())


def _stable_hash(value: Any, length: int) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

import elliott_waves.export as export_module
from elliott_waves.models import WaveNode
from elliott_waves.service import _merge_wave_map_history
from elliott_waves.summary import build_summary
from elliott_waves.wave_map import WaveMapBuilder


def _bars(count: int = 41) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=count, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": range(100, 100 + count),
            "high": range(101, 101 + count),
            "low": range(99, 99 + count),
            "close": range(100, 100 + count),
            "is_closed": True,
        }
    )


def _node(
    node_id: str,
    pattern: str,
    start: int,
    end: int,
    k: float,
    labels: list[tuple[int, str]],
    *,
    forming: bool = False,
) -> WaveNode:
    dates = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    label_rows = [
        {
            "bar_index": index,
            "pivot_time": dates[index].isoformat(),
            "price": float(100 + index),
            "label": label,
            "status": "PIVOT_CONFIRMED",
            "confirmed_at": dates[index].isoformat(),
        }
        for index, label in labels
    ]
    return WaveNode(
        node_id=node_id,
        pattern_type=pattern,
        subtype=None,
        profile_id="TEST",
        direction=1,
        orientation_direction=1,
        relative_degree="D0",
        start_point={"pivot_time": dates[start].isoformat(), "price": 100 + start},
        end_point={"pivot_time": dates[end].isoformat(), "price": 100 + end},
        internal_high=float(101 + end),
        internal_low=float(99 + start),
        extreme_times={},
        source_bar_range=[str(start), str(end)],
        source_timeframes=["1D"],
        duration_bars=end - start,
        duration_calendar=end - start,
        endpoint_status="FORMING" if forming else "PIVOT_CONFIRMED",
        geometry_status="PENDING" if forming else "VALID",
        context_status="UNRESOLVED",
        subdivision_status="PARTIAL" if forming else "VERIFIED",
        verified_depth=2,
        verification_coverage=0.8,
        known_at=dates[end].isoformat(),
        first_observed_at=dates[end].isoformat(),
        last_updated_at=dates[end].isoformat(),
        labels=label_rows,
        pivot_stream_k=k,
        start_pivot_index=start,
        end_pivot_index=end,
        invalidation={"scope": "Minor" if k < 2.5 else "Intermediate", "level": 95.0, "basis": "High/Low"},
        targets=[{
            "target_id": f"t-{node_id}", "node_id": node_id, "role": "next", "coefficient": 1.0,
            "price_low": 140.0, "price_high": 145.0, "status": "ACTIVE", "invalidation_scope": "Minor",
        }],
    )


def _tree_nodes() -> list[WaveNode]:
    return [
        _node("major-wxy", "DOUBLE_THREE", 0, 30, 6.0, [(0, "S"), (10, "W"), (20, "X"), (30, "Y")]),
        _node("intermediate-abc", "ZIGZAG", 20, 30, 3.0, [(20, "S"), (23, "A"), (25, "B"), (30, "C")]),
        _node("minor-impulse", "IMPULSE", 25, 30, 1.5, [(25, "0"), (26, "1"), (27, "2"), (28, "3"), (29, "4"), (30, "5")], forming=True),
    ]


def _map(nodes: list[WaveNode] | None = None) -> dict:
    return WaveMapBuilder().build(
        nodes or _tree_nodes(),
        _bars(),
        analysis_window="5Y",
        analysis_start="2020-01-01T00:00:00+00:00",
    )


def test_map_01_local_abc_is_nested_in_larger_wxy():
    result = _map()
    lookup = {node["node_id"]: node for node in result["wave_nodes"]}
    assert lookup["intermediate-abc"]["parent_wave_id"] == "major-wxy"
    assert {node["degree"] for node in lookup.values()} == {"Major", "Intermediate", "Minor"}


def test_map_02_completed_minor_remains_historical():
    nodes = _tree_nodes() + [
        _node("minor-old", "ZIGZAG", 20, 25, 1.5, [(20, "S"), (22, "A"), (23, "B"), (25, "C")])
    ]
    result = _map(nodes)
    assert "minor-old" in result["historical_completed_node_ids"]
    assert result["active_minor_node_id"] == "minor-impulse"


def test_map_03_timeframe_projection_is_snapshot_only():
    source = Path("elliott_waves_tab.py").read_text(encoding="utf-8")
    assert "read_chart_bars" in source
    assert "ElliottWaveEngine" not in source
    assert 'chart_timeframe in {"1W", "1M"}' in source


def test_map_04_viewport_is_not_engine_input():
    source = Path("elliott_waves/engine.py").read_text(encoding="utf-8")
    assert "viewport" not in source
    assert "zoom" not in source
    assert "analysis_window" in source


def test_map_05_degree_toggles_are_display_only():
    source = Path("elliott_waves_tab.py").read_text(encoding="utf-8")
    for key in ("elliott_major_", "elliott_intermediate_", "elliott_minor_"):
        assert key in source
    assert "visible_degrees" in source


def test_map_06_minor_invalidation_is_scope_aware():
    result = _map()
    lookup = {node["node_id"]: node for node in result["wave_nodes"]}
    assert lookup[result["active_minor_node_id"]]["invalidation"]["scope"] == "Minor"
    assert result["active_major_node_id"] == "major-wxy"


def test_map_07_conflicting_same_degree_candidate_is_alternative():
    nodes = _tree_nodes() + [
        _node("minor-conflict", "ENDING_DIAGONAL_CONTRACTING_33333", 25, 30, 1.5, [(25, "0"), (26, "1"), (27, "2"), (28, "3"), (29, "4"), (30, "5")])
    ]
    result = _map(nodes)
    main_ids = set(result["root_scenarios"][0]["node_ids"])
    assert len({"minor-impulse", "minor-conflict"} & main_ids) == 1
    assert any(edge["edge_type"] == "ALTERNATIVE_TO" for edge in result["structural_edges"])


def test_map_08_cross_degree_overlap_requires_parent_child_edge():
    result = _map()
    pairs = {(edge["source_node_id"], edge["target_node_id"]) for edge in result["structural_edges"] if edge["edge_type"] == "PARENT_CHILD"}
    assert ("major-wxy", "intermediate-abc") in pairs
    assert ("intermediate-abc", "minor-impulse") in pairs


def test_map_09_removed_history_becomes_superseded():
    old = {"wave_nodes": [{**_map()["wave_nodes"][0], "node_id": "old-major"}]}
    current = {"analysis_start": "2020-01-01T00:00:00+00:00", **_map()}
    _merge_wave_map_history(old, current)
    carried = next(node for node in current["wave_nodes"] if node["node_id"] == "old-major")
    assert carried["map_status"] == "SUPERSEDED"
    assert any(edge["edge_type"] == "SUPERSEDES" for edge in current["structural_edges"])
    assert any(event["event_type"] == "NODE_SUPERSEDED" for event in current["events"])


def test_map_10_unresolved_gaps_are_explicit():
    result = _map([_node("major-late", "DOUBLE_THREE", 10, 30, 6.0, [(10, "S"), (16, "W"), (22, "X"), (30, "Y")])])
    assert result["unresolved_intervals"]
    assert result["unresolved_intervals"][0]["status"] == "UNRESOLVED"


def test_map_11_local_flat_is_child_not_root():
    nodes = _tree_nodes()[:1] + [
        _node("intermediate-flat", "FLAT_REGULAR", 20, 30, 3.0, [(20, "S"), (23, "A"), (25, "B"), (30, "C")])
    ]
    result = _map(nodes)
    flat = next(node for node in result["wave_nodes"] if node["node_id"] == "intermediate-flat")
    assert flat["parent_wave_id"] == "major-wxy"


def test_map_12_summary_mentions_all_active_degrees():
    result = _map()
    snapshot = {**result, "unresolved_reasons": []}
    text = build_summary(snapshot, result["main_root_scenario_id"])
    assert all(degree in text for degree in ("Major", "Intermediate", "Minor"))


def test_map_13_targets_and_invalidation_have_scope():
    result = _map()
    active = {node["node_id"]: node for node in result["wave_nodes"]}[result["active_minor_node_id"]]
    assert active["invalidation"]["scope"] == "Minor"
    assert active["targets"][0]["invalidation_scope"] == "Minor"


def test_map_14_alternative_switch_replaces_consistent_tree_member():
    nodes = _tree_nodes() + [
        _node("minor-conflict", "ENDING_DIAGONAL_CONTRACTING_33333", 25, 30, 1.5, [(25, "0"), (26, "1"), (27, "2"), (28, "3"), (29, "4"), (30, "5")])
    ]
    result = _map(nodes)
    assert len(result["root_scenarios"]) >= 2
    main, alternative = result["root_scenarios"][:2]
    assert len(main["node_ids"]) == len(alternative["node_ids"])
    assert len(set(main["node_ids"]) ^ set(alternative["node_ids"])) == 2


def test_map_15_exports_hierarchy_and_historical_nodes():
    result = _map(_tree_nodes() + [_node("minor-old", "ZIGZAG", 20, 25, 1.5, [(20, "S"), (22, "A"), (23, "B"), (25, "C")])])
    snapshot = {
        "canonical_asset_id": "SPX", "snapshot_id": "map-15", "parameters": {}, "pivot_streams": [],
        "events": [], "analysis_window": "5Y", "analysis_start": "2020-01-01T00:00:00+00:00", **result,
    }
    manifest = {"manifest_id": "m15", "published_at": "2020-02-10"}
    settings = {"SPX": {"scenario_id": result["main_root_scenario_id"], "chart_timeframe": "1D", "analysis_window": "5Y"}}
    original_base, original_chart = export_module.read_base_bars, export_module.read_chart_bars
    export_module.read_base_bars = lambda *_: _bars()
    export_module.read_chart_bars = lambda *_: _bars()
    try:
        json_payload = json.loads(export_module.build_json_export(manifest, {"SPX": snapshot}, settings))
        xlsx_payload = export_module.build_xlsx_export(manifest, {"SPX": snapshot}, settings)
        edges = pd.read_excel(io.BytesIO(xlsx_payload), sheet_name="StructuralEdges")
        nodes = pd.read_excel(io.BytesIO(xlsx_payload), sheet_name="WaveNodes")
    finally:
        export_module.read_base_bars, export_module.read_chart_bars = original_base, original_chart
    assert json_payload["assets"]["SPX"]["snapshot"]["structural_edges"]
    assert "PARENT_CHILD" in set(edges["edge_type"])
    assert "minor-old" in set(nodes["node_id"])


if __name__ == "__main__":
    failures = []
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_map_") and callable(value)]
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(1)
    print(f"PASS all {len(tests)} Wave Map tests")

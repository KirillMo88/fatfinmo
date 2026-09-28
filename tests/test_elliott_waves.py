from __future__ import annotations

import copy
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from elliott_waves.config import ASSET_SPECS
from elliott_waves.data import aggregate_daily_bars, data_version
from elliott_waves.engine import ElliottWaveEngine
import elliott_waves.export as export_module
import elliott_waves.storage as storage_module
from elliott_waves.lifecycle import evaluate_lifecycle
from elliott_waves.models import Pivot, WaveNode
from elliott_waves.pivots import build_causal_pivot_streams
from elliott_waves.summary import build_summary
from elliott_waves.validators import (
    validate_double_correction,
    validate_ending_diagonal,
    validate_flat,
    validate_impulse,
    validate_triangle,
    validate_zigzag,
)


def make_points(prices: list[float], start: str = "2020-01-01") -> list[Pivot]:
    dates = pd.date_range(start, periods=len(prices), freq="7D")
    points = []
    for idx, (date, price) in enumerate(zip(dates, prices)):
        kind = "LOW" if idx % 2 == 0 else "HIGH"
        points.append(
            Pivot(
                pivot_id=f"p{idx}",
                pivot_time=date.isoformat(),
                confirmed_at=(date + pd.Timedelta(days=1)).isoformat(),
                known_at=(date + pd.Timedelta(days=1)).isoformat(),
                price=float(price),
                kind=kind,
                status="PIVOT_CONFIRMED",
                source_timeframe="1D",
                k=1.5,
                atr_reference=1.0,
                evidence_bar_ids=[f"b{idx}"],
                bar_index=idx,
            )
        )
    return points


def bars_from_prices(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="7D")
    return pd.DataFrame(
        {
            "bar_id": [f"b{i}" for i in range(len(prices))],
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "is_closed": True,
        }
    )


def normalized_daily(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "bar_id": [f"b{i}" for i in range(len(prices))],
            "timestamp": dates,
            "period_start": dates,
            "period_end": dates + pd.Timedelta(days=1),
            "open": prices,
            "high": np.asarray(prices) + 0.5,
            "low": np.asarray(prices) - 0.5,
            "close": prices,
            "volume": np.nan,
            "timeframe": "1D",
            "is_closed": True,
            "source_id": "test",
            "provider_symbol": "TEST",
            "source_timeframe": "1D",
            "last_source_bar_time": dates,
            "lineage": [[f"b{i}"] for i in range(len(prices))],
        }
    )


def failed(result, rule_id: str) -> bool:
    return any(check.rule_id == rule_id and check.result == "FAIL" for check in result.checks)


def lifecycle_node(target: dict, *, direction: int = 1, pattern_type: str = "IMPULSE", invalidation: float = 90) -> WaveNode:
    return WaveNode(
        node_id="n1",
        pattern_type=pattern_type,
        subtype=None,
        profile_id="TEST",
        direction=direction,
        orientation_direction=direction,
        relative_degree="D0",
        start_point={},
        end_point={},
        internal_high=0,
        internal_low=0,
        extreme_times={},
        source_bar_range=[],
        source_timeframes=["1D"],
        duration_bars=0,
        duration_calendar=0,
        endpoint_status="FORMING",
        geometry_status="VALID",
        context_status="UNRESOLVED",
        subdivision_status="UNVERIFIED",
        verified_depth=0,
        verification_coverage=0,
        known_at="2020-01-01T00:00:00+00:00",
        first_observed_at="2020-01-01T00:00:00+00:00",
        last_updated_at="2020-01-01T00:00:00+00:00",
        invalidation={"level": invalidation},
        targets=[target],
    )


def test_asset_contracts_use_confirmed_symbols_and_timeframes():
    assert ASSET_SPECS["SPX"].provider_symbol == "^GSPC"
    assert ASSET_SPECS["NDX"].provider_symbol == "^NDX"
    assert ASSET_SPECS["GOLD"].provider_symbol == "TVC:GOLD"
    assert ASSET_SPECS["GOLD"].instrument_type == "cfd_index_proxy"
    assert ASSET_SPECS["GOLD"].base_timeframe == "1W"
    assert ASSET_SPECS["BTCUSD"].provider_symbol == "INDEX:BTCUSD"
    assert ASSET_SPECS["BTCUSD"].base_timeframe == "1W"


def test_daily_aggregation_preserves_missing_volume_and_calendar_periods():
    daily = normalized_daily([100, 101, 99, 104, 103, 106, 107, 108, 110, 109])
    weekly = aggregate_daily_bars(daily, "1W", ASSET_SPECS["SPX"])
    monthly = aggregate_daily_bars(daily, "1M", ASSET_SPECS["SPX"])
    assert weekly.iloc[0]["open"] == 100
    assert weekly.iloc[0]["high"] == 101.5
    assert weekly.iloc[0]["low"] == 98.5
    assert pd.isna(weekly.iloc[0]["volume"])
    assert len(monthly) == 1
    assert monthly.iloc[0]["close"] == 109


def test_t01_basic_impulse_geometry_passes_without_claiming_subdivision():
    result = validate_impulse(make_points([100, 120, 110, 150, 130, 154.72]), bars_from_prices([100, 120, 110, 150, 130, 154.72]))
    assert result.valid
    assert "wave3_subdivision" in result.unknown_requirements


def test_t02_wave2_exact_return_to_zero_invalidates_impulse():
    result = validate_impulse(make_points([100, 120, 100, 150, 130, 155]), bars_from_prices([100, 120, 100, 150, 130, 155]))
    assert failed(result, "IMPULSE_2_NO_FULL_RETRACE")


def test_t03_wick_breach_is_not_ignored():
    points = make_points([100, 120, 110, 150, 130, 155])
    bars = bars_from_prices([100, 120, 110, 150, 130, 155])
    bars.loc[2, "low"] = 99
    result = validate_impulse(points, bars)
    assert failed(result, "IMPULSE_2_NO_FULL_RETRACE")


def test_t04_wave4_overlap_invalidates_ordinary_impulse():
    result = validate_impulse(make_points([100, 120, 110, 150, 119, 155]), bars_from_prices([100, 120, 110, 150, 119, 155]))
    assert failed(result, "IMPULSE_ENDPOINT_1_4_3") or failed(result, "IMPULSE_4_NO_WAVE1_RANGE_OVERLAP")


def test_t05_wave4_checks_full_wave1_range_not_only_endpoint():
    points = make_points([100, 120, 110, 150, 123, 155])
    bars = bars_from_prices([100, 120, 110, 150, 123, 155])
    bars.loc[1, "high"] = 125
    result = validate_impulse(points, bars)
    assert failed(result, "IMPULSE_4_NO_WAVE1_RANGE_OVERLAP")


def test_t06_third_wave_cannot_be_shortest():
    result = validate_impulse(make_points([100, 130, 120, 140, 135, 160]), bars_from_prices([100, 130, 120, 140, 135, 160]))
    assert failed(result, "IMPULSE_3_NOT_SHORTEST")


def test_t07_large_third_extension_is_not_a_hard_failure():
    result = validate_impulse(make_points([100, 110, 105, 140, 130, 145]), bars_from_prices([100, 110, 105, 140, 130, 145]))
    assert result.valid


def test_t08_incompatible_wave5_target_is_exported_as_excluded():
    points = make_points([100, 120, 110, 150, 121, 155])
    result = validate_impulse(points, bars_from_prices([100, 120, 110, 150, 121, 155]))
    targets = ElliottWaveEngine()._targets(result, points)
    assert any(target.status == "EXCLUDED" and "wave3" in str(target.exclusion_reason) for target in targets)


def test_t09_fifth_length_cap_when_third_is_shorter_than_first():
    result = validate_impulse(make_points([100, 130, 115, 140, 135, 165]), bars_from_prices([100, 130, 115, 140, 135, 165]))
    assert failed(result, "IMPULSE_5_LENGTH_CAP")


def test_t10_ordinary_zigzag_rejects_b_crossing_start():
    result = validate_zigzag(make_points([150, 130, 155, 125]), bars_from_prices([150, 130, 155, 125]))
    assert not result.valid
    assert failed(result, "ZIGZAG_B_BETWEEN_A_START")


def test_t11_expanded_flat_has_separate_validator():
    outputs = validate_flat(make_points([150, 130, 155, 125]), bars_from_prices([150, 130, 155, 125]))
    expanded = next(result for result in outputs if result.pattern_type == "FLAT_EXPANDED")
    assert expanded.valid
    assert "A_corrective_subdivision" in expanded.unknown_requirements


def test_t13_zigzag_target_that_does_not_clear_a_is_excluded():
    points = make_points([100, 80, 92.36, 70])
    result = validate_zigzag(points, bars_from_prices([100, 80, 92.36, 70]))
    targets = ElliottWaveEngine()._targets(result, points)
    target_0618 = next(target for target in targets if target.coefficient == 0.618)
    assert target_0618.status == "EXCLUDED"
    assert target_0618.exclusion_reason == "correction_center_does_not_exceed_wave_A"


def test_t14_running_flat_is_disabled_by_default():
    outputs = validate_flat(make_points([150, 130, 155, 140]))
    assert all(result.pattern_type != "FLAT_RUNNING" for result in outputs)


def test_t15_triangle_c_only_is_not_a_complete_triangle():
    try:
        validate_triangle(make_points([150, 130, 146, 135]))
    except ValueError:
        return
    raise AssertionError("A triangle ending at C must not validate as complete")


def test_t16_triangle_e_line_throw_through_is_not_a_standalone_failure():
    outputs = validate_triangle(make_points([150, 130, 146, 135, 143, 138]), bars_from_prices([150, 130, 146, 135, 143, 138]))
    assert any(result.valid for result in outputs)
    assert all(check.rule_id != "TRIANGLE_E_AC_LINE" for result in outputs for check in result.checks)


def test_t19_double_zigzag_requires_two_zigzags():
    points = make_points([150, 130, 142, 120])
    valid = validate_double_correction(points, ["ZIGZAG", "FLAT_REGULAR", "ZIGZAG"], family="DOUBLE_ZIGZAG")
    invalid = validate_double_correction(points, ["FLAT_REGULAR", "ZIGZAG", "ZIGZAG"], family="DOUBLE_ZIGZAG")
    assert valid.valid
    assert not invalid.valid


def test_t20_triangle_cannot_be_w_of_double_three():
    result = validate_double_correction(
        make_points([150, 130, 142, 125]),
        ["TRIANGLE_CONTRACTING", "ZIGZAG", "FLAT_REGULAR"],
        family="DOUBLE_THREE",
    )
    assert not result.valid


def test_t18_triangle_is_allowed_as_terminal_y_of_double_three():
    result = validate_double_correction(
        make_points([150, 130, 142, 125]),
        ["FLAT_REGULAR", "ZIGZAG", "TRIANGLE_CONTRACTING"],
        family="DOUBLE_THREE",
    )
    assert result.valid


def test_t21_impulse_cannot_be_x_connector():
    result = validate_double_correction(
        make_points([150, 130, 142, 120]),
        ["ZIGZAG", "IMPULSE", "ZIGZAG"],
        family="DOUBLE_ZIGZAG",
    )
    assert failed(result, "DOUBLE_X_CORRECTIVE")


def test_t23_ending_diagonal_allows_one_four_overlap():
    points = make_points([100, 120, 110, 125, 117, 128])
    result = validate_ending_diagonal(points, bars_from_prices([100, 120, 110, 125, 117, 128]))
    assert result.valid


def test_t24_diagonal_throw_over_is_not_a_standalone_rule_breach():
    result = validate_ending_diagonal(
        make_points([100, 120, 110, 125, 117, 129]),
        bars_from_prices([100, 120, 110, 125, 117, 129]),
    )
    assert all("THROW" not in check.rule_id for check in result.checks)


def test_t25_truncated_fifth_remains_candidate_without_subdivision():
    result = validate_impulse(make_points([100, 120, 110, 150, 130, 145]), bars_from_prices([100, 120, 110, 150, 130, 145]), truncated=True)
    assert result.valid
    assert "wave5_five_part_subdivision" in result.unknown_requirements


def test_t26_truncated_fifth_rejects_hidden_internal_new_high():
    points = make_points([100, 120, 110, 150, 130, 145])
    bars = bars_from_prices([100, 120, 110, 150, 130, 145])
    bars.loc[5, "high"] = 151
    result = validate_impulse(points, bars, truncated=True)
    assert failed(result, "TRUNCATED_5_INTERNAL_EXTREME")


def test_t29_target_reached_before_issue_is_retrospective_not_hit():
    target = {
        "target_id": "t1", "issued_at": "2020-01-03T00:00:00+00:00",
        "price_low": 104.0, "price_high": 106.0, "status": "ACTIVE",
    }
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-04"], utc=True),
        "open": [100, 102], "high": [105, 103], "low": [99, 101], "close": [104, 102],
    })
    node = lifecycle_node(target)
    events = evaluate_lifecycle([node], bars)
    assert node.targets[0]["status"] == "RETROSPECTIVE_LEVEL"
    assert any(event["event_type"] == "RETROSPECTIVE_LEVEL" for event in events)


def test_t30_gap_over_zone_is_not_reported_as_touch():
    target = {
        "target_id": "t1", "issued_at": "2020-01-02T00:00:00+00:00",
        "price_low": 105.0, "price_high": 110.0, "status": "ACTIVE",
    }
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [100, 112], "high": [102, 115], "low": [99, 111], "close": [101, 114],
    })
    node = lifecycle_node(target)
    events = evaluate_lifecycle([node], bars)
    assert node.targets[0]["status"] == "TARGET_PASSED_BY_GAP"
    assert all(event["event_type"] != "TARGET_ZONE_ENTERED" for event in events)


def test_t31_same_bar_target_and_invalidation_is_ambiguous():
    target = {
        "target_id": "t1", "issued_at": "2020-01-02T00:00:00+00:00",
        "price_low": 105.0, "price_high": 110.0, "status": "ACTIVE",
    }
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [100, 100], "high": [102, 108], "low": [99, 89], "close": [101, 101],
    })
    node = lifecycle_node(target, invalidation=90)
    events = evaluate_lifecycle([node], bars)
    assert node.targets[0]["status"] == "AMBIGUOUS_BAR"
    assert any(event["event_type"] == "AMBIGUOUS_BAR" for event in events)


def test_t28_target_hit_does_not_complete_a_forming_wave():
    target = {
        "target_id": "t1", "issued_at": "2020-01-02T00:00:00+00:00",
        "price_low": 105.0, "price_high": 110.0, "status": "ACTIVE",
    }
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "open": [100, 103], "high": [102, 108], "low": [99, 102], "close": [101, 107],
    })
    node = lifecycle_node(target)
    evaluate_lifecycle([node], bars)
    assert node.targets[0]["status"] == "TARGET_ZONE_ENTERED"
    assert node.endpoint_status == "FORMING"


def test_t32_engine_is_deterministic_on_identical_input():
    wave = np.sin(np.linspace(0, 18 * np.pi, 900)) * np.linspace(2, 25, 900) + np.linspace(100, 220, 900)
    bars = normalized_daily(wave.tolist(), start="2022-01-01")
    engine = ElliottWaveEngine({"pivot_atr_multipliers": [1.5], "max_active_scenarios": 16})
    first = engine.analyze(bars, ASSET_SPECS["SPX"])
    second = engine.analyze(bars, ASSET_SPECS["SPX"])
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["nodes"] == second["nodes"]
    assert first["scenarios"] == second["scenarios"]
    assert first["events"] == second["events"]


def test_t33_pivot_prefix_invariance():
    values = (100 + np.sin(np.linspace(0, 30, 300)) * 10 + np.linspace(0, 15, 300)).tolist()
    bars = normalized_daily(values)
    short = build_causal_pivot_streams(bars.iloc[:220], asset_id="TEST", source_timeframe="1D", multipliers=[1.5])
    long = build_causal_pivot_streams(bars, asset_id="TEST", source_timeframe="1D", multipliers=[1.5])
    cutoff = pd.Timestamp(bars.iloc[219]["timestamp"])
    short_confirmed = [(p.kind, p.pivot_time, p.confirmed_at) for p in short[0].pivots if p.status == "PIVOT_CONFIRMED"]
    long_confirmed = [(p.kind, p.pivot_time, p.confirmed_at) for p in long[0].pivots if p.status == "PIVOT_CONFIRMED" and pd.Timestamp(p.confirmed_at) <= cutoff]
    assert short_confirmed == long_confirmed


def test_t34_same_snapshot_id_is_immutable_on_retry():
    original_paths = (
        storage_module.STORAGE_DIR,
        storage_module.SNAPSHOT_DIR,
        storage_module.MANIFEST_PATH,
        storage_module.JOURNAL_PATH,
        storage_module.QUOTE_PATH,
        storage_module._atomic_parquet,
    )
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        storage_module.STORAGE_DIR = root
        storage_module.SNAPSHOT_DIR = root / "snapshots"
        storage_module.MANIFEST_PATH = root / "manifest.json"
        storage_module.JOURNAL_PATH = root / "events.jsonl"
        storage_module.QUOTE_PATH = root / "quotes.json"
        storage_module._atomic_parquet = lambda path, frame: None
        try:
            first = {"canonical_asset_id": "SPX", "snapshot_id": "same", "created_at": "first", "events": []}
            second = {"canonical_asset_id": "SPX", "snapshot_id": "same", "created_at": "second", "events": []}
            storage_module.write_snapshot(first, pd.DataFrame(), {})
            storage_module.write_snapshot(second, pd.DataFrame(), {})
            saved = storage_module.read_snapshot("SPX", "same")
            assert saved["created_at"] == "first"
            assert second["created_at"] == "first"
        finally:
            (
                storage_module.STORAGE_DIR,
                storage_module.SNAPSHOT_DIR,
                storage_module.MANIFEST_PATH,
                storage_module.JOURNAL_PATH,
                storage_module.QUOTE_PATH,
                storage_module._atomic_parquet,
            ) = original_paths


def test_t35_historical_revision_changes_data_version():
    first = normalized_daily([100, 101, 102])
    revised = first.copy()
    revised.loc[1, ["open", "high", "low", "close"]] = [101, 102, 100, 100]
    assert data_version(first) != data_version(revised)


def test_t36_current_vintage_history_is_not_called_live_pit():
    bars = normalized_daily((100 + np.sin(np.linspace(0, 15, 180)) * 5).tolist())
    snapshot = ElliottWaveEngine({"pivot_atr_multipliers": [1.5]}).analyze(bars, ASSET_SPECS["SPX"])
    assert snapshot["availability_mode"] == "CAUSAL_REPLAY_CURRENT_VINTAGE"
    assert "HISTORICAL_RECEIVED_AT_UNAVAILABLE" in snapshot["quality_flags"]


def test_t38_display_scale_is_not_an_engine_parameter():
    assert "price_scale" not in ElliottWaveEngine().parameters
    assert ElliottWaveEngine().parameters["measurement_mode"] == "arithmetic"


def test_t37_visible_range_is_not_an_engine_input():
    engine = ElliottWaveEngine()
    assert "date_range" not in engine.parameters
    assert "visible_start" not in engine.parameters


def test_t39_momentum_volume_and_macro_are_not_hard_rule_inputs():
    parameters = ElliottWaveEngine().parameters
    assert parameters["momentum_in_hard_rules"] is False
    assert parameters["macro_inputs_in_engine"] is False


def test_t40_same_hypothesis_from_multiple_k_is_deduplicated():
    first = lifecycle_node({"target_id": "a", "issued_at": "2020-01-01", "price_low": 1, "price_high": 2})
    second = copy.deepcopy(first)
    first.node_id, second.node_id = "k15", "k30"
    first.pivot_stream_k, second.pivot_stream_k = 1.5, 3.0
    first.start_point = second.start_point = {"pivot_time": "2020-01-01"}
    first.end_point = second.end_point = {"pivot_time": "2020-02-01"}
    scenarios = ElliottWaveEngine()._rank_scenarios({"k15": first, "k30": second}, {"k15", "k30"}, 100)
    assert len(scenarios) == 1


def test_t41_search_limit_is_explicit():
    wave = (100 + np.sin(np.linspace(0, 30, 400)) * 10).tolist()
    snapshot = ElliottWaveEngine({"pivot_atr_multipliers": [1.5], "max_active_scenarios": 0}).analyze(
        normalized_daily(wave), ASSET_SPECS["SPX"]
    )
    assert snapshot["search_statistics"]["search_truncated"] is True
    assert "SEARCH_LIMITED" in snapshot["quality_flags"]
    assert any(event["event_type"] == "SEARCH_LIMIT_REACHED" for event in snapshot["events"])


def test_t42_no_valid_candidate_is_explicitly_unresolved():
    snapshot = ElliottWaveEngine({"pivot_atr_multipliers": [6.0]}).analyze(
        normalized_daily(list(np.linspace(100, 110, 80))), ASSET_SPECS["SPX"]
    )
    assert snapshot["main_scenario_id"] is None
    assert snapshot["unresolved_reasons"]
    assert any(event["event_type"] == "UNRESOLVED" for event in snapshot["events"])


def test_t44_mirrored_ohlc_keeps_impulse_validity():
    points = make_points([100, 120, 110, 150, 130, 155])
    bars = bars_from_prices([100, 120, 110, 150, 130, 155])
    original = validate_impulse(points, bars)
    mirrored_points = copy.deepcopy(points)
    for point in mirrored_points:
        point.price = -point.price
        point.kind = "HIGH" if point.kind == "LOW" else "LOW"
    mirrored = bars.copy()
    mirrored_high = -bars["low"]
    mirrored_low = -bars["high"]
    mirrored["open"] = -bars["open"]
    mirrored["close"] = -bars["close"]
    mirrored["high"] = mirrored_high
    mirrored["low"] = mirrored_low
    reflected = validate_impulse(mirrored_points, mirrored)
    assert original.valid == reflected.valid
    assert [(c.rule_id, c.result) for c in original.checks] == [(c.rule_id, c.result) for c in reflected.checks]


def test_t45_anchored_review_does_not_override_hard_rule():
    points = make_points([100, 120, 100, 150, 130, 155])
    bars = normalized_daily([100, 120, 100, 150, 130, 155])
    review = ElliottWaveEngine().review_anchored(bars, ASSET_SPECS["SPX"], "IMPULSE", points)
    assert review["mode"] == "ANCHORED_REVIEW"
    assert review["status"] == "MODEL_INVALID"
    assert review["auto_history_modified"] is False
    assert any(check["rule_id"] == "IMPULSE_2_NO_FULL_RETRACE" and check["result"] == "FAIL" for check in review["rule_checks"])


def test_t43_json_and_xlsx_use_the_same_snapshot_and_scenario():
    bars = normalized_daily([100, 101, 102])
    snapshot = {
        "canonical_asset_id": "SPX",
        "snapshot_id": "snap-43",
        "main_scenario_id": "scenario-43",
        "scenarios": [{"scenario_id": "scenario-43", "root_node_id": "node-43"}],
        "nodes": [{
            "node_id": "node-43", "pattern_type": "IMPULSE", "relative_degree": "D0",
            "labels": [{"label": "3"}], "endpoint_status": "FORMING", "context_status": "UNRESOLVED",
            "verified_depth": 0, "targets": [], "invalidation": None, "rule_checks": [], "ratios": [], "channels": [],
        }],
        "parameters": {}, "pivot_streams": [], "events": [],
    }
    manifest = {"manifest_id": "manifest-43", "published_at": "2020-01-01"}
    settings = {"SPX": {"scenario_id": "scenario-43", "focus_node_id": "node-43", "chart_timeframe": "1W", "visible_degree": "Auto"}}
    old_base, old_chart = export_module.read_base_bars, export_module.read_chart_bars
    export_module.read_base_bars = lambda asset_id, snapshot_id: bars.copy()
    export_module.read_chart_bars = lambda asset_id, snapshot_id, timeframe: bars.copy()
    try:
        json_payload = json.loads(export_module.build_json_export(manifest, {"SPX": snapshot}, settings).decode("utf-8"))
        xlsx_payload = export_module.build_xlsx_export(manifest, {"SPX": snapshot}, settings)
        metadata = pd.read_excel(io.BytesIO(xlsx_payload), sheet_name="Metadata")
    finally:
        export_module.read_base_bars, export_module.read_chart_bars = old_base, old_chart
    assert json_payload["assets"]["SPX"]["snapshot"]["snapshot_id"] == "snap-43"
    assert json_payload["assets"]["SPX"]["view"]["scenario_id"] == "scenario-43"
    assert metadata.iloc[0]["snapshot_id"] == "snap-43"
    assert metadata.iloc[0]["scenario_id"] == "scenario-43"


def test_ui01_and_ui20_navigation_is_exact_and_render_path_is_snapshot_only():
    app_source = Path("app.py").read_text(encoding="utf-8")
    tab_source = Path("elliott_waves_tab.py").read_text(encoding="utf-8")
    assert '"Eliot waves"' in app_source
    assert "render_elliott_waves_tab()" in app_source
    assert "ElliottWaveEngine" not in tab_source
    assert "refresh_all_assets" not in tab_source


def test_summary_does_not_present_fib_fit_as_probability():
    snapshot = {
        "main_scenario_id": "s1",
        "scenarios": [{"scenario_id": "s1", "root_node_id": "n1"}],
        "nodes": [{
            "node_id": "n1",
            "pattern_type": "IMPULSE",
            "relative_degree": "D0",
            "labels": [{"label": "5"}],
            "endpoint_status": "FORMING",
            "context_status": "UNRESOLVED",
            "targets": [],
            "invalidation": {"level": 100, "basis": "High/Low"},
        }],
    }
    text = build_summary(snapshot, "s1")
    assert "вероят" not in text.lower()
    assert "формируется" in text.lower()


if __name__ == "__main__":
    failures = []
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(1)
    print(f"PASS all {len([name for name in globals() if name.startswith('test_')])} Elliott tests")

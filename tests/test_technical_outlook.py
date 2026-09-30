from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from market_data import _normalized_frame
from technical_outlook.analytics import (
    build_support_resistance,
    calculate_indicators,
    classify_structure,
    detect_pivots,
    historical_analogs,
    scenario_probabilities,
)
from technical_outlook.elliott import (
    generate_candidates,
    select_primary_alternative,
    validate_compound,
    validate_diagonal,
    validate_flat,
    validate_impulse,
    validate_triangle,
    validate_zigzag,
)
from technical_outlook.config import CORE_ASSETS
from technical_outlook.engine import TechnicalOutlookEngine, _chart_frame
from technical_outlook.llm import LLM_INPUT_MAX_CHARS, LLM_TEXT_FIELDS, serialize_llm_input, structured_llm_input, validate_llm_output
from technical_outlook.legacy_v0 import build_confirmation_matrix_v0, build_scenarios_v0, build_support_resistance_v0
from technical_outlook.scenario_v1 import apply_relevance, build_scenarios_v1, support_resistance_component_v1
from technical_outlook.service import apply_llm_schedule
from technical_outlook.sr_v1 import build_cross_timeframe_zones, build_support_resistance_v1
from technical_outlook import storage
from technical_outlook_tab import _analysis_content, _visible_support_resistance, _zone_distance_from_price, build_technical_chart


def pivot(price: float, index: int, kind: str, status: str = "CONFIRMED", degree: str = "INTERMEDIATE") -> dict:
    timestamp = pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=index)
    return {
        "pivot_id": f"{degree}:{index}", "price": price, "bar_index": index, "kind": kind,
        "pivot_time": timestamp.isoformat(), "confirmation_time": (timestamp + pd.Timedelta(weeks=1)).isoformat() if status == "CONFIRMED" else None,
        "confirmed_at": (timestamp + pd.Timedelta(weeks=1)).isoformat() if status == "CONFIRMED" else None,
        "status": status, "degree": degree, "timeframe": "WEEKLY", "atr_reference": 1.0,
        "threshold_used": 2.0, "reversal_magnitude": 3.0 if status == "CONFIRMED" else None,
    }


def points(prices: list[float], first_kind: str = "LOW", status: str = "CONFIRMED") -> list[dict]:
    kinds = [first_kind if index % 2 == 0 else ("HIGH" if first_kind == "LOW" else "LOW") for index in range(len(prices))]
    return [pivot(price, index, kinds[index], status=status) for index, price in enumerate(prices)]


def weekly_frame(prices: list[float], atr: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=len(prices), freq="W-FRI")
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": np.asarray(prices) + 0.5,
        "low": np.asarray(prices) - 0.5,
        "close": prices,
        "volume": np.arange(len(prices)) + 100,
        "atr14": atr,
    })


def test_indicators_calculate_ma_rsi_macd_roc_and_extension() -> None:
    prices = np.linspace(100, 300, 260)
    frame = calculate_indicators(weekly_frame(prices.tolist()))
    row = frame.iloc[-1]
    assert row["sma50"] > row["sma100"] > row["sma200"]
    assert row["sma50_slope"] > 0
    assert row["rsi14"] > 50
    assert row["macd"] > row["macd_signal"]
    assert row["roc12"] > 0
    assert row["extension200"] > 0


def test_causal_pivot_confirmation_time_is_after_extreme() -> None:
    frame = weekly_frame([100, 104, 110, 109, 108, 106, 101, 102, 103], atr=2.0)
    detected = detect_pivots(frame, "WEEKLY", "INTERMEDIATE", atr_multiplier=1.0, minimum_reversal_pct=1.0)
    confirmed_highs = [item for item in detected if item["kind"] == "HIGH" and item["status"] == "CONFIRMED"]
    assert confirmed_highs
    high = confirmed_highs[0]
    assert pd.Timestamp(high["confirmation_time"]) > pd.Timestamp(high["pivot_time"])
    assert high["reversal_magnitude"] >= high["threshold_used"]
    assert detected[-1]["status"] == "POTENTIAL"
    assert detected[-1]["confirmation_time"] is None


def test_market_structure_ignores_potential_pivot() -> None:
    pivots = [
        pivot(100, 0, "HIGH"), pivot(80, 1, "LOW"), pivot(110, 2, "HIGH"), pivot(90, 3, "LOW"),
        pivot(70, 4, "LOW", status="POTENTIAL"),
    ]
    result = classify_structure(weekly_frame([100] * 30), pivots)
    assert result["state"] == "BULL"
    assert result["sequence"] == "HH-HL"


@pytest.mark.parametrize(
    ("prices", "valid"),
    [
        ([0, 10, 5, 30, 20, 40], True),
        ([0, 10, -1, 30, 20, 40], False),
        ([0, 10, 5, 12, 11, 30], False),
        ([0, 10, 5, 30, 8, 40], False),
    ],
)
def test_impulse_hard_rules(prices: list[float], valid: bool) -> None:
    actual, checks = validate_impulse(points(prices))
    assert actual is valid
    assert all(check["passed"] for check in checks) is valid


def test_diagonal_overlap_is_handled_separately() -> None:
    sequence = points([0, 20, 10, 28, 18, 30])
    impulse_valid, _ = validate_impulse(sequence)
    diagonal_valid, checks = validate_diagonal(sequence, ending=True)
    assert impulse_valid is False
    assert diagonal_valid is True
    assert any(check["rule"] == "diagonal_wave4_overlap" and check["passed"] for check in checks)


def test_corrective_pattern_validators() -> None:
    assert validate_zigzag(points([0, 10, 5, 18]))[0]
    assert validate_flat(points([0, 10, 1, 9]), expanded=False)[0]
    assert validate_flat(points([0, 10, -2, 12]), expanded=True)[0]
    assert validate_triangle(points([0, 10, 2, 8, 4, 6]))[0]
    assert validate_compound(points([0, 10, 5, 8, 3, 9, 4]), triple=False)[0]
    assert validate_compound(points([0, 10, 5, 8, 3, 9, 4, 7, 5, 6]), triple=True)[0]


def test_complexity_penalty_prefers_simple_candidate_when_fit_is_equal() -> None:
    momentum = {"score": 30.0}
    volume = {"status": "NOT_AVAILABLE", "score": None}
    universe = generate_candidates(points([0, 10, 5, 18, 9, 24, 12, 20, 14, 18]), "INTERMEDIATE", [], momentum, [], volume, parent=None)
    penalties = {item["pattern"]: item["complexity_penalty"] for item in universe}
    if "WXY" in penalties:
        assert penalties["WXY"] == 5.0
    if "WXYXZ" in penalties:
        assert penalties["WXYXZ"] == 10.0


def test_developing_wave_and_primary_alternative_are_preserved() -> None:
    universe = generate_candidates(points([0, 10, 5, 30, 20]), "INTERMEDIATE", [], {"score": 60.0}, [], {"status": "NOT_AVAILABLE"}, parent=None)
    impulse = next(item for item in universe if item["pattern"] == "IMPULSE" and item["open_wave"])
    assert impulse["wave_state"] == "DEVELOPING"
    assert impulse["completion_state"] == "DEVELOPING"
    assert impulse["waves"][-1]["wave_status"] == "DEVELOPING"
    primary, alternative, confidence = select_primary_alternative(universe)
    assert primary["candidate_id"]
    assert alternative["candidate_id"]
    assert confidence in {"HIGH", "MEDIUM", "LOW"}


def test_scenario_probabilities_always_sum_to_100() -> None:
    result = scenario_probabilities({
        "trend": 70, "momentum": 40, "elliott": 50, "volume": None,
        "extension_risk": -20, "divergence": 0, "support_resistance": 35, "historical_analog": 15,
    }, "MEDIUM")
    assert set(result) == {"BULLISH", "NEUTRAL", "BEARISH"}
    assert sum(result.values()) == 100


def test_historical_analogs_exclude_outcomes_that_end_after_as_of() -> None:
    frame = calculate_indicators(weekly_frame((100 + np.arange(420) * 0.2 + np.sin(np.arange(420) / 8) * 3).tolist()))
    result = historical_analogs(frame)
    dates = {pd.Timestamp(item["date"]) for item in result["periods"]}
    cutoff = pd.Timestamp(frame.iloc[-53]["timestamp"]).normalize()
    assert all(date <= cutoff for date in dates)
    assert result["sampling"].startswith("Nearest causal weekly states")


def test_snapshot_serialization_preserves_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(storage, "CHART_DIR", tmp_path / "charts")
    first = {"ticker": "TEST", "snapshot_id": "one", "value": np.float64(1.5)}
    second = {"ticker": "TEST", "snapshot_id": "two", "value": np.float64(2.5)}
    storage.write_snapshot(first, {"1W": pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "close": [1.0]})})
    storage.write_snapshot(second, {})
    assert (tmp_path / "snapshots" / "TEST" / "one.json").exists()
    assert (tmp_path / "snapshots" / "TEST" / "two.json").exists()
    assert storage.read_latest_snapshot("TEST")["snapshot_id"] == "two"


def valid_llm_payload() -> dict:
    candidate = {
        "label": "UNRESOLVED", "pattern": "UNRESOLVED", "direction": "NEUTRAL",
        "current_wave": "N/A", "wave_state": "UNRESOLVED", "completion_state": "UNRESOLVED",
        "targets": "N/A", "invalidation": "N/A", "rationale": "Insufficient evidence", "waves": [],
    }
    return {
        **{field: "ok" for field in LLM_TEXT_FIELDS},
        "risk_factors": [],
        "key_confirmation_points": [],
        "elliott_structure": {
            "primary": dict(candidate), "alternative": dict(candidate),
            "confidence": "LOW", "current_wave_state": "UNRESOLVED",
        },
    }


def test_chart_payload_is_limited_to_latest_500_bars() -> None:
    frame = weekly_frame(np.linspace(100, 300, 650).tolist())
    result = _chart_frame(frame)
    assert len(result) == 500
    assert result.iloc[0]["timestamp"] == frame.iloc[-500]["timestamp"]


def test_engine_builds_weekly_and_daily_frames_without_quant_elliott() -> None:
    spec = CORE_ASSETS["SPY"]
    dates = pd.date_range("2019-01-02", periods=1800, freq="B")
    close = 100 + np.arange(len(dates)) * 0.08 + np.sin(np.arange(len(dates)) / 12) * 4
    daily = _normalized_frame(
        dates=dates,
        opens=pd.Series(close - 0.2),
        highs=pd.Series(close + 1.0),
        lows=pd.Series(close - 1.0),
        closes=pd.Series(close),
        volumes=pd.Series(1_000_000 + np.arange(len(dates)) * 100),
        timeframe="1D",
        spec=spec,
        now=pd.Timestamp("2030-01-01"),
    )
    snapshot, charts = TechnicalOutlookEngine().analyze(daily, spec, created_at=datetime(2030, 1, 2, tzinfo=timezone.utc))
    assert set(charts) == {"1W", "1D"}
    assert len(charts["1D"]) == 500
    assert "weekly_structure" in snapshot and "daily_structure" in snapshot
    assert "monthly_structure" not in snapshot
    assert snapshot["elliott_source"] == "LLM"
    assert snapshot["scenario_components"]["elliott"] is None
    assert snapshot["weekly_volume_profile"]["lookback_bars"] == len(charts["1W"])
    assert snapshot["weekly_volume_profile"]["lookback_bars"] <= 500
    assert all(
        zone["timeframes"] == ["WEEKLY"]
        for zone in snapshot["weekly_support_resistance"]
    )


def test_structural_weekly_levels_use_weekly_smas_and_all_window_pivots() -> None:
    prices = np.linspace(100, 600, 500)
    frame = calculate_indicators(weekly_frame(prices.tolist(), atr=8.0))
    pivots = [pivot(float(price), index, "LOW" if index % 2 == 0 else "HIGH") for index, price in enumerate(np.linspace(120, 580, 24))]
    zones = build_support_resistance(
        frame,
        pivots,
        {"status": "NOT_AVAILABLE"},
        timeframe="WEEKLY",
        max_pivots=None,
        max_zones=20,
    )
    assert len(zones) <= 20
    assert all(zone["timeframes"] == ["WEEKLY"] for zone in zones)
    sources = {source for zone in zones for source in zone["sources"]}
    assert {"sma50", "sma100", "sma200"}.issubset(sources)


def test_legacy_v0_is_a_physically_separate_frozen_calculation() -> None:
    frame = calculate_indicators(weekly_frame(np.linspace(100, 160, 260).tolist(), atr=2.0))
    pivots = [pivot(value, index + 20, "LOW" if index % 2 == 0 else "HIGH") for index, value in enumerate([120.0, 128.0, 124.0, 140.0])]
    profile = {"status": "NOT_AVAILABLE"}
    original = build_support_resistance(frame, pivots, profile, timeframe="DAILY", max_pivots=10, max_zones=10)
    frozen = build_support_resistance_v0(frame, pivots, profile, timeframe="DAILY", max_pivots=10, max_zones=10)
    assert frozen == original
    probabilities = {"BULLISH": 45, "NEUTRAL": 25, "BEARISH": 30}
    scenarios = build_scenarios_v0(160.0, frozen, probabilities, {"classification": "POSITIVE"}, timeframe_label="Daily")
    matrix = build_confirmation_matrix_v0(scenarios, frozen, 160.0, timeframe_label="Daily")
    assert [item["probability"] for item in scenarios] == [45, 25, 30]
    assert matrix[0]["interpretation"] == "Bull scenario confirmed"


@pytest.mark.parametrize(
    ("ticker", "scale", "golden_hash"),
    [
        ("SPY", 1.0, "cd1fff5cb7f4be9e7a2c2a42d91bcf5094ef46a1805e91fd4895dc2d03019acc"),
        ("QQQ", 1.7, "9b5c77728dcfab2b50bce3f1f3cf7751e0c0061dc4466469e81dda71dd6808f3"),
        ("GLD", 0.65, "4483744d79a23e9f8ceb928d08c917425c0aa96bba74b7b39b89af9a33e4098a"),
        ("BTC-USD", 150.0, "9d8cc6389cda15e6f6659ab2eb0028b786f734ea4037c0275c74d0601d28afc8"),
    ],
)
def test_legacy_v0_golden_snapshots(ticker: str, scale: float, golden_hash: str) -> None:
    dates = pd.date_range("2020-01-03", periods=260, freq="W-FRI")
    close = (100 + np.arange(260) * 0.2 + np.sin(np.arange(260) / 8) * 3) * scale
    frame = calculate_indicators(pd.DataFrame({
        "timestamp": dates,
        "open": close - 0.2 * scale,
        "high": close + scale,
        "low": close - scale,
        "close": close,
        "volume": 1_000_000 + np.arange(260),
        "atr14": 2 * scale,
    }))
    pivots = []
    for index, (price, kind) in enumerate(((120, "LOW"), (140, "HIGH"), (128, "LOW"), (155, "HIGH"))):
        timestamp = pd.Timestamp("2022-01-07") + pd.Timedelta(weeks=index * 10)
        pivots.append({
            "pivot_id": f"{ticker}:{index}", "price": price * scale, "bar_index": 100 + index * 10,
            "kind": kind, "pivot_time": timestamp.isoformat(),
            "confirmation_time": (timestamp + pd.Timedelta(weeks=1)).isoformat(), "status": "CONFIRMED",
        })
    zones = build_support_resistance_v0(frame, pivots, {"status": "NOT_AVAILABLE"}, timeframe="DAILY", max_pivots=10, max_zones=10)
    probabilities = scenario_probabilities({
        "trend": 70, "momentum": 40, "elliott": None, "volume": 0,
        "extension_risk": -20, "divergence": 0, "support_resistance": 35, "historical_analog": 15,
    }, "MEDIUM")
    scenarios = build_scenarios_v0(float(close[-1]), zones, probabilities, {"classification": "POSITIVE"}, timeframe_label="Daily")
    confirmation = build_confirmation_matrix_v0(scenarios, zones, float(close[-1]), timeframe_label="Daily")
    encoded = json.dumps(
        {"zones": zones, "probabilities": probabilities, "scenarios": scenarios, "confirmation": confirmation},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == golden_hash


def test_sr_v1_repeated_daily_swings_count_once_in_confluence() -> None:
    frame = weekly_frame([100.0] * 80, atr=1.0)
    daily_pivots = [
        {**pivot(99.8 + index * 0.1, index + 20, "LOW", degree="INTERMEDIATE"), "timeframe": "DAILY"}
        for index in range(3)
    ]
    zones = build_support_resistance_v1(
        frame,
        daily_pivots,
        {"status": "NOT_AVAILABLE"},
        timeframe="DAILY",
        max_pivots=None,
        max_zones=20,
    )
    swing_zone = next(zone for zone in zones if "SWING_STRUCTURE" in zone["source_families"])
    assert swing_zone["family_scores"]["SWING_STRUCTURE"] == 2.0
    assert swing_zone["family_sources"]["SWING_STRUCTURE"] == ["daily_swing"]


def test_sr_v1_moving_averages_are_one_family_with_capped_breadth() -> None:
    frame = calculate_indicators(weekly_frame([100.0] * 260, atr=1.0))
    zones = build_support_resistance_v1(
        frame,
        [],
        {"status": "NOT_AVAILABLE"},
        timeframe="DAILY",
        max_pivots=None,
        max_zones=20,
    )
    zone = next(zone for zone in zones if set(zone["family_sources"].get("MOVING_AVERAGE", [])) == {"sma50", "sma100", "sma200"})
    assert zone["family_scores"]["MOVING_AVERAGE"] == 3.0
    assert zone["independent_family_count"] == 2  # moving average + round number


def test_sr_v1_clustering_is_input_order_invariant() -> None:
    frame = weekly_frame([100.0] * 100, atr=1.0)
    pivots = [
        {**pivot(value, index + 30, "LOW", degree="INTERMEDIATE"), "timeframe": "DAILY"}
        for index, value in enumerate([98.9, 99.2, 99.5, 100.1])
    ]
    first = build_support_resistance_v1(frame, pivots, {"status": "NOT_AVAILABLE"}, timeframe="DAILY", max_pivots=None, max_zones=20)
    second = build_support_resistance_v1(frame, list(reversed(pivots)), {"status": "NOT_AVAILABLE"}, timeframe="DAILY", max_pivots=None, max_zones=20)
    assert [(zone["zone_id"], zone["low"], zone["high"]) for zone in first] == [(zone["zone_id"], zone["low"], zone["high"]) for zone in second]


def test_sr_v1_cross_timeframe_requires_same_role_and_does_not_mutate_parents() -> None:
    frame = calculate_indicators(weekly_frame([100.0] * 100, atr=1.0))
    base = {
        "source_members": [], "source_families": [], "family_scores": {}, "sources": [],
        "confluence_class": "MEDIUM", "confluence": "MEDIUM", "strength_class": "WEAK",
        "strength_score": 0.0, "valid_from": frame.iloc[0]["timestamp"].isoformat(),
    }
    daily = [{**base, "zone_id": "d", "role": "SUPPORT", "low": 98.0, "high": 102.0, "center": 100.0, "width": 4.0, "timeframe": "DAILY"}]
    weekly = [{**base, "zone_id": "w", "role": "SUPPORT", "low": 99.0, "high": 103.0, "center": 101.0, "width": 4.0, "timeframe": "WEEKLY"}]
    original_daily = dict(daily[0])
    derived = build_cross_timeframe_zones(daily, weekly, frame)
    assert len(derived) == 1
    assert derived[0]["low"] == 99.0 and derived[0]["high"] == 102.0
    assert derived[0]["cross_timeframe_bonus"] == 1.0
    assert daily[0] == original_daily
    weekly[0]["role"] = "RESISTANCE"
    assert build_cross_timeframe_zones(daily, weekly, frame) == []


def test_scenario_v1_can_rank_near_strong_zone_above_distant_confluence() -> None:
    frame = calculate_indicators(weekly_frame(np.linspace(100, 120, 100).tolist(), atr=2.0))
    common = {
        "timeframe": "DAILY", "role": "RESISTANCE", "cross_timeframe": False,
        "source_families": [], "parent_daily_zone_ids": [], "parent_weekly_zone_ids": [],
    }
    zones = [
        {**common, "zone_id": "near", "low": 122.0, "high": 123.0, "center": 122.5, "confluence_class": "MEDIUM", "strength_class": "VERY_STRONG", "strength_score": 8.0, "final_confluence_score": 3.0},
        {**common, "zone_id": "far", "low": 170.0, "high": 172.0, "center": 171.0, "confluence_class": "VERY_HIGH", "strength_class": "WEAK", "strength_score": 0.0, "final_confluence_score": 9.0},
    ]
    enriched = apply_relevance(zones, 120.0, frame, is_crypto=False)
    scenarios = build_scenarios_v1(120.0, enriched, {"BULLISH": 40, "NEUTRAL": 30, "BEARISH": 30}, {"classification": "POSITIVE"}, daily_atr=2.0, weekly_atr=5.0)
    bullish = next(item for item in scenarios if item["scenario"] == "BULLISH")
    assert bullish["primary_target"]["zone_id"] == "near"


def test_sr_v1_component_uses_canonical_cross_timeframe_zone_once() -> None:
    frame = calculate_indicators(weekly_frame(np.linspace(100, 120, 100).tolist(), atr=2.0))
    zone = {
        "zone_id": "d", "role": "SUPPORT", "low": 115.0, "high": 116.0, "center": 115.5,
        "timeframe": "DAILY", "cross_timeframe": False, "confluence_class": "HIGH", "strength_class": "STRONG",
        "parent_daily_zone_ids": [], "parent_weekly_zone_ids": [],
    }
    parent = apply_relevance([zone], 120.0, frame, is_crypto=False)[0]
    cross = {
        **parent, "zone_id": "x", "timeframe": "CROSS_TIMEFRAME", "cross_timeframe": True,
        "parent_daily_zone_ids": ["d"], "parent_weekly_zone_ids": ["w"],
    }
    assert support_resistance_component_v1([parent, cross]) == support_resistance_component_v1([cross])


def test_chart_has_no_pan_or_zoom_interaction() -> None:
    frame = calculate_indicators(weekly_frame(np.linspace(100, 160, 260).tolist()))
    chart = build_technical_chart(
        frame,
        {"support_resistance": [], "volume_profile": {}},
        primary=None,
        alternative=None,
        show_levels=False,
        show_profile=False,
    )
    assert "params" not in chart.to_dict()


def test_support_resistance_overlay_keeps_only_bright_high_confluence_zones() -> None:
    zones = _visible_support_resistance({"support_resistance": [
        {"role": "SUPPORT", "confluence": "LOW", "low": 90, "high": 91},
        {"role": "SUPPORT", "confluence": "HIGH", "low": 95, "high": 96},
        {"role": "RESISTANCE", "confluence": "VERY_HIGH", "low": 105, "high": 106},
    ]})
    assert list(zones["confluence"]) == ["HIGH", "VERY_HIGH"]
    assert list(zones["color"]) == ["#00ff88", "#ff4d5a"]
    assert zones.loc[zones["confluence"] == "VERY_HIGH", "zone_opacity"].iloc[0] > zones.loc[zones["confluence"] == "HIGH", "zone_opacity"].iloc[0]


def test_zone_distance_from_price_uses_zone_average() -> None:
    assert _zone_distance_from_price(110.0, {"low": 99.0, "high": 101.0}) == "+10.00%"
    assert _zone_distance_from_price(90.0, {"low": 99.0, "high": 101.0}) == "-10.00%"
    assert _zone_distance_from_price(100.0, {"low": 0.0, "high": 0.0}) == "N/A"


def test_technical_outlook_v0_uses_only_pre_weekly_sr_outputs() -> None:
    snapshot = {
        "support_resistance": [{"id": "v0-low", "confluence": "LOW"}],
        "daily_support_resistance": [{"id": "daily-new", "confluence": "VERY_HIGH"}],
        "weekly_support_resistance": [{"id": "weekly-new", "confluence": "VERY_HIGH"}],
        "scenarios": [{"scenario": "v0"}],
        "daily_scenarios": [{"scenario": "daily-new"}],
        "weekly_scenarios": [{"scenario": "weekly-new"}],
        "confirmation_matrix": [{"event": "v0"}],
        "daily_confirmation_matrix": [{"event": "daily-new"}],
        "weekly_confirmation_matrix": [{"event": "weekly-new"}],
    }
    content = _analysis_content(snapshot, legacy=True, use_weekly=True)
    assert content["zones"] == snapshot["support_resistance"]
    assert content["scenarios"] == snapshot["scenarios"]
    assert content["confirmation"] == snapshot["confirmation_matrix"]


def test_engine_persists_separate_legacy_and_v1_stacks() -> None:
    spec = CORE_ASSETS["SPY"]
    dates = pd.date_range("2017-01-02", periods=2400, freq="B")
    close = 100 + np.arange(len(dates)) * 0.05 + np.sin(np.arange(len(dates)) / 10) * 3
    daily = _normalized_frame(
        dates=dates,
        opens=pd.Series(close - 0.2), highs=pd.Series(close + 1.0), lows=pd.Series(close - 1.0),
        closes=pd.Series(close), volumes=pd.Series(1_000_000 + np.arange(len(dates))),
        timeframe="1D", spec=spec, now=pd.Timestamp("2030-01-01"),
    )
    snapshot, _ = TechnicalOutlookEngine().analyze(daily, spec, created_at=datetime(2030, 1, 2, tzinfo=timezone.utc))
    assert snapshot["sr_engine_version"] == "SR_ENGINE_V1"
    assert snapshot["scenario_engine_version"] == "SCENARIO_ENGINE_V1"
    assert snapshot["legacy_sr_engine_version"] == "SR_ENGINE_V0"
    assert snapshot["legacy_scenario_engine_version"] == "SCENARIO_ENGINE_V0"
    assert snapshot["support_resistance"] == snapshot["legacy_daily_support_resistance"]
    assert snapshot["scenarios"] == snapshot["legacy_scenarios"]
    assert snapshot["daily_support_resistance"] != snapshot["legacy_daily_support_resistance"]


def test_llm_input_uses_weekly_and_daily_frames_only() -> None:
    value = structured_llm_input({"weekly_structure": {"state": "BULL"}, "daily_structure": {"state": "BEAR"}})
    assert "weekly" in value
    assert "daily" in value
    assert "monthly" not in value


def test_llm_request_is_compact_and_bounded() -> None:
    many_pivots = [pivot(float(100 + index), index, "LOW" if index % 2 == 0 else "HIGH") for index in range(200)]
    snapshot = {
        "ticker": "QQQ",
        "weekly_pivots": many_pivots,
        "daily_pivots": many_pivots,
        "minor_pivots": many_pivots,
        "divergences": [{"type": "BULLISH_RSI", "start_pivot": item, "end_pivot": item} for item in many_pivots],
        "weekly_support_resistance": [{"low": index, "high": index + 1, "center": index + 0.5, "role": "SUPPORT", "sources": ["major_swing"], "timeframes": ["WEEKLY"], "confluence_score": 8, "confluence": "VERY_HIGH"} for index in range(200)],
        "daily_support_resistance": [],
    }
    encoded = serialize_llm_input(snapshot)
    assert len(encoded) < LLM_INPUT_MAX_CHARS
    decoded = __import__("json").loads(encoded)
    assert len(decoded["weekly"]["pivots"]) == 18
    assert len(decoded["daily"]["pivots"]) == 24
    assert len(decoded["divergences"]) == 12


def test_llm_elliott_points_must_match_supplied_pivots() -> None:
    payload = valid_llm_payload()
    supplied = pivot(123.0, 2, "HIGH")
    payload["elliott_structure"]["primary"]["waves"] = [{
        "pivot_time": supplied["pivot_time"], "price": supplied["price"],
        "wave_label": "1", "wave_status": "CONFIRMED",
    }]
    snapshot = {"weekly_pivots": [supplied], "daily_pivots": [], "minor_pivots": []}
    validate_llm_output(payload, snapshot)
    payload["elliott_structure"]["primary"]["waves"][0]["price"] = 999.0
    with pytest.raises(ValueError):
        validate_llm_output(payload, snapshot)


def test_llm_schema_rejects_numeric_replacements() -> None:
    payload = valid_llm_payload()
    payload["summary"] = 123
    with pytest.raises(ValueError):
        validate_llm_output(payload)


def test_llm_schedule_off_and_non_friday_never_call() -> None:
    calls = []
    caller = lambda snapshot: (calls.append(snapshot) or valid_llm_payload(), "test-model")
    base = {"llm_interpretation": None, "llm_updated_at": None}
    friday = datetime(2026, 10, 2, 23, tzinfo=timezone.utc)
    monday = datetime(2026, 10, 5, 23, tzinfo=timezone.utc)
    apply_llm_schedule(dict(base), None, {"use_llm_interpretation": False}, friday, allow_scheduled_llm=True, llm_caller=caller)
    apply_llm_schedule(dict(base), None, {"use_llm_interpretation": True}, monday, allow_scheduled_llm=True, llm_caller=caller)
    assert calls == []


def test_friday_llm_failure_preserves_quant_and_previous_llm() -> None:
    previous = {"llm_interpretation": valid_llm_payload(), "llm_model": "old", "llm_updated_at": "2026-09-25T23:00:00+00:00"}
    snapshot = {"ticker": "SPY", "llm_interpretation": None, "llm_updated_at": None}
    result = apply_llm_schedule(
        snapshot,
        previous,
        {"use_llm_interpretation": True},
        datetime(2026, 10, 2, 23, tzinfo=timezone.utc),
        allow_scheduled_llm=True,
        llm_caller=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert result["llm_status"] == "UPDATE_FAILED_USING_PREVIOUS"
    assert result["llm_interpretation"] == previous["llm_interpretation"]

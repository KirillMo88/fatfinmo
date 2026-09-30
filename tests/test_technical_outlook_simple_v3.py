from __future__ import annotations

import importlib
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from elliott_waves.data import ElliottDataError, _normalized_frame
from technical_outlook.analytics import calculate_indicators
from technical_outlook.engine import TechnicalOutlookEngine
from technical_outlook_simple_v3.config import CONFIG, CORE_ASSETS, swing_config
from technical_outlook_simple_v3.engine import TechnicalOutlookSimpleV3Engine
from technical_outlook_simple_v3.fibonacci import active_fibonacci_framework
from technical_outlook_simple_v3.scenario import build_weekly_scenario_matrix, scenario_probabilities
from technical_outlook_simple_v3.service import _validate_completed_bars
from technical_outlook_simple_v3.strength import classify_strength
from technical_outlook_simple_v3.support_resistance import (
    _family_scores,
    assign_role,
    confluence_class,
    deterministic_clusters,
    visible_zones,
)
from technical_outlook_simple_v3.swings import detect_causal_swings
from technical_outlook_simple_v3.volume_profile import build_volume_profile
from technical_outlook_simple_v3_tab import build_simple_v3_chart, filter_chart_zones, visible_zone_frame, zone_source_color


def _frame(periods: int = 600, *, freq: str = "B", amplitude: float = 12.0) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=periods, freq=freq)
    index = np.arange(periods)
    close = 100 + index * 0.08 + np.sin(index / 25.0) * amplitude
    return pd.DataFrame({
        "timestamp": dates,
        "open": close - 0.5,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "volume": 1_000_000 + (np.sin(index / 11.0) + 1.5) * 250_000,
        "is_closed": True,
    })


def _pivot(price: float, index: int, kind: str, status: str = "CONFIRMED") -> dict:
    date = pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=index)
    return {
        "pivot_id": f"{kind}:{index}", "price": price, "bar_index": index,
        "kind": kind, "status": status, "pivot_time": date.isoformat(),
        "confirmation_time": (date + pd.Timedelta(weeks=1)).isoformat() if status == "CONFIRMED" else None,
    }


def test_scenario_weights_are_normalized() -> None:
    assert sum(CONFIG["scenario"]["weights"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "WEAK"), (2.49, "WEAK"), (2.5, "MODERATE"), (4.99, "MODERATE"), (5.0, "STRONG"), (7.49, "STRONG"), (7.5, "VERY_STRONG")],
)
def test_strength_thresholds_are_frozen(score: float, expected: str) -> None:
    assert classify_strength(score) == expected


def test_zone_role_and_signed_distance_convention() -> None:
    assert assign_role(80, 90, 100) == "SUPPORT"
    assert assign_role(110, 120, 100) == "RESISTANCE"
    assert assign_role(95, 105, 100) == "TESTING"
    assert 80 / 100 - 1 == pytest.approx(-0.20)


@pytest.mark.parametrize("families", [1, 2, 3, 4])
def test_confluence_is_strict_family_count(families: int) -> None:
    assert confluence_class(families) == {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "VERY_HIGH"}[families]


def test_family_quality_deduplicates_repeated_methods() -> None:
    members = [
        {"family": "SWING_STRUCTURE", "source": "weekly_swing", "weight": 3.0},
        {"family": "SWING_STRUCTURE", "source": "weekly_swing", "weight": 3.0},
        {"family": "VOLUME_ACCEPTANCE", "source": "poc", "weight": 3.0},
        {"family": "VOLUME_ACCEPTANCE", "source": "hvn", "weight": 2.5},
        {"family": "MOVING_AVERAGE", "source": "sma50", "weight": 1.5},
        {"family": "MOVING_AVERAGE", "source": "sma100", "weight": 2.0},
        {"family": "MOVING_AVERAGE", "source": "sma200", "weight": 2.5},
    ]
    scores, sources = _family_scores(members)
    assert set(scores) == {"SWING_STRUCTURE", "VOLUME_ACCEPTANCE", "MOVING_AVERAGE"}
    assert sources["SWING_STRUCTURE"] == ["weekly_swing"]
    assert scores["VOLUME_ACCEPTANCE"] == 3.25
    assert scores["MOVING_AVERAGE"] == 3.0


def test_clustering_is_order_invariant_and_chain_protected() -> None:
    members = [
        {"member_id": "a", "price": 100.0, "family": "SWING_STRUCTURE", "source": "weekly_swing", "weight": 3.0},
        {"member_id": "b", "price": 100.5, "family": "FIBONACCI", "source": "fib_0.618", "weight": 1.75},
        {"member_id": "c", "price": 101.0, "family": "MOVING_AVERAGE", "source": "sma200", "weight": 2.5},
        {"member_id": "far", "price": 110.0, "family": "VOLUME_ACCEPTANCE", "source": "poc", "weight": 3.0},
    ]
    first = deterministic_clusters(members, current=100.0, atr=2.0, timeframe="WEEKLY")
    second = deterministic_clusters(list(reversed(members)), current=100.0, atr=2.0, timeframe="WEEKLY")
    signature = lambda groups: sorted(sorted(item["member_id"] for item in group) for group in groups)
    assert signature(first) == signature(second)
    assert ["far"] in signature(first)


def test_volume_interval_participates_across_its_full_price_node() -> None:
    members = [
        {
            "member_id": "ma", "price": 656.0, "low": 656.0, "high": 656.0,
            "family": "MOVING_AVERAGE", "source": "sma100", "weight": 2.0,
        },
        {
            "member_id": "volume", "price": 676.0, "low": 660.0, "high": 692.0,
            "family": "VOLUME_ACCEPTANCE", "source": "hvn", "weight": 2.5,
        },
        {
            "member_id": "fib", "price": 661.0, "low": 661.0, "high": 661.0,
            "family": "FIBONACCI", "source": "fib_0.786", "weight": 1.25,
        },
        {
            "member_id": "far", "price": 705.0, "low": 705.0, "high": 705.0,
            "family": "SWING_STRUCTURE", "source": "weekly_swing", "weight": 3.0,
        },
    ]
    clusters = deterministic_clusters(members, current=765.0, atr=17.5, timeframe="WEEKLY")
    signature = sorted(sorted(item["member_id"] for item in cluster) for cluster in clusters)
    assert ["fib", "ma", "volume"] in signature
    assert ["far"] in signature
    joined = next(cluster for cluster in clusters if {item["member_id"] for item in cluster} == {"fib", "ma", "volume"})
    assert len(_family_scores(joined)[0]) == 3


def test_interval_aware_weekly_radius_is_bounded_by_zone_width() -> None:
    weekly = CONFIG["clustering"]["weekly"]
    assert CONFIG["config_version"] == "TECHNICAL_OUTLOOK_SIMPLE_V3_CONFIG_V2"
    assert weekly["base_price_fraction"] == pytest.approx(0.0135)
    assert weekly["atr_multiplier"] == pytest.approx(0.825)
    assert weekly["radius_cap_fraction"] == pytest.approx(weekly["max_total_width_fraction"] / 2.0)


def test_unfinished_invalid_yahoo_bar_is_removed_before_validation() -> None:
    spec = CORE_ASSETS["BTC-USD"]
    frame = _normalized_frame(
        dates=pd.Series(pd.to_datetime(["2026-09-28", "2026-09-30"])),
        opens=pd.Series([100.0, 105.0]), highs=pd.Series([110.0, 106.0]),
        lows=pd.Series([95.0, 104.0]), closes=pd.Series([105.0, 103.0]),
        volumes=pd.Series([1_000.0, 1_100.0]), timeframe="1D", spec=spec,
        now=pd.Timestamp("2026-09-30 12:00:00"),
    )
    assert frame["is_closed"].tolist() == [True, False]
    validated = _validate_completed_bars(frame, spec)
    assert len(validated) == 1
    assert pd.Timestamp(validated.iloc[0]["timestamp"]) == pd.Timestamp("2026-09-28")

    frame.loc[1, "is_closed"] = True
    with pytest.raises(ElliottDataError):
        _validate_completed_bars(frame, spec)


def test_asset_specific_and_default_swing_configs() -> None:
    assert swing_config("SPY")["weekly"] == {"atr_multiplier": 2.75, "min_reversal_pct": 0.09}
    assert swing_config("QQQ")["weekly"] == {"atr_multiplier": 2.5, "min_reversal_pct": 0.10}
    assert swing_config("GLD")["daily"] == {"atr_multiplier": 2.25, "min_reversal_pct": 0.055}
    assert swing_config("BTC-USD")["calibration_start"] == "2020-01-01"
    custom = swing_config("IWM")
    assert custom["source"] == "SPY_DEFAULT"
    assert custom["daily"]["min_reversal_pct"] == 0.05


def test_causal_swing_requires_confirmation_and_persists_potential() -> None:
    frame = calculate_indicators(pd.DataFrame({
        "timestamp": pd.date_range("2020-01-03", periods=12, freq="W-FRI"),
        "open": [100, 102, 108, 112, 110, 105, 96, 92, 95, 103, 109, 111],
        "high": [101, 104, 110, 114, 112, 107, 98, 94, 97, 105, 111, 113],
        "low": [99, 100, 106, 110, 107, 102, 94, 90, 93, 101, 107, 109],
        "close": [100, 103, 109, 111, 109, 104, 96, 92, 96, 104, 110, 112],
        "volume": [1000] * 12,
    }))
    frame["atr14"] = 2.0
    pivots = detect_causal_swings(frame, timeframe="WEEKLY", atr_multiplier=1.0, min_reversal_pct=0.03)
    confirmed = [item for item in pivots if item["status"] == "CONFIRMED"]
    assert confirmed
    assert all(pd.Timestamp(item["confirmation_time"]) > pd.Timestamp(item["pivot_time"]) for item in confirmed)
    assert pivots[-1]["status"] == "POTENTIAL"
    assert pivots[-1]["confirmation_time"] is None


def test_one_active_fibonacci_framework_and_developing_multiplier() -> None:
    pivots = [_pivot(80, 1, "LOW"), _pivot(120, 5, "HIGH", status="POTENTIAL")]
    result = active_fibonacci_framework(pivots, timeframe="WEEKLY")
    assert result["status"] == "DEVELOPING"
    assert len(result["retracements"]) == 5
    assert {item["ratio"] for item in result["retracements"]} == {0.236, 0.382, 0.5, 0.618, 0.786}
    assert all(item["quality_multiplier"] == 0.75 for item in result["retracements"])
    assert [item["ratio"] for item in result["extensions"]] == [1.0, 1.272, 1.618]


def test_volume_profile_lookbacks_and_hvn_limits_are_explicit() -> None:
    daily = calculate_indicators(_frame(600))
    weekly = calculate_indicators(_frame(300, freq="W-FRI", amplitude=20.0))
    daily_profile = build_volume_profile(daily, timeframe="DAILY")
    weekly_profile = build_volume_profile(weekly, timeframe="WEEKLY")
    assert daily_profile["lookback_bars"] == 252
    assert weekly_profile["lookback_bars"] == 260
    assert daily_profile["number_of_bins"] == CONFIG["volume_profile"]["profile_bins"]
    assert daily_profile["smoothing_method"] == "gaussian"
    assert len(daily_profile["hvns"]) <= 4
    assert all(item["relative_prominence"] >= 0.08 - 1e-12 for item in daily_profile["hvns"])


def _scenario_inputs() -> dict:
    return {
        "current_price": 100.0,
        "weekly_atr": 5.0,
        "weekly_bars_count": 300,
        "structure": {"state": "BULL"},
        "momentum": {"score": 45.0, "classification": "POSITIVE"},
        "ma_structure": {"state": "BULL"},
        "volume_context": {"status": "AVAILABLE", "score": 60.0},
        "analogs": {"returns": {"6M": {"median": 5.0}}},
        "extension_pct": 10.0,
        "weekly_zones": [
            {"zone_id": "s1", "role": "SUPPORT", "low": 88.0, "high": 92.0, "center": 90.0, "confluence_class": "HIGH", "quality_score": 7.0, "strength_class": "STRONG", "hidden_by_60pct_filter": False},
            {"zone_id": "r1", "role": "RESISTANCE", "low": 108.0, "high": 112.0, "center": 110.0, "confluence_class": "HIGH", "quality_score": 7.0, "strength_class": "MODERATE", "hidden_by_60pct_filter": False},
            {"zone_id": "r2", "role": "RESISTANCE", "low": 120.0, "high": 124.0, "center": 122.0, "confluence_class": "VERY_HIGH", "quality_score": 9.0, "strength_class": "WEAK", "hidden_by_60pct_filter": False},
        ],
        "fibonacci": {"extensions": [{"ratio": 1.272, "price": 127.2, "direction": "UP"}]},
    }


def test_weekly_scenario_is_independent_from_daily_and_has_no_synthetic_target() -> None:
    first = build_weekly_scenario_matrix(**_scenario_inputs())
    second = build_weekly_scenario_matrix(**_scenario_inputs())
    assert first == second
    assert sum(first["probabilities"].values()) == 100
    assert first["scenarios"][0]["trigger"].startswith("Confirmed Weekly close above")
    encoded = str(first)
    assert "1.08" not in encoded and "0.92" not in encoded


def test_testing_zone_boundaries_are_triggers_not_targets() -> None:
    values = _scenario_inputs()
    values["weekly_zones"] = [
        {"zone_id": "testing", "role": "TESTING", "low": 95.0, "high": 105.0, "center": 100.0, "confluence_class": "VERY_HIGH", "quality_score": 10.0, "strength_class": "STRONG", "hidden_by_60pct_filter": False},
        {"zone_id": "support", "role": "SUPPORT", "low": 82.0, "high": 86.0, "center": 84.0, "confluence_class": "HIGH", "quality_score": 7.0, "strength_class": "STRONG", "hidden_by_60pct_filter": False},
        {"zone_id": "resistance", "role": "RESISTANCE", "low": 114.0, "high": 118.0, "center": 116.0, "confluence_class": "HIGH", "quality_score": 7.0, "strength_class": "STRONG", "hidden_by_60pct_filter": False},
    ]
    result = build_weekly_scenario_matrix(**values)
    bull, _, bear = result["scenarios"]
    assert bull["trigger"] == "Confirmed Weekly close above 105.00"
    assert bear["trigger"] == "Confirmed Weekly close below 95.00"
    assert (bull["primary_target"] or {}).get("zone_id") != "testing"
    assert (bear["primary_target"] or {}).get("zone_id") != "testing"


def test_missing_component_weights_are_renormalized() -> None:
    components = {
        "directional_components": {
            "market_structure": {"value": 100.0, "configured_weight": 0.30},
            "momentum": {"value": None, "configured_weight": 0.25},
            "ma_structure": {"value": 50.0, "configured_weight": 0.15},
        },
        "extension_risk": {"available": False, "risk_fraction": 0.0},
    }
    result = scenario_probabilities(components)
    assert sum(result.values()) == 100
    assert sum(components["effective_weights"].values()) == pytest.approx(1.0)
    assert components["effective_weights"]["market_structure"] == pytest.approx(2 / 3)


def test_extension_risk_cannot_independently_flip_bull_to_bear() -> None:
    components = {
        "directional_components": {
            "market_structure": {"value": 60.0, "configured_weight": 0.50},
            "momentum": {"value": 50.0, "configured_weight": 0.45},
        },
        "extension_risk": {"available": True, "risk_fraction": 1.0, "direction": "POSITIVE", "configured_weight": 0.05},
    }
    result = scenario_probabilities(components)
    assert result["BULLISH"] > result["BEARISH"]
    assert max(result, key=result.get) != "BEARISH"


def test_chart_visibility_is_not_limited_by_confluence() -> None:
    zones = [
        {"zone_id": "low-strong", "confluence_class": "LOW", "strength_class": "VERY_STRONG", "hidden_by_60pct_filter": False, "visible_on_chart": False},
        {"zone_id": "medium", "confluence_class": "MEDIUM", "strength_class": "WEAK", "hidden_by_60pct_filter": False, "visible_on_chart": False},
        {"zone_id": "high-weak", "confluence_class": "HIGH", "strength_class": "WEAK", "hidden_by_60pct_filter": False, "visible_on_chart": True},
    ]
    assert [zone["zone_id"] for zone in visible_zones(zones)] == ["low-strong", "medium", "high-weak"]
    rendered = visible_zone_frame([
        {**zones[0], "role": "SUPPORT", "low": 80, "high": 81},
        {**zones[1], "role": "SUPPORT", "low": 90, "high": 91},
        {**zones[2], "role": "RESISTANCE", "low": 110, "high": 111},
    ], timeframe="WEEKLY")
    assert list(rendered["zone_id"]) == ["low-strong", "medium", "high-weak"]
    assert rendered.loc[rendered["zone_id"] == "medium", "zone_opacity"].iloc[0] < rendered.loc[rendered["zone_id"] == "high-weak", "zone_opacity"].iloc[0]


def test_chart_source_filter_keeps_multi_family_zone_when_one_source_is_enabled() -> None:
    zones = [
        {"zone_id": "swing-volume", "source_families": ["SWING_STRUCTURE", "VOLUME_ACCEPTANCE"]},
        {"zone_id": "fib", "source_families": ["FIBONACCI"]},
        {"zone_id": "ma", "source_families": ["MOVING_AVERAGE"]},
    ]
    assert [zone["zone_id"] for zone in filter_chart_zones(zones, {"VOLUME_ACCEPTANCE"})] == ["swing-volume"]
    assert [zone["zone_id"] for zone in filter_chart_zones(zones, {"FIBONACCI", "MOVING_AVERAGE"})] == ["fib", "ma"]
    assert filter_chart_zones(zones, set()) == []


def test_chart_zone_colors_are_source_based() -> None:
    assert zone_source_color({"source_families": ["SWING_STRUCTURE"]}) == "#f97316"
    assert zone_source_color({"source_families": ["VOLUME_ACCEPTANCE"]}) == "#06b6d4"
    assert zone_source_color({"source_families": ["FIBONACCI"]}) == "#a855f7"
    assert zone_source_color({"source_families": ["MOVING_AVERAGE"]}) == "#facc15"
    assert zone_source_color({"source_families": ["FIBONACCI", "SWING_STRUCTURE"]}) == "#f8fafc"


def test_60pct_filter_hides_but_does_not_delete_zone() -> None:
    zones = [
        {"zone_id": "hidden", "confluence_class": "VERY_HIGH", "hidden_by_60pct_filter": True, "visible_on_chart": False},
        {"zone_id": "visible", "confluence_class": "HIGH", "hidden_by_60pct_filter": False, "visible_on_chart": True},
    ]
    assert len(zones) == 2
    assert [item["zone_id"] for item in visible_zones(zones)] == ["visible"]


def test_chart_is_fixed_to_500_bars_and_has_no_scale_binding() -> None:
    bars = calculate_indicators(_frame(650))
    chart = build_simple_v3_chart(bars, [], primary=None, alternative=None, timeframe="DAILY")
    spec = chart.to_dict()
    datasets = spec.get("datasets") or {}
    assert datasets and max(len(value) for value in datasets.values()) == 500
    assert "selection" not in str(spec).lower()
    assert '"bind":"scales"' not in str(spec).replace(" ", "").lower()


def test_simple_v3_never_imports_elliott_wave_engine() -> None:
    modules = [
        importlib.import_module("technical_outlook_simple_v3.engine"),
        importlib.import_module("technical_outlook_simple_v3.scenario"),
        importlib.import_module("technical_outlook_simple_v3.support_resistance"),
    ]
    assert all("ElliottWaveEngine" not in vars(module) for module in modules)


def test_engine_snapshot_is_isolated_and_weekly_scenario_has_no_daily_input() -> None:
    spec = CORE_ASSETS["SPY"]
    base = _frame(2600, amplitude=18.0)
    daily = _normalized_frame(
        dates=base["timestamp"], opens=base["open"], highs=base["high"], lows=base["low"],
        closes=base["close"], volumes=base["volume"], timeframe="1D", spec=spec,
        now=pd.Timestamp("2035-01-01"),
    )
    snapshot, charts = TechnicalOutlookSimpleV3Engine().analyze(
        daily, spec, created_at=datetime(2030, 1, 2, tzinfo=timezone.utc),
    )
    assert snapshot["model_version"] == "TECHNICAL_OUTLOOK_SIMPLE_V3"
    assert snapshot["config_version"] == "TECHNICAL_OUTLOOK_SIMPLE_V3_CONFIG_V2"
    assert snapshot["sr_engine_version"] == "SR_ENGINE_SIMPLE_V3_V2"
    assert snapshot["scenario_engine_version"] == "SCENARIO_ENGINE_SIMPLE_V3"
    assert "cross_timeframe_support_resistance" not in snapshot
    assert len(charts["1D"]) == 500
    assert len(charts["1W"]) <= 500
    assert all(zone["family_count"] == len(zone["source_families"]) for zone in snapshot["weekly_zones"])
    volume_members = [
        member
        for zone in snapshot["weekly_zones"]
        for member in zone["source_members"]
        if member["family"] == "VOLUME_ACCEPTANCE"
    ]
    assert volume_members and any(member["high"] > member["low"] for member in volume_members)
    assert sum(snapshot["scenario_probabilities"].values()) == 100


@pytest.mark.parametrize(
    ("ticker", "scale", "golden_hash"),
    [
        ("SPY", 1.0, "4e5ca7bfbfd2247b4e6aec3eaad7c15b48831b8897514f239f4f4310e27504b8"),
        ("QQQ", 1.7, "2734fdfb34e98da3d64250ed5931c72b88ecf71fec82c1397ce1c53501618efb"),
        ("GLD", 0.65, "e2b45bfc3bf8edfd9ef7af83b4e12b7e4f94a0f9eb629fb40811977636a25331"),
        ("BTC-USD", 150.0, "c4127b786171101173724c2a9da1dc0cc6d67e6850673ed2c87b05336ed34329"),
    ],
)
def test_current_technical_outlook_golden_outputs_remain_unchanged(ticker: str, scale: float, golden_hash: str) -> None:
    spec = CORE_ASSETS[ticker]
    periods = 1400
    dates = pd.date_range("2018-01-02", periods=periods, freq="B")
    index = np.arange(periods)
    close = (100 + index * 0.06 + np.sin(index / 18.0) * 8) * scale
    daily = _normalized_frame(
        dates=dates,
        opens=pd.Series(close - 0.3 * scale), highs=pd.Series(close + 1.2 * scale),
        lows=pd.Series(close - 1.2 * scale), closes=pd.Series(close),
        volumes=pd.Series(1_000_000 + index * 10), timeframe="1D", spec=spec,
        now=pd.Timestamp("2035-01-01"),
    )
    snapshot, _ = TechnicalOutlookEngine().analyze(
        daily, spec, created_at=datetime(2030, 1, 2, tzinfo=timezone.utc),
    )
    canonical = {
        "model_version": snapshot["model_version"],
        "config_version": snapshot["config_version"],
        "sr_engine_version": snapshot["sr_engine_version"],
        "scenario_engine_version": snapshot["scenario_engine_version"],
        "daily_support_resistance": snapshot["daily_support_resistance"],
        "weekly_support_resistance": snapshot["weekly_support_resistance"],
        "cross_timeframe_support_resistance": snapshot["cross_timeframe_support_resistance"],
        "scenario_probabilities": snapshot["scenario_probabilities"],
        "daily_scenarios": snapshot["daily_scenarios"],
        "weekly_scenarios": snapshot["weekly_scenarios"],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == golden_hash

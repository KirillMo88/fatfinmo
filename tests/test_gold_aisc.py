from __future__ import annotations

import numpy as np
import pandas as pd

from gold_regime.aisc import (
    ANNUAL_AISC_HISTORY,
    AISC_QUARTERLY_STATUS,
    NORMAL_AISC_MULTIPLE,
    aisc_valuation_state,
    build_gold_aisc_valuation,
    build_quarterly_aisc,
)
from gold_regime.aisc_view import build_gold_aisc_valuation_fig, filter_aisc_range
from gold_regime.demand_structure import DEMAND_CATEGORIES, build_demand_structure_frame
from gold_regime.demand_structure_view import build_demand_structure_fig, filter_demand_range


def test_quarterly_aisc_forecast_compounds_from_last_actual() -> None:
    quarterly = build_quarterly_aisc(
        "2026Q4",
        actual_quarterly={"2026Q1": 1785.0},
    )

    assert quarterly["aisc_source"].tolist() == ["ACTUAL", "ESTIMATED", "ESTIMATED", "ESTIMATED"]
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q2"), "aisc"].iloc[0], 1829.625)
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q3"), "aisc"].iloc[0], 1875.365625)
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q4"), "aisc"].iloc[0], 1922.249765625)


def test_annual_aisc_history_is_repeated_for_each_quarter_with_source_status() -> None:
    quarterly = build_quarterly_aisc("2012Q4")

    assert len(ANNUAL_AISC_HISTORY) == 13
    for year, annual_value in ANNUAL_AISC_HISTORY.items():
        year_rows = quarterly.loc[quarterly["quarter"].str.startswith(f"{year} ")]
        assert year_rows["aisc"].tolist() == [annual_value] * 4

    assert [AISC_QUARTERLY_STATUS[pd.Period(f"2012Q{quarter}", freq="Q")] for quarter in range(1, 5)] == [
        "METALS_FOCUS", "METALS_FOCUS", "METALS_FOCUS", "METALS_FOCUS"
    ]
    assert AISC_QUARTERLY_STATUS[pd.Period("2000Q1", freq="Q")] == "RECONSTRUCTED"
    assert AISC_QUARTERLY_STATUS[pd.Period("2010Q1", freq="Q")] == "METALS_FOCUS_RETROSPECTIVE"
    assert quarterly.loc[quarterly["quarter"].eq("2013 Q1"), "aisc"].iloc[0] == 1120.0


def test_historical_annual_aisc_values_appear_in_gold_chart_with_readable_sources() -> None:
    dates = pd.to_datetime(["2000-01-03", "2010-01-04", "2012-01-03"])
    gold = pd.Series([280.0, 1100.0, 1600.0], index=dates)
    valuation = build_gold_aisc_valuation(gold)
    figure = build_gold_aisc_valuation_fig(valuation)
    historical = next(trace for trace in figure.data if trace.name == "Historical AISC")

    assert historical.y.tolist() == [250.0, 801.0, 1112.0]
    assert [row[1] for row in historical.customdata] == [
        "Reconstructed", "Metals Focus retrospective", "Metals Focus"
    ]


def test_daily_aisc_is_a_quarterly_step_function_and_actual_overrides_estimate() -> None:
    dates = pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02"])
    gold = pd.Series([4600.0, 4671.792, 4700.0, 4710.0], index=dates)
    valuation = build_gold_aisc_valuation(
        gold,
        actual_quarterly={"2026Q1": 1785.0, "2026Q2": 1800.0},
    )

    assert valuation["aisc"].tolist() == [1785.0, 1785.0, 1800.0, 1800.0]
    assert valuation["aisc_source"].tolist() == ["ACTUAL", "ACTUAL", "ACTUAL", "ACTUAL"]
    assert np.isclose(valuation.loc[1, "gold_aisc_ratio"], 4671.792 / 1785.0)
    assert np.isclose(
        valuation.loc[1, "premium_discount_pct"],
        (4671.792 / (1785.0 * NORMAL_AISC_MULTIPLE) - 1.0) * 100.0,
    )


def test_new_actual_quarter_replaces_estimate_and_state_thresholds_are_dynamic() -> None:
    dates = pd.date_range("2026-04-01", periods=2, freq="D")
    gold = pd.Series([4600.0, 4600.0], index=dates)
    estimated = build_gold_aisc_valuation(gold, actual_quarterly={"2026Q1": 1785.0})
    actual = build_gold_aisc_valuation(gold, actual_quarterly={"2026Q1": 1785.0, "2026Q2": 1829.625})

    assert estimated["aisc_source"].tolist() == ["ESTIMATED", "ESTIMATED"]
    assert actual["aisc_source"].tolist() == ["ACTUAL", "ACTUAL"]
    assert aisc_valuation_state(1.20) == "Very compressed producer economics"
    assert aisc_valuation_state(1.625) == "Normal historical range"
    assert aisc_valuation_state(2.50) == "Extreme / unusual"


def test_aisc_range_uses_the_shared_global_range_anchor() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2022-01-01", "2026-12-31", freq="MS")})
    filtered = filter_aisc_range(frame, "1Y", pd.Timestamp("2026-06-30"))

    assert filtered["date"].min() == pd.Timestamp("2025-07-01")
    assert filtered["date"].max() == pd.Timestamp("2026-06-01")


def test_demand_structure_has_three_quarterly_stacked_series_and_shared_range() -> None:
    frame = build_demand_structure_frame()
    filtered = filter_demand_range(frame, "5Y", pd.Timestamp("2026-09-27"))
    figure = build_demand_structure_fig(
        filtered,
        (pd.Timestamp("2021-09-27"), pd.Timestamp("2026-09-27")),
    )

    assert len(frame) == 264
    assert DEMAND_CATEGORIES == ["Jewellery", "Technology", "Investment", "Central Banks"]
    assert frame["date"].min() == pd.Timestamp("2010-03-31")
    assert frame["date"].max() == pd.Timestamp("2026-06-30")
    assert len(figure.data) == 12
    assert figure.layout.barmode == "relative"
    assert list(figure.layout.xaxis.range) == [pd.Timestamp("2021-09-27"), pd.Timestamp("2026-09-27")]

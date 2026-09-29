from __future__ import annotations

import numpy as np
import pandas as pd

from gold_regime.aisc import (
    NORMAL_AISC_MULTIPLE,
    aisc_valuation_state,
    build_gold_aisc_valuation,
    build_quarterly_aisc,
)
from gold_regime.aisc_view import filter_aisc_range


def test_quarterly_aisc_forecast_compounds_from_last_actual() -> None:
    quarterly = build_quarterly_aisc(
        "2026Q4",
        actual_quarterly={"2026Q1": 1785.0},
    )

    assert quarterly["aisc_source"].tolist() == ["ACTUAL", "ESTIMATED", "ESTIMATED", "ESTIMATED"]
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q2"), "aisc"].iloc[0], 1829.625)
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q3"), "aisc"].iloc[0], 1875.365625)
    assert np.isclose(quarterly.loc[quarterly["quarter"].eq("2026 Q4"), "aisc"].iloc[0], 1922.249765625)


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

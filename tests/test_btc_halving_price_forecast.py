import numpy as np

from btc_halving_price_forecast import (
    DEFAULT_BTC_BOTTOM_2026,
    DEFAULT_HALVING_TO_TOP_MULTIPLIERS,
    build_btc_halving_price_forecast,
    historical_return_table,
    historical_timing_table,
    validate_btc_halving_price_forecast,
)
from btc_cycle_tab import _halving_forecast_summary


def test_default_btc_halving_forecast_calculates_all_exposed_fields():
    forecast = build_btc_halving_price_forecast()
    expected_fields = {
        "BTC_Bottom_2026",
        "BTC_BottomToHalving_2015_2017",
        "BTC_BottomToHalving_2018_2021",
        "BTC_BottomToHalving_2022_2025",
        "BTC_Avg_BottomToHalving",
        "BTC_HalvingToTop_Conservative",
        "BTC_HalvingToTop_Base",
        "BTC_HalvingToTop_StrongLiquidity",
        "BTC_PriceAtHalving",
        "BTC_CycleTop_Conservative",
        "BTC_CycleTop_Base",
        "BTC_CycleTop_StrongLiquidity",
        "BTC_CycleTop_Average",
    }

    assert expected_fields <= forecast.keys()
    assert forecast["BTC_Bottom_2026"] == DEFAULT_BTC_BOTTOM_2026
    assert np.isclose(forecast["BTC_Avg_BottomToHalving"], (3.20 + 3.04 + 3.86) / 3)
    assert np.isclose(forecast["BTC_PriceAtHalving"], 194_421.63333333333)
    assert np.isclose(forecast["BTC_CycleTop_Conservative"], 233_305.96)
    assert np.isclose(forecast["BTC_CycleTop_Base"], 262_469.205)
    assert np.isclose(forecast["BTC_CycleTop_StrongLiquidity"], 291_632.45)
    assert np.isclose(forecast["BTC_CycleTop_Average"], forecast["BTC_CycleTop_Base"])
    assert not validate_btc_halving_price_forecast(forecast)
    assert set(DEFAULT_HALVING_TO_TOP_MULTIPLIERS) == {
        "Conservative",
        "Base",
        "Strong liquidity",
    }


def test_bottom_price_and_scenario_inputs_recalculate_forecast():
    forecast = build_btc_halving_price_forecast(
        bottom_price=60_000,
        halving_to_top_multipliers={
            "Conservative": 1.10,
            "Base": 1.30,
            "Strong liquidity": 1.70,
        },
    )

    assert np.isclose(forecast["BTC_PriceAtHalving"], 60_000 * (3.20 + 3.04 + 3.86) / 3)
    assert np.isclose(forecast["BTC_CycleTop_Conservative"], forecast["BTC_PriceAtHalving"] * 1.10)
    assert np.isclose(forecast["BTC_CycleTop_Base"], forecast["BTC_PriceAtHalving"] * 1.30)
    assert np.isclose(forecast["BTC_CycleTop_StrongLiquidity"], forecast["BTC_PriceAtHalving"] * 1.70)
    assert not np.isclose(forecast["BTC_CycleTop_Average"], forecast["BTC_CycleTop_Base"])


def test_historical_return_table_formats_multiples_and_returns():
    table = historical_return_table()

    assert list(table.columns) == ["Cycle", "Bottom → Halving", "Halving → Top", "Bottom → Top"]
    assert table.iloc[0].tolist() == ["2015–2017", "3.20x / +220%", "26.57x / +2,557%", "85.09x / +8,409%"]
    assert table.iloc[1]["Halving → Top"] == "6.52x / +552%"
    assert table.iloc[2]["Bottom → Top"] == "7.40x / +640%"


def test_historical_timing_table_formats_elapsed_time_from_day_inputs():
    table = historical_timing_table()

    assert list(table.columns) == ["Cycle", "Bottom → Top", "Bottom → Halving", "Halving → Top"]
    assert table.iloc[0].tolist() == [
        "2015–2017",
        "1064 days / 152 weeks / 35.0 months",
        "540 days / 77.1 weeks / 17.7 months",
        "524 days / 74.9 weeks / 17.2 months",
    ]
    assert table.iloc[1]["Bottom → Halving"] == "514 days / 73.4 weeks / 16.9 months"
    assert table.iloc[2]["Halving → Top"] == "531 days / 75.9 weeks / 17.4 months"


def test_forecast_validation_detects_broken_scenario_calculations():
    forecast = build_btc_halving_price_forecast()
    forecast["BTC_CycleTop_Base"] += 1

    assert validate_btc_halving_price_forecast(forecast) == [
        "Base cycle-top estimate does not reconcile",
        "Average cycle-top forecast does not reconcile",
    ]


def test_dynamic_summary_reflects_reordered_scenario_multipliers():
    forecast = build_btc_halving_price_forecast(
        60_000,
        {"Conservative": 1.80, "Base": 1.40, "Strong liquidity": 1.00},
    )
    summary = _halving_forecast_summary(forecast)

    assert "2026 bottom of $60.0k" in summary
    assert "1.00x–1.80x" in summary
    assert "projected cycle-top range of approximately $202k–$364k" in summary

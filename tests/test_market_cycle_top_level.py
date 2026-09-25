from __future__ import annotations

import numpy as np
import pandas as pd

from market_cycle import (
    build_momentum_monthly,
    build_structural_monthly,
    calculate_current_spx_performance,
    overlay_latest_daily_current_risk,
)
from market_cycle_tab import build_top_level_analytics, range_domain


def monthly_fixture() -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=280, freq="ME")
    trend = np.linspace(100.0, 500.0, len(dates))
    cycle = 15.0 * np.sin(np.arange(len(dates)) / 9.0)
    return pd.DataFrame({"Date": dates, "SPX_Close": trend + cycle})


def test_structural_and_medium_term_momentum_windows_are_calculated() -> None:
    structural = build_structural_monthly(monthly_fixture())
    medium = build_momentum_monthly(monthly_fixture())

    for months in [1, 3, 6, 12]:
        structural_col = f"StructuralROCMomentum_{months}M"
        medium_col = f"SPX_ROC_Momentum_{months}M"
        assert structural_col in structural
        assert medium_col in medium
        expected_structural = structural["SPX_ROC36M_3MMA"] - structural["SPX_ROC36M_3MMA"].shift(months)
        expected_medium = medium["SPX_ROC12M_3MMA"] - medium["SPX_ROC12M_3MMA"].shift(months)
        pd.testing.assert_series_equal(structural[structural_col], expected_structural, check_names=False)
        pd.testing.assert_series_equal(medium[medium_col], expected_medium, check_names=False)

    close = monthly_fixture()["SPX_Close"]
    for months in [1, 3, 6, 12]:
        expected = close.pct_change(months, fill_method=None)
        pd.testing.assert_series_equal(medium[f"SPX_ROC{months}M"], expected, check_names=False)


def test_top_level_analytics_contains_requested_three_sections() -> None:
    cards = build_top_level_analytics(
        {
            "StructuralExtensionZone": "OVEREXTENDED",
            "StructuralExtensionPercentile": 94.8,
            "SMA200WZone": "OVEREXTENDED",
            "SMA200WExtensionPercentile": 89.4,
            "SPX_ROC36M_3MMA": 0.71,
            "StructuralROCMomentum_12M": 0.07,
            "SPX_ROC12M_3MMA": 0.19,
            "SPX_ROC_Momentum_3M": -0.053,
            "PerformanceROC1M": 0.010,
            "PerformanceROC3M": 0.035,
            "PerformanceROC6M": 0.189,
            "PerformanceROC12M": 0.161,
            "CurrentMarketRiskState": "NORMAL",
            "CurrentRiskBreadthRisk": 80.0,
            "CurrentRiskRSIDivergenceRisk": 45.0,
            "CurrentRiskVIXRisk": 20.0,
            "CurrentRiskHighBetaRisk": 10.0,
            "CurrentRiskHYRisk": 5.0,
        }
    )

    assert [title for title, _ in cards] == ["Structural Market Cycle", "Medium Term Market Cycle", "Performance", "Current Risk"]
    assert cards[0][1][0] == ("SMA 200M Extension", "Overextended (94.8)")
    assert dict(cards[2][1]) == {
        "ROC 1M": "1.0%",
        "ROC 3M": "3.5%",
        "ROC 6M": "18.9%",
        "ROC 12M": "16.1%",
    }
    assert dict(cards[3][1]) == {
        "Status": "NORMAL",
        "Breadth Risk": "HIGH",
        "RSI Divergence Risk": "MODERATE",
        "VIX Risk": "LOW",
        "High Beta Risk": "LOW",
        "High Yield Risk": "LOW",
    }


def test_current_performance_includes_the_partial_current_month() -> None:
    dates = pd.date_range("2025-09-30", periods=12, freq="ME").append(pd.DatetimeIndex(["2026-09-22"]))
    daily = pd.DataFrame(
        {"SPX_Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0, 111.0, 112.0, 115.0, 116.0, 118.0, 120.0]},
        index=dates,
    )

    performance = calculate_current_spx_performance(daily)

    assert performance["PerformanceROC1M"] == 120.0 / 118.0 - 1.0
    assert performance["PerformanceROC3M"] == 120.0 / 115.0 - 1.0
    assert performance["PerformanceROC6M"] == 120.0 / 110.0 - 1.0
    assert performance["PerformanceROC12M"] == 120.0 / 100.0 - 1.0


def test_latest_daily_current_risk_overlays_stale_weekly_summary() -> None:
    daily = pd.DataFrame(
        {
            "CurrentRiskVIXRisk": [20.0, 35.0],
            "CurrentMarketRiskState": ["NORMAL", "MODERATE"],
            "CurrentRiskFrequency": ["DAILY", "DAILY"],
        },
        index=pd.to_datetime(["2026-09-18", "2026-09-22"]),
    )
    daily.index.name = "Date"

    current = overlay_latest_daily_current_risk(
        {"CurrentRiskVIXRisk": 10.0, "CurrentMarketRiskState": "NORMAL", "CycleContext": "UNCHANGED"},
        daily,
    )

    assert current["CurrentRiskVIXRisk"] == 35.0
    assert current["CurrentMarketRiskState"] == "MODERATE"
    assert current["CurrentRiskAsOfDate"] == pd.Timestamp("2026-09-22")
    assert current["CycleContext"] == "UNCHANGED"


def test_daily_range_uses_latest_index_date_not_last_completed_week() -> None:
    daily = pd.DataFrame(index=pd.to_datetime(["2026-09-18", "2026-09-21", "2026-09-22"]))

    start, end = range_domain(daily, "1Y")

    assert end == pd.Timestamp("2026-09-22")
    assert start == pd.Timestamp("2025-09-22")

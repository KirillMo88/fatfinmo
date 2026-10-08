import numpy as np
import pandas as pd

from top_analytics import build_top_analytics


def test_top_analytics_uses_requested_order_and_source_metrics():
    dates = pd.date_range("2020-01-01", periods=2400, freq="D")
    vix = pd.DataFrame({"Date": dates, "VIX": np.linspace(15.0, 30.0, len(dates))})
    move = pd.DataFrame({"Date": dates, "MOVE": np.linspace(70.0, 110.0, len(dates))})
    funding = pd.DataFrame(
        {"Date": dates, "FundingState": ["NORMAL"] * (len(dates) - 1) + ["TREASURY_VOLATILITY"]}
    )
    liquidity_dates = pd.date_range("2021-01-01", periods=210, freq="W-FRI")
    liquidity = pd.DataFrame(
        {"date": liquidity_dates, "global_liquidity_score": np.linspace(30.0, 60.0, len(liquidity_dates))}
    )
    inflation = pd.DataFrame(
        {
            "Date": dates,
            "PPIACO": np.linspace(200.0, 260.0, len(dates)),
            "US10Y": np.linspace(3.0, 4.25, len(dates)),
            "DXY": np.linspace(90.0, 102.0, len(dates)),
            "WTI": np.linspace(55.0, 75.0, len(dates)),
        }
    )

    metrics = build_top_analytics(
        "HIGH RISK", liquidity, vix, move, funding, inflation, "RISING", -0.06
    )

    assert [title for title, _, _ in metrics] == [
        "Current Risk",
        "Global Liquidity Score",
        "VIX",
        "MOVE",
        "Funding Stress",
        "Inflation",
    ]
    assert metrics[0][1] == "HIGH RISK"
    assert metrics[1][1] == "60.0"
    assert metrics[1][2][0].startswith("ROC 1M +")
    assert metrics[1][2][1].startswith("ROC 3M +")
    assert metrics[1][2][2].startswith("ROC 6M +")
    assert metrics[2][1] == "30.00" and metrics[2][2][2] == "5Y percentile 100%"
    assert metrics[2][2][0].startswith("1W change +")
    assert metrics[3][1] == "110.00" and metrics[3][2][1].startswith("4W change +")
    assert metrics[3][2][0].startswith("1W change +")
    assert metrics[4][1] == "TREASURY VOLATILITY"
    assert metrics[5][1] == "RISING (-0.06)"
    assert metrics[5][2][0].startswith("PPIACO 1M +")
    assert metrics[5][2][1].startswith("US10Y 4.25% · 1M +")
    assert metrics[5][2][2].startswith("DXY 102.00 · 1M +")
    assert metrics[5][2][3].startswith("WTI 75.00 · 1M +")


def test_inflation_card_reports_na_for_missing_history():
    metrics = build_top_analytics(
        None,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert metrics[-1] == (
        "Inflation",
        "N/A",
        [
            "PPIACO 1M N/A · 3M N/A · 6M N/A",
            "US10Y N/A · 1M N/A · 3M N/A · 6M N/A",
            "DXY N/A · 1M N/A · 3M N/A · 6M N/A",
            "WTI N/A · 1M N/A · 3M N/A · 6M N/A",
        ],
    )


def test_top_analytics_reports_na_when_five_year_history_is_incomplete():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    frame = pd.DataFrame({"Date": dates, "VIX": np.linspace(15.0, 25.0, len(dates))})

    metrics = build_top_analytics(None, pd.DataFrame(), frame, pd.DataFrame(), pd.DataFrame())

    assert metrics[2][2][2] == "5Y percentile N/A"


def test_current_risk_lists_components_with_high_or_higher_status():
    current = {
        "CurrentMarketRiskState": "ELEVATED",
        "CurrentRiskDrawdownRiskState": "HIGH",
        "CurrentRiskPriceCycleVulnerabilityRiskState": "MODERATE",
        "CurrentRiskBreadthRiskState": "RED FLAG",
        "CurrentRiskRSIDivergenceRiskState": "LOW",
        "CurrentRiskVIXRiskState": "HIGH RISK",
        "CurrentRiskHighBetaRiskState": "DATA INCOMPLETE",
        "CurrentRiskHYRiskState": "ELEVATED",
    }

    metrics = build_top_analytics(current, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert metrics[0][1] == "ELEVATED"
    assert metrics[0][2] == ["Market Cycle", "High+ components: Drawdown Risk, Breadth Risk, VIX Risk"]

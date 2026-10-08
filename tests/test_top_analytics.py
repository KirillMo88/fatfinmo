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

    metrics = build_top_analytics("HIGH RISK", liquidity, vix, move, funding)

    assert [title for title, _, _ in metrics] == [
        "Current Risk",
        "Global Liquidity Score",
        "VIX",
        "MOVE",
        "Funding Stress",
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

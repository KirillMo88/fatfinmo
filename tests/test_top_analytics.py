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
    assert "ROC 1M +" in metrics[1][2] and "ROC 3M +" in metrics[1][2]
    assert metrics[2][1] == "30.00" and "5Y percentile 100%" in metrics[2][2]
    assert metrics[3][1] == "110.00" and "4W change +" in metrics[3][2]
    assert metrics[4][1] == "TREASURY VOLATILITY"


def test_top_analytics_reports_na_when_five_year_history_is_incomplete():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    frame = pd.DataFrame({"Date": dates, "VIX": np.linspace(15.0, 25.0, len(dates))})

    metrics = build_top_analytics(None, pd.DataFrame(), frame, pd.DataFrame(), pd.DataFrame())

    assert "5Y percentile N/A" in metrics[2][2]

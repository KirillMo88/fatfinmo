import numpy as np
import pandas as pd

from alpha_engine import classify_opportunity_state
from market_model import (
    calculate_alpha_confidence,
    calculate_fast_transition_risk,
    calculate_fast_transition_risk_history,
    calculate_macro_transition_risk,
    calculate_macro_transition_risk_history,
    calculate_overall_transition_status,
    transition_state,
)


def weekly_series(values):
    return pd.Series(values, index=pd.date_range("2020-01-03", periods=len(values), freq="W-FRI"))


def test_transition_state_bands():
    assert transition_state(20.0, "ALERT") == "LOW"
    assert transition_state(40.0, "ALERT") == "WATCH"
    assert transition_state(60.0, "ALERT") == "DETERIORATING"
    assert transition_state(80.0, "ALERT") == "HIGH_RISK"
    assert transition_state(81.0, "ALERT") == "ALERT"


def test_fast_transition_risk_uses_vix_and_dxy():
    vix = weekly_series([20.0] * 26 + [40.0])
    dxy = weekly_series(np.linspace(100.0, 112.0, 27))

    result = calculate_fast_transition_risk(vix, dxy)

    assert result["Fast_Transition_State"] in {"HIGH_RISK", "TRANSITION_ALERT"}
    assert result["VIX_Risk"] > result["DXY_Risk"]


def test_macro_transition_risk_uses_dxy_liquidity_and_us2y():
    dates = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    fred = pd.DataFrame(
        [
            *({"Series_ID": "FED_LIQUIDITY", "Date": date, "Value": 1000.0 - i * 5.0} for i, date in enumerate(dates)),
            *({"Series_ID": "DGS2", "Date": date, "Value": 1.0 + i * 0.03} for i, date in enumerate(dates)),
        ]
    )
    dxy = pd.Series(np.linspace(100.0, 108.0, 30), index=dates)

    result = calculate_macro_transition_risk(dxy, fred)

    assert result["Macro_Transition_State"] != "DATA_INCOMPLETE"
    assert result["Fed_Liquidity_Risk"] > 10.0
    assert result["US2Y_Risk"] > 10.0


def test_transition_risk_history_returns_weekly_scores():
    dates = pd.date_range("2020-01-03", periods=35, freq="W-FRI")
    vix = pd.Series([20.0] * 26 + [35.0] * 9, index=dates)
    dxy = pd.Series(np.linspace(100.0, 110.0, 35), index=dates)
    fred = pd.DataFrame(
        [
            *({"Series_ID": "FED_LIQUIDITY", "Date": date, "Value": 1000.0 - i * 6.0} for i, date in enumerate(dates)),
            *({"Series_ID": "DGS2", "Date": date, "Value": 1.0 + i * 0.02} for i, date in enumerate(dates)),
        ]
    )

    fast = calculate_fast_transition_risk_history(vix, dxy)
    macro = calculate_macro_transition_risk_history(dxy, fred)

    assert fast["Fast_Transition_Risk"].dropna().iloc[-1] > 0.0
    assert macro["Macro_Transition_Risk"].dropna().iloc[-1] > 0.0
    assert fast["Date"].iloc[-1] == dates[-1]
    assert macro["Date"].iloc[-1] == dates[-1]


def test_overall_status_and_alpha_confidence_include_transition_risk():
    assert calculate_overall_transition_status(10.0, 10.0, 0.0) == "STABLE"
    assert calculate_overall_transition_status(85.0, 20.0, 0.0) == "TRANSITION_ALERT"
    assert calculate_alpha_confidence("BULL", 65.0, 65.0) == 50.0
    assert calculate_alpha_confidence("STRESS", 10.0, 10.0) == 20.0


def test_opportunity_state_handles_transition_layers():
    row = pd.Series(
        {
            "Alpha_Score": 78.0,
            "Entry_Risk_Score": 20.0,
            "Market_Regime": "BULL",
            "Fast_Transition_Risk": 65.0,
            "Macro_Transition_Risk": 20.0,
        }
    )

    assert classify_opportunity_state(row) == "FAST_TRANSITION_WARNING"

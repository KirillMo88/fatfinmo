import numpy as np
import pandas as pd

from alpha_engine import classify_opportunity_state
from market_model import (
    calculate_alpha_confidence,
    calculate_confirmations_history,
    calculate_fast_transition_risk,
    calculate_fast_transition_risk_history,
    calculate_macro_transition_risk,
    calculate_macro_transition_risk_history,
    calculate_overall_transition_status,
    classify_global_liquidity_backdrop,
    fast_transition_state,
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


def test_fast_transition_state_bands():
    assert fast_transition_state(10.0) == "LOW"
    assert fast_transition_state(20.0) == "NORMAL"
    assert fast_transition_state(40.0) == "WATCH"
    assert fast_transition_state(60.0) == "HIGH"
    assert fast_transition_state(61.0) == "EXTREME"


def test_fast_transition_risk_uses_vix_and_dxy():
    vix = weekly_series([20.0] * 26 + [40.0])
    dxy = weekly_series(np.linspace(100.0, 112.0, 27))

    result = calculate_fast_transition_risk(vix, dxy)

    assert result["Fast_Transition_State"] in {"HIGH", "EXTREME"}
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


def test_confirmations_history_returns_negative_count():
    dates = pd.date_range("2020-01-03", periods=35, freq="W-FRI")
    yahoo_weekly = {
        "CL=F": pd.DataFrame({"Close": np.linspace(50.0, 70.0, 35)}, index=dates),
        "SPY": pd.DataFrame({"Close": np.linspace(100.0, 130.0, 35)}, index=dates),
        "IWM": pd.DataFrame({"Close": np.linspace(100.0, 90.0, 35)}, index=dates),
        "XLI": pd.DataFrame({"Close": np.linspace(100.0, 95.0, 35)}, index=dates),
        "XLP": pd.DataFrame({"Close": np.linspace(100.0, 105.0, 35)}, index=dates),
    }
    fred = pd.DataFrame(
        [
            *({"Series_ID": "DFII10", "Date": date, "Value": 1.0 + i * 0.02} for i, date in enumerate(dates)),
        ]
    )

    history = calculate_confirmations_history(yahoo_weekly, fred)

    assert not history.empty
    assert "Negative_Confirmation_Count" in history.columns
    assert history["Negative_Confirmation_Count"].dropna().iloc[-1] >= 1.0


def test_overall_status_and_alpha_confidence_include_transition_risk():
    assert calculate_overall_transition_status(10.0, 10.0, 0.0, structural_regime="BULL") == "BULL"
    assert calculate_overall_transition_status(10.0, 10.0, 3.0, structural_regime="BULL") == "BULL"
    assert calculate_overall_transition_status(85.0, 20.0, 0.0, structural_regime="BULL") == "DETERIORATING"
    assert (
        calculate_overall_transition_status(
            5.0,
            18.0,
            3.0,
            structural_regime="BULL",
            global_liquidity_backdrop="LIQUIDITY_WARNING",
            global_liquidity_score=44.0,
            global_liquidity_direction_13w=-24.0,
            global_liquidity_direction_state="DETERIORATING_FAST",
        )
        == "BULL_LIQUIDITY_WARNING"
    )
    assert calculate_alpha_confidence("BULL", 65.0, 65.0) == 50.0
    assert calculate_alpha_confidence("STRESS", 10.0, 10.0) == 20.0


def test_global_liquidity_backdrop_classification():
    assert classify_global_liquidity_backdrop(65.0, 8.0, "IMPROVING") == "SUPPORTIVE"
    assert classify_global_liquidity_backdrop(44.0, -24.0, "DETERIORATING_FAST") == "LIQUIDITY_WARNING"
    assert classify_global_liquidity_backdrop(18.0, -12.0, "DETERIORATING_FAST") == "STRONGLY_NEGATIVE"


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

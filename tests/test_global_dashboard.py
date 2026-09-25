from types import SimpleNamespace

import numpy as np
import pandas as pd

from global_dashboard import build_global_dashboard_snapshot, classify_cross_cycle, format_forward_outlook
from market_cycle import build_asset_historical_outlook


def _namespace(**kwargs):
    defaults = {
        "current": {}, "history": pd.DataFrame(), "outlook": pd.DataFrame(),
        "asset_outlook": pd.DataFrame(), "daily": pd.DataFrame(), "weekly": pd.DataFrame(), "status": {},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_cross_cycle_uses_ordered_states_without_mega_score():
    assert classify_cross_cycle("BOTTOMING", "RISK CONTRACTION", "DETERIORATING") == "EARLY LIQUIDITY TURN"
    assert classify_cross_cycle("EXPANSION", "RISK EXPANSION", "IMPROVING") == "BROAD EXPANSION"
    assert classify_cross_cycle("PEAKING", "RISK EXPANSION", "IMPROVING") == "LIQUIDITY PEAKING"


def test_asset_outlook_reuses_independent_analog_rows():
    analogs = pd.DataFrame(
        {
            "Similarity": [74.0, 78.0],
            "Coverage": [0.9, 0.8],
            "ForwardReturn_SPY_12M": [0.10, -0.05],
            "ForwardMaxDrawdown_SPY_12M": [-0.08, -0.20],
            "ForwardReturn_QQQ_12M": [0.20, 0.00],
            "ForwardMaxDrawdown_QQQ_12M": [-0.12, -0.30],
        }
    )
    analogs.attrs["concentration_warning"] = "OK"
    result = build_asset_historical_outlook(analogs)
    spy = result.loc[(result["Asset"] == "SPY") & (result["Horizon"] == "12M")].iloc[0]
    qqq = result.loc[(result["Asset"] == "QQQ") & (result["Horizon"] == "12M")].iloc[0]
    assert spy["Independent Analog N"] == 2
    assert qqq["Independent Analog N"] == 2
    assert spy["Median Forward Return"] == 0.025
    assert qqq["Risk of >25% Drawdown"] == 0.5


def test_dashboard_maps_production_outputs_and_marks_missing_fragility():
    liquidity = pd.DataFrame(
        [{
            "date": "2026-09-18", "final_regime_label": "POSITIVE", "direction_13w_state": "ACCELERATING",
            "global_liquidity_score": 72, "long_cycle_phase": "ACCELERATING_EXPANSION",
        }]
    )
    forecast = pd.DataFrame(
        [{
            "Date": "2026-09-18", "LiquidityForecastState": "EXPANSION", "LiquidityPressureScore": 42,
            "PolicyResponseScore": 65, "BreadthParticipationState": "BROADENING", "RiskOnState": "CONFIRMED",
            "RiskReductionWarning": "NONE",
        }]
    )
    asset_rows = []
    for asset in ("SPY", "QQQ", "GLD", "BTC"):
        for horizon, value in (("3M", 0.03), ("6M", 0.06), ("12M", 0.12)):
            asset_rows.append({
                "Asset": asset, "Horizon": horizon, "Median Forward Return": value,
                "Positive Return Probability": 0.65, "P25 Return": -0.05, "P75 Return": 0.20,
                "Risk of >15% Drawdown": 0.2, "Risk of >25% Drawdown": 0.1, "Risk of >45% Drawdown": 0.0,
                "Independent Analog N": 30, "Average Similarity": 73, "Average Coverage": 0.9,
                "Confidence": "HIGH", "Concentration Warning": "OK",
            })
    market = _namespace(
        current={
            "Date": pd.Timestamp("2026-09-18"), "MomentumCyclePhase": "RISK EXPANSION",
            "CurrentMarketRiskState": "NORMAL", "CurrentMarketRiskDirection": "STABLE",
        },
        outlook=pd.DataFrame([{
            "Horizon": "12M", "Median Forward Return": 0.12, "Positive Return Probability": 0.65,
            "Risk of >15% Drawdown": 0.2, "Independent Analog N": 30, "Average Similarity": 73,
            "Average Coverage": 0.9, "Confidence": "HIGH", "Concentration Warning": "OK",
        }]),
        asset_outlook=pd.DataFrame(asset_rows),
    )
    business = _namespace(current={
        "date": pd.Timestamp("2026-09-11"), "BusinessCycleState": "STRONG EXPANSION",
        "BusinessCycleDirection": "IMPROVING", "InflationState": "FALLING", "EconomyRegime": "GOLDILOCKS",
    })
    macro = _namespace(current={"GrowthState": "POSITIVE", "CPISurpriseState": "COOLER"})
    rates = _namespace(history=pd.DataFrame([{"Date": "2026-09-18", "FinancialConditionsDirection": "EASING"}]))
    funding = _namespace(daily=pd.DataFrame([{"Date": "2026-09-18", "FundingState": "NORMAL"}]))
    treasury = _namespace(weekly=pd.DataFrame([{
        "Date": "2026-09-18", "TreasuryLiquidityState": "MILD INJECTION",
        "FiscalImpulseState": "POSITIVE / ACCELERATING", "TreasuryFinancingPressure": "MODERATE",
        "PolicyMix": "FISCAL & LIQUIDITY SUPPORT",
    }]))
    result = build_global_dashboard_snapshot(
        liquidity_regime=liquidity, forecast_frame=forecast, forecast_status={"DataAsOf": "2026-09-18"},
        market_snapshot=market, business_snapshot=business, macro_snapshot=macro, rates_snapshot=rates,
        funding_snapshot=funding, treasury_snapshot=treasury,
        transition_snapshot={"Fast_Transition_Risk": 20, "Macro_Transition_Risk": 30},
    )
    assert result.cross_cycle["CrossCycleState"] == "BROAD EXPANSION"
    assert result.fragility["FinancialFragilityState"] == "UNAVAILABLE"
    assert "Global Macro Score" not in result.fields
    assert len(result.forward_outlook) == 4
    assert result.forward_outlook.loc[result.forward_outlook["Asset"] == "BTC", "12M Median"].iloc[0] == 0.12


def test_forward_outlook_handles_partial_asset_history():
    raw = pd.DataFrame([
        {
            "Asset": "SPY", "Horizon": "12M", "Median Forward Return": 0.1,
            "Positive Return Probability": 0.6, "P25 Return": -0.1, "P75 Return": 0.2,
            "Risk of >15% Drawdown": 0.2, "Risk of >25% Drawdown": 0.1, "Risk of >45% Drawdown": 0.0,
            "Independent Analog N": 20, "Average Similarity": 70, "Average Coverage": 0.85,
            "Confidence": "MEDIUM", "Concentration Warning": "OK",
        }
    ])
    result = format_forward_outlook(raw)
    assert list(result["Asset"]) == ["SPY"]
    assert np.isnan(result.iloc[0]["3M Median"])

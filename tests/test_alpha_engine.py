import numpy as np
import pandas as pd

from alpha_engine import (
    calculate_adx_di_score,
    calculate_alpha_engine,
    calculate_sma200d_robust_z_36m,
    calculate_sma_regime_score,
    sort_by_alpha,
)
from entry_risk import perf12m_extreme_risk, regime_dependent_smaz_risk


def market(regime: str = "BULL", confidence: float = 100.0) -> dict:
    return {
        "Market_Regime": regime,
        "Alpha_Confidence": confidence,
        "SPY_vs_SMA40W_%": 5.0,
        "SPY_Drawdown_52W_%": -2.0,
        "SPY_Volatility_13W_%": 12.0,
        "SPY_Volatility_Percentile": 40.0,
    }


def alpha_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["LOW", "MID", "HIGH"],
            "Perf_1M_%": [1.0, 5.0, 12.0],
            "Perf_3M_%": [2.0, 8.0, 25.0],
            "Perf_6M_%": [3.0, 12.0, 35.0],
            "Perf_12M_%": [4.0, 20.0, 70.0],
            "Perf_12M_Percentile": [30.0, 70.0, 85.0],
            "ADX_14": [18.0, 27.0, 34.0],
            "DI_Plus_14": [14.0, 24.0, 35.0],
            "DI_Minus_14": [20.0, 16.0, 10.0],
            "SMA50w_vs_SMA200w_Spread_%": [-3.0, 2.0, 6.0],
            "SMA200d_Robust_Z_36M": [-0.4, 0.3, 1.0],
            "Price_vs_52W_High_%": [-24.0, -10.0, -2.0],
            "SMA200W_Distance_Percentile": [40.0, 75.0, 99.0],
            "RSI_14": [42.0, 58.0, 85.0],
        }
    )


def test_alpha_engine_outputs_scores_in_range_and_ranks_highest_first():
    scored = calculate_alpha_engine(alpha_input_frame(), market_regime=market())

    assert scored["Alpha_Score"].between(0.0, 100.0).all()
    assert scored["Momentum_Score"].between(0.0, 100.0).all()
    assert scored["Trend_Quality_Score"].between(0.0, 100.0).all()
    assert scored["Persistence_Score"].between(0.0, 100.0).all()
    assert scored["Entry_Risk_Score"].between(0.0, 100.0).all()
    assert scored["Opportunity_Score"].between(0.0, 100.0).all()
    assert scored["Alpha_Data_Complete"].all()
    assert scored.loc[scored["Ticker"] == "HIGH", "Alpha_Rank"].iloc[0] == 1.0


def test_alpha_score_equals_base_alpha_without_overextension_subtraction():
    scored = calculate_alpha_engine(alpha_input_frame(), market_regime=market()).set_index("Ticker")

    assert scored.loc["HIGH", "SMA200W_Distance_Percentile"] == 99.0
    assert scored.loc["HIGH", "RSI_14"] == 85.0
    assert scored.loc["HIGH", "Alpha_Score"] == scored.loc["HIGH", "Base_Alpha"]


def test_momentum_score_increases_with_stronger_relative_performance():
    scored = calculate_alpha_engine(alpha_input_frame(), market_regime=market()).set_index("Ticker")

    assert scored.loc["LOW", "Momentum_Score"] < scored.loc["MID", "Momentum_Score"]
    assert scored.loc["MID", "Momentum_Score"] < scored.loc["HIGH", "Momentum_Score"]


def test_sort_by_alpha_uses_entry_risk_as_last_tie_breaker():
    df = pd.DataFrame(
        {
            "Ticker": ["RISKIER", "BETTER"],
            "Alpha_Score": [80.0, 80.0],
            "Persistence_Score": [75.0, 75.0],
            "Trend_Quality_Score": [80.0, 80.0],
            "Momentum_Score": [80.0, 80.0],
            "Entry_Risk_Score": [35.0, 10.0],
            "Opportunity_Score": [52.0, 72.0],
        }
    )

    sorted_df = sort_by_alpha(df)

    assert sorted_df["Ticker"].tolist() == ["BETTER", "RISKIER"]


def test_sort_by_opportunity_score_keeps_alpha_score_separate():
    df = pd.DataFrame(
        {
            "Ticker": ["HIGH_ALPHA_HIGH_RISK", "LOWER_ALPHA_LOW_RISK"],
            "Alpha_Score": [90.0, 75.0],
            "Persistence_Score": [90.0, 70.0],
            "Trend_Quality_Score": [90.0, 70.0],
            "Momentum_Score": [90.0, 70.0],
            "Entry_Risk_Score": [70.0, 5.0],
            "Opportunity_Score": [27.0, 71.25],
        }
    )

    sorted_df = sort_by_alpha(df, sort_by="Opportunity Score")

    assert sorted_df["Ticker"].tolist() == ["LOWER_ALPHA_LOW_RISK", "HIGH_ALPHA_HIGH_RISK"]


def test_di_denominator_zero_sets_balance_to_zero():
    result = calculate_adx_di_score(
        pd.Series([30.0]),
        pd.Series([0.0]),
        pd.Series([0.0]),
    )

    assert result["DI_Balance"].iloc[0] == 0.0
    assert result["ADX_DI_Trend_Score"].iloc[0] == 0.0


def test_bearish_di_dominance_does_not_score_as_bullish_trend():
    result = calculate_adx_di_score(
        pd.Series([35.0]),
        pd.Series([10.0]),
        pd.Series([35.0]),
    )

    assert result["DI_Balance"].iloc[0] < 0.0
    assert result["ADX_DI_Trend_Score"].iloc[0] == 0.0


def test_relative_sma_score_uses_v13_piecewise_curve():
    df = alpha_input_frame()
    df["SMA200d_Robust_Z_36M"] = [-2.0, 2.0, 3.0]

    scored = calculate_sma_regime_score(df)

    assert scored["Relative_SMA_Score"].tolist() == [0.0, 90.0, 100.0]


def test_entry_risk_is_regime_dependent_for_smaz():
    assert regime_dependent_smaz_risk("BULL", 3.2) == 10.0
    assert regime_dependent_smaz_risk("BULL_HIGH_VOL", 2.75) == 5.0
    assert regime_dependent_smaz_risk("CORRECTION", 3.2) == 50.0
    assert regime_dependent_smaz_risk("STRESS", 3.2) == 80.0


def test_stress_with_high_momentum_and_extreme_smaz_forces_high_entry_risk():
    df = alpha_input_frame()
    df["SMA200d_Robust_Z_36M"] = [3.2, 3.3, 3.4]

    scored = calculate_alpha_engine(df, market_regime=market("STRESS", 20.0))

    assert scored.loc[scored["Ticker"] == "HIGH", "Entry_Risk_Score"].iloc[0] >= 80.0
    assert scored.loc[scored["Ticker"] == "HIGH", "Opportunity_State"].iloc[0] == "STRESS_AVOID_CHASING"


def test_perf12m_extreme_risk_is_soft_only_above_98():
    risk = perf12m_extreme_risk(
        pd.Series([98.0, 98.5, 99.5]),
        soft_threshold=98.0,
        extreme_threshold=99.0,
        soft_risk=5.0,
        extreme_risk=10.0,
    )

    assert risk.tolist() == [0.0, 5.0, 10.0]


def test_robust_z_returns_zero_when_mad_is_zero():
    dates = pd.date_range("2020-01-01", periods=760, freq="B")
    close = pd.Series(100.0, index=dates)

    assert calculate_sma200d_robust_z_36m(close) == 0.0


def test_robust_z_returns_nan_with_insufficient_history():
    dates = pd.date_range("2026-01-01", periods=220, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)

    assert np.isnan(calculate_sma200d_robust_z_36m(close))


def test_missing_critical_data_marks_alpha_incomplete():
    df = alpha_input_frame()
    df.loc[0, "ADX_14"] = np.nan

    scored = calculate_alpha_engine(df, market_regime=market())

    assert not scored.loc[0, "Alpha_Data_Complete"]
    assert np.isnan(scored.loc[0, "Alpha_Score"])
    assert scored.loc[0, "Alpha_State"] == "Missing Data"

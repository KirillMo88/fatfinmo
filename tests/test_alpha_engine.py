import numpy as np
import pandas as pd

from alpha_engine import (
    calculate_adx_di_score,
    calculate_alpha_engine,
    calculate_sma200d_robust_z_36m,
    quadratic_penalty,
    sort_by_alpha,
)


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
            "SMA200W_Distance_Percentile": [40.0, 75.0, 88.0],
            "RSI_14": [42.0, 58.0, 66.0],
        }
    )


def test_alpha_engine_outputs_scores_in_range_and_ranks_highest_first():
    scored = calculate_alpha_engine(alpha_input_frame())

    assert scored["Alpha_Score"].between(0.0, 100.0).all()
    assert scored["Momentum_Score"].between(0.0, 100.0).all()
    assert scored["Trend_Quality_Score"].between(0.0, 100.0).all()
    assert scored["Persistence_Score"].between(0.0, 100.0).all()
    assert scored["Alpha_Data_Complete"].all()
    assert scored.loc[scored["Ticker"] == "HIGH", "Alpha_Rank"].iloc[0] == 1.0


def test_momentum_score_increases_with_stronger_relative_performance():
    scored = calculate_alpha_engine(alpha_input_frame()).set_index("Ticker")

    assert scored.loc["LOW", "Momentum_Score"] < scored.loc["MID", "Momentum_Score"]
    assert scored.loc["MID", "Momentum_Score"] < scored.loc["HIGH", "Momentum_Score"]


def test_sort_by_alpha_uses_requested_tie_breakers():
    df = pd.DataFrame(
        {
            "Ticker": ["LOWER_PERSISTENCE", "BETTER"],
            "Alpha_Score": [80.0, 80.0],
            "Persistence_Score": [70.0, 75.0],
            "Trend_Quality_Score": [90.0, 80.0],
            "Momentum_Score": [90.0, 80.0],
            "Overextension_Penalty": [1.0, 5.0],
        }
    )

    sorted_df = sort_by_alpha(df)

    assert sorted_df["Ticker"].tolist() == ["BETTER", "LOWER_PERSISTENCE"]


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


def test_quadratic_penalty_is_zero_before_threshold_and_monotonic_after():
    penalties = quadratic_penalty(pd.Series([80.0, 90.0, 95.0, 100.0]), start=90.0, end=100.0, max_penalty=10.0)

    assert penalties.iloc[0] == 0.0
    assert penalties.iloc[1] == 0.0
    assert 0.0 < penalties.iloc[2] < penalties.iloc[3]
    assert penalties.iloc[3] == 10.0


def test_smaz_penalty_starts_after_one_and_half_sigma():
    penalties = quadratic_penalty(pd.Series([1.0, 1.5, 2.25, 3.0]), start=1.5, end=3.0, max_penalty=5.0)

    assert penalties.iloc[0] == 0.0
    assert penalties.iloc[1] == 0.0
    assert 0.0 < penalties.iloc[2] < penalties.iloc[3]
    assert penalties.iloc[3] == 5.0


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

    scored = calculate_alpha_engine(df)

    assert not scored.loc[0, "Alpha_Data_Complete"]
    assert np.isnan(scored.loc[0, "Alpha_Score"])
    assert scored.loc[0, "Alpha_State"] == "Missing Data"

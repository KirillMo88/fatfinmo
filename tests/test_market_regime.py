import numpy as np
import pandas as pd

from alpha_engine import alpha_config
from market_regime import calculate_market_regime


def weekly_spy_frame(close_values, high_offset=0.0):
    dates = pd.date_range("2020-01-03", periods=len(close_values), freq="W-FRI")
    close = pd.Series(close_values, index=dates, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * (1.0 + high_offset),
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


def test_market_regime_returns_bull_when_structural_bull_and_normal_vol():
    close = np.linspace(100.0, 150.0, 120)
    regime = calculate_market_regime(weekly_spy_frame(close), alpha_config())

    assert regime["Market_Regime"] == "BULL"
    assert regime["Alpha_Confidence"] == 100.0
    assert regime["SPY_vs_SMA40W_%"] > 0.0


def test_market_regime_returns_stress_when_below_sma_and_high_vol():
    trend = np.linspace(120.0, 160.0, 100)
    volatile_selloff = np.array([150, 130, 155, 120, 145, 110, 135, 100, 125, 95, 120, 90, 115, 85, 110], dtype=float)
    close = np.r_[trend, volatile_selloff]

    regime = calculate_market_regime(weekly_spy_frame(close), alpha_config())

    assert regime["Market_Regime"] == "STRESS"
    assert regime["Alpha_Confidence"] == 20.0
    assert regime["SPY_Drawdown_52W_%"] <= -10.0
    assert regime["SPY_Volatility_Percentile"] >= 75.0

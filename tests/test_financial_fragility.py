from types import SimpleNamespace

import numpy as np
import pandas as pd

from financial_fragility import _move_status, _rates_pressure_status, build_financial_fragility_snapshot


def test_stress_layer_uses_latest_daily_market_and_funding_values():
    dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    market_history = pd.DataFrame({
        "Date": dates,
        "SPX_Close": np.linspace(4000, 5000, len(dates)),
        "SMA200WExtensionPercentile": np.linspace(40, 80, len(dates)),
        "SPX_ROC_Momentum_3M": np.linspace(0.02, -0.02, len(dates)),
        "PositioningRisk": np.linspace(25, 55, len(dates)),
        "PositioningVulnerability": "NORMAL",
        "HistoricalBreadthRisk": 30.0,
        "HistoricalHighBetaRisk": 25.0,
        "HistoricalRSIDivergenceRisk": 20.0,
        "MomentumCyclePhase": "RISK EXPANSION",
        "VIX": 14.87,
        "VIX3M": 18.0,
        "RealizedVol20D": 0.12,
        "BAMLH0A0HYM2": 3.5,
    })
    daily_dates = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=22)
    market_daily = pd.DataFrame({
        "VIX": np.linspace(16.0, 18.0, len(daily_dates)),
        "VIX3M": 20.0,
        "RealizedVol20D": 0.22,
        "HY_OAS": 6.0,
    }, index=daily_dates)
    rates = pd.DataFrame({
        "Date": dates,
        "BAMLH0A0HYM2": 3.5,
        "RatesPressureScore": np.linspace(0, 1, len(dates)),
        "YieldCurveRegime_26W": "BEAR FLATTENING",
    })
    funding_weekly = pd.DataFrame({
        "Date": dates,
        "FundingCore": 0.25,
        "MoneyMarketStress": 0.10,
        "FundingState": "NORMAL",
    })
    funding_daily = pd.DataFrame({
        "Date": daily_dates[-3:],
        "FundingCore": [0.5, 0.75, 1.0],
        "MoneyMarketStress": [0.2, 0.7, 1.2],
        "FundingState": ["NORMAL", "NORMAL", "TECHNICAL FUNDING PRESSURE"],
        "PersistentFundingFlag": [False, False, False],
        "ReservePressure": [0.3, 0.4, 0.5],
        "CollateralStress": [0.0, 0.0, 0.0],
        "MOVE": [100.0, 110.0, 120.0],
        "MOVE_Z": [np.nan, np.nan, 1.25],
    })
    snapshot = build_financial_fragility_snapshot(
        liquidity_regime=pd.DataFrame(),
        forecast_frame=pd.DataFrame({"Date": dates, "LiquidityForecastState": "EXPANSION", "LiquidityPressureScore": 20.0}),
        market_snapshot=SimpleNamespace(history=market_history, daily=market_daily),
        business_snapshot=None,
        rates_snapshot=SimpleNamespace(history=rates),
        funding_snapshot=SimpleNamespace(weekly=funding_weekly, daily=funding_daily),
        treasury_snapshot=None,
    )

    assert snapshot.current["VIX"] == 18.0
    assert snapshot.current["HY_OAS"] == 6.0
    assert snapshot.current["FundingCore"] == 1.0
    assert snapshot.current["FundingState"] == "TECHNICAL FUNDING PRESSURE"
    assert snapshot.current["Curve26W"] == "BEAR FLATTENING"
    assert snapshot.current["MOVE"] == 120.0
    assert np.isclose(snapshot.current["MOVE_Z"], np.sqrt(1.5))
    assert snapshot.current["CollateralStressStatus"] == "HIGH"
    assert snapshot.current["MarketDataAsOf"] == daily_dates[-1]
    assert snapshot.current["FundingDataAsOf"] == daily_dates[-1]
    assert snapshot.current["RatesPressureStatus"] == "EXTREME"
    assert np.isfinite(snapshot.current["RealizedVolRisk"])
    assert np.isfinite(snapshot.current["MarketVolatilityStress"])


def test_stress_layer_status_thresholds():
    assert [_rates_pressure_status(value) for value in (0, 25, 25.1, 50, 50.1, 75, 75.1, 100)] == [
        "LOW", "LOW", "ELEVATED", "ELEVATED", "HIGH", "HIGH", "EXTREME", "EXTREME"
    ]
    assert [_move_status(value) for value in (49.9, 50, 80, 80.1, 110, 110.1, 141, 141.1)] == [
        "LOW", "NORMAL", "NORMAL", "ELEVATED", "ELEVATED", "HIGH", "HIGH", "EXTREME"
    ]

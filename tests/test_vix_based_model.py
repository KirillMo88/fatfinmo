import numpy as np
import pandas as pd

from vix_based_model import _market_weekly_vix, build_vix_model_data


def test_build_vix_model_data_from_market_cycle_daily_frame() -> None:
    dates = pd.bdate_range("2003-01-02", "2026-10-09")
    phase = np.linspace(0.0, 40.0 * np.pi, len(dates))
    source = pd.DataFrame(
        {
            "Date": dates,
            "SPX_Close": 900.0 * np.exp(np.linspace(0.0, 2.0, len(dates))),
            "VIX": 22.0 + 12.0 * np.sin(phase),
        }
    )

    model = build_vix_model_data(source)

    assert model.diagnostics["status"] == "ok"
    assert not model.frame.empty
    assert {
        "DailySpread",
        "WeeklySpread",
        "WeeklyRegime",
        "StrategyAWeight",
        "StrategyBWeight",
    }.issubset(model.frame.columns)
    assert model.frame["Date"].min() >= pd.Timestamp("2003-08-01")
    assert model.frame["StrategyAWeight"].between(0.0, 1.0).all()
    assert model.frame["StrategyBWeight"].between(0.0, 1.0).all()


def test_market_weekly_vix_excludes_open_exchange_week() -> None:
    source = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-10-02", "2026-10-05", "2026-10-06"]),
            "VIX_Close": [17.0, 18.0, 19.0],
        }
    )

    weekly = _market_weekly_vix(
        source,
        now=pd.Timestamp("2026-10-06 12:00", tz="America/New_York"),
    )

    assert weekly["Date"].tolist() == [pd.Timestamp("2026-10-02")]
    assert weekly["VIX_Close"].tolist() == [17.0]

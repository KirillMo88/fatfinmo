import numpy as np
import pandas as pd

from vix_based_model import (
    _deduplicated_signal,
    _market_weekly_vix,
    backtest_vxx_strategy,
    build_vix_model_data,
)


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


def _vix_signals(dates: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Date": dates,
            "DailyRapid": False,
            "VIX_Close": 16.0,
            "WeeklyRegime": "WHITE",
        }
    )
    frame.loc[frame["Date"].eq(signal_date), ["DailyRapid", "VIX_Close"]] = [True, 12.0]
    return frame


def _vxx_prices(
    dates: pd.DatetimeIndex,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=dates,
    )


def test_vxx_strategy_enters_next_close_and_takes_gap_profit() -> None:
    dates = pd.bdate_range("2026-01-02", periods=3)
    result = backtest_vxx_strategy(
        _vix_signals(dates, dates[0]),
        _vxx_prices(
            dates,
            opens=[10.0, 10.0, 18.0],
            highs=[10.5, 10.5, 19.0],
            lows=[9.5, 9.5, 17.5],
            closes=[10.0, 10.0, 18.5],
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["Entry Signal Date"] == dates[0]
    assert trade["Entry Date"] == dates[1]
    assert trade["Entry Price"] == 10.0
    assert trade["Exit Date"] == dates[2]
    assert trade["Exit Price"] == 18.0
    assert trade["Exit Reason"] == "TAKE PROFIT — GAP"
    expected_capital = ((1_000.0 * (1.0 - 0.0005)) / 10.0 * 18.0) * (1.0 - 0.0005)
    assert np.isclose(trade["Capital After ($)"], expected_capital)


def test_vxx_strategy_uses_stop_first_when_tp_and_stop_touch_same_bar() -> None:
    dates = pd.bdate_range("2026-01-02", periods=3)
    result = backtest_vxx_strategy(
        _vix_signals(dates, dates[0]),
        _vxx_prices(
            dates,
            opens=[10.0, 10.0, 10.0],
            highs=[10.5, 10.5, 18.0],
            lows=[9.5, 9.5, 7.0],
            closes=[10.0, 10.0, 12.0],
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["Exit Reason"] == "TRAILING STOP"
    assert trade["Exit Price"] == 8.0
    assert trade["Holding Sessions"] == 1


def test_vxx_strategy_updates_trailing_high_after_completed_bar() -> None:
    dates = pd.bdate_range("2026-01-02", periods=4)
    result = backtest_vxx_strategy(
        _vix_signals(dates, dates[0]),
        _vxx_prices(
            dates,
            opens=[10.0, 10.0, 10.0, 13.0],
            highs=[10.5, 10.5, 15.0, 14.0],
            lows=[9.5, 9.5, 9.0, 11.0],
            closes=[10.0, 10.0, 14.0, 12.5],
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["Exit Reason"] == "TRAILING STOP"
    assert trade["Exit Price"] == 12.0
    assert trade["Holding Sessions"] == 2


def test_vxx_strategy_exits_at_close_of_40th_full_holding_session() -> None:
    dates = pd.bdate_range("2026-01-02", periods=42)
    result = backtest_vxx_strategy(
        _vix_signals(dates, dates[0]),
        _vxx_prices(
            dates,
            opens=[10.0] * len(dates),
            highs=[10.5] * len(dates),
            lows=[9.5] * len(dates),
            closes=[10.0] * len(dates),
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["Exit Reason"] == "TIME EXIT"
    assert trade["Holding Sessions"] == 40
    assert trade["Exit Date"] == dates[41]
    assert trade["Exit Price"] == 10.0


def test_rapid_signal_cooldown_applies_before_vxx_filters() -> None:
    raw = pd.Series([True] + [False] * 9 + [True] + [False] * 52 + [True])

    deduplicated = _deduplicated_signal(raw, minimum_gap=63)

    assert deduplicated.iloc[0]
    assert not deduplicated.iloc[10]
    assert deduplicated.iloc[63]

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from finance_core import (
    _scenario_label,
    analyze_ticker,
    crossover_dates,
    drop_incomplete_daily_bar,
    ticker_metrics,
    weekly_ohlcv,
)


def make_ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1000.0,
        },
        index=close.index,
    )


def test_incomplete_current_utc_bar_is_removed() -> None:
    index = pd.to_datetime(["2026-06-12", "2026-06-14"])
    frame = make_ohlcv(pd.Series([100.0, 101.0], index=index))
    result = drop_incomplete_daily_bar(
        frame, datetime(2026, 6, 14, 6, tzinfo=timezone.utc)
    )
    assert result.index.tolist() == [pd.Timestamp("2026-06-12")]


def test_crosses_are_limited_to_last_fourteen_bars() -> None:
    index = pd.bdate_range("2025-01-01", periods=260)
    recent_golden = pd.Series([100.0] * 255 + [200.0] * 5, index=index)
    old_golden = pd.Series([100.0] * 220 + [200.0] * 40, index=index)
    golden, death = crossover_dates(recent_golden, 14)
    assert golden
    assert not death
    assert crossover_dates(old_golden, 14) == ([], [])


def test_daily_and_weekly_crosses_include_dates() -> None:
    index = pd.bdate_range("2020-01-01", periods=1300)
    close = pd.Series([100.0] * 1295 + [200.0] * 5, index=index)
    metrics = ticker_metrics("TEST", make_ohlcv(close))
    assert metrics["crosses"]["daily_golden"]
    assert all(
        len(value) == 10
        for values in metrics["crosses"].values()
        for value in values
    )
    assert weekly_ohlcv(make_ohlcv(close)).index[-1].weekday() == 4


def test_scenario_thresholds() -> None:
    assert _scenario_label(0.20, 20.0) == "bullish"
    assert _scenario_label(-0.20, 20.0) == "bearish"
    assert _scenario_label(0.05, 20.0) == "neutral"


def test_historical_probabilities_sum_to_one_hundred() -> None:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2018-01-01", periods=1400)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index)))),
        index=index,
    )
    report = analyze_ticker("TEST", make_ohlcv(close))
    assert report["analog_count"] == 75
    assert report["horizon_bars"] == 63
    assert sum(report["probabilities"].values()) == 100.0
    assert set(report["probabilities"]) == {"bullish", "bearish", "neutral"}


import numpy as np
import pandas as pd

from screener_metrics import (
    calendar_performance_series,
    correction_risk_from_percentile_analogs,
    expanding_percentile_rank,
    forward_calendar_return_series,
    historical_momentum_52w_metrics,
    percentile_rank,
    sma200w_distance_percentile,
)


def test_percentile_rank_uses_asset_own_distribution():
    values = pd.Series([10.0, 20.0, 30.0, 40.0])

    assert percentile_rank(values, 30.0) == 75.0


def test_expanding_percentile_rank_uses_history_available_at_each_point():
    values = pd.Series([10.0, 30.0, 20.0, 40.0])

    percentiles = expanding_percentile_rank(values)

    assert [round(value, 10) for value in percentiles.tolist()] == [
        100.0,
        100.0,
        round(2 / 3 * 100, 10),
        100.0,
    ]


def test_calendar_performance_series_matches_calendar_day_lookup():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-08"])
    close = pd.Series([100.0, 110.0, 121.0], index=dates)

    perf = calendar_performance_series(close, days=5)

    assert np.isnan(perf.iloc[0])
    assert round(perf.iloc[2], 2) == 10.0


def test_forward_calendar_return_series_excludes_incomplete_forward_windows():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 105.0, 110.0, 120.0, 130.0], index=dates)

    forward = forward_calendar_return_series(close, days=2)

    assert round(forward.iloc[0], 2) == 10.0
    assert round(forward.iloc[2], 2) == 18.18
    assert np.isnan(forward.iloc[-1])


def test_sma200w_distance_percentile_returns_current_distance_rank():
    dates = pd.date_range("2020-01-03", periods=230, freq="W-FRI")
    close = pd.Series(
        np.r_[np.linspace(100.0, 120.0, 220), np.linspace(125.0, 220.0, 10)],
        index=dates,
    )

    percentile = sma200w_distance_percentile(close)

    assert percentile > 90.0


def test_historical_momentum_52w_metrics_returns_percentile_and_75_analogs_average():
    dates = pd.date_range("2020-01-01", periods=650, freq="D")
    close = pd.Series(np.linspace(100.0, 220.0, len(dates)), index=dates)

    momentum_percentile, avg_forward_return_6m = historical_momentum_52w_metrics(close)

    assert 0.0 <= momentum_percentile <= 100.0
    assert avg_forward_return_6m > 0.0


def test_correction_risk_uses_sma_and_momentum_percentile_analogs():
    weekly_dates = pd.date_range("2018-01-05", periods=380, freq="W-FRI")
    trend = np.linspace(100.0, 180.0, len(weekly_dates))
    cycle = 18.0 * np.sin(np.linspace(0.0, 12.0 * np.pi, len(weekly_dates)))
    weekly_close = pd.Series(trend + cycle, index=weekly_dates)
    daily_dates = pd.date_range(weekly_dates[0], weekly_dates[-1], freq="D")
    daily_close = weekly_close.reindex(daily_dates).interpolate(method="time").ffill()

    correction_risk, correction_3m, correction_6m = correction_risk_from_percentile_analogs(
        daily_close,
        weekly_close,
    )

    assert 0.0 <= correction_risk <= 100.0
    assert 0.0 <= correction_3m <= 100.0
    assert 0.0 <= correction_6m <= 100.0
    assert round(correction_risk, 10) == round((0.7 * correction_3m) + (0.3 * correction_6m), 10)

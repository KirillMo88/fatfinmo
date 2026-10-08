import numpy as np
import pandas as pd

from commodity_cycle.ppi_led_etf import (
    EQUITY_ASSETS,
    build_performance_table,
    calculate_period_returns,
    relative_median_frame,
)


def test_returns_use_calendar_month_targets_and_previous_available_close():
    dates = pd.to_datetime(["2023-01-30", "2023-02-27", "2023-03-30", "2023-04-28"])
    prices = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates)

    result = calculate_period_returns(prices, "2023-04-30", horizons=[1, 2, 3])

    assert np.isclose(result["1M"], 40.0 / 30.0 - 1.0)
    assert np.isclose(result["2M"], 40.0 / 20.0 - 1.0)
    assert np.isclose(result["3M"], 40.0 / 10.0 - 1.0)


def test_performance_table_order_and_aggregates_exclude_ppiaco():
    dates = pd.date_range("2022-01-01", "2026-01-01", freq="MS")
    ppi = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    markets = {
        ticker: pd.Series(np.linspace(100, 150 + idx, len(dates)), index=dates)
        for idx, ticker in enumerate(EQUITY_ASSETS)
    }
    table, as_of = build_performance_table(ppi, markets)

    assert as_of == dates[-1]
    assert table.iloc[0]["Asset"] == "PPIACO"
    assert table.iloc[1]["Asset"] == "RTSI — Russia"
    assert table.iloc[-2]["Asset"] == "Average"
    assert table.iloc[-1]["Asset"] == "Median"
    assert len(table) == 20
    equity_returns = table.iloc[1:-2]["12M"]
    assert np.isclose(table.loc[table["Asset"].eq("Average"), "12M"].iloc[0], equity_returns.mean())
    assert np.isclose(table.loc[table["Asset"].eq("Median"), "12M"].iloc[0], equity_returns.median())
    assert not np.isclose(table.loc[table["Asset"].eq("Average"), "12M"].iloc[0], ppi.iloc[-1] / ppi.iloc[-13] - 1)


def test_relative_performance_is_equity_return_minus_equity_median_sorted_and_ranked():
    ppi = pd.Series(np.arange(100.0, 140.0), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
    markets = {
        ticker: pd.Series(np.arange(100.0, 140.0) * (1 + idx * 0.01), index=ppi.index)
        for idx, ticker in enumerate(EQUITY_ASSETS)
    }
    table, _ = build_performance_table(ppi, markets)
    relative = relative_median_frame(table, "12M")

    assert len(relative) == len(EQUITY_ASSETS)
    assert relative.iloc[0]["Rank"] == 1
    assert relative.iloc[-1]["Rank"] == len(EQUITY_ASSETS)
    assert relative["Relative"].is_monotonic_decreasing
    assert np.isclose(relative["Relative"].median(), 0.0)

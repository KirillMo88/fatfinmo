import numpy as np
import pandas as pd
import pytest

from ai_dashboard import (
    aggregate_return,
    breadth_percent,
    build_group_table,
    company_price_metrics,
    convert_krw_ohlcv_to_usd,
    convert_krw_value_to_usd,
    positive_percent,
)


def test_aggregate_return_equal_weighted_skips_missing_values():
    block = pd.DataFrame(
        {
            "Perf 1M": [10.0, np.nan, -4.0],
            "Market Cap": [100.0, 200.0, 300.0],
        }
    )

    assert aggregate_return(block, "Perf 1M", "Equal Weighted") == pytest.approx(3.0)


def test_aggregate_return_market_cap_weighted_reweights_valid_rows():
    block = pd.DataFrame(
        {
            "Perf 1M": [10.0, np.nan, -10.0],
            "Market Cap": [100.0, 900.0, 300.0],
        }
    )

    assert aggregate_return(block, "Perf 1M", "Market Cap Weighted") == pytest.approx(-5.0)


def test_breadth_and_positive_percent_skip_missing_values():
    assert breadth_percent(pd.Series([True, False, np.nan, True])) == pytest.approx(66.6667, rel=1e-4)
    assert positive_percent(pd.Series([5.0, -2.0, np.nan, 1.0])) == pytest.approx(66.6667, rel=1e-4)


def test_company_price_metrics_calculates_ath_and_trend_flags():
    dates = pd.bdate_range("2023-01-02", periods=260)
    close = pd.Series(np.linspace(100.0, 200.0, len(dates)), index=dates)
    high = close.copy()
    high.iloc[200] = 240.0
    frame = pd.DataFrame({"Close": close, "High": high})

    metrics = company_price_metrics(frame)

    assert metrics["Price vs ATH"] == pytest.approx((200.0 / 240.0) - 1.0)
    assert metrics["ATH Date"] == dates[200].date().isoformat()
    assert metrics["Above SMA50"] is True
    assert metrics["Above SMA200"] is True


def test_build_group_table_includes_group_momentum_percentile():
    companies = pd.DataFrame(
        {
            "Group": ["Compute", "Compute"],
            "Ticker": ["AAA", "BBB"],
            "Market Cap": [100.0, 300.0],
            "Perf 1D": [1.0, 2.0],
            "Perf 1W": [1.0, 2.0],
            "Perf 1M": [10.0, 20.0],
            "Perf 3M": [10.0, 20.0],
            "Perf 6M": [10.0, 20.0],
            "Perf 12M": [10.0, 20.0],
            "Perf 3Y": [10.0, 20.0],
            "Perf 5Y": [10.0, 20.0],
            "Perf 10Y": [10.0, 20.0],
            "SMA200W Percentile": [40.0, 80.0],
            "Perf 12M Percentile": [30.0, 70.0],
            "SMA200D Robust Z 36M": [0.5, 1.5],
            "Price vs SMA200D": [0.1, 0.2],
            "Above SMA50": [True, False],
            "Above SMA200": [True, True],
            "Trailing P/E": [20.0, 30.0],
            "Forward P/E": [18.0, 28.0],
            "PEG Ratio": [1.0, 2.0],
            "Price/Sales": [8.0, 10.0],
            "Quarterly Revenue Growth YoY": [0.2, 0.4],
            "Revenue Growth 3Y": [0.1, 0.3],
            "Operating Margin TTM": [0.25, 0.35],
            "Price vs ATH": [-0.1, -0.2],
        }
    )

    groups = build_group_table(companies, weighting="Market Cap Weighted", benchmark_returns={"Perf 1M": 5.0})
    row = groups.iloc[0]

    assert row["Total Market Cap"] == pytest.approx(400.0)
    assert row["Perf 1M"] == pytest.approx(17.5)
    assert row["SMA200W Percentile"] == pytest.approx(60.0)
    assert row["Perf 12M Percentile"] == pytest.approx(50.0)


def test_convert_krw_ohlcv_to_usd_divides_price_columns_only():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    frame = pd.DataFrame(
        {
            "Open": [1300.0, 2600.0],
            "High": [1430.0, 2860.0],
            "Low": [1170.0, 2340.0],
            "Close": [1365.0, 2730.0],
            "Volume": [100.0, 200.0],
        },
        index=dates,
    )
    fx = pd.DataFrame({"Close": [1300.0, 1300.0]}, index=dates)

    converted = convert_krw_ohlcv_to_usd(frame, fx)

    assert converted["Open"].tolist() == pytest.approx([1.0, 2.0])
    assert converted["High"].tolist() == pytest.approx([1.1, 2.2])
    assert converted["Low"].tolist() == pytest.approx([0.9, 1.8])
    assert converted["Close"].tolist() == pytest.approx([1.05, 2.1])
    assert converted["Volume"].tolist() == pytest.approx([100.0, 200.0])


def test_convert_krw_market_cap_to_usd_for_korean_tickers_only():
    assert convert_krw_value_to_usd("000660.KS", 1_300_000.0, 1300.0) == pytest.approx(1000.0)
    assert convert_krw_value_to_usd("005930.KS", 2_600_000.0, 1300.0) == pytest.approx(2000.0)
    assert convert_krw_value_to_usd("006930.KS", 3_900_000.0, 1300.0) == pytest.approx(3000.0)
    assert convert_krw_value_to_usd("MSFT", 1_300_000.0, 1300.0) == pytest.approx(1_300_000.0)

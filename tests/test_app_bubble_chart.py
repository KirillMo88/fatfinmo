import pandas as pd
import pytest

from app import (
    CHARTS_BAR_HEIGHT,
    CHARTS_CATEGORY_LABEL_LIMIT,
    CHARTS_RSI_HEIGHT,
    PERFORMANCE_BUBBLE_GROUP_BY_TICKER,
    _prepare_performance_gainers_losers_frame,
    _prepare_performance_sma200w_bubble_frame,
    bulk_current_prices,
)


def test_categorical_charts_keep_enough_vertical_plot_area():
    assert CHARTS_BAR_HEIGHT >= 360
    assert CHARTS_RSI_HEIGHT >= 390
    assert CHARTS_CATEGORY_LABEL_LIMIT <= 130


def test_bubble_frame_filters_only_selected_negative_performance():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "Perf_1W_%": [5.0, -1.0, 3.0],
            "SMA200W_Distance_Percentile": [20.0, 80.0, 40.0],
            "Avg_Forward_Return_6M_%": [10.0, 15.0, -12.0],
        }
    )

    result = _prepare_performance_sma200w_bubble_frame(df, "Perf_1W_%")

    assert result["Ticker"].tolist() == ["AAA", "CCC"]
    assert result["Forward_Return_Sign"].tolist() == ["Positive", "Negative"]


def test_bubble_frame_sizes_by_absolute_forward_return():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Perf_1M_%": [5.0, 6.0],
            "SMA200W_Distance_Percentile": [20.0, 80.0],
            "Avg_Forward_Return_6M_%": [4.0, -14.0],
        }
    )

    result = _prepare_performance_sma200w_bubble_frame(df, "Perf_1M_%")

    assert result["Bubble_Size_Value"].tolist() == [4.0, 14.0]
    assert result.loc[result["Ticker"] == "BBB", "Forward_Return_Sign"].iloc[0] == "Negative"


def test_bubble_group_view_uses_medians_and_excludes_unmapped_tickers():
    df = pd.DataFrame(
        {
            "Ticker": ["GLD", "GDX", "GDXJ", "SPY", "UNMAPPED"],
            "Perf_1M_%": [1.0, 5.0, 9.0, -2.0, 100.0],
            "SMA200W_Distance_Percentile": [10.0, 30.0, 50.0, 90.0, 100.0],
            "Avg_Forward_Return_6M_%": [-5.0, 5.0, 15.0, 20.0, 100.0],
        }
    )

    result = _prepare_performance_sma200w_bubble_frame(df, "Perf_1M_%", view_mode="Group")

    assert result["Ticker"].tolist() == ["Gold"]
    assert result["Selected_Performance"].tolist() == [5.0]
    assert result["SMA200W_Percentile"].tolist() == [30.0]
    assert result["Avg_Forward_Return_6M"].tolist() == [5.0]
    assert result["Group_Size"].tolist() == [3]


def test_bubble_group_mapping_uses_corrected_crypto_altcoins_name():
    assert PERFORMANCE_BUBBLE_GROUP_BY_TICKER["ETH-USD"] == "Crypto Altcoins"
    assert "Crypro Altcoins" not in set(PERFORMANCE_BUBBLE_GROUP_BY_TICKER.values())


def test_bubble_frame_rejects_unknown_view_mode():
    with pytest.raises(ValueError, match="Unsupported Performance vs SMA200W view mode"):
        _prepare_performance_sma200w_bubble_frame(pd.DataFrame(), "Perf_1M_%", view_mode="Unknown")


def test_gainers_losers_uses_ten_from_each_side_of_selected_period():
    values = [float(value) for value in range(-12, 0)] + [float(value) for value in range(1, 13)]
    df = pd.DataFrame(
        {
            "Ticker": [f"T{i}" for i in range(len(values))],
            "Perf_1W_%": values,
            "Perf_1M_%": list(reversed(values)),
            "SMA200W_Distance_Percentile": [50.0] * len(values),
            "Avg_Forward_Return_6M_%": [2.0] * len(values),
        }
    )

    result = _prepare_performance_gainers_losers_frame(df, "Perf_1W_%")

    assert len(result) == 20
    assert result["Selected_Performance"].tolist() == [
        12.0,
        11.0,
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -12.0,
    ]
    assert result["Performance_Direction"].value_counts().to_dict() == {"Gainer": 10, "Loser": 10}


def test_bulk_current_prices_uses_one_download_for_all_tickers(monkeypatch):
    dates = pd.to_datetime(["2026-10-15", "2026-10-16"])
    columns = pd.MultiIndex.from_product(
        [["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    raw = pd.DataFrame(index=dates, columns=columns, dtype="float64")
    raw[("AAA", "Close")] = [10.0, 11.0]
    raw[("BBB", "Close")] = [20.0, 22.0]
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw

    monkeypatch.setattr("app.yf.download", fake_download)

    result = bulk_current_prices(["AAA", "BBB", "AAA"], refresh_bucket=123)

    assert result == {"AAA": 11.0, "BBB": 22.0}
    assert len(calls) == 1
    assert calls[0][0] == ["AAA", "BBB"]
    assert calls[0][1]["threads"] is True

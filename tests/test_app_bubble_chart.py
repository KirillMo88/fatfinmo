import pandas as pd

from app import _prepare_performance_sma200w_bubble_frame


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

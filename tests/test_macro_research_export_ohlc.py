import pandas as pd

from macro_research_export import weekly_ohlc


def test_weekly_ohlc_high_low_contain_open_and_close():
    daily = pd.DataFrame(
        {
            "Open": [101.0, 103.0],
            "High": [99.0, 102.0],
            "Low": [102.0, 100.0],
            "Close": [103.0, 98.0],
            "Volume": [10.0, 20.0],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )

    weekly = weekly_ohlc(daily)

    assert len(weekly) == 1
    bar = weekly.iloc[0]
    assert bar["High"] >= max(bar["Open"], bar["Close"])
    assert bar["Low"] <= min(bar["Open"], bar["Close"])
    assert bar["Volume"] == 30.0

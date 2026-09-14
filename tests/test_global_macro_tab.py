import pandas as pd

from global_macro_tab import MacroSeriesSpec, _macro_change_color_row, _series_row


def test_series_row_formats_percent_point_and_state_horizons():
    dates = pd.date_range("2024-01-05", periods=170, freq="W-FRI")
    percent_series = pd.Series(range(100, 270), index=dates, dtype="float64")
    bps_series = pd.Series([4.0 + idx * 0.01 for idx in range(170)], index=dates)
    state_series = pd.Series(["A"] * 166 + ["B"] * 4, index=dates)
    now = pd.Timestamp("2027-04-09")

    percent_row = _series_row(
        MacroSeriesSpec("Markets", "Test Price", "TEST", "percent", "number", "weekly", "price", percent_series),
        now,
    )
    bps_row = _series_row(
        MacroSeriesSpec("Rates", "Test Yield", "TEST", "bps", "percent", "weekly", "percentage points", bps_series),
        now,
    )
    state_row = _series_row(
        MacroSeriesSpec("Global Liquidity", "Test State", "TEST", "state", "number", "weekly", "state", state_series),
        now,
    )

    assert list(percent_row.keys())[2:9] == ["Current", "1W", "1M", "3M", "6M", "12M", "36M"]
    assert percent_row["1W"].endswith("%")
    assert bps_row["1W"] == "+0.01%"
    assert state_row["Current"] == "B"
    assert state_row["1M"] == "A"


def test_macro_change_color_row_uses_directional_rules():
    positive_row = pd.Series({"Instrument": "S&P 500 ETF", "1W": "+3.0%", "1M": "-1.0%"})
    inverse_row = pd.Series({"Instrument": "U.S. Dollar Index", "1W": "+3.0%", "1M": "-1.0%"})

    positive_styles = _macro_change_color_row(positive_row, "Markets")
    inverse_styles = _macro_change_color_row(inverse_row, "FX / USD")

    assert "#4ade80" in positive_styles[1]
    assert "#fdba74" in positive_styles[2]
    assert "#ef4444" in inverse_styles[1]
    assert "#86efac" in inverse_styles[2]

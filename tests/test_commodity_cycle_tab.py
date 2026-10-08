import numpy as np
import pandas as pd

from commodity_cycle_tab import (
    DISPLAY_COLUMN_NAMES,
    PRIMARY_COMMODITY_COLUMNS,
    build_top_analytics,
    _commodity_confirmation_frames,
    _commodity_numeric_formatters,
    _curve_scatter_marker,
    _price_momentum_series,
    _raw_curve_state_style,
    _return_gradient_styles,
)


def test_top_analytics_formats_five_horizontal_market_metrics_from_shared_sources():
    dates = pd.date_range("2020-01-01", periods=2400, freq="D")
    vix = pd.DataFrame({"Date": dates, "VIX": np.linspace(15.0, 30.0, len(dates))})
    move = pd.DataFrame({"Date": dates, "MOVE": np.linspace(70.0, 110.0, len(dates))})
    funding = pd.DataFrame({"Date": dates, "FundingState": ["NORMAL"] * (len(dates) - 1) + ["TREASURY VOLATILITY"]})
    liquidity_dates = pd.date_range("2021-01-01", periods=210, freq="W-FRI")
    liquidity = pd.DataFrame({"Date": liquidity_dates, "global_liquidity_score": np.linspace(30.0, 60.0, len(liquidity_dates))})

    cards = build_top_analytics("HIGH RISK", liquidity, vix, move, funding)

    assert [label for label, _, _ in cards] == [
        "Current Risk", "Global Liquidity Score", "VIX", "MOVE", "Funding Stress"
    ]
    assert cards[0][1] == "HIGH RISK"
    assert cards[1][1] == "60.0"
    assert "ROC 1M" in cards[1][2] and "ROC 3M" in cards[1][2]
    assert "5Y percentile" in cards[2][2] and cards[2][1] == "30.00"
    assert "4W change +" in cards[3][2] and cards[3][1] == "110.00"
    assert cards[4][1] == "TREASURY VOLATILITY"


def test_primary_commodity_table_has_requested_columns_and_auxiliary_keeps_rest():
    frame = pd.DataFrame([{
        **{column: None for column in PRIMARY_COMMODITY_COLUMNS},
        "Leg 1": "CLX26.NYM",
        "CFTC Status": "CURRENT",
    }])

    primary, auxiliary = _commodity_confirmation_frames(frame)

    assert primary.columns.tolist() == [DISPLAY_COLUMN_NAMES.get(column, column) for column in PRIMARY_COMMODITY_COLUMNS]
    assert auxiliary.columns.tolist() == ["Sector", "Commodity", "Leg 1", "CFTC Status"]


def test_percentile_labels_identify_cot_and_spread_sources():
    frame = pd.DataFrame([{
        "Sector": "Energy", "Commodity": "WTI", "5Y Percentile": 80.0,
        "3Y Percentile": 70.0, "Seasonal Percentile 5Y": 60.0, "Seasonal Percentile 10Y": 50.0,
    }])

    primary, auxiliary = _commodity_confirmation_frames(frame)

    assert "COT 5Y Percentile" in primary
    assert "Spread 5Y Seasonal Percentile" in primary
    assert "COT 3Y Percentile" in auxiliary
    assert "Spread 10Y Seasonal Percentile" in auxiliary


def test_commodity_table_formats_all_numeric_values_to_two_decimals():
    frame = pd.DataFrame({
        "Price": [90.421],
        "Return 3M": [0.30123],
        "Curve Spread": [0.02567],
        "5Y Percentile": [20.3846],
    })

    formats = _commodity_numeric_formatters(frame)

    assert formats["Price"].format(frame.at[0, "Price"]) == "90.42"
    assert formats["Return 3M"].format(frame.at[0, "Return 3M"]) == "30.12%"
    assert formats["Curve Spread"].format(frame.at[0, "Curve Spread"]) == "2.57%"
    assert formats["5Y Percentile"].format(frame.at[0, "5Y Percentile"]) == "20.38"


def test_return_gradient_and_curve_state_colors_follow_requested_direction():
    styles = _return_gradient_styles(pd.Series([-0.2, 0.0, 0.3, np.nan]))

    assert "rgb(127, 29, 29)" in styles[0]
    assert "rgb(20, 83, 45)" in styles[2]
    assert styles[3] == ""
    assert "#14532d" in _raw_curve_state_style("Contango")
    assert "#7f1d1d" in _raw_curve_state_style("Backwardation")


def test_price_positioning_markers_are_large_and_follow_absolute_curve_state():
    backwardation = _curve_scatter_marker("Backwardation")
    contango = _curve_scatter_marker("Contango")
    unavailable = _curve_scatter_marker("N/A")

    assert backwardation["size"] == 14
    assert backwardation["color"] == "#ef4444"
    assert contango["color"] == "#22c55e"
    assert unavailable["color"] == "#94a3b8"


def test_asset_price_momentum_uses_the_selected_weekly_horizon():
    dates = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
    prices = pd.Series(np.arange(100.0, 160.0), index=dates)

    one_month = _price_momentum_series(prices, "1M")
    three_month = _price_momentum_series(prices, "3M")
    six_month = _price_momentum_series(prices, "6M")
    twelve_month = _price_momentum_series(prices, "12M")

    assert one_month.iloc[4] == 4.0
    assert three_month.iloc[13] == 13.0
    assert six_month.iloc[26] == 26.0
    assert twelve_month.iloc[52] == 52.0
    assert _price_momentum_series(prices, "2M").empty

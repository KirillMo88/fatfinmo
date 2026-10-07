import numpy as np
import pandas as pd

from commodity_cycle_tab import (
    PRIMARY_COMMODITY_COLUMNS,
    _commodity_confirmation_frames,
    _commodity_numeric_formatters,
    _raw_curve_state_style,
    _return_gradient_styles,
)


def test_primary_commodity_table_has_requested_columns_and_auxiliary_keeps_rest():
    frame = pd.DataFrame([{
        **{column: None for column in PRIMARY_COMMODITY_COLUMNS},
        "Leg 1": "CLX26.NYM",
        "CFTC Status": "CURRENT",
    }])

    primary, auxiliary = _commodity_confirmation_frames(frame)

    assert primary.columns.tolist() == list(PRIMARY_COMMODITY_COLUMNS)
    assert auxiliary.columns.tolist() == ["Sector", "Commodity", "Leg 1", "CFTC Status"]


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

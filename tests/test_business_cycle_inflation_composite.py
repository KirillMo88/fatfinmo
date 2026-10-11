import numpy as np
import pandas as pd

from business_cycle import add_inflation_layer
from business_cycle_tab import build_inflation_price_income_fig


def _inflation_history() -> pd.DataFrame:
    periods = 110
    values = np.arange(periods, dtype=float)
    return pd.DataFrame({
        "T5YIE": 2.0 + values * 0.001,
        "T10YIE": 2.1 + values * 0.001,
        "EXPINF1YR": 2.2 + values * 0.001,
        "EXPINF5YR": 2.0 + values * 0.001,
        "MICH": 3.0 + values * 0.001,
        "T5YIFR": 2.2 + values * 0.001,
        "CPIAUCSL": 250.0 + values,
        "PI": 18_000.0 + values * 20.0,
        "CPILFESL": 255.0 + values,
        "PCEPILFE": 110.0 + values * 0.4,
        "PPIACO": 190.0 + values * 0.8,
    })


def test_inflation_layer_builds_weighted_pi_ppi_contributions():
    history = _inflation_history()
    add_inflation_layer(history)
    latest = history.iloc[-1]

    assert latest["PI_WeightedContribution"] == latest["PI_YoY"] * 0.70
    assert latest["PPIACO_WeightedContribution"] == latest["PPIACO_YoY"] * 0.30
    assert latest["PI_PPI_Weighted12M"] == (
        latest["PI_WeightedContribution"] + latest["PPIACO_WeightedContribution"]
    )


def test_inflation_price_income_figure_has_three_lines_and_two_bar_components():
    history = _inflation_history()
    add_inflation_layer(history)
    history["date"] = pd.date_range("2024-01-05", periods=len(history), freq="W-FRI")
    fig = build_inflation_price_income_fig(history)

    assert [trace.name for trace in fig.data] == [
        "CPIAUCSL 12M change",
        "PI 12M change",
        "PPIACO 12M change",
        "PI 12M change × 70%",
        "PPIACO 12M change × 30%",
    ]
    assert fig.data[0].line.dash in (None, "solid")
    assert fig.data[1].line.dash == "dot"
    assert fig.data[2].line.dash == "dot"
    assert fig.data[3].type == "bar"
    assert fig.data[4].type == "bar"
    assert fig.data[3].marker.color != fig.data[4].marker.color
    assert fig.layout.barmode == "relative"

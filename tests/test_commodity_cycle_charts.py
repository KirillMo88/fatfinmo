import pandas as pd
import plotly.graph_objects as go

from commodity_cycle_tab import (
    CORE_COLORS,
    INVENTORY_SALES_SERIES,
    STATE_BAND_OPACITY,
    _add_state_bands,
    _add_state_legend,
    _build_inventory_sales_figure,
)


def test_state_bands_are_bright_and_have_no_inline_text_annotations():
    frame = pd.DataFrame(
        {"Core State": ["Early Easing", "Early Easing", "Mature"]},
        index=pd.date_range("2026-01-01", periods=3, freq="MS"),
    )
    figure = go.Figure()

    _add_state_bands(figure, frame, "Core State", CORE_COLORS)

    assert len(figure.layout.shapes) == 2
    assert all(shape.opacity == STATE_BAND_OPACITY for shape in figure.layout.shapes)
    assert len(figure.layout.annotations or []) == 0


def test_state_legend_contains_only_present_states_as_bottom_ready_swatches():
    frame = pd.DataFrame({"Core State": ["Mature", "Early Easing", "Mature"]})
    figure = go.Figure()

    _add_state_legend(figure, frame, "Core State", CORE_COLORS)

    assert [trace.name for trace in figure.data] == ["Mature", "Early Easing"]
    assert all(trace.marker.symbol == "square" for trace in figure.data)


def test_inventory_sales_figure_includes_all_eight_categories_and_obeys_range():
    dates = pd.date_range("2018-01-01", periods=108, freq="MS")
    frame = pd.DataFrame(
        {name: [1.0 + index / 100 for index in range(len(dates))] for name in INVENTORY_SALES_SERIES},
        index=dates,
    )

    figure = _build_inventory_sales_figure(frame, "5Y")

    assert len(INVENTORY_SALES_SERIES) == 8
    assert [trace.name for trace in figure.data] == list(INVENTORY_SALES_SERIES)
    assert min(trace.x[0] for trace in figure.data) >= dates[-1] - pd.DateOffset(years=5)

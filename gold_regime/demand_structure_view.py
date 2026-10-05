from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .aisc_view import aisc_range_window
from .demand_structure import DEMAND_CATEGORIES, build_demand_structure_frame


DEMAND_COLORS = {
    "Jewellery": "#22d3ee",
    "Technology": "#c084fc",
    "Investment": "#fbbf24",
    "Central Banks": "#fb7185",
    "OTC and other": "#94a3b8",
}


def filter_demand_range(
    frame: pd.DataFrame,
    selected_range: str,
    anchor_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    window = aisc_range_window(frame, selected_range, anchor_date)
    if window is None:
        return frame.copy()
    start, end = window
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame.loc[dates.ge(start) & dates.le(end)].copy()


def render_demand_structure(
    selected_range: str = "5Y",
    range_end: pd.Timestamp | None = None,
) -> None:
    st.markdown("#### Demand Structure")
    source = build_demand_structure_frame()
    data = filter_demand_range(source, selected_range, range_end)
    if data.empty:
        st.info("Demand Structure data is unavailable for the selected range.")
        return
    st.plotly_chart(
        build_demand_structure_fig(data, aisc_range_window(source, selected_range, range_end)),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def build_demand_structure_fig(
    frame: pd.DataFrame,
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> go.Figure:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["demand_share"] = pd.to_numeric(data["demand_share"], errors="coerce")
    data["demand_12m_change_tn"] = pd.to_numeric(data["demand_12m_change_tn"], errors="coerce")
    data = data.dropna(subset=["date"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.31, 0.31],
        vertical_spacing=0.07,
        subplot_titles=("Demand Share", "Demand 12M Change (tn)", "Demand 3M Change (tn)"),
    )
    for category in DEMAND_CATEGORIES:
        category_data = data[data["category"].eq(category)]
        if category_data.empty:
            continue
        color = DEMAND_COLORS[category]
        fig.add_trace(
            go.Bar(
                x=category_data["date"],
                y=category_data["demand_share"] * 100.0,
                name=category,
                marker_color=color,
                customdata=category_data["quarter"],
                hovertemplate="Quarter: %{customdata}<br>" + category + ": %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        change_data = category_data.dropna(subset=["demand_12m_change_tn"])
        if not change_data.empty:
            fig.add_trace(
                go.Bar(
                    x=change_data["date"],
                    y=change_data["demand_12m_change_tn"],
                    name=category,
                    marker_color=color,
                    customdata=change_data["quarter"],
                    showlegend=False,
                    hovertemplate="Quarter: %{customdata}<br>" + category + ": %{y:+.1f} tn<extra></extra>",
                ),
                row=2,
                col=1,
            )
        change_data = category_data.dropna(subset=["demand_3m_change_tn"])
        if not change_data.empty:
            fig.add_trace(
                go.Bar(
                    x=change_data["date"],
                    y=change_data["demand_3m_change_tn"],
                    name=category,
                    marker_color=color,
                    customdata=change_data["quarter"],
                    showlegend=False,
                    hovertemplate="Quarter: %{customdata}<br>" + category + ": %{y:+.1f} tn<extra></extra>",
                ),
                row=3,
                col=1,
            )

    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    fig.update_yaxes(title_text="Share (%)", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="Change (tn)", row=2, col=1)
    fig.update_yaxes(title_text="Change (tn)", row=3, col=1)
    fig.update_xaxes(title_text="Quarter", row=3, col=1)
    fig.update_layout(
        barmode="relative",
        height=820,
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#11161f",
        margin={"l": 70, "r": 25, "t": 60, "b": 45},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return fig

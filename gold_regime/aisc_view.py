from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .aisc import AISC_ZONE_THRESHOLDS, NORMAL_AISC_MULTIPLE, aisc_valuation_state
from .models import GoldAISCValuationSnapshot


AISC_ZONE_COLORS = (
    (0.0, 1.25, "#166534", "Very compressed producer economics"),
    (1.25, 1.45, "#86efac", "Below-normal margin environment"),
    (1.45, 1.80, "#2563eb", "Normal historical range"),
    (1.80, 2.10, "#f97316", "Strong producer-margin environment"),
    (2.10, 2.40, "#fca5a5", "Historically elevated"),
    (2.40, 5.0, "#991b1b", "Extreme / unusual"),
)


def filter_aisc_range(frame: pd.DataFrame, selected_range: str) -> pd.DataFrame:
    if frame.empty or selected_range == "MAX":
        return frame.copy()
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(selected_range)
    if years is None:
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    end = dates.max()
    if pd.isna(end):
        return frame.iloc[0:0].copy()
    return frame.loc[dates.ge(end - pd.DateOffset(years=years))].copy()


def render_gold_aisc_valuation(
    snapshot: GoldAISCValuationSnapshot | None,
    selected_range: str = "5Y",
) -> None:
    st.markdown("#### Gold / AISC Valuation")
    if snapshot is None or snapshot.history.empty:
        st.info("Gold / AISC valuation data is unavailable.")
        return

    data = filter_aisc_range(snapshot.history, selected_range)
    data = data.dropna(subset=["gold_aisc_ratio", "aisc"])
    if data.empty:
        st.info("Gold / AISC valuation data is unavailable for the selected range.")
        return

    current = snapshot.current or data.iloc[-1].to_dict()
    ratio = _number(current.get("gold_aisc_ratio"))
    gold = _number(current.get("gold_close"))
    aisc = _number(current.get("aisc"))
    source = str(current.get("aisc_source") or "N/A")
    quarter = str(current.get("quarter") or "N/A")
    state = aisc_valuation_state(ratio)

    cards = st.columns(6)
    with cards[0]:
        st.metric("Gold / AISC", "N/A" if not np.isfinite(ratio) else f"{ratio:.2f}×")
    with cards[1]:
        st.metric("Interpretation", state)
    with cards[2]:
        st.metric("Gold Price", "N/A" if not np.isfinite(gold) else f"${gold:,.0f}")
    with cards[3]:
        st.metric("Current AISC", "N/A" if not np.isfinite(aisc) else f"${aisc:,.0f}")
    with cards[4]:
        st.metric("AISC Source", source.title())
    with cards[5]:
        st.metric("Quarter", quarter)

    st.plotly_chart(
        build_gold_aisc_valuation_fig(data),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def build_gold_aisc_valuation_fig(frame: pd.DataFrame) -> go.Figure:
    data = frame.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["gold_close"] = pd.to_numeric(data["gold_close"], errors="coerce")
    data["aisc"] = pd.to_numeric(data["aisc"], errors="coerce")
    data["gold_aisc_ratio"] = pd.to_numeric(data["gold_aisc_ratio"], errors="coerce")
    data["premium_discount_pct"] = pd.to_numeric(data["premium_discount_pct"], errors="coerce")
    data = data.dropna(subset=["date", "aisc", "gold_aisc_ratio"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.20, 0.30],
        vertical_spacing=0.045,
        subplot_titles=(
            "Gold Price / Global AISC",
            f"Gold Premium / Discount vs {NORMAL_AISC_MULTIPLE:g}× AISC",
            "Global AISC",
        ),
    )

    regime = data.get("regime", pd.Series("N/A", index=data.index)).fillna("N/A").astype(str)
    source = data["aisc_source"].fillna("N/A").astype(str)
    custom = np.column_stack(
        [
            data["gold_close"].to_numpy(),
            data["aisc"].to_numpy(),
            regime.to_numpy(),
            source.to_numpy(),
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["gold_aisc_ratio"],
            mode="lines",
            name="Gold / AISC",
            line={"color": "#f8fafc", "width": 2.0},
            customdata=custom,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Gold Close: $%{customdata[0]:,.0f}<br>"
                "AISC: $%{customdata[1]:,.0f}<br>Gold / AISC: %{y:.2f}×<br>"
                "Regime: %{customdata[2]}<br>AISC Source: %{customdata[3]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    for lower, upper, color, _ in AISC_ZONE_COLORS:
        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=color,
            opacity=0.16,
            line_width=0,
            row=1,
            col=1,
        )
    for level in AISC_ZONE_THRESHOLDS:
        fig.add_hline(y=level, line={"color": "#cbd5e1", "width": 1, "dash": "dot"}, row=1, col=1)

    premium = data["premium_discount_pct"]
    premium_colors = np.where(premium.ge(0.0), "#ef4444", "#22c55e")
    premium_custom = np.column_stack(
        [data["gold_close"].to_numpy(), (data["aisc"] * NORMAL_AISC_MULTIPLE).to_numpy()]
    )
    fig.add_trace(
        go.Bar(
            x=data["date"],
            y=premium,
            name="Premium / Discount",
            marker={"color": premium_colors},
            customdata=premium_custom,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Gold Close: $%{customdata[0]:,.0f}<br>"
                f"{NORMAL_AISC_MULTIPLE:g}× AISC: $%{{customdata[1]:,.0f}}<br>Premium / Discount: %{{y:+.1f}}%<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": "#f8fafc", "width": 1.2, "dash": "dot"}, row=2, col=1)

    for source_name, dash in (("ACTUAL", "solid"), ("ESTIMATED", "dash")):
        segment = data[source.eq(source_name)]
        if segment.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=segment["date"],
                y=segment["aisc"],
                mode="lines",
                name=f"{source_name.title()} AISC",
                line={"color": "#38bdf8" if source_name == "ACTUAL" else "#f59e0b", "width": 2.2, "dash": dash, "shape": "hv"},
                customdata=np.column_stack([segment["quarter"].astype(str), segment["aisc_source"].astype(str)]),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Quarter: %{customdata[0]}<br>AISC: $%{y:,.0f}<br>%{customdata[1]}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Gold / AISC (×)", row=1, col=1)
    fig.update_yaxes(title_text="Premium / Discount (%)", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="AISC ($/oz)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#11161f",
        margin={"l": 70, "r": 25, "t": 60, "b": 45},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return fig


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan

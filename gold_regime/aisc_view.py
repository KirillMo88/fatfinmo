from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .aisc import AISC_ZONE_THRESHOLDS, aisc_valuation_state, median_gold_aisc_ratio
from .models import GoldAISCValuationSnapshot


AISC_ZONE_COLORS = (
    (0.0, 1.25, "#16a34a", "Very compressed producer economics"),
    (1.25, 1.45, "#86efac", "Below-normal margin environment"),
    (1.45, 1.80, "#2563eb", "Normal historical range"),
    (1.80, 2.10, "#f97316", "Strong producer-margin environment"),
    (2.10, 2.40, "#fb7185", "Historically elevated"),
    (2.40, 5.0, "#dc2626", "Extreme / unusual"),
)

AISC_SOURCE_LABELS = {
    "RECONSTRUCTED": "Reconstructed",
    "METALS_FOCUS_RETROSPECTIVE": "Metals Focus retrospective",
    "METALS_FOCUS": "Metals Focus",
    "ACTUAL": "Actual",
    "ESTIMATED": "Estimated",
}


def aisc_range_window(
    frame: pd.DataFrame,
    selected_range: str,
    anchor_date: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if frame.empty or selected_range == "MAX":
        return None
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(selected_range)
    if years is None:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    end = pd.to_datetime(anchor_date, errors="coerce") if anchor_date is not None else dates.max()
    if pd.isna(end):
        return None
    return end - pd.DateOffset(years=years), end


def filter_aisc_range(
    frame: pd.DataFrame,
    selected_range: str,
    anchor_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    window = aisc_range_window(frame, selected_range, anchor_date)
    if window is None:
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    start, end = window
    return frame.loc[dates.ge(start) & dates.le(end)].copy()


def render_gold_aisc_valuation(
    snapshot: GoldAISCValuationSnapshot | None,
    selected_range: str = "5Y",
    range_end: pd.Timestamp | None = None,
) -> None:
    st.markdown("#### Gold / AISC Valuation")
    if snapshot is None or snapshot.history.empty:
        st.info("Gold / AISC valuation data is unavailable.")
        return

    data = filter_aisc_range(snapshot.history, selected_range, range_end)
    data = data.dropna(subset=["gold_aisc_ratio", "aisc"])
    if data.empty:
        st.info("Gold / AISC valuation data is unavailable for the selected range.")
        return

    current = snapshot.current or data.iloc[-1].to_dict()
    ratio = _number(current.get("gold_aisc_ratio"))
    median_ratio = median_gold_aisc_ratio(snapshot.history)
    gold = _number(current.get("gold_close"))
    aisc = _number(current.get("aisc"))
    source = str(current.get("aisc_source") or "N/A")
    quarter = str(current.get("quarter") or "N/A")
    state = aisc_valuation_state(ratio)

    cards = st.columns(7)
    with cards[0]:
        st.metric("Gold / AISC", "N/A" if not np.isfinite(ratio) else f"{ratio:.2f}×")
    with cards[1]:
        st.metric("Median Gold / AISC", "N/A" if not np.isfinite(median_ratio) else f"{median_ratio:.2f}×")
    with cards[2]:
        st.metric("Interpretation", state)
    with cards[3]:
        st.metric("Gold Price", "N/A" if not np.isfinite(gold) else f"${gold:,.0f}")
    with cards[4]:
        st.metric("Current AISC", "N/A" if not np.isfinite(aisc) else f"${aisc:,.0f}")
    with cards[5]:
        st.metric("AISC Source", AISC_SOURCE_LABELS.get(source.upper(), source.title()))
    with cards[6]:
        st.metric("Quarter", quarter)

    st.plotly_chart(
        build_gold_aisc_valuation_fig(
            data,
            aisc_range_window(snapshot.history, selected_range, range_end),
            median_multiple=median_ratio,
        ),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def build_gold_aisc_valuation_fig(
    frame: pd.DataFrame,
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    median_multiple: float | None = None,
) -> go.Figure:
    data = frame.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["gold_close"] = pd.to_numeric(data["gold_close"], errors="coerce")
    data["aisc"] = pd.to_numeric(data["aisc"], errors="coerce")
    data["gold_aisc_ratio"] = pd.to_numeric(data["gold_aisc_ratio"], errors="coerce")
    data["premium_discount_pct"] = pd.to_numeric(data["premium_discount_pct"], errors="coerce")
    data = data.dropna(subset=["date", "aisc", "gold_aisc_ratio"])
    if median_multiple is None:
        median_multiple = median_gold_aisc_ratio(data)
    multiple_label = f"{median_multiple:.2f}×" if np.isfinite(median_multiple) else "Median multiple"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.20, 0.30],
        vertical_spacing=0.045,
        subplot_titles=(
            "Gold Price / Global AISC",
            f"Gold Premium / Discount vs {multiple_label} AISC"
            if np.isfinite(median_multiple)
            else "Gold Premium / Discount vs median × AISC",
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
            line={"color": "#ffffff", "width": 2.8},
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
            opacity=0.30,
            line_width=0,
            row=1,
            col=1,
        )
    for level in AISC_ZONE_THRESHOLDS:
        fig.add_hline(y=level, line={"color": "#cbd5e1", "width": 1, "dash": "dot"}, row=1, col=1)

    premium = data["premium_discount_pct"]
    premium_colors = np.where(premium.ge(0.0), "#ff3b30", "#00e676")
    premium_custom = np.column_stack([data["gold_close"].to_numpy(), data["normal_gold_value"].to_numpy()])
    fig.add_trace(
        go.Bar(
            x=data["date"],
            y=premium,
            name="Premium / Discount",
            marker={"color": premium_colors, "opacity": 0.95, "line": {"color": premium_colors, "width": 0.5}},
            customdata=premium_custom,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Gold Close: $%{customdata[0]:,.0f}<br>"
                f"{multiple_label} AISC: $%{{customdata[1]:,.0f}}<br>Premium / Discount: %{{y:+.1f}}%<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": "#f8fafc", "width": 1.2, "dash": "dot"}, row=2, col=1)

    quarterly = data.sort_values("date").groupby("quarter", as_index=False, sort=True).tail(1)
    quarterly_source = quarterly["aisc_source"].fillna("N/A").astype(str)
    bar_width_ms = 70 * 24 * 60 * 60 * 1000
    for source_name, selector, color in (
        ("Historical", quarterly_source.ne("ESTIMATED"), "#38bdf8"),
        ("Estimated", quarterly_source.eq("ESTIMATED"), "#f59e0b"),
    ):
        segment = quarterly.loc[selector]
        if segment.empty:
            continue
        source_labels = segment["aisc_source"].fillna("N/A").astype(str).map(
            lambda value: AISC_SOURCE_LABELS.get(value.upper(), value.title())
        )
        fig.add_trace(
            go.Bar(
                x=segment["date"],
                y=segment["aisc"],
                width=bar_width_ms,
                name=f"{source_name} AISC",
                marker={"color": color, "line": {"color": color, "width": 0.5}},
                customdata=np.column_stack([segment["quarter"].astype(str), source_labels]),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Quarter: %{customdata[0]}<br>AISC: $%{y:,.0f}<br>Source: %{customdata[1]}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Gold / AISC (×)", row=1, col=1)
    fig.update_yaxes(title_text="Premium / Discount (%)", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="AISC ($/oz)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#11161f",
        margin={"l": 70, "r": 25, "t": 60, "b": 45},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
        barmode="overlay",
    )
    return fig


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan

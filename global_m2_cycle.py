"""Monthly cycle diagnostics for the Global M2 series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator

from market_cycle import cycle_direction, fft_bandpass_cycle, standard_zscore


GLOBAL_M2_CYCLE_COLUMNS = [
    "Date",
    "GlobalM2",
    "GlobalM2Log",
    "M2SMA50M",
    "M2StructuralExtensionPct",
    "M2ROC12M",
    "M2RSI14M",
    "PrimaryCycleComposite",
    "PrimaryMarketCycle",
    "PrimaryCycleDirection",
    "PrimaryCycleState",
]


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=GLOBAL_M2_CYCLE_COLUMNS)


def _primary_cycle_state(value: Any, direction: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "DATA INCOMPLETE"
    if not np.isfinite(value):
        return "DATA INCOMPLETE"
    direction = str(direction)
    if value >= 0 and direction == "UPSWING":
        return "ACCELERATING EXPANSION"
    if value >= 0:
        return "DECELERATING EXPANSION"
    if direction == "DOWNSWING":
        return "ACCELERATING CONTRACTION"
    return "RECOVERY / REACCELERATION"


def build_global_m2_cycle_history(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build the Global M2 cycle on its native monthly frequency.

    Structural extension is measured against SMA50M. The primary cycle is a
    standardized composite of 12M ROC and RSI(14M), filtered through the same
    full-history FFT band-pass approach used by Market Cycle, with a 30-54M
    passband. The structural extension remains a separate diagnostic so it is
    not double-counted in the primary composite.
    """
    if monthly is None or monthly.empty or "global_m2_usd_bn" not in monthly.columns:
        return _empty_history()

    frame = monthly.copy()
    date_source = frame["date"] if "date" in frame.columns else frame.get("Date")
    frame["Date"] = pd.to_datetime(date_source, errors="coerce")
    frame["GlobalM2"] = pd.to_numeric(frame["global_m2_usd_bn"], errors="coerce")
    frame = (
        frame.dropna(subset=["Date", "GlobalM2"])
        .loc[lambda item: item["GlobalM2"] > 0]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    if frame.empty:
        return _empty_history()

    frame = frame[["Date", "GlobalM2"]].reset_index(drop=True)
    level = frame["GlobalM2"]
    frame["GlobalM2Log"] = np.log(level)
    frame["M2SMA50M"] = level.rolling(50, min_periods=50).mean()
    frame["M2StructuralExtensionPct"] = level / frame["M2SMA50M"] - 1.0
    frame["M2ROC12M"] = level.pct_change(12, fill_method=None)
    frame["M2RSI14M"] = RSIIndicator(close=level, window=14).rsi()

    components = pd.DataFrame(
        {"roc": frame["M2ROC12M"], "rsi": frame["M2RSI14M"]},
        index=frame.index,
    )
    complete = components.dropna()
    zscores = components.copy()
    for column in components.columns:
        std = complete[column].std(ddof=0)
        zscores[column] = (
            (components[column] - complete[column].mean()) / std
            if np.isfinite(std) and std > 0
            else np.nan
        )
    frame["PrimaryCycleComposite"] = zscores.mean(axis=1, skipna=False)
    frame["PrimaryMarketCycle"] = standard_zscore(
        fft_bandpass_cycle(frame["PrimaryCycleComposite"], min_period=30.0, max_period=54.0)
    )
    frame["PrimaryCycleDirection"] = cycle_direction(frame["PrimaryMarketCycle"])
    frame["PrimaryCycleState"] = [
        _primary_cycle_state(value, direction)
        for value, direction in zip(
            frame["PrimaryMarketCycle"], frame["PrimaryCycleDirection"], strict=False
        )
    ]
    return frame[GLOBAL_M2_CYCLE_COLUMNS]


def build_global_m2_cycle_fig(
    full_history: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> go.Figure:
    """Render only the primary monthly Global M2 cycle layer."""
    fig = go.Figure()
    if full_history is None or full_history.empty:
        return _style_figure(fig, "Global M2 Monthly Cycle", 690)

    full = full_history.copy()
    full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
    full = full.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if start is None:
        start = full["Date"].min()
    if end is None:
        end = full["Date"].max()
    visible = full.loc[full["Date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if visible.empty:
        visible = full.tail(1).copy()

    states = full["PrimaryCycleState"].astype(str)
    state_colors = {
        "ACCELERATING EXPANSION": "#22c55e",
        "DECELERATING EXPANSION": "#facc15",
        "ACCELERATING CONTRACTION": "#ef4444",
        "RECOVERY / REACCELERATION": "#38bdf8",
    }
    state_start = 0
    for idx in range(1, len(full) + 1):
        if idx == len(full) or states.iloc[idx] != states.iloc[state_start]:
            x0 = max(full.loc[state_start, "Date"], pd.Timestamp(start))
            x1 = min(
                full.loc[idx, "Date"] if idx < len(full) else full.loc[idx - 1, "Date"] + pd.offsets.MonthEnd(1),
                pd.Timestamp(end),
            )
            if x1 >= x0 and states.iloc[state_start] in state_colors:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=state_colors[states.iloc[state_start]],
                    opacity=0.10,
                    line_width=0,
                )
            state_start = idx

    dates = visible["Date"]
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pd.to_numeric(visible["PrimaryMarketCycle"], errors="coerce"),
            mode="lines",
            name="Primary Cycle",
            line={"color": "#38bdf8", "width": 2.0},
            customdata=visible[["PrimaryCycleState"]].to_numpy(),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Cycle: %{y:.2f}<br>State: %{customdata[0]}"
                "<extra></extra>"
            ),
        ),
    )
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1})
    fig.update_yaxes(title_text="Normalized")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#94a3b8", spikethickness=1)
    return _style_figure(
        fig,
        "Global M2 Primary Liquidity Cycle<br><sup>12M ROC + RSI14M | 30-54M band-pass</sup>",
        390,
    )


def _style_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 72, "t": 68, "b": 52},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickformat="%b'%y", showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569", ticks="outside")
    return fig


def global_m2_cycle_current(history: pd.DataFrame) -> dict[str, Any]:
    if history is None or history.empty:
        return {}
    return history.iloc[-1].to_dict()

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rates_financial_conditions import REGIMES, RatesSnapshot, read_snapshot, refresh_snapshot, validation_episodes


REGIME_COLORS = {
    REGIMES[0]: "#16a34a",
    REGIMES[1]: "#38bdf8",
    REGIMES[2]: "#f59e0b",
    REGIMES[3]: "#ef4444",
    "DATA INCOMPLETE": "#64748b",
}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
CURVE_26W_EXPLANATIONS = {
    "BULL STEEPENING": (
        "Rates are falling and the spread is widening. Typically, the short end declines more sharply. "
        "This often reflects expectations of Fed rate cuts. It can occur both in a positive disinflation / "
        "soft-landing scenario and in a stress scenario when the market expects aggressive cuts."
    ),
    "BULL FLATTENING": (
        "Rates are falling and the spread is narrowing. Yields decline, but primarily at the long end "
        "of the curve. This is more consistent with falling long-term growth and/or inflation expectations, "
        "without a major shift in Fed expectations."
    ),
    "BEAR STEEPENING": (
        "Rates are rising and the spread is widening. The long end rises faster. This can indicate that "
        "the key pressure is coming less from near-term Fed policy and more from long-term inflation "
        "expectations, fiscal pressure, or a rising term premium."
    ),
    "BEAR FLATTENING": (
        "Rates are rising and the spread is narrowing. The short end rises faster. This is the clearest "
        "form of policy-tightening pressure: the market is repricing the expected Fed path and short-term "
        "interest rates higher."
    ),
}


def _style(fig: go.Figure, height: int = 310) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=45, r=15, t=24, b=50),
        paper_bgcolor="#14181e", plot_bgcolor="#14181e", font=dict(size=11),
        legend=dict(orientation="h", y=-0.25), hovermode="closest",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#334155", zeroline=False)
    return fig


def _strip(fig: go.Figure, frame: pd.DataFrame, column: str, row: int, label: str, colors: dict[str, str]) -> None:
    keys = list(colors)
    states = frame[column].fillna("DATA INCOMPLETE").astype(str)
    values = [keys.index(value) if value in colors else keys.index("DATA INCOMPLETE") for value in states]
    scale = [(edge, color) for index, color in enumerate(colors.values()) for edge in (index / len(keys), (index + 1) / len(keys))]
    fig.add_trace(go.Heatmap(
        x=frame["Date"], y=[label], z=[values], text=[states.tolist()],
        zmin=-0.5, zmax=len(keys) - 0.5, colorscale=scale,
        showscale=False, hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
    ), row=row, col=1)
    fig.update_yaxes(showgrid=False, row=row, col=1)


def _add_regime_shading(fig: go.Figure, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    date = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    states = frame["RatesFinancialConditionsRegime"].reset_index(drop=True)
    start = 0
    for position in range(1, len(frame) + 1):
        if position < len(frame) and states.iloc[position] == states.iloc[start]:
            continue
        color = REGIME_COLORS.get(states.iloc[start], "#64748b")
        right = date.iloc[position] if position < len(frame) else date.iloc[-1] + pd.Timedelta(days=7)
        fig.add_vrect(x0=date.iloc[start], x1=right, fillcolor=color, opacity=0.07, line_width=0, layer="below", row=1, col=1)
        start = position


def build_transmission_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.86, 0.14], vertical_spacing=0.035)
    _add_regime_shading(fig, frame)
    for column, label, color in (
        ("RatesPressureScore", "Rates Pressure", "#38bdf8"),
        ("FinancialConditionsDirectionScore", "FC Direction", "#fb7185"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], mode="lines", name=label, line=dict(color=color, width=2),
                                 customdata=frame["RatesFinancialConditionsRegime"], hovertemplate="%{x|%Y-%m-%d}<br>" + label + " %{y:.2f}<br>%{customdata}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=1, col=1)
    _strip(fig, frame, "RatesFinancialConditionsRegime", 2, "Rates / FC Regime", REGIME_COLORS)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    return _style(fig, 390)


def build_confirmation_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        ("FinancialConditionsDirectionScore", "Core FC", "#fb7185"),
        ("NFCIMomentum", "NFCI", "#38bdf8"),
        ("ANFCIMomentum", "ANFCI", "#facc15"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], mode="lines", name=label, line=dict(color=color, width=1.8)))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    return _style(fig)


def build_component_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        ("CreditDirectionScore", "Credit", "#facc15"),
        ("DXYMomentum", "DXY", "#38bdf8"),
        ("MOVEMomentum", "MOVE", "#fb7185"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], mode="lines", name=label, line=dict(color=color, width=1.8)))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    return _style(fig)


def build_regime_map(frame: pd.DataFrame, weeks: int) -> go.Figure:
    tail = frame.tail(weeks + 1).dropna(subset=["RatesPressureScore", "FinancialConditionsDirectionScore"])
    fig = go.Figure()
    for x0, x1, y0, y1, regime in (
        (-5, 0, -5, 0, REGIMES[0]), (0, 5, -5, 0, REGIMES[1]),
        (-5, 0, 0, 5, REGIMES[2]), (0, 5, 0, 5, REGIMES[3]),
    ):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=REGIME_COLORS[regime], opacity=0.07, line_width=0, layer="below")
    fig.add_hline(y=0, line_color="#64748b")
    fig.add_vline(x=0, line_color="#64748b")
    if not tail.empty:
        fig.add_trace(go.Scatter(x=tail["RatesPressureScore"], y=tail["FinancialConditionsDirectionScore"],
                                 mode="lines+markers", name="Weekly path", line=dict(color="#94a3b8"), marker=dict(size=6),
                                 customdata=np.column_stack([tail["Date"].dt.strftime("%Y-%m-%d"), tail["RatesFinancialConditionsRegime"]]),
                                 hovertemplate="%{customdata[0]}<br>Rates %{x:.2f}<br>FC %{y:.2f}<br>%{customdata[1]}<extra></extra>"))
        latest = tail.iloc[-1]
        fig.add_trace(go.Scatter(x=[latest["RatesPressureScore"]], y=[latest["FinancialConditionsDirectionScore"]],
                                 mode="markers", name="Latest", marker=dict(size=14, color="#ffffff")))
    fig.update_xaxes(title="Rates Pressure", range=[-5, 5])
    fig.update_yaxes(title="Financial Conditions Direction", range=[-5, 5])
    return _style(fig, 360)


def build_curve_chart(frame: pd.DataFrame, weeks: int) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.48, 0.40, 0.12], vertical_spacing=0.04)
    for column, label, color in (("DGS2", "US 2Y", "#38bdf8"), ("DGS10", "US 10Y", "#facc15")):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, mode="lines", line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["US10Y_2Y_Spread"], name="10Y - 2Y", mode="lines", line=dict(color="#e2e8f0")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    colors = {"BULL STEEPENING": "#22c55e", "BULL FLATTENING": "#38bdf8", "BEAR STEEPENING": "#facc15", "BEAR FLATTENING": "#ef4444", "UNCHANGED": "#94a3b8", "DATA INCOMPLETE": "#64748b"}
    _strip(fig, frame, f"YieldCurveRegime_{weeks}W", 3, "Curve", colors)
    fig.update_yaxes(title_text="Yield %", row=1, col=1)
    fig.update_yaxes(title_text="Spread pp", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    return _style(fig, 390)


def _number(value: object) -> str:
    return f"{float(value):+.2f}" if pd.notna(value) else "n/a"


def _status_panel(title: str, lines: list[tuple[str, str]]) -> None:
    st.markdown(f"**{title}**")
    for label, value in lines:
        st.markdown(f"{label}: **{value}**")


def render_rates_financial_conditions_tab(api_key: str | None) -> None:
    st.subheader("Rates & Financial Conditions")
    if st.button("Refresh Rates & Financial Conditions", key="rates_fc_refresh"):
        try:
            with st.spinner("Updating rates and financial conditions..."):
                refresh_snapshot(api_key)
        except Exception as exc:
            st.error(f"Refresh failed; previous snapshot retained: {exc}")
    snapshot: RatesSnapshot = read_snapshot()
    if snapshot.history.empty:
        try:
            with st.spinner("Building Rates & Financial Conditions history..."):
                snapshot = refresh_snapshot(api_key)
        except Exception as exc:
            st.error(f"Required model inputs are unavailable: {exc}")
            return
    history = snapshot.history.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    valid = history.loc[history["RatesFinancialConditionsRegime"].isin(REGIMES)]
    if valid.empty:
        st.warning("No complete Rates / FC regime observations yet.")
        return
    current = valid.iloc[-1]
    st.caption(f"Model {snapshot.status.get('ModelVersion', 'n/a')}  |  Data as of {snapshot.status.get('DataAsOf', 'n/a')}")
    stale = [name for name, state in snapshot.status.get("SourceStatus", {}).items() if "STALE" in state]
    if stale:
        st.warning("Using stale source cache: " + ", ".join(stale))

    cards = st.columns(4, gap="small")
    with cards[0]:
        _status_panel("Rates Pressure", [("Direction", current["RatesDirection"]), ("Score", _number(current["RatesPressureScore"])),
                                         ("US 2Y momentum", _number(current["US2YMomentum"])), ("Real yield momentum", _number(current["RealYieldMomentum"])),
                                         ("Fed Funds", current["FedFundsDirection"]), ("Curve 13W", current["YieldCurveRegime_13W"])])
    with cards[1]:
        _status_panel("Financial Conditions", [("Direction", current["FinancialConditionsDirection"]),
                                               ("Direction score", _number(current["FinancialConditionsDirectionScore"])),
                                               ("Level", _number(current["FinancialConditionsLevel"])), ("Stress", current["FinancialConditionsStressLevel"]),
                                               ("Credit", current["CreditDirection"]), ("DXY", current["DXYDirection"]), ("MOVE", current["MOVEDirection"])])
    with cards[2]:
        _status_panel("Confirmation", [("NFCI", current["NFCIDirection"]), ("ANFCI", current["ANFCIDirection"]),
                                       ("Core FC", current["FinancialConditionsDirection"]),
                                       ("Status", current["FCConfirmationStatus"]), ("Confidence", current["FCConfirmationConfidence"])])
    with cards[3]:
        _status_panel("Rates / FC Regime", [("Regime", current["RatesFinancialConditionsRegime"]),
                                              ("Curve 26W", current["YieldCurveRegime_26W"]), ("FC level", _number(current["FinancialConditionsLevel"]))])
        explanation = CURVE_26W_EXPLANATIONS.get(current["YieldCurveRegime_26W"])
        if explanation:
            st.markdown(f"**Curve 26W interpretation:** {explanation}")

    selected_range = st.radio("Time range", ["3Y", "5Y", "10Y", "MAX"], index=2, horizontal=True, key="rates_fc_time_range")
    years = {"3Y": 3, "5Y": 5, "10Y": 10}.get(selected_range)
    view = history.loc[history["Date"] >= history["Date"].max() - pd.DateOffset(years=years)] if years else history
    st.markdown("### Rates Pressure vs Financial Conditions Direction")
    st.plotly_chart(build_transmission_chart(view), use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Yield Curve Diagnostic")
        curve_weeks = st.radio("Curve horizon", ["13W", "26W"], horizontal=True)
        st.plotly_chart(build_curve_chart(view, int(curve_weeks[:-1])), use_container_width=True, config=CHART_CONFIG)
        st.markdown("### Rates / FC Regime Map")
        map_weeks = st.radio("Map range", ["13W", "26W", "52W"], index=1, horizontal=True)
        st.plotly_chart(build_regime_map(history, int(map_weeks[:-1])), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### Credit / DXY / MOVE")
        st.plotly_chart(build_component_chart(view), use_container_width=True, config=CHART_CONFIG)
        st.markdown("### Core FC vs NFCI / ANFCI")
        st.plotly_chart(build_confirmation_chart(view), use_container_width=True, config=CHART_CONFIG)
        st.caption(
            "1. **NFCI** shows how tight or loose overall financial conditions in the U.S. are relative "
            "to their historical norm.\n"
            "2. **ANFCI** shows how tight or loose financial conditions are **after adjusting for the "
            "current state of the economy and inflation**."
        )

    st.markdown("### Historical Asset Returns by Rates / FC Regime")
    controls = st.columns(3)
    with controls[0]:
        asset = st.selectbox("Asset", ["ALL", "SPY", "QQQ", "GLD", "BTC"])
    with controls[1]:
        horizon = st.selectbox("Horizon", ["ALL", "3M", "6M", "12M"])
    with controls[2]:
        metric = st.selectbox("Metric", ["Average Return", "Median Return", "Hit Rate", "Downside Excursion"])
    returns = snapshot.returns.copy()
    if asset != "ALL":
        returns = returns.loc[returns["Asset"].eq(asset)]
    if horizon != "ALL":
        returns = returns.loc[returns["Horizon"].eq(horizon)]
    returns["Sample"] = np.where(returns["N"] < 20, "LOW SAMPLE SIZE", "")
    selected_cols = ["Average Adverse Excursion", "Median Adverse Excursion", "P10 Adverse Excursion"] if metric == "Downside Excursion" else [metric]
    st.dataframe(returns[["Regime", "Asset", "Horizon", "N", *selected_cols, "Sample"]].round(2), use_container_width=True, hide_index=True)
    with st.expander("All forward-return and downside statistics", expanded=False):
        st.dataframe(returns.round(2), use_container_width=True, hide_index=True)
    chart_asset = st.selectbox("Chart asset", ["SPY", "QQQ", "GLD", "BTC"])
    chart_horizon = st.selectbox("Chart horizon", ["3M", "6M", "12M"])
    chart_data = snapshot.returns.loc[snapshot.returns["Asset"].eq(chart_asset) & snapshot.returns["Horizon"].eq(chart_horizon)]
    chart_metric = "Average Adverse Excursion" if metric == "Downside Excursion" else metric
    fig = go.Figure(go.Bar(x=chart_data["Regime"], y=chart_data[chart_metric], marker_color=[REGIME_COLORS.get(label, "#64748b") for label in chart_data["Regime"]],
                           customdata=chart_data["N"], hovertemplate="%{x}<br>Value %{y:.2f}<br>N %{customdata}<extra></extra>"))
    fig.update_yaxes(title=chart_metric + (" %" if chart_metric != "Hit Rate" else ""))
    st.plotly_chart(_style(fig, 330), use_container_width=True, config=CHART_CONFIG)
    st.markdown("### Validation / Research")
    st.dataframe(validation_episodes(history).round(2), use_container_width=True, hide_index=True)
    st.dataframe(snapshot.transitions, use_container_width=True, hide_index=True)
    with st.expander("Source status and timing", expanded=False):
        st.caption(snapshot.status.get("TimingConvention", ""))
        st.dataframe(snapshot.quality, use_container_width=True, hide_index=True)

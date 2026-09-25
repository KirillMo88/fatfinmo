from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from funding_conditions import FundingSnapshot, STATES, read_snapshot, refresh_snapshot


STATE_COLORS = {
    "NORMAL": "#22c55e",
    "TECHNICAL FUNDING PRESSURE": "#facc15",
    "PERSISTENT FUNDING PRESSURE": "#f97316",
    "TREASURY VOLATILITY": "#38bdf8",
    "SYSTEMIC FUNDING STRESS": "#ef4444",
    "DATA INCOMPLETE": "#64748b",
}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
METRIC_FORMULAS = {
    "Funding Core": "60% Money Market Stress + 40% Reserve Pressure",
    "Money Market Stress": "clip(robust Z(median5(SOFR99 - DFF)), 0, 4)",
    "Reserve Pressure": "65% Reserve Vulnerability + 35% Reserve Drain",
    "Collateral Stress": "(MOVE - mean) / std",
}
def _style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=46, r=16, t=22, b=55),
        paper_bgcolor="#14181e", plot_bgcolor="#14181e", font=dict(size=11),
        legend=dict(orientation="h", y=-0.25), hovermode="closest",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#334155", zeroline=False)
    return fig


def _strip(fig: go.Figure, frame: pd.DataFrame, column: str, row: int, colors: dict[str, str]) -> None:
    labels = list(colors)
    values = frame[column].fillna("DATA INCOMPLETE").astype(str)
    z = [labels.index(value) if value in colors else labels.index("DATA INCOMPLETE") for value in values]
    scale = [(edge, color) for index, color in enumerate(colors.values())
             for edge in (index / len(labels), (index + 1) / len(labels))]
    fig.add_trace(go.Heatmap(
        x=frame["Date"], y=[column], z=[z], text=[values.tolist()], zmin=-0.5,
        zmax=len(labels) - 0.5, colorscale=scale, showscale=False,
        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
    ), row=row, col=1)
    fig.update_yaxes(showgrid=False, row=row, col=1)


def build_core_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.84, 0.16], vertical_spacing=0.04)
    for column, label, color in (
        ("FundingCore", "Funding Core", "#e2e8f0"),
        ("MoneyMarketStress", "Money Market Stress", "#fb7185"),
        ("ReservePressure", "Reserve Pressure", "#38bdf8"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>"), row=1, col=1)
    _strip(fig, frame, "FundingState", 2, STATE_COLORS)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return _style(fig, 390)


def build_money_market_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.52, 0.32, 0.16], vertical_spacing=0.05)
    for column, label, color in (
        ("FundingSpreadRaw", "SOFR99 - DFF", "#94a3b8"),
        ("FundingSpreadSmooth", "5-observation median", "#e2e8f0"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, mode="lines",
                                 line=dict(color=color),
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}} pp<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["MoneyMarketStress"], name="Money Market Stress",
                             mode="lines", line=dict(color="#fb7185", width=2),
                             hovertemplate="%{x|%Y-%m-%d}<br>Money Market Stress: %{y:.2f}<extra></extra>"), row=2, col=1)
    flags = frame["PersistentFundingFlag"].fillna(False).astype(bool).map({True: "Persistent", False: "None"})
    flags = flags.mask(frame["CalendarTechnicalFlag"].fillna(False).astype(bool), "Calendar")
    flags = flags.mask(frame["PersistentFundingFlag"].fillna(False).astype(bool), "Persistent")
    marked = frame.assign(PressureFlag=flags)
    _strip(fig, marked, "PressureFlag", 3,
           {"None": "#334155", "Calendar": "#facc15", "Persistent": "#ef4444", "DATA INCOMPLETE": "#64748b"})
    fig.update_yaxes(title_text="pp", row=1, col=1)
    fig.update_yaxes(title_text="0-4", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    return _style(fig, 460)


def build_reserve_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.36, 0.28, 0.36], vertical_spacing=0.055)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["WRESBAL"] / 1_000_000,
                             name="Reserves", mode="lines", line=dict(color="#e2e8f0"),
                             hovertemplate="%{x|%Y-%m-%d}<br>Reserves: %{y:.2f} tn USD<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["ReserveGDP"] * 100,
                             name="Reserve / GDP", mode="lines", line=dict(color="#38bdf8"),
                             hovertemplate="%{x|%Y-%m-%d}<br>Reserve / GDP: %{y:.2f}%<extra></extra>"), row=2, col=1)
    for column, label, color in (
        ("ReserveVulnerability", "Vulnerability", "#facc15"),
        ("ReserveDrain", "Drain", "#fb7185"),
        ("ReservePressure", "Reserve Pressure", "#e2e8f0"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, mode="lines",
                                 line=dict(color=color),
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>"), row=3, col=1)
    fig.update_yaxes(title_text="USD tn", row=1, col=1)
    fig.update_yaxes(title_text="% GDP", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    return _style(fig, 470)


def build_collateral_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.48, 0.52], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["MOVE"], name="MOVE", mode="lines",
                             line=dict(color="#38bdf8"),
                             hovertemplate="%{x|%Y-%m-%d}<br>MOVE: %{y:.2f}<extra></extra>"), row=1, col=1)
    for column, label, color in (
        ("CollateralStress", "Collateral Stress", "#38bdf8"),
        ("MoneyMarketStress", "Money Market Stress", "#fb7185"),
    ):
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, mode="lines",
                                 line=dict(color=color),
                                 hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="MOVE", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return _style(fig, 390)


def build_state_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    _strip(fig, frame, "FundingState", 1, STATE_COLORS)
    return _style(fig, 140)


def _number(value: object) -> str:
    return f"{float(value):.2f}" if pd.notna(value) else "n/a"


def render_funding_conditions_tab(api_key: str | None) -> None:
    st.subheader("Funding Conditions")
    if st.button("Refresh Funding Conditions", key="funding_conditions_refresh"):
        try:
            with st.spinner("Updating funding conditions..."):
                refresh_snapshot(api_key, refresh=True)
        except Exception as exc:
            st.error(f"Refresh failed; previous snapshot retained: {exc}")
    snapshot: FundingSnapshot = read_snapshot()
    if snapshot.weekly.empty:
        try:
            with st.spinner("Building funding conditions history..."):
                snapshot = refresh_snapshot(api_key)
        except Exception as exc:
            st.error(f"Funding data unavailable: {exc}")
            return
    weekly = snapshot.weekly.copy()
    daily = snapshot.daily.copy()
    weekly["Date"] = pd.to_datetime(weekly["Date"])
    daily["Date"] = pd.to_datetime(daily["Date"])
    latest = daily.iloc[-1]
    st.caption(f"Model {snapshot.status.get('ModelVersion', 'n/a')}  |  Data as of {snapshot.status.get('DataAsOf', 'n/a')}")
    if latest["DataCoverage"] != "FULL":
        st.warning(f"{latest['DataCoverage']}: one or more components are unavailable.")
    if latest["UnconfirmedFundingPressureFlag"]:
        st.warning("Elevated SOFR tail stress is present, but persistence is not yet confirmed.")
    stale = [name for name, state in snapshot.status.get("SourceStatus", {}).items() if "STALE" in state]
    if stale:
        st.warning("Using stale source cache: " + ", ".join(stale))
    cards = st.columns(6, gap="small")
    for column, title, value in zip(cards,
                                    ("Funding State", "Direction", "Funding Core", "Money Market Stress", "Reserve Pressure", "Collateral Stress"),
                                    (latest["FundingState"], latest["FundingDirection"], _number(latest["FundingCore"]),
                                     _number(latest["MoneyMarketStress"]), _number(latest["ReservePressure"]),
                                     _number(latest["CollateralStress"]))):
        with column:
            st.markdown(f"**{title}**")
            st.markdown(str(value))
            if title in METRIC_FORMULAS:
                st.caption(METRIC_FORMULAS[title])
    st.markdown(
        f"Calendar technical: **{'Yes' if latest['CalendarTechnicalFlag'] else 'No'}**  |  "
        f"Persistent funding: **{'Yes' if latest['PersistentFundingFlag'] else 'No'}**  |  "
        f"Unconfirmed funding pressure: **{'Yes' if latest['UnconfirmedFundingPressureFlag'] else 'No'}**  |  "
        f"Reserve vulnerability watch: **{'Yes' if latest['ReserveVulnerabilityWatch'] else 'No'}**  |  "
        f"Primary driver: **{latest['PrimaryDriver']}**"
    )
    selected_range = st.radio("Time range", ["1Y", "3Y", "5Y", "MAX"], index=2,
                              horizontal=True, key="funding_conditions_time_range")
    years = {"1Y": 1, "3Y": 3, "5Y": 5}.get(selected_range)
    start = daily["Date"].max() - pd.DateOffset(years=years) if years else daily["Date"].min()
    weekly_view = weekly.loc[weekly["Date"].ge(start)]
    daily_view = daily.loc[daily["Date"].ge(start)]

    st.markdown("### Funding Core History")
    st.plotly_chart(build_core_chart(weekly_view), use_container_width=True, config=CHART_CONFIG)
    st.markdown("### SOFR Tail Funding Stress")
    st.plotly_chart(build_money_market_chart(daily_view), use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Reserve Conditions")
        st.plotly_chart(build_reserve_chart(weekly_view), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### Treasury / Collateral Stress")
        st.plotly_chart(build_collateral_chart(weekly_view), use_container_width=True, config=CHART_CONFIG)
    with st.expander("Funding Conditions Methodology & Component Logic", expanded=False):
        st.markdown(
            "1. **Money Market Stress:** SOFR99 minus DFF, smoothed with a five-observation median, "
            "then converted to a one-sided expanding robust Z-score (minimum 52 observations, clipped to 0-4).\n"
            "2. **Technical vs persistent pressure:** the calendar window spans two business days before "
            "through three after month-end. Persistent pressure requires five severe observations or "
            "at least ten elevated observations in fifteen, with a majority outside calendar windows. "
            "Elevated off-calendar stress before confirmation is flagged separately as unconfirmed.\n"
            "3. **Reserve Vulnerability:** the expanding Z-score of WRESBAL / nominal GDP; low ratios raise vulnerability.\n"
            "4. **Reserve Drain:** the negative point-in-time Z-score of the 13-week reserve change.\n"
            "5. **Reserve Pressure:** 65% vulnerability plus 35% drain. Low reserves alone do not imply current stress.\n"
            "6. **Funding Core:** 60% Money Market Stress plus 40% Reserve Pressure.\n"
            "7. **Collateral Stress:** positive expanding MOVE Z-score; MOVE stays outside Funding Core. "
            "Treasury volatility alone is not a funding crisis.\n"
            "8. **Funding State:** systemic requires persistent funding stress plus reserve or collateral confirmation; "
            "otherwise persistent, calendar-technical, Treasury-volatility, or normal rules apply.\n"
            "9. **Funding Direction:** four-week Funding Core change compared with its expanding historical "
            "absolute-change distribution."
        )
        st.caption(snapshot.status.get("TimingConvention", ""))
        st.dataframe(pd.DataFrame([{"Series": key, "Source": value} for key, value in snapshot.status.get("SourceStatus", {}).items()]),
                     use_container_width=True, hide_index=True)

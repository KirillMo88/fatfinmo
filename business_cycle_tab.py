from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from business_cycle import (
    BUSINESS_PHASES,
    ECONOMY_REGIMES,
    BusinessCycleSnapshot,
    build_business_cycle_snapshot,
)
from macro_surprises import MacroSurprisesSnapshot, build_macro_surprises_snapshot


BUSINESS_CYCLE_TTL_SECONDS = 21600
BUSINESS_CYCLE_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

REGIME_COLORS = {
    "GOLDILOCKS": "#22c55e",
    "REFLATION": "#facc15",
    "STAGFLATION": "#ef4444",
    "DISINFLATIONARY SLOWDOWN": "#38bdf8",
    "DATA INCOMPLETE": "#64748b",
}
PHASE_COLORS = {
    "STRONG EXPANSION": "#16a34a",
    "LATE / SLOWING EXPANSION": "#f59e0b",
    "DETERIORATING CONTRACTION": "#ef4444",
    "EARLY RECOVERY": "#38bdf8",
    "DATA INCOMPLETE": "#64748b",
    "TRANSITION": "#94a3b8",
}
CONFIRMATION_COLORS = {
    "CONFIRMED RISING": "#ef4444",
    "EARLY RISING": "#f97316",
    "CONFIRMED FALLING": "#22c55e",
    "EARLY FALLING": "#38bdf8",
    "DATA INCOMPLETE": "#64748b",
}
TURNING_SIGNAL_COLORS = {
    "CONFIRMED IMPROVEMENT": "#16a34a",
    "SLOWDOWN WARNING": "#f97316",
    "NEUTRAL": "#64748b",
    "CONFIRMED DETERIORATION": "#ef4444",
    "POTENTIAL BOTTOMING": "#38bdf8",
    "DATA INCOMPLETE": "#334155",
}


@st.cache_data(show_spinner=True, ttl=BUSINESS_CYCLE_TTL_SECONDS)
def load_business_cycle_snapshot_cached(api_key: str | None, refresh_nonce: int = 0) -> BusinessCycleSnapshot:
    _ = refresh_nonce
    return build_business_cycle_snapshot(api_key=api_key)


@st.cache_data(show_spinner=True, ttl=BUSINESS_CYCLE_TTL_SECONDS)
def load_macro_surprises_snapshot_cached(history: pd.DataFrame, refresh_nonce: int = 0) -> MacroSurprisesSnapshot:
    return build_macro_surprises_snapshot(history, refresh_live=refresh_nonce > 0)


def render_business_cycle_tab(api_key: str | None) -> None:
    st.subheader("Business Cycle")

    c1, c2, c3 = st.columns([1.1, 1.1, 7.8])
    with c1:
        refresh = st.button("Refresh Business Cycle", use_container_width=True)
    with c2:
        st.caption("Refresh interval: 6 hours")
    if refresh:
        st.session_state["business_cycle_refresh_nonce"] = int(pd.Timestamp.now(tz="UTC").value)
        st.rerun()

    try:
        snapshot = load_business_cycle_snapshot_cached(api_key, int(st.session_state.get("business_cycle_refresh_nonce", 0)))
    except Exception as exc:
        st.error(f"Business Cycle model failed: {exc}")
        return

    history = snapshot.history.copy()
    if history.empty:
        st.info("Business Cycle history is unavailable.")
        return
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    current = snapshot.current or {}

    missing = snapshot.data_quality.loc[snapshot.data_quality["Status"].ne("OK")]
    if not missing.empty:
        st.warning("Some required macro series are incomplete: " + ", ".join(missing["Series"].astype(str).head(8).tolist()))

    render_current_status(current)

    st.markdown("### Primary Macro View")
    chart_range = st.radio(
        "Business Cycle chart range",
        options=["3Y", "5Y", "10Y", "MAX"],
        index=1,
        horizontal=True,
        key="business_cycle_chart_range",
    )
    d = filter_history_range(history, chart_range)
    st.plotly_chart(build_primary_macro_fig(d), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)

    st.markdown("### Regime Dynamics")
    map_col, lm_col = st.columns([1.05, 1.55])
    with map_col:
        map_window = st.radio("Regime map window", ["13W", "26W", "52W"], index=1, horizontal=True, key="business_cycle_map_window")
        st.plotly_chart(build_regime_map_fig(history, map_window), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    with lm_col:
        show_components = st.checkbox("Show pillar scores", value=False, key="business_cycle_show_pillars")
        st.plotly_chart(build_level_momentum_fig(d, show_components), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)

    inf_col, curve_col = st.columns(2)
    with inf_col:
        show_inflation_components = st.checkbox("Show inflation channel scores", value=False, key="business_cycle_show_inflation_components")
        st.plotly_chart(build_inflation_direction_fig(d, show_inflation_components), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    with curve_col:
        st.plotly_chart(build_expinf_curve_fig(d), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)

    st.markdown("### Inflation Details")
    lead_col, structural_col = st.columns(2)
    with lead_col:
        st.plotly_chart(build_leading_realized_fig(d), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    with structural_col:
        st.plotly_chart(build_t5yifr_fig(d), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)

    try:
        macro_snapshot = load_macro_surprises_snapshot_cached(
            history,
            int(st.session_state.get("business_cycle_refresh_nonce", 0)),
        )
        render_macro_surprises_section(macro_snapshot)
    except Exception as exc:
        st.warning(f"Macro Surprises are unavailable: {exc}")

    st.markdown("### Historical Asset Returns by Macro State")
    render_returns_section(snapshot)

    st.markdown("### Research Diagnostics")
    diag_col, quality_col = st.columns(2)
    with diag_col:
        st.dataframe(format_diagnostics(snapshot.diagnostics), use_container_width=True, hide_index=True)
    with quality_col:
        st.dataframe(snapshot.data_quality, use_container_width=True, hide_index=True)


def render_macro_surprises_section(snapshot: MacroSurprisesSnapshot) -> None:
    st.markdown("### Macro Surprises & Turning Points")
    st.caption(
        "Early-warning and confirmation overlay. These signals do not change the official Business Cycle Phase or Inflation State."
    )
    current = snapshot.current or {}
    latest = snapshot.latest_releases.copy()
    latest_lookup = {str(row["Indicator"]): row for _, row in latest.iterrows()} if not latest.empty else {}

    cards = st.columns([1.0, 1.45, 1.0])
    with cards[0]:
        render_status_card(
            "Growth Surprise",
            fmt_number(current.get("GrowthSurpriseScore"), 2),
            [
                ("Growth State", fmt_text(current.get("GrowthState"))),
                ("Turning Signal", fmt_text(current.get("TurningSignal"))),
                ("Persistence", fmt_text(current.get("Persistence"))),
                ("Growth Breadth", fmt_percent(current.get("GrowthBreadth"), 0)),
                ("Available Weight", fmt_percent(current.get("GrowthAvailableWeight"), 0)),
            ],
        )
    with cards[1]:
        render_growth_decomposition_card(latest_lookup)
    with cards[2]:
        cpi = latest_lookup.get("CPI", {})
        render_status_card(
            "Inflation Surprise",
            fmt_text(current.get("InflationSignal")),
            [
                ("Inflation State", fmt_text(current.get("InflationState"))),
                ("CPI Surprise", fmt_number(current.get("CPIContribution"), 2)),
                ("CPI State", fmt_text(current.get("CPISurpriseState"))),
                ("Released", fmt_release_date(cpi.get("Release Date"))),
                ("Actual", fmt_release_value("CPI", cpi.get("Actual"))),
                ("Forecast", fmt_release_value("CPI", cpi.get("Forecast"))),
            ],
        )

    macro_range = st.radio(
        "Macro Surprise range",
        options=["3Y", "5Y", "10Y", "MAX"],
        index=1,
        horizontal=True,
        key="business_cycle_macro_surprise_range",
    )
    chart_data = filter_history_range(snapshot.history, macro_range)
    st.plotly_chart(build_growth_surprise_fig(chart_data), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    component_col, cpi_col = st.columns(2)
    with component_col:
        st.plotly_chart(build_growth_components_fig(chart_data), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    with cpi_col:
        st.plotly_chart(build_cpi_surprise_fig(chart_data), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)

    st.markdown("#### Latest Macro Releases")
    if latest.empty:
        st.info("No completed releases with both Actual and Forecast are available.")
    else:
        display = latest.copy()
        display["Release Date"] = pd.to_datetime(display["Release Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for indicator in display["Indicator"].unique():
            mask = display["Indicator"].eq(indicator)
            for column in ["Actual", "Forecast", "Previous"]:
                display.loc[mask, column] = display.loc[mask, column].map(lambda value, name=indicator: fmt_release_value(name, value))
        for column in ["Raw Surprise", "Z Surprise", "Current Contribution"]:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: fmt_number(value, 2))
        display["Age"] = display["Age"].map(lambda value: f"{int(value)}d")
        st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("Macro Surprise Diagnostics"):
        st.dataframe(snapshot.diagnostics, use_container_width=True, hide_index=True)


def render_growth_decomposition_card(latest: dict[str, Any]) -> None:
    blocks: list[str] = []
    for indicator in ["PMI", "Retail Sales", "Initial Jobless Claims"]:
        row = latest.get(indicator, {})
        blocks.append(
            "<div style='padding:0.18rem 0 0.38rem;border-bottom:1px solid #263241;'>"
            f"<b style='color:#f8fafc'>{html.escape(indicator)}</b>"
            f"<div style='color:#cbd5e1'>Actual {html.escape(fmt_release_value(indicator, row.get('Actual')))} | "
            f"Forecast {html.escape(fmt_release_value(indicator, row.get('Forecast')))}</div>"
            f"<div style='color:#94a3b8'>Z {html.escape(fmt_number(row.get('Z Surprise'), 2))} | "
            f"Contribution {html.escape(fmt_number(row.get('Current Contribution'), 2))} | "
            f"{html.escape(fmt_release_date(row.get('Release Date')))}</div></div>"
        )
    st.markdown(
        "<div style='border:1px solid #263241;border-radius:8px;padding:0.75rem 0.85rem;background:#0f131a;min-height:13.5rem;'>"
        "<div style='font-size:0.72rem;color:#94a3b8;font-weight:800;text-transform:uppercase;'>Growth decomposition</div>"
        "<div style='font-size:0.76rem;margin-top:0.35rem;'>" + "".join(blocks) + "</div></div>",
        unsafe_allow_html=True,
    )


def build_growth_surprise_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    if not d.empty:
        add_background_bands(
            fig,
            d,
            "BusinessCycleDirection",
            {"IMPROVING": "#16a34a", "DETERIORATING": "#ef4444", "DATA INCOMPLETE": "#64748b"},
            opacity=0.08,
            row=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["GrowthSurpriseScore"],
                mode="lines",
                name="Growth Surprise Score",
                line={"color": "#f8fafc", "width": 2.2},
                customdata=d["BusinessCycleDirection"].astype(str),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Growth Surprise: %{y:.2f}<br>Business Cycle Direction: %{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        strip_colors = d["TurningSignal"].astype(str).map(lambda value: TURNING_SIGNAL_COLORS.get(value, "#64748b"))
        fig.add_trace(
            go.Bar(
                x=d["date"],
                y=np.ones(len(d)),
                marker={"color": strip_colors},
                name="Turning Signal",
                customdata=d["TurningSignal"].astype(str),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Turning Signal: %{customdata}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    for threshold, color, dash in [(0.20, "#22c55e", "dash"), (0.0, "#94a3b8", "dot"), (-0.20, "#ef4444", "dash")]:
        fig.add_hline(y=threshold, line={"color": color, "dash": dash, "width": 1}, row=1, col=1)
    fig.update_yaxes(title_text="Z", row=1, col=1)
    fig.update_yaxes(showticklabels=False, range=[0, 1], row=2, col=1)
    return style_business_fig(fig, "Growth Surprise Score and Turning Signal", 430)


def build_growth_components_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, name, color in [
        ("PMIContribution", "PMI contribution", "#38bdf8"),
        ("RetailSalesContribution", "Retail Sales contribution", "#f59e0b"),
        ("ClaimsContribution", "Claims contribution", "#22c55e"),
        ("GrowthSurpriseScore", "Growth Surprise Score", "#f8fafc"),
    ]:
        if column not in d.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d[column],
                mode="lines",
                name=name,
                line={"color": color, "width": 2.4 if column == "GrowthSurpriseScore" else 1.4},
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{name}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    return style_business_fig(fig, "Growth Surprise Components", 340)


def build_cpi_surprise_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not d.empty:
        add_background_bands(
            fig,
            d,
            "InflationState",
            {"RISING": "#ef4444", "FALLING": "#22c55e", "DATA INCOMPLETE": "#64748b"},
            opacity=0.08,
        )
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["CPIContribution"],
                mode="lines",
                name="CPI Surprise",
                line={"color": "#e879f9", "width": 2.1},
                customdata=d["InflationSignal"].astype(str),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>CPI Surprise: %{y:.2f}<br>Inflation Signal: %{customdata}<extra></extra>",
            )
        )
    for threshold, color, dash in [(0.20, "#ef4444", "dash"), (0.0, "#94a3b8", "dot"), (-0.20, "#22c55e", "dash")]:
        fig.add_hline(y=threshold, line={"color": color, "dash": dash, "width": 1})
    return style_business_fig(fig, "CPI Surprise / Inflation Confirmation-Challenge", 340)


def render_current_status(current: dict[str, Any]) -> None:
    cols = st.columns(3)
    with cols[0]:
        render_status_card(
            "Business Cycle",
            fmt_text(current.get("BusinessCycleState")),
            [
                ("Level", fmt_number(current.get("BusinessCycleLevel"), 2)),
                ("Position", fmt_text(current.get("BusinessCyclePosition"))),
                ("Momentum", fmt_number(current.get("BusinessCycleMomentum"), 2)),
                ("Direction", fmt_text(current.get("BusinessCycleDirection"))),
                ("Confidence", fmt_text(current.get("BusinessCycleConfidence"))),
                ("Transition Zone", "YES" if bool(current.get("BusinessCycleTransitionZone")) else "NO"),
            ],
        )
    with cols[1]:
        render_status_card(
            "Inflation",
            fmt_text(current.get("InflationState")),
            [
                ("Direction Score", fmt_number(current.get("InflationDirectionScore"), 2)),
                ("Confidence", fmt_text(current.get("InflationConfidence"))),
                ("Transition Zone", "YES" if bool(current.get("InflationTransitionZone")) else "NO"),
                ("Market Pricing", fmt_text(current.get("MarketPricingDirection"))),
                ("Model Implied", fmt_text(current.get("ModelImpliedDirection"))),
                ("Survey", fmt_text(current.get("SurveyInflationDirection"))),
                ("Agreement", fmt_text(current.get("InflationChannelAgreement"))),
                ("Confirmation", fmt_text(current.get("InflationConfirmationStatus"))),
            ],
        )
    with cols[2]:
        render_status_card(
            "Economy Regime",
            fmt_text(current.get("EconomyRegime")),
            [
                ("Business Direction", fmt_text(current.get("BusinessCycleDirection"))),
                ("Inflation Direction", fmt_text(current.get("InflationState"))),
                ("Regime Confidence", fmt_text(current.get("RegimeConfidence"))),
                ("Labor Cycle", fmt_text(current.get("LaborCycleState"))),
                ("Productivity Flag", "YES" if bool(current.get("ProductivityExpansionFlag")) else "NO"),
                ("Model", fmt_text(current.get("EconomyRegimeModelVersion"))),
            ],
        )


def render_status_card(title: str, headline: str, rows: list[tuple[str, str]]) -> None:
    body = "".join(
        "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:0.75rem;'>"
        f"<span style='color:#cbd5e1'>{html.escape(label)}</span>"
        f"<b style='color:#f8fafc;text-align:right;overflow-wrap:anywhere'>{html.escape(value)}</b></div>"
        for label, value in rows
    )
    st.markdown(
        f"""
<div style="border:1px solid #263241; border-radius:8px; padding:0.75rem 0.85rem; background:#0f131a; min-height:13.5rem;">
  <div style="font-size:0.72rem; color:#94a3b8; font-weight:800; text-transform:uppercase;">{html.escape(title)}</div>
  <div style="font-size:1.15rem; color:#f8fafc; font-weight:900; margin:0.25rem 0 0.65rem;">{html.escape(headline)}</div>
  <div style="display:grid; gap:0.28rem; font-size:0.78rem; color:#cbd5e1;">
    {body}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_primary_macro_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.04)
    add_background_bands(fig, d, "EconomyRegime", REGIME_COLORS, opacity=0.16, row=1)
    pmi = pd.to_numeric(d.get("ISM_3MMA"), errors="coerce")
    custom = np.stack(
        [
            d.get("BusinessCycleState", pd.Series(index=d.index, dtype="object")).astype(str),
            pd.to_numeric(d.get("BusinessCycleLevel"), errors="coerce"),
            pd.to_numeric(d.get("BusinessCycleMomentum"), errors="coerce"),
            d.get("InflationState", pd.Series(index=d.index, dtype="object")).astype(str),
            pd.to_numeric(d.get("InflationDirectionScore"), errors="coerce"),
            d.get("EconomyRegime", pd.Series(index=d.index, dtype="object")).astype(str),
            d.get("BusinessCycleConfidence", pd.Series(index=d.index, dtype="object")).astype(str),
            d.get("InflationConfidence", pd.Series(index=d.index, dtype="object")).astype(str),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=pmi,
            mode="lines",
            name="ISM Manufacturing PMI 3MMA",
            line={"color": "#f8fafc", "width": 2.0},
            customdata=custom,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>PMI 3MMA: %{y:.1f}<br>"
                "Business Cycle: %{customdata[0]}<br>Level: %{customdata[1]:.2f}<br>Momentum: %{customdata[2]:.2f}<br>"
                "Inflation: %{customdata[3]} / %{customdata[4]:.2f}<br>Economy Regime: %{customdata[5]}<br>"
                "BC Confidence: %{customdata[6]}<br>Inflation Confidence: %{customdata[7]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=50, line={"color": "#94a3b8", "dash": "dot", "width": 1}, row=1, col=1)
    strip_colors = d["BusinessCycleState"].astype(str).map(lambda state: PHASE_COLORS.get(state, "#64748b"))
    fig.add_trace(
        go.Bar(
            x=d["date"],
            y=np.ones(len(d)),
            marker={"color": strip_colors},
            name="Business Cycle State",
            customdata=d["BusinessCycleState"].astype(str),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Business Cycle: %{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="PMI", row=1, col=1)
    fig.update_yaxes(showticklabels=False, range=[0, 1], row=2, col=1)
    return style_business_fig(fig, "PMI with Economy Regime Background and Business Cycle Phase Strip", 430)


def build_level_momentum_fig(d: pd.DataFrame, show_components: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=-0.10, y1=0.10, fillcolor="#94a3b8", opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=d["date"], y=d["BusinessCycleLevel"], mode="lines", name="BusinessCycleLevel", line={"color": "#38bdf8", "width": 2.0}))
    fig.add_trace(go.Scatter(x=d["date"], y=d["BusinessCycleMomentum"], mode="lines", name="BusinessCycleMomentum", line={"color": "#facc15", "width": 2.0}))
    if show_components:
        for col, color in [("SurveyScore", "#60a5fa"), ("ProductionScore", "#22c55e"), ("DemandIncomeScore", "#f97316"), ("LaborScore", "#e879f9")]:
            fig.add_trace(go.Scatter(x=d["date"], y=d[col], mode="lines", name=col, line={"color": color, "width": 1.1, "dash": "dot"}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    return style_business_fig(fig, "Business Cycle Level and Momentum", 360)


def build_regime_map_fig(history: pd.DataFrame, window: str) -> go.Figure:
    n = {"13W": 13, "26W": 26, "52W": 52}.get(window, 26)
    d = history.dropna(subset=["BusinessCycleMomentum", "InflationDirectionScore"]).tail(n).copy()
    fig = go.Figure()
    if d.empty:
        return style_business_fig(fig, "Growth / Inflation Regime Map", 360)
    colors = d["EconomyRegime"].astype(str).map(lambda state: REGIME_COLORS.get(state, "#64748b"))
    fig.add_trace(
        go.Scatter(
            x=d["BusinessCycleMomentum"],
            y=d["InflationDirectionScore"],
            mode="lines+markers",
            marker={"size": 8, "color": colors, "line": {"color": "#0f172a", "width": 0.6}},
            line={"color": "#94a3b8", "width": 1.2},
            customdata=np.stack([d["date"].dt.strftime("%Y-%m-%d"), d["EconomyRegime"].astype(str)], axis=-1),
            hovertemplate="Date: %{customdata[0]}<br>Momentum: %{x:.2f}<br>Inflation Score: %{y:.2f}<br>Regime: %{customdata[1]}<extra></extra>",
            name="Weekly path",
        )
    )
    latest = d.tail(1)
    fig.add_trace(go.Scatter(x=latest["BusinessCycleMomentum"], y=latest["InflationDirectionScore"], mode="markers", marker={"size": 15, "color": "#f8fafc", "symbol": "circle-open", "line": {"width": 2}}, name="Current"))
    fig.add_vline(x=0, line={"color": "#94a3b8", "dash": "dash", "width": 1})
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dash", "width": 1})
    fig.add_annotation(x=0.96, y=0.96, xref="paper", yref="paper", text="REFLATION", showarrow=False, font={"color": "#facc15", "size": 10})
    fig.add_annotation(x=0.04, y=0.96, xref="paper", yref="paper", text="STAGFLATION", showarrow=False, font={"color": "#ef4444", "size": 10})
    fig.add_annotation(x=0.96, y=0.04, xref="paper", yref="paper", text="GOLDILOCKS", showarrow=False, font={"color": "#22c55e", "size": 10})
    fig.add_annotation(x=0.04, y=0.04, xref="paper", yref="paper", text="DISINFLATIONARY SLOWDOWN", showarrow=False, font={"color": "#38bdf8", "size": 10})
    fig.update_xaxes(title="BusinessCycleMomentum")
    fig.update_yaxes(title="InflationDirectionScore")
    return style_business_fig(fig, "Growth / Inflation Regime Map", 360)


def build_inflation_direction_fig(d: pd.DataFrame, show_components: bool) -> go.Figure:
    fig = go.Figure()
    add_background_bands(fig, d, "InflationState", {"RISING": "#ef4444", "FALLING": "#22c55e", "DATA INCOMPLETE": "#64748b"}, opacity=0.10)
    fig.add_hrect(y0=-0.20, y1=0.20, fillcolor="#94a3b8", opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=d["date"], y=d["InflationDirectionScore"], mode="lines", name="InflationDirectionScore", line={"color": "#f8fafc", "width": 2.0}))
    if show_components:
        for col, color in [("MarketPricingScore", "#38bdf8"), ("ModelImpliedInflationScore", "#facc15"), ("SurveyInflationScore", "#e879f9")]:
            fig.add_trace(go.Scatter(x=d["date"], y=d[col], mode="lines", name=col, line={"color": color, "width": 1.4, "dash": "dot"}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    return style_business_fig(fig, "Inflation Direction", 340)


def build_expinf_curve_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=d["date"], y=d["EXPINF1YR"], mode="lines", name="EXPINF1YR", line={"color": "#facc15", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["EXPINF5YR"], mode="lines", name="EXPINF5YR", line={"color": "#38bdf8", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["InflationCurve"], mode="lines", name="EXPINF1YR - EXPINF5YR", line={"color": "#f8fafc", "width": 1.8}), row=2, col=1)
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1}, row=2, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Curve", row=2, col=1)
    return style_business_fig(fig, "EXPINF Term Structure", 340)


def build_leading_realized_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_background_bands(fig, d, "InflationConfirmationStatus", CONFIRMATION_COLORS, opacity=0.10)
    fig.add_trace(go.Scatter(x=d["date"], y=d["InflationDirectionScore"], mode="lines", name="InflationDirectionScore", line={"color": "#f8fafc", "width": 2.0}))
    fig.add_trace(go.Scatter(x=d["date"], y=d["RealizedInflationMomentum"], mode="lines", name="RealizedInflationMomentum", line={"color": "#f97316", "width": 1.8}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    return style_business_fig(fig, "Leading vs Realized Inflation", 320)


def build_t5yifr_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=d["date"], y=d["T5YIFR"], mode="lines", name="T5YIFR", line={"color": "#22c55e", "width": 1.9}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["T5YIFR_Change_13W"], mode="lines", name="13W change", line={"color": "#38bdf8", "width": 1.5}), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["T5YIFR_Change_26W"], mode="lines", name="26W change", line={"color": "#facc15", "width": 1.5}), row=2, col=1)
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1}, row=2, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Change", row=2, col=1)
    return style_business_fig(fig, "Structural Inflation Expectations / T5YIFR", 320)


def render_returns_section(snapshot: BusinessCycleSnapshot) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        classifier = st.radio("Return classifier", ["Economy Regime", "Business Cycle"], index=0, horizontal=True, key="business_cycle_return_classifier")
    with c2:
        asset = st.selectbox("Asset", ["ALL", "SPY", "QQQ", "GLD", "BTC"], index=0, key="business_cycle_return_asset")
    with c3:
        horizon = st.selectbox("Horizon", ["ALL", "3M", "6M", "12M"], index=0, key="business_cycle_return_horizon")
    with c4:
        metric = st.selectbox("Metric", ["Average Return", "Median Return", "Hit Rate"], index=0, key="business_cycle_return_metric")

    phase = snapshot.phase_returns.copy()
    regime = snapshot.regime_returns.copy()
    selected = regime if classifier == "Economy Regime" else phase
    filtered = filter_return_stats(selected, asset, horizon)
    st.dataframe(style_return_stats(filtered), use_container_width=True, hide_index=True, height=table_height(filtered))

    viz_col, table_col = st.columns([1, 1.45])
    with viz_col:
        viz_asset = "SPY" if asset == "ALL" else asset
        st.plotly_chart(build_return_heatmap(selected, viz_asset, metric), use_container_width=True, config=BUSINESS_CYCLE_PLOTLY_CONFIG)
    with table_col:
        st.caption("Descriptive historical statistics only. Forward returns are not used to optimize model parameters.")
        st.dataframe(style_return_stats(filter_return_stats(phase, asset, horizon)), use_container_width=True, hide_index=True, height=table_height(filter_return_stats(phase, asset, horizon)))
        st.dataframe(style_return_stats(filter_return_stats(regime, asset, horizon)), use_container_width=True, hide_index=True, height=table_height(filter_return_stats(regime, asset, horizon)))

    st.markdown("#### Information Content Comparison")
    st.dataframe(format_eta(snapshot.eta_squared), use_container_width=True, hide_index=True)


def filter_return_stats(frame: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
    out = frame.copy()
    if asset != "ALL":
        out = out[out["Asset"].eq(asset)]
    if horizon != "ALL":
        out = out[out["Horizon"].eq(horizon)]
    return out


def style_return_stats(frame: pd.DataFrame) -> Any:
    out = frame.copy()
    for col in ["AverageReturn", "MedianReturn", "HitRate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return (
        out.style.format(
            {
                "AverageReturn": "{:.1%}",
                "MedianReturn": "{:.1%}",
                "HitRate": "{:.0%}",
                "EtaSquared": "{:.3f}",
            },
            na_rep="N/A",
        )
        .map(lambda value: return_cell_style(value), subset=[col for col in ["AverageReturn", "MedianReturn", "HitRate"] if col in out.columns])
        .set_properties(**{"background-color": "#0f131a", "color": "#e5e7eb", "border-color": "#263241"})
    )


def build_return_heatmap(frame: pd.DataFrame, asset: str, metric: str) -> go.Figure:
    metric_col = {"Average Return": "AverageReturn", "Median Return": "MedianReturn", "Hit Rate": "HitRate"}[metric]
    d = frame[frame["Asset"].eq(asset)].copy()
    fig = go.Figure()
    if d.empty:
        return style_business_fig(fig, f"{asset} Forward Return Heatmap", 320)
    pivot = d.pivot_table(index="State", columns="Horizon", values=metric_col, aggfunc="mean")
    pivot = pivot.reindex(BUSINESS_PHASES if frame["Classifier"].astype(str).str.contains("Business").any() else ECONOMY_REGIMES)
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[[0.0, "#ef4444"], [0.5, "#f8fafc"], [1.0, "#16a34a"]],
            zmid=0 if metric_col != "HitRate" else 0.5,
            text=np.vectorize(lambda x: "N/A" if not np.isfinite(x) else (f"{x:.0%}" if metric_col == "HitRate" else f"{x:.1%}"))(pivot.values.astype(float)),
            texttemplate="%{text}",
            hovertemplate="State: %{y}<br>Horizon: %{x}<br>Value: %{text}<extra></extra>",
        )
    )
    return style_business_fig(fig, f"{asset} {metric} by Macro State", 320)


def format_eta(frame: pd.DataFrame) -> Any:
    out = frame.copy()
    out["Classifier"] = out["Classifier"].replace({"BusinessCycleState": "Business Cycle", "EconomyRegime": "Economy Regime"})
    return out.style.format({"EtaSquared": "{:.3f}"}, na_rep="N/A").set_properties(**{"background-color": "#0f131a", "color": "#e5e7eb", "border-color": "#263241"})


def format_diagnostics(frame: pd.DataFrame) -> Any:
    return frame.style.format(
        {
            "TransitionsPerYear": "{:.2f}",
            "MedianDurationWeeks": "{:.1f}",
        },
        na_rep="N/A",
    ).set_properties(**{"background-color": "#0f131a", "color": "#e5e7eb", "border-color": "#263241"})


def add_background_bands(fig: go.Figure, d: pd.DataFrame, state_col: str, colors: dict[str, str], opacity: float, row: int | None = None) -> None:
    if d.empty or state_col not in d.columns:
        return
    dates = pd.to_datetime(d["date"], errors="coerce").reset_index(drop=True)
    states = d[state_col].astype(str).reset_index(drop=True)
    start_idx = 0
    for idx in range(1, len(d) + 1):
        if idx == len(d) or states.iloc[idx] != states.iloc[start_idx]:
            x0 = dates.iloc[start_idx]
            x1 = dates.iloc[idx - 1] + pd.Timedelta(days=7)
            state = states.iloc[start_idx]
            kwargs = {"x0": x0, "x1": x1, "fillcolor": colors.get(state, "#64748b"), "opacity": opacity, "line_width": 0}
            if row is None:
                fig.add_vrect(**kwargs)
            else:
                fig.add_vrect(**kwargs, row=row, col=1)
            start_idx = idx


def style_business_fig(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 40, "t": 58, "b": 44},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "top", "y": -0.13, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickformat="%b'%y", showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569")
    fig.update_yaxes(showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569")
    return fig


def filter_history_range(history: pd.DataFrame, chart_range: str) -> pd.DataFrame:
    d = history.dropna(subset=["date"]).copy()
    if chart_range == "MAX" or d.empty:
        return d
    years = {"3Y": 3, "5Y": 5, "10Y": 10}.get(chart_range, 5)
    cutoff = pd.to_datetime(d["date"]).max() - pd.DateOffset(years=years)
    return d[d["date"].ge(cutoff)].copy()


def table_height(frame: pd.DataFrame) -> int:
    return min(max(170, 44 + len(frame) * 28), 520)


def fmt_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(value)


def fmt_number(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{decimals}f}"


def fmt_percent(value: Any, decimals: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{decimals}%}"


def fmt_release_date(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    return date.strftime("%d %b %Y") if pd.notna(date) else "N/A"


def fmt_release_value(indicator: str, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    if indicator in {"Retail Sales", "CPI"}:
        return f"{number:.1%}"
    if indicator == "Initial Jobless Claims":
        return f"{number:,.0f}"
    return f"{number:.1f}"


def return_cell_style(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    if number >= 0.10:
        return "background-color:#16a34a; color:#ffffff;"
    if number > 0:
        return "background-color:#86efac; color:#111827;"
    if number == 0:
        return "background-color:#f8fafc; color:#111827;"
    if number > -0.10:
        return "background-color:#fdba74; color:#111827;"
    return "background-color:#ef4444; color:#ffffff;"

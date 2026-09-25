from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from financial_fragility import MODEL_VERSION, FinancialFragilitySnapshot, _number, build_financial_fragility_snapshot
from global_dashboard_tab import _load_snapshots


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
STATE_COLORS = {
    "NORMAL": "#22c55e",
    "WATCH": "#facc15",
    "HIGH": "#f97316",
    "EXTREME": "#ef4444",
    "STABLE": "#22c55e",
    "MILD TRANSITION": "#94a3b8",
    "ELEVATED": "#facc15",
    "EXTREME TRANSITION": "#ef4444",
    "UNAVAILABLE": "#334155",
}


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}{suffix}" if np.isfinite(number) else "n/a"


def _state_color(value: Any) -> str:
    text = str(value).upper()
    return STATE_COLORS.get(text, "#64748b")


def _card(title: str, subtitle: str, state: Any, score: Any, rows: list[tuple[str, Any]], *, wide: bool = False) -> None:
    detail = "".join(
        f"<div class='ff-row'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class='ff-card{' ff-wide' if wide else ''}' style='border-top-color:{_state_color(state)}'>
          <div class='ff-card-head'><div><div class='ff-title'>{html.escape(title)}</div><div class='ff-subtitle'>{html.escape(subtitle)}</div></div>
          <div class='ff-score'><span>{html.escape(str(state))}</span><b>{_fmt(score, 0)} /100</b></div></div>
          <div class='ff-details'>{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _styles() -> None:
    st.markdown(
        """
        <style>
        .ff-card {border:1px solid #273244;border-top:3px solid #64748b;border-radius:6px;padding:13px 15px;background:#111720;min-height:185px;margin-bottom:12px;overflow:hidden}
        .ff-wide {min-height:120px}.ff-card-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.ff-title{font-size:.86rem;color:#f8fafc;font-weight:800;text-transform:uppercase}.ff-subtitle{font-size:.70rem;color:#8ea4bd;margin-top:3px}.ff-score{text-align:right;white-space:nowrap}.ff-score span{display:block;color:#f8fafc;font-size:1rem;font-weight:800}.ff-score b{display:block;color:#cbd5e1;font-size:.78rem;margin-top:3px}.ff-details{display:grid;gap:5px;margin-top:14px}.ff-row{display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:12px;align-items:start;font-size:.75rem;line-height:1.25}.ff-row span{color:#94a3b8}.ff-row strong{color:#e5edf6;text-align:right;max-width:260px;overflow-wrap:anywhere}.ff-formula{font-size:.72rem;color:#8ea4bd;margin:-5px 0 10px}.ff-method{font-size:.78rem;color:#cbd5e1;line-height:1.45}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _filter(history: pd.DataFrame, selected: str) -> pd.DataFrame:
    if history.empty:
        return history
    end = pd.to_datetime(history["Date"]).max()
    years = {"1Y": 1, "5Y": 5, "10Y": 10}.get(selected)
    if years is None:
        return history
    return history.loc[pd.to_datetime(history["Date"]).ge(end - pd.DateOffset(years=years))].copy()


def _discrete_colors(domain: list[str]) -> list[list[Any]]:
    names = domain + ["UNAVAILABLE"]
    scale: list[list[Any]] = []
    denom = len(names)
    for index, name in enumerate(names):
        left = index / denom
        right = (index + 1) / denom
        color = STATE_COLORS.get(name, "#334155")
        scale.extend([[left, color], [right, color]])
    scale[-1][0] = 1.0
    return scale


def _add_strip(fig: go.Figure, frame: pd.DataFrame, column: str, row: int, domain: list[str], label: str) -> None:
    values = frame[column].astype(str) if column in frame else pd.Series("UNAVAILABLE", index=frame.index)
    codes = values.map({name: index for index, name in enumerate(domain)}).fillna(len(domain)).to_numpy()
    fig.add_trace(
        go.Heatmap(
            x=frame["Date"], y=[label], z=[codes], zmin=0, zmax=len(domain),
            colorscale=_discrete_colors(domain), showscale=False, name=label,
            text=[values.tolist()], hovertemplate="Date: %{x|%Y-%m-%d}<br>" + label + ": %{text}<extra></extra>",
        ),
        row=row, col=1,
    )


def build_historical_figure(history: pd.DataFrame) -> go.Figure:
    rows = ["General Regime", "Macro Pressure", "Market Vulnerability", "Credit Stress", "Funding Stress", "Market Volatility Stress", "Transition Risk"]
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[0.36] + [0.09] * 7, subplot_titles=("SPX Log", *rows))
    fig.add_trace(go.Scatter(x=history["Date"], y=np.log(pd.to_numeric(history["SPX"], errors="coerce")), mode="lines", name="SPX Log", line={"color": "#f1f5f9", "width": 1.5}, hovertemplate="Date: %{x|%Y-%m-%d}<br>SPX: %{customdata:.2f}<extra></extra>", customdata=pd.to_numeric(history["SPX"], errors="coerce")), row=1, col=1)
    _add_strip(fig, history, "GeneralRegime", 2, ["WATCH", "HIGH", "EXTREME"], "General Regime")
    for row, column, label in ((3, "MacroPressure", "Macro Pressure"), (4, "MarketVulnerability", "Market Vulnerability"), (5, "CreditStress", "Credit Stress"), (6, "FundingStress", "Funding Stress"), (7, "MarketVolatilityStress", "Market Volatility Stress"), (8, "TransitionRisk", "Transition Risk")):
        fig.add_trace(go.Scatter(x=history["Date"], y=pd.to_numeric(history[column], errors="coerce"), mode="lines", name=label, line={"color": "#38bdf8", "width": 1.3}, hovertemplate="Date: %{x|%Y-%m-%d}<br>" + label + ": %{y:.1f}<extra></extra>"), row=row, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    for row in range(3, 9):
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    fig.update_layout(height=820, margin={"l": 55, "r": 20, "t": 60, "b": 35}, paper_bgcolor="#0d1117", plot_bgcolor="#111720", font={"color": "#dbeafe", "size": 10}, legend={"orientation": "h", "y": -0.04}, hovermode="x unified")
    return fig


def _interpretation(snapshot: FinancialFragilitySnapshot) -> str:
    return str(snapshot.current.get("Interpretation", "Financial fragility is unavailable."))


def _render_methodology() -> None:
    st.markdown("### FINANCIAL FRAGILITY METHODOLOGY & COMPONENT LOGIC")
    sections = {
        "Macro Pressure": "0.60 Liquidity Pressure Risk + 0.20 Rates Pressure Risk + 0.20 Core FC Pressure. Liquidity uses the production forecast state and pressure score; rates and FC levels use trailing point-in-time percentiles.",
        "Market Vulnerability": "0.55 SMA200W Stretch + 0.20 ROC Momentum 3M Risk + 0.10 Positioning Vulnerability + 0.05 Reserve Vulnerability + 0.10 reduced Current Risk Vulnerability. Structural SMA200M extension is excluded.",
        "Credit Stress": "HY OAS 3Y point-in-time percentile only. IG OAS and HY direction remain diagnostics and do not receive an independent score weight.",
        "Funding Stress": "max(25 x Funding Core, Funding State floor), clipped to 0-100. Reserve Vulnerability is kept separate from actual Funding Stress.",
        "Market Volatility Stress": "0.30 VIX Level + 0.30 VIX Momentum + 0.25 VIX/VIX3M Term Structure + 0.15 Realized Volatility 20D, with PIT percentiles where applicable.",
        "Transition Risk": "0.30 Liquidity + 0.25 Market Cycle + 0.20 Business Cycle + 0.15 Rates/FC + 0.10 Cross-Cycle Divergence. It measures instability and turning points, not stress itself.",
        "General Regime": "Rule-based priority EXTREME, HIGH, WATCH, NORMAL. Actual stress transmission dominates; Macro Pressure and Market Vulnerability represent latent fragility and cannot create EXTREME on their own.",
    }
    for title, body in sections.items():
        with st.expander(title):
            st.markdown(f"<div class='ff-method'>{html.escape(body)}</div>", unsafe_allow_html=True)
    st.caption("LATENT FRAGILITY = Macro Pressure + Market Vulnerability | ACTUAL STRESS TRANSMISSION = Credit + Funding + Volatility | SYSTEM INSTABILITY = Transition Risk")


def render_financial_fragility_tab(api_key: str | None, liquidity_regime: pd.DataFrame) -> None:
    _styles()
    st.subheader("Financial Fragility")
    st.caption(f"Model {MODEL_VERSION} | synthesis of existing production modules; calculation history remains full-length")
    loaded, errors = _load_snapshots(api_key)
    try:
        snapshot = build_financial_fragility_snapshot(
            liquidity_regime=liquidity_regime,
            forecast_frame=loaded["forecast"],
            market_snapshot=loaded["market"],
            business_snapshot=loaded["business"],
            rates_snapshot=loaded["rates"],
            funding_snapshot=loaded["funding"],
            treasury_snapshot=loaded["treasury"],
            transition_snapshot={},
        )
    except Exception as exc:
        st.error(f"Financial Fragility model failed: {type(exc).__name__}: {exc}")
        return
    if snapshot.history.empty:
        st.warning("Financial Fragility history is unavailable.")
        return
    current = snapshot.current
    st.markdown(f"**Dashboard As Of:** {pd.Timestamp(snapshot.as_of).strftime('%Y-%m-%d') if pd.notna(snapshot.as_of) else 'unavailable'}")
    if errors:
        st.warning("Some source modules are unavailable; affected components remain blank. " + " | ".join(errors))

    st.markdown("### MACRO LAYER")
    st.caption("Slow build-up of macro-financial pressure and market vulnerability")
    col1, col2 = st.columns(2)
    with col1:
        _card("Macro Pressure", "Slow macro-financial pressure on the system", current.get("MacroPressureState"), current.get("MacroPressure"), [
            ("Formula", "60% Liquidity | 20% Rates | 20% Core FC"),
            ("Liquidity Pressure Risk", _fmt(current.get("LiquidityPressureRisk"), 1)),
            ("Rates Pressure Risk", _fmt(current.get("RatesPressureRisk"), 1)),
            ("Core FC Pressure", _fmt(current.get("CoreFCPressure"), 1)),
            ("Forecast / direction", f"{current.get('LiquidityForecastState', 'n/a')} / {current.get('GlobalLiquidityDirection', 'n/a')}"),
            ("Primary drivers", current.get("PrimaryDrivers", "n/a")),
        ])
    with col2:
        _card("Market Vulnerability", "Internal market fragility and susceptibility to shock", current.get("MarketVulnerabilityState"), current.get("MarketVulnerability"), [
            ("Formula", "55% SMA200W | 20% ROC Mom 3M | 10% Positioning | 5% Reserve | 10% Current Risk"),
            ("SMA200W Stretch", _fmt(current.get("SMA200WStretch"), 1)),
            ("ROC Momentum 3M Risk", _fmt(current.get("ROCMomentum3MRisk"), 1)),
            ("Positioning Vulnerability", _fmt(current.get("PositioningVulnerability"), 1)),
            ("Reserve Vulnerability", _fmt(current.get("ReserveVulnerability"), 1)),
            ("Breadth / Beta / RSI", f"{_fmt(current.get('BreadthRisk'), 0)} / {_fmt(current.get('HighBetaRisk'), 0)} / {_fmt(current.get('RSIDivergenceRisk'), 0)}"),
        ])

    st.markdown("### STRESS REALIZATION LAYER")
    st.caption("Bottom-up confirmation through actual market stress channels")
    col1, col2, col3 = st.columns(3)
    with col1:
        _card("Credit Stress", "Stress transmission through HY credit", current.get("CreditStressState"), current.get("CreditStress"), [("HY OAS 3Y Percentile", _fmt(current.get("HYOAS_3Y_Percentile"), 1)), ("Current HY OAS", _fmt(current.get("HY_OAS"), 2)), ("Interpretation", "Credit market does not confirm stress." if _number(current.get("CreditStress")) < 50 else "Credit stress is transmitting.")])
    with col2:
        _card("Funding Stress", "Funding system and money-market strain", current.get("FundingStressState"), current.get("FundingStress"), [("Funding Core", _fmt(current.get("FundingCore"), 2)), ("Money Market Stress", _fmt(current.get("MoneyMarketStress"), 2)), ("Funding State", current.get("FundingState", "n/a")), ("Persistent Flag", current.get("PersistentFundingFlag", "n/a"))])
    with col3:
        _card("Market Volatility Stress", "Fast market shock channel", current.get("MarketVolatilityStressState"), current.get("MarketVolatilityStress"), [("VIX", _fmt(current.get("VIX"), 2)), ("VIX / VIX3M", _fmt(current.get("VIX_VIX3M_Ratio"), 3)), ("VIX Momentum Risk", _fmt(current.get("VIXMomentumRisk"), 1)), ("Realized Volatility", _fmt(current.get("RealizedVolatility20D"), 3))])

    st.markdown("### TRANSITION RISK")
    _card("Transition Risk", "How stable the current regime configuration is", current.get("TransitionRiskState"), current.get("TransitionRisk"), [
        ("Liquidity Transition", _fmt(current.get("LiquidityTransition"), 1)),
        ("Market Cycle Transition", _fmt(current.get("MarketCycleTransition"), 1)),
        ("Business Cycle Transition", _fmt(current.get("BusinessCycleTransition"), 1)),
        ("Rates / FC Transition", _fmt(current.get("RatesFCTransition"), 1)),
        ("Cross-Cycle Divergence", _fmt(current.get("CrossCycleDivergence"), 1)),
        ("Key Divergence", current.get("KeyDivergence", "n/a")),
    ], wide=True)
    st.caption(_interpretation(snapshot))

    st.markdown("### OVERALL ASSESSMENT")
    st.markdown(f"<div class='ff-card ff-wide' style='border-top-color:{_state_color(current.get('GeneralRegime'))}'><div class='ff-title'>GENERAL REGIME</div><div class='ff-score' style='text-align:left;margin-top:5px'><span>{html.escape(str(current.get('GeneralRegime', 'UNAVAILABLE')))}</span></div><div class='ff-details'><div class='ff-row'><span>Risk Drivers</span><strong>{html.escape(str(current.get('PrimaryDrivers', 'UNAVAILABLE')))}</strong></div><div class='ff-row'><span>Stabilizers</span><strong>{html.escape(str(current.get('PrimaryStabilizers', 'UNAVAILABLE')))}</strong></div><div class='ff-row'><span>Data Coverage</span><strong>{_fmt(current.get('DataCoverage'), 1)}%</strong></div></div></div>", unsafe_allow_html=True)

    st.markdown("### SPX LOG AND REGIME STATES")
    range_choice = st.radio("Financial Fragility history range", ["1Y", "5Y", "10Y", "FULL"], index=1, horizontal=True, key="financial_fragility_range")
    st.plotly_chart(build_historical_figure(_filter(snapshot.history, range_choice)), use_container_width=True, config=PLOT_CONFIG)
    st.markdown("### DATA QUALITY")
    st.dataframe(snapshot.data_quality, use_container_width=True, hide_index=True)
    _render_methodology()

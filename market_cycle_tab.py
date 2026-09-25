from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from current_risk import classify_component_state, current_risk_new_event
from market_cycle import MarketCycleSnapshot, build_market_cycle_snapshot
from spy_macro_outlook import (
    SPY_MACRO_HORIZONS,
    SPY_MACRO_RANGE_OPTIONS,
    SPYMacroOutlook,
    build_model_details,
    build_spy_macro_outlook,
    build_spy_macro_workbook,
    score_direction,
    score_state,
    term_structure_state,
)


MARKET_CYCLE_TTL_SECONDS = 21600
MARKET_CYCLE_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
MARKET_CYCLE_RANGE_OPTIONS = ["1Y", "5Y", "10Y", "20Y", "FULL"]
MARKET_CYCLE_RANGE_YEARS = {"1Y": 1, "5Y": 5, "10Y": 10, "20Y": 20}

PHASE_COLORS = {
    "EARLY STRUCTURAL EXPANSION": "#38bdf8",
    "MID STRUCTURAL EXPANSION": "#22c55e",
    "MATURE STRUCTURAL EXPANSION": "#facc15",
    "LATE STRUCTURAL EXPANSION": "#f97316",
    "STRUCTURAL TOP ZONE": "#ef4444",
    "STRUCTURAL CONTRACTION": "#a855f7",
    "EARLY RECOVERY": "#38bdf8",
    "EARLY EXPANSION": "#22c55e",
    "EXPANSION": "#16a34a",
    "MATURE EXPANSION": "#facc15",
    "LATE EXPANSION": "#f97316",
    "CONTRACTION": "#ef4444",
    "STRETCHED EXPANSION": "#f59e0b",
    "CAPITULATION / OVERSOLD": "#38bdf8",
    "RISK EXPANSION": "#22c55e",
    "LATE RISK EXPANSION": "#facc15",
    "RISK CONTRACTION": "#ef4444",
    "EARLY RISK RECOVERY": "#38bdf8",
    "TRANSITION": "#94a3b8",
    "LOW": "#22c55e",
    "NORMAL": "#94a3b8",
    "ELEVATED": "#facc15",
    "CROWDED": "#f97316",
    "EXTREME": "#ef4444",
    "HIGH": "#f97316",
    "ACUTE": "#ef4444",
    "LOW CONFIRMATION": "#22c55e",
    "MODERATE": "#facc15",
    "HIGH RISK": "#f97316",
    "RED FLAG": "#ef4444",
    "CORRECTION WATCH": "#facc15",
    "CORRECTION": "#f97316",
    "STRESS BUILDING": "#fb923c",
    "CAPITULATION": "#ef4444",
    "EARLY EXHAUSTION": "#38bdf8",
    "BOTTOMING": "#22c55e",
    "RE-ACCELERATION WATCH": "#a855f7",
    "RE-ACCELERATION": "#ef4444",
    "LOW UNDERCUT": "#e879f9",
    "CAPITULATION RECLAIM": "#10b981",
    "CONFIRMED BREAKDOWN": "#991b1b",
    "RECOVERY CONFIRMED": "#00c853",
    "NEW CORRECTION WAVE WATCH": "#f59e0b",
    "NEW CORRECTION WAVE": "#dc2626",
}

CYCLE_RISK_REGIME_ORDER = [
    "RISK ON",
    "SHORT-CYCLE CORRECTION RISK",
    "HIGH CORRECTION RISK",
    "MAXIMUM DRAWDOWN RISK",
    "HIGH VOLATILITY / BOTTOMING",
]

CYCLE_RISK_REGIME_COLORS = {
    "RISK ON": "#22C55E",
    "SHORT-CYCLE CORRECTION RISK": "#F59E0B",
    "HIGH CORRECTION RISK": "#EF4444",
    "MAXIMUM DRAWDOWN RISK": "#B91C1C",
    "HIGH VOLATILITY / BOTTOMING": "#A855F7",
}


@st.cache_data(show_spinner=True, ttl=MARKET_CYCLE_TTL_SECONDS)
def load_market_cycle_snapshot_cached(refresh_nonce: int = 0) -> MarketCycleSnapshot:
    _ = refresh_nonce
    return build_market_cycle_snapshot()


@st.cache_data(show_spinner=True, ttl=MARKET_CYCLE_TTL_SECONDS)
def load_spy_macro_outlook_cached(history: pd.DataFrame, api_key: str | None = None, refresh_nonce: int = 0) -> SPYMacroOutlook:
    _ = refresh_nonce
    return build_spy_macro_outlook(history, api_key)


def render_market_cycle_tab(api_key: str | None = None) -> None:
    st.subheader("Market Cycle")
    c1, c2, c3 = st.columns([1.05, 1.2, 7.75])
    with c1:
        refresh = st.button("Refresh Market Cycle", use_container_width=True)
    with c2:
        st.caption("Model: MARKET_CYCLE_V1")
    if refresh:
        st.session_state["market_cycle_refresh_nonce"] = int(st.session_state.get("market_cycle_refresh_nonce", 0)) + 1
        st.rerun()

    try:
        snapshot = load_market_cycle_snapshot_cached(int(st.session_state.get("market_cycle_refresh_nonce", 0)))
    except Exception as exc:
        st.error(f"Market Cycle model failed: {exc}")
        return

    history = snapshot.history.copy()
    if history.empty:
        st.info("Market Cycle data is unavailable.")
        return
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    current = snapshot.current or {}

    try:
        spy_macro_outlook = load_spy_macro_outlook_cached(
            history,
            api_key,
            int(st.session_state.get("market_cycle_refresh_nonce", 0)),
        )
    except Exception as exc:
        spy_macro_outlook = None
        st.warning(f"SPY Macro Outlook is temporarily unavailable: {exc}")

    render_summary_cards(current, spy_macro_outlook)
    st.markdown("### Historical Outlook - Current Market State")
    render_outlook_table(snapshot.outlook)

    normalize_range_state("market_cycle_range")
    market_cycle_range = st.radio("Market Cycle range", MARKET_CYCLE_RANGE_OPTIONS, index=3, horizontal=True, key="market_cycle_range")
    range_start, range_end = range_domain(history, market_cycle_range)
    display_history = filter_range_domain(history, range_start, range_end)

    st.markdown("### SPX Multi-Layer Market Cycles")
    cycle_col, cycle_summary_col = st.columns([3.25, 1.0])
    with cycle_col:
        st.plotly_chart(
            apply_common_x_range(build_multi_layer_market_cycles_fig(history, range_start, range_end), range_start, range_end),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )
        st.caption(
            "SPX, the ~41M primary market cycle, the ~80-95M long extension cycle, and actual structural extension are separate layers. "
            "Trough and peak markers are historical context, not deterministic buy/sell dates."
        )
    with cycle_summary_col:
        render_multi_layer_cycle_summary(current)

    st.markdown("### Structural")
    structural_history = display_history
    st.plotly_chart(
        apply_common_x_range(build_structural_sync_fig(structural_history), range_start, range_end),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )
    structural_col_1, structural_col_2 = st.columns(2)
    with structural_col_1:
        st.plotly_chart(apply_common_x_range(build_structural_momentum_cycle_fig(structural_history), range_start, range_end), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with structural_col_2:
        st.plotly_chart(apply_common_x_range(build_sma200m_extension_fig(structural_history), range_start, range_end), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)

    st.markdown("### Medium Term")
    momentum_history = display_history
    st.plotly_chart(
        apply_common_x_range(build_medium_term_sync_fig(momentum_history), range_start, range_end),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )
    medium_col_1, medium_col_2 = st.columns(2)
    with medium_col_1:
        st.plotly_chart(apply_common_x_range(build_momentum_cycle_fig(momentum_history), range_start, range_end), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with medium_col_2:
        st.plotly_chart(apply_common_x_range(build_sma200w_extension_fig(momentum_history), range_start, range_end), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)

    if spy_macro_outlook is not None:
        render_spy_macro_outlook(spy_macro_outlook, history)

    st.markdown("### Current Risk")
    risk_col, confirm_col = st.columns([1.05, 1.25])
    with risk_col:
        st.plotly_chart(build_current_risk_components_fig(current), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with confirm_col:
        render_current_risk_signal_table(current)
    risk_range_start, risk_range_end = range_domain(snapshot.daily, market_cycle_range)
    risk_history_start = max(pd.Timestamp("2015-01-01"), pd.Timestamp(risk_range_start))
    st.plotly_chart(
        apply_common_x_range(
            build_current_risk_signals_fig(snapshot.daily, risk_history_start, risk_range_end),
            risk_history_start,
            risk_range_end,
        ),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )
    st.plotly_chart(
        apply_common_x_range(
            build_current_risk_indicators_fig(snapshot.daily, risk_history_start, risk_range_end),
            risk_history_start,
            risk_range_end,
        ),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )

    st.markdown("### Positioning Risk")
    st.plotly_chart(
        apply_common_x_range(
            build_positioning_vulnerability_fig(display_history),
            range_start,
            range_end,
        ),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )

    st.markdown("### Correction Bottom Indicator")
    correction_range_start, correction_range_end = range_domain(snapshot.correction_daily, market_cycle_range)
    correction_start = max(pd.Timestamp("2005-01-01"), pd.Timestamp(correction_range_start))
    render_correction_bottom_indicator(snapshot.correction_daily, current, correction_start, correction_range_end)

    st.markdown("### Timeline and Analog Distributions")
    st.plotly_chart(build_timeline_fig(filter_range(history, years=25)), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    dist_col_1, dist_col_2 = st.columns(2)
    horizon = st.radio("Analog horizon", ["3M", "6M", "12M"], index=2, horizontal=True, key="market_cycle_analog_horizon")
    with dist_col_1:
        st.plotly_chart(build_return_distribution_fig(snapshot.analogs, horizon), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with dist_col_2:
        st.plotly_chart(build_drawdown_distribution_fig(snapshot.analogs, horizon), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    render_analog_details_table(snapshot.analogs)

    st.markdown("### Market State Validation")
    val_col, quality_col = st.columns(2)
    with val_col:
        st.dataframe(style_percent_table(snapshot.category_validation), use_container_width=True, hide_index=True)
    with quality_col:
        st.dataframe(snapshot.data_quality, use_container_width=True, hide_index=True)

    st.markdown("### Interpretation")
    st.markdown(
        f"<div style='white-space:pre-wrap; color:#cbd5e1; line-height:1.45; font-size:0.9rem;'>{html.escape(snapshot.interpretation)}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Historical analog output is a conditional historical distribution, not an investment recommendation.")
    render_market_cycle_methodology()


def render_summary_cards(current: dict[str, Any], spy_macro_outlook: SPYMacroOutlook | None = None) -> None:
    cards = build_top_level_analytics(current, spy_macro_outlook)
    cols = st.columns(len(cards))
    for idx, (title, rows) in enumerate(cards):
        with cols[idx]:
            render_top_level_panel(title, rows)


def build_top_level_analytics(current: dict[str, Any], spy_macro_outlook: SPYMacroOutlook | None = None) -> list[tuple[str, list[tuple[str, str]]]]:
    cards = [
        (
            "Structural Market Cycle",
            [
                ("SMA 200M Extension", extension_summary(current.get("StructuralExtensionZone"), current.get("StructuralExtensionPercentile"))),
                ("36M ROC 3MMA", pct(current.get("SPX_ROC36M_3MMA"))),
                ("ROC Momentum 12M", pct(current.get("StructuralROCMomentum_12M"))),
                ("Structural Market Cycle Maturity", maturity_pct(current.get("StructuralMaturityPct"))),
            ],
        ),
        (
            "Medium Term Market Cycle",
            [
                ("SMA 200W Extension", extension_summary(current.get("SMA200WZone"), current.get("SMA200WExtensionPercentile"))),
                ("12M ROC 3MMA", pct(current.get("SPX_ROC12M_3MMA"))),
                ("ROC Momentum 3M", pct(current.get("SPX_ROC_Momentum_3M"))),
                ("LONG EXTENSION CYCLE MATURITY", maturity_pct(current.get("LongCycleMaturityPct"))),
                ("PRIMARY MARKET CYCLE MATURITY", maturity_pct(current.get("PrimaryCycleMaturityPct"))),
            ],
        ),
        (
            "Performance",
            [
                ("ROC 1M", pct(current.get("PerformanceROC1M"))),
                ("ROC 3M", pct(current.get("PerformanceROC3M"))),
                ("ROC 6M", pct(current.get("PerformanceROC6M"))),
                ("ROC 12M", pct(current.get("PerformanceROC12M"))),
            ],
        ),
        (
            "Current Risk",
            [
                ("Status", text(current.get("CurrentMarketRiskState"))),
                ("Breadth Risk", classify_component_state(current.get("CurrentRiskBreadthRisk"))),
                ("RSI Divergence Risk", classify_component_state(current.get("CurrentRiskRSIDivergenceRisk"))),
                ("VIX Risk", classify_component_state(current.get("CurrentRiskVIXRisk"))),
                ("High Beta Risk", classify_component_state(current.get("CurrentRiskHighBetaRisk"))),
                ("High Yield Risk", classify_component_state(current.get("CurrentRiskHYRisk"))),
            ],
        ),
    ]
    if spy_macro_outlook is not None:
        macro_rows: list[tuple[str, str]] = []
        for horizon in SPY_MACRO_HORIZONS:
            score = spy_macro_outlook.current.get(f"Macro_{horizon}")
            macro_rows.append(
                (
                    f"{horizon} State",
                    f"{score_state(score)} ({_spy_macro_value(score)}), {score_direction(spy_macro_outlook, horizon)}",
                )
            )
        macro_rows.append(("Financial Transmission", _spy_macro_value(spy_macro_outlook.current.get("Transmission_Score"), " / 100")))
        cards.append(("Macro Outlook", macro_rows))
    return cards


def render_top_level_panel(title: str, rows: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<div style="display:grid;grid-template-columns:auto auto;justify-content:start;column-gap:0.75rem;align-items:baseline;">'
        f'<span style="color:#cbd5e1;">{html.escape(label)}</span>'
        f'<b style="color:#f8fafc;white-space:nowrap;">{html.escape(value)}</b></div>'
        for label, value in rows
    )
    st.markdown(
        f"""
<div style="padding:0.55rem 0 0.9rem;line-height:1.35;">
  <div style="font-size:1.3rem;color:#f8fafc;font-weight:900;margin-bottom:0.35rem;">{html.escape(title)}</div>
  <div style="display:grid;gap:0.18rem;font-size:1.06rem;">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def extension_summary(zone: Any, percentile: Any) -> str:
    label = text(zone)
    if label != "N/A":
        label = label.replace("_", " ").title()
    pctl = num(percentile)
    return label if pctl == "N/A" else f"{label} ({pctl})"


def render_card(title: str, headline: str, rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<div><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>" for k, v in rows)
    st.markdown(
        f"""
<div style="padding:0.65rem 0; min-height:9.2rem; line-height:1.15;">
  <div style="font-size:0.68rem; color:#94a3b8; font-weight:800;">{html.escape(title)}</div>
  <div style="font-size:0.96rem; color:#f8fafc; font-weight:900; margin:0.25rem 0 0.45rem;">{html.escape(headline)}</div>
  <div style="display:grid; grid-template-columns:1fr auto; gap:0.22rem 0.55rem; font-size:0.70rem; color:#cbd5e1;">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_market_cycle_methodology() -> None:
    st.markdown("### Market Cycle Methodology")
    st.caption("Definitions below describe the regime strips used in Market Cycle. They are descriptive state labels, not deterministic timing signals.")

    sma_rows = [
        ("EXTREME OVERSOLD", "Extension percentile < 10"),
        ("OVERSOLD", "10 <= extension percentile < 25"),
        ("NORMAL", "25 <= extension percentile < 75"),
        ("EXTENDED", "75 <= extension percentile < 90"),
        ("OVEREXTENDED", "90 <= extension percentile < 97.5"),
        ("EXTREME OVEREXTENSION", "Extension percentile >= 97.5"),
    ]
    st.markdown("#### SMA Extension Phase")
    st.dataframe(pd.DataFrame(sma_rows, columns=["Phase", "Rule"]), use_container_width=True, hide_index=True, height=250)
    st.caption("SMA200W Extension uses SPX/SMA200W percentile. SMA200M Extension uses SPX/SMA200M percentile with the same phase cutoffs.")

    momentum_rows = [
        ("RISK EXPANSION", "12M ROC 3MMA > 0 and 3M ROC momentum > 0"),
        ("LATE RISK EXPANSION", "12M ROC 3MMA > 0 and 3M ROC momentum < 0"),
        ("RISK CONTRACTION", "12M ROC 3MMA < 0 and 3M ROC momentum < 0"),
        ("EARLY RISK RECOVERY", "12M ROC 3MMA < 0 and 3M ROC momentum > 0"),
        ("TRANSITION", "ROC or momentum is exactly neutral"),
    ]
    structural_momentum_rows = [
        ("RISK EXPANSION", "36M ROC 3MMA > 0 and 12M ROC momentum > 0"),
        ("LATE RISK EXPANSION", "36M ROC 3MMA > 0 and 12M ROC momentum < 0"),
        ("RISK CONTRACTION", "36M ROC 3MMA < 0 and 12M ROC momentum < 0"),
        ("EARLY RISK RECOVERY", "36M ROC 3MMA < 0 and 12M ROC momentum > 0"),
        ("TRANSITION", "ROC or momentum is exactly neutral"),
    ]
    momentum_col, structural_momentum_col = st.columns(2)
    with momentum_col:
        st.markdown("#### Momentum Cycle Phase")
        st.dataframe(pd.DataFrame(momentum_rows, columns=["Phase", "Rule"]), use_container_width=True, hide_index=True, height=220)
    with structural_momentum_col:
        st.markdown("#### SPX Structural Momentum Cycle")
        st.dataframe(pd.DataFrame(structural_momentum_rows, columns=["Phase", "Rule"]), use_container_width=True, hide_index=True, height=220)
    st.caption("Medium-term momentum uses 12M ROC smoothed by 3MMA and 3M momentum lookback. Structural momentum uses 36M ROC smoothed by 3MMA and 12M momentum lookback.")

    sma_roc_rows = [
        ("EARLY RECOVERY", "Momentum = EARLY RISK RECOVERY and SMA extension is oversold/normal"),
        ("EXPANSION", "Momentum = RISK EXPANSION without stretched extension"),
        ("STRETCHED EXPANSION", "Momentum = RISK EXPANSION and SMA200W percentile >= 75"),
        ("LATE EXPANSION", "Momentum = LATE RISK EXPANSION"),
        ("CONTRACTION", "Momentum = RISK CONTRACTION"),
        ("CAPITULATION / OVERSOLD", "Momentum = RISK CONTRACTION and SMA200W percentile <= 25"),
    ]
    structural_sma_roc_rows = [
        ("EARLY RECOVERY", "SMA200M percentile <= 25 or extension <= 0, structural ROC momentum accelerating and positive"),
        ("EARLY EXPANSION", "SMA200M percentile <= 55 and structural ROC momentum accelerating"),
        ("EXPANSION", "Default positive structural SMA+ROC regime"),
        ("MATURE EXPANSION", "SMA200M percentile >= 75 and ROC positive or not decelerating"),
        ("LATE EXPANSION", "SMA200M percentile >= 90 and structural ROC momentum decelerating"),
        ("CONTRACTION", "Extension weak/negative with decelerating or negative structural ROC"),
    ]
    st.markdown("#### SMA+ROC Phase")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame(sma_roc_rows, columns=["Medium-Term Phase", "Rule"]), use_container_width=True, hide_index=True, height=260)
    with c2:
        st.dataframe(pd.DataFrame(structural_sma_roc_rows, columns=["Structural Phase", "Rule"]), use_container_width=True, hide_index=True, height=260)

    structural_rows = [
        ("STRUCTURAL CONTRACTION", "SPX/SMA200M < -12% and 12M extension direction < 0"),
        ("EARLY STRUCTURAL EXPANSION", "SMA200M percentile <= 45, structural ROC momentum accelerating, and cycle age < 45% of historical median"),
        ("MID STRUCTURAL EXPANSION", "Default structural expansion state"),
        ("MATURE STRUCTURAL EXPANSION", "SMA200M percentile >= 75 or cycle age >= 55% of historical median"),
        ("LATE STRUCTURAL EXPANSION", "SMA200M percentile >= 90 and cycle age >= 65% of historical median"),
        ("STRUCTURAL TOP ZONE", "SMA200M percentile >= 97.5, cycle age >= 75%, and structural ROC momentum decelerating"),
    ]
    st.markdown("#### Structural Market Cycle Phase")
    st.dataframe(pd.DataFrame(structural_rows, columns=["Phase", "Rule"]), use_container_width=True, hide_index=True, height=260)

    risk_rows = [
        ("NORMAL", "VIX Risk Activation is false"),
        ("LOW CONFIRMATION", "Risk Activation is true and Escalation Score V2 is 0-1"),
        ("MODERATE", "Risk Activation is true and Escalation Score V2 equals 2"),
        ("HIGH RISK", "Risk Activation is true and Escalation Score V2 equals 3"),
        ("RED FLAG", "Risk Activation is true and Escalation Score V2 is 4-6; QQQ and credit confirmations refine the marker"),
    ]
    st.markdown("#### Current Risk State")
    st.dataframe(pd.DataFrame(risk_rows, columns=["State", "Rule"]), use_container_width=True, hide_index=True, height=240)


def render_outlook_table(outlook: pd.DataFrame) -> None:
    if outlook is None or outlook.empty:
        st.info("Historical outlook is unavailable.")
        return
    metrics = [
        "Median Forward Return",
        "Mean Forward Return",
        "Positive Return Probability",
        "P25 Return",
        "P75 Return",
        "Risk of >15% Drawdown",
        "Risk of >25% Drawdown",
        "Risk of >45% Drawdown",
        "Independent Analog N",
        "Raw Candidate N",
        "Average Similarity",
        "Average Coverage",
        "Concentration Warning",
        "Confidence",
    ]
    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for _, source in outlook.iterrows():
            value = source.get(metric)
            if metric in {"Independent Analog N", "Raw Candidate N"}:
                row[source["Horizon"]] = "" if pd.isna(value) else f"{int(value)}"
            elif metric in {"Confidence", "Concentration Warning"}:
                row[source["Horizon"]] = text(value)
            elif metric in {"Average Similarity", "Average Coverage"}:
                row[source["Horizon"]] = "" if pd.isna(value) else f"{float(value):.0%}" if float(value) <= 1 else f"{float(value):.0f}%"
            else:
                row[source["Horizon"]] = pct(value)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)


def cycle_state(cycle_value: Any, direction: Any) -> str:
    """Classify one normalized cycle without changing the underlying cycle series."""
    cycle = safe_float(cycle_value)
    change = safe_float(direction)
    if not np.isfinite(cycle) or not np.isfinite(change):
        # Keep the combined layer total: incomplete early observations are a
        # constructive fallback and never create an unshaded interval.
        return "EXPANSION"
    if cycle >= 0 and change > 0:
        return "EXPANSION"
    if cycle >= 0 and change <= 0:
        return "LATE / PEAK"
    if cycle < 0 and change <= 0:
        return "CONTRACTION"
    return "RECOVERY"


def combined_cycle_risk_regime(long_state: str, primary_state: str) -> str:
    """Map the two internal cycle states to one of five visible regimes."""
    if long_state == "CONTRACTION":
        if primary_state == "CONTRACTION":
            return "MAXIMUM DRAWDOWN RISK"
        if primary_state == "RECOVERY":
            return "HIGH VOLATILITY / BOTTOMING"
        if primary_state == "LATE / PEAK":
            return "MAXIMUM DRAWDOWN RISK"
    if long_state == "LATE / PEAK" and primary_state in {"LATE / PEAK", "CONTRACTION"}:
        return "HIGH CORRECTION RISK"
    if long_state == "EXPANSION" and primary_state in {"LATE / PEAK", "CONTRACTION"}:
        return "SHORT-CYCLE CORRECTION RISK"
    return "RISK ON"


def add_combined_cycle_risk_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Add validation fields for the background regime using displayed cycles."""
    if frame.empty:
        return frame.copy()
    enriched = frame.copy()
    primary_display = normalize_centered_full(enriched["PrimaryMarketCycle"])
    long_display = normalize_centered_full(enriched["LongMarketExtensionCycle"])
    enriched["PrimaryCycle3MChange"] = primary_display.diff(3)
    enriched["LongCycle3MChange"] = long_display.diff(3)
    enriched["PrimaryCycleState"] = [
        cycle_state(value, change)
        for value, change in zip(primary_display, enriched["PrimaryCycle3MChange"], strict=False)
    ]
    enriched["LongCycleState"] = [
        cycle_state(value, change)
        for value, change in zip(long_display, enriched["LongCycle3MChange"], strict=False)
    ]
    enriched["CombinedCycleRiskRegime"] = [
        combined_cycle_risk_regime(long_state, primary_state)
        for long_state, primary_state in zip(enriched["LongCycleState"], enriched["PrimaryCycleState"], strict=False)
    ]
    return enriched


def add_combined_cycle_risk_background(fig: go.Figure, frame: pd.DataFrame) -> None:
    """Shade contiguous monthly regimes behind only the SPX Log subplot."""
    if frame.empty or "CombinedCycleRiskRegime" not in frame:
        return
    dates = pd.to_datetime(frame["Date"], errors="coerce").reset_index(drop=True)
    regimes = frame["CombinedCycleRiskRegime"].astype(str).reset_index(drop=True)
    if dates.dropna().empty:
        return
    first_axis_domain = fig.layout.yaxis.domain or [0.0, 1.0]
    start_idx = 0
    for idx in range(1, len(frame) + 1):
        if idx == len(frame) or regimes.iloc[idx] != regimes.iloc[start_idx]:
            x0 = dates.iloc[start_idx]
            if idx < len(frame):
                x1 = dates.iloc[idx]
            else:
                x1 = dates.iloc[idx - 1] + pd.offsets.MonthEnd(1)
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=first_axis_domain[0],
                y1=first_axis_domain[1],
                fillcolor=CYCLE_RISK_REGIME_COLORS.get(regimes.iloc[start_idx], "#16A34A"),
                opacity=0.27,
                line={"width": 0},
                layer="below",
            )
            start_idx = idx


def add_cycle_risk_regime_legend_traces(fig: go.Figure) -> None:
    """Provide the exact five-item regime legend for the multi-layer chart."""
    for rank, regime in enumerate(CYCLE_RISK_REGIME_ORDER, start=1):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=regime,
                marker={"symbol": "square", "size": 10, "color": CYCLE_RISK_REGIME_COLORS[regime]},
                hoverinfo="skip",
                showlegend=True,
                legendrank=rank,
            ),
            row=1,
            col=1,
        )


def build_multi_layer_market_cycles_fig(full_history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    full = add_combined_cycle_risk_regime(monthly_context_frame(full_history))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.19, 0.19, 0.20],
        vertical_spacing=0.035,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("SPX Log", "Primary Market Cycle (~41M)", "Long Extension Cycle (~80-95M)", "Structural Extension"),
    )
    if full.empty:
        return style_fig(fig, "SPX Multi-Layer Market Cycles<br><sup>SPX | ~41M Primary Cycle | ~80-95M Long Extension Cycle | Structural Extension</sup>", 760)
    full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
    visible = full.loc[full["Date"].between(start, end)].copy()
    if visible.empty:
        visible = full.tail(1).copy()

    primary_norm = normalize_centered_full(full["PrimaryMarketCycle"]).reindex(visible.index)
    long_norm = normalize_centered_full(full["LongMarketExtensionCycle"]).reindex(visible.index)

    add_combined_cycle_risk_background(fig, full)

    fig.add_trace(
        go.Scatter(
            x=visible["Date"],
            y=pd.to_numeric(visible["SPX_Close"], errors="coerce"),
            mode="lines",
            name="SPX",
            line={"color": "#f8fafc", "width": 1.9},
            showlegend=False,
            customdata=np.column_stack(
                [
                    visible["CombinedCycleRiskRegime"].astype(str),
                    visible["PrimaryCycleState"].astype(str),
                    visible["LongCycleState"].astype(str),
                ]
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>SPX Close: %{y:,.0f}<br>"
                "Cycle Risk Regime: %{customdata[0]}<br>"
                "Primary Cycle State: %{customdata[1]}<br>"
                "Long Cycle State: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=visible["Date"],
            y=primary_norm,
            mode="lines",
            name="~41M Primary Cycle",
            line={"color": "#38bdf8", "width": 1.8},
            showlegend=False,
            customdata=np.stack(
                [
                    pd.to_numeric(visible["PrimaryMarketCycle"], errors="coerce"),
                    visible["PrimaryCycleTrough"].astype(str),
                    pd.to_numeric(visible["PrimaryCycleMonthsSinceTrough"], errors="coerce"),
                    pd.to_numeric(visible["PrimaryCycleMaturityPct"], errors="coerce"),
                    visible["PrimaryCycleLastTroughDate"].map(date_text),
                    pd.to_numeric(visible["PrimaryCycleTroughToTroughMonths"], errors="coerce"),
                    visible["PrimaryCycleDirection"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Normalized cycle: %{y:.2f}<br>"
                "Cycle value: %{customdata[0]:.2f}<br>Trough: %{customdata[1]}<br>"
                "Last trough: %{customdata[3]}<br>Trough-to-trough: %{customdata[4]:.0f}M<br>"
                "Months since trough: %{customdata[2]:.0f}M (%{customdata[3]:.1f}%)<br>Direction: %{customdata[6]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    add_cycle_marker_trace(fig, visible, "Primary", "PrimaryMarketCycle", primary_norm, "#38bdf8", row=2)

    fig.add_trace(
        go.Scatter(
            x=visible["Date"],
            y=long_norm,
            mode="lines",
            name="~80-95M Long Extension Cycle",
            line={"color": "#facc15", "width": 1.8},
            showlegend=False,
            customdata=np.stack(
                [
                    pd.to_numeric(visible["LongMarketExtensionCycle"], errors="coerce"),
                    visible["LongCycleTrough"].astype(str),
                    pd.to_numeric(visible["LongCycleMonthsSinceTrough"], errors="coerce"),
                    pd.to_numeric(visible["LongCycleMaturityPct"], errors="coerce"),
                    visible["LongCycleLastTroughDate"].map(date_text),
                    pd.to_numeric(visible["LongCycleTroughToTroughMonths"], errors="coerce"),
                    visible["LongCycleDirection"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Normalized cycle: %{y:.2f}<br>"
                "Cycle value: %{customdata[0]:.2f}<br>Trough: %{customdata[1]}<br>"
                "Last trough: %{customdata[3]}<br>Trough-to-trough: %{customdata[4]:.0f}M<br>"
                "Months since trough: %{customdata[2]:.0f}M (%{customdata[3]:.1f}%)<br>Direction: %{customdata[6]}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )
    add_cycle_marker_trace(fig, visible, "Long", "LongMarketExtensionCycle", long_norm, "#facc15", row=3)

    fig.add_trace(
        go.Scatter(
            x=visible["Date"],
            y=pd.to_numeric(visible["StructuralExtensionSmooth"], errors="coerce"),
            mode="lines",
            name="Structural Extension",
            line={"color": "#f8fafc", "width": 1.9},
            showlegend=False,
            customdata=np.stack(
                [
                    pd.to_numeric(visible["StructuralExtensionPct"], errors="coerce") * 100,
                    visible["StructuralTrough"].astype(str),
                    visible["StructuralPeak"].astype(str),
                    visible["StructuralLastTroughDate"].map(date_text),
                    pd.to_numeric(visible["StructuralMonthsSinceTrough"], errors="coerce"),
                    pd.to_numeric(visible["StructuralMaturityPct"], errors="coerce"),
                    pd.to_numeric(visible["StructuralReferenceCycle"], errors="coerce"),
                    pd.to_numeric(visible["StructuralReferenceProgressPct"], errors="coerce"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Structural Extension 3MMA: %{y:.1f}%<br>"
                "Raw SPX/SMA200M: %{customdata[0]:.1f}%<br>"
                "Structural trough: %{customdata[1]}<br>Structural peak: %{customdata[2]}<br>"
                "Last structural trough: %{customdata[3]}<br>Months since trough: %{customdata[4]:.0f}M (%{customdata[5]:.1f}%)<br>"
                "Reference cycle: %{customdata[6]:.1f}<br>Reference progress: %{customdata[7]:.1f}%<extra></extra>"
            ),
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=visible["Date"],
            y=pd.to_numeric(visible["StructuralReferenceCycle"], errors="coerce"),
            mode="lines",
            name="Structural Reference 0-100",
            line={"color": "#a78bfa", "width": 1.5, "dash": "dash"},
            showlegend=False,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Reference cycle: %{y:.1f}<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    add_structural_marker_trace(fig, visible, mode="trough")
    add_structural_marker_trace(fig, visible, mode="peak")

    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1}, row=2, col=1)
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1}, row=3, col=1)
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot", "width": 1}, row=4, col=1)
    add_cycle_risk_regime_legend_traces(fig)
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(title_text="Normalized", row=2, col=1)
    fig.update_yaxes(title_text="Normalized", row=3, col=1)
    fig.update_yaxes(title_text="Extension %", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Reference 0-100", row=4, col=1, secondary_y=True, range=[0, 105], showgrid=False)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#94a3b8", spikethickness=1)
    return style_fig(fig, "SPX Multi-Layer Market Cycles<br><sup>SPX | ~41M Primary Cycle | ~80-95M Long Extension Cycle | Structural Extension</sup>", 760)


def render_multi_layer_cycle_summary(current: dict[str, Any]) -> None:
    rows = [
        ("PRIMARY MARKET CYCLE", ""),
        ("Historical center", "~40-42M"),
        ("Months since last trough", f"{num(current.get('PrimaryCycleMonthsSinceTrough'), 0)} ({maturity_pct(current.get('PrimaryCycleMaturityPct'))})"),
        ("Current direction", text(current.get("PrimaryCycleDirection"))),
        ("LONG EXTENSION CYCLE", ""),
        ("Historical center", "~85-86M"),
        ("Months since last trough", f"{num(current.get('LongCycleMonthsSinceTrough'), 0)} ({maturity_pct(current.get('LongCycleMaturityPct'))})"),
        ("Current direction", text(current.get("LongCycleDirection"))),
        ("STRUCTURAL CYCLE", ""),
        ("Actual SPX vs SMA200M", pct(current.get("StructuralExtensionPct"))),
        ("Months since last structural trough", f"{num(current.get('StructuralMonthsSinceTrough'), 0)} ({maturity_pct(current.get('StructuralMaturityPct'))})"),
        ("Reference trough-to-peak", "~291M"),
        ("Reference full cycle", "~461M"),
        ("Reference cycle progress", f"{num(current.get('StructuralReferenceProgressPct'))}%"),
        ("Structural state", text(current.get("StructuralMarketCyclePhase"))),
    ]
    html_rows = []
    for label, value in rows:
        if not value:
            html_rows.append(f"<div style='color:#93c5fd;font-size:0.72rem;font-weight:900;margin-top:0.45rem'>{html.escape(label)}</div>")
        else:
            html_rows.append(
                f"<div style='border-bottom:1px solid #263241;padding:0.32rem 0;'>"
                f"<div style='color:#94a3b8;font-size:0.70rem;font-weight:700'>{html.escape(label)}</div>"
                f"<div style='font-size:0.92rem;font-weight:800'>{html.escape(value)}</div></div>"
            )
    st.markdown(f"<div style='background:#0f131a;border:1px solid #263241;border-radius:6px;padding:0.65rem'>{''.join(html_rows)}</div>", unsafe_allow_html=True)


def add_cycle_marker_trace(fig: go.Figure, d: pd.DataFrame, prefix: str, cycle_col: str, normalized: pd.Series, color: str, row: int) -> None:
    flag_col = f"{prefix}CycleTrough"
    if flag_col not in d:
        return
    flags = d[flag_col].map(is_true_flag)
    markers = d.loc[flags].copy()
    if markers.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=markers["Date"],
            y=normalized.loc[markers.index],
            mode="markers",
            name=f"{prefix} cycle trough",
            showlegend=False,
            marker={"color": color, "size": 7, "line": {"color": "#0f131a", "width": 1}},
            customdata=np.stack(
                [
                    markers[f"{prefix}CycleLastTroughDate"].map(date_text),
                    markers[f"{prefix}CyclePreviousTroughDate"].map(date_text),
                    pd.to_numeric(markers[f"{prefix}CycleTroughToTroughMonths"], errors="coerce"),
                    pd.to_numeric(markers[cycle_col], errors="coerce"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Trough date: %{customdata[0]}<br>Previous trough: %{customdata[1]}<br>"
                "Trough-to-trough: %{customdata[2]:.0f}M<br>Cycle value: %{customdata[3]:.2f}<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def add_structural_marker_trace(fig: go.Figure, d: pd.DataFrame, mode: str) -> None:
    flag_col = "StructuralTrough" if mode == "trough" else "StructuralPeak"
    if flag_col not in d:
        return
    flags = d[flag_col].map(is_true_flag)
    markers = d.loc[flags].copy()
    if markers.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=markers["Date"],
            y=pd.to_numeric(markers["StructuralExtensionSmooth"], errors="coerce"),
            mode="markers",
            name="Structural trough" if mode == "trough" else "Structural peak",
            showlegend=False,
            marker={
                "symbol": "circle" if mode == "trough" else "square",
                "color": "#22c55e" if mode == "trough" else "#ef4444",
                "size": 10 if mode == "trough" else 8,
                "line": {"color": "#0f131a", "width": 1},
            },
            hovertemplate=("Structural " + mode + "<br>Date: %{x|%Y-%m-%d}<br>Extension: %{y:.1f}%<extra></extra>"),
        ),
        row=4,
        col=1,
        secondary_y=False,
    )


def build_structural_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.06)
    add_background_bands(fig, d, "StructuralMarketCyclePhase", row=1, opacity=0.12)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX_Close"], mode="lines", name="SPX", line={"color": "#f8fafc", "width": 1.7}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX_SMA200M"], mode="lines", name="SMA200M", line={"color": "#38bdf8", "width": 1.5}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["StructuralExtensionPct"], errors="coerce") * 100, mode="lines", name="SPX/SMA200M %", line={"color": "#facc15", "width": 1.6}), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["StructuralExtensionPercentile"], mode="lines", name="Percentile", line={"color": "#22c55e", "width": 1.3, "dash": "dot"}), row=2, col=1)
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(title_text="% / percentile", row=2, col=1)
    return style_fig(fig, "Structural Market Cycle", 520)


def build_structural_roc_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_background_bands(fig, d, "StructuralROCMomentumState", opacity=0.10)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["SPX_ROC36M"], errors="coerce") * 100, mode="lines", name="ROC36M", line={"color": "#38bdf8", "width": 1.4}))
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["SPX_ROC36M_3MMA"], errors="coerce") * 100, mode="lines", name="ROC36M 3MMA", line={"color": "#f8fafc", "width": 1.8}))
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["StructuralROCMomentum"], errors="coerce") * 100, mode="lines", name="Structural ROC Momentum", line={"color": "#facc15", "width": 1.5}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    return style_fig(fig, "Structural ROC Momentum", 320)


def build_structural_momentum_cycle_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.36, 0.12],
        vertical_spacing=0.045,
        subplot_titles=("SPX", "ROC", "SPX Structural Momentum Cycle Phase"),
    )
    add_background_bands(fig, d, "StructuralMomentumCyclePhase", row=1, opacity=0.14)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX_Close"], mode="lines", name="SPX", line={"color": "#f8fafc", "width": 1.6}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["SPX_ROC36M_3MMA"], errors="coerce") * 100, mode="lines", name="36M ROC 3MMA", line={"color": "#38bdf8", "width": 1.7}), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["StructuralROCMomentum"], errors="coerce") * 100, mode="lines", name="ROC Momentum 12M", line={"color": "#facc15", "width": 1.5}), row=2, col=1)
    fig.add_trace(
        build_state_strip_trace(
            d,
            "StructuralMomentumCyclePhase",
            [
                "EARLY RISK RECOVERY",
                "RISK EXPANSION",
                "LATE RISK EXPANSION",
                "RISK CONTRACTION",
                "TRANSITION",
            ],
            "Structural Momentum Cycle",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1}, row=2, col=1)
    fig.update_yaxes(title_text="SPX", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    return style_fig(fig, "SPX Structural Momentum Cycle", 430)


def build_structural_sync_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.64, 0.12, 0.12, 0.12],
        vertical_spacing=0.035,
        subplot_titles=("SPX Log", "Structural SMA+ROC Phase", "Structural Market Cycle Phase", "Structural ROC Momentum State"),
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["SPX_Close"],
            mode="lines",
            name="SPX",
            line={"color": "#f8fafc", "width": 1.8},
            customdata=np.stack(
                [
                    d["StructuralMarketCyclePhase"].astype(str),
                    d.get("Structural_SMA_ROC_Phase", pd.Series("DATA INCOMPLETE", index=d.index)).astype(str),
                    d["StructuralROCMomentumState"].astype(str),
                    pd.to_numeric(d["StructuralExtensionPct"], errors="coerce") * 100,
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<br>"
                "Structural Phase: %{customdata[0]}<br>"
                "SMA+ROC Phase: %{customdata[1]}<br>"
                "ROC State: %{customdata[2]}<br>"
                "SPX/SMA200M: %{customdata[3]:.1f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "Structural_SMA_ROC_Phase",
            [
                "EARLY RECOVERY",
                "EARLY EXPANSION",
                "EXPANSION",
                "MATURE EXPANSION",
                "LATE EXPANSION",
                "CONTRACTION",
            ],
            "SMA+ROC",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "StructuralMarketCyclePhase",
            [
                "EARLY STRUCTURAL EXPANSION",
                "MID STRUCTURAL EXPANSION",
                "MATURE STRUCTURAL EXPANSION",
                "LATE STRUCTURAL EXPANSION",
                "STRUCTURAL TOP ZONE",
                "STRUCTURAL CONTRACTION",
            ],
            "Structural",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "StructuralROCMomentumState",
            ["ACCELERATING", "STABLE", "DECELERATING"],
            "ROC State",
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=4, col=1)
    return style_fig(fig, "Structural Synchronized View", 560)


def build_momentum_cycle_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.36, 0.12],
        vertical_spacing=0.045,
        subplot_titles=("SPX", "ROC", "SPX Momentum Cycle Phase"),
    )
    add_background_bands(fig, d, "MomentumCyclePhase", row=1, opacity=0.14)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX_Close"], mode="lines", name="SPX", line={"color": "#f8fafc", "width": 1.6}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["SPX_ROC12M_3MMA"], errors="coerce") * 100, mode="lines", name="12M ROC 3MMA", line={"color": "#38bdf8", "width": 1.7}), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pd.to_numeric(d["SPX_ROC_Momentum_3M"], errors="coerce") * 100, mode="lines", name="ROC Momentum 3M", line={"color": "#facc15", "width": 1.5}), row=2, col=1)
    fig.add_trace(
        build_state_strip_trace(
            d,
            "MomentumCyclePhase",
            [
                "EARLY RISK RECOVERY",
                "RISK EXPANSION",
                "LATE RISK EXPANSION",
                "RISK CONTRACTION",
            ],
            "Momentum",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1}, row=2, col=1)
    fig.update_yaxes(title_text="SPX", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    return style_fig(fig, "SPX Momentum Cycle", 430)


def build_sma200w_extension_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y0, y1, color, label in [
        (0, 10, "#38bdf8", "Extreme Oversold"),
        (10, 25, "#60a5fa", "Oversold"),
        (25, 75, "#64748b", "Normal"),
        (75, 90, "#facc15", "Extended"),
        (90, 97.5, "#f97316", "Overextended"),
        (97.5, 100, "#ef4444", "Extreme"),
    ]:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=0.28,
            line_width=0,
            annotation_text=label,
            annotation_position="left",
            annotation_font={"color": "#f8fafc", "size": 10},
        )
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SMA200WExtensionPercentile"], mode="lines", name="SMA200W percentile", line={"color": "#f8fafc", "width": 1.9}, customdata=np.stack([pd.to_numeric(d["SMA200WExtensionPct"], errors="coerce") * 100, d["SMA200WZone"].astype(str)], axis=-1), hovertemplate="Date: %{x|%Y-%m-%d}<br>Percentile: %{y:.1f}<br>Extension: %{customdata[0]:.1f}%<br>Zone: %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(range=[0, 100], title="0-100")
    return style_fig(fig, "SMA200W Extension", 390)


def build_sma200m_extension_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y0, y1, color, label in [
        (0, 10, "#38bdf8", "Extreme Oversold"),
        (10, 25, "#60a5fa", "Oversold"),
        (25, 75, "#64748b", "Normal"),
        (75, 90, "#facc15", "Extended"),
        (90, 97.5, "#f97316", "Overextended"),
        (97.5, 100, "#ef4444", "Extreme"),
    ]:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            opacity=0.28,
            line_width=0,
            annotation_text=label,
            annotation_position="left",
            annotation_font={"color": "#f8fafc", "size": 10},
        )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=pd.to_numeric(d["StructuralExtensionPercentile"], errors="coerce"),
            mode="lines",
            name="SMA200M percentile",
            line={"color": "#f8fafc", "width": 1.9},
            customdata=np.stack(
                [
                    pd.to_numeric(d["StructuralExtensionPct"], errors="coerce") * 100,
                    d.get("StructuralMarketCyclePhase", pd.Series("DATA INCOMPLETE", index=d.index)).astype(str),
                ],
                axis=-1,
            ),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Percentile: %{y:.1f}<br>Extension: %{customdata[0]:.1f}%<br>Structural Phase: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 100], title="0-100")
    return style_fig(fig, "SMA200M Extension", 390)


def render_maturity_status(current: dict[str, Any], prefix: str, label: str, reference_length: str) -> None:
    cols = st.columns(6)
    elapsed = safe_float(current.get(f"{prefix}CycleTimeProgressPct"))
    ref_value = safe_float(current.get(f"{prefix}ReferenceCycleValue"))
    actual = safe_float(current.get(f"{prefix}ActualMaturity"))
    gap = safe_float(current.get(f"{prefix}CycleTimingGap"))
    items = [
        ("Elapsed Time", f"{elapsed:.1f}%" if np.isfinite(elapsed) else "N/A"),
        ("Reference Length", reference_length),
        ("Reference Value", f"{ref_value:.1f}" if np.isfinite(ref_value) else "N/A"),
        ("Actual Maturity", f"{actual:.1f}" if np.isfinite(actual) else "N/A"),
        ("Timing Gap", f"{gap:+.1f}" if np.isfinite(gap) else "N/A"),
        ("Status", text(current.get(f"{prefix}MaturityStatus"))),
    ]
    for col, (name, value) in zip(cols, items):
        with col:
            st.markdown(f"<div style='color:#93c5fd;font-size:0.72rem;font-weight:700'>{html.escape(label)} {html.escape(name)}</div><div style='font-weight:800;font-size:1rem'>{html.escape(value)}</div>", unsafe_allow_html=True)


def build_maturity_fig(
    d: pd.DataFrame,
    prefix: str,
    cycle_start: pd.Timestamp,
    reference_months: int,
    title: str,
    subtitle: str,
    actual_label: str,
) -> go.Figure:
    actual = d.copy()
    actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")
    actual = actual.loc[actual["Date"].ge(cycle_start)].copy()
    cycle_end = cycle_start + pd.DateOffset(months=reference_months)
    latest_actual = actual["Date"].max() if not actual.empty else pd.NaT
    ref_dates = pd.date_range(cycle_start, cycle_end, freq="W-FRI")
    elapsed_months = (ref_dates - cycle_start).days / 30.4375
    ref_values = 100.0 * np.sin(np.pi * elapsed_months / float(reference_months))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ref_dates,
            y=ref_values,
            mode="lines",
            name="Reference cycle",
            line={"color": "#94a3b8", "width": 1.8, "dash": "dash"},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Reference cycle: %{y:.1f}<extra></extra>",
        )
    )
    actual_col = f"{prefix}ActualMaturity"
    raw_col = "StructuralExtensionPct" if prefix == "Structural" else "SMA200WExtensionPct"
    ma_col = "SPX_SMA200M" if prefix == "Structural" else "SPX_SMA200W"
    if actual_col in actual.columns:
        fig.add_trace(
            go.Scatter(
                x=actual["Date"],
                y=pd.to_numeric(actual[actual_col], errors="coerce"),
                mode="lines",
                name=actual_label,
                line={"color": "#38bdf8", "width": 2.2},
                customdata=np.stack(
                    [
                        pd.to_numeric(actual["SPX_Close"], errors="coerce"),
                        pd.to_numeric(actual.get(ma_col, pd.Series(np.nan, index=actual.index)), errors="coerce"),
                        pd.to_numeric(actual.get(raw_col, pd.Series(np.nan, index=actual.index)), errors="coerce") * 100,
                        pd.to_numeric(actual.get(f"{prefix}ReferenceCycleValue", pd.Series(np.nan, index=actual.index)), errors="coerce"),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Actual percentile: %{y:.1f}<br>"
                    "SPX: %{customdata[0]:,.0f}<br>MA: %{customdata[1]:,.0f}<br>"
                    "Raw extension: %{customdata[2]:.1f}%<br>Reference: %{customdata[3]:.1f}<extra></extra>"
                ),
            )
        )
    add_vertical_marker(fig, cycle_start, "Cycle Start", "#22c55e", y=1.02, xanchor="left")
    add_vertical_marker(fig, cycle_end, "Reference Cycle End", "#f97316", y=1.02, xanchor="right")
    if pd.notna(latest_actual):
        add_vertical_marker(fig, latest_actual, "Latest Actual", "#38bdf8", y=0.02, xanchor="right")
    fig.update_yaxes(range=[0, 105], title="0-100")
    fig.update_xaxes(range=[cycle_start, max(cycle_end, latest_actual) if pd.notna(latest_actual) else cycle_end])
    styled = style_fig(fig, title, 390)
    styled.update_layout(title={"text": f"{title}<br><sup>{subtitle}</sup>"})
    return styled


def add_vertical_marker(fig: go.Figure, x: Any, label: str, color: str, y: float, xanchor: str) -> None:
    marker_x = pd.Timestamp(x).strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=marker_x,
        x1=marker_x,
        y0=0,
        y1=1,
        line={"color": color, "dash": "dot", "width": 1},
    )
    fig.add_annotation(
        x=marker_x,
        y=y,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        xanchor=xanchor,
        yanchor="bottom" if y >= 1 else "top",
        font={"color": color, "size": 10},
    )


def build_medium_term_sync_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.64, 0.12, 0.12, 0.12],
        vertical_spacing=0.035,
        subplot_titles=("SPX Log", "Medium-Term SMA+ROC Phase", "SMA200W Extension Phase", "SPX Momentum Cycle Phase"),
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["SPX_Close"],
            mode="lines",
            name="SPX",
            line={"color": "#f8fafc", "width": 1.8},
            customdata=np.stack(
                [
                    d["SMA200WZone"].astype(str),
                    d["MomentumCyclePhase"].astype(str),
                    d.get("MediumTerm_SMA_ROC_Phase", pd.Series("DATA INCOMPLETE", index=d.index)).astype(str),
                    pd.to_numeric(d["SMA200WExtensionPct"], errors="coerce") * 100,
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<br>"
                "SMA200W Zone: %{customdata[0]}<br>"
                "Momentum Phase: %{customdata[1]}<br>"
                "SMA+ROC Phase: %{customdata[2]}<br>"
                "SMA200W Extension: %{customdata[3]:.1f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "MediumTerm_SMA_ROC_Phase",
            [
                "EARLY RECOVERY",
                "EXPANSION",
                "STRETCHED EXPANSION",
                "LATE EXPANSION",
                "CONTRACTION",
                "CAPITULATION / OVERSOLD",
            ],
            "SMA+ROC",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "SMA200WZone",
            [
                "OVERSOLD",
                "NORMAL",
                "EXTENDED",
                "OVEREXTENDED",
                "EXTREME OVEREXTENSION",
            ],
            "SMA200W",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "MomentumCyclePhase",
            [
                "EARLY RISK RECOVERY",
                "RISK EXPANSION",
                "LATE RISK EXPANSION",
                "RISK CONTRACTION",
            ],
            "Momentum",
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=4, col=1)
    return style_fig(fig, "Medium-Term Synchronized View", 560)


def build_state_strip_trace(d: pd.DataFrame, state_col: str, domain: list[str], name: str) -> go.Heatmap:
    values = d[state_col].astype(str) if state_col in d.columns else pd.Series("DATA INCOMPLETE", index=d.index)
    if state_col == "SMA200WZone":
        values = values.replace({"EXTREME OVERSOLD": "OVERSOLD"})
    colors = strip_colors_for_domain(domain)
    names = domain + ["DATA INCOMPLETE"]
    code_lookup = {state: idx for idx, state in enumerate(names)}
    z = [code_lookup.get(value, code_lookup["DATA INCOMPLETE"]) for value in values]
    hover = [
        f"Date: {date:%Y-%m-%d}<br>{name}: {state}"
        for date, state in zip(pd.to_datetime(d["Date"], errors="coerce"), values)
    ]
    return go.Heatmap(
        x=d["Date"],
        y=[name],
        z=[z],
        colorscale=colors,
        zmin=0,
        zmax=max(len(names) - 1, 1),
        showscale=False,
        text=[hover],
        hovertemplate="%{text}<extra></extra>",
        name=name,
    )


def strip_colors_for_domain(domain: list[str]) -> list[list[Any]]:
    risk_domain = ["LOW", "NORMAL", "ELEVATED", "HIGH", "ACUTE"]
    risk_palette = {
        "LOW": "#00c853",
        "NORMAL": "#7ee787",
        "ELEVATED": "#facc15",
        "HIGH": "#f97316",
        "ACUTE": "#ef233c",
        "DATA INCOMPLETE": "#64748b",
    }
    if domain == risk_domain:
        names = domain + ["DATA INCOMPLETE"]
        denom = len(names)
        scale: list[list[Any]] = []
        for idx, name in enumerate(names):
            color = risk_palette.get(name, "#64748b")
            left = idx / denom
            right = (idx + 1) / denom
            scale.append([left, color])
            scale.append([right, color])
        scale[-1][0] = 1.0
        return scale

    palette = {
        "EXTREME OVERSOLD": "#38bdf8",
        "OVERSOLD": "#60a5fa",
        "NORMAL": "#94a3b8",
        "EXTENDED": "#facc15",
        "CROWDED": "#f97316",
        "EXTREME": "#ef4444",
        "OVEREXTENDED": "#f97316",
        "EXTREME OVEREXTENSION": "#ef4444",
        "EARLY RISK RECOVERY": "#38bdf8",
        "RISK EXPANSION": "#22c55e",
        "LATE RISK EXPANSION": "#facc15",
        "RISK CONTRACTION": "#ef4444",
        "EARLY RECOVERY": "#38bdf8",
        "EARLY EXPANSION": "#22c55e",
        "EXPANSION": "#16a34a",
        "MATURE EXPANSION": "#facc15",
        "LATE EXPANSION": "#f97316",
        "CONTRACTION": "#ef4444",
        "STRETCHED EXPANSION": "#f59e0b",
        "CAPITULATION / OVERSOLD": "#38bdf8",
        "EARLY STRUCTURAL EXPANSION": "#38bdf8",
        "MID STRUCTURAL EXPANSION": "#22c55e",
        "MATURE STRUCTURAL EXPANSION": "#facc15",
        "LATE STRUCTURAL EXPANSION": "#f97316",
        "STRUCTURAL TOP ZONE": "#ef4444",
        "STRUCTURAL CONTRACTION": "#a855f7",
        "ACCELERATING": "#22c55e",
        "STABLE": "#94a3b8",
        "DECELERATING": "#ef4444",
        "CORRECTION WATCH": "#facc15",
        "CORRECTION": "#f97316",
        "STRESS BUILDING": "#fb923c",
        "CAPITULATION": "#ef4444",
        "EARLY EXHAUSTION": "#38bdf8",
        "BOTTOMING": "#22c55e",
        "RE-ACCELERATION WATCH": "#a855f7",
        "RE-ACCELERATION": "#ef4444",
        "LOW UNDERCUT": "#e879f9",
        "CAPITULATION RECLAIM": "#10b981",
        "CONFIRMED BREAKDOWN": "#991b1b",
        "RECOVERY CONFIRMED": "#00c853",
        "NEW CORRECTION WAVE WATCH": "#f59e0b",
        "NEW CORRECTION WAVE": "#dc2626",
    }
    names = domain + ["DATA INCOMPLETE"]
    if len(names) == 1:
        color = palette.get(names[0], PHASE_COLORS.get(names[0], "#64748b"))
        return [[0, color], [1, color]]
    denom = len(names)
    scale: list[list[Any]] = []
    for idx, name in enumerate(names):
        color = palette.get(name, PHASE_COLORS.get(name, "#64748b"))
        left = idx / denom
        right = (idx + 1) / denom
        scale.append([left, color])
        scale.append([right, color])
    scale[-1][0] = 1.0
    return scale


def build_current_risk_components_fig(current: dict[str, Any]) -> go.Figure:
    rows = [
        ("Risk of >15% Drawdown", "CurrentRiskDrawdownRisk", f"Existing Market Cycle probability; N={num(current.get('SPX_1M_DD20_AnalogN'), 0)}"),
        ("Price Cycle Vulnerability Risk", "CurrentRiskPriceCycleVulnerabilityRisk", f"PVC MAX10D; current PVC={num(current.get('PVC_V2'))}"),
        ("Breadth Risk", "CurrentRiskBreadthRisk", f"S5FI={num(current.get('SPXAboveSMA50D'))}; 5D change={num(current.get('CurrentRiskBreadth5DChange'))}pp"),
        ("RSI Divergence Risk", "CurrentRiskRSIDivergenceRisk", f"Daily divergence={text(current.get('PVC_DailyRSIDivergence'))}; weekly={text(current.get('PVC_WeeklyRSIDivergence'))}"),
        ("VIX Risk", "CurrentRiskVIXRisk", f"VIX={num(current.get('VIX'))}; 3D change={pct(current.get('CurrentRiskVIX3DChange'))}"),
        ("High Beta Risk (QQQ)", "CurrentRiskHighBetaRisk", f"QQQ/SPY depth={pct(current.get('CurrentRiskQQQDivergenceDepth'))}"),
        ("HY Risk (BAMLH0A0HYM2)", "CurrentRiskHYRisk", f"HY OAS={num(current.get('HY_OAS'), 2)}; 5D change={num(current.get('CurrentRiskHY5DChange'), 2)}"),
    ]
    labels = [label for label, _, _ in rows]
    values = [safe_float(current.get(key)) for _, key, _ in rows]
    states = [risk_zone_state(value) for value in values]
    details = [detail for _, _, detail in rows]
    value_text = ["N/A" if not np.isfinite(v) else f"{v:.1f} | {state}" for v, state in zip(values, states)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": [risk_zone_color(state) for state in states]},
            customdata=np.stack([states, details], axis=-1),
            text=value_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Score: %{x:.1f}<br>State: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 120], title="Risk (0-100)")
    return style_fig(fig, f"Current Market Risk Components - {text(current.get('CurrentMarketRiskState'))}", 420)


def render_current_risk_signal_table(current: dict[str, Any]) -> None:
    rows = [
        ("Risk of >15% Drawdown", "CurrentRiskDrawdownRisk"),
        ("Price Cycle Vulnerability Risk", "CurrentRiskPriceCycleVulnerabilityRisk"),
        ("Breadth Risk", "CurrentRiskBreadthRisk"),
        ("RSI Divergence Risk", "CurrentRiskRSIDivergenceRisk"),
        ("VIX Risk", "CurrentRiskVIXRisk"),
        ("High Beta Risk (QQQ)", "CurrentRiskHighBetaRisk"),
        ("HY Risk (BAMLH0A0HYM2)", "CurrentRiskHYRisk"),
    ]
    table_rows = []
    for label, key in rows:
        value = safe_float(current.get(key))
        state = text(current.get(f"{key}State"))
        if state in {"N/A", "DATA INCOMPLETE"}:
            state = risk_zone_state(value)
        table_rows.append({"Signal": label, "State": state, "Value": value})
    table = pd.DataFrame(table_rows)
    st.dataframe(
        table.style.format({"Value": "{:.1f}"}, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
        height=300,
    )
    activation = "ACTIVE" if is_true_flag(current.get("CurrentRiskActivation")) else "OFF"
    st.caption(
        f"VIX Risk Activation: {activation} | Escalation Score V2: {num(current.get('CurrentRiskEscalationScoreV2'), 0)}/6 | "
        f"Model: {text(current.get('CurrentRiskModelVersion'))}"
    )


def build_current_risk_signals_fig(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    d = current_risk_history_frame(history, start, end)
    fig = go.Figure()
    if d.empty:
        return style_fig(fig, "Log SPY with Current Risk Signals", 560)
    spy_raw = d["CurrentRiskSPYRawClose"].combine_first(d["SPY_Close"])
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=spy_raw,
            mode="lines",
            name="SPY",
            line={"color": "#f8fafc", "width": 1.7},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SPY raw: %{y:,.2f}<extra></extra>",
        )
    )
    categories = [
        ("LOW CONFIRMATION", "#22c55e", "circle", 8),
        ("MODERATE", "#facc15", "diamond", 9),
        ("HIGH RISK", "#f97316", "triangle-up", 10),
        ("RED FLAG", "#fb7185", "x", 11),
        ("RED FLAG + TOP CONFIRMATION", "#ef4444", "star", 12),
        ("RED FLAG + CREDIT CONFIRMATION", "#7f1d1d", "star-diamond", 13),
    ]
    for category, color, symbol, size in categories:
        points = d[
            d["CurrentRiskNewEvent"].fillna(False).astype(bool)
            & d["CurrentRiskSignalClass"].eq(category)
        ].copy()
        if points.empty:
            continue
        custom = np.column_stack(
            [
                points["PVC_MAX10D"],
                points["CurrentRiskEscalationScoreV2"],
                points["VIX"],
                points["SPXAboveSMA50D"],
                points["HY_OAS"],
                points["CurrentRiskQQQSPY"],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=points["Date"],
                y=points["CurrentRiskSPYRawClose"].combine_first(points["SPY_Close"]),
                mode="markers",
                name=category,
                marker={"color": color, "symbol": symbol, "size": size, "line": {"color": "#0f131a", "width": 1}},
                customdata=custom,
                hovertemplate=(
                    f"{category}<br>Date: %{{x|%Y-%m-%d}}<br>SPY raw: %{{y:,.2f}}<br>"
                    "PVC MAX10D: %{customdata[0]:.0f}<br>Escalation: %{customdata[1]:.0f}/6<br>"
                    "VIX: %{customdata[2]:.1f}<br>S5FI: %{customdata[3]:.1f}<br>"
                    "HY OAS: %{customdata[4]:.2f}<br>QQQ/SPY: %{customdata[5]:.4f}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(type="log", title="SPY log")
    fig.update_xaxes(range=[pd.Timestamp(start), pd.Timestamp(end)])
    return style_fig(fig, "Log SPY with Current Risk Signals", 560)


def build_current_risk_indicators_fig(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    d = current_risk_history_frame(history, start, end)
    titles = (
        "Price Cycle Vulnerability",
        "RSI Divergence",
        "ROC Deterioration / Blow-off",
        "VIX",
        "S5FI",
        "BAMLH0A0HYM2 10Y Percentile",
        "QQQ/SPY Divergence",
        "Escalation Score V2",
    )
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.022, subplot_titles=titles)
    if d.empty:
        return style_fig(fig, "Current Risk Indicators", 1500)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["PVC_V2"], name="PVC V2", line={"color": "#38bdf8", "width": 1.4}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["PVC_MAX10D"], name="PVC MAX10D", line={"color": "#facc15", "width": 1.2, "dash": "dash"}), row=1, col=1)
    fig.add_hline(y=70, line={"color": "#ef4444", "dash": "dot"}, row=1, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["CurrentRiskDailyRSI14"], name="Daily RSI14", line={"color": "#e5e7eb", "width": 1.4}), row=2, col=1)
    add_boolean_zones(fig, d, "PVC_DailyRSIDivergence", row=2, color="#f97316", opacity=0.17)
    add_boolean_zones(fig, d, "PVC_WeeklyRSIDivergence", row=2, color="#ef4444", opacity=0.13)
    fig.add_hline(y=78, line={"color": "#facc15", "dash": "dot"}, row=2, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=100.0 * d["CurrentRiskROC20D"], name="SPY ROC20D", line={"color": "#38bdf8", "width": 1.4}), row=3, col=1)
    add_boolean_zones(fig, d, "PVC_ROCCondition", row=3, color="#f97316", opacity=0.15)
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dot"}, row=3, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["VIX"], name="VIX", line={"color": "#ef4444", "width": 1.4}), row=4, col=1)
    for level in [20, 30]:
        fig.add_hline(y=level, line={"color": "#64748b", "dash": "dot"}, row=4, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPXAboveSMA50D"], name="S5FI", line={"color": "#22c55e", "width": 1.4}), row=5, col=1)
    for level in [50, 75]:
        fig.add_hline(y=level, line={"color": "#64748b", "dash": "dot"}, row=5, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["CurrentRiskHY10YPercentile"], name="HY 10Y Percentile", line={"color": "#f97316", "width": 1.4}), row=6, col=1)
    for level in [50, 75]:
        fig.add_hline(y=level, line={"color": "#64748b", "dash": "dot"}, row=6, col=1)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["CurrentRiskQQQSPY"], name="QQQ/SPY", line={"color": "#a78bfa", "width": 1.4}), row=7, col=1)
    add_boolean_zones(fig, d, "CurrentRiskTopConfirmation", row=7, color="#ef4444", opacity=0.16)

    fig.add_trace(go.Scatter(x=d["Date"], y=d["CurrentRiskEscalationScoreV2"], name="Escalation Score V2", line={"color": "#fb7185", "width": 1.5}, fill="tozeroy", fillcolor="rgba(251,113,133,0.10)"), row=8, col=1)
    for level, color in [(2, "#facc15"), (3, "#f97316"), (4, "#ef4444")]:
        fig.add_hline(y=level, line={"color": color, "dash": "dot"}, row=8, col=1)

    for row in [1, 5, 6]:
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="VIX", row=4, col=1)
    fig.update_yaxes(title_text="QQQ/SPY", row=7, col=1)
    fig.update_yaxes(title_text="0-6", range=[0, 6.2], row=8, col=1)
    fig.update_xaxes(range=[pd.Timestamp(start), pd.Timestamp(end)])
    fig.update_layout(showlegend=False)
    return style_fig(fig, "Current Risk Indicators", 1500)


def add_boolean_zones(fig: go.Figure, frame: pd.DataFrame, column: str, *, row: int, color: str, opacity: float) -> None:
    if column not in frame or frame.empty:
        return
    mask = frame[column].fillna(False).astype(bool).reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"], errors="coerce").reset_index(drop=True)
    start_idx: int | None = None
    for idx in range(len(frame) + 1):
        active = idx < len(frame) and bool(mask.iloc[idx])
        if active and start_idx is None:
            start_idx = idx
        if not active and start_idx is not None:
            end_idx = idx - 1
            fig.add_vrect(
                x0=dates.iloc[start_idx],
                x1=dates.iloc[end_idx] + pd.Timedelta(days=1),
                fillcolor=color,
                opacity=opacity,
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )
            start_idx = None


def risk_zone_state(value: Any) -> str:
    score = safe_float(value)
    if not np.isfinite(score):
        return "DATA INCOMPLETE"
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MODERATE"
    if score <= 75:
        return "ELEVATED"
    return "HIGH"


def risk_zone_color(state: str) -> str:
    return {
        "LOW": "#22c55e",
        "MODERATE": "#facc15",
        "ELEVATED": "#f97316",
        "HIGH": "#ef4444",
    }.get(state, "#64748b")


def build_current_risk_history_fig(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.42, 0.12],
        vertical_spacing=0.04,
        subplot_titles=("SPX Log", "Current Risk Indicators", "Current Risk State"),
    )
    d = current_risk_history_frame(history, start, end)
    if d.empty:
        return style_fig(fig, "Current Risk History<br><sup>SPX Log vs Historical Current Risk Regime</sup>", 620)

    custom = np.stack(
        [
            pd.to_numeric(d["HistoricalDrawdownRisk"], errors="coerce"),
            pd.to_numeric(d["HistoricalHighBetaRisk"], errors="coerce"),
            pd.to_numeric(d["HistoricalBreadthRisk"], errors="coerce"),
            pd.to_numeric(d["HistoricalRSIDivergenceRisk"], errors="coerce"),
            pd.to_numeric(d["HistoricalCombinedVIXRisk"], errors="coerce"),
            pd.to_numeric(d["HistoricalCurrentRiskScore"], errors="coerce"),
            d["HistoricalCurrentRiskState"].astype(str),
            d["HistoricalCurrentRiskPrimaryDrivers"].astype(str),
        ],
        axis=-1,
    )
    shared_hover = (
        "Date: %{x|%Y-%m-%d}<br>"
        "Drawdown Risk: %{customdata[0]:.1f}% probability of >20% 1M drawdown<br>"
        "High Beta Risk: %{customdata[1]:.1f}<br>"
        "Breadth Risk: %{customdata[2]:.1f}<br>"
        "RSI Divergence Risk: %{customdata[3]:.1f}<br>"
        "Combined VIX Risk: %{customdata[4]:.1f}<br>"
        "Current Risk Score: %{customdata[5]:.1f}<br>"
        "Current Risk State: %{customdata[6]}<br>"
        "Primary Drivers: %{customdata[7]}<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=pd.to_numeric(d["SPX_Close"], errors="coerce"),
            mode="lines",
            name="SPX",
            line={"color": "#f8fafc", "width": 1.7},
            customdata=custom,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<br>Current Risk Score: %{customdata[5]:.1f}<br>Current Risk State: %{customdata[6]}<br>Primary Drivers: %{customdata[7]}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for name, col, color, dash in [
        ("Current Risk Score", "HistoricalCurrentRiskScore", "#a78bfa", "dash"),
        ("Drawdown Risk", "HistoricalDrawdownRisk", "#facc15", "solid"),
        ("High Beta Risk", "HistoricalHighBetaRisk", "#38bdf8", "solid"),
        ("Breadth Risk", "HistoricalBreadthRisk", "#22c55e", "solid"),
        ("RSI Divergence Risk", "HistoricalRSIDivergenceRisk", "#f97316", "solid"),
        ("Combined VIX Risk", "HistoricalCombinedVIXRisk", "#ef4444", "solid"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=d["Date"],
                y=pd.to_numeric(d[col], errors="coerce"),
                mode="lines",
                name=name,
                line={"color": color, "width": 1.5, "dash": dash},
                customdata=custom,
                hovertemplate=shared_hover,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "HistoricalCurrentRiskState",
            ["LOW", "NORMAL", "ELEVATED", "HIGH", "ACUTE"],
            "Current Risk",
        ),
        row=3,
        col=1,
    )
    for level in [20, 40, 60, 80]:
        fig.add_hline(y=level, line={"color": "#334155", "dash": "dot", "width": 1}, row=2, col=1)
    for level in [30, 50, 70, 90]:
        fig.add_hline(y=level, line={"color": "#475569", "dash": "dot", "width": 0.7}, opacity=0.45, row=2, col=1)
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(title_text="0-100", range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100], row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    fig.update_xaxes(range=[pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")])
    return style_fig(fig, "Current Risk History<br><sup>SPX Log vs Historical Current Risk Regime</sup>", 620)


def render_current_risk_history_summary(current: dict[str, Any]) -> None:
    rows = [
        ("Current Risk State", text(current.get("HistoricalCurrentRiskState") or current.get("CurrentMarketRiskState"))),
        ("Weeks in Current State", num(current.get("HistoricalCurrentRiskWeeksInState"), 0)),
        ("Current Risk Score", num(current.get("HistoricalCurrentRiskScore"))),
        ("Primary Drivers", text(current.get("HistoricalCurrentRiskPrimaryDrivers"))),
        ("Drawdown Risk", f"{num(current.get('HistoricalDrawdownRisk'))}%"),
        ("Combined VIX Risk", num(current.get("HistoricalCombinedVIXRisk"))),
    ]
    html_rows = "".join(
        f"<div style='border-bottom:1px solid #263241;padding:0.35rem 0;'>"
        f"<div style='color:#93c5fd;font-size:0.72rem;font-weight:700'>{html.escape(label)}</div>"
        f"<div style='font-size:0.94rem;font-weight:800'>{html.escape(value)}</div></div>"
        for label, value in rows
    )
    st.markdown(f"<div style='background:#0f131a;border:1px solid #263241;border-radius:6px;padding:0.65rem'>{html_rows}</div>", unsafe_allow_html=True)


def current_risk_history_frame(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    d = history.copy()
    d.index.name = None
    if "Date" not in d:
        d["Date"] = pd.to_datetime(d.index, errors="coerce")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    effective_start = max(pd.Timestamp("2015-01-01"), pd.Timestamp(start))
    d = d.loc[d["Date"].between(effective_start, pd.Timestamp(end))].copy()
    numeric_columns = [
        "SPX_Close",
        "SPY_Close",
        "CurrentRiskSPYRawClose",
        "VIX",
        "SPXAboveSMA50D",
        "HY_OAS",
        "PVC_V2",
        "PVC_MAX10D",
        "CurrentRiskDailyRSI14",
        "CurrentRiskROC20D",
        "CurrentRiskHY10YPercentile",
        "CurrentRiskQQQSPY",
        "CurrentRiskEscalationScoreV2",
        "CurrentRiskBreadth5DChange",
        "CurrentRiskVIX3DChange",
        "CurrentRiskHY5DChange",
        "CurrentRiskQQQDivergenceDepth",
    ]
    for col in numeric_columns:
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in [
        "PVC_DailyRSIDivergence",
        "PVC_WeeklyRSIDivergence",
        "PVC_ROCCondition",
        "CurrentRiskTopConfirmation",
        "CurrentRiskCreditConfirmation",
        "CurrentRiskActivation",
    ]:
        if col not in d:
            d[col] = False
    if "CurrentRiskNewEvent" not in d:
        d["CurrentRiskNewEvent"] = current_risk_new_event(d["CurrentRiskActivation"], lookback_sessions=5)
    else:
        d["CurrentRiskNewEvent"] = d["CurrentRiskNewEvent"].fillna(False).astype(bool)
    if "CurrentRiskSignalClass" not in d:
        d["CurrentRiskSignalClass"] = "INACTIVE"
    return d


CORRECTION_STATE_DOMAIN = [
    "NORMAL",
    "CORRECTION WATCH",
    "CORRECTION",
    "STRESS BUILDING",
    "CAPITULATION",
    "EARLY EXHAUSTION",
    "BOTTOMING",
    "RE-ACCELERATION WATCH",
    "RE-ACCELERATION",
    "LOW UNDERCUT",
    "CAPITULATION RECLAIM",
    "CONFIRMED BREAKDOWN",
    "RECOVERY CONFIRMED",
    "NEW CORRECTION WAVE WATCH",
    "NEW CORRECTION WAVE",
]


def render_correction_bottom_indicator(history: pd.DataFrame, current: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> None:
    d = correction_history_frame(history, start, end)
    if d.empty or "CorrectionStatus" not in d:
        st.info("Correction Bottom Indicator data is unavailable.")
        return

    latest_durable_date = pd.to_datetime(current.get("LatestDurableBottomDate"), errors="coerce")
    durable_active = pd.notna(latest_durable_date)
    bear_score = safe_float(current.get("LatestDurableBearRiskScore"))
    cards = [
        (
            "Correction Status",
            text(current.get("CorrectionStatus")),
            [
                ("Current drawdown", pct(current.get("CorrectionCurrentDrawdown"))),
                ("Worst drawdown", pct(current.get("CorrectionWorstDrawdown"))),
                ("Peak", num(current.get("CorrectionPeakPrice"), 2)),
            ],
        ),
        (
            "Extreme Capitulation",
            text(current.get("ExtremeCapitulationStatus")),
            [
                ("Level 2", text(current.get("ExtremeCapitulation2Status"))),
                ("Stress mean", num(current.get("ExtremeStressMean5"), 1)),
                ("Conditions", f"{num(current.get('ExtremeConditionsMet'), 0)}/5"),
            ],
        ),
        (
            "Durable Bottom",
            "CONFIRMED" if durable_active else "NOT CONFIRMED",
            [
                ("Date", latest_durable_date.strftime("%Y-%m-%d") if durable_active else "N/A"),
                ("SPY", num(current.get("LatestDurableBottomPrice"), 2)),
                ("Route", text(current.get("LatestDurableRoute"))),
            ],
        ),
        (
            "Bear Risk",
            f"{bear_score:.0f}/6" if durable_active and np.isfinite(bear_score) else "N/A",
            [
                ("Category", text(current.get("LatestDurableBearRiskCategory")) if durable_active else "N/A"),
                ("Wave", text(current.get("CorrectionWaveID"))),
                ("Coverage", f"{num(current.get('CorrectionDataCoverage'), 0)}%"),
            ],
        ),
    ]
    cols = st.columns(4)
    for idx, card in enumerate(cards):
        with cols[idx]:
            render_card(*card)

    st.plotly_chart(build_correction_primary_fig(d), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)

    st.markdown("#### Current Components")
    st.dataframe(correction_component_table(current), use_container_width=True, hide_index=True)

    extreme_validation, durable_validation, events = correction_validation_tables_v2(history)
    val_col, event_col = st.columns(2)
    with val_col:
        st.markdown("#### Extreme Capitulation Validation")
        st.dataframe(extreme_validation, use_container_width=True, hide_index=True, height=260)
    with event_col:
        st.markdown("#### Durable Bottom Validation")
        st.dataframe(durable_validation, use_container_width=True, hide_index=True, height=260)

    st.markdown("#### Historical Signals")
    st.dataframe(
        events.style.format(
            {
                "Drawdown": "{:.1%}",
                "Forward 5D": "{:.1%}",
                "Forward 20D": "{:.1%}",
                "Forward 60D": "{:.1%}",
                "Additional Downside 20D": "{:.1%}",
                "Additional Downside 60D": "{:.1%}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    with st.expander("Correction Bottom Indicator Logic"):
        st.markdown(
            """
`CORRECTION_BOTTOM_V2` uses SPY daily OHLCV and point-in-time market internals.

**Correction zone.** A master correction starts at a 5% decline from the observed SPY peak and ends only when SPY recovers that original peak. A new internal wave starts after a post-Tactical recovery high is followed by another 5% decline.

**Extreme Capitulation.** The real-time event requires a drawdown of at least 15% and simultaneous extreme 1-year stress percentiles in VIX, RSI, SPY 5-day return, Breadth50, and Breadth200. Nearby raw dates are clustered only for the historical marker; clustering never delays the real-time signal.

**Extreme Capitulation 2.** This is a severity-gated subset of Classic Extreme Capitulation. It additionally requires either a drawdown of at least 20% or a VIX stress percentile of at least 99.8%. Both categories use separate 10-session episode clustering; Level 2 has display priority when both belong to the same episode.

**Durable Bottom.** The internal Tactical trigger requires a 10-day price breakout plus VIX retreat and breadth recovery. It is not displayed. Durable confirmation is evaluated on the same day through either the trend-and-breadth route or the credit-and-stress route.

**Bear Risk.** Every Durable signal receives 0-6 points from SPY versus SMA200D, SMA200D slope, 3M and 6M momentum, HY OAS percentile, and HY OAS direction. The score adds context and never vetoes the signal.

Missing breadth or credit inputs remain unavailable; they are not converted to zero or a passing condition.
            """.strip()
        )


def correction_history_frame(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if history.empty or "Date" not in history:
        return pd.DataFrame()
    d = history.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    d = d.loc[d["Date"].between(start, end)].copy()
    for col in [
        "CorrectionStatus",
        "CorrectionSeverity",
        "CorrectionState",
        "CorrectionType",
        "CorrectionEventType",
        "DurableRoute",
        "BearRiskCategory",
        "BreadthSourceFrequency",
        "VolumeConfirmation",
    ]:
        if col not in d:
            d[col] = "DATA INCOMPLETE"
    numeric_cols = [
        "SPY_Close",
        "SPY_Low",
        "SPY_Volume",
        "SPX_Close",
        "CorrectionDrawdown",
        "CorrectionCurrentDrawdown",
        "CorrectionWorstDrawdown",
        "CorrectionDataCoverage",
        "SPXAboveSMA50D",
        "SPXAboveSMA200D",
        "VIX",
        "VIX_Stress",
        "RSI_Stress",
        "SPY5D_DownsideStress",
        "Breadth50_Stress",
        "Breadth200_Stress",
        "ExtremeStressMean5",
        "ExtremeConditionsMet",
        "SPYVolumePct1Y",
        "SPYVolumeRatio20",
        "SPYVolumeMaxPct5",
        "SPYVolumeMaxRatio5",
        "BearRiskScore",
        "HY_OAS",
        "HYOAS20DChange",
        "HYOASPercentile3Y",
        "HYOASRetreat20D",
        "VIXRetreat20D",
    ]
    for col in numeric_cols:
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def build_correction_primary_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    severity_fill = {
        "PULLBACK / CORRECTION": "rgba(148,163,184,0.08)",
        "SIGNIFICANT CORRECTION": "rgba(250,204,21,0.08)",
        "DEEP CORRECTION": "rgba(249,115,22,0.09)",
        "BEAR / CRASH": "rgba(239,68,68,0.10)",
    }
    zone_x: list[pd.Timestamp] = []
    zone_y: list[float] = []
    zone_text: list[str] = []
    if "CorrectionCycleID" in d:
        for cycle_id, rows in d.dropna(subset=["CorrectionCycleID"]).groupby("CorrectionCycleID", sort=False):
            rows = rows.sort_values("Date")
            start_date = pd.to_datetime(rows["CorrectionStartDate"], errors="coerce").dropna().min()
            recovery = pd.to_datetime(rows.get("HistoricalCorrectionRecoveryDate"), errors="coerce").dropna()
            end_date = recovery.iloc[-1] if not recovery.empty else pd.to_datetime(rows["Date"], errors="coerce").max()
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            severity = text(rows["CorrectionSeverity"].iloc[-1])
            fig.add_vrect(x0=start_date, x1=end_date, fillcolor=severity_fill.get(severity, "rgba(148,163,184,0.07)"), line_width=0, layer="below")
            peak_date = pd.to_datetime(rows["CorrectionPeakDate"], errors="coerce").dropna().iloc[0]
            peak_price = safe_float(rows["CorrectionPeakPrice"].dropna().iloc[0]) if rows["CorrectionPeakPrice"].notna().any() else np.nan
            low_date = pd.to_datetime(rows.get("HistoricalCorrectionFinalLowDate"), errors="coerce").dropna()
            max_dd = safe_float(pd.to_numeric(rows.get("HistoricalCorrectionMaxDrawdown"), errors="coerce").dropna().iloc[-1]) if pd.to_numeric(rows.get("HistoricalCorrectionMaxDrawdown"), errors="coerce").notna().any() else np.nan
            duration = safe_float(pd.to_numeric(rows.get("HistoricalCorrectionDurationDays"), errors="coerce").dropna().iloc[-1]) if pd.to_numeric(rows.get("HistoricalCorrectionDurationDays"), errors="coerce").notna().any() else np.nan
            zone_x.append(start_date)
            zone_y.append(peak_price)
            zone_text.append(
                f"Correction: {cycle_id}<br>Peak date: {peak_date:%Y-%m-%d}<br>Peak price: {peak_price:,.2f}<br>"
                f"Start: {start_date:%Y-%m-%d}<br>Final low: {low_date.iloc[-1]:%Y-%m-%d}" if not low_date.empty else f"Correction: {cycle_id}<br>Start: {start_date:%Y-%m-%d}"
            )
            if zone_text:
                zone_text[-1] += f"<br>Maximum drawdown: {max_dd:.1%}<br>Recovery: {end_date:%Y-%m-%d}<br>Duration: {duration:.0f} days"

    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["SPY_Close"],
            mode="lines",
            name="SPY",
            line={"color": "#e5e7eb", "width": 1.7},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SPY: %{y:,.2f}<extra></extra>",
        )
    )
    if zone_x:
        fig.add_trace(
            go.Scatter(
                x=zone_x,
                y=zone_y,
                mode="markers",
                name="Correction zones",
                marker={"size": 7, "color": "rgba(148,163,184,0.25)"},
                text=zone_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    classic_marker_column = "ExtremeEpisodeDisplayMarker" if "ExtremeEpisodeDisplayMarker" in d else "ExtremeEpisodeMarker"
    extreme = d[d.get(classic_marker_column, pd.Series(False, index=d.index)).apply(is_true_flag)].copy()
    if not extreme.empty:
        classic_custom = np.column_stack(
            [
                extreme["CorrectionCurrentDrawdown"],
                extreme["ExtremeStressMean5"],
                extreme["VIX"],
                extreme["VIX_Stress"],
                extreme["SPY_RSI14"],
                extreme["RSI_Stress"],
                extreme["SPY_5D_Return"],
                extreme["SPY5D_DownsideStress"],
                extreme["SPXAboveSMA50D"],
                extreme["Breadth50_Stress"],
                extreme["SPXAboveSMA200D"],
                extreme["Breadth200_Stress"],
                extreme["SPY_Volume"],
                extreme["SPYVolumeRatio20"],
                extreme["SPYVolumePct1Y"],
                extreme["SPYVolumeMaxPct5"],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=extreme["Date"],
                y=extreme["SPY_Close"],
                mode="markers",
                name="Extreme Capitulation",
                marker={"symbol": "diamond", "size": 11, "color": "#38bdf8", "line": {"color": "#082f49", "width": 1}},
                customdata=classic_custom,
                hovertemplate=(
                    "Extreme Capitulation<br>Date: %{x|%Y-%m-%d}<br>SPY: %{y:,.2f}<br>Drawdown: %{customdata[0]:.1%}<br>"
                    "Stress mean: %{customdata[1]:.1f}<br>VIX: %{customdata[2]:.1f} (%{customdata[3]:.1f})<br>"
                    "RSI: %{customdata[4]:.1f} (%{customdata[5]:.1f})<br>SPY 5D: %{customdata[6]:.1%} (%{customdata[7]:.1f})<br>"
                    "Breadth50: %{customdata[8]:.1f} (%{customdata[9]:.1f})<br>Breadth200: %{customdata[10]:.1f} (%{customdata[11]:.1f})<br>"
                    "Volume: %{customdata[12]:,.0f}<br>Volume/SMA20: %{customdata[13]:.2f}x<br>Volume percentile: %{customdata[14]:.1f}<br>"
                    "Recent 5D max volume percentile: %{customdata[15]:.1f}<extra></extra>"
                ),
            )
        )

    extreme_2 = d[
        d.get("ExtremeCapitulation2EpisodeMarker", pd.Series(False, index=d.index)).apply(is_true_flag)
    ].copy()
    if not extreme_2.empty:
        extreme_2_custom = np.column_stack(
            [
                extreme_2["CorrectionCurrentDrawdown"],
                extreme_2["ExtremeStressMean5"],
                extreme_2["VIX"],
                extreme_2["VIX_Stress"],
                extreme_2["SPY_RSI14"],
                extreme_2["RSI_Stress"],
                extreme_2["SPY_5D_Return"],
                extreme_2["SPY5D_DownsideStress"],
                extreme_2["SPXAboveSMA50D"],
                extreme_2["Breadth50_Stress"],
                extreme_2["SPXAboveSMA200D"],
                extreme_2["Breadth200_Stress"],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=extreme_2["Date"],
                y=extreme_2["SPY_Close"],
                mode="markers",
                name="Extreme Capitulation 2",
                marker={"symbol": "star-diamond", "size": 16, "color": "#991b1b", "line": {"color": "#fecaca", "width": 1.5}},
                customdata=extreme_2_custom,
                hovertemplate=(
                    "Extreme Capitulation 2<br>Date: %{x|%Y-%m-%d}<br>SPY: %{y:,.2f}<br>Drawdown: %{customdata[0]:.1%}<br>"
                    "Stress mean: %{customdata[1]:.1f}<br>VIX: %{customdata[2]:.1f} (%{customdata[3]:.1f})<br>"
                    "RSI: %{customdata[4]:.1f} (%{customdata[5]:.1f})<br>SPY 5D: %{customdata[6]:.1%} (%{customdata[7]:.1f})<br>"
                    "Breadth50: %{customdata[8]:.1f} (%{customdata[9]:.1f})<br>"
                    "Breadth200: %{customdata[10]:.1f} (%{customdata[11]:.1f})<extra></extra>"
                ),
            )
        )

    durable = d[d.get("DurableBottomConfirmed", pd.Series(False, index=d.index)).apply(is_true_flag)].copy()
    if not durable.empty:
        custom = np.column_stack(
            [
                durable["CorrectionCycleID"].astype(str),
                durable["CorrectionWaveID"].astype(str),
                durable["CorrectionWorstDrawdown"],
                durable["DurableRoute"].astype(str),
                durable["BearRiskScore"],
                durable["BearRiskCategory"].astype(str),
                durable["SPY_Close"] / durable["SPY_SMA200D"] - 1.0,
                durable["SMA200D_Slope20D"],
                durable["SPY_ROC3M"],
                durable["SPY_ROC6M"],
                durable["Breadth200_5D_Change"],
                durable["HY_OAS"],
                durable["HYOAS20DChange"],
                durable["HYOASPercentile3Y"],
                durable["HYOASRetreat20D"],
                durable["VIXRetreat20D"],
            ]
        )
        labels = durable["BearRiskScore"].apply(lambda value: "" if not np.isfinite(safe_float(value)) else f"{safe_float(value):.0f}")
        fig.add_trace(
            go.Scatter(
                x=durable["Date"],
                y=durable["SPY_Close"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"color": "#fdba74", "size": 10},
                name="Durable Bottom Confirmed",
                marker={"symbol": "star", "size": 13, "color": "#f97316", "line": {"color": "#431407", "width": 1}},
                customdata=custom,
                hovertemplate=(
                    "Durable Bottom Confirmed<br>Date: %{x|%Y-%m-%d}<br>SPY: %{y:,.2f}<br>Cycle: %{customdata[0]}<br>Wave: %{customdata[1]}<br>"
                    "Worst drawdown so far: %{customdata[2]:.1%}<br>Route: %{customdata[3]}<br>Bear risk: %{customdata[4]:.0f}/6 (%{customdata[5]})<br>"
                    "SPY vs SMA200D: %{customdata[6]:.1%}<br>SMA200D 20D slope: %{customdata[7]:.2f}<br>3M ROC: %{customdata[8]:.1%}<br>6M ROC: %{customdata[9]:.1%}<br>"
                    "Breadth200 5-observation change: %{customdata[10]:.1f}pp<br>HY OAS: %{customdata[11]:.2f}<br>HY OAS 20D change: %{customdata[12]:.2f}<br>"
                    "HY OAS 3Y percentile: %{customdata[13]:.1f}<br>HY OAS retreat: %{customdata[14]:.1%}<br>VIX retreat: %{customdata[15]:.1%}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(type="log", title="SPY log")
    fig.update_xaxes(title=None)
    return style_fig(fig, "SPY - Extreme Capitulation and Durable Bottoms", 620)


def correction_component_table(current: dict[str, Any]) -> pd.DataFrame:
    def threshold_status(value: Any, threshold: float) -> bool | str:
        number = safe_float(value)
        return number >= threshold if np.isfinite(number) else "UNAVAILABLE"

    def flag_status(value: Any) -> bool | str:
        if value is None or value is pd.NA:
            return "UNAVAILABLE"
        try:
            if pd.isna(value):
                return "UNAVAILABLE"
        except (TypeError, ValueError):
            pass
        if str(value).strip().upper() in {"", "N/A", "NAN", "NONE", "DATA INCOMPLETE", "UNAVAILABLE"}:
            return "UNAVAILABLE"
        return is_true_flag(value)

    rows = [
        ("Extreme Capitulation", "VIX", current.get("VIX"), current.get("VIX_Stress"), ">=97.6 percentile", threshold_status(current.get("VIX_Stress"), 97.6)),
        ("Extreme Capitulation", "RSI", current.get("SPY_RSI14"), current.get("RSI_Stress"), ">=89.3 stress percentile", threshold_status(current.get("RSI_Stress"), 89.3)),
        ("Extreme Capitulation", "SPY 5D Return", current.get("SPY_5D_Return"), current.get("SPY5D_DownsideStress"), ">=90.5 downside percentile", threshold_status(current.get("SPY5D_DownsideStress"), 90.5)),
        ("Extreme Capitulation", "Breadth50", current.get("SPXAboveSMA50D"), current.get("Breadth50_Stress"), ">=93.2 stress percentile", threshold_status(current.get("Breadth50_Stress"), 93.2)),
        ("Extreme Capitulation", "Breadth200", current.get("SPXAboveSMA200D"), current.get("Breadth200_Stress"), ">=94.4 stress percentile", threshold_status(current.get("Breadth200_Stress"), 94.4)),
        ("Extreme Capitulation 2", "Classic signal", current.get("ExtremeCapRaw"), np.nan, "Classic Extreme Capitulation = TRUE", flag_status(current.get("ExtremeCapRaw"))),
        ("Extreme Capitulation 2", "SPY Drawdown", current.get("CorrectionCurrentDrawdown"), np.nan, "<=-20% OR VIX stress >=99.8", flag_status(safe_float(current.get("CorrectionCurrentDrawdown")) <= -0.20 or safe_float(current.get("VIX_Stress")) >= 99.8)),
        ("Diagnostic only", "SPY Volume", current.get("SPY_Volume"), current.get("SPYVolumePct1Y"), ">=95 percentile or >=1.5x", flag_status(text(current.get("VolumeConfirmation")) == "EXTREME") if np.isfinite(safe_float(current.get("SPYVolumePct1Y"))) else "UNAVAILABLE"),
        ("Durable", "Internal Tactical Trigger", current.get("TacticalInternalEvent"), np.nan, "Same-day causal trigger", flag_status(current.get("TacticalInternalEvent"))),
        ("Durable", "Trend + Breadth Route", current.get("DurableRouteTrendBreadth"), np.nan, "All route conditions", flag_status(current.get("DurableRouteTrendBreadth"))),
        ("Durable", "Credit / Stress Route", current.get("DurableRouteCreditStress"), np.nan, "Any route condition", flag_status(current.get("DurableRouteCreditStress"))),
    ]
    bear_rows = [
        ("SPY below SMA200D", "BearRisk_SPYBelowSMA200"),
        ("SMA200D slope negative", "BearRisk_SMA200SlopeNegative"),
        ("3M ROC negative", "BearRisk_ROC3MNegative"),
        ("6M ROC negative", "BearRisk_ROC6MNegative"),
        ("HY OAS >=90th percentile", "BearRisk_HYOASPct90"),
        ("HY OAS 20D non-improving", "BearRisk_HY20DNonImproving"),
    ]
    for label, key in bear_rows:
        value = current.get(key)
        rows.append(("Bear Risk", label, value, np.nan, "1 point when true", flag_status(value)))
    table = pd.DataFrame(rows, columns=["Block", "Component", "Current Value", "Percentile / Stress", "Threshold", "Pass"])
    table["Pass"] = table["Pass"].map(lambda value: "PASS" if value is True else "FAIL" if value is False else str(value))
    return table


def correction_validation_tables_v2(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if history.empty or "Date" not in history:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    h = history.copy().reset_index(drop=True)
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
    close = pd.to_numeric(h.get("SPY_Close"), errors="coerce")
    low = pd.to_numeric(h.get("SPY_Low", close), errors="coerce").fillna(close)
    for horizon in [5, 20, 60]:
        h[f"ForwardReturn_{horizon}D"] = close.shift(-horizon) / close - 1.0
        h[f"AdditionalDownside_{horizon}D"] = [
            (low.iloc[idx + 1 : min(len(h), idx + horizon + 1)].min() / close.iloc[idx] - 1.0)
            if idx + 1 < len(h) and np.isfinite(safe_float(close.iloc[idx]))
            else np.nan
            for idx in range(len(h))
        ]

    extreme = h[h.get("ExtremeEpisodeMarker", pd.Series(False, index=h.index)).apply(is_true_flag)].copy()
    classic_display = h[
        h.get("ExtremeEpisodeDisplayMarker", h.get("ExtremeEpisodeMarker", pd.Series(False, index=h.index))).apply(is_true_flag)
    ].copy()
    extreme_2 = h[
        h.get("ExtremeCapitulation2EpisodeMarker", pd.Series(False, index=h.index)).apply(is_true_flag)
    ].copy()
    extreme_rows = [
        ("Episodes", len(extreme)),
        ("Median 5D return", extreme["ForwardReturn_5D"].median()),
        ("Median 20D return", extreme["ForwardReturn_20D"].median()),
        ("Median 60D return", extreme["ForwardReturn_60D"].median()),
        ("Median additional downside 20D", extreme["AdditionalDownside_20D"].median()),
        ("Median additional downside 60D", extreme["AdditionalDownside_60D"].median()),
        ("Extreme Capitulation 2 episodes", len(extreme_2)),
    ]
    extreme_validation = pd.DataFrame(extreme_rows, columns=["Metric", "Value"])

    durable = h[h.get("DurableBottomConfirmed", pd.Series(False, index=h.index)).apply(is_true_flag)].copy()
    validation_rows = []
    groups = [("ALL", durable)]
    for category in ["LOW BEAR RISK", "MODERATE BEAR RISK", "HIGH BEAR RISK"]:
        groups.append((category, durable[durable.get("BearRiskCategory", "").eq(category)]))
    for label, sample in groups:
        validation_rows.append(
            {
                "Bear Risk": label,
                "Signals": len(sample),
                "Independent Corrections": sample.get("CorrectionCycleID", pd.Series(dtype="object")).nunique(),
                "20D Success": float((sample["AdditionalDownside_20D"].dropna() >= -0.03).mean()) if sample["AdditionalDownside_20D"].notna().any() else np.nan,
                "60D Success": float((sample["AdditionalDownside_60D"].dropna() >= -0.05).mean()) if sample["AdditionalDownside_60D"].notna().any() else np.nan,
                "Median Downside 20D": sample["AdditionalDownside_20D"].median(),
                "Median Downside 60D": sample["AdditionalDownside_60D"].median(),
                "Median Return 20D": sample["ForwardReturn_20D"].median(),
                "Median Return 60D": sample["ForwardReturn_60D"].median(),
            }
        )
    durable_validation = pd.DataFrame(validation_rows)

    extreme_events = classic_display.assign(Event="EXTREME CAPITULATION", **{"Bear Risk": ""})
    extreme_2_events = extreme_2.assign(Event="EXTREME CAPITULATION 2", **{"Bear Risk": ""})
    durable_events = durable.assign(Event="DURABLE BOTTOM CONFIRMED", **{"Bear Risk": durable.get("BearRiskCategory", "")})
    events = pd.concat([extreme_events, extreme_2_events, durable_events], ignore_index=True).sort_values("Date", ascending=False)
    event_table = pd.DataFrame(
        {
            "Date": events.get("Date"),
            "Event": events.get("Event"),
            "Cycle": events.get("CorrectionCycleID"),
            "Wave": events.get("CorrectionWaveID"),
            "Drawdown": events.get("CorrectionCurrentDrawdown"),
            "Route": events.get("DurableRoute"),
            "Bear Risk Score": events.get("BearRiskScore"),
            "Bear Risk": events.get("Bear Risk"),
            "Forward 5D": events.get("ForwardReturn_5D"),
            "Forward 20D": events.get("ForwardReturn_20D"),
            "Forward 60D": events.get("ForwardReturn_60D"),
            "Additional Downside 20D": events.get("AdditionalDownside_20D"),
            "Additional Downside 60D": events.get("AdditionalDownside_60D"),
        }
    ).head(80)
    return extreme_validation, durable_validation, event_table


def build_correction_spx_state_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.03, subplot_titles=("SPX Log", "Correction State"))
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["SPX_Close"],
            mode="lines",
            name="SPX",
            line={"color": "#e5e7eb", "width": 1.6},
            customdata=np.column_stack(
                [
                    d["CorrectionState"].astype(str),
                    d["CorrectionType"].astype(str),
                    pd.to_numeric(d["CorrectionDrawdown"], errors="coerce"),
                    pd.to_numeric(d["CorrectionExhaustionScore"], errors="coerce"),
                    pd.to_numeric(d["CorrectionRecoveryScore"], errors="coerce"),
                ]
            ),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<br>State: %{customdata[0]}<br>Type: %{customdata[1]}<br>Drawdown: %{customdata[2]:.1%}<br>Exhaustion: %{customdata[3]:.1f}<br>Recovery: %{customdata[4]:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    event_symbols = {
        "BOTTOMING": "triangle-up",
        "RECOVERY CONFIRMED": "circle",
        "RE-ACCELERATION": "x",
        "LOW UNDERCUT": "diamond",
        "CONFIRMED BREAKDOWN": "triangle-down",
        "NEW CORRECTION WAVE": "square",
        "CAPITULATION RECLAIM": "star",
    }
    events = d[d["CorrectionEventType"].astype(str).str.len().gt(0)].copy()
    for event, symbol in event_symbols.items():
        sample = events[events["CorrectionEventType"].eq(event)]
        if sample.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sample["Date"],
                y=sample["SPX_Close"],
                mode="markers",
                name=event,
                marker={"symbol": symbol, "size": 10, "color": PHASE_COLORS.get(event, "#f8fafc"), "line": {"color": "#0f131a", "width": 1}},
                hovertemplate=f"{event}<br>Date: %{{x|%Y-%m-%d}}<br>SPX: %{{y:,.0f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(build_state_strip_trace(d, "CorrectionState", CORRECTION_STATE_DOMAIN, "Correction"), row=2, col=1)
    fig.update_yaxes(type="log", title_text="SPX log", row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    return style_fig(fig, "SPX + Correction State", 520)


def build_correction_scores_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["CorrectionExhaustionScore"], mode="lines", name="Exhaustion Score", line={"color": "#38bdf8", "width": 2}))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["CorrectionRecoveryScore"], mode="lines", name="Recovery Score", line={"color": "#22c55e", "width": 2}))
    fig.add_hline(y=50, line={"color": "#38bdf8", "dash": "dot"}, annotation_text="Exhaustion 50")
    fig.add_hline(y=40, line={"color": "#22c55e", "dash": "dot"}, annotation_text="Recovery 40")
    fig.update_yaxes(range=[0, 100], title="0-100")
    return style_fig(fig, "Exhaustion Score vs Recovery Score", 380)


def build_correction_components_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [
        ("Price / Momentum", "PriceMomentumExhaustion", "#38bdf8"),
        ("Volatility", "VolatilityExhaustion", "#f97316"),
        ("Breadth", "BreadthExhaustion", "#22c55e"),
        ("Credit / Positioning", "CreditPositioningExhaustion", "#a78bfa"),
        ("Risk Appetite", "RiskAppetiteExhaustion", "#facc15"),
    ]
    for name, col, color in series:
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], mode="lines", name=name, line={"color": color, "width": 1.5}))
    fig.update_yaxes(range=[0, 100], title="0-100")
    return style_fig(fig, "Exhaustion Components", 380)


def build_correction_reacceleration_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["Date"],
            y=d["CorrectionActiveStressDomains"],
            name="Active Stress Domains",
            marker={"color": "#f97316"},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Active stress domains: %{y:.0f}<extra></extra>",
        )
    )
    for col, name, color in [
        ("CorrectionReaccelerationWatch", "Re-acceleration Watch", "#a855f7"),
        ("CorrectionLowUndercut", "Low Undercut", "#e879f9"),
        ("CorrectionConfirmedBreakdown", "Confirmed Breakdown", "#ef4444"),
        ("CorrectionNewWaveWatch", "New Wave Watch", "#f59e0b"),
    ]:
        if col in d:
            sample = d[d[col].apply(is_true_flag)]
            fig.add_trace(go.Scatter(x=sample["Date"], y=[4] * len(sample), mode="markers", name=name, marker={"color": color, "size": 8}, hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<extra></extra>"))
    fig.update_yaxes(range=[0, 4.4], title="Stress domains")
    return style_fig(fig, "Re-acceleration Monitor", 360)


def build_correction_breadth_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.05, subplot_titles=("SPX", "Breadth"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX_Close"], mode="lines", name="SPX", line={"color": "#e5e7eb", "width": 1.4}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPXAboveSMA50D"], mode="lines", name="% Above SMA50D / S5FI", line={"color": "#38bdf8", "width": 1.6}), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPXAboveSMA200D"], mode="lines", name="% Above SMA200D / S5TH", line={"color": "#22c55e", "width": 1.6}), row=2, col=1)
    fig.add_hline(y=20, line={"color": "#ef4444", "dash": "dot"}, row=2, col=1)
    fig.add_hline(y=50, line={"color": "#64748b", "dash": "dot"}, row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 100], title="%", row=2, col=1)
    return style_fig(fig, "SPX + Breadth", 430)


def build_correction_volatility_fig(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05, subplot_titles=("Volatility", "VIX / VIX3M"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["VIX"], mode="lines", name="VIX", line={"color": "#ef4444", "width": 1.5}), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["VIX3M"], mode="lines", name="VIX3M", line={"color": "#facc15", "width": 1.3}), row=1, col=1)
    ratio = d["VIX"] / d["VIX3M"]
    fig.add_trace(go.Scatter(x=d["Date"], y=ratio, mode="lines", name="VIX/VIX3M", line={"color": "#38bdf8", "width": 1.5}), row=2, col=1)
    fig.add_hline(y=1.0, line={"color": "#ef4444", "dash": "dot"}, row=2, col=1)
    return style_fig(fig, "Volatility Exhaustion", 430)


def build_correction_credit_positioning_fig(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["CreditPositioningExhaustion"], mode="lines", name="Credit / Positioning Exhaustion", line={"color": "#a78bfa", "width": 2}))
    if "AAII_Bearish_Percentile" in d:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["AAII_Bearish_Percentile"], mode="lines", name="AAII Bearish Percentile", line={"color": "#facc15", "width": 1.2}))
    if "VIX_AssetManager_Percentile" in d:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["VIX_AssetManager_Percentile"], mode="lines", name="VIX AM Percentile", line={"color": "#38bdf8", "width": 1.2}))
    fig.update_yaxes(range=[0, 100], title="0-100 / percentile")
    return style_fig(fig, "Credit / CFTC Deleveraging", 430)


def build_positioning_vulnerability_fig(d: pd.DataFrame) -> go.Figure:
    positioning = pd.to_numeric(d.get("PositioningRisk", pd.Series(np.nan, index=d.index)), errors="coerce")
    aaii = pd.to_numeric(d.get("AAII_Bearish_Percentile", pd.Series(np.nan, index=d.index)), errors="coerce")
    vix_asset_manager = pd.to_numeric(
        d.get("VIX_AssetManager_Percentile", pd.Series(np.nan, index=d.index)), errors="coerce"
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.82, 0.18],
        vertical_spacing=0.045,
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Positioning Risk and Percentile Components", "Positioning Vulnerability"),
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=positioning,
            mode="lines",
            name="Positioning Risk",
            line={"color": "#f8fafc", "width": 2.2},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Positioning Risk: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=aaii,
            mode="lines",
            name="AAII Bearish 3Y Percentile",
            line={"color": "#facc15", "width": 1.5},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>AAII Bearish 3Y Percentile: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=vix_asset_manager,
            mode="lines",
            name="VIX Asset Manager Net %OI 3Y Percentile",
            line={"color": "#38bdf8", "width": 1.5},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>VIX Asset Manager Net %%OI 3Y Percentile: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        build_state_strip_trace(
            d,
            "PositioningVulnerability",
            ["LOW", "NORMAL", "ELEVATED", "CROWDED", "EXTREME"],
            "Positioning Vulnerability",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[0, 100], title_text="Positioning Risk", row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 100], title_text="Percentile", row=1, col=1, secondary_y=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    return style_fig(fig, "Positioning Risk and Vulnerability", 430)


def correction_validation_tables(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history.empty or "CorrectionEventType" not in history:
        return pd.DataFrame(), pd.DataFrame()
    h = history.copy()
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
    h = h.dropna(subset=["Date"]).sort_values("Date")
    close = pd.to_numeric(h.get("SPX_Close", pd.Series(np.nan, index=h.index)), errors="coerce")
    h["ForwardReturn_5D"] = close.shift(-5) / close - 1.0
    h["ForwardReturn_20D"] = close.shift(-20) / close - 1.0
    h["ForwardReturn_60D"] = close.shift(-60) / close - 1.0
    events = h[h["CorrectionEventType"].astype(str).str.len().gt(0)].copy()
    if events.empty:
        return pd.DataFrame([{"Event Type": "No historical events", "N": 0}]), pd.DataFrame()
    validation = (
        events.groupby("CorrectionEventType", dropna=False)
        .agg(
            N=("Date", "count"),
            Median_5D=("ForwardReturn_5D", "median"),
            Median_20D=("ForwardReturn_20D", "median"),
            Median_60D=("ForwardReturn_60D", "median"),
            Positive_60D=("ForwardReturn_60D", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
        )
        .reset_index()
        .rename(
            columns={
                "CorrectionEventType": "Event Type",
                "Median_5D": "Median 5D",
                "Median_20D": "Median 20D",
                "Median_60D": "Median 60D",
                "Positive_60D": "Positive 60D",
            }
        )
    )
    event_cols = [
        "Date",
        "CorrectionEventType",
        "CorrectionState",
        "CorrectionDrawdown",
        "CorrectionExhaustionScore",
        "CorrectionRecoveryScore",
        "ForwardReturn_5D",
        "ForwardReturn_20D",
        "ForwardReturn_60D",
    ]
    event_table = events[[col for col in event_cols if col in events.columns]].tail(40).sort_values("Date", ascending=False)
    return validation, event_table


def render_confirmations_table(current: dict[str, Any]) -> None:
    rsi_state = text(current.get("RSIDivergenceState"))
    rsi_duration = safe_float(current.get("RSIDivergenceDurationWeeks"))
    rsi_label = rsi_state if not np.isfinite(rsi_duration) or rsi_state in {"N/A", "NONE"} else f"{rsi_state} - {rsi_duration:.0f}W"
    rows = [
        ("QQQ/SPX", text(current.get("QQQ_SPX_State"))),
        ("BTC/SPX", text(current.get("BTC_SPX_State"))),
        ("RSP/SPX", text(current.get("RSP_SPX_State"))),
        ("IWM/SPX", text(current.get("IWM_SPX_State"))),
        ("XLI/XLP", text(current.get("XLI_XLP_State"))),
        ("Breadth Participation", text(current.get("BreadthParticipationState"))),
        ("Market Turning Signal", text(current.get("MarketTurningSignal"))),
        ("Positioning Vulnerability", text(current.get("PositioningVulnerability"))),
        ("RSI Divergence", rsi_label),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Signal", "State"]), use_container_width=True, hide_index=True, height=340)


def build_timeline_fig(d: pd.DataFrame) -> go.Figure:
    strips = [
        ("StructuralMarketCyclePhase", "Structural"),
        ("StructuralROCMomentumState", "Structural ROC"),
        ("MomentumCyclePhase", "Momentum"),
        ("SMA200WZone", "SMA200W"),
        ("CurrentMarketRiskState", "Current Risk"),
        ("MarketTurningSignal", "Turning Signal"),
    ]
    fig = go.Figure()
    for idx, (col, label) in enumerate(strips):
        codes, names = encode_states(d[col].astype(str))
        fig.add_trace(go.Heatmap(x=d["Date"], y=[label], z=[codes], colorscale=state_colorscale(names), showscale=False, customdata=[d[col].astype(str)], hovertemplate="Date: %{x|%Y-%m-%d}<br>%{y}: %{customdata}<extra></extra>"))
    return style_fig(fig, "Market Cycle Timeline", 280)


def build_return_distribution_fig(analogs: pd.DataFrame, horizon: str) -> go.Figure:
    col = f"ForwardReturn_{horizon}"
    values = pd.to_numeric(analogs.get(col, pd.Series(dtype="float64")), errors="coerce").dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=24, marker={"color": "#38bdf8", "opacity": 0.72}, name="Forward returns"))
    if not values.empty:
        fig.add_vline(x=float(values.median()), line={"color": "#facc15", "width": 2}, annotation_text="Median")
        fig.add_vline(x=float(values.mean()), line={"color": "#f8fafc", "dash": "dash"}, annotation_text="Mean")
        fig.add_vline(x=0, line={"color": "#94a3b8", "dash": "dot"})
    fig.update_xaxes(title=f"{horizon} forward return (%)")
    return style_fig(fig, f"Historical Analog Return Distribution - Independent N={len(values)}", 320)


def build_drawdown_distribution_fig(analogs: pd.DataFrame, horizon: str) -> go.Figure:
    col = f"ForwardMaxDrawdown_{horizon}"
    values = pd.to_numeric(analogs.get(col, pd.Series(dtype="float64")), errors="coerce").dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=24, marker={"color": "#ef4444", "opacity": 0.72}, name="Forward max drawdown"))
    fig.add_vline(x=-15, line={"color": "#facc15", "width": 2}, annotation_text="-15%")
    if not values.empty:
        prob = (values <= -15).mean()
        fig.add_annotation(x=0.98, y=0.94, xref="paper", yref="paper", text=f"P(>15% DD): {prob:.0%}<br>Independent N={len(values)}", showarrow=False, align="right")
    fig.update_xaxes(title=f"{horizon} forward max drawdown (%)")
    return style_fig(fig, f"Historical Analog Drawdown Distribution - Independent N={len(values)}", 320)


def render_analog_details_table(analogs: pd.DataFrame) -> None:
    if analogs is None or analogs.empty:
        st.info("Historical analog episode details are unavailable.")
        return
    d = analogs.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    if "AnalogEpisodeStart" not in d:
        d["AnalogEpisodeStart"] = d["Date"]
    if "AnalogEpisodeEnd" not in d:
        d["AnalogEpisodeEnd"] = d["Date"]
    d["Episode"] = d.apply(
        lambda row: f"{date_text(row.get('AnalogEpisodeStart'))} -> {date_text(row.get('AnalogEpisodeEnd'))}",
        axis=1,
    )
    cols = [
        "Date",
        "Episode",
        "Similarity",
        "StructuralSimilarity",
        "MomentumSimilarity",
        "SMA200WSimilarity",
        "CurrentRiskSimilarity",
        "PositioningSimilarity",
        "FeatureCoveragePct",
        "ForwardReturn_3M",
        "ForwardReturn_6M",
        "ForwardReturn_12M",
        "ForwardMaxDrawdown_12M",
    ]
    for col in cols:
        if col not in d:
            d[col] = np.nan
    display = d[cols].sort_values("Similarity", ascending=False).head(60).copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    st.markdown("#### Historical Analog Episodes")
    st.dataframe(
        display.style.format(
            {
                "Similarity": "{:.1f}",
                "StructuralSimilarity": "{:.1f}",
                "MomentumSimilarity": "{:.1f}",
                "SMA200WSimilarity": "{:.1f}",
                "CurrentRiskSimilarity": "{:.1f}",
                "PositioningSimilarity": "{:.1f}",
                "FeatureCoveragePct": "{:.0f}%",
                "ForwardReturn_3M": "{:.1%}",
                "ForwardReturn_6M": "{:.1%}",
                "ForwardReturn_12M": "{:.1%}",
                "ForwardMaxDrawdown_12M": "{:.1%}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )


def add_background_bands(fig: go.Figure, d: pd.DataFrame, state_col: str, opacity: float = 0.12, row: int | None = None) -> None:
    if d.empty or state_col not in d.columns:
        return
    dates = pd.to_datetime(d["Date"], errors="coerce").reset_index(drop=True)
    states = d[state_col].astype(str).reset_index(drop=True)
    start_idx = 0
    for idx in range(1, len(d) + 1):
        if idx == len(d) or states.iloc[idx] != states.iloc[start_idx]:
            state = states.iloc[start_idx]
            kwargs = {"x0": dates.iloc[start_idx], "x1": dates.iloc[idx - 1] + pd.Timedelta(days=7), "fillcolor": PHASE_COLORS.get(state, "#64748b"), "opacity": opacity, "line_width": 0}
            if row is None:
                fig.add_vrect(**kwargs)
            else:
                fig.add_vrect(**kwargs, row=row, col=1)
            start_idx = idx


def style_fig(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 34, "t": 58, "b": 44},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "top", "y": -0.13, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickformat="%b'%y", showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569")
    fig.update_yaxes(showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569")
    return fig


def style_percent_table(frame: pd.DataFrame) -> Any:
    if frame is None or frame.empty:
        return pd.DataFrame().style
    return frame.style.format(
        {
            "Median Return": "{:.1%}",
            "Risk of >15% Drawdown": "{:.0%}",
        },
        na_rep="N/A",
    ).set_properties(**{"background-color": "#0f131a", "color": "#e5e7eb", "border-color": "#263241"})


def filter_range(history: pd.DataFrame, years: int | None) -> pd.DataFrame:
    if years is None or history.empty:
        return history.copy()
    cutoff = pd.to_datetime(history["Date"]).max() - pd.DateOffset(years=years)
    return history[history["Date"].ge(cutoff)].copy()


def range_domain(history: pd.DataFrame, selection: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    date_values = history["Date"] if "Date" in history else pd.Series(history.index, index=history.index)
    dates = pd.to_datetime(date_values, errors="coerce").dropna()
    if dates.empty:
        now = pd.Timestamp.utcnow().tz_localize(None)
        return now - pd.DateOffset(years=20), now
    end = dates.max()
    years = range_years(selection)
    start = dates.min() if years is None else end - pd.DateOffset(years=years)
    return pd.Timestamp(start), pd.Timestamp(end)


def filter_range_domain(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    dates = pd.to_datetime(history["Date"], errors="coerce")
    return history.loc[dates.between(start, end)].copy()


def apply_common_x_range(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[start_s, end_s])
    return fig


def range_years(selection: str) -> int | None:
    return MARKET_CYCLE_RANGE_YEARS.get(selection)


def normalize_range_state(key: str) -> None:
    if st.session_state.get(key) not in MARKET_CYCLE_RANGE_OPTIONS:
        st.session_state[key] = "20Y"


def monthly_context_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "Date" not in history:
        return pd.DataFrame()
    frame = history.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").set_index("Date").resample("ME").last().reset_index()
    needed = [
        "PrimaryCycleComposite",
        "PrimaryMarketCycle",
        "PrimaryCycleTrough",
        "PrimaryCycleLastTroughDate",
        "PrimaryCyclePreviousTroughDate",
        "PrimaryCycleTroughToTroughMonths",
        "PrimaryCycleMonthsSinceTrough",
        "PrimaryCycleMaturityPct",
        "PrimaryCycleDirection",
        "LongCycleTrough",
        "LongCycleLastTroughDate",
        "LongCyclePreviousTroughDate",
        "LongCycleTroughToTroughMonths",
        "LongCycleMonthsSinceTrough",
        "LongCycleMaturityPct",
        "LongCycleDirection",
        "StructuralExtensionSmooth",
        "StructuralMomentum12M",
        "StructuralDiagnosticRegime",
        "LongMarketExtensionCycle",
        "LongCycleAmplitudeHilbert",
        "LongCycleAmplitudeCausal",
        "LongCycleAmplitudePercentile",
        "LongCycleAmplitudeState",
        "Corr_LongAmplitude_vs_StructuralExtension",
        "Corr_LongAmplitude_vs_StructuralMomentum",
        "StructuralTrough",
        "StructuralPeak",
        "StructuralLastTroughDate",
        "StructuralMonthsSinceTrough",
        "StructuralMaturityPct",
        "StructuralReferenceCycle",
        "StructuralReferenceFullLengthMonths",
        "StructuralReferenceTroughToPeakMonths",
        "StructuralReferenceProgressPct",
    ]
    for col in needed:
        if col not in frame:
            frame[col] = np.nan
    return frame


def normalize_centered_full(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < 12:
        return pd.Series(np.nan, index=series.index)
    scale = valid.std()
    if not np.isfinite(safe_float(scale)) or scale == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - valid.mean()) / scale


def encode_states(series: pd.Series) -> tuple[list[int], list[str]]:
    names = list(dict.fromkeys(series.astype(str).tolist()))
    lookup = {name: idx for idx, name in enumerate(names)}
    return [lookup.get(value, 0) for value in series.astype(str)], names


def state_colorscale(names: list[str]) -> list[list[Any]]:
    if not names:
        return [[0, "#64748b"], [1, "#64748b"]]
    if len(names) == 1:
        color = PHASE_COLORS.get(names[0], "#64748b")
        return [[0, color], [1, color]]
    scale = []
    denom = max(len(names) - 1, 1)
    for idx, name in enumerate(names):
        scale.append([idx / denom, PHASE_COLORS.get(name, "#64748b")])
    return scale


def breadth_risk_numeric(value: Any) -> float:
    return {"LOW": 20.0, "NORMAL": 45.0, "WEAKENING": 70.0, "STRESSED": 90.0}.get(str(value), np.nan)


def pct(value: Any) -> str:
    v = safe_float(value)
    return "N/A" if not np.isfinite(v) else f"{v:.1%}"


def _spy_macro_value(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}{suffix}"


def _spy_macro_range_domain(history: pd.DataFrame, selection: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
    end = dates.max()
    if selection == "2015 -> Latest":
        return max(dates.min(), pd.Timestamp("2015-01-01")), end
    years = {"1Y": 1, "5Y": 5, "10Y": 10, "20Y": 20}.get(selection)
    return (dates.min() if years is None else end - pd.DateOffset(years=years)), end


def _spy_macro_filtered_history(history: pd.DataFrame, selection: str) -> pd.DataFrame:
    start, end = _spy_macro_range_domain(history, selection)
    dates = pd.to_datetime(history["Date"], errors="coerce")
    return history.loc[dates.between(start, end)].copy()


def _spy_macro_narrative(current: dict[str, Any]) -> str:
    scores = ", ".join(f"{horizon} {_spy_macro_value(current.get(f'Macro_{horizon}'))}" for horizon in SPY_MACRO_HORIZONS)
    transition_columns = ("Transition_US2Y", "Transition_USD", "Transition_Liquidity", "Transition_HY", "Transition_ANFCI", "Transition_BusinessCycle")
    transition = ", ".join(str(current.get(column)) for column in transition_columns if current.get(column) not in (None, "Neutral", "nan")) or "No active transition flags"
    modifiers = ", ".join(str(current.get(column)) for column in ("T5YIE_Modifier", "WTI_ShockRisk", "JP10Y_ShockRisk") if current.get(column) not in (None, "Neutral", "nan")) or "No active macro modifier"
    return f"Current macro scores: {scores}. Term structure is {str(current.get('TermStructureState', 'Insufficient data')).lower()}. Transition signals: {transition}. Modifiers: {modifiers}."


def build_spy_macro_history_fig(history: pd.DataFrame, horizon: str, show_windows: bool) -> go.Figure:
    d = history.copy()
    score_column = f"Macro_{horizon}"
    growth_column = "Growth_3M" if horizon == "3M" else "Growth_6M" if horizon == "6M" else "Growth_9M" if horizon == "9M" else "BusinessCycle_12M"
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06, subplot_titles=("SPX Log", f"SPY Macro Outlook Score - {horizon}"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["SPX"], mode="lines", name="SPX", line={"color": "#f8fafc", "width": 1.7}, customdata=np.column_stack([d[score_column], d["Monetary_Score"], d[f"Liquidity_{horizon}"], d[growth_column]]), hovertemplate="Date: %{x|%Y-%m-%d}<br>SPX: %{y:,.0f}<br>Macro score: %{customdata[0]:.1f}<br>Monetary: %{customdata[1]:.1f}<br>Liquidity: %{customdata[2]:.1f}<br>Growth: %{customdata[3]:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d[score_column], mode="lines", name=f"Macro score {horizon}", line={"color": "#38bdf8", "width": 2.0}, customdata=np.column_stack([d["SPX"], d["Monetary_Score"], d[f"Liquidity_{horizon}"], d[growth_column], d["Transition_Liquidity"].astype(str)]), hovertemplate="Date: %{x|%Y-%m-%d}<br>Score: %{y:.1f}<br>SPX: %{customdata[0]:,.0f}<br>Monetary: %{customdata[1]:.1f}<br>Liquidity: %{customdata[2]:.1f}<br>Growth: %{customdata[3]:.1f}<br>Liquidity transition: %{customdata[4]}<extra></extra>"), row=2, col=1)
    for y0, y1, color, opacity in ((0, 25, "#ef4444", 0.14), (25, 45, "#f97316", 0.12), (45, 55, "#facc15", 0.10), (55, 75, "#4ade80", 0.10), (75, 100, "#16a34a", 0.14)):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=opacity, line_width=0, row=2, col=1)
    fig.add_hline(y=50, line={"color": "#94a3b8", "dash": "dot"}, row=2, col=1)
    drawdowns = d[d["Drawdown_10pct_Event"].eq(1)]
    if not drawdowns.empty:
        fig.add_trace(go.Scatter(x=drawdowns["Date"], y=drawdowns["SPX"], mode="markers", name="SPX >10% drawdown", marker={"color": "#ef4444", "size": 8, "symbol": "triangle-down"}), row=1, col=1)
        fig.add_trace(go.Scatter(x=drawdowns["Date"], y=drawdowns[score_column], mode="markers", name="Drawdown event", marker={"color": "#ef4444", "size": 8, "symbol": "triangle-down"}), row=2, col=1)
        if show_windows:
            for event_date in drawdowns["Date"]:
                fig.add_vrect(x0=event_date - pd.Timedelta(days=30), x1=event_date, fillcolor="#ef4444", opacity=0.08, line_width=0, row=2, col=1)
    pre_value = pd.to_numeric(d.get(f"PreDrawdown_Avg_{horizon}"), errors="coerce").dropna()
    if not pre_value.empty:
        fig.add_hline(y=float(pre_value.iloc[0]), line={"color": "#ef4444", "dash": "dash"}, annotation_text=f"Pre-drawdown avg {pre_value.iloc[0]:.1f}", row=2, col=1)
    fig.update_yaxes(type="log", title="SPX", row=1, col=1)
    fig.update_yaxes(range=[0, 100], title="Score", row=2, col=1)
    return style_fig(fig, "SPY Macro Outlook History", 620)


def render_spy_macro_outlook_legacy(outlook: SPYMacroOutlook, market_cycle_history: pd.DataFrame) -> None:
    current = outlook.current
    st.markdown("### SPY Macro Outlook")
    st.caption("Point-in-time macro support score for future SPX returns. Higher values are more supportive.")
    cards = st.columns(4)
    for card, horizon in zip(cards, SPY_MACRO_HORIZONS):
        with card:
            render_card(f"{horizon} OUTLOOK", _spy_macro_value(current.get(f"Macro_{horizon}"), " / 100"), [("State", score_state(current.get(f"Macro_{horizon}"))), ("Data", str(current.get("SPYMacroDataStatus", "INSUFFICIENT DATA")))])
    term_frame = pd.DataFrame({"Horizon": list(SPY_MACRO_HORIZONS), "Score": [current.get(f"Macro_{horizon}") for horizon in SPY_MACRO_HORIZONS]})
    term_fig = go.Figure(go.Bar(x=term_frame["Horizon"], y=term_frame["Score"], marker_color=["#38bdf8", "#22c55e", "#facc15", "#f97316"], text=term_frame["Score"].round(1), textposition="outside", hovertemplate="%{x}: %{y:.1f}/100<extra></extra>"))
    term_fig.update_yaxes(range=[0, 100], title="Score")
    st.plotly_chart(style_fig(term_fig, "Current Macro Term Structure", 300), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    st.caption(f"Term structure: {current.get('TermStructureState', 'Insufficient data')}")

    horizon = st.radio("SPY Macro horizon", list(SPY_MACRO_HORIZONS), index=0, horizontal=True, key="spy_macro_horizon")
    growth_column = "Growth_3M" if horizon == "3M" else "Growth_6M" if horizon == "6M" else "Growth_9M" if horizon == "9M" else "BusinessCycle_12M"
    weights = {"3M": (45, 35, 20), "6M": (40, 30, 30), "9M": (25, 55, 20), "12M": (20, 50, 30)}[horizon]
    breakdown = pd.DataFrame(
        [
            {"Block": "Monetary", "Score": current.get("Monetary_Score"), "Weight": weights[0]},
            {"Block": "Liquidity", "Score": current.get(f"Liquidity_{horizon}"), "Weight": weights[1]},
            {"Block": "Growth / Business Cycle", "Score": current.get(growth_column), "Weight": weights[2]},
            {"Block": "Macro Score", "Score": current.get(f"Macro_{horizon}"), "Weight": 100},
        ]
    )
    st.markdown("#### Macro Score Breakdown")
    st.dataframe(breakdown.style.format({"Score": "{:.1f}", "Weight": "{:.0f}%"}), use_container_width=True, hide_index=True)
    with st.expander("Liquidity factor details", expanded=False):
        rows = []
        for label, prefix in (("Global M2", "Global_M2"), ("Global CB Assets", "Global_CB_Assets"), ("US Net Liquidity", "US_Net_Liquidity")):
            rows.append({"Factor": label, "Growth Percentile": current.get(f"{prefix}_GrowthLevelPct"), "Fast Impulse Percentile": current.get(f"{prefix}_FastImpulsePercentile"), "Medium Impulse Percentile": current.get(f"{prefix}_MediumImpulsePercentile"), "Slow Impulse Percentile": current.get(f"{prefix}_SlowImpulsePercentile"), f"{horizon} Score": current.get(f"{prefix}_{horizon}")})
        liquidity_frame = pd.DataFrame(rows)
        st.dataframe(liquidity_frame.style.format({column: "{:.1f}" for column in liquidity_frame.columns if column != "Factor"}, na_rep="N/A"), use_container_width=True, hide_index=True)
    transmission = st.columns(3)
    with transmission[0]:
        render_card("FINANCIAL TRANSMISSION", _spy_macro_value(current.get("Transmission_Score"), " / 100"), [("HY Health", _spy_macro_value(current.get("HY_Health"))), ("ANFCI Health", _spy_macro_value(current.get("ANFCI_Health")))])
    with transmission[1]:
        render_card("TRANSITION SIGNALS", str(current.get("Transition_Liquidity", "Watch")), [("US2Y / USD", f"{current.get('Transition_US2Y', 'Watch')} / {current.get('Transition_USD', 'Watch')}"), ("HY / ANFCI", f"{current.get('Transition_HY', 'Watch')} / {current.get('Transition_ANFCI', 'Watch')}")])
    with transmission[2]:
        render_card("MODIFIERS", str(current.get("T5YIE_Modifier", "Neutral")), [("WTI", str(current.get("WTI_ShockRisk", "Neutral"))), ("JP10Y", str(current.get("JP10Y_ShockRisk", "Neutral")))])
    st.markdown("#### SPY Macro Outlook Narrative")
    st.markdown(f"<div style='color:#cbd5e1;line-height:1.45;'>{html.escape(_spy_macro_narrative(current))}</div>", unsafe_allow_html=True)

    range_selection = st.radio("SPY Macro history range", SPY_MACRO_RANGE_OPTIONS, index=4, horizontal=True, key="spy_macro_range")
    historical = _spy_macro_filtered_history(outlook.history, range_selection)
    show_windows = st.checkbox("Show drawdown windows", value=False, key="spy_macro_drawdown_windows")
    st.plotly_chart(build_spy_macro_history_fig(historical, horizon, show_windows), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with st.expander("Model Details", expanded=False):
        details = build_model_details(outlook, horizon)
        for numeric_column in ("Raw Value", "Level Percentile", "Direction Percentile", "Factor Score", "Weight", "Contribution"):
            details[numeric_column] = pd.to_numeric(details[numeric_column], errors="coerce")
        st.dataframe(details.style.format({"Raw Value": "{:.3f}", "Level Percentile": "{:.1f}", "Direction Percentile": "{:.1f}", "Factor Score": "{:.1f}", "Weight": "{:.2f}", "Contribution": "{:.1f}"}, na_rep="N/A"), use_container_width=True, hide_index=True)
        st.dataframe(outlook.data_quality, use_container_width=True, hide_index=True)
    st.download_button("Export Market Cycle Workbook", data=build_spy_macro_workbook(market_cycle_history, outlook), file_name="market_cycle_with_spy_macro.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _spy_macro_report_state(value: Any) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "Insufficient data", "#94a3b8"
    score = float(value)
    if score >= 75:
        return "Supportive", "#22c55e"
    if score >= 55:
        return "Constructive", "#4ade80"
    if score >= 45:
        return "Neutral / slight headwind", "#facc15"
    if score >= 25:
        return "Mildly unfavorable", "#fb923c"
    return "Unfavorable", "#ef4444"


def _spy_macro_report_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}"


def _spy_macro_report_formula(horizon: str) -> str:
    if horizon == "3M":
        return "Macro_3M = 0.45 Monetary + 0.35 Liquidity_3M + 0.20 Growth_3M"
    if horizon == "6M":
        return "Macro_6M = 0.40 Monetary + 0.30 Liquidity_6M + 0.30 Growth_6M"
    if horizon == "9M":
        return "Macro_9M = 0.25 Monetary + 0.55 Liquidity_9M + 0.20 Growth_9M"
    return "Macro_12M = 0.20 Monetary + 0.50 Liquidity_12M + 0.30 BusinessCycle_12M"


def _render_spy_macro_report_table(outlook: SPYMacroOutlook) -> None:
    current = outlook.current
    weights = {"3M": (45, 35, 20), "6M": (40, 30, 30), "9M": (25, 55, 20), "12M": (20, 50, 30)}
    growth_columns = {"3M": "Growth_3M", "6M": "Growth_6M", "9M": "Growth_9M", "12M": "BusinessCycle_12M"}
    rows = []
    for horizon in SPY_MACRO_HORIZONS:
        final_score = current.get(f"Macro_{horizon}")
        state, color = _spy_macro_report_state(final_score)
        rows.append(
            f"<tr>"
            f"<td><b>{horizon}</b></td>"
            f"<td style='text-align:left;'>{_spy_macro_report_num(current.get('Monetary_Score'))}</td>"
            f"<td style='text-align:left;'>{_spy_macro_report_num(current.get(f'Liquidity_{horizon}'))}</td>"
            f"<td style='text-align:left;'>{_spy_macro_report_num(current.get(growth_columns[horizon]))}</td>"
            f"<td style='text-align:left;background:{color};color:#0f131a;font-weight:900;'>{_spy_macro_report_num(final_score)}</td>"
            f"<td>{html.escape(state)}</td>"
            f"<td style='font-family:monospace;font-size:0.78rem;color:#cbd5e1;'>{html.escape(_spy_macro_report_formula(horizon))}</td>"
            f"</tr>"
        )
    formula_rows = [
        ("Monetary", "MonetaryScore = 0.50 US2Y + 0.43 DXY + 0.07 US10Y"),
        ("Liquidity", "LiquidityScore_H = 0.30 GlobalM2_H + 0.10 GlobalCBAssets_H + 0.60 USNetLiquidity_H"),
        ("Growth / Cycle", "Growth_3M = 0.80 ISM DirectionPct + 0.20 Retail DirectionPct"),
        ("Growth / Cycle", "Growth_6M = 0.75 ISM DirectionPct + 0.25 Retail DirectionPct"),
        ("Growth / Cycle", "Growth_9M = 0.60 ISM DirectionPct + 0.20 ISM Maturity + 0.20 Retail DirectionPct"),
        ("Growth / Cycle", "BusinessCycle_12M = 0.40 ISM DirectionPct + 0.30 ISM Maturity + 0.30 Retail Maturity"),
    ]
    formula_html = "".join(
        f"<tr><td style='font-weight:800;color:#f8fafc;'>{html.escape(label)}</td><td colspan='6' style='font-family:monospace;color:#cbd5e1;'>{html.escape(formula)}</td></tr>"
        for label, formula in formula_rows
    )
    st.markdown(
        f"""
<div class="spy-macro-report" style="border:1px solid #263241;border-radius:6px;overflow:hidden;background:#0f131a;">
  <table style="width:100%;border-collapse:collapse;color:#e5e7eb;font-size:0.82rem;">
    <thead style="background:#315f8f;color:#f8fafc;">
      <tr>
        <th style="padding:0.42rem;text-align:left;">Horizon</th>
        <th style="padding:0.42rem;text-align:left;">Monetary</th>
        <th style="padding:0.42rem;text-align:left;">Liquidity</th>
        <th style="padding:0.42rem;text-align:left;">Growth / Cycle</th>
        <th style="padding:0.42rem;text-align:left;">Final Macro Score</th>
        <th style="padding:0.42rem;text-align:left;">State</th>
        <th style="padding:0.42rem;text-align:left;">Formula</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
      <tr><td colspan="7" style="height:1.15rem;border:0;background:#0f131a;"></td></tr>
      {formula_html}
    </tbody>
  </table>
</div>
<style>
  .spy-macro-report table td, .spy-macro-report table th {{ border-bottom:1px solid #263241; padding:0.42rem; }}
  .spy-macro-report table tbody tr:nth-child(even) {{ background:#141a23; }}
  .spy-macro-report table tbody tr:last-child td {{ border-bottom:0; }}
</style>
""",
        unsafe_allow_html=True,
    )


def render_spy_macro_outlook(outlook: SPYMacroOutlook, market_cycle_history: pd.DataFrame) -> None:
    current = outlook.current
    st.markdown("### SPY Macro Outlook")
    st.caption("Point-in-time macro support score for future SPX returns. Higher values are more supportive.")
    _render_spy_macro_report_table(outlook)
    signal_columns = st.columns(3)
    with signal_columns[0]:
        render_top_level_panel(
            "FINANCIAL TRANSMISSION",
            [
                ("Score", _spy_macro_value(current.get("Transmission_Score"), " / 100")),
                ("HY Health", _spy_macro_value(current.get("HY_Health"))),
                ("ANFCI Health", _spy_macro_value(current.get("ANFCI_Health"))),
            ],
        )
    with signal_columns[1]:
        render_top_level_panel(
            "TRANSITION SIGNALS",
            [
                ("US2Y", text(current.get("Transition_US2Y"))),
                ("USD", text(current.get("Transition_USD"))),
                ("Liquidity", text(current.get("Transition_Liquidity"))),
                ("HY / ANFCI", f"{text(current.get('Transition_HY'))} / {text(current.get('Transition_ANFCI'))}"),
            ],
        )
    with signal_columns[2]:
        render_top_level_panel(
            "MODIFIERS",
            [
                ("T5YIE", text(current.get("T5YIE_Modifier"))),
                ("WTI", text(current.get("WTI_ShockRisk"))),
                ("JP10Y", text(current.get("JP10Y_ShockRisk"))),
                ("Data", text(current.get("SPYMacroDataStatus"))),
            ],
        )
    st.markdown(
        f"<div style='color:#cbd5e1;line-height:1.45;margin:0.2rem 0 0.8rem;'>{html.escape(_spy_macro_narrative(current))}</div>",
        unsafe_allow_html=True,
    )

    horizon = st.radio("History chart horizon", list(SPY_MACRO_HORIZONS), index=0, horizontal=True, key="spy_macro_horizon")
    range_selection = st.radio("SPY Macro history range", SPY_MACRO_RANGE_OPTIONS, index=4, horizontal=True, key="spy_macro_range")
    historical = _spy_macro_filtered_history(outlook.history, range_selection)
    show_windows = st.checkbox("Show drawdown windows", value=False, key="spy_macro_drawdown_windows")
    st.plotly_chart(build_spy_macro_history_fig(historical, horizon, show_windows), use_container_width=True, config=MARKET_CYCLE_PLOTLY_CONFIG)
    with st.expander("Model Details", expanded=False):
        details = build_model_details(outlook, horizon)
        for numeric_column in ("Raw Value", "Level Percentile", "Direction Percentile", "Factor Score", "Weight", "Contribution"):
            details[numeric_column] = pd.to_numeric(details[numeric_column], errors="coerce")
        st.dataframe(details.style.format({"Raw Value": "{:.3f}", "Level Percentile": "{:.1f}", "Direction Percentile": "{:.1f}", "Factor Score": "{:.1f}", "Weight": "{:.2f}", "Contribution": "{:.1f}"}, na_rep="N/A"), use_container_width=True, hide_index=True)
        st.dataframe(outlook.data_quality, use_container_width=True, hide_index=True)
    st.download_button("Export Market Cycle Workbook", data=build_spy_macro_workbook(market_cycle_history, outlook), file_name="market_cycle_with_spy_macro.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def maturity_pct(value: Any) -> str:
    v = safe_float(value)
    return "N/A" if not np.isfinite(v) else f"{v:.1f}%"


def num(value: Any, decimals: int = 1) -> str:
    v = safe_float(value)
    return "N/A" if not np.isfinite(v) else f"{v:.{decimals}f}"


def text(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    return str(value)


def date_text(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return "N/A"
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def is_true_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(safe_float(value)):
        return bool(int(value))
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan

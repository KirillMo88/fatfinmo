from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .macro2 import HORIZONS
from .models import GoldStructuralMacro2Snapshot


SCORE_COLORS = {
    "STRONGLY_UNFAVORABLE": "#ef4444",
    "UNFAVORABLE": "#f97316",
    "NEUTRAL_MIXED": "#facc15",
    "SUPPORTIVE": "#22c55e",
    "STRONGLY_SUPPORTIVE": "#16a34a",
}


def render_gold_structural_macro2(snapshot: GoldStructuralMacro2Snapshot | None) -> None:
    st.markdown("### Gold Structural Macro 2")
    st.caption("Macro conditions for future Gold returns across 3–12 month horizons")
    if snapshot is None or snapshot.history.empty:
        st.info("Gold Structural Macro 2 data is unavailable.")
        return

    history = snapshot.history.copy()
    current = snapshot.current or {}
    render_horizon_table(current)
    render_formula_details()
    render_current_term_structure(current)
    render_macro2_diagnostics(current)
    render_macro2_narrative(current)
    render_structural_macro2_history(history)
    render_gold_macro2_price_chart(history)
    render_macro2_model_details(current)


def render_horizon_table(current: dict[str, Any]) -> None:
    rows = []
    formulas = {
        "3M": "0.60 Liquidity + 0.30 Structural + 0.10 Inflation + Overlay + BC",
        "6M": "0.30 Liquidity + 0.20 Structural + 0.50 Inflation + Overlay + BC",
        "9M": "0.50 Structural + 0.50 Inflation + Overlay + BC",
        "12M": "0.60 Structural + 0.40 Inflation + Overlay + BC",
    }
    for horizon in HORIZONS:
        rows.append(
            {
                "Horizon": horizon,
                "Structural Macro": _fmt_score(current.get("StructuralMacro")),
                "Liquidity": _fmt_score(current.get(f"GoldLiquidity_{horizon}")) if horizon in {"3M", "6M"} else "—",
                "Inflation Relief": _fmt_score(current.get(f"InflationRelief_{horizon}")),
                "Sovereign Stress": _fmt_signed(current.get("SovereignStressOverlay")),
                "Business Cycle": _fmt_signed(current.get(f"Gold_BC_Modifier_{horizon}")),
                "Final Gold Score": _fmt_score(current.get(f"GLD_MACRO_{horizon}")),
                "State": str(current.get(f"GLD_MACRO_{horizon}_State") or "DATA_INCOMPLETE"),
                "Formula": formulas[horizon],
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(_style_score_table(frame), hide_index=True, use_container_width=True)


def _style_score_table(frame: pd.DataFrame):
    def color_score(value: Any) -> str:
        text = str(value)
        for state, color in SCORE_COLORS.items():
            if state in text:
                return f"background-color: {color}; color: #fff; font-weight: 700"
        try:
            score = float(text)
        except Exception:
            return ""
        if score >= 80:
            color = SCORE_COLORS["STRONGLY_SUPPORTIVE"]
        elif score >= 60:
            color = SCORE_COLORS["SUPPORTIVE"]
        elif score >= 40:
            color = SCORE_COLORS["NEUTRAL_MIXED"]
        elif score >= 20:
            color = SCORE_COLORS["UNFAVORABLE"]
        else:
            color = SCORE_COLORS["STRONGLY_UNFAVORABLE"]
        return f"background-color: {color}; color: #fff; font-weight: 700"

    style = frame.style
    if hasattr(style, "map"):
        return style.map(color_score, subset=["Final Gold Score", "State"])
    return style.applymap(color_score, subset=["Final Gold Score", "State"])


def render_formula_details() -> None:
    st.markdown("#### Formula Details")
    st.dataframe(
        pd.DataFrame(
            [
                ["Structural Macro", "0.35 DXYBull + 0.55 RealYieldBull + 0.10 US2YBull"],
                ["Liquidity", "0.20 Global M2 + 0.20 Global CB Assets + 0.60 US Net Liquidity"],
                ["Inflation Relief", "Horizon-specific inverse percentile of T5YIE / T10YIE changes"],
                ["Sovereign Stress", "Curve regime plus JP10Y shock confirmation"],
                ["Business Cycle", "Canonical BusinessCycleState plus horizon modifier"],
            ],
            columns=["Block", "Formula"],
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_current_term_structure(current: dict[str, Any]) -> None:
    st.markdown("#### Current Term Structure")
    scores = [current.get(f"GLD_MACRO_{horizon}") for horizon in HORIZONS]
    if not any(np.isfinite(float(value)) for value in scores if _is_number(value)):
        st.info("Current term structure is unavailable.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(HORIZONS),
            y=[_number(value) for value in scores],
            mode="lines+markers+text",
            text=[_fmt_score(value) for value in scores],
            textposition="top center",
            line={"color": "#f8fafc", "width": 2},
            marker={"color": "#38bdf8", "size": 8},
            hovertemplate="%{x}<br>Score: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 100], title="Score")
    fig.update_layout(height=250, margin={"l": 45, "r": 20, "t": 15, "b": 35}, template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#11161f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    st.dataframe(
        pd.DataFrame(
            {"Horizon": list(HORIZONS), "Score": [_fmt_score(value) for value in scores], "State": [current.get(f"GLD_MACRO_{horizon}_State", "DATA_INCOMPLETE") for horizon in HORIZONS]}
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_macro2_diagnostics(current: dict[str, Any]) -> None:
    st.markdown("#### Diagnostics")
    structural = pd.DataFrame(
        [
            ["Structural Macro Score", _fmt_score(current.get("StructuralMacro"))],
            ["DXYBull", _fmt_score(current.get("DXYBull"))],
            ["RealYieldBull", _fmt_score(current.get("RealYieldBull"))],
            ["US2YBull", _fmt_score(current.get("US2YBull"))],
        ],
        columns=["Structural Components", "Value"],
    )
    signals = pd.DataFrame(
        [
            ["Gold Rate Regime", current.get("Gold_Rate_Regime", "DATA_INCOMPLETE")],
            ["US2Y 13W Change", _fmt_signed(current.get("US2Y_13W_Change"))],
            ["US10Y 13W Change", _fmt_signed(current.get("US10Y_13W_Change"))],
            ["Curve 13W Change", _fmt_signed(current.get("Curve_13W_Change"))],
            ["JP10Y Shock Percentile", _fmt_score(current.get("JP10Y_Shock_Pct"))],
            ["Sovereign Stress Overlay", _fmt_signed(current.get("SovereignStressOverlay"))],
            ["Business Cycle State", current.get("BusinessCycleState", "DATA_INCOMPLETE")],
            ["BC Modifier (selected 3M)", _fmt_signed(current.get("Gold_BC_Modifier_3M"))],
        ],
        columns=["Regime Signals", "Value"],
    )
    modifiers = pd.DataFrame(
        [
            ["Inflation Relief 3M", _fmt_score(current.get("InflationRelief_3M"))],
            ["Inflation Relief 6M", _fmt_score(current.get("InflationRelief_6M"))],
            ["Gold Liquidity 3M", _fmt_score(current.get("GoldLiquidity_3M"))],
            ["Gold Liquidity 6M", _fmt_score(current.get("GoldLiquidity_6M"))],
            ["Liquidity State / Score", _liquidity_state(current)],
            ["Data Quality", _macro2_data_quality(current)],
            ["Latest Data Date", _fmt_date(current.get("date"))],
        ],
        columns=["Modifiers / Data", "Value"],
    )
    cols = st.columns(3)
    for col, frame in zip(cols, [structural, signals, modifiers]):
        with col:
            st.dataframe(frame, hide_index=True, use_container_width=True)


def render_macro2_narrative(current: dict[str, Any]) -> None:
    structural = _number(current.get("StructuralMacro"))
    liquidity = _number(current.get("GoldLiquidity_3M"))
    inflation = _number(current.get("InflationRelief_6M"))
    regime = str(current.get("Gold_Rate_Regime") or "MIXED")
    cycle = str(current.get("BusinessCycleState") or "DATA_INCOMPLETE")
    macro_word = "supportive" if structural >= 60 else "unfavorable" if structural < 40 else "mixed"
    liquidity_word = "supportive" if liquidity >= 60 else "restrictive" if liquidity < 40 else "mixed"
    inflation_word = "relieving pressure" if inflation >= 60 else "adding pressure" if inflation < 40 else "mixed"
    text = (
        f"Gold's conventional structural macro backdrop is {macro_word}. "
        f"Short-term liquidity is {liquidity_word}, while inflation expectations are {inflation_word}. "
        f"The current sovereign curve regime is {regime}. "
        f"The longer-term outlook is additionally affected by the {cycle} Business Cycle state."
    )
    st.markdown("#### Macro Interpretation")
    st.markdown(f"<div style='color:#cbd5e1; line-height:1.45;'>{html.escape(text)}</div>", unsafe_allow_html=True)


def render_structural_macro2_history(history: pd.DataFrame) -> None:
    st.markdown("#### Gold Structural Macro 2 — Structural Components")
    data = history.dropna(subset=["date"]).copy()
    range_label = st.radio("Structural history", ["5Y", "10Y", "2018-Latest", "Full"], index=2, horizontal=True, key="gold_macro2_structural_range")
    latest = pd.to_datetime(data["date"], errors="coerce").max()
    if pd.notna(latest):
        if range_label != "Full":
            cutoff = {"5Y": latest - pd.DateOffset(years=5), "10Y": latest - pd.DateOffset(years=10), "2018-Latest": pd.Timestamp("2018-01-01")}[range_label]
            data = data.loc[pd.to_datetime(data["date"], errors="coerce") >= cutoff]
    fig = go.Figure()
    for column, name, color, width, dash in [
        ("StructuralMacro", "Gold Structural Macro 2", "#00ff66", 2.6, "solid"),
        ("DXYBull", "DXYBull", "#38bdf8", 1.2, "dot"),
        ("RealYieldBull", "RealYieldBull", "#facc15", 1.2, "dot"),
        ("US2YBull", "US2YBull", "#a78bfa", 1.2, "dot"),
    ]:
        if column in data.columns:
            fig.add_trace(go.Scatter(x=data["date"], y=data[column], name=name, mode="lines", line={"color": color, "width": width, "dash": dash}))
    for level, color in [(20, "#ef4444"), (40, "#f97316"), (60, "#facc15"), (80, "#22c55e")]:
        fig.add_hline(y=level, line={"color": color, "dash": "dash", "width": 1})
    fig.update_yaxes(range=[0, 100], title="Score")
    fig.update_layout(height=300, margin={"l": 45, "r": 20, "t": 15, "b": 35}, template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#11161f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def render_gold_macro2_price_chart(history: pd.DataFrame) -> None:
    st.markdown("#### Gold — Log Scale and Final Gold Macro Score")
    horizon = st.radio("Macro horizon", list(HORIZONS), horizontal=True, index=0, key="gold_macro2_horizon")
    score_column = f"GLD_MACRO_{horizon}"
    data = history.dropna(subset=["date"]).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.58, 0.42], subplot_titles=("Gold — Log Scale", f"Final Gold Macro Score — {horizon}"))
    if "gold_price" in data.columns:
        fig.add_trace(go.Scatter(x=data["date"], y=data["gold_price"], name="GLD", line={"color": "#f8fafc", "width": 1.8}, hovertemplate="%{x|%Y-%m-%d}<br>GLD: %{y:.2f}<extra></extra>"), row=1, col=1)
    bands = [(0, 20, "#ef4444"), (20, 40, "#f97316"), (40, 60, "#facc15"), (60, 80, "#86efac"), (80, 100, "#22c55e")]
    for low, high, color in bands:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.08, line_width=0, row=2, col=1)
    if score_column in data.columns:
        liquidity_column = f"GoldLiquidity_{horizon}" if horizon in {"3M", "6M"} else None
        tooltip_columns = [column for column in ["gold_price", "StructuralMacro", liquidity_column, f"InflationRelief_{horizon}", "SovereignStressOverlay", "BusinessCycleState", f"Gold_BC_Modifier_{horizon}", "Gold_Rate_Regime", "JP10Y_Shock_Pct"] if column and column in data.columns]
        customdata = data[tooltip_columns].to_numpy() if tooltip_columns else None
        labels = "<br>".join(f"{column}: %{{customdata[{index}]}}" for index, column in enumerate(tooltip_columns))
        fig.add_trace(go.Scatter(x=data["date"], y=data[score_column], name=f"GLD Macro {horizon}", line={"color": "#38bdf8", "width": 2.2}, customdata=customdata, hovertemplate=f"%{{x|%Y-%m-%d}}<br>Score: %{{y:.1f}}<br>{labels}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(type="log", title="GLD", row=1, col=1)
    fig.update_yaxes(range=[0, 100], title="Score", row=2, col=1)
    fig.update_layout(height=620, margin={"l": 50, "r": 25, "t": 45, "b": 35}, template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#11161f", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def render_macro2_model_details(current: dict[str, Any]) -> None:
    with st.expander("Model Details"):
        rows = []

        structural_inputs = [
            ("DXY", "DXY_13W_Return", "DXY_13W_Pct", "DXYBull", 0.35),
            ("Real Yield", "RealYield_13W_Change", "RealYield_13W_Pct", "RealYieldBull", 0.55),
            ("US2Y", "US2Y_13W_Change", "US2Y_13W_Pct", "US2YBull", 0.10),
        ]
        for horizon in HORIZONS:
            core_structural_weight = {"3M": 0.30, "6M": 0.20, "9M": 0.50, "12M": 0.60}[horizon]
            for factor, raw_column, pct_column, score_column, weight in structural_inputs:
                score = _number(current.get(score_column))
                effective_weight = weight * core_structural_weight
                rows.append([horizon, factor, _fmt_signed(current.get(raw_column)), _fmt_score(current.get(pct_column)), _fmt_score(score), f"{effective_weight:.3f}", _fmt_score(score * effective_weight)])

            if horizon in {"3M", "6M"}:
                liquidity_weight = {"Global M2": 0.20, "Global CB Assets": 0.20, "US Net Liquidity": 0.60}
                suffix = "3M" if horizon == "3M" else "6M"
                core_liquidity_weight = 0.60 if horizon == "3M" else 0.30
                for factor, prefix in [("Global M2", "Global_M2"), ("Global CB Assets", "Global_CB_Assets"), ("US Net Liquidity", "US_Net_Liquidity")]:
                    score = _number(current.get(f"{prefix}_Factor_{suffix}"))
                    effective_weight = liquidity_weight[factor] * core_liquidity_weight
                    rows.append([
                        horizon,
                        f"{factor} Liquidity Factor",
                        _fmt_percent(current.get(f"{prefix}_Growth")),
                        _fmt_score(current.get(f"{prefix}_GrowthPct")),
                        _fmt_score(score),
                        f"{effective_weight:.3f}",
                        _fmt_score(score * effective_weight),
                    ])

            inflation_inputs = {
                "3M": [("T5YIE", "T5YIE_13W_Change", "T5YIE_13W_Change_Pct", 1.00)],
                "6M": [("T5YIE", "T5YIE_26W_Change", "T5YIE_26W_Change_Pct", 0.70), ("T10YIE", "T10YIE_26W_Change", "T10YIE_26W_Change_Pct", 0.30)],
                "9M": [("T5YIE", "T5YIE_26W_Change", "T5YIE_26W_Change_Pct", 0.50), ("T10YIE", "T10YIE_39W_Change", "T10YIE_39W_Change_Pct", 0.50)],
                "12M": [("T5YIE", "T5YIE_39W_Change", "T5YIE_39W_Change_Pct", 0.40), ("T10YIE", "T10YIE_39W_Change", "T10YIE_39W_Change_Pct", 0.60)],
            }[horizon]
            core_inflation_weight = {"3M": 0.10, "6M": 0.50, "9M": 0.50, "12M": 0.40}[horizon]
            for factor, raw_column, pct_column, weight in inflation_inputs:
                score = _number(current.get(pct_column))
                relief = 100.0 - score if np.isfinite(score) else np.nan
                effective_weight = weight * core_inflation_weight
                rows.append([horizon, f"{factor} Inflation Relief", _fmt_signed(current.get(raw_column)), _fmt_score(score), _fmt_score(relief), f"{effective_weight:.3f}", _fmt_score(relief * effective_weight)])

        for horizon in HORIZONS:
            structural_score = _number(current.get("StructuralMacro"))
            structural_weight = {"3M": 0.30, "6M": 0.20, "9M": 0.50, "12M": 0.60}[horizon]
            rows.append([horizon, "Structural Macro (aggregate)", "n/a", "n/a", _fmt_score(structural_score), f"{structural_weight:.2f}", _fmt_score(structural_score * structural_weight)])
            if horizon in {"3M", "6M"}:
                liquidity_weight = 0.60 if horizon == "3M" else 0.30
                liquidity_score = _number(current.get(f"GoldLiquidity_{horizon}"))
                rows.append([horizon, "Liquidity (aggregate)", "n/a", "n/a", _fmt_score(liquidity_score), f"{liquidity_weight:.2f}", _fmt_score(liquidity_score * liquidity_weight)])
            inflation_weight = {"3M": 0.10, "6M": 0.50, "9M": 0.50, "12M": 0.40}[horizon]
            inflation_score = _number(current.get(f"InflationRelief_{horizon}"))
            rows.append([horizon, "Inflation Relief (aggregate)", "n/a", "n/a", _fmt_score(inflation_score), f"{inflation_weight:.2f}", _fmt_score(inflation_score * inflation_weight)])
            rows.extend(
                [
                    [horizon, "Core Macro", "n/a", "n/a", _fmt_score(current.get(f"CoreMacro_{horizon}")), "1.00", _fmt_score(current.get(f"CoreMacro_{horizon}"))],
                    [horizon, "Sovereign Overlay", "n/a", "n/a", _fmt_signed(current.get("SovereignStressOverlay")), "n/a", _fmt_signed(current.get("SovereignStressOverlay"))],
                    [horizon, "Business Cycle Modifier", "n/a", "n/a", _fmt_signed(current.get(f"Gold_BC_Modifier_{horizon}")), "n/a", _fmt_signed(current.get(f"Gold_BC_Modifier_{horizon}"))],
                    [horizon, "Final Gold Score", "n/a", "n/a", _fmt_score(current.get(f"GLD_MACRO_{horizon}")), "clamp 0-100", _fmt_score(current.get(f"GLD_MACRO_{horizon}_BeforeClamp"))],
                ]
            )
        st.dataframe(pd.DataFrame(rows, columns=["Horizon", "Factor", "Raw Value", "Percentile", "Bull / Relief Score", "Weight", "Contribution / Reconciliation"]), hide_index=True, use_container_width=True)


def _is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _fmt_score(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:.1f}"


def _fmt_signed(value: Any) -> str:
    number = _number(value)
    return "n/a" if not np.isfinite(number) else f"{number:+.1f}"


def _fmt_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return "n/a"


def _liquidity_state(current: dict[str, Any]) -> str:
    score = _number(current.get("GoldLiquidity_3M"))
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score >= 60:
        return f"SUPPORTIVE ({score:.1f})"
    if score < 40:
        return f"RESTRICTIVE ({score:.1f})"
    return f"MIXED ({score:.1f})"


def _macro2_data_quality(current: dict[str, Any]) -> str:
    required = ["StructuralMacro", "InflationRelief_3M", "InflationRelief_6M", "InflationRelief_9M", "InflationRelief_12M", "GoldLiquidity_3M", "GoldLiquidity_6M"]
    return "COMPLETE" if all(_is_number(current.get(column)) for column in required) else "PARTIAL_DATA"

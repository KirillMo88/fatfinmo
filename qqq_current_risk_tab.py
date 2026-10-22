from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_cycle_tab import MARKET_CYCLE_PLOTLY_CONFIG, MARKET_CYCLE_TTL_SECONDS, style_fig
from qqq_current_risk import qqq_component_columns
from qqq_current_risk_data import QQQCurrentRiskSnapshot, build_qqq_current_risk_snapshot


@st.cache_data(show_spinner=True, ttl=MARKET_CYCLE_TTL_SECONDS)
def load_qqq_current_risk_cached(refresh_nonce: int = 0) -> QQQCurrentRiskSnapshot:
    _ = refresh_nonce
    return build_qqq_current_risk_snapshot()


def render_qqq_current_risk(refresh_nonce: int = 0) -> None:
    st.markdown("### Current Risk")
    try:
        snapshot = load_qqq_current_risk_cached(refresh_nonce)
    except Exception as exc:
        st.error(f"QQQ Current Risk data is unavailable: {exc}")
        return
    if not snapshot.variants:
        st.warning("QQQ Current Risk data is unavailable. " + _source_caption(snapshot.source_status))
        return

    selected = st.segmented_control(
        "Volatility Indicator",
        options=["VIX", "VXN"],
        default="VIX",
        key="qqq_current_risk_volatility_indicator",
    ) or "VIX"
    history = snapshot.variants.get(selected, pd.DataFrame())
    if history.empty:
        st.warning(f"QQQ Current Risk — {selected} is DATA INCOMPLETE.")
        return
    current = history.iloc[-1].to_dict()
    render_status_panel(current, selected, snapshot.as_of)

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("#### Price Cycle Vulnerability")
        st.dataframe(
            pvc_table(current).style.format(na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
            height=283,
        )
        st.caption(
            f"PVC: {_number(current.get('PVC'), 0)} / 100 | "
            f"PVC MAX10D: {_number(current.get('PVC_MAX10D'), 0)} / 100 | "
            f"Activation threshold: 50 | Conditions met: {pvc_conditions_met(current)}/7"
        )
    with detail_right:
        st.markdown("#### Volatility Activation")
        st.dataframe(
            activation_table(current, selected).style.format(na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
            height=178,
        )
        activation = "ACTIVE" if _flag(current.get("CurrentRiskActivation")) else "INACTIVE"
        st.caption(f"VOLATILITY ACTIVATION: {activation}")

        st.markdown("#### Escalation Score")
        st.dataframe(
            escalation_table(current, selected).style.format(na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
            height=248,
        )
        st.caption(
            f"ESCALATION SCORE: {_number(current.get('CurrentRiskEscalationScore'), 0)} / 6 | "
            f"Raw channels: {_number(current.get('CurrentRiskEscalationScoreRaw'), 0)} / 6"
        )

    st.markdown("#### Diagnostic Risk Components")
    diagnostics = diagnostic_table(current, selected)
    diagnostic_col, table_col = st.columns([1.45, 1.0])
    with diagnostic_col:
        st.plotly_chart(
            build_diagnostic_fig(diagnostics, current, selected),
            use_container_width=True,
            config=MARKET_CYCLE_PLOTLY_CONFIG,
        )
    with table_col:
        st.dataframe(
            diagnostics.style.format({"Score": "{:.1f}"}, na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
            height=300,
        )
        st.caption(
            f"Strict component average: {_number(current.get('CurrentRiskComponentAverage'))} | "
            f"Primary drivers: {current.get('HistoricalCurrentRiskPrimaryDrivers', 'DATA INCOMPLETE')}"
        )

    chart_history = history.loc[pd.to_datetime(history["Date"], errors="coerce").ge(pd.Timestamp("2015-01-01"))]
    st.plotly_chart(
        build_qqq_current_risk_signals_fig(chart_history, selected),
        use_container_width=True,
        config=MARKET_CYCLE_PLOTLY_CONFIG,
    )
    st.caption(
        f"Model: {current.get('CurrentRiskConfiguration', 'N/A')} | "
        f"Sources: {_source_caption(snapshot.source_status)}"
    )


def render_status_panel(current: dict[str, Any], source: str, as_of: pd.Timestamp | None) -> None:
    state = str(current.get("CurrentMarketRiskState", "DATA INCOMPLETE"))
    color = {
        "NORMAL": "#94a3b8",
        "LOW CONFIRMATION": "#22c55e",
        "MODERATE": "#facc15",
        "HIGH RISK": "#f97316",
        "RED FLAG": "#ef4444",
    }.get(state, "#64748b")
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:10px 14px;background:#141a23'>"
        f"<b>CURRENT MARKET RISK — {source}</b><br><span style='font-size:1.45rem;color:{color}'>{state}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    row_one = st.columns(5)
    values_one = [
        ("Confirmation", current.get("CurrentRiskConfirmation", "NONE")),
        ("Escalation", f"{_number(current.get('CurrentRiskEscalationScore'), 0)} / 6"),
        ("Direction", current.get("CurrentMarketRiskDirection", "DATA INCOMPLETE")),
        ("PVC", f"{_number(current.get('PVC'), 0)} / 100"),
        ("PVC MAX10D", f"{_number(current.get('PVC_MAX10D'), 0)} / 100"),
    ]
    for column, (label, value) in zip(row_one, values_one):
        column.metric(label, value)
    last_event = pd.to_datetime(current.get("CurrentRiskLastEventDate"), errors="coerce")
    last_event_text = last_event.strftime("%Y-%m-%d") if pd.notna(last_event) else "—"
    latest_text = as_of.strftime("%Y-%m-%d") if as_of is not None else "—"
    row_two = st.columns(5)
    values_two = [
        (source, _number(current.get("VolatilityLevel"), 2)),
        (f"{source} 3D Change", _percent(current.get("Volatility3DChange"))),
        ("Activation", "ACTIVE" if _flag(current.get("CurrentRiskActivation")) else "INACTIVE"),
        ("Last Event", last_event_text),
        ("Latest Observation", latest_text),
    ]
    for column, (label, value) in zip(row_two, values_two):
        column.metric(label, value)


def pvc_table(current: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Price Extension", _percent(current.get("QQQ_Extension200W")), "≥40%", "PVC_ExtensionCondition", "PVC_ExtensionContribution", 20),
        ("Near 20D High", _percent(current.get("QQQ_NearHighRatio")), "≥99%", "PVC_NearHighCondition", "PVC_NearHighContribution", 10),
        ("ROC Deterioration / High", f"{_number(current.get('QQQ_ROCDeterioration'), 1)} p.p. / {_number(current.get('QQQ_ROC12'), 1)}%", "≥5 p.p. OR ≥6%", "PVC_ROCCondition", "PVC_ROCContribution", 15),
        ("Breadth Deterioration / High", f"{_number(current.get('NDFIDeterioration'), 1)} p.p. / {_number(current.get('NDFI'), 1)}%", "≥12 p.p. OR ≥80%", "PVC_BreadthCondition", "PVC_BreadthContribution", 15),
        ("Daily RSI Divergence", _number(current.get("QQQ_DailyRSIDecline"), 1), "Near high AND ≥12", "PVC_DailyRSIDivergence", "PVC_DailyRSIDivergenceContribution", 15),
        ("Weekly RSI Divergence", _number(current.get("QQQ_WeeklyRSIDecline"), 1), "Near high AND ≥8", "PVC_WeeklyRSIDivergence", "PVC_WeeklyRSIDivergenceContribution", 10),
        ("RSI Blow-off", _number(current.get("QQQ_DailyRSI14"), 1), "≥76", "PVC_RSIBlowoff", "PVC_RSIBlowoffContribution", 15),
    ]
    return pd.DataFrame(
        [
            {
                "Component": label,
                "Actual Value": actual,
                "Threshold": threshold,
                "Contribution": f"{_number(current.get(contribution), 0)}/{weight}",
                "Status": _status(current.get(flag)),
            }
            for label, actual, threshold, flag, contribution, weight in rows
        ]
    )


def activation_table(current: dict[str, Any], source: str) -> pd.DataFrame:
    rows = [
        ("PVC_MAX10D", _number(current.get("PVC_MAX10D"), 0), "≥50", _safe(current.get("PVC_MAX10D")) >= 50),
        (f"{source} Breakout", f"{_number(current.get('VolatilityLevel'), 2)} vs {_number(current.get('VolatilityPrevious3DHigh'), 2)}", "Current higher", current.get("VolatilityBreakout")),
        (f"{source} 3D Change", _percent(current.get("Volatility3DChange")), "≥10%", _safe(current.get("Volatility3DChange")) >= 0.10),
        ("QQQ Near High", _percent(_safe(current.get("QQQ_RawClose")) / _safe(current.get("QQQ_20D_High"))), "≥95%", current.get("QQQActivationNearHigh")),
    ]
    return pd.DataFrame(
        [{"Condition": name, "Current Value": value, "Threshold": threshold, "Status": _status(flag)} for name, value, threshold, flag in rows]
    )


def escalation_table(current: dict[str, Any], source: str) -> pd.DataFrame:
    calm_threshold = 16 if source == "VIX" else 18
    rows = [
        (f"Calm {source}", _number(current.get("CurrentRiskCalmVolatilityLevel"), 2), f"≤{calm_threshold}", current.get("CurrentRiskCalmVolatility")),
        ("HY Level", f"{_number(current.get('HY_OAS'), 2)}%", "≥3.25%", current.get("CurrentRiskHYLevelConfirmation")),
        ("HY Widening", f"{_number(current.get('HY_OAS_5D_Change'), 2)} p.p.", ">0", current.get("CurrentRiskHYWidening")),
        ("Weak Breadth", f"{_number(current.get('NDFI'), 1)}%", "≤50%", current.get("CurrentRiskWeakBreadth")),
        ("Breadth Collapse", f"{_number(current.get('NDFI_5D_Change'), 1)} p.p.", "≤−12 p.p.", current.get("CurrentRiskBreadthCollapse")),
        ("IWM/QQQ Divergence", _percent(current.get("IWMQQQDivergence")), "≥3%", current.get("CurrentRiskTopConfirmation")),
    ]
    return pd.DataFrame(
        [
            {"Channel": name, "Actual Value": value, "Threshold": threshold, "Points": "1/1" if _flag(flag) else "0/1", "Status": _status(flag)}
            for name, value, threshold, flag in rows
        ]
    )


def diagnostic_table(current: dict[str, Any], source: str) -> pd.DataFrame:
    labels = [
        "Historical Drawdown Risk",
        "Price Cycle Vulnerability Risk",
        "Breadth Risk",
        "RSI Divergence Risk",
        f"{source} Risk",
        "High Beta Risk",
        "HY Credit Risk",
    ]
    rows = []
    for label, column in zip(labels, qqq_component_columns()):
        score = _safe(current.get(column))
        rows.append({"Component": label, "Score": score, "Classification": current.get(f"{column}State", "DATA INCOMPLETE")})
    return pd.DataFrame(rows)


def build_diagnostic_fig(table: pd.DataFrame, current: dict[str, Any], source: str) -> go.Figure:
    colors = {"LOW": "#22c55e", "MODERATE": "#facc15", "ELEVATED": "#f97316", "HIGH": "#ef4444", "DATA INCOMPLETE": "#64748b"}
    fig = go.Figure(
        go.Bar(
            x=table["Score"],
            y=table["Component"],
            orientation="h",
            marker={"color": [colors.get(str(value), "#64748b") for value in table["Classification"]]},
            customdata=table[["Classification"]].to_numpy(),
            hovertemplate="%{y}<br>Score: %{x:.1f}<br>State: %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100], title="Risk (0–100)")
    return style_fig(fig, f"QQQ Current Risk Components — {source} — {current.get('CurrentMarketRiskState', 'N/A')}", 400)


def build_qqq_current_risk_signals_fig(history: pd.DataFrame, source: str) -> go.Figure:
    fig = go.Figure()
    if history.empty:
        return style_fig(fig, f"Log QQQ with Current Risk Signals — {source}", 560)
    d = history.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", "QQQ_RawClose"]).sort_values("Date")
    fig.add_trace(
        go.Scatter(
            x=d["Date"], y=d["QQQ_RawClose"], mode="lines", name="QQQ",
            line={"color": "#f8fafc", "width": 1.7},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>QQQ raw close: %{y:,.2f}<extra></extra>",
        )
    )
    categories = [
        ("LOW CONFIRMATION", "#22c55e", "circle", 8, "legendonly"),
        ("MODERATE", "#facc15", "diamond", 9, "legendonly"),
        ("HIGH RISK", "#f97316", "triangle-up", 10, True),
        ("RED FLAG", "#3b82f6", "diamond", 11, True),
        ("RED FLAG + TOP CONFIRMATION", "#ef4444", "star", 12, True),
        ("RED FLAG + CREDIT CONFIRMATION", "#ef4444", "diamond", 13, True),
    ]
    custom_columns = [
        "QQQ_RawClose", "CurrentRiskDisplayClass", "PVC", "PVC_MAX10D", "VolatilityLevel",
        "Volatility3DChange", "VolatilityBreakout", "CurrentRiskEscalationScore", "HY_OAS",
        "HY_OAS_5D_Change", "NDFI", "NDFI_5D_Change", "IWMQQQDivergence", "CurrentRiskCreditConfirmation",
    ]
    for category, color, symbol, size, visible in categories:
        points = d.loc[
            d["CurrentRiskEventMarker"].fillna(False).astype(bool)
            & d["CurrentRiskEventMarkerClass"].eq(category)
        ].copy()
        if points.empty:
            continue
        custom = points.reindex(columns=custom_columns).to_numpy(dtype=object)
        fig.add_trace(
            go.Scatter(
                x=points["Date"], y=points["QQQ_RawClose"], mode="markers", name=category, visible=visible,
                marker={"color": color, "symbol": symbol, "size": size, "line": {"color": "#0f131a", "width": 1}},
                customdata=custom,
                hovertemplate=(
                    f"Date: %{{x|%Y-%m-%d}}<br>QQQ Close: %{{customdata[0]:,.2f}}<br>Volatility source: {source}<br>"
                    "Market Risk State: %{customdata[1]}<br>PVC: %{customdata[2]:.0f}<br>PVC_MAX10D: %{customdata[3]:.0f}<br>"
                    "Volatility: %{customdata[4]:.2f}<br>Volatility 3D Change: %{customdata[5]:.1%}<br>"
                    "Volatility Breakout: %{customdata[6]}<br>Escalation Score: %{customdata[7]:.0f}/6<br>"
                    "HY OAS: %{customdata[8]:.2f}<br>HY OAS 5D Change: %{customdata[9]:.2f}<br>"
                    "NDFI: %{customdata[10]:.1f}<br>NDFI 5D Change: %{customdata[11]:.1f}<br>"
                    "IWM/QQQ Divergence: %{customdata[12]:.1%}<br>Credit Confirmation: %{customdata[13]}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(type="log", title="QQQ log")
    fig.update_xaxes(range=[pd.Timestamp("2015-01-01"), d["Date"].max()])
    return style_fig(fig, f"Log QQQ with Current Risk Signals — {source}", 560)


def pvc_conditions_met(current: dict[str, Any]) -> int:
    return sum(
        _flag(current.get(column))
        for column in [
            "PVC_ExtensionCondition", "PVC_NearHighCondition", "PVC_ROCCondition", "PVC_BreadthCondition",
            "PVC_DailyRSIDivergence", "PVC_WeeklyRSIDivergence", "PVC_RSIBlowoff",
        ]
    )


def _status(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return "Met" if _flag(value) else "Not Met"


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "active", "met"}
    return bool(value) if pd.notna(value) else False


def _safe(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _number(value: Any, digits: int = 1) -> str:
    number = _safe(value)
    return "N/A" if not np.isfinite(number) else f"{number:.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    number = _safe(value)
    return "N/A" if not np.isfinite(number) else f"{100.0 * number:.{digits}f}%"


def _source_caption(status: dict[str, str]) -> str:
    return " | ".join(f"{name}: {value}" for name, value in status.items())

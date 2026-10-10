from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from btc_cycle import BTC_ACTUAL_HALVINGS, BTC_PROJECTED_HALVING
from btc_data import load_btc_weekly_price
from finance_core import download_completed_ohlcv
from gold_regime.service import load_gold_mcp_weekly
from portfolio_prioritization import (
    DEFAULT_CONFIG,
    build_portfolio_snapshot,
    completed_monthly_ratio,
    completed_weekly_ratio,
    reconcile_btc_gold_signals,
)


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
HALVING_COLORS = {
    "EARLY_EXPANSION": "rgba(16,185,129,0.13)",
    "LATE_EXPANSION": "rgba(132,204,22,0.13)",
    "CYCLE_TOP_RISK": "rgba(245,158,11,0.15)",
    "BEAR_DELEVERAGING": "rgba(239,68,68,0.13)",
    "BOTTOMING_TRANSITION": "rgba(59,130,246,0.13)",
    "ACCUMULATION_PRE_HALVING": "rgba(168,85,247,0.13)",
}


def _to_completed_weekly(series: pd.Series) -> pd.Series:
    values = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(), index=pd.to_datetime(series.index, errors="coerce", utc=True).tz_localize(None))
    values = values.loc[~values.index.isna()].dropna().sort_index()
    return values.resample("W-FRI").last().dropna() if not values.empty else pd.Series(dtype="float64")


@st.cache_data(show_spinner=False, ttl=21600)
def load_portfolio_model() -> dict[str, Any]:
    spx = download_completed_ohlcv("^GSPC", period="max")
    gold_weekly = load_gold_mcp_weekly()
    # Use the long canonical weekly Gold series for both engines. Normalizing both
    # legs to completed W-FRI observations preserves the pre-1970 history and
    # guarantees that every ratio observation has one synchronized timestamp.
    spx_weekly = _to_completed_weekly(spx.get("Close", pd.Series(dtype="float64")))
    gold_weekly = _to_completed_weekly(gold_weekly)
    spx_monthly = completed_monthly_ratio(spx_weekly, gold_weekly)
    btc_weekly = load_btc_weekly_price()
    btc_ratio = completed_weekly_ratio(btc_weekly, gold_weekly)
    snapshot = build_portfolio_snapshot(spx_monthly, btc_ratio)
    snapshot["diagnostics"] = {
        "spx_source": "Yahoo Finance ^GSPC completed weekly closes (W-FRI)",
        "gold_source": "TradingView TVC:GOLD completed weekly closes (W-FRI)",
        "btc_source": "Canonical BTC Cycle weekly history (TradingView INDEX:BTCUSD + Yahoo BTC-USD)",
        "spx_gold_start": None if spx_monthly.empty else spx_monthly["Date"].min(),
        "spx_gold_end": None if spx_monthly.empty else spx_monthly["Date"].max(),
        "btc_gold_start": None if btc_ratio.empty else btc_ratio["Date"].min(),
        "btc_gold_end": None if btc_ratio.empty else btc_ratio["Date"].max(),
    }
    return snapshot


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _fmt(value: Any, digits: int = 2) -> str:
    numeric = _number(value)
    return "n/a" if not np.isfinite(numeric) else f"{numeric:,.{digits}f}"


def _date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d") if pd.notna(value) else "n/a"
    except Exception:
        return "n/a"


def _status_label(row: dict[str, Any], direction: str) -> str:
    if bool(row.get("sustained_confirmation_reached")):
        return f"{direction} Sustained"
    run = int(_number(row.get(f"{direction}_condition_run"))) if np.isfinite(_number(row.get(f"{direction}_condition_run"))) else 0
    return f"{direction} Initial ({run}/3)" if run else f"{direction} Pending"


def _evidence_html(items: list[tuple[str, bool, str]]) -> str:
    lines = []
    for label, passed, value in items:
        icon = "✓" if passed else "✗"
        color = "#4ade80" if passed else "#f87171"
        suffix = f" <span style='color:#94a3b8'>({html.escape(value)})</span>" if value else ""
        lines.append(f"<div style='margin:0.18rem 0;color:{color};font-weight:700'>{icon} {html.escape(label)}{suffix}</div>")
    return "".join(lines)


def _card(title: str, rows: list[tuple[str, str]], evidence: list[tuple[str, bool, str]], observation: Any) -> None:
    detail = "".join(
        f"<div style='display:grid;grid-template-columns:1fr auto;gap:0.5rem;margin:0.25rem 0'>"
        f"<span style='color:#94a3b8'>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
        for label, value in rows
    )
    met = sum(bool(item[1]) for item in evidence)
    st.markdown(
        f"""
<div style="border:1px solid #334155;border-radius:10px;padding:1rem;background:#0f141d;height:100%">
  <div style="font-size:1.05rem;font-weight:900;margin-bottom:0.7rem">{html.escape(title)}</div>
  {detail}
  <div style="font-size:0.72rem;color:#7dd3fc;font-weight:800;text-transform:uppercase;margin-top:0.8rem">Structural Evidence</div>
  {_evidence_html(evidence)}
  <div style="margin-top:0.45rem;font-weight:800">Conditions Met: {met}/{len(evidence)}</div>
  <div style="font-size:0.68rem;color:#64748b;margin-top:0.7rem">Latest completed observation: {_date(observation)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _range_frame(frame: pd.DataFrame, choice: str) -> pd.DataFrame:
    if frame.empty or choice == "MAX":
        return frame
    years = int(choice.rstrip("Y"))
    return frame.loc[pd.to_datetime(frame["Date"]) >= pd.to_datetime(frame["Date"]).max() - pd.DateOffset(years=years)]


def _add_position_zones(
    fig: go.Figure,
    frame: pd.DataFrame,
    position_col: str,
    colors: dict[str, str],
    row: int | None = None,
) -> None:
    if frame.empty:
        return
    data = frame[["Date", position_col]].dropna().copy()
    if data.empty:
        return
    data["group"] = data[position_col].ne(data[position_col].shift()).cumsum()
    for _, segment in data.groupby("group"):
        position = str(segment[position_col].iloc[0])
        kwargs = {
            "x0": segment["Date"].iloc[0],
            "x1": segment["Date"].iloc[-1],
            "fillcolor": colors.get(position, "rgba(148,163,184,0.08)"),
            "opacity": 1,
            "line_width": 0,
            "layer": "below",
        }
        if row is None:
            fig.add_vrect(**kwargs)
        else:
            fig.add_vrect(**kwargs, row=row, col=1)


def _style_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title, height=height, margin={"l": 50, "r": 35, "t": 55, "b": 40},
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font={"color": "#e2e8f0", "size": 11},
        hovermode="x unified", legend={"orientation": "h", "y": -0.12},
    )
    fig.update_xaxes(gridcolor="#253041", rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#253041")
    return fig


def _spx_gold_chart(history: pd.DataFrame, choice: str) -> go.Figure:
    data = _range_frame(history, choice)
    fig = go.Figure()
    _add_position_zones(fig, data, "position", {"SPX": "rgba(56,189,248,0.10)", "Gold": "rgba(234,179,8,0.11)"})
    specs = [
        ("close", "SPX / Gold", "#f8fafc", None, 2.3),
        ("sma50", "SMA50M", "#facc15", None, 1.8),
        ("upper", "SMA50M × 1.05", "#fb7185", "dash", 1.2),
        ("lower", "SMA50M × 0.95", "#4ade80", "dash", 1.2),
        ("sma100", "SMA100M", "#60a5fa", "dot", 1.2),
        ("sma200", "SMA200M", "#c084fc", "dot", 1.2),
    ]
    for column, name, color, dash, width in specs:
        fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name=name, mode="lines", line={"color": color, "width": width, "dash": dash or "solid"}))
    switches = data.loc[data["trading_switch"]]
    fig.add_trace(go.Scatter(x=switches["Date"], y=switches["close"], name="Trading Switch", mode="markers", marker={"symbol": "diamond", "size": 10, "color": "#ffffff"}))
    for direction, color in (("SPX", "#38bdf8"), ("Gold", "#facc15")):
        initial = data.loc[data[f"{direction}_initial_event"]]
        sustained = data.loc[data[f"{direction}_sustained_event"]]
        fig.add_trace(go.Scatter(x=initial["Date"], y=initial["close"], name=f"{direction} Initial", mode="markers", marker={"symbol": "circle-open", "size": 8, "color": color}))
        fig.add_trace(go.Scatter(x=sustained["Date"], y=sustained["close"], name=f"{direction} Sustained", mode="markers", marker={"symbol": "star", "size": 11, "color": color}))
    fig.update_yaxes(title="SPX / Gold ratio")
    return _style_figure(fig, "SPX / Gold Trading System", 520)


def _btc_gold_chart(history: pd.DataFrame, choice: str) -> go.Figure:
    data = _range_frame(history, choice)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.60, 0.23, 0.17], vertical_spacing=0.04)
    _add_position_zones(fig, data, "position", {"BTC": "rgba(249,115,22,0.10)", "Gold": "rgba(234,179,8,0.11)"}, row=1)
    for column, name, color, dash in (
        ("ratio", "BTC / Gold", "#f8fafc", "solid"),
        ("regression_trend", "208W Regression", "#fb923c", "solid"),
        ("channel_lower", f"{DEFAULT_CONFIG.btc_buy_percentile:.1f}% Channel", "#4ade80", "dash"),
        ("channel_upper", f"{DEFAULT_CONFIG.btc_sell_percentile:.1f}% Channel", "#fb7185", "dash"),
        ("sma50m", "SMA50M", "#60a5fa", "dot"),
    ):
        fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name=name, mode="lines", line={"color": color, "dash": dash, "width": 1.5}), row=1, col=1)
    switches = data.loc[data["trading_switch"]]
    fig.add_trace(go.Scatter(x=switches["Date"], y=switches["ratio"], name="Trading Switch", mode="markers", marker={"symbol": "diamond", "size": 10, "color": "#ffffff"}), row=1, col=1)
    for direction, color in (("BTC", "#fb923c"), ("Gold", "#facc15")):
        sustained = data.loc[data[f"{direction}_sustained_event"]]
        fig.add_trace(go.Scatter(x=sustained["Date"], y=sustained["ratio"], name=f"{direction} Sustained", mode="markers", marker={"symbol": "star", "size": 10, "color": color}), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["channel_percentile"], name="Channel Percentile", mode="lines", line={"color": "#38bdf8"}), row=2, col=1)
    for level, color, label in (
        (DEFAULT_CONFIG.btc_buy_percentile, "#4ade80", f"Buy BTC {DEFAULT_CONFIG.btc_buy_percentile:.1f}%"),
        (50.0, "#94a3b8", "Median"),
        (DEFAULT_CONFIG.btc_sell_percentile, "#fb7185", f"Sell BTC {DEFAULT_CONFIG.btc_sell_percentile:.1f}%"),
    ):
        fig.add_hline(y=level, line={"color": color, "dash": "dash", "width": 1}, annotation_text=label, row=2, col=1)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["halving_progress"], name="Halving Progress", mode="lines", line={"color": "#a78bfa", "width": 1.5}), row=3, col=1)
    phases = data[["Date", "halving_phase"]].dropna().copy()
    phases["group"] = phases["halving_phase"].ne(phases["halving_phase"].shift()).cumsum()
    for _, segment in phases.groupby("group"):
        phase = str(segment["halving_phase"].iloc[0])
        fig.add_vrect(x0=segment["Date"].iloc[0], x1=segment["Date"].iloc[-1], fillcolor=HALVING_COLORS.get(phase, "rgba(148,163,184,0.08)"), opacity=1, line_width=0, layer="below", row=3, col=1)
    for event in list(BTC_ACTUAL_HALVINGS) + [BTC_PROJECTED_HALVING]:
        if not data.empty and data["Date"].min() <= event <= data["Date"].max():
            fig.add_vline(x=event, line={"color": "#a78bfa", "dash": "dot", "width": 1}, row=3, col=1)
    fig.update_yaxes(title="BTC / Gold", type="log", row=1, col=1)
    fig.update_yaxes(title="Percentile", range=[-3, 103], row=2, col=1)
    fig.update_yaxes(title="Halving %", range=[0, 100], row=3, col=1)
    return _style_figure(fig, "BTC / Gold Trading System", 760)


def _interpretation(current: dict[str, Any]) -> str:
    if not current.get("data_complete"):
        return "The ranking is unavailable because one or both trading engines are not initialized."
    spx = current["spx_gold"]
    btc = current["btc_gold"]
    lines = [
        f"{current['priority_asset']} currently ranks first in **{current['ranking']}**. "
        f"SPX/Gold holds **{spx.get('position')}** under the completed-month SMA50M ±5% state machine, "
        f"while BTC/Gold holds **{btc.get('position')}** under the 208W residual-percentile state machine."
    ]
    lines.append(
        f"SPX/Gold structural confirmation is **{spx.get('confirmation_health')}**; "
        f"BTC/Gold structural confirmation is **{btc.get('confirmation_health')}**."
    )
    recovery = [str(value) for value in (spx.get("recovery_phase"), btc.get("recovery_phase")) if value and value != "None / Transition"]
    if recovery:
        lines.append("Current recovery diagnostics: **" + "** and **".join(recovery) + "**. These diagnostics do not change either trading position.")
    if spx.get("position") == "Gold":
        lines.append(f"The next Layer 1 switch to SPX requires a completed monthly SPX/Gold close above **{_fmt(spx.get('upper'), 3)}**.")
    else:
        lines.append(f"The next Layer 1 switch to Gold requires a completed monthly SPX/Gold close below **{_fmt(spx.get('lower'), 3)}**.")
    if btc.get("position") == "BTC":
        lines.append(
            f"The next Layer 2 switch to Gold requires a completed weekly channel percentile at or above "
            f"**{DEFAULT_CONFIG.btc_sell_percentile:.1f}%**."
        )
    else:
        lines.append("The next Layer 2 switch to BTC requires a completed weekly channel percentile at or below **0.5%**.")
    return "\n\n".join(lines)


def render_portfolio_tab() -> None:
    st.subheader("Portfolio")
    st.caption("Asset Prioritization V5.1 · deterministic SPX, Gold and BTC hierarchy · no weights or liquidity overlays")
    try:
        snapshot = load_portfolio_model()
    except Exception as exc:
        st.error(f"Portfolio inputs are unavailable: {exc}")
        return
    current = snapshot.get("current", {})
    if not current:
        st.warning("Portfolio ranking is unavailable because synchronized completed observations were not returned.")
        return
    ranking = current.get("ranking") or "INCOMPLETE DATA — ENGINE UNINITIALIZED"
    priority = current.get("priority_asset") or "n/a"
    spx = current["spx_gold"]
    btc = current["btc_gold"]
    st.markdown(
        f"""
<div style="border:1px solid #334155;border-radius:12px;padding:1.2rem;text-align:center;background:linear-gradient(135deg,#101722,#111827)">
  <div style="color:#7dd3fc;font-size:0.72rem;font-weight:900;letter-spacing:0.08em">ASSET PRIORITIZATION V5.1</div>
  <div style="color:#94a3b8;font-size:0.75rem;font-weight:800;margin-top:0.55rem">CURRENT PRIORITY ASSET</div>
  <div style="font-size:clamp(1.7rem,4vw,3.1rem);font-weight:950;margin:0.35rem 0">{html.escape(ranking)}</div>
  <div style="color:#cbd5e1">Layer 1: <b>{html.escape(str(spx.get('position') or 'Uninitialized'))}</b>&nbsp;&nbsp;·&nbsp;&nbsp;Layer 2: <b>{html.escape(str(btc.get('position') or 'Uninitialized'))}</b>&nbsp;&nbsp;·&nbsp;&nbsp;Priority Asset: <b>{html.escape(priority)}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Structural Confirmation")
    left, right = st.columns(2)
    spx_direction = str(spx.get("position") or "SPX")
    spx_evidence = (
        [("Higher High (24M)", bool(spx.get("hh")), ""), ("Higher Low (24M)", bool(spx.get("hl")), ""), ("Above SMA100M", _number(spx.get("close")) > _number(spx.get("sma100")), _fmt(spx.get("sma100"), 3)), ("Above SMA200M", _number(spx.get("close")) > _number(spx.get("sma200")), _fmt(spx.get("sma200"), 3))]
        if spx_direction == "SPX"
        else [("Lower High (24M)", bool(spx.get("lh")), ""), ("Lower Low (24M)", bool(spx.get("ll")), ""), ("Below SMA100M", _number(spx.get("close")) < _number(spx.get("sma100")), _fmt(spx.get("sma100"), 3)), ("Below SMA200M", _number(spx.get("close")) < _number(spx.get("sma200")), _fmt(spx.get("sma200"), 3))]
    )
    with left:
        _card("SPX / GOLD", [
            ("Trading Position", spx_direction), ("Ratio", _fmt(spx.get("close"), 3)),
            ("SMA50M", _fmt(spx.get("sma50"), 3)),
            (f"Next Switch to {'Gold' if spx_direction == 'SPX' else 'SPX'}", _fmt(spx.get("lower") if spx_direction == "SPX" else spx.get("upper"), 3)),
            ("Structural Confirmation", _status_label(spx, spx_direction)),
            ("Confirmation Health", str(spx.get("confirmation_health", "Pending"))),
            ("Recovery Phase", str(spx.get("recovery_phase", "None / Transition"))),
        ], spx_evidence, spx.get("Date"))
    btc_direction = str(btc.get("position") or "BTC")
    btc_observation = pd.Timestamp(btc.get("Date"))
    previous_halving = pd.Timestamp(btc.get("previous_halving"))
    next_halving = pd.Timestamp(btc.get("next_halving"))
    days_since_halving = max(0, (btc_observation.normalize() - previous_halving.normalize()).days)
    days_until_halving = max(0, (next_halving.normalize() - btc_observation.normalize()).days)
    btc_evidence = (
        [("Above SMA50M", _number(btc.get("ratio")) > _number(btc.get("sma50m")), _fmt(btc.get("sma50m"), 2)), ("ROC13W Positive", _number(btc.get("roc13w")) > 0, _fmt(_number(btc.get("roc13w")) * 100, 1) + "%"), ("ROC26W Positive", _number(btc.get("roc26w")) > 0, _fmt(_number(btc.get("roc26w")) * 100, 1) + "%")]
        if btc_direction == "BTC"
        else [("Below SMA50M", _number(btc.get("ratio")) < _number(btc.get("sma50m")), _fmt(btc.get("sma50m"), 2)), ("ROC13W Negative", _number(btc.get("roc13w")) < 0, _fmt(_number(btc.get("roc13w")) * 100, 1) + "%"), ("ROC26W Negative", _number(btc.get("roc26w")) < 0, _fmt(_number(btc.get("roc26w")) * 100, 1) + "%")]
    )
    with right:
        _card("BTC / GOLD", [
            ("Trading Position", btc_direction), ("BTC/Gold Ratio", _fmt(btc.get("ratio"), 2)),
            ("Channel Percentile", _fmt(btc.get("channel_percentile"), 1) + "%"),
            ("Buy BTC", f"≤{DEFAULT_CONFIG.btc_buy_percentile:.1f}%"),
            ("Sell BTC", f"≥{DEFAULT_CONFIG.btc_sell_percentile:.1f}%"),
            ("Structural Confirmation", _status_label(btc, btc_direction)),
            ("Confirmation Health", str(btc.get("confirmation_health", "Pending"))),
            ("Halving Phase", str(btc.get("halving_phase", "DATA_INCOMPLETE"))),
            ("Halving Cycle Progress", _fmt(btc.get("halving_progress"), 1) + "%"),
            ("Days Since Last Halving", f"{days_since_halving:,}"),
            ("Estimated Next Halving", _date(next_halving)),
            ("Days Until Next Halving", f"{days_until_halving:,}"),
            ("Recovery Phase", str(btc.get("recovery_phase", "None / Transition"))),
        ], btc_evidence, btc.get("Date"))

    st.markdown("### Market Interpretation")
    st.markdown(_interpretation(current))
    st.markdown("### Historical Charts")
    spx_range = st.radio("SPX / Gold range", ["5Y", "10Y", "20Y", "MAX"], index=3, horizontal=True, key="portfolio_spx_gold_range")
    st.plotly_chart(_spx_gold_chart(snapshot["spx_gold"], spx_range), use_container_width=True, config=PLOTLY_CONFIG)
    btc_range = st.radio("BTC / Gold range", ["1Y", "3Y", "5Y", "MAX"], index=3, horizontal=True, key="portfolio_btc_gold_range")
    st.plotly_chart(_btc_gold_chart(snapshot["btc_gold"], btc_range), use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Trading Signal History")
    signal_filter = st.radio("Signals", ["All Signals", "SPX / Gold", "BTC / Gold"], horizontal=True, label_visibility="collapsed", key="portfolio_signal_filter")
    signals = snapshot["signals"].copy()
    if signal_filter != "All Signals":
        signals = signals.loc[signals["Layer"] == signal_filter]
    st.dataframe(signals.sort_values("Signal Date", ascending=False), use_container_width=True, hide_index=True)

    with st.expander("Methodology, reconciliation and data diagnostics", expanded=False):
        st.markdown(
            "**BTC/Gold percentile methodology:** fit log(BTC/Gold) on the 208 completed weeks strictly preceding each signal week; "
            "forecast the current week at x=208; rank the current residual against the 208 in-sample residuals using weak empirical rank "
            "`count(residual <= current) / 208 × 100`. Ties are included. Insufficient history remains uninitialized."
        )
        reconciliation = reconcile_btc_gold_signals(snapshot["btc_gold"])
        st.dataframe(reconciliation, use_container_width=True, hide_index=True)
        diagnostics = snapshot.get("diagnostics", {})
        st.json({key: _date(value) if key.endswith(("_start", "_end")) else value for key, value in diagnostics.items()})

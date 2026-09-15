from __future__ import annotations

import html
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_regime import build_gold_regime_snapshot, gold_regime_config
from global_liquidity import read_global_liquidity


GOLD_X_AXIS_DATE_FORMAT = "%b'%y"
GOLD_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=True, ttl=21600)
def load_gold_regime_snapshot(gold_alpha: float | None, fred_api_key: str | None) -> Any:
    return build_gold_regime_snapshot(gold_alpha=gold_alpha, fred_api_key=fred_api_key, config=gold_regime_config())


@st.cache_data(show_spinner=False, ttl=21600)
def load_gold_global_m2_context() -> dict[str, Any]:
    try:
        _, monthly, weekly = read_global_liquidity()
    except Exception as exc:
        return {"status": "DATA_UNAVAILABLE", "error": str(exc)}

    source = weekly if weekly is not None and not weekly.empty else monthly
    if source is None or source.empty or "global_m2_usd_bn" not in source.columns:
        return {"status": "DATA_UNAVAILABLE"}

    frame = source.copy()
    date_col = "date" if "date" in frame.columns else "observation_date" if "observation_date" in frame.columns else None
    if date_col is None:
        return {"status": "DATA_UNAVAILABLE"}
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["global_m2_usd_bn"] = pd.to_numeric(frame["global_m2_usd_bn"], errors="coerce")
    frame = frame.dropna(subset=[date_col, "global_m2_usd_bn"]).sort_values(date_col)
    if frame.empty:
        return {"status": "DATA_UNAVAILABLE"}

    weekly_m2 = frame.set_index(date_col)["global_m2_usd_bn"].resample("W-FRI").last().ffill()
    growth_13w = weekly_m2.pct_change(13, fill_method=None)
    growth_26w = weekly_m2.pct_change(26, fill_method=None)
    latest_date = weekly_m2.dropna().index.max()
    latest_13w = safe_series_last(growth_13w)
    latest_26w = safe_series_last(growth_26w)
    acceleration = latest_13w - latest_26w if np.isfinite(latest_13w) and np.isfinite(latest_26w) else np.nan

    impulse = np.nan
    if "global_m2_impulse" in frame.columns:
        impulse_values = pd.to_numeric(frame["global_m2_impulse"], errors="coerce").dropna()
        if not impulse_values.empty:
            impulse = float(impulse_values.iloc[-1])

    if np.isfinite(latest_13w) and latest_13w < 0 and np.isfinite(acceleration) and acceleration < 0:
        state = "CONTRACTING"
    elif np.isfinite(acceleration) and acceleration > 0 and np.isfinite(latest_13w) and latest_13w > 0:
        state = "ACCELERATING"
    elif np.isfinite(latest_13w) and latest_13w > 0:
        state = "STABLE"
    elif np.isfinite(latest_13w):
        state = "DECELERATING"
    else:
        state = "DATA_INCOMPLETE"

    return {
        "date": latest_date,
        "global_m2_growth_13w": latest_13w,
        "global_m2_growth_26w": latest_26w,
        "global_m2_acceleration": acceleration,
        "global_m2_impulse": impulse,
        "context_state": state,
        "status": "CURRENT",
    }


def render_gold_regime_tab(table_df: pd.DataFrame, fred_api_key: str | None = None) -> None:
    st.subheader("Gold Regime")
    gold_alpha = extract_gold_alpha(table_df)
    with st.spinner("Computing Gold Regime..."):
        snapshot = load_gold_regime_snapshot(gold_alpha, fred_api_key)
    current = snapshot.current
    if not current:
        st.warning("Gold Regime data is unavailable.")
        return

    render_summary(current)
    selected_range = st.radio(
        "Time range",
        ["1Y", "3Y", "5Y", "10Y", "MAX"],
        index=2,
        horizontal=True,
        key="gold_charts_range",
    )
    render_gold_history_chart(snapshot, selected_range)
    render_signal_explanation(current)
    render_macro_detail(current)
    render_global_monetary_liquidity_context(current)
    render_flow_detail(current, snapshot)
    render_structural_demand(current, snapshot)
    render_freshness(snapshot)
    render_history_table(snapshot.history)


def extract_gold_alpha(table_df: pd.DataFrame) -> float | None:
    if table_df is None or table_df.empty or "Ticker" not in table_df.columns or "Alpha_Score" not in table_df.columns:
        return None
    gold = table_df[table_df["Ticker"].astype(str).str.upper().isin(["GLD", "GLD.US"])]
    if gold.empty:
        return None
    value = pd.to_numeric(gold.iloc[0].get("Alpha_Score"), errors="coerce")
    return float(value) if np.isfinite(value) else None


def render_summary(current: dict[str, Any]) -> None:
    metrics = [
        ("Final Gold State", fmt_text(current.get("gold_regime")), "rule-based"),
        ("Gold Price", fmt_number(current.get("gold_price"), decimals=2), "GLD weekly close"),
        ("Gold Alpha", f"{fmt_score(current.get('gold_alpha'))} / {fmt_text(current.get('gold_alpha_state'))}", "PRICE"),
        ("Structural Macro", f"{fmt_score(current.get('structural_macro_score'))} / {fmt_text(current.get('structural_macro_state'))}", "DXY 35% + real yield 55% + US2Y 10%"),
        ("Forward Macro Risk", f"{fmt_score(current.get('forward_macro_risk'))} / {fmt_text(current.get('forward_macro_risk_state'))}", "US2Y 60% + WTI 40%"),
        ("Tactical Flow", f"{fmt_score(current.get('tactical_flow_score'))} / {fmt_text(current.get('flow_state'))}", "Gold ETF flows 85% + COT 15%"),
        ("ETF Flow Score", fmt_score(current.get("etf_flow_score")), "4W flow intensity, 3Y PIT percentile"),
        ("COT Momentum Score", fmt_score(current.get("cot_momentum_score")), "4W change in MM net % OI"),
        ("Long Liquidity Cycle", fmt_text(current.get("long_liquidity_cycle")), "context only"),
        ("Structural Demand Status", fmt_text(current.get("structural_demand_status")), "manual / quarterly context"),
        ("Active Divergence Flags", fmt_text(current.get("ACTIVE_DIVERGENCE_FLAGS")), "informational only"),
        ("Last Updated", fmt_date(current.get("date")), "latest weekly observation"),
    ]
    for start in range(0, len(metrics), 4):
        cols = st.columns(4)
        for col, (label, value, detail) in zip(cols, metrics[start : start + 4]):
            with col:
                render_metric(label, value, detail)


def render_metric(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
<div style="padding: 0.75rem 0; line-height: 1.15;">
  <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">{html.escape(label)}</div>
  <div style="font-size: 1.0rem; color: #f8fafc; font-weight: 800;">{html.escape(value)}</div>
  <div style="font-size: 0.72rem; color: #cbd5e1;">{html.escape(detail)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_gold_history_chart(snapshot: Any, selected_range: str) -> None:
    history = snapshot.history
    st.markdown("### Gold Price + Gold Regime")
    if history.empty or "gold_price" not in history.columns:
        st.info("Gold price history is unavailable.")
        return

    d = filter_gold_history_range(history, selected_range)
    if d.empty:
        st.info("Gold price history is unavailable.")
        return
    price_data = d.dropna(subset=["date", "gold_price"]).copy()
    if price_data.empty:
        st.info("Gold price history is unavailable.")
        return
    price_data["next_date"] = price_data["date"].shift(-1)
    price_data.loc[price_data["next_date"].isna(), "next_date"] = price_data.loc[price_data["next_date"].isna(), "date"] + pd.Timedelta(days=7)
    ymin = float(price_data["gold_price"].min())
    ymax = float(price_data["gold_price"].max())
    pad = max((ymax - ymin) * 0.06, 1.0)
    price_data["y_min"] = ymin - pad
    price_data["y_max"] = ymax + pad

    st.plotly_chart(build_gold_price_plotly(price_data, ymin - pad, ymax + pad), use_container_width=True, config=GOLD_PLOTLY_CONFIG)
    st.plotly_chart(
        build_gold_score_components_plotly(
            d,
            "structural_macro_score",
            [("dxy_score", "DXYBull", "#38bdf8"), ("real_yield_score", "RealYieldBull", "#facc15"), ("us2y_bull_score", "US2YBull", "#a78bfa")],
            "Gold Structural Macro",
            "#00ff66",
            higher_is_better=True,
        ),
        use_container_width=True,
        config=GOLD_PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_gold_score_components_plotly(
            d,
            "forward_macro_risk",
            [("us2y_risk_score", "US2YRisk", "#facc15"), ("wti_risk_score", "WTIRisk", "#fb923c")],
            "Gold Forward Macro Risk",
            "#ff1744",
        ),
        use_container_width=True,
        config=GOLD_PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_gold_score_components_plotly(
            d,
            "tactical_flow_score",
            [("etf_flow_score", "ETFFlowScore", "#22d3ee"), ("cot_momentum_score", "COTMomentumScore", "#facc15")],
            "Gold Tactical Flow",
            "#00e5ff",
            higher_is_better=True,
        ),
        use_container_width=True,
        config=GOLD_PLOTLY_CONFIG,
    )
    etf_fig = build_gold_etf_flows_plotly(d, positioning_chart_subtitle(snapshot, "ETF flows"))
    if etf_fig is not None:
        st.plotly_chart(etf_fig, use_container_width=True, config=GOLD_PLOTLY_CONFIG)
    cot_fig = build_gold_cot_plotly(d, positioning_chart_subtitle(snapshot, "COT"))
    if cot_fig is not None:
        st.plotly_chart(cot_fig, use_container_width=True, config=GOLD_PLOTLY_CONFIG)


def prepare_weekly_gold_chart_data(history: pd.DataFrame) -> pd.DataFrame:
    d = history.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return d
    d["date"] = d["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return d.sort_values("date").groupby("date", as_index=False).last()


def get_gold_chart_x_domain(history: pd.DataFrame) -> list[pd.Timestamp]:
    d = history.dropna(subset=["date"]).copy()
    if d.empty:
        today = pd.Timestamp.today().normalize()
        return [today - pd.Timedelta(days=365), today]
    start = pd.Timestamp(d["date"].min()).normalize()
    end = pd.Timestamp(d["date"].max()).normalize() + pd.Timedelta(days=7)
    return [start, end]


def configure_gold_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            gridColor="#263241",
            domainColor="#475569",
            tickColor="#475569",
            labelColor="#cbd5e1",
            titleColor="#cbd5e1",
        )
        .configure_view(stroke=None)
    )


def style_gold_plotly(fig: go.Figure, height: int, title: str, subtitle: str | None = None) -> go.Figure:
    full_title = title if not subtitle else f"{title}<br><sup>{html.escape(subtitle)}</sup>"
    fig.update_layout(
        title=full_title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 72, "t": 62, "b": 38},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(
        tickformat="%b'%y",
        showgrid=False,
        zeroline=False,
        color="#cbd5e1",
        linecolor="#475569",
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#263241",
        zeroline=False,
        color="#cbd5e1",
        linecolor="#475569",
        ticks="outside",
    )
    return fig


def build_gold_price_plotly(price_data: pd.DataFrame, ymin: float, ymax: float) -> go.Figure:
    regime_colors = {
        "HIGH_CONVICTION_LONG": "#00ff66",
        "BULLISH": "#8cff3d",
        "STRONG_TREND_WITH_FLOW_SUPPORT": "#22d3ee",
        "MACRO_WARNING": "#f59e0b",
        "FLOW_DIVERGENCE_WARNING": "#ff7a18",
        "MACRO_TURNING_BULLISH": "#a3e635",
        "NEUTRAL": "#94a3b8",
        "BEARISH": "#ef4444",
        "HIGH_RISK": "#be123c",
        "DATA_INCOMPLETE": "#64748b",
    }
    fig = go.Figure()
    for _, row in price_data.iterrows():
        regime = str(row.get("gold_regime") or "DATA_INCOMPLETE")
        fig.add_vrect(
            x0=row["date"],
            x1=row["next_date"],
            fillcolor=regime_colors.get(regime, "#64748b"),
            opacity=0.25,
            line_width=0,
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=price_data["date"],
            y=price_data["gold_price"],
            mode="lines",
            name="GLD",
            line={"color": "#f8fafc", "width": 1.8},
            customdata=price_data[["gold_regime"]],
            hovertemplate="Week: %{x|%Y-%m-%d}<br>GLD: %{y:.2f}<br>Gold Regime: %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_yaxes(title="GLD", range=[ymin, ymax])
    return style_gold_plotly(fig, 390, "GLD Weekly Price with Gold Regime Zones")


def build_gold_score_plotly(data: pd.DataFrame, metric: str, title: str, color: str, higher_is_better: bool = False) -> go.Figure:
    d = data.dropna(subset=["date", metric]).copy() if metric in data.columns else pd.DataFrame(columns=["date", metric])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["date"],
            y=pd.to_numeric(d[metric], errors="coerce"),
            name=title,
            marker={"color": color, "opacity": 0.9},
            hovertemplate="Week: %{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
        )
    )
    threshold_colors = (
        {20: "#ef4444", 40: "#f97316", 60: "#facc15", 80: "#22c55e"}
        if higher_is_better
        else {20: "#22c55e", 40: "#facc15", 60: "#f97316", 80: "#ef4444"}
    )
    for level, line_color in threshold_colors.items():
        fig.add_hline(y=level, line={"color": line_color, "dash": "dash", "width": 1})
    fig.update_yaxes(title="Score", range=[0, 100])
    return style_gold_plotly(fig, 190, title)


def build_gold_score_components_plotly(
    data: pd.DataFrame,
    metric: str,
    components: list[tuple[str, str, str]],
    title: str,
    color: str,
    higher_is_better: bool = False,
) -> go.Figure:
    d = data.dropna(subset=["date"]).copy()
    fig = go.Figure()
    if metric in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=pd.to_numeric(d[metric], errors="coerce"),
                mode="lines",
                name=title,
                line={"color": color, "width": 2.4},
                hovertemplate="Week: %{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
            )
        )
    for component, name, component_color in components:
        if component not in d.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=pd.to_numeric(d[component], errors="coerce"),
                mode="lines",
                name=name,
                line={"color": component_color, "width": 1.3, "dash": "dot"},
                opacity=0.72,
                hovertemplate=f"Week: %{{x|%Y-%m-%d}}<br>{name}: %{{y:.1f}}<extra></extra>",
            )
        )
    threshold_colors = (
        {20: "#ef4444", 40: "#f97316", 60: "#facc15", 80: "#22c55e"}
        if higher_is_better
        else {20: "#22c55e", 40: "#facc15", 60: "#f97316", 80: "#ef4444"}
    )
    for level, line_color in threshold_colors.items():
        fig.add_hline(y=level, line={"color": line_color, "dash": "dash", "width": 1})
    fig.update_yaxes(title="Score", range=[0, 100])
    return style_gold_plotly(fig, 250, title)


def build_gold_etf_flows_plotly(data: pd.DataFrame, subtitle: str | None = None) -> go.Figure | None:
    required = ["date", "etf_flow_intensity_4w", "etf_flow_score"]
    if data.empty or not all(column in data.columns for column in required):
        return None
    return build_dual_axis_plotly(
        data,
        left_metric="etf_flow_intensity_4w",
        right_metric="etf_flow_score",
        title="Gold ETF Fund Flows — 4W Flow Intensity and Trailing 3Y Percentile",
        subtitle=subtitle,
        left_name="4W Flow Intensity",
        right_name="3Y Percentile",
        left_color="#22d3ee",
        right_color="#facc15",
        left_axis_title="4W Flow Intensity, normalized",
        right_axis_title="Trailing 3Y Percentile",
    )


def build_gold_cot_plotly(data: pd.DataFrame, subtitle: str | None = None) -> go.Figure | None:
    required = ["date", "cot_mm_net_pct_oi", "cot_mm_net_pct_oi_percentile"]
    if data.empty or not all(column in data.columns for column in required):
        return None
    return build_dual_axis_plotly(
        data,
        left_metric="cot_mm_net_pct_oi",
        right_metric="cot_mm_net_pct_oi_percentile",
        title="Gold COT — Managed Money Net % OI and Trailing 3Y Percentile",
        subtitle=subtitle,
        left_name="Managed Money Net % OI",
        right_name="3Y Percentile",
        left_color="#60a5fa",
        right_color="#facc15",
        left_axis_title="Managed Money Net % of Open Interest",
        right_axis_title="Trailing 3Y Percentile",
        left_is_percent=True,
    )


def build_dual_axis_plotly(
    data: pd.DataFrame,
    left_metric: str,
    right_metric: str,
    title: str,
    subtitle: str | None,
    left_name: str,
    right_name: str,
    left_color: str,
    right_color: str,
    left_axis_title: str,
    right_axis_title: str,
    left_is_percent: bool = False,
) -> go.Figure:
    d = data.dropna(subset=["date"]).copy()
    d[left_metric] = pd.to_numeric(d[left_metric], errors="coerce")
    d[right_metric] = pd.to_numeric(d[right_metric], errors="coerce")
    fig = go.Figure()
    left_hover = "%{y:.1%}" if left_is_percent else "%{y:.2f}"
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d[left_metric],
            mode="lines",
            name=left_name,
            line={"color": left_color, "width": 1.8},
            hovertemplate=f"Week: %{{x|%Y-%m-%d}}<br>{left_name}: {left_hover}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d[right_metric],
            mode="lines",
            name=right_name,
            yaxis="y2",
            line={"color": right_color, "width": 1.8, "dash": "dash"},
            hovertemplate=f"Week: %{{x|%Y-%m-%d}}<br>{right_name}: %{{y:.0f}}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(
        yaxis={"title": left_axis_title, "tickformat": ".1%" if left_is_percent else None},
        yaxis2={
            "title": right_axis_title,
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
            "color": "#cbd5e1",
            "linecolor": "#475569",
        },
    )
    return style_gold_plotly(fig, 320, title, subtitle)


def build_gold_component_chart(
    data: pd.DataFrame,
    metric: str,
    title: str,
    color: str,
    higher_is_better: bool = False,
    x_domain: list[pd.Timestamp] | None = None,
) -> alt.Chart:
    if metric not in data.columns:
        return alt.Chart(pd.DataFrame({"date": [], metric: []})).mark_bar().properties(height=120, title=title)
    chart_data = data.dropna(subset=["date", metric]).copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"], errors="coerce").dt.to_period("W-FRI").dt.end_time.dt.normalize()
    chart_data = chart_data.dropna(subset=["date"]).sort_values("date").groupby("date", as_index=False).last()
    chart_data["next_date"] = chart_data["date"].shift(-1)
    chart_data.loc[chart_data["next_date"].isna(), "next_date"] = chart_data.loc[chart_data["next_date"].isna(), "date"] + pd.Timedelta(days=7)
    x_encoding = (
        alt.X("date:T", scale=alt.Scale(domain=x_domain), axis=alt.Axis(title=None, format=GOLD_X_AXIS_DATE_FORMAT))
        if x_domain is not None
        else alt.X("date:T", axis=alt.Axis(title=None, format=GOLD_X_AXIS_DATE_FORMAT))
    )
    bar = alt.Chart(chart_data).mark_bar(color=color, opacity=0.82).encode(
        x=x_encoding,
        x2="next_date:T",
        y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="Score")),
        tooltip=[
            alt.Tooltip("date:T", title="Week", format="%Y-%m-%d"),
            alt.Tooltip(f"{metric}:Q", title=title, format=".1f"),
            alt.Tooltip("gold_regime:N", title="Gold Regime"),
        ],
    )
    threshold_colors = (
        {20: "#ef4444", 40: "#f97316", 60: "#facc15", 80: "#22c55e"}
        if higher_is_better
        else {20: "#22c55e", 40: "#facc15", 60: "#f97316", 80: "#ef4444"}
    )
    threshold_20 = alt.Chart(chart_data).mark_rule(color=threshold_colors[20], strokeDash=[4, 3], opacity=0.7).encode(y=alt.datum(20))
    threshold_40 = alt.Chart(chart_data).mark_rule(color=threshold_colors[40], strokeDash=[4, 3], opacity=0.7).encode(y=alt.datum(40))
    threshold_60 = alt.Chart(chart_data).mark_rule(color=threshold_colors[60], strokeDash=[4, 3], opacity=0.7).encode(y=alt.datum(60))
    threshold_80 = alt.Chart(chart_data).mark_rule(color=threshold_colors[80], strokeDash=[4, 3], opacity=0.7).encode(y=alt.datum(80))
    return (bar + threshold_20 + threshold_40 + threshold_60 + threshold_80).properties(height=125, title=title)


def render_signal_explanation(current: dict[str, Any]) -> None:
    st.markdown("### Gold Regime Interpretation")
    explanation = fmt_text(current.get("explanation"))
    demand_context = fmt_text(current.get("additional_structural_demand_context"))
    if demand_context != "n/a":
        explanation = f"{explanation}\n\nStructural Demand: {demand_context}."
    st.markdown(
        f"<div style='white-space: pre-wrap; color:#cbd5e1; line-height:1.45;'>{html.escape(explanation)}</div>",
        unsafe_allow_html=True,
    )


def render_macro_detail(current: dict[str, Any]) -> None:
    st.markdown("### Macro Detail")
    render_table(
        [
            ("DXY 13W Return", fmt_percent(current.get("dxy_13w"))),
            ("DXY Percentile", fmt_score(current.get("dxy_percentile"))),
            ("DXY Bull Score", fmt_score(current.get("dxy_score"))),
            ("10Y Real Yield", fmt_number(current.get("real_yield"))),
            ("10Y Real Yield 13W Change bp", fmt_number(current.get("real_yield_change_13w"))),
            ("Real Yield Percentile", fmt_score(current.get("real_yield_percentile"))),
            ("Real Yield Bull Score", fmt_score(current.get("real_yield_score"))),
            ("US2Y", fmt_number(current.get("us2y"))),
            ("US2Y 13W Change bp", fmt_number(current.get("us2y_change_13w"))),
            ("US2Y Percentile", fmt_score(current.get("us2y_percentile"))),
            ("US2Y Bull Score", fmt_score(current.get("us2y_bull_score"))),
            ("US2Y Risk Score", fmt_score(current.get("us2y_risk_score"))),
            ("WTI 26W Return", fmt_percent(current.get("wti_26w"))),
            ("WTI Percentile", fmt_score(current.get("wti_percentile"))),
            ("WTI Risk Score", fmt_score(current.get("wti_risk_score"))),
            ("Structural Macro Score", fmt_score(current.get("structural_macro_score"))),
            ("Structural Macro Formula", "0.35 * DXYBull + 0.55 * RealYieldBull + 0.10 * US2YBull"),
            ("Forward Macro Risk", fmt_score(current.get("forward_macro_risk"))),
            ("Forward Macro Risk Formula", "0.60 * US2YRisk + 0.40 * WTIRisk"),
        ]
    )


def render_global_monetary_liquidity_context(current: dict[str, Any]) -> None:
    st.markdown("### Global Monetary Liquidity Context")
    context = load_gold_global_m2_context()
    if context.get("status") == "DATA_UNAVAILABLE":
        st.info("Global monetary liquidity context is unavailable.")
        return
    render_table(
        [
            ("Global M2 13W Growth", fmt_percent(context.get("global_m2_growth_13w"))),
            ("Global M2 26W Growth", fmt_percent(context.get("global_m2_growth_26w"))),
            ("Global M2 Acceleration", fmt_percentage_points(context.get("global_m2_acceleration"))),
            ("Global M2 Impulse", fmt_score(context.get("global_m2_impulse"))),
            ("Long Liquidity Cycle Phase", fmt_text(current.get("long_liquidity_cycle"))),
            ("Context State", fmt_text(context.get("context_state"))),
            ("Mode", "INFORMATIONAL / DOES NOT CHANGE GOLD FINAL REGIME"),
            ("Last Observation", fmt_date(context.get("date"))),
        ]
    )


def render_flow_detail(current: dict[str, Any], snapshot: Any) -> None:
    st.markdown("### Positioning & Flows")
    st.markdown("#### ETF Flows")
    render_table(
        [
            ("Gold ETF Net Flow 1W", fmt_money(current.get("etf_flow_1w"))),
            ("Gold ETF Net Flow 4W", fmt_money(current.get("etf_flow_4w"))),
            ("Gold ETF Net Flow 13W", fmt_money(current.get("etf_flow_13w"))),
            ("ETF Flow Intensity 4W", fmt_signed_number(current.get("etf_flow_intensity_4w"))),
            ("ETF Flow 3Y Percentile", fmt_score(current.get("etf_flow_score"))),
            ("ETF Flow Score", fmt_score(current.get("etf_flow_score"))),
            ("ETF Flow State", fmt_text(current.get("etf_flow_state"))),
            ("ETF Coverage", f"{int(current.get('etf_coverage_count', 0) or 0)} / {int(current.get('etf_coverage_total', 0) or 0)}"),
            ("Available ETF", ", ".join(snapshot.etf_available_tickers) or "n/a"),
            ("Unavailable ETF", ", ".join(snapshot.etf_unavailable_tickers) or "n/a"),
        ]
    )

    st.markdown("#### COT Positioning")
    render_table(
        [
            ("Managed Money Long", fmt_number(current.get("cot_mm_long"))),
            ("Managed Money Short", fmt_number(current.get("cot_mm_short"))),
            ("Managed Money Net", fmt_number(current.get("cot_mm_net"))),
            ("Open Interest", fmt_number(current.get("cot_open_interest"))),
            ("Managed Money Net % OI", fmt_percent(current.get("cot_mm_net_pct_oi"))),
            ("Managed Money Net % OI Percentile", fmt_score(current.get("cot_mm_net_pct_oi_percentile"))),
            ("COT Net Position State", fmt_text(current.get("cot_net_position_state"))),
            ("4W Change in MM Net % OI", fmt_percent(current.get("cot_change_4w"))),
            ("COT Momentum Score", fmt_score(current.get("cot_momentum_score"))),
            ("COT Report Date", fmt_date(current.get("cot_report_date"))),
            ("COT Contract", snapshot.cot_contract_market_name or "n/a"),
            ("Gold Tactical Flow Score", fmt_score(current.get("tactical_flow_score"))),
            ("Gold Tactical Flow Formula", "0.85 * ETFFlowScore + 0.15 * COTMomentumScore"),
            ("Flow State", fmt_text(current.get("flow_state"))),
            ("FLOW_CONFIRMATION_BULLISH", fmt_bool(current.get("FLOW_CONFIRMATION_BULLISH"))),
            ("FLOW_DIVERGENCE_WARNING", fmt_bool(current.get("FLOW_DIVERGENCE_WARNING"))),
            ("FLOW_RECOVERY", fmt_bool(current.get("FLOW_RECOVERY"))),
            ("FLOW_DETERIORATION", fmt_bool(current.get("FLOW_DETERIORATION"))),
            ("COT_ETF_DIVERGENCE", fmt_text(current.get("COT_ETF_DIVERGENCE"))),
            ("ETF_LED_BULLISH_DIVERGENCE", fmt_bool(current.get("ETF_LED_BULLISH_DIVERGENCE"))),
            ("SPECULATIVE_COT_DIVERGENCE", fmt_bool(current.get("SPECULATIVE_COT_DIVERGENCE"))),
            ("MACRO_FLOW_CONFLICT", fmt_bool(current.get("MACRO_FLOW_CONFLICT"))),
            ("LONG_CYCLE_HEADWIND", fmt_bool(current.get("LONG_CYCLE_HEADWIND"))),
            ("Active Divergence Flags", fmt_text(current.get("ACTIVE_DIVERGENCE_FLAGS"))),
        ]
    )


def filter_gold_history_range(history: pd.DataFrame, selected_range: str) -> pd.DataFrame:
    d = prepare_weekly_gold_chart_data(history)
    if d.empty or selected_range == "MAX":
        return d
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(selected_range)
    if not years:
        return d
    end_date = d["date"].max()
    start_date = end_date - pd.DateOffset(years=years)
    return d.loc[d["date"] >= start_date].copy()


def positioning_chart_subtitle(snapshot: Any, freshness_key: str) -> str:
    freshness = snapshot.freshness.get(freshness_key)
    last_updated = fmt_date(freshness.last_updated) if freshness else "n/a"
    status = freshness.status if freshness else "n/a"
    return f"Last Updated: {last_updated} · {status}"


def build_gold_etf_flows_chart(
    history: pd.DataFrame,
    x_domain: list[pd.Timestamp] | None = None,
    title: str = "Gold ETF Fund Flows — 4W Flow Intensity and Trailing 3Y Percentile",
    subtitle: str | None = None,
) -> alt.Chart | None:
    required = ["date", "etf_flow_intensity_4w", "etf_flow_score"]
    if history.empty or not all(column in history.columns for column in required):
        return None
    d = history.copy()
    d["tooltip_date"] = d["date"].map(fmt_date)
    d["etf_flow_1w_label"] = d.get("etf_flow_1w", pd.Series(index=d.index, dtype="float64")).map(fmt_money)
    d["etf_flow_4w_label"] = d.get("etf_flow_4w", pd.Series(index=d.index, dtype="float64")).map(fmt_money)
    d["etf_flow_13w_label"] = d.get("etf_flow_13w", pd.Series(index=d.index, dtype="float64")).map(fmt_money)
    d["etf_flow_intensity_4w_label"] = d["etf_flow_intensity_4w"].map(fmt_signed_number)
    d["etf_flow_score_label"] = d["etf_flow_score"].map(fmt_score)
    d["etf_coverage_label"] = d.apply(
        lambda row: f"{int(row.get('etf_coverage_count', 0) or 0)} / {int(row.get('etf_coverage_total', 0) or 0)}",
        axis=1,
    )
    tooltip = [
        alt.Tooltip("tooltip_date:N", title="Date"),
        alt.Tooltip("etf_flow_1w_label:N", title="ETF Net Flow 1W"),
        alt.Tooltip("etf_flow_4w_label:N", title="ETF Net Flow 4W"),
        alt.Tooltip("etf_flow_13w_label:N", title="ETF Net Flow 13W"),
        alt.Tooltip("etf_flow_intensity_4w_label:N", title="4W Flow Intensity"),
        alt.Tooltip("etf_flow_score_label:N", title="3Y Percentile"),
        alt.Tooltip("etf_coverage_label:N", title="ETF Coverage"),
        alt.Tooltip("etf_flow_state:N", title="State"),
    ]
    return build_dual_axis_positioning_chart(
        d,
        left_metric="etf_flow_intensity_4w",
        right_metric="etf_flow_score",
        left_title="4W Flow Intensity, normalized",
        right_title="Trailing 3Y Percentile",
        left_series="4W Flow Intensity",
        right_series="3Y Percentile",
        left_color="#22d3ee",
        right_color="#facc15",
        left_format=".2f",
        tooltip=tooltip,
        latest_left_label="Flow Intensity",
        latest_right_label="Pct",
        x_domain=x_domain,
        title=title,
        subtitle=subtitle,
    )


def build_gold_cot_positioning_chart(
    history: pd.DataFrame,
    x_domain: list[pd.Timestamp] | None = None,
    title: str = "Gold COT — Managed Money Net % OI and Trailing 3Y Percentile",
    subtitle: str | None = None,
) -> alt.Chart | None:
    required = ["date", "cot_mm_net_pct_oi", "cot_mm_net_pct_oi_percentile"]
    if history.empty or not all(column in history.columns for column in required):
        return None
    d = history.copy()
    d["tooltip_report_date"] = d.get("cot_report_date", pd.Series(index=d.index, dtype="object")).map(fmt_date)
    d["cot_mm_long_label"] = d.get("cot_mm_long", pd.Series(index=d.index, dtype="float64")).map(lambda value: fmt_number(value, decimals=0))
    d["cot_mm_short_label"] = d.get("cot_mm_short", pd.Series(index=d.index, dtype="float64")).map(lambda value: fmt_number(value, decimals=0))
    d["cot_mm_net_label"] = d.get("cot_mm_net", pd.Series(index=d.index, dtype="float64")).map(lambda value: fmt_number(value, decimals=0))
    d["cot_open_interest_label"] = d.get("cot_open_interest", pd.Series(index=d.index, dtype="float64")).map(lambda value: fmt_number(value, decimals=0))
    d["cot_mm_net_pct_oi_label"] = d["cot_mm_net_pct_oi"].map(fmt_percent)
    d["cot_mm_net_pct_oi_percentile_label"] = d["cot_mm_net_pct_oi_percentile"].map(fmt_score)
    d["cot_change_4w_label"] = d.get("cot_change_4w", pd.Series(index=d.index, dtype="float64")).map(fmt_percentage_points)
    d["cot_momentum_score_label"] = d.get("cot_momentum_score", pd.Series(index=d.index, dtype="float64")).map(fmt_score)
    tooltip = [
        alt.Tooltip("tooltip_report_date:N", title="Report Date"),
        alt.Tooltip("cot_mm_long_label:N", title="Managed Money Long"),
        alt.Tooltip("cot_mm_short_label:N", title="Managed Money Short"),
        alt.Tooltip("cot_mm_net_label:N", title="Managed Money Net"),
        alt.Tooltip("cot_open_interest_label:N", title="Open Interest"),
        alt.Tooltip("cot_mm_net_pct_oi_label:N", title="Managed Money Net % OI"),
        alt.Tooltip("cot_mm_net_pct_oi_percentile_label:N", title="3Y Percentile"),
        alt.Tooltip("cot_change_4w_label:N", title="4W Change"),
        alt.Tooltip("cot_momentum_score_label:N", title="COT Momentum Score"),
        alt.Tooltip("cot_net_position_state:N", title="Positioning State"),
    ]
    return build_dual_axis_positioning_chart(
        d,
        left_metric="cot_mm_net_pct_oi",
        right_metric="cot_mm_net_pct_oi_percentile",
        left_title="Managed Money Net % of Open Interest",
        right_title="Trailing 3Y Percentile",
        left_series="Managed Money Net % OI",
        right_series="3Y Percentile",
        left_color="#60a5fa",
        right_color="#facc15",
        left_format=".1%",
        tooltip=tooltip,
        latest_left_label="Net % OI",
        latest_right_label="Pct",
        x_domain=x_domain,
        title=title,
        subtitle=subtitle,
    )


def build_dual_axis_positioning_chart(
    data: pd.DataFrame,
    left_metric: str,
    right_metric: str,
    left_title: str,
    right_title: str,
    left_series: str,
    right_series: str,
    left_color: str,
    right_color: str,
    left_format: str,
    tooltip: list[Any],
    latest_left_label: str,
    latest_right_label: str,
    x_domain: list[pd.Timestamp] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> alt.Chart:
    d = data.copy()
    d[left_metric] = pd.to_numeric(d[left_metric], errors="coerce")
    d[right_metric] = pd.to_numeric(d[right_metric], errors="coerce")
    left_data = add_weekly_gap_segments(d.dropna(subset=["date", left_metric]))
    right_data = add_weekly_gap_segments(d.dropna(subset=["date", right_metric]))
    if left_data.empty and right_data.empty:
        return alt.Chart(pd.DataFrame({"date": []})).mark_line().properties(height=230)

    x_axis = (
        alt.X("date:T", scale=alt.Scale(domain=x_domain), axis=alt.Axis(title=None, format=GOLD_X_AXIS_DATE_FORMAT, labelOverlap=True))
        if x_domain is not None
        else alt.X("date:T", axis=alt.Axis(title=None, format=GOLD_X_AXIS_DATE_FORMAT, labelOverlap=True))
    )
    left_line = (
        alt.Chart(left_data)
        .mark_line(strokeWidth=1.7)
        .encode(
            x=x_axis,
            y=alt.Y(f"{left_metric}:Q", axis=alt.Axis(title=left_title, grid=True, format=left_format)),
            color=alt.Color("series:N", scale=alt.Scale(domain=[left_series, right_series], range=[left_color, right_color]), legend=alt.Legend(title=None, orient="bottom")),
            detail="segment:N",
            tooltip=tooltip,
        )
        .transform_calculate(series=f"'{left_series}'")
    )
    zero_line = alt.Chart(left_data).mark_rule(color="#94a3b8", strokeDash=[2, 3], opacity=0.55).encode(y=alt.datum(0))
    latest_left = build_latest_value_label(left_data, left_metric, latest_left_label, left_format, left_color)
    left_layer = alt.layer(left_line, zero_line, latest_left)

    right_line = (
        alt.Chart(right_data)
        .mark_line(strokeWidth=1.7, strokeDash=[6, 4])
        .encode(
            x=x_axis,
            y=alt.Y(
                f"{right_metric}:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title=right_title, orient="right", grid=False),
            ),
            color=alt.Color("series:N", scale=alt.Scale(domain=[left_series, right_series], range=[left_color, right_color]), legend=alt.Legend(title=None, orient="bottom")),
            detail="segment:N",
            tooltip=tooltip,
        )
        .transform_calculate(series=f"'{right_series}'")
    )
    pct_refs = [
        alt.Chart(right_data).mark_rule(color="#64748b", strokeDash=[2, 4], opacity=0.45).encode(y=alt.datum(level))
        for level in [10, 50, 90]
    ]
    latest_right = build_latest_value_label(right_data, right_metric, latest_right_label, ".0f", right_color)
    right_layer = alt.layer(right_line, *pct_refs, latest_right)
    chart_title = alt.TitleParams(text=title, subtitle=subtitle) if title and subtitle else title or alt.Undefined

    return (
        alt.layer(left_layer, right_layer)
        .resolve_scale(y="independent")
        .properties(
            height=260,
            title=chart_title,
        )
    )


def add_weekly_gap_segments(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    gaps = d["date"].diff().dt.days.fillna(7)
    d["segment"] = (gaps > 14).cumsum()
    return d


def build_latest_value_label(data: pd.DataFrame, metric: str, label: str, value_format: str, color: str) -> alt.Chart:
    if data.empty:
        return alt.Chart(pd.DataFrame({"date": [], metric: []})).mark_text()
    latest = data.dropna(subset=[metric]).tail(1).copy()
    if latest.empty:
        return alt.Chart(pd.DataFrame({"date": [], metric: []})).mark_text()
    latest["label"] = latest[metric].map(lambda value: f"{label}: {format_latest_value(value, value_format)}")
    return alt.Chart(latest).mark_text(align="left", dx=6, dy=-6, color=color, fontSize=11, fontWeight="bold").encode(
        x="date:T",
        y=f"{metric}:Q",
        text="label:N",
    )


def render_structural_demand(current: dict[str, Any], snapshot: Any) -> None:
    st.markdown("### Structural Demand")
    freshness = snapshot.freshness.get("Structural Demand")
    status = freshness.status if freshness else "ERROR"
    render_table(
        [
            ("Structural Demand Score", fmt_score(current.get("structural_demand_score"))),
            ("Monetary Demand Share", fmt_text(current.get("monetary_demand_share"))),
            ("Central Bank 4Q Purchases", fmt_text(current.get("central_bank_4q_purchases"))),
            ("Demand Rotation YoY", fmt_text(current.get("demand_rotation_yoy"))),
            ("Last Updated", fmt_text(current.get("structural_demand_last_updated"))),
            ("Mode", "MANUAL / INFORMATIONAL"),
            ("Status", status),
        ]
    )


def render_freshness(snapshot: Any) -> None:
    st.markdown("### Data Freshness")
    rows = []
    current = snapshot.current or {}
    for block, freshness in snapshot.freshness.items():
        coverage = ""
        if block == "ETF flows":
            coverage = f"{int(current.get('etf_coverage_count', 0) or 0)} / {int(current.get('etf_coverage_total', 0) or 0)}"
        rows.append(
            {
                "Block": block,
                "Last Updated": fmt_date(freshness.last_updated),
                "Data Age": "n/a" if freshness.data_age_days is None else f"{freshness.data_age_days} days",
                "Coverage": coverage,
                "Status": freshness.status,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_history_table(history: pd.DataFrame) -> None:
    st.markdown("### Weekly History")
    if history.empty:
        st.info("Gold Regime history is unavailable.")
        return
    columns = [
        "date",
        "gold_price",
        "gold_alpha",
        "structural_macro_score",
        "forward_macro_risk",
        "etf_flow_1w",
        "etf_flow_4w",
        "etf_flow_13w",
        "etf_flow_score",
        "cot_mm_net_pct_oi",
        "cot_mm_net_pct_oi_percentile",
        "cot_change_4w",
        "cot_momentum_score",
        "tactical_flow_score",
        "long_liquidity_cycle",
        "ACTIVE_DIVERGENCE_FLAGS",
        "gold_regime",
    ]
    view = history[[column for column in columns if column in history.columns]].tail(260).copy()
    st.dataframe(view, hide_index=True, use_container_width=True)


def render_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)


def safe_series_last(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def fmt_number(value: Any, decimals: int = 1) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:,.{decimals}f}"


def fmt_score(value: Any) -> str:
    return fmt_number(value, decimals=0)


def fmt_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100.0:.1f}%"


def fmt_percentage_points(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100.0:+.1f} pp"


def fmt_signed_number(value: Any, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:+,.{decimals}f}"


def format_latest_value(value: Any, value_format: str) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    if value_format.endswith("%"):
        return f"{numeric * 100.0:.1f}%"
    if value_format == ".0f":
        return f"{numeric:.0f}"
    return f"{numeric:.2f}"


def fmt_money(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"${numeric / 1_000_000.0:,.1f}M"


def fmt_text(value: Any) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return text if text and text.lower() != "nan" else "n/a"


def fmt_bool(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).date().isoformat()

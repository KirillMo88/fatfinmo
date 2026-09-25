from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from liquidity_forecast import VALIDATION_PATH, read_forecast_snapshot


FORECAST_PHASES = {"BOTTOMING", "EXPANSION", "PEAKING", "CONTRACTION"}
FORWARD_SIGNAL_COLORS = {
    "NEGATIVE": "#ef4444",
    "NEUTRAL": "#64748b",
    "POSITIVE": "#22c55e",
    "DATA_INCOMPLETE": "#475569",
}
FORECAST_STATE_COLORS = {
    "BOTTOMING": "#facc15",
    "EXPANSION": "#86efac",
    "PEAKING": "#15803d",
    "CONTRACTION": "#ef4444",
    "NEUTRAL": "#64748b",
    "DATA_INCOMPLETE": "#475569",
}
PRESSURE_ZONES = ((0, 40, "#22c55e"), (40, 60, "#facc15"), (60, 100, "#ef4444"))
RESPONSE_ZONES = ((0, 40, "#ef4444"), (40, 60, "#facc15"), (60, 100, "#22c55e"))
PRESSURE_COMPONENTS = (
    ("DXY change 13W", "DXY_13W_Change_Pctl", "DXY_13W_Change", "%", ""),
    ("MOVE", "MOVE_Pctl", "MOVE", "", ""),
    ("US 2Y yield", "US2Y_Pctl", "US2Y", "%", ""),
    ("ISM Manufacturing", "Inverse_ISM_Pctl", "ISM_Manufacturing", "", "Inverted"),
    ("CFNAI", "Inverse_CFNAI_Pctl", "CFNAI", "", "Inverted"),
    ("Continuing Claims change 13W", "ContinuingClaims_13W_Pctl", "ContinuingClaims_13W_Change", "", ""),
    ("10Y Term Premium change 26W", "TermPremium_26W_Change_Pctl", "US10Y_TermPremium_26W_Change", " pp", ""),
)
RESPONSE_COMPONENTS = (
    ("CB Impulse", "CBImpulse", None, "", ""),
    ("US Net Liquidity Impulse", "USNLImpulse", None, "", ""),
    ("Bank Reserves Impulse", "BankReservesImpulse", "US_BankReserves_13W_PctChange", "%", ""),
)


def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=35, r=15, t=20, b=40),
        paper_bgcolor="#14181e", plot_bgcolor="#14181e", legend=dict(orientation="h", y=-0.14),
        font=dict(size=11), hovermode="closest",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#334155", zeroline=False)
    return fig


def filter_forecast_history(frame: pd.DataFrame, range_start: pd.Timestamp | None) -> pd.DataFrame:
    view = frame.copy()
    view["Date"] = pd.to_datetime(view["Date"], errors="coerce")
    view = view.dropna(subset=["Date"]).sort_values("Date")
    if range_start is not None and pd.notna(range_start):
        view = view.loc[view["Date"] >= pd.Timestamp(range_start)]
    return view


def _discrete_colorscale(colors: list[str]) -> list[tuple[float, str]]:
    count = len(colors)
    return [(edge, color) for index, color in enumerate(colors) for edge in (index / count, (index + 1) / count)]


def _status_strip(fig: go.Figure, view: pd.DataFrame, column: str, colors: dict[str, str], row: int, label: str) -> None:
    states = view[column].fillna("DATA_INCOMPLETE").astype(str).tolist()
    keys = list(colors)
    values = [keys.index(state) if state in colors else keys.index("DATA_INCOMPLETE") for state in states]
    fig.add_trace(
        go.Heatmap(
            x=view["Date"], y=[label], z=[values], text=[states],
            zmin=-0.5, zmax=len(keys) - 0.5, colorscale=_discrete_colorscale(list(colors.values())),
            showscale=False, xgap=0, ygap=0,
            hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.update_yaxes(showgrid=False, showticklabels=True, tickfont=dict(size=10), row=row, col=1)


def build_forecast_timeline(view: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.52, 0.25, 0.11, 0.12], vertical_spacing=0.035)
    pressure = pd.to_numeric(view["LiquidityPressureScore"], errors="coerce")
    response = pd.to_numeric(view["PolicyResponseScore"], errors="coerce")
    difference = pressure - response
    details = view[["LiquidityForwardSignal", "LiquidityForecastState"]].fillna("DATA_INCOMPLETE").to_numpy()
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=pressure, name="Liquidity Pressure", mode="lines",
            line=dict(color="#fb7185", width=2), customdata=details,
            hovertemplate="%{x|%Y-%m-%d}<br>Pressure %{y:.1f}<br>Forward signal %{customdata[0]}<br>Forecast state %{customdata[1]}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=response, name="Policy Response", mode="lines",
            line=dict(color="#34d399", width=2), customdata=details,
            hovertemplate="%{x|%Y-%m-%d}<br>Response %{y:.1f}<br>Forward signal %{customdata[0]}<br>Forecast state %{customdata[1]}<extra></extra>",
        ),
        row=1, col=1,
    )
    for threshold in (40, 60):
        fig.add_hline(y=threshold, line_dash="dot", line_color="#64748b", row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text="Score", row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=view["Date"], y=difference, name="Pressure - Response",
            marker_color=["#22c55e" if value > 0 else "#ef4444" if value < 0 else "#64748b" for value in difference],
            hovertemplate="%{x|%Y-%m-%d}<br>Pressure - Response %{y:+.1f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="P - R", row=2, col=1)
    _status_strip(fig, view, "LiquidityForwardSignal", FORWARD_SIGNAL_COLORS, 3, "Signal")
    _status_strip(fig, view, "LiquidityForecastState", FORECAST_STATE_COLORS, 4, "Forecast")
    for row in (1, 2, 3):
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    start = (view["Date"].min() - timedelta(days=3)).to_pydatetime()
    end = (view["Date"].max() + timedelta(days=3)).to_pydatetime()
    fig.update_xaxes(range=[start, end])
    return _style(fig, 540)


def build_score_history(
    view: pd.DataFrame, score_col: str, band_col: str,
    zones: tuple[tuple[int, int, str], ...], color: str,
) -> go.Figure:
    fig = go.Figure()
    for low, high, zone_color in zones:
        fig.add_hrect(y0=low, y1=high, fillcolor=zone_color, opacity=0.13, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=pd.to_numeric(view[score_col], errors="coerce"),
            mode="lines", name=score_col, line=dict(color=color, width=2),
            customdata=view[band_col].fillna("DATA_INCOMPLETE"),
            hovertemplate="%{x|%Y-%m-%d}<br>Score %{y:.1f}<br>Band %{customdata}<extra></extra>",
        )
    )
    for threshold in (40, 60):
        fig.add_hline(y=threshold, line_dash="dot", line_color="#94a3b8")
    fig.update_yaxes(range=[0, 100], title="Score")
    fig.update_layout(showlegend=False)
    return _style(fig, 260)


def build_component_history(
    view: pd.DataFrame, label: str, value_col: str,
    raw_col: str | None, raw_unit: str, orientation: str,
) -> go.Figure:
    fig = go.Figure()
    hover = f"%{{x|%Y-%m-%d}}<br>{label}<br>Model value %{{y:.1f}}"
    kwargs = {}
    if raw_col is not None and raw_col in view:
        raw = pd.to_numeric(view[raw_col], errors="coerce")
        kwargs["text"] = raw.map(lambda value: "n/a" if pd.isna(value) else f"{value:.2f}{raw_unit}")
        hover += "<br>Raw %{text}"
    if orientation:
        hover += f"<br>{orientation}"
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=pd.to_numeric(view[value_col], errors="coerce"),
            mode="lines", name=label, line=dict(color="#fb7185" if value_col in {item[1] for item in PRESSURE_COMPONENTS} else "#34d399", width=1.7),
            hovertemplate=hover + "<extra></extra>", **kwargs,
        )
    )
    for threshold in (40, 60):
        fig.add_hline(y=threshold, line_dash="dot", line_color="#64748b")
    fig.update_yaxes(range=[0, 100], tickvals=[0, 40, 60, 100])
    fig.update_layout(showlegend=False)
    return _style(fig, 175)


def build_pressure_response_figure(frame: pd.DataFrame, trajectory: str) -> go.Figure:
    weeks = {"13W": 13, "26W": 26, "52W": 52}[trajectory]
    points = frame.tail(weeks + 1).dropna(subset=["LiquidityPressureScore", "PolicyResponseScore"])
    fig = go.Figure()
    fig.add_vline(x=40, line_dash="dot", line_color="#64748b")
    fig.add_vline(x=60, line_dash="dot", line_color="#64748b")
    fig.add_hline(y=40, line_dash="dot", line_color="#64748b")
    fig.add_hline(y=60, line_dash="dot", line_color="#64748b")
    fig.add_trace(go.Scatter(x=points["LiquidityPressureScore"], y=points["PolicyResponseScore"], mode="lines+markers", name="Trajectory", line=dict(color="#94a3b8"), marker=dict(size=6, color="#94a3b8"), text=points["Date"].dt.strftime("%Y-%m-%d"), hovertemplate="%{text}<br>Pressure %{x:.1f}<br>Response %{y:.1f}<extra></extra>"))
    if not points.empty:
        point = points.iloc[-1]
        fig.add_trace(go.Scatter(x=[point["LiquidityPressureScore"]], y=[point["PolicyResponseScore"]], mode="markers", name="Latest", marker=dict(size=16, color="#facc15", line=dict(color="#ffffff", width=1))))
    fig.add_annotation(x=77, y=20, text="Bottoming setup", showarrow=False, font=dict(color="#38bdf8", size=10))
    fig.add_annotation(x=22, y=80, text="Peaking setup", showarrow=False, font=dict(color="#fbbf24", size=10))
    fig.update_xaxes(title="Liquidity Pressure", range=[0, 100])
    fig.update_yaxes(title="Policy Response", range=[0, 100])
    return _style(fig)


def render_historical_validation() -> None:
    validation = pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else pd.DataFrame()
    liquidity = validation.loc[validation["Analysis"].eq("Liquidity State")] if not validation.empty else pd.DataFrame()
    if liquidity.empty:
        st.info("Validation snapshot is not available yet.")
        return
    fig = go.Figure()
    for horizon, color in ((13, "#38bdf8"), (26, "#facc15")):
        fig.add_trace(go.Bar(x=liquidity["Group"], y=liquidity[f"Avg_{horizon}W"], name=f"Future GLS Δ {horizon}W", marker_color=color, customdata=liquidity[[f"N_{horizon}W", f"HitRate_{horizon}W"]].to_numpy(), hovertemplate="%{x}<br>Avg ΔGLS %{y:.1f}<br>N %{customdata[0]}<br>Hit rate %{customdata[1]:.1f}%<extra></extra>"))
    fig.update_layout(barmode="group")
    st.plotly_chart(_style(fig), use_container_width=True)
    with st.expander("Breadth confirmation and threshold sensitivity", expanded=False):
        st.caption("Forward worst path and drawdown probability use weekly closes, not intraday lows. Thresholds 50 and 70 are diagnostics; production remains 60/40.")
        st.dataframe(validation.loc[validation["Analysis"].eq("Breadth Confirmation")].round(2), use_container_width=True, hide_index=True)


def render_liquidity_forecast(range_start: pd.Timestamp | None = None) -> None:
    frame, status = read_forecast_snapshot()
    st.markdown("### Liquidity Forecast")
    if status.get("LatestRefreshError"):
        error = status["LatestRefreshError"]
        st.warning(f"Latest forecast refresh failed ({error.get('Reason', 'error')}); displaying the previous successful snapshot. Failed at {error.get('FailedAt', 'n/a')}.")
    if frame.empty:
        st.info("Forecast snapshot is not available yet. The nightly liquidity-forecast job will backfill it.")
        return
    frame = filter_forecast_history(frame, None)
    if frame.empty:
        st.warning("Forecast snapshot has no valid weekly dates.")
        return
    valid = frame.loc[frame["LiquidityForecastState"].isin(FORECAST_PHASES)]
    latest = valid.iloc[-1] if not valid.empty else frame.iloc[-1]
    strip = [
        ("Current Liquidity", f"{_number(latest.get('GlobalLiquidityScore'))} / {_number(latest.get('GlobalLiquidity_Direction_13W'), signed=True)}"),
        ("Forecast", str(latest.get("LiquidityForecastState", "n/a"))),
        ("Pressure", f"{_number(latest.get('LiquidityPressureScore'))} {latest.get('LiquidityPressureBand', '')}"),
        ("Response", f"{_number(latest.get('PolicyResponseScore'))} {latest.get('PolicyResponseBand', '')}"),
        ("Breadth", str(latest.get("BreadthParticipationState", "n/a"))),
        ("Risk-On", str(latest.get("RiskOnState", "n/a"))),
        ("Risk Reduction", str(latest.get("RiskReductionWarning", "n/a"))),
        ("Funding Stress", f"{_number(latest.get('SOFR_EFFR_Spread'), signed=True)} pp / {latest.get('FundingStressContext', 'n/a')}"),
        ("Long Cycle", str(latest.get("LongLiquidityCyclePhase", "n/a"))),
    ]
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            font-size: 1.4rem !important;
            line-height: 1.15 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for offset, count in ((0, 5), (5, 4)):
        for col, (label, value) in zip(st.columns(count), strip[offset:offset + count]):
            col.metric(label, value)
    st.caption(f"Weekly data as of {status.get('DataAsOf', frame['Date'].max().date())} | {status.get('ModelVersion', 'LIQ_FORECAST_V1')}")

    view = filter_forecast_history(frame, range_start)
    if view.empty:
        st.info("No completed forecast weeks in the selected range.")
        return
    st.markdown("#### Pressure + Response + Forward Signal + Forecast State")
    st.plotly_chart(build_forecast_timeline(view), use_container_width=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Liquidity Pressure Score History")
        st.plotly_chart(build_score_history(view, "LiquidityPressureScore", "LiquidityPressureBand", PRESSURE_ZONES, "#fb7185"), use_container_width=True)
        for label, value_col, raw_col, raw_unit, orientation in PRESSURE_COMPONENTS:
            st.markdown(f"**{label}**")
            st.plotly_chart(build_component_history(view, label, value_col, raw_col, raw_unit, orientation), use_container_width=True)
    with right:
        st.markdown("#### Policy Response Score History")
        st.plotly_chart(build_score_history(view, "PolicyResponseScore", "PolicyResponseBand", RESPONSE_ZONES, "#34d399"), use_container_width=True)
        for label, value_col, raw_col, raw_unit, orientation in RESPONSE_COMPONENTS:
            st.markdown(f"**{label}**")
            st.plotly_chart(build_component_history(view, label, value_col, raw_col, raw_unit, orientation), use_container_width=True)

    validation_col, trajectory_col = st.columns(2, gap="medium")
    with validation_col:
        st.markdown("#### Historical Forecast Validation")
        render_historical_validation()
    with trajectory_col:
        st.markdown("#### Pressure vs Policy Response")
        trajectory = st.radio("Trajectory", ["13W", "26W", "52W"], horizontal=True, key="liquidity_forecast_trajectory")
        st.plotly_chart(build_pressure_response_figure(frame, trajectory), use_container_width=True)

    with st.expander("Forecast source status", expanded=False):
        st.dataframe(pd.DataFrame([{"Source": key, "Status": value} for key, value in status.get("SourceStatus", {}).items()]), use_container_width=True, hide_index=True)
        st.caption(status.get("TimingConvention", ""))


def _number(value: object, signed: bool = False) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "n/a"
    return f"{float(number):+.1f}" if signed else f"{float(number):.1f}"

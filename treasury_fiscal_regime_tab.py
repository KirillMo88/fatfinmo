from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from treasury_fiscal_regime import (
    FINANCING_MAP_TRAIL_QUARTERS, SOURCE_SPECS,
    TreasuryFiscalSnapshot, read_snapshot, refresh_snapshot,
)
from treasury_funding_policy import (
    POLICY_WEIGHTS, TENORS, TreasuryFundingPolicySnapshot,
    load_config as load_funding_config,
    read_snapshot as read_funding_policy_snapshot,
    refresh_snapshot as refresh_funding_policy_snapshot,
    save_config as save_funding_config,
)


CHART_CONFIG = {"displayModeBar": False, "responsive": True}
BG = "#14181e"
GRID = "#334155"
COLORS = {
    "fed": "#38bdf8", "tga": "#facc15", "rrp": "#34d399",
    "liquidity": "#e2e8f0", "fiscal": "#fb7185", "duration": "#f97316",
}
MIX_COLORS = {
    "FISCAL & LIQUIDITY SUPPORT": "#22c55e",
    "FISCAL SUPPORT / TREASURY PRESSURE": "#f97316",
    "LIQUIDITY SUPPORT / FISCAL PRESSURE": "#38bdf8",
    "FISCAL & TREASURY PRESSURE": "#ef4444",
    "LIQUIDITY SUPPORT / FISCAL NEUTRAL": "#86efac",
    "TREASURY PRESSURE / FISCAL NEUTRAL": "#fb7185",
    "FISCAL SUPPORT / LIQUIDITY NEUTRAL": "#a3e635",
    "FISCAL PRESSURE / LIQUIDITY NEUTRAL": "#fbbf24",
    "NEUTRAL POLICY MIX": "#94a3b8",
    "DATA UNAVAILABLE": "#475569",
}


def _style(fig: go.Figure, height: int = 345) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=47, r=18, t=24, b=55),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(size=11),
        legend=dict(orientation="h", y=-0.20), hovermode="closest",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _line(fig: go.Figure, frame: pd.DataFrame, column: str, label: str, color: str,
          row: int = 1, col: int = 1, scale: float = 1.0, suffix: str = "") -> None:
    fig.add_trace(
        go.Scatter(
            x=frame["Date"], y=pd.to_numeric(frame[column], errors="coerce") * scale,
            name=label, mode="lines", line=dict(color=color, width=2),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}{suffix}<extra></extra>",
        ), row=row, col=col,
    )


def _shade(fig: go.Figure, frame: pd.DataFrame, column: str, colors: dict[str, str], row: int = 1) -> None:
    if frame.empty:
        return
    states = frame[column].fillna("DATA UNAVAILABLE").astype(str).tolist()
    dates = pd.to_datetime(frame["Date"]).tolist()
    start = 0
    for end in range(1, len(frame) + 1):
        if end < len(frame) and states[end] == states[start]:
            continue
        right = dates[end] if end < len(frame) else dates[-1] + pd.Timedelta(days=7)
        fig.add_vrect(
            x0=dates[start], x1=right, fillcolor=colors.get(states[start], "#475569"),
            opacity=0.08, line_width=0, layer="below", row=row, col=1,
        )
        start = end


def build_net_liquidity_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.84, 0.16],
        vertical_spacing=0.045, specs=[[{"secondary_y": True}], [{}]],
    )
    _line(fig, frame, "NetLiquidity", "Net Liquidity", COLORS["liquidity"], suffix=" USD bn")
    fig.add_trace(
        go.Scatter(
            x=frame["Date"], y=pd.to_numeric(frame["FiscalStanceRaw"], errors="coerce") * 100,
            name="12M Deficit / GDP", mode="lines", line=dict(color=COLORS["fiscal"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>12M Deficit / GDP: %{y:.2f}%<extra></extra>",
        ), row=1, col=1, secondary_y=True,
    )
    _policy_mix_strip(fig, frame, row=2)
    fig.update_yaxes(title_text="Net Liquidity, USD bn", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="12M Deficit / GDP, %", row=1, col=1, secondary_y=True)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return _style(fig, 430)


def build_contributions_chart(frame: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        (f"FedImpulse_{horizon}W", "Fed", COLORS["fed"]),
        (f"TGAImpulse_{horizon}W", "TGA", COLORS["tga"]),
        (f"RRPImpulse_{horizon}W", "RRP", COLORS["rrp"]),
    ):
        fig.add_bar(x=frame["Date"], y=frame[column], name=label, marker_color=color,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:+.1f}} USD bn<extra></extra>")
    fig.add_scatter(x=frame["Date"], y=frame[{"4": "FastLiquidityImpulse", "13": "MediumLiquidityImpulse",
                                               "26": "SlowLiquidityImpulse"}[str(horizon)]],
                    mode="lines", name="Net impulse", line=dict(color="#f8fafc", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Net impulse: %{y:+.1f} USD bn<extra></extra>")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="USD bn")
    fig.add_hline(y=0, line_color="#64748b")
    return _style(fig, 340)


def build_buffers_chart(frame: pd.DataFrame, raw: bool) -> go.Figure:
    fig = go.Figure()
    series = (
        (("RRPBufferRaw", "RRP / GDP", COLORS["rrp"]), ("ReserveBufferRaw", "Reserves / GDP", COLORS["fed"]))
        if raw else
        (("RRPBufferPercentile", "RRP buffer", COLORS["rrp"]),
         ("ReserveBufferPercentile", "Reserve buffer", COLORS["fed"]))
    )
    for column, label, color in series:
        fig.add_scatter(x=frame["Date"], y=frame[column] * 100, mode="lines", name=label,
                        line=dict(color=color, width=2),
                        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.1f}}%<extra></extra>")
    fig.update_yaxes(title="% GDP" if raw else "PIT percentile", range=None if raw else [0, 100])
    return _style(fig, 340)


def build_fiscal_chart(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.54, 0.46], vertical_spacing=0.06)
    _line(fig, frame, "FiscalStanceRaw", "12M deficit / GDP", COLORS["fiscal"], scale=100, suffix="%")
    for column, label, color in (
        ("FastFiscalZ", "3M Z", "#38bdf8"), ("MediumFiscalZ", "6M Z", "#facc15"),
        ("StructuralFiscalZ", "12M Z", "#fb7185"),
    ):
        _line(fig, frame, column, label, color, row=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    fig.update_yaxes(title_text="% GDP", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return _style(fig, 370)


def build_fiscal_decomposition_chart(frame: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        (f"SpendingImpulse_{horizon}M", "Spending", COLORS["fiscal"]),
        (f"RevenueImpulse_{horizon}M", "Revenue", COLORS["fed"]),
    ):
        fig.add_bar(x=frame["Date"], y=frame[column] * 100, name=label, marker_color=color,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:+.2f}} pp GDP<extra></extra>")
    fig.add_scatter(x=frame["Date"], y=frame["FiscalStanceRaw"].diff(horizon) * 100,
                    name="Total stance change", mode="lines", line=dict(color="#f8fafc", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Total stance change: %{y:+.2f} pp GDP<extra></extra>")
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_color="#64748b")
    fig.update_yaxes(title="pp GDP")
    return _style(fig, 340)


def build_financing_chart(frame: pd.DataFrame, percent_gdp: bool) -> go.Figure:
    fig = go.Figure()
    scale = 100 if percent_gdp else 1
    unit = "pp GDP" if percent_gdp else "USD bn"
    for column, label, color in (
        ("BillSupplyLoadRaw" if percent_gdp else "BillNetIssuance4Q", "Bills", COLORS["tga"]),
        ("DurationSupplyLoadRaw" if percent_gdp else "DurationNetIssuance4Q", "Duration proxy", COLORS["duration"]),
    ):
        fig.add_bar(x=frame["Date"], y=frame[column] * scale, name=label, marker_color=color,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:+.2f}} {unit}<extra></extra>")
    total = "TotalSupplyLoadRaw" if percent_gdp else "TotalNetIssuance4Q"
    fig.add_scatter(x=frame["Date"], y=frame[total] * scale, mode="lines", name="Total issuance",
                    line=dict(color="#f8fafc", width=2),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>Total: %{{y:+.2f}} {unit}<extra></extra>")
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_color="#64748b")
    fig.update_yaxes(title=unit)
    return _style(fig, 340)


def build_financing_map(frame: pd.DataFrame) -> go.Figure:
    points = frame.dropna(subset=["FinancingObservationDate", "DurationSupplyPercentile", "AbsorptionTightness"])
    points = points.drop_duplicates("FinancingObservationDate", keep="last").tail(FINANCING_MAP_TRAIL_QUARTERS + 1)
    fig = go.Figure()
    for threshold in (75, 90):
        fig.add_vline(x=threshold, line_dash="dot", line_color="#64748b")
    if not points.empty:
        details = np.column_stack([
            points["FinancingObservationDate"].dt.to_period("Q").astype(str),
            points["TotalSupplyPercentile"].mul(100).round(1).astype(str),
            points["FinancingMix"].astype(str), points["AbsorptionCapacity"].astype(str),
            points["FundingState"].astype(str), points["TreasuryFinancingPressure"].astype(str),
        ])
        fig.add_scatter(
            x=points["DurationSupplyPercentile"] * 100, y=points["AbsorptionTightness"],
            mode="lines+markers", name="Quarterly path", line=dict(color="#94a3b8"),
            marker=dict(color="#38bdf8", size=8), customdata=details,
            hovertemplate=("Period: %{customdata[0]}<br>Duration supply: %{x:.1f}%"
                           "<br>Absorption tightness: %{y:.0f}<br>Total supply: %{customdata[1]}%"
                           "<br>Mix: %{customdata[2]}<br>Absorption: %{customdata[3]}"
                           "<br>Funding: %{customdata[4]}<br>Pressure: %{customdata[5]}<extra></extra>"),
        )
        latest = points.iloc[-1]
        fig.add_scatter(x=[latest["DurationSupplyPercentile"] * 100], y=[latest["AbsorptionTightness"]],
                        mode="markers", name="Current", marker=dict(color="#ffffff", size=14, symbol="diamond"),
                        hovertemplate="Current quarter<br>Duration supply: %{x:.1f}%<br>Absorption: %{y:.0f}<extra></extra>")
    fig.update_xaxes(title="Duration supply percentile", range=[0, 100])
    fig.update_yaxes(title="Absorption tightness", range=[0, 100], tickvals=[25, 50, 85],
                     ticktext=["Ample", "Normal", "Tight"])
    return _style(fig, 340)


def _policy_mix_strip(fig: go.Figure, frame: pd.DataFrame, row: int | None = None) -> None:
    names = list(MIX_COLORS)
    states = frame["PolicyMix"].fillna("DATA UNAVAILABLE").astype(str)
    values = [names.index(value) if value in MIX_COLORS else names.index("DATA UNAVAILABLE") for value in states]
    scale = [(edge, color) for index, color in enumerate(MIX_COLORS.values())
             for edge in (index / len(names), (index + 1) / len(names))]
    trace = go.Heatmap(
        x=frame["Date"], y=["Policy mix"], z=[values], text=[states.tolist()], zmin=-0.5,
        zmax=len(names) - 0.5, colorscale=scale, showscale=False,
        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def build_policy_mix_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _policy_mix_strip(fig, frame)
    fig.update_yaxes(showgrid=False)
    return _style(fig, 150)


def _number(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if pd.notna(value) else "n/a"


def _freshness(snapshot: TreasuryFiscalSnapshot) -> pd.DataFrame:
    rows = []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    for series_id, meta in snapshot.status.get("SourceStatus", {}).items():
        available = pd.to_datetime(meta.get("AvailableDate"), errors="coerce")
        frequency = meta.get("Frequency")
        max_age = {"daily": 7, "weekly": 14, "monthly": 45, "quarterly": 120}.get(frequency, 30)
        age = (now - available).days if pd.notna(available) else None
        rows.append({
            "Series": series_id, "Block": meta.get("Domain"), "Frequency": frequency,
            "Observation": meta.get("ObservationDate"), "Available": meta.get("AvailableDate"),
            "Days since release": age, "Status": "STALE" if age is not None and age > max_age else meta.get("State"),
        })
    return pd.DataFrame(rows)


def _interpretation(row: pd.Series) -> str:
    if row["PolicyMix"] == "DATA UNAVAILABLE":
        return "The policy mix is unavailable until all three model blocks and Funding Conditions have released data."
    fiscal = "accelerating" if row["FiscalImpulseState"].startswith("POSITIVE") else (
        "decelerating" if row["FiscalImpulseState"].startswith("NEGATIVE") else "neutral"
    )
    liquidity = row["TreasuryLiquidityState"].lower()
    driver = row["LiquidityDriver"].lower()
    pressure = row["TreasuryFinancingPressure"].lower()
    absorption = row["AbsorptionCapacity"].lower()
    return (f"Observed fiscal impulse is {fiscal}. Treasury liquidity is {liquidity} ({driver}); "
            f"financing pressure is {pressure} with {absorption} absorption capacity. "
            f"The resulting policy mix is {row['PolicyMix']}.")


def _diagnostic_table(row: pd.Series, fields: tuple[str, ...]) -> pd.DataFrame:
    values = []
    for field in fields:
        value = row.get(field, np.nan)
        values.append({"Metric": field, "Value": _number(value, 4) if isinstance(value, (float, np.floating))
                       else str(value) if pd.notna(value) else "n/a"})
    return pd.DataFrame(values)


def _scenario_chart(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    actual_column: str,
    forecast_column: str,
    y_title: str,
    show_forecast: bool = True,
) -> go.Figure:
    fig = go.Figure()
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["Year"], y=actual[actual_column], name="BASE actual",
            mode="lines+markers", line=dict(color="#e2e8f0", width=2.2),
            hovertemplate="Year %{x}<br>BASE actual %{y:.2f}<extra></extra>",
        ))
    if show_forecast and not forecast.empty:
        colors = {"LONG": "#22c55e", "BASE": "#38bdf8", "SHORT": "#f97316"}
        for scenario in ("LONG", "BASE", "SHORT"):
            data = forecast.loc[forecast["Scenario"].eq(scenario)]
            fig.add_trace(go.Scatter(
                x=data["Year"], y=data[forecast_column], name=scenario,
                mode="lines+markers", line=dict(color=colors[scenario], width=2, dash="dash"),
                hovertemplate=f"Year %{{x}}<br>{scenario} %{{y:.2f}}<extra></extra>",
            ))
    fig.update_yaxes(title=y_title)
    return _style(fig, 330)


def build_rollover_chart(actual: pd.DataFrame, forecast: pd.DataFrame, show_forecast: bool = True) -> go.Figure:
    return _scenario_chart(actual, forecast, "Rollover12M", "PrincipalRollover", "USD tn", show_forecast)


def build_rollover_yoy_chart(actual: pd.DataFrame, forecast: pd.DataFrame, show_forecast: bool = True) -> go.Figure:
    trajectory = actual[["Year", "Rollover12M"]].copy()
    if show_forecast and not forecast.empty:
        base = forecast.loc[forecast["Scenario"].eq("BASE"), ["Year", "PrincipalRollover"]].rename(
            columns={"PrincipalRollover": "Rollover12M"}
        )
        trajectory = pd.concat([trajectory, base], ignore_index=True).sort_values("Year").drop_duplicates("Year", keep="last")
    trajectory["YoYAbs"] = trajectory["Rollover12M"].diff()
    trajectory["YoYPct"] = trajectory["Rollover12M"].pct_change(fill_method=None) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08)
    fig.add_trace(go.Bar(
        x=trajectory["Year"], y=trajectory["YoYAbs"], name="YoY change",
        marker_color=np.where(trajectory["YoYAbs"].ge(0), "#f97316", "#22c55e"),
        hovertemplate="Year %{x}<br>YoY %{y:.2f} tn<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trajectory["Year"], y=trajectory["YoYPct"], name="YoY %",
        line=dict(color="#38bdf8", width=2), hovertemplate="Year %{x}<br>YoY %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title="USD tn", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    return _style(fig, 390)


def build_policy_score_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=frame["Date"], y=frame["PolicyResponseScore"], name="Policy Response Score",
        line=dict(color="#e2e8f0", width=2),
        hovertemplate="Date %{x|%Y-%m-%d}<br>Score %{y:.1f}<extra></extra>",
    ))
    fig.add_hrect(y0=0, y1=40, fillcolor="#ef4444", opacity=.12, line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor="#facc15", opacity=.10, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="#22c55e", opacity=.12, line_width=0)
    fig.add_hline(y=40, line=dict(color="#64748b", dash="dot"))
    fig.add_hline(y=60, line=dict(color="#64748b", dash="dot"))
    fig.update_yaxes(title="0-100", range=[0, 100])
    return _style(fig, 330)


def build_policy_components_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        ("CBImpulsePercentile", "Central Bank Impulse", "#34d399"),
        ("USNLImpulsePercentile", "US Net Liquidity Impulse", "#38bdf8"),
        ("BankReservesImpulsePercentile", "Bank Reserves Impulse", "#facc15"),
    ):
        fig.add_trace(go.Scatter(
            x=frame["Date"], y=frame[column], name=label, line=dict(color=color, width=1.7),
            hovertemplate=f"Date %{{x|%Y-%m-%d}}<br>{label} %{{y:.1f}}<extra></extra>",
        ))
    fig.update_yaxes(title="Percentile", range=[0, 100])
    return _style(fig, 330)


def build_raw_policy_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        ("CBImpulse13W", "CB Assets 13W", "#34d399"),
        ("USNLImpulse13W", "US Net Liquidity 13W", "#38bdf8"),
        ("BankReservesImpulse13W", "Bank Reserves 13W", "#facc15"),
    ):
        fig.add_trace(go.Scatter(
            x=frame["Date"], y=frame[column], name=label, line=dict(color=color, width=1.5),
            hovertemplate=f"Date %{{x|%Y-%m-%d}}<br>{label} %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#64748b", dash="dot"))
    fig.update_yaxes(title="13W change, %")
    return _style(fig, 320)


def build_near_term_refinancing_chart(frame: pd.DataFrame, start: pd.Timestamp | None = None) -> go.Figure:
    effective_start = max(pd.Timestamp("2020-01-01"), pd.Timestamp(start) if start is not None else pd.Timestamp("2020-01-01"))
    view = frame.loc[pd.to_datetime(frame["Date"]).ge(effective_start)].copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.055)
    custom_3m = np.column_stack([
        view["next_3m_rollover"], view["trailing_12m_rollover"], view["pressure_3m_percentile"],
    ])
    fig.add_trace(go.Scatter(
        x=view["Date"], y=view["pressure_ratio_3m"], name="3M Pressure Ratio",
        line=dict(color="#f97316", width=2), customdata=custom_3m,
        hovertemplate=("Date %{x|%Y-%m-%d}<br>3M ratio %{y:.2f}x"
                       "<br>Next 3M $%{customdata[0]:.2f}tn"
                       "<br>Trailing 12M $%{customdata[1]:.2f}tn"
                       "<br>3M PIT percentile %{customdata[2]:.1f}<extra></extra>"),
    ), row=1, col=1)
    custom_6m = np.column_stack([
        view["next_6m_rollover"], view["trailing_12m_rollover"], view["pressure_6m_percentile"],
    ])
    fig.add_trace(go.Scatter(
        x=view["Date"], y=view["pressure_ratio_6m"], name="6M Pressure Ratio",
        line=dict(color="#38bdf8", width=2), customdata=custom_6m,
        hovertemplate=("Date %{x|%Y-%m-%d}<br>6M ratio %{y:.2f}x"
                       "<br>Next 6M $%{customdata[0]:.2f}tn"
                       "<br>Trailing 12M $%{customdata[1]:.2f}tn"
                       "<br>6M PIT percentile %{customdata[2]:.1f}<extra></extra>"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=view["Date"], y=view["near_term_refinancing_pressure"], name="Composite Pressure",
        line=dict(color="#e2e8f0", width=2.2),
        hovertemplate="Date %{x|%Y-%m-%d}<br>Near-term pressure %{y:.1f}<extra></extra>",
    ), row=3, col=1)
    for row in (1, 2):
        fig.add_hline(y=1, line=dict(color="#94a3b8", dash="dot"), row=row, col=1)
    fig.add_hrect(y0=0, y1=40, fillcolor="#22c55e", opacity=.10, line_width=0, row=3, col=1)
    fig.add_hrect(y0=40, y1=60, fillcolor="#facc15", opacity=.09, line_width=0, row=3, col=1)
    fig.add_hrect(y0=60, y1=100, fillcolor="#ef4444", opacity=.11, line_width=0, row=3, col=1)
    fig.add_hline(y=40, line=dict(color="#64748b", dash="dot"), row=3, col=1)
    fig.add_hline(y=60, line=dict(color="#64748b", dash="dot"), row=3, col=1)
    fig.update_yaxes(title="3M, x", row=1, col=1)
    fig.update_yaxes(title="6M, x", row=2, col=1)
    fig.update_yaxes(title="Score", range=[0, 100], row=3, col=1)
    return _style(fig, 345)


def _render_near_term_refinancing(monthly: pd.DataFrame, start: pd.Timestamp) -> None:
    latest = monthly.dropna(subset=["near_term_refinancing_pressure"]).iloc[-1]
    st.markdown("### Near-Term Treasury Refinancing")
    headline = st.columns(3)
    headline[0].metric("Near-Term Refinancing Pressure", f"{latest['near_term_refinancing_pressure']:.1f} / 100")
    headline[1].metric("3M Pressure", f"{latest['pressure_3m_percentile']:.1f}", f"{latest['pressure_ratio_3m']:.2f}x run-rate")
    headline[2].metric("6M Pressure", f"{latest['pressure_6m_percentile']:.1f}", f"{latest['pressure_ratio_6m']:.2f}x run-rate")
    amounts = st.columns(3)
    amounts[0].metric("Next 3M Rollover", f"${latest['next_3m_rollover']:.2f}tn")
    amounts[1].metric("Next 6M Rollover", f"${latest['next_6m_rollover']:.2f}tn")
    amounts[2].metric("Treasury Data As Of", f"{latest['Date']:%Y-%m-%d}")
    with st.expander("Near-Term Refinancing Diagnostics"):
        fields = (
            "next_3m_rollover", "next_6m_rollover", "trailing_12m_rollover",
            "next_3m_monthly_run_rate", "next_6m_monthly_run_rate", "trailing_12m_monthly_run_rate",
            "pressure_ratio_3m", "pressure_3m_percentile", "pressure_ratio_6m",
            "pressure_6m_percentile", "near_term_refinancing_pressure",
        )
        st.dataframe(_diagnostic_table(latest, fields), use_container_width=True, hide_index=True)
    st.markdown("#### Near-Term Treasury Refinancing Pressure")
    st.plotly_chart(build_near_term_refinancing_chart(monthly, start), use_container_width=True, config=CHART_CONFIG)


def _funding_table(snapshot: TreasuryFundingPolicySnapshot) -> pd.DataFrame:
    historical = snapshot.annual[["Year", "Rollover12M", "RolloverYoYAbs", "RolloverYoYPct", "RolloverIntensity", "PeriodLabel"]].copy()
    historical = historical.rename(columns={
        "Rollover12M": "BASE Rollover $tn", "RolloverYoYAbs": "BASE YoY Change $tn",
        "RolloverYoYPct": "BASE YoY Change %", "RolloverIntensity": "BASE Rollover Intensity %",
    })
    historical["LONG Rollover $tn"] = np.nan
    historical["SHORT Rollover $tn"] = np.nan
    historical["LONG Rollover Intensity %"] = np.nan
    historical["SHORT Rollover Intensity %"] = np.nan
    forecasts = []
    for year, data in snapshot.forecast.groupby("Year"):
        indexed = data.set_index("Scenario")
        base_rollover = indexed.loc["BASE", "PrincipalRollover"]
        previous = historical.iloc[-1]["BASE Rollover $tn"] if not forecasts else forecasts[-1]["BASE Rollover $tn"]
        forecasts.append({
            "Year": year, "PeriodLabel": str(year),
            "LONG Rollover $tn": indexed.loc["LONG", "PrincipalRollover"],
            "BASE Rollover $tn": base_rollover,
            "SHORT Rollover $tn": indexed.loc["SHORT", "PrincipalRollover"],
            "BASE YoY Change $tn": base_rollover - previous,
            "BASE YoY Change %": (base_rollover / previous - 1) * 100 if previous else np.nan,
            "LONG Rollover Intensity %": indexed.loc["LONG", "RolloverIntensity"],
            "BASE Rollover Intensity %": indexed.loc["BASE", "RolloverIntensity"],
            "SHORT Rollover Intensity %": indexed.loc["SHORT", "RolloverIntensity"],
        })
    output = pd.concat([historical, pd.DataFrame(forecasts)], ignore_index=True)
    columns = ["PeriodLabel", "LONG Rollover $tn", "BASE Rollover $tn", "SHORT Rollover $tn",
               "BASE YoY Change $tn", "BASE YoY Change %", "LONG Rollover Intensity %",
               "BASE Rollover Intensity %", "SHORT Rollover Intensity %"]
    return output[columns].rename(columns={"PeriodLabel": "Year"})


def _render_funding_config(api_key: str | None, base_snapshot: TreasuryFiscalSnapshot) -> None:
    config = load_funding_config()
    with st.expander("Treasury Funding Assumptions"):
        financing_share = st.number_input(
            "Marketable Financing Share", min_value=0.0, max_value=100.0,
            value=float(config["marketable_financing_share"]) * 100, step=1.0,
            key="treasury_funding_market_share",
        )
        cbo = pd.DataFrame({"Year": list(map(int, config["cbo_deficit_bn"])),
                            "CBO Deficit, USD bn": list(config["cbo_deficit_bn"].values())})
        cbo_edit = st.data_editor(cbo, hide_index=True, disabled=["Year"], key="treasury_funding_cbo")
        curve = pd.DataFrame({"Tenor": list(TENORS),
                              "Reference Yield, %": [config["reference_yield_pct"][tenor] for tenor in TENORS]})
        curve_edit = st.data_editor(curve, hide_index=True, disabled=["Tenor"], key="treasury_funding_curve")
        shares = pd.DataFrame({"Tenor": list(TENORS)})
        for scenario in ("LONG", "BASE", "SHORT"):
            shares[f"{scenario}, %"] = [config["issuance_shares"][scenario][tenor] * 100 for tenor in TENORS]
        shares_edit = st.data_editor(shares, hide_index=True, disabled=["Tenor"], key="treasury_funding_shares")
        totals = " | ".join(f"{scenario} {shares_edit[f'{scenario}, %'].sum():.2f}%" for scenario in ("LONG", "BASE", "SHORT"))
        st.caption(f"Issuance share validation: {totals}. CBO source: {config['cbo_source']}.")
        if st.button("Save Funding Assumptions", key="save_treasury_funding_assumptions"):
            updated = dict(config)
            updated["marketable_financing_share"] = float(financing_share) / 100
            updated["cbo_deficit_bn"] = {str(int(row["Year"])): float(row["CBO Deficit, USD bn"]) for _, row in cbo_edit.iterrows()}
            updated["reference_yield_pct"] = {str(row["Tenor"]): float(row["Reference Yield, %"]) for _, row in curve_edit.iterrows()}
            updated["issuance_shares"] = {
                scenario: {str(row["Tenor"]): float(row[f"{scenario}, %"]) / 100 for _, row in shares_edit.iterrows()}
                for scenario in ("LONG", "BASE", "SHORT")
            }
            try:
                save_funding_config(updated)
                with st.spinner("Recalculating Treasury funding scenarios..."):
                    refresh_funding_policy_snapshot(api_key, refresh=False, treasury_snapshot=base_snapshot)
                st.success("Funding assumptions saved and forecast recalculated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Assumptions were not saved: {exc}")


def render_treasury_funding_policy_block(
    snapshot: TreasuryFundingPolicySnapshot,
    start: pd.Timestamp,
    api_key: str | None,
    base_snapshot: TreasuryFiscalSnapshot,
) -> None:
    st.divider()
    st.markdown("## Treasury Funding")
    st.caption("Structural funding pressure, maturity concentration and Treasury debt-management scenarios.")
    if snapshot.monthly.empty:
        st.warning("Treasury Funding snapshot is not available. Use Refresh Treasury & Fiscal Data.")
        return
    latest = snapshot.monthly.iloc[-1]
    st.markdown("### Structural Treasury Funding")
    metrics = st.columns(3)
    metrics[0].metric("12M Principal Rollover", f"${latest['Rollover12M']:.2f}tn", f"{latest['RolloverYoYAbs']:+.2f}tn YoY")
    metrics[1].metric("Rollover Intensity", f"{latest['RolloverIntensity']:.1f}%", f"{latest['RolloverIntensityYoYDeltaPP']:+.1f}pp")
    metrics[2].metric("Gross Financing Requirement", f"${latest['GrossFinancingRequirement']:.2f}tn" if pd.notna(latest['GrossFinancingRequirement']) else "n/a",
                      f"{latest['GFRToDebt']:.1f}% debt" if pd.notna(latest['GFRToDebt']) else None)
    metrics = st.columns(4)
    metrics[0].metric("Portfolio Avg Rate", f"{latest['PortfolioAvgRate']:.2f}%")
    metrics[1].metric("Marginal Funding Rate", f"{latest['MarginalFundingRate']:.2f}%")
    metrics[2].metric("WAM", f"{latest['WAMMonths']:.1f} months")
    metrics[3].metric("Bill Share", f"{latest['BillShare']:.1f}%")
    st.caption(f"MSPD {latest['Date']:%Y-%m-%d} | reconciliation {latest['ReconciliationFlag']} ({latest['ReconciliationErrorPct']:+.3f}%)")

    _render_near_term_refinancing(snapshot.monthly, start)

    show_forecast = st.toggle("Show 2027-2031 funding forecast", value=True, key="treasury_funding_show_forecast")
    actual_view = snapshot.annual.loc[snapshot.annual["Date"].ge(start)].copy()
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Annual Treasury Principal Rollover")
        st.plotly_chart(build_rollover_chart(actual_view, snapshot.forecast, show_forecast), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### YoY Change in Treasury Rollover")
        st.plotly_chart(build_rollover_yoy_chart(actual_view, snapshot.forecast, show_forecast), use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### 12M Rollover Intensity")
        st.plotly_chart(_scenario_chart(actual_view, snapshot.forecast, "RolloverIntensity", "RolloverIntensity", "%", show_forecast), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### Average Interest Rate of Marketable Treasury Debt")
        st.plotly_chart(_scenario_chart(actual_view, snapshot.forecast, "PortfolioAvgRate", "PortfolioAvgRate", "%", show_forecast), use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Weighted Average Maturity")
        st.plotly_chart(_scenario_chart(actual_view, snapshot.forecast, "WAMMonths", "WAMMonths", "Months", show_forecast), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### Bill Share of Marketable Debt")
        st.plotly_chart(_scenario_chart(actual_view, snapshot.forecast, "BillShare", "BillShare", "%", show_forecast), use_container_width=True, config=CHART_CONFIG)

    st.markdown("### Historical / Forecast Treasury Funding")
    st.dataframe(_funding_table(snapshot), use_container_width=True, hide_index=True)
    _render_funding_config(api_key, base_snapshot)
    with st.expander("Treasury Funding Audit"):
        treasury_columns = [
            "Date", "TotalMarketableDebt", "BillsOutstanding", "NotesOutstanding", "BondsOutstanding",
            "TIPSOutstanding", "FRNOutstanding", "Rollover12M", "RolloverIntensity", "RolloverBills",
            "RolloverNotes", "RolloverBonds", "RolloverTIPS", "RolloverFRN", "RolloverYoYAbs",
            "RolloverYoYPct", "RolloverIntensityYoYDeltaPP", "GrossFinancingRequirement", "GFRToDebt",
            "WAMMonths", "BillShare", "PortfolioAvgRate", "MarginalFundingRate",
            "PublishedMarketableDebt", "PublishedExclusionAdjustment", "AdjustedPublishedMarketableDebt",
            "ReconciliationErrorPct", "ReconciliationFlag", "AdjustedReconciliationErrorPct",
            "AdjustedReconciliationFlag",
            "next_3m_rollover", "next_6m_rollover", "trailing_12m_rollover",
            "next_3m_monthly_run_rate", "next_6m_monthly_run_rate", "trailing_12m_monthly_run_rate",
            "pressure_ratio_3m", "pressure_ratio_6m", "pressure_3m_percentile",
            "pressure_6m_percentile", "near_term_refinancing_pressure",
            "next_3m_bills", "next_3m_notes", "next_3m_bonds", "next_3m_tips", "next_3m_frn",
            "next_6m_bills", "next_6m_notes", "next_6m_bonds", "next_6m_tips", "next_6m_frn",
        ]
        st.dataframe(snapshot.monthly[treasury_columns].tail(24), use_container_width=True, hide_index=True)
        st.download_button("Download Treasury audit CSV", snapshot.monthly[treasury_columns].to_csv(index=False).encode(),
                           "treasury_funding_audit.csv", "text/csv", key="download_treasury_funding_audit")
        st.download_button("Download cohort forecast CSV", snapshot.forecast.to_csv(index=False).encode(),
                           "treasury_funding_forecast.csv", "text/csv", key="download_treasury_funding_forecast")
        st.json(snapshot.status)

    with st.expander("Treasury Funding Methodology"):
        st.write("12M Principal Rollover measures the outstanding principal of marketable Treasury securities scheduled to mature during the next 12 months. Rollover Intensity divides this amount by total marketable Treasury debt.")
        st.write("The measure includes Bills as well as Notes, Bonds, TIPS and FRNs approaching maturity. It is not gross auction turnover: a short-dated Bill is counted as principal when it matures, not repeatedly because it may be reissued during a year.")
        st.write("Forecast scenarios change Treasury issuance maturity composition, not the assumed macro yield environment. V1 therefore isolates debt-management effects.")
    with st.expander("Near-Term Treasury Refinancing Methodology"):
        st.write("Near-Term Treasury Refinancing Pressure measures whether the contractual Treasury maturity wall over the next 3 and 6 months is heavier or lighter than the principal rollover pace absorbed during the previous 12 months.")
        st.write("The model converts each horizon to a comparable monthly run-rate. The 3M ratio is 4 × Next 3M Rollover / Trailing 12M Rollover. The 6M ratio is 2 × Next 6M Rollover / Trailing 12M Rollover. A ratio above 1.0 means the upcoming maturity pace is greater than the pace absorbed over the previous year.")
        st.write("Both ratios become expanding point-in-time percentiles using only the post-2020 Treasury funding regime. Near-Term Refinancing Pressure = 60% × 3M Pressure Percentile + 40% × 6M Pressure Percentile.")
        st.write("12M Rollover Intensity measures the structural share of Treasury debt approaching maturity. Near-Term Refinancing Pressure instead measures the timing and concentration of that rollover relative to the recent refinancing pace. High structural intensity can coexist with low near-term pressure, and vice versa.")


def render_treasury_fiscal_regime_tab(api_key: str | None) -> None:
    st.subheader("Treasury & Fiscal Regime")
    if st.button("Refresh Treasury & Fiscal Data", key="treasury_fiscal_refresh"):
        try:
            with st.spinner("Updating Treasury and fiscal series..."):
                refreshed = refresh_snapshot(api_key, refresh=True)
                refresh_funding_policy_snapshot(api_key, refresh=True, treasury_snapshot=refreshed)
        except Exception as exc:
            st.error(f"Refresh failed; previous snapshot retained: {exc}")
    snapshot = read_snapshot()
    if snapshot.weekly.empty:
        try:
            with st.spinner("Building Treasury & Fiscal history..."):
                snapshot = refresh_snapshot(api_key)
        except Exception as exc:
            st.error(f"Data unavailable: {exc}")
            return
    weekly = snapshot.weekly.copy()
    weekly["Date"] = pd.to_datetime(weekly["Date"])
    fiscal = snapshot.fiscal.copy()
    financing = snapshot.financing.copy()
    available_through = weekly["Date"].max()
    for frame in (fiscal, financing):
        if not frame.empty:
            frame["Date"] = pd.to_datetime(frame["AvailableDate"])
    if not fiscal.empty:
        fiscal = fiscal.loc[fiscal["Date"].le(available_through)].copy()
    if not financing.empty:
        financing = financing.loc[financing["Date"].le(available_through)].copy()
    latest = weekly.iloc[-1]
    st.caption(f"Model {snapshot.status.get('ModelVersion', 'n/a')} | Weekly state {latest['Date']:%Y-%m-%d}")
    revised = [key for key, meta in snapshot.status.get("SourceStatus", {}).items()
               if "REVISED_HISTORY" in meta.get("State", "")]
    if revised:
        st.caption("Historical current-vintage FRED values with conservative release lags: " +
                   ", ".join(revised) + ". Publication timing is guarded; past revisions are not fully PIT-recoverable.")
    primary = st.columns(4, gap="small")
    for column, title, value in zip(
        primary,
        ("Policy Mix", "Treasury Liquidity", "Observed Fiscal Impulse", "Treasury Financing Pressure"),
        (latest["PolicyMix"], latest["TreasuryLiquidityState"], latest["FiscalImpulseState"],
         latest["TreasuryFinancingPressure"]),
    ):
        with column:
            st.markdown(f"**{title}**")
            st.write(value)
    secondary = st.columns(4, gap="small")
    for column, title, value in zip(
        secondary, ("Fiscal Stance", "Financing Mix", "RRP Buffer", "Absorption Capacity"),
        (latest["FiscalStanceState"], latest["FinancingMix"], latest["RRPBuffer"],
         latest["AbsorptionCapacity"]),
    ):
        with column:
            st.markdown(f"**{title}**")
            st.write(value)
    st.caption(
        f"Liquidity as of {snapshot.status.get('SourceStatus', {}).get('WALCL', {}).get('ObservationDate', 'n/a')} | "
        f"Fiscal as of {latest['FiscalObservationDate']:%Y-%m} | "
        f"Financing as of {pd.Timestamp(latest['FinancingObservationDate']).to_period('Q')}"
        if pd.notna(latest["FiscalObservationDate"]) and pd.notna(latest["FinancingObservationDate"])
        else "One or more lower-frequency sources are unavailable."
    )
    selected = st.radio("Time range", ["1Y", "3Y", "5Y", "10Y", "MAX"], index=2,
                        horizontal=True, key="treasury_fiscal_range")
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(selected)
    start = weekly["Date"].max() - pd.DateOffset(years=years) if years else weekly["Date"].min()
    view = weekly.loc[weekly["Date"].ge(start)]
    fiscal_view = fiscal.loc[fiscal["Date"].ge(start)] if not fiscal.empty else fiscal
    financing_view = financing.loc[financing["Date"].ge(start)] if not financing.empty else financing

    st.markdown("### Treasury & Fiscal Policy Mix History")
    st.caption(
        f"Liquidity level: {_number(latest['NetLiquidityLevel'] * 100)}% GDP "
        f"(PIT percentile {_number(latest['NetLiquidityLevelPercentile'] * 100, 0)}); "
        f"liquidity impulse: {_number(latest['TreasuryLiquidityImpulse'])}"
    )
    st.plotly_chart(build_net_liquidity_chart(view), use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Treasury Liquidity Contributions")
        horizon = st.segmented_control("Horizon", [4, 13, 26], default=13, format_func=lambda value: f"{value}W",
                                       key="treasury_contribution_horizon")
        horizon = horizon or 13
        st.plotly_chart(build_contributions_chart(view, horizon), use_container_width=True, config=CHART_CONFIG)
        st.caption(f"Current driver: {latest[f'LiquidityDriver_{horizon}W']}")
    with right:
        st.markdown("### System Liquidity Buffers")
        mode = st.segmented_control("Scale", ["Normalized", "Raw"], default="Normalized",
                                    key="treasury_buffer_scale")
        st.plotly_chart(build_buffers_chart(view, mode == "Raw"), use_container_width=True, config=CHART_CONFIG)
        st.caption("Low RRP is a reduced future shock absorber, not a stress signal by itself.")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Fiscal Impulse")
        if fiscal_view.empty:
            st.info("Fiscal data unavailable.")
        else:
            st.plotly_chart(build_fiscal_chart(fiscal_view), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.markdown("### Fiscal Impulse Decomposition")
        horizon = st.segmented_control("Horizon", [3, 6, 12], default=3, format_func=lambda value: f"{value}M",
                                       key="treasury_fiscal_horizon") or 3
        if fiscal_view.empty:
            st.info("Fiscal data unavailable.")
        else:
            st.plotly_chart(build_fiscal_decomposition_chart(fiscal_view, horizon),
                            use_container_width=True, config=CHART_CONFIG)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### Treasury Financing Structure")
        units = st.segmented_control("Units", ["% GDP", "USD"], default="% GDP",
                                     key="treasury_financing_units")
        if financing_view.empty:
            st.info("Financing data unavailable.")
        else:
            st.plotly_chart(build_financing_chart(financing_view, units != "USD"),
                            use_container_width=True, config=CHART_CONFIG)
            st.caption(f"Financing mix: {latest['FinancingMix']}. Duration is a non-bill debt proxy.")
    with right:
        st.markdown("### Treasury Financing Pressure Map")
        st.plotly_chart(build_financing_map(view), use_container_width=True, config=CHART_CONFIG)
    st.markdown("### Current Interpretation")
    st.write(_interpretation(latest))

    with st.expander("Model Diagnostics"):
        st.caption("Independent liquidity, fiscal and financing models. No aggregate numerical score.")
        for title, row, fields in (
            ("Treasury Liquidity", latest, (
                "WALCL", "WTREGEN", "WDTGAL", "TGAFastGap", "RRP", "RRPBufferRaw",
                "RRPBufferPercentile", "WRESBAL", "ReserveBufferRaw", "ReserveBufferPercentile",
                "NetLiquidity", "NetLiquidityLevel", "NetLiquidityLevelPercentile",
                "FastLiquidityImpulse", "MediumLiquidityImpulse", "SlowLiquidityImpulse",
                "RRPImpulse_4W", "RRPImpulse_13W", "RRPImpulse_26W",
                "FastLiquidityZ", "MediumLiquidityZ", "SlowLiquidityZ", "TreasuryLiquidityImpulse",
                "TreasuryLiquidityState", "LiquidityDriver",
            )),
            ("Fiscal", latest, (
                "MonthlyReceipts", "MonthlyOutlays", "Receipts12M", "Outlays12M", "Deficit12M",
                "FiscalStanceRaw", "FiscalStancePercentile", "FiscalStanceState",
                "FastFiscalImpulseRaw", "MediumFiscalImpulseRaw",
                "StructuralFiscalImpulseRaw", "FastFiscalZ", "MediumFiscalZ", "StructuralFiscalZ",
                "FiscalImpulse", "FiscalImpulseState", "SpendingImpulse", "RevenueImpulse",
            )),
            ("Treasury Financing", latest, (
                "FGTSL", "Bills", "NonBillDebt", "TotalNetIssuance4Q", "BillNetIssuance4Q",
                "DurationNetIssuance4Q", "TotalSupplyLoadRaw", "BillSupplyLoadRaw",
                "DurationSupplyLoadRaw", "TotalSupplyPercentile", "BillSupplyPercentile",
                "DurationSupplyPercentile", "BillFinancingShare", "DurationFinancingShare",
                "FinancingMix", "AbsorptionCapacity", "TreasuryFinancingPressure", "PolicyMix",
            )),
        ):
            st.markdown(f"**{title}**")
            st.dataframe(_diagnostic_table(row, fields), use_container_width=True, hide_index=True)
    st.markdown("### Data Freshness")
    st.dataframe(_freshness(snapshot), use_container_width=True, hide_index=True)
    with st.expander("Data Sources"):
        st.caption(snapshot.status.get("TimingConvention", ""))
        st.dataframe(pd.DataFrame([
            {"Series": key, "Frequency": frequency, "Units": unit, "Block": domain,
             "Source": f"https://fred.stlouisfed.org/series/{key}"}
            for key, (frequency, unit, domain) in SOURCE_SPECS.items()
        ]), use_container_width=True, hide_index=True)
        st.caption(f"Funding Conditions: {snapshot.status.get('FundingStatus', 'DATA UNAVAILABLE')}")
        st.caption("Treasury Liquidity decomposes U.S. liquidity; do not add it to the Global Liquidity score.")

    funding_policy = read_funding_policy_snapshot()
    if funding_policy.monthly.empty or funding_policy.policy.empty:
        try:
            with st.spinner("Building Treasury Funding history..."):
                funding_policy = refresh_funding_policy_snapshot(api_key, treasury_snapshot=snapshot)
        except Exception as exc:
            st.warning(f"Treasury Funding data unavailable: {exc}")
    render_treasury_funding_policy_block(funding_policy, start, api_key, snapshot)

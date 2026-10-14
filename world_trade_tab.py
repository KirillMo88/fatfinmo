from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from world_trade import REGION_ORDER, TradeBalanceService, load_world_trade_module


@st.cache_data(show_spinner="Loading World Trade data…", ttl=180 * 24 * 60 * 60)
def load_world_trade_snapshot(refresh_nonce: int = 0) -> dict[str, Any]:
    return load_world_trade_module(force_refresh=refresh_nonce > 0)


def render_world_trade_module(snapshot: dict[str, Any]) -> None:
    st.header("World Trade")
    st.caption("Historical cycle diagnostics, merchandise trade balances, and China export geography")

    _render_global_trade(snapshot["global_trade"])
    st.divider()
    _render_trade_balances(snapshot["trade_balances"])
    st.divider()
    _render_china_exports(snapshot["china_exports"])


def _render_global_trade(result: dict[str, Any]) -> None:
    st.subheader("WORLD TRADE | Historical Cycle Diagnostics 1970–Present")
    warning = result.get("warning")
    if warning:
        st.warning(warning)
    frame = result.get("data", pd.DataFrame()).copy()
    if frame.empty:
        st.info("World Bank trade history is not available yet. Use Refresh to try again.")
        return
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.sort_values("year")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Annual growth: trade vs GDP", "Real and nominal global exports", "Globalization measure: nominal exports / GDP"),
    )
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["real_exports_growth_pct"], name="Real exports YoY",
        line={"color": "#60a5fa", "width": 2.2},
        hovertemplate="%{x}: %{y:.2f}%<extra>Real exports YoY</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["real_exports_growth_5y_avg_pct"], name="5Y moving average",
        line={"color": "#22d3ee", "width": 1.7, "dash": "dash"},
        hovertemplate="%{x}: %{y:.2f}%<extra>5Y moving average</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["real_gdp_growth_pct"], name="Real GDP YoY",
        line={"color": "#fb923c", "width": 2.0},
        hovertemplate="%{x}: %{y:.2f}%<extra>Real GDP YoY</extra>",
    ), row=1, col=1)
    forecast = result.get("metadata", {}).get("forecast")
    if forecast:
        fig.add_trace(go.Scatter(
            x=[forecast["year"]], y=[forecast["growth_pct"]], name="WTO goods forecast",
            mode="markers", marker={"color": "#ef4444", "size": 11, "symbol": "diamond"},
            customdata=[[forecast.get("publication_date", "unknown")]],
            hovertemplate=("WTO Merchandise Trade Volume Forecast — Goods Only"
                           "<br>Year: %{x}<br>Growth: %{y:.2f}%"
                           "<br>Publication: %{customdata[0]}<extra></extra>"),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["real_exports_usd_trn"], name="Real exports (constant USD)",
        line={"color": "#60a5fa", "width": 2.2},
        hovertemplate="%{x}: $%{y:.2f}tn<extra>Real exports</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["nominal_exports_usd_trn"], name="Nominal exports (current USD)",
        line={"color": "#fb923c", "width": 2.2},
        hovertemplate="%{x}: $%{y:.2f}tn<extra>Nominal exports</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame["year"], y=frame["exports_gdp_pct"], name="Exports / GDP",
        line={"color": "#38bdf8", "width": 2.3}, fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        hovertemplate="%{x}: %{y:.2f}%<extra>Exports / GDP</extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line={"color": "#64748b", "width": 1}, row=1, col=1)
    fig.add_vline(x=2008, line={"color": "#64748b", "width": 1, "dash": "dot"}, row=3, col=1)
    latest_real = frame.dropna(subset=["real_exports_usd_trn"]).iloc[-1]
    latest_nominal = frame.dropna(subset=["nominal_exports_usd_trn"]).iloc[-1]
    fig.add_annotation(
        x=latest_real["year"], y=latest_real["real_exports_usd_trn"], row=2, col=1,
        text=f"{int(latest_real['year'])}: ${latest_real['real_exports_usd_trn']:.2f}T",
        showarrow=False, xanchor="left", xshift=8, font={"color": "#60a5fa", "size": 11},
    )
    fig.add_annotation(
        x=latest_nominal["year"], y=latest_nominal["nominal_exports_usd_trn"], row=2, col=1,
        text=f"{int(latest_nominal['year'])}: ${latest_nominal['nominal_exports_usd_trn']:.2f}T",
        showarrow=False, xanchor="left", xshift=8, font={"color": "#fb923c", "size": 11},
    )
    fig.update_yaxes(title_text="Annual growth, %", ticksuffix="%", gridcolor="#293241", row=1, col=1)
    fig.update_yaxes(title_text="USD trillion", tickprefix="$", gridcolor="#293241", row=2, col=1)
    fig.update_yaxes(title_text="Share of GDP, %", ticksuffix="%", gridcolor="#293241", row=3, col=1)
    fig.update_xaxes(gridcolor="#293241", dtick=5, row=3, col=1)
    fig.update_layout(
        height=980, margin={"l": 62, "r": 25, "t": 58, "b": 38}, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    metadata = result.get("metadata", {})
    st.caption(
        f"Source: {metadata.get('source', 'World Bank WDI')} · "
        f"Retrieved: {metadata.get('retrieved_at_utc', 'unknown')} · "
        "Only completed calendar years are shown."
    )


def _render_trade_balances(result: dict[str, Any]) -> None:
    st.subheader("Top 5 Net Exporters and Net Importers")
    metadata = result.get("metadata", {})
    frame = result.get("data", pd.DataFrame()).copy()
    comparison = st.segmented_control(
        "Period", ["Previous Year", "10 Years Ago"], default="Previous Year", key="world_trade_balance_period",
    )
    if frame.empty or not metadata.get("latest_valid_year"):
        st.info("Data incomplete: no completed year has passed the WTO 150-economy and 95% coverage checks.")
        if result.get("warning"):
            st.caption(result["warning"])
        return
    year = int(metadata["latest_valid_year"] if comparison == "Previous Year" else metadata["ten_year_comparison_year"])
    top = TradeBalanceService.top_balances(frame, year)
    if top.empty:
        st.info(f"No validated trade-balance records are available for {year}.")
        return
    colors = np.where(top["trade_balance_usd_bn"].ge(0), "#3b82f6", "#f97316")
    labels = [f"{value:+,.0f}" for value in top["trade_balance_usd_bn"]]
    hover_values = top[["exports_usd_mn", "imports_usd_mn"]].to_numpy(dtype=float) / 1000.0
    fig = go.Figure(go.Bar(
        x=top["trade_balance_usd_bn"], y=top["economy"], orientation="h",
        marker_color=colors, text=labels, textposition="outside", cliponaxis=False,
        customdata=hover_values,
        hovertemplate=("<b>%{y}</b><br>Balance: $%{x:,.1f}bn"
                       "<br>Exports: $%{customdata[0]:,.1f}bn"
                       "<br>Imports: $%{customdata[1]:,.1f}bn"
                       f"<br>Year: {year}<br>Source: WTO<extra></extra>"),
    ))
    fig.add_vline(x=0, line={"color": "#94a3b8", "width": 1.4})
    max_abs = max(float(top["trade_balance_usd_bn"].abs().max()), 1.0)
    fig.update_layout(
        title={"text": f"Merchandise trade balance, USD bn | {year}", "x": 0.5},
        height=520, margin={"l": 110, "r": 80, "t": 52, "b": 48}, showlegend=False,
        xaxis={"title": "USD billions", "range": [-max_abs * 1.25, max_abs * 1.25], "gridcolor": "#293241"},
        yaxis={"title": None},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    validation = metadata.get("validation", {}).get(str(year), metadata.get("validation", {}).get(year, {}))
    st.caption(
        f"Source: {metadata.get('source', 'WTO')} · Validated year: {year} · "
        f"Economies: {validation.get('economies', 'n/a')} · "
        f"Export coverage: {_fmt_pct(validation.get('export_coverage_pct'))} · "
        f"Import coverage: {_fmt_pct(validation.get('import_coverage_pct'))}."
    )
    if result.get("warning"):
        st.warning(result["warning"])


def _render_china_exports(result: dict[str, Any]) -> None:
    st.subheader("China Export Geography")
    metadata = result.get("metadata", {})
    frame = result.get("data", pd.DataFrame()).copy()
    latest = metadata.get("latest_valid_year")
    status_labels = {
        "current": "Latest official GACC data", "cached": "Cached valid data",
        "historical_only": "Historical imported data", "unavailable": "Unavailable",
    }
    meta_left, meta_middle, meta_right = st.columns(3)
    meta_left.metric("Latest annual observation", int(latest) if latest else "Not loaded")
    meta_middle.metric("Source", "GACC" if latest else "Historical import")
    meta_right.metric("Data status", status_labels.get(result.get("status"), str(result.get("status", "unknown"))))
    period_options = ["2000", "2010", "2020", "Previous Year"]
    metric_col, period_col = st.columns([1, 2])
    with metric_col:
        metric = st.segmented_control("Metric", ["USD bn", "Share %"], default="Share %", key="china_export_metric")
    with period_col:
        period = st.segmented_control("Period", period_options, default="Previous Year", key="china_export_period")
    selected_year = int(latest) if period == "Previous Year" and latest else int(period) if period != "Previous Year" else None
    if selected_year is None:
        st.info("Previous Year is unavailable until a completed GACC year passes validation. Historical comparisons remain available below.")
    else:
        selected = frame.loc[pd.to_numeric(frame["year"], errors="coerce").eq(selected_year)].copy()
        if selected.empty:
            st.info(f"No validated China export geography is available for {selected_year}.")
        else:
            selected["region"] = pd.Categorical(selected["region"], categories=REGION_ORDER, ordered=True)
            selected = selected.sort_values("region")
            value_col = "exports_usd_bn" if metric == "USD bn" else "share_pct"
            suffix = "" if metric == "USD bn" else "%"
            prefix = "$" if metric == "USD bn" else ""
            color_map = dict(zip(REGION_ORDER, ["#38bdf8", "#818cf8", "#f97316", "#f59e0b", "#a78bfa", "#ec4899", "#22c55e", "#94a3b8", "#14b8a6"]))
            custom = selected[["exports_usd_bn", "share_pct", "source"]].to_numpy()
            fig = go.Figure(go.Bar(
                y=selected["region"], x=selected[value_col], orientation="h",
                marker_color=[color_map[str(region)] for region in selected["region"]],
                text=selected[value_col], texttemplate=("$%{text:,.1f}" if metric == "USD bn" else "%{text:.1f}%"),
                textposition="outside", cliponaxis=False,
                customdata=custom,
                hovertemplate=("<b>%{y}</b>"
                               f"<br>Year: {selected_year}"
                               "<br>Exports: $%{customdata[0]:,.1f}bn"
                               "<br>Share: %{customdata[1]:.2f}%"
                               "<br>Source: %{customdata[2]}"
                               f"<br>Status: {status_labels.get(result.get('status'), result.get('status', 'unknown'))}<extra></extra>"),
            ))
            fig.update_layout(
                title={"text": f"China exports by region | {selected_year}", "x": 0.5}, height=470,
                margin={"l": 120, "r": 55, "t": 52, "b": 48}, showlegend=False,
                xaxis={"title": metric, "ticksuffix": suffix, "tickprefix": prefix, "gridcolor": "#293241"},
                yaxis={"title": None, "categoryorder": "array", "categoryarray": REGION_ORDER, "autorange": "reversed"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    _render_china_comparison_table(frame, latest, selected_year, metric)
    if result.get("warning"):
        st.warning(result["warning"])
    source = metadata.get("source", "Historical research dataset; GACC Previous Year not loaded")
    st.caption(
        f"Source: {source} · Previous Year means the latest completed, validated GACC calendar year"
        + (f" ({int(latest)})." if latest else ".")
    )


def _render_china_comparison_table(frame: pd.DataFrame, latest: int | None, selected_year: int | None, metric: str) -> None:
    years = [2000, 2010, 2020] + ([int(latest)] if latest and int(latest) not in {2000, 2010, 2020} else [])
    if frame.empty:
        return
    visible = frame.loc[pd.to_numeric(frame["year"], errors="coerce").isin(years)].copy()
    value_column = "exports_usd_bn" if metric == "USD bn" else "share_pct"
    values = visible.pivot_table(index="region", columns="year", values=value_column, aggfunc="first").reindex(REGION_ORDER)
    table = pd.DataFrame({"Region": REGION_ORDER})
    for year in years:
        table[str(year)] = values[year].to_numpy() if year in values else np.nan
    total = {"Region": "Total"}
    for year in years:
        total[str(year)] = float(values[year].sum()) if year in values else np.nan
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)

    def highlight(column: pd.Series) -> list[str]:
        if selected_year is not None and str(selected_year) == str(column.name):
            return ["background-color: rgba(56, 189, 248, 0.16); font-weight: 600"] * len(column)
        return [""] * len(column)

    formats = {column: ("{:,.1f}" if metric == "USD bn" else "{:.1f}%") for column in table.columns if column != "Region"}
    styled = table.style.format(formats, na_rep="—").apply(highlight, axis=0)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=390)


def _fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.1f}%" if np.isfinite(number) else "n/a"

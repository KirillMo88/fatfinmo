from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from economic_efficiency import (
    PWT_VERSION,
    build_country_history,
    build_efficiency_table,
    eligible_countries,
    load_economic_efficiency_data,
    resolve_reference_year,
)


COUNTRY_COLORS = [
    "#4f8fe8", "#e76f51", "#2a9d8f", "#f4a261", "#9b7ede",
    "#e0568f", "#54b96b", "#c49a5a", "#45a6c8", "#8d99ae", "#f2c14e",
]
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner="Loading Penn World Table 11.0…", ttl=30 * 24 * 60 * 60)
def load_economic_efficiency_snapshot(refresh_nonce: int = 0) -> dict[str, Any]:
    return load_economic_efficiency_data(force_refresh=refresh_nonce > 0)


def render_economic_efficiency_module(
    snapshot: dict[str, Any], world_gdp_snapshot: dict[str, Any]
) -> None:
    st.header("Countries Economic Efficiency")
    if snapshot.get("warning"):
        st.warning(snapshot["warning"])
    pwt = snapshot.get("pwt", pd.DataFrame()).copy()
    services = snapshot.get("services", pd.DataFrame()).copy()
    world_gdp = world_gdp_snapshot.get("data", pd.DataFrame()).copy()
    if pwt.empty or world_gdp.empty:
        st.info("PWT 11.0 or synchronized World Bank GDP data is unavailable. Use Refresh World Economy to try again.")
        return

    try:
        reference_year = resolve_reference_year(pwt, world_gdp)
    except Exception as exc:
        st.error(f"Economic Efficiency data could not be synchronized: {exc}")
        return

    countries = eligible_countries(pwt, reference_year)
    selected_iso3 = _country_selector(countries)
    table = build_efficiency_table(
        pwt,
        services,
        world_gdp,
        reference_year=reference_year,
        selected_iso3=selected_iso3,
    )
    mode = "Country Historical Analysis" if selected_iso3 else "Global Comparison"
    metadata = snapshot.get("metadata", {})
    st.caption(
        f"Mode: {mode} · Synchronized observation year: {reference_year} · "
        f"PWT {metadata.get('pwt_version', PWT_VERSION)} · Rankings: World Bank nominal GDP"
    )

    _render_efficiency_table(table, reference_year)
    st.divider()
    if selected_iso3:
        history = build_country_history(
            pwt,
            services,
            world_gdp,
            iso3=selected_iso3,
            reference_year=reference_year,
        )
        if history.empty:
            st.info("The selected country has no valid historical observations.")
        else:
            _render_historical_charts(history)
    else:
        _render_global_charts(table, reference_year)

    retrieved = metadata.get("retrieved_at_utc", "unknown")
    st.caption(
        "Sources: Penn World Table 11.0 (ctfp, rtfpna, cgdpo, cn, rgdpna, rnna, emp, avh) · "
        "World Bank WDI (NY.GDP.MKTP.CD, NV.SRV.TOTL.ZS) · "
        f"PWT license: CC BY 4.0 · Retrieved: {retrieved}"
    )


def _country_selector(countries: pd.DataFrame) -> str | None:
    names = dict(zip(countries["iso3"], countries["country"]))
    options: list[str | None] = [None, *countries["iso3"].tolist()]
    key = "economic_efficiency_country"
    if st.session_state.get(key) not in options:
        st.session_state[key] = None
    selector_col, reset_col = st.columns([5, 1])
    with selector_col:
        selected = st.selectbox(
            "Select Country",
            options,
            format_func=lambda code: "Global Comparison — Top 10" if code is None else f"{names.get(code, code)} ({code})",
            key=key,
        )
    with reset_col:
        st.write("")
        if st.button("Back to Global", disabled=selected is None, key="economic_efficiency_reset", use_container_width=True):
            st.session_state[key] = None
            st.rerun()
    return selected


def _render_efficiency_table(table: pd.DataFrame, reference_year: int) -> None:
    st.subheader("Economic Efficiency Table")
    st.caption(
        f"Top 10 by nominal GDP in {reference_year}. Ten-year changes compare exact observations in "
        f"{reference_year} and {reference_year - 10}; missing endpoints remain N/A."
    )
    display = pd.DataFrame({
        "Rank": table["position"],
        "Country": table["country_name"] + " (" + table["iso3"] + ")",
        "TFP Level, US = 100": table["tfp_level"],
        "TFP Growth 10Y": table["tfp_growth_10y_pct"],
        "GDP / Capital": table["capital_productivity_current_pct"],
        "GDP / Capital Change 10Y": table["capital_productivity_change_10y_pct"],
        "Services / GDP": table["services_gdp_pct"],
        "Services Year": table["services_year"].astype("Int64"),
        "GDP per Hour Worked": table["gdp_per_hour_ppp"],
        "GDP per Hour Worked 10Y": table["gdp_per_hour_change_10y_pct"],
        "_selected": table["is_selected"].fillna(False),
    })
    metric_columns = [
        "TFP Growth 10Y", "GDP / Capital Change 10Y", "GDP per Hour Worked 10Y"
    ]
    styler = display.drop(columns="_selected").style.format({
        "TFP Level, US = 100": lambda value: _fmt(value, "{:.1f}"),
        "TFP Growth 10Y": _fmt_pct,
        "GDP / Capital": _fmt_pct,
        "GDP / Capital Change 10Y": _fmt_pct,
        "Services / GDP": _fmt_pct,
        "Services Year": lambda value: "N/A" if pd.isna(value) else str(int(value)) + ("*" if int(value) != reference_year else ""),
        "GDP per Hour Worked": lambda value: _fmt(value, "${:,.1f}"),
        "GDP per Hour Worked 10Y": _fmt_pct,
    }, na_rep="N/A")
    styler = styler.map(_change_color, subset=metric_columns)
    selected_mask = display["_selected"].to_numpy(dtype=bool)
    styler = styler.apply(
        lambda _: ["background-color: rgba(79,143,232,0.16)" if flag else "" for flag in selected_mask],
        axis=0,
    )
    st.dataframe(styler, use_container_width=True, hide_index=True, height=min(470, 42 + 35 * len(display)))
    if table["services_is_older"].fillna(False).any():
        st.caption("* Services/GDP uses the most recent earlier WDI observation; the actual source year is shown.")


def _render_global_charts(table: pd.DataFrame, reference_year: int) -> None:
    top10 = table.head(10).copy()
    st.subheader("Productivity Charts")
    left, right = st.columns(2, gap="medium")
    with left:
        _plot_global_bar(
            top10, "tfp_level", "TFP Level — US = 100", "TFP level index",
            reference_year=reference_year, reference_line=100.0,
            hover_lines=[("PWT variable", "ctfp × 100")],
        )
    with right:
        _plot_global_bar(
            top10, "tfp_growth_10y_pct", "TFP Growth — Last 10 Years", "10Y growth (%)",
            reference_year=reference_year, zero_line=True, signed=True,
            hover_lines=[("Formula", "rtfpna[t] / rtfpna[t-10] − 1")],
        )

    left, right = st.columns(2, gap="medium")
    with left:
        show_services = st.toggle("Show Services Share", key="efficiency_global_services_overlay")
        _plot_capital_productivity(top10, reference_year, show_services)
    with right:
        _plot_global_bar(
            top10,
            "capital_productivity_change_10y_pct",
            "Capital Productivity Change — 10Y",
            "10Y change (%)",
            reference_year=reference_year,
            zero_line=True,
            signed=True,
            hover_lines=[("Formula", "change in rgdpna / rnna")],
        )

    left, _ = st.columns(2, gap="medium")
    with left:
        _plot_labor_productivity(top10, reference_year)

    st.subheader("Economic Efficiency Matrix — Top 10 Economies")
    size_mode = st.segmented_control(
        "Bubble Size",
        ["Nominal GDP", "GDP PPP", "Capital Stock"],
        default="Nominal GDP",
        key="efficiency_bubble_size",
    ) or "Nominal GDP"
    _plot_efficiency_matrix(top10, reference_year, size_mode)


def _plot_global_bar(
    frame: pd.DataFrame,
    column: str,
    title: str,
    y_title: str,
    *,
    reference_year: int,
    reference_line: float | None = None,
    zero_line: bool = False,
    signed: bool = False,
    hover_lines: list[tuple[str, str]] | None = None,
) -> None:
    data = frame.dropna(subset=[column]).sort_values(column, ascending=False)
    if data.empty:
        st.info(f"No valid data for {title}.")
        return
    values = pd.to_numeric(data[column], errors="coerce")
    colors = np.where(values.ge(0), "#4f8fe8", "#e76f51") if signed else "#4f8fe8"
    extra = "".join(f"<br>{label}: {value}" for label, value in (hover_lines or []))
    fig = go.Figure(go.Bar(
        x=data["country_name"], y=values, marker_color=colors,
        text=values, texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False,
        customdata=data[["iso3"]],
        hovertemplate=(
            "<b>%{x} (%{customdata[0]})</b><br>Value: %{y:.2f}<br>Observation year: "
            f"{reference_year}{extra}<br>Source: PWT {PWT_VERSION}<extra></extra>"
        ),
    ))
    if reference_line is not None:
        fig.add_hline(y=reference_line, line={"color": "#f2c14e", "dash": "dash", "width": 1.4}, annotation_text="USA = 100")
    if zero_line:
        fig.add_hline(y=0, line={"color": "#94a3b8", "width": 1.2})
    fig.update_layout(
        title={"text": title, "x": 0.5}, height=430,
        margin={"l": 55, "r": 25, "t": 55, "b": 85}, showlegend=False, hovermode="closest",
        xaxis={"title": None, "tickangle": -25},
        yaxis={"title": y_title, "gridcolor": "rgba(148,163,184,0.20)"},
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _plot_capital_productivity(frame: pd.DataFrame, reference_year: int, show_services: bool) -> None:
    data = frame.dropna(subset=["capital_productivity_current_pct"]).sort_values(
        "capital_productivity_current_pct", ascending=False
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    inverse = 100.0 / data["capital_productivity_current_pct"]
    custom = np.column_stack((inverse, data["services_gdp_pct"], data["services_year"].fillna(-1), data["iso3"]))
    fig.add_trace(go.Bar(
        x=data["country_name"], y=data["capital_productivity_current_pct"], name="GDP / Capital",
        marker_color="#4f8fe8", customdata=custom,
        hovertemplate=(
            "<b>%{x} (%{customdata[3]})</b><br>GDP / Capital: %{y:.2f}%"
            "<br>Capital / GDP: %{customdata[0]:.2f}x"
            "<br>Services / GDP: %{customdata[1]:.2f}% (year %{customdata[2]:.0f})"
            f"<br>Observation year: {reference_year}<br>Formula: cgdpo / cn<br>Source: PWT {PWT_VERSION} + WDI<extra></extra>"
        ),
    ), secondary_y=False)
    if show_services:
        fig.add_trace(go.Scatter(
            x=data["country_name"], y=data["services_gdp_pct"], mode="markers", name="Services / GDP",
            marker={"size": 10, "color": "#f2c14e", "line": {"color": "#111827", "width": 1}},
            customdata=data[["services_year"]],
            hovertemplate="<b>%{x}</b><br>Services / GDP: %{y:.2f}%<br>WDI year: %{customdata[0]:.0f}<extra></extra>",
        ), secondary_y=True)
    fig.update_layout(
        title={"text": "Capital Productivity — GDP / Capital Stock", "x": 0.5}, height=430,
        margin={"l": 55, "r": 55, "t": 55, "b": 85}, hovermode="closest",
        legend={"orientation": "h", "y": -0.24, "x": 0.5, "xanchor": "center"},
        xaxis={"tickangle": -25},
    )
    fig.update_yaxes(title_text="GDP / Capital (%)", gridcolor="rgba(148,163,184,0.20)", secondary_y=False)
    fig.update_yaxes(title_text="Services / GDP (%)", showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _plot_labor_productivity(frame: pd.DataFrame, reference_year: int) -> None:
    data = frame.dropna(subset=["gdp_per_hour_ppp"]).sort_values("gdp_per_hour_ppp", ascending=False)
    custom = data[["iso3", "avh", "gdp_per_hour_change_10y_pct"]].to_numpy()
    fig = go.Figure(go.Bar(
        x=data["country_name"], y=data["gdp_per_hour_ppp"], marker_color="#4f8fe8",
        text=data["gdp_per_hour_ppp"], texttemplate="$%{text:.1f}", textposition="outside", cliponaxis=False,
        customdata=custom,
        hovertemplate=(
            "<b>%{x} (%{customdata[0]})</b><br>GDP per hour: $%{y:,.2f} PPP"
            "<br>Average annual hours: %{customdata[1]:,.0f}"
            "<br>10Y growth: %{customdata[2]:+.2f}%"
            f"<br>Observation year: {reference_year}<br>Formula: cgdpo / (emp × avh)<br>Source: PWT {PWT_VERSION}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title={"text": "Labor Productivity — GDP per Hour Worked", "x": 0.5}, height=430,
        margin={"l": 55, "r": 25, "t": 55, "b": 85}, showlegend=False, hovermode="closest",
        xaxis={"tickangle": -25}, yaxis={"title": "PPP USD/hour", "tickprefix": "$", "gridcolor": "rgba(148,163,184,0.20)"},
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _plot_efficiency_matrix(frame: pd.DataFrame, reference_year: int, size_mode: str) -> None:
    size_column = {"Nominal GDP": "nominal_gdp_usd", "GDP PPP": "cgdpo", "Capital Stock": "cn"}[size_mode]
    data = frame.dropna(subset=["tfp_level", "tfp_growth_10y_pct", size_column]).copy()
    if data.empty:
        st.info("No complete observations are available for the efficiency matrix.")
        return
    x_median = float(data["tfp_level"].median())
    y_median = float(data["tfp_growth_10y_pct"].median())
    sizes = pd.to_numeric(data[size_column], errors="coerce").clip(lower=0)
    max_size = max(float(sizes.max()), 1.0)
    x_pad = max(float(data["tfp_level"].max() - data["tfp_level"].min()) * 0.15, 5.0)
    y_pad = max(float(data["tfp_growth_10y_pct"].max() - data["tfp_growth_10y_pct"].min()) * 0.18, 2.0)
    x0, x1 = float(data["tfp_level"].min()) - x_pad, float(data["tfp_level"].max()) + x_pad
    y0, y1 = float(data["tfp_growth_10y_pct"].min()) - y_pad, float(data["tfp_growth_10y_pct"].max()) + y_pad
    custom = data[[
        "country_name", "nominal_gdp_usd", "capital_productivity_current_pct",
        "capital_productivity_change_10y_pct", "services_gdp_pct", "services_year",
        "gdp_per_hour_ppp", "gdp_per_hour_change_10y_pct",
    ]].to_numpy()
    fig = go.Figure(go.Scatter(
        x=data["tfp_level"], y=data["tfp_growth_10y_pct"], mode="markers+text",
        text=data["iso3"], textposition=["top center" if i % 2 == 0 else "middle right" for i in range(len(data))],
        textfont={"size": 11}, marker={
            "size": sizes, "sizemode": "area", "sizeref": 2.0 * max_size / (62.0 ** 2), "sizemin": 10,
            "color": COUNTRY_COLORS[:len(data)], "opacity": 0.78, "line": {"color": "#e2e8f0", "width": 1.1},
        }, customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]} (%{text})</b><br>TFP Level: %{x:.2f}<br>TFP Growth 10Y: %{y:+.2f}%"
            "<br>Nominal GDP: $%{customdata[1]:,.0f}<br>GDP / Capital: %{customdata[2]:.2f}%"
            "<br>GDP / Capital Change 10Y: %{customdata[3]:+.2f}%"
            "<br>Services / GDP: %{customdata[4]:.2f}% (year %{customdata[5]:.0f})"
            "<br>GDP per Hour: $%{customdata[6]:,.2f}<br>GDP per Hour Growth 10Y: %{customdata[7]:+.2f}%"
            f"<br>PWT year: {reference_year}<br>Source: PWT {PWT_VERSION} + World Bank WDI<extra></extra>"
        ),
    ))
    fig.add_shape(type="rect", x0=x_median, x1=x1, y0=y_median, y1=y1, fillcolor="rgba(74,222,128,0.12)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=x_median, y0=y0, y1=y_median, fillcolor="rgba(245,158,11,0.10)", line_width=0, layer="below")
    fig.add_vline(x=x_median, line={"color": "#94a3b8", "dash": "dash"})
    fig.add_hline(y=y_median, line={"color": "#94a3b8", "dash": "dash"})
    for x, y, label, anchor in (
        (x0 + x_pad * 0.2, y1 - y_pad * 0.2, "Improving, low level", "left"),
        (x1 - x_pad * 0.2, y1 - y_pad * 0.2, "High & improving", "right"),
        (x0 + x_pad * 0.2, y0 + y_pad * 0.2, "Low & declining", "left"),
        (x1 - x_pad * 0.2, y0 + y_pad * 0.2, "High, but declining", "right"),
    ):
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, xanchor=anchor, font={"size": 13})
    fig.update_layout(
        height=650, margin={"l": 70, "r": 35, "t": 25, "b": 60}, showlegend=False, hovermode="closest",
        xaxis={"title": "TFP Level →", "range": [x0, x1], "gridcolor": "rgba(148,163,184,0.16)"},
        yaxis={"title": "TFP Growth 10Y ↑", "range": [y0, y1], "ticksuffix": "%", "gridcolor": "rgba(148,163,184,0.16)"},
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    st.caption(f"Dashed lines are Top 10 medians. Bubble area represents {size_mode}; quadrant labels are relative to those medians.")


def _render_historical_charts(history: pd.DataFrame) -> None:
    country = str(history["country"].iloc[-1])
    iso3 = str(history["iso3"].iloc[-1])
    start_year, end_year = int(history["year"].min()), int(history["year"].max())
    st.subheader(f"{country} — Economic Efficiency, {start_year}–{end_year}")

    left, right = st.columns(2, gap="medium")
    with left:
        tfp_mode = st.segmented_control(
            "TFP view", ["Domestic TFP Index", "Relative TFP Level"],
            default="Domestic TFP Index", key="efficiency_history_tfp_mode",
        ) or "Domestic TFP Index"
        if tfp_mode == "Domestic TFP Index":
            valid = history["rtfpna"].dropna()
            base = float(valid.iloc[0]) if not valid.empty else np.nan
            values = history["rtfpna"] / base * 100.0
            y_title = f"Domestic TFP index ({start_year} = 100)"
        else:
            values = history["tfp_level"]
            y_title = "TFP level (USA = 100)"
        _plot_history_line(history, values, "TFP Evolution — 30Y", y_title, "PWT: rtfpna" if tfp_mode.startswith("Domestic") else "PWT: ctfp × 100")
    with right:
        _plot_history_line(
            history, history["tfp_growth_10y_pct"], "TFP Growth 10Y — Historical", "10Y growth (%)",
            "PWT: rtfpna[t] / rtfpna[t-10] − 1", zero_line=True, signed_fill=True,
        )

    left, right = st.columns(2, gap="medium")
    with left:
        controls = st.columns(2)
        with controls[0]:
            capital_mode = st.segmented_control(
                "Capital metric", ["GDP / Capital", "Capital / GDP"],
                default="GDP / Capital", key="efficiency_history_capital_mode",
            ) or "GDP / Capital"
        with controls[1]:
            services_overlay = st.toggle("Services/GDP — right axis", key="efficiency_history_services")
        _plot_historical_capital(history, capital_mode, services_overlay)
    with right:
        _plot_history_line(
            history, history["capital_productivity_change_10y_pct"],
            "Capital Productivity Change 10Y — Historical", "10Y change (%)",
            "PWT: change in rgdpna / rnna", zero_line=True, signed_fill=True,
        )

    left, right = st.columns(2, gap="medium")
    with left:
        labor_mode = st.segmented_control(
            "Labor productivity", ["GDP per Hour — Level", "GDP per Hour — 10Y Growth"],
            default="GDP per Hour — Level", key="efficiency_history_labor_mode",
        ) or "GDP per Hour — Level"
        labor_values = history["gdp_per_hour_real"] if labor_mode.endswith("Level") else history["gdp_per_hour_change_10y_pct"]
        labor_axis = "Constant-price USD/hour" if labor_mode.endswith("Level") else "10Y growth (%)"
        _plot_history_line(
            history, labor_values, "GDP per Hour Worked — Historical", labor_axis,
            "PWT: rgdpna / (emp × avh)", zero_line=not labor_mode.endswith("Level"),
            signed_fill=not labor_mode.endswith("Level"),
        )
    with right:
        marker_mode = st.segmented_control(
            "Trajectory marker size", ["Constant", "Nominal GDP"], default="Constant",
            key="efficiency_trajectory_size",
        ) or "Constant"
        _plot_historical_trajectory(history, marker_mode)

    st.subheader("Economic Structure — Services Share of GDP")
    structural = history.dropna(subset=["services_gdp_pct"])
    if structural.empty:
        st.info("No Services/GDP observations are available for this historical window.")
    else:
        _plot_history_line(
            structural, structural["services_gdp_pct"], "Services Share of GDP — 30Y", "Services / GDP (%)",
            "World Bank WDI: NV.SRV.TOTL.ZS",
        )
    st.info(
        "A high capital-to-output ratio does not necessarily indicate inefficient capital allocation. "
        "It can also reflect industry composition, infrastructure intensity, natural resources, housing "
        "and the maturity of the capital stock. A higher Services/GDP share is not proof of lower capital requirements."
    )
    st.caption(f"Selected country: {country} ({iso3}). Missing years are not interpolated; rolling changes require exact 10-year endpoints.")


def _plot_history_line(
    history: pd.DataFrame,
    values: pd.Series,
    title: str,
    y_title: str,
    source: str,
    *,
    zero_line: bool = False,
    signed_fill: bool = False,
) -> None:
    data = pd.DataFrame({"year": history["year"], "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if data.empty:
        st.info(f"No valid observations for {title}.")
        return
    fig = go.Figure()
    if signed_fill:
        fig.add_trace(go.Scatter(
            x=data["year"], y=data["value"].clip(lower=0), mode="lines", line={"width": 0},
            fill="tozeroy", fillcolor="rgba(34,197,94,0.15)", hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=data["year"], y=data["value"].clip(upper=0), mode="lines", line={"width": 0},
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)", hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=data["year"], y=data["value"], mode="lines+markers", line={"color": "#4f8fe8", "width": 2.5},
        marker={"size": 5}, customdata=np.full((len(data), 1), source),
        hovertemplate="Year: %{x}<br>Value: %{y:,.2f}<br>Source/formula: %{customdata[0]}<extra></extra>",
        showlegend=False,
    ))
    if zero_line:
        fig.add_hline(y=0, line={"color": "#94a3b8", "width": 1.1})
    fig.update_layout(
        title={"text": title, "x": 0.5}, height=430, margin={"l": 60, "r": 25, "t": 55, "b": 50},
        hovermode="closest", xaxis={"title": "Year", "dtick": 5},
        yaxis={"title": y_title, "gridcolor": "rgba(148,163,184,0.20)"},
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _plot_historical_capital(history: pd.DataFrame, mode: str, show_services: bool) -> None:
    ratio = history["capital_productivity_real_pct"]
    values = ratio if mode == "GDP / Capital" else 100.0 / ratio.where(ratio.gt(0))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=history["year"], y=values, mode="lines+markers", name=mode,
        line={"color": "#4f8fe8", "width": 2.5}, marker={"size": 5},
        hovertemplate=f"Year: %{{x}}<br>{mode}: %{{y:,.2f}}{'%' if mode == 'GDP / Capital' else 'x'}<br>Source: PWT rgdpna / rnna<extra></extra>",
    ), secondary_y=False)
    if show_services:
        structural = history.dropna(subset=["services_gdp_pct"])
        fig.add_trace(go.Scatter(
            x=structural["year"], y=structural["services_gdp_pct"], mode="lines+markers", name="Services / GDP",
            line={"color": "#f2c14e", "width": 1.8, "dash": "dot"}, marker={"size": 4},
            hovertemplate="Year: %{x}<br>Services / GDP: %{y:.2f}%<br>Source: WDI NV.SRV.TOTL.ZS<extra></extra>",
        ), secondary_y=True)
    fig.update_layout(
        title={"text": "Capital Productivity — Historical", "x": 0.5}, height=430,
        margin={"l": 60, "r": 60, "t": 55, "b": 50}, hovermode="closest",
        legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
    )
    fig.update_xaxes(title_text="Year", dtick=5)
    fig.update_yaxes(title_text="GDP / Capital (%)" if mode == "GDP / Capital" else "Capital / GDP (x)", secondary_y=False)
    fig.update_yaxes(title_text="Services / GDP (%)", showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _plot_historical_trajectory(history: pd.DataFrame, marker_mode: str) -> None:
    data = history.dropna(subset=["tfp_level", "tfp_growth_10y_pct"]).copy()
    if data.empty:
        st.info("No complete observations are available for the historical efficiency trajectory.")
        return
    if marker_mode == "Nominal GDP":
        sizes = pd.to_numeric(data["nominal_gdp_usd"], errors="coerce").fillna(0).clip(lower=0)
        maximum = max(float(sizes.max()), 1.0)
        marker_sizes: Any = sizes
        sizeref = 2.0 * maximum / (34.0 ** 2)
    else:
        marker_sizes = 11
        sizeref = 1
    opacity = np.linspace(0.25, 1.0, len(data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["tfp_level"], y=data["tfp_growth_10y_pct"], mode="lines",
        line={"color": "rgba(79,143,232,0.45)", "width": 2}, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=data["tfp_level"], y=data["tfp_growth_10y_pct"], mode="markers",
        marker={"size": marker_sizes, "sizemode": "area", "sizeref": sizeref, "sizemin": 6,
                "color": data["year"], "colorscale": "Blues", "opacity": opacity, "showscale": True,
                "colorbar": {"title": "Year"}},
        customdata=data[["year", "nominal_gdp_usd"]],
        hovertemplate="Year: %{customdata[0]}<br>TFP Level: %{x:.2f}<br>TFP Growth 10Y: %{y:+.2f}%<br>Nominal GDP: $%{customdata[1]:,.0f}<extra></extra>",
        showlegend=False,
    ))
    first, latest = data.iloc[0], data.iloc[-1]
    for row, label in ((first, f"Start {int(first['year'])}"), (latest, f"Latest {int(latest['year'])}")):
        fig.add_annotation(x=row["tfp_level"], y=row["tfp_growth_10y_pct"], text=label, showarrow=True, arrowhead=2)
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot"})
    fig.update_layout(
        title={"text": "Economic Efficiency Trajectory — 30Y", "x": 0.5}, height=430,
        margin={"l": 60, "r": 45, "t": 55, "b": 50}, hovermode="closest",
        xaxis={"title": "TFP Level, USA = 100"}, yaxis={"title": "TFP Growth 10Y (%)", "ticksuffix": "%"},
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def _fmt(value: Any, pattern: str) -> str:
    return "N/A" if pd.isna(value) else pattern.format(float(value))


def _fmt_pct(value: Any) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):+.1f}%".replace("-", "−")


def _change_color(value: Any) -> str:
    if pd.isna(value):
        return "color: #94a3b8"
    return "color: #48c78e" if float(value) >= 0 else "color: #ff6b6b"

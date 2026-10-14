from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

from world_gdp import WorldGDPService, load_world_gdp_data


MAJOR_COUNTRY_COLORS = {
    "USA": "#1f77b4",
    "CHN": "#d62728",
    "JPN": "#9467bd",
    "DEU": "#ff7f0e",
    "GBR": "#2ca02c",
    "IND": "#c5b0d5",
    "FRA": "#e377c2",
    "ITA": "#7f7f7f",
    "RUS": "#8c564b",
    "CAN": "#17becf",
    "BRA": "#ffbb78",
    "KOR": "#c49c94",
    "ESP": "#ff9896",
    "MEX": "#f7b6d2",
    "AUS": "#bcbd22",
    "ARG": "#c7c7c7",
    "NLD": "#9edae5",
    "WLD": "#f8fafc",
    "OTHER": "#cbd5e1",
}
FALLBACK_COLORS = [*qualitative.Safe, *qualitative.Set3, *qualitative.Dark24]


@st.cache_data(show_spinner="Loading World Bank GDP data…", ttl=30 * 24 * 60 * 60)
def load_world_gdp_snapshot(refresh_nonce: int = 0) -> dict[str, Any]:
    return load_world_gdp_data(force_refresh=refresh_nonce > 0)


def render_world_gdp_module(snapshot: dict[str, Any]) -> None:
    st.header("World GDP")
    warning = snapshot.get("warning")
    if warning:
        st.warning(warning)
    frame = snapshot.get("data", pd.DataFrame()).copy()
    metadata = snapshot.get("metadata", {})
    if frame.empty:
        st.info("World Bank GDP data is not available yet. Use Refresh World Economy to try again.")
        return

    reference_year = int(metadata.get("reference_year") or WorldGDPService.latest_common_year(frame))
    table = WorldGDPService.build_top10_table(frame, reference_year=reference_year)
    _render_top10_table(table, reference_year)
    st.divider()
    _render_world_gdp_history(frame, reference_year)

    coverage = metadata.get("coverage", {})
    retrieved = metadata.get("retrieved_at_utc", "unknown")
    st.caption(
        f"Source: World Bank World Development Indicators · Common validated year: {reference_year} · "
        f"Coverage: nominal {coverage.get('nominal_gdp_usd', 'N/A')}, "
        f"PPP {coverage.get('gdp_ppp_intl_usd', 'N/A')}, "
        f"per capita {coverage.get('gdp_per_capita_usd', 'N/A')} economies · Retrieved: {retrieved}"
    )


def _render_top10_table(table: pd.DataFrame, reference_year: int) -> None:
    st.subheader("Top 10 Economies by Nominal GDP")
    st.caption(f"{reference_year} values; cumulative 10-year changes compare {reference_year} with {reference_year - 10}.")
    display = table.drop(columns="iso3").rename(columns={
        "rank": "Rank",
        "country_name": "Country",
        "nominal_gdp_usd_trn": "Nominal GDP, $T",
        "nominal_gdp_usd_change_10y_pct": "Nominal GDP 10Y Change",
        "gdp_ppp_intl_usd_trn": "GDP PPP, Intl$T",
        "gdp_ppp_intl_usd_change_10y_pct": "GDP PPP 10Y Change",
        "gdp_per_capita_usd": "GDP per capita, $",
        "gdp_per_capita_usd_change_10y_pct": "GDP per capita 10Y Change",
    })
    change_columns = [column for column in display if "Change" in column]
    styler = display.style.format({
        "Nominal GDP, $T": lambda value: _format_number(value, "${:,.2f}T"),
        "GDP PPP, Intl$T": lambda value: _format_number(value, "Intl${:,.2f}T"),
        "GDP per capita, $": lambda value: _format_number(value, "${:,.0f}"),
        **{column: _format_change for column in change_columns},
    }, na_rep="N/A")
    styler = styler.applymap(_change_color, subset=change_columns)
    event = st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=425,
        on_select="rerun",
        selection_mode="single-row",
        key="world_gdp_top10_table",
    )
    rows = _selected_rows(event)
    row_number = int(rows[0]) if rows else -1
    selected_iso3 = str(table.iloc[row_number]["iso3"]) if 0 <= row_number < len(table) else None
    previous_table_selection = st.session_state.get("world_gdp_last_table_selection")
    if selected_iso3 != previous_table_selection:
        st.session_state["world_gdp_last_table_selection"] = selected_iso3
        if selected_iso3 is not None and st.session_state.get("world_gdp_country_focus") != selected_iso3:
            st.session_state["world_gdp_country_focus"] = selected_iso3
            st.rerun()


def _selected_rows(event: Any) -> list[int]:
    if hasattr(event, "selection"):
        selection = event.selection
        if hasattr(selection, "rows"):
            return list(selection.rows)
        if isinstance(selection, dict):
            return list(selection.get("rows", []))
    if isinstance(event, dict):
        return list(event.get("selection", {}).get("rows", []))
    return []


def _format_number(value: Any, pattern: str) -> str:
    return "N/A" if pd.isna(value) else pattern.format(float(value))


def _format_change(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):+.1f}%".replace("-", "−")


def _change_color(value: Any) -> str:
    if pd.isna(value):
        return "color: #94a3b8"
    return "color: #6fbf8f" if float(value) >= 0 else "color: #e88888"


def _render_world_gdp_history(frame: pd.DataFrame, reference_year: int) -> None:
    st.subheader("WORLD GDP | Historical Dynamics")
    metric_col, mode_col, focus_col = st.columns([1.1, 0.9, 1.6])
    with metric_col:
        metric_label = st.segmented_control(
            "GDP metric",
            ["Nominal GDP", "GDP PPP", "GDP per capita"],
            default="Nominal GDP",
            key="world_gdp_metric",
        )
    with mode_col:
        display_mode = st.segmented_control(
            "Display mode", ["Absolute", "Share"], default="Share", key="world_gdp_display_mode"
        )
    with focus_col:
        selected_iso3 = _country_focus_selector(frame, reference_year)

    metric_label = metric_label or "Nominal GDP"
    display_mode = display_mode or "Share"
    if metric_label in {"Nominal GDP", "GDP PPP"}:
        st.caption("Top 10 economies recomputed independently for each year; all others grouped in Other.")
        metric = "nominal" if metric_label == "Nominal GDP" else "ppp"
        history = WorldGDPService.build_additive_history(
            frame, metric=metric, reference_year=reference_year, selected_iso3=selected_iso3
        )
        _render_additive_chart(history, metric=metric, display_mode=display_mode, selected_iso3=selected_iso3)
    else:
        history = WorldGDPService.build_per_capita_history(
            frame, reference_year=reference_year, selected_iso3=selected_iso3
        )
        _render_per_capita_chart(history, display_mode=display_mode, selected_iso3=selected_iso3)


def _country_focus_selector(frame: pd.DataFrame, reference_year: int) -> str | None:
    countries = (
        frame.loc[frame["is_country"] & frame["year"].le(reference_year), ["iso3", "country_name"]]
        .drop_duplicates("iso3", keep="last")
        .sort_values("country_name")
    )
    names = dict(zip(countries["iso3"], countries["country_name"]))
    options: list[str | None] = [None, *countries["iso3"].tolist()]
    key = "world_gdp_country_focus"
    if st.session_state.get(key) not in options:
        st.session_state[key] = None
    return st.selectbox(
        "Country Focus",
        options,
        format_func=lambda code: "All Countries" if code is None else f"{names.get(code, code)} ({code})",
        key=key,
    )


def _render_additive_chart(
    history: pd.DataFrame, *, metric: str, display_mode: str, selected_iso3: str | None
) -> None:
    share_mode = display_mode == "Share"
    title_metric = "NOMINAL GDP" if metric == "nominal" else "GDP PPP"
    value_label = "Current US dollars" if metric == "nominal" else "Current international dollars"
    title = f"WORLD {title_metric} | Changing share of the 10 largest economies"
    years = sorted(history["year"].unique())
    latest_year = int(max(years))
    latest = history.loc[history["year"].eq(latest_year)]
    _additive_latest_summary(latest, metric=metric, latest_year=latest_year)

    codes = [code for code in history["iso3"].drop_duplicates() if code != "OTHER"] + ["OTHER"]
    fig = go.Figure()
    for code in codes:
        subset = history.loc[history["iso3"].eq(code)].sort_values("year")
        if subset.empty:
            continue
        name = str(subset["country_name"].iloc[-1])
        y = subset["share_pct"] if share_mode else subset["value"] / 1e12
        ranks = subset["rank"].map(lambda value: "N/A" if pd.isna(value) else str(int(value)))
        notes = subset["selected_outside_top10"].map(
            lambda value: "Selected separately; outside annual top 10" if value else "Annual top 10" if code != "OTHER" else "Residual to WLD total"
        )
        custom = np.column_stack((subset["value"] / 1e12, subset["share_pct"], ranks, notes))
        opacity = _trace_opacity(code, selected_iso3)
        fig.add_trace(go.Bar(
            x=subset["year"],
            y=y,
            name=name,
            marker={"color": _country_color(code), "line": {"color": "rgba(255,255,255,0.55)", "width": 0.45}},
            opacity=opacity,
            customdata=custom,
            hovertemplate=(
                f"<b>{name}</b><br>Year: %{{x}}<br>{title_metric}: %{{customdata[0]:.3f}}T"
                "<br>World share: %{customdata[1]:.2f}%<br>Annual rank: %{customdata[2]}"
                "<br>%{customdata[3]}<br>Source: World Bank WDI<extra></extra>"
            ),
        ))
    subtitle = f"{min(years)}–{latest_year} · {value_label} · Top 10 recomputed independently each year"
    fig.update_layout(
        title={"text": f"<b>{title}</b><br><sup>{subtitle}</sup>", "x": 0.01, "xanchor": "left"},
        barmode="stack",
        height=720,
        margin={"l": 65, "r": 20, "t": 85, "b": 145},
        bargap=0.05,
        legend={"orientation": "h", "yanchor": "top", "y": -0.17, "x": 0.5, "xanchor": "center"},
        xaxis={"title": "Year", "dtick": 5, "tick0": min(years), "showgrid": False},
        yaxis={
            "title": f"Share of World {title_metric.title()} (%)" if share_mode else f"{title_metric.title()} (trillions)",
            "range": [0, 100] if share_mode else None,
            "ticksuffix": "%" if share_mode else "",
            "gridcolor": "rgba(148,163,184,0.20)",
        },
        hovermode="x unified" if selected_iso3 is None else "closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _additive_latest_summary(latest: pd.DataFrame, *, metric: str, latest_year: int) -> None:
    other = _share_for(latest, "OTHER")
    if metric == "nominal":
        first = f"United States {_share_for(latest, 'USA'):.1f}%"
        second = f"China {_share_for(latest, 'CHN'):.1f}%"
    else:
        leaders = latest.loc[latest["iso3"].ne("OTHER")].nlargest(2, "value")
        labels = [f"{row.country_name} {row.share_pct:.1f}%" for row in leaders.itertuples(index=False)]
        first, second = (labels + ["N/A", "N/A"])[:2]
    st.markdown(f"**{latest_year}: {first}  |  {second}  |  Other {other:.1f}%**")


def _share_for(frame: pd.DataFrame, iso3: str) -> float:
    values = frame.loc[frame["iso3"].eq(iso3), "share_pct"]
    return float(values.iloc[0]) if not values.empty else float("nan")


def _render_per_capita_chart(
    history: pd.DataFrame, *, display_mode: str, selected_iso3: str | None
) -> None:
    relative = display_mode == "Share"
    y_column = "relative_world_pct" if relative else "gdp_per_capita_usd"
    title = "GDP per Capita Relative to World Average" if relative else "WORLD GDP PER CAPITA | Historical Dynamics"
    _per_capita_latest_summary(history, selected_iso3)
    fig = go.Figure()
    codes = [code for code in history["iso3"].drop_duplicates() if code != "WLD"] + ["WLD"]
    for code in codes:
        subset = history.loc[history["iso3"].eq(code)].sort_values("year")
        if subset.empty:
            continue
        name = str(subset["country_name"].iloc[-1])
        is_world = code == "WLD"
        highlighted = selected_iso3 is None or code == selected_iso3 or is_world
        custom = np.column_stack((subset["gdp_per_capita_usd"], subset["relative_world_pct"]))
        fig.add_trace(go.Scatter(
            x=subset["year"],
            y=subset[y_column],
            mode="lines",
            name=name,
            line={
                "color": _country_color(code),
                "width": 3.2 if code == selected_iso3 else 2.4 if is_world else 1.8,
                "dash": "dash" if is_world else "solid",
            },
            opacity=1.0 if highlighted else 0.42,
            customdata=custom,
            hovertemplate=(
                f"<b>{name}</b><br>Year: %{{x}}<br>GDP per capita: $%{{customdata[0]:,.0f}}"
                "<br>World average index: %{customdata[1]:.1f}%<br>Source: World Bank WDI<extra></extra>"
            ),
        ))
    if relative:
        fig.add_hline(y=100, line={"color": "#94a3b8", "width": 1, "dash": "dot"})
    min_year = int(history["year"].min())
    max_year = int(history["year"].max())
    fig.update_layout(
        title={"text": f"<b>{title}</b><br><sup>{min_year}–{max_year} · Current US dollars per person</sup>", "x": 0.01},
        height=620,
        margin={"l": 70, "r": 20, "t": 85, "b": 110},
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "x": 0.5, "xanchor": "center"},
        xaxis={"title": "Year", "dtick": 5, "tick0": min_year, "showgrid": False},
        yaxis={
            "title": "World average = 100%" if relative else "GDP per capita (current USD)",
            "ticksuffix": "%" if relative else "",
            "tickprefix": "" if relative else "$",
            "gridcolor": "rgba(148,163,184,0.20)",
        },
        hovermode="x unified" if selected_iso3 is None else "closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _per_capita_latest_summary(history: pd.DataFrame, selected_iso3: str | None) -> None:
    code = selected_iso3 or "WLD"
    subset = history.loc[history["iso3"].eq(code)].dropna(subset=["gdp_per_capita_usd"])
    if subset.empty:
        return
    latest = subset.sort_values("year").iloc[-1]
    label = latest["country_name"]
    st.markdown(
        f"**Latest year: {int(latest['year'])}  |  {label}: ${latest['gdp_per_capita_usd']:,.0f}  |  "
        f"World average index: {latest['relative_world_pct']:.1f}%**"
    )


def _trace_opacity(code: str, selected_iso3: str | None) -> float:
    if selected_iso3 is None:
        return 0.96
    if code == selected_iso3:
        return 1.0
    if code == "OTHER":
        return 0.42
    return 0.28


def _country_color(iso3: str) -> str:
    if iso3 in MAJOR_COUNTRY_COLORS:
        return MAJOR_COUNTRY_COLORS[iso3]
    digest = hashlib.sha256(iso3.encode("utf-8")).digest()
    return FALLBACK_COLORS[int.from_bytes(digest[:2], "big") % len(FALLBACK_COLORS)]

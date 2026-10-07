from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from positioning import (
    CFTC_3Y_PERCENTILE_MIN_PERIODS,
    CFTC_5Y_PERCENTILE_MIN_PERIODS,
    CFTC_STALE_DAYS,
    CFTC_DASHBOARD_LAYOUT,
    cftc_asset_config,
    cftc_asset_series,
    cftc_categories_for_report,
    cftc_contract_status,
    cftc_latest_status,
    export_positioning_xlsx,
    load_positioning_data,
)


CFTC_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
CFTC_RANGE_OPTIONS = ("3Y", "5Y", "10Y", "MAX")


@st.cache_data(show_spinner=True, ttl=21600)
def load_positioning_snapshot(cache_key: str, force_update: bool = False) -> dict[str, Any]:
    _ = cache_key
    return load_positioning_data(force_update=force_update)


def render_cftc_cot_tab() -> None:
    st.subheader("CFTC COT")
    force_update = st.button("Refresh Positioning Data", key="refresh_positioning_data")
    if force_update:
        load_positioning_snapshot.clear()
    with st.spinner("Loading positioning data..."):
        data = load_positioning_snapshot("positioning", force_update=force_update)
    master = data.get("cftc_master", pd.DataFrame())
    aaii = data.get("aaii", pd.DataFrame())
    naaim = data.get("naaim", pd.DataFrame())
    status = data.get("status", {})

    if status.get("CFTC Refresh", {}).get("status") == "ALREADY_RUNNING":
        st.info("Positioning data refresh is already running; showing the last saved data.")
    render_source_status(master, status)
    range_choice = st.radio("Time range", CFTC_RANGE_OPTIONS, index=1, horizontal=True, key="cftc_cot_range")
    if master.empty:
        st.warning("CFTC positioning data is unavailable.")
        return

    export_col, download_col = st.columns([1, 3])
    with export_col:
        if st.button("Prepare Positioning Export", key="prepare_positioning_export", use_container_width=True):
            with st.spinner("Preparing positioning_data.xlsx..."):
                st.session_state["positioning_export_xlsx"] = export_positioning_xlsx(master, aaii, naaim, status)
    if "positioning_export_xlsx" in st.session_state:
        with download_col:
            st.download_button(
                "Download Positioning Data",
                data=st.session_state["positioning_export_xlsx"],
                file_name="positioning_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=False,
                key="export_positioning_data",
            )

    for row_index, row_assets in enumerate(CFTC_DASHBOARD_LAYOUT, start=1):
        st.markdown(f"#### Row {row_index}")
        cols = st.columns(len(row_assets))
        for col, asset in zip(cols, row_assets):
            with col:
                cfg = cftc_asset_config(asset)
                source_key = "CFTC Commodities" if cfg and cfg.report_type == "Disaggregated" else "CFTC Financials"
                source_error = str(status.get(source_key, {}).get("status", "")).startswith("SOURCE_FAILED")
                render_cftc_asset_card(master, asset, range_choice, source_error=source_error)


def render_source_status(master: pd.DataFrame, status: dict[str, Any]) -> None:
    commodities = cftc_latest_status(master, "Disaggregated")
    financials = cftc_latest_status(master, "TFF")
    cols = st.columns(4)
    items = [
        ("CFTC Commodities", commodities.get("last_report_date") or "n/a", cftc_source_detail(status, "CFTC Commodities", commodities.get("status", "DATA UNAVAILABLE"))),
        ("CFTC Financials", financials.get("last_report_date") or "n/a", cftc_source_detail(status, "CFTC Financials", financials.get("status", "DATA UNAVAILABLE"))),
        ("AAII", source_timestamp(status, "AAII"), source_state(status, "AAII")),
        ("NAAIM", source_timestamp(status, "NAAIM"), source_state(status, "NAAIM")),
    ]
    for col, (label, value, detail) in zip(cols, items):
        with col:
            st.markdown(
                f"""
<div style="padding:0.35rem 0 0.75rem 0; line-height:1.15;">
  <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;">{html.escape(label)}</div>
  <div style="font-size:0.95rem; color:#f8fafc; font-weight:800;">{html.escape(str(value))}</div>
  <div style="font-size:0.72rem; color:#cbd5e1;">{html.escape(str(detail))}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_cftc_asset_card(master: pd.DataFrame, asset: str, range_choice: str, *, source_error: bool = False) -> None:
    cfg = cftc_asset_config(asset)
    if cfg is None:
        render_unavailable_card(asset, "No dashboard config")
        return
    categories = cftc_categories_for_report(cfg.report_type)
    default_index = categories.index(cfg.default_participant) if cfg.default_participant in categories else 0
    participant = st.selectbox(
        asset,
        categories,
        index=default_index,
        key=f"cftc_participant_{asset}",
        label_visibility="visible",
    )
    series = cftc_asset_series(master, asset, participant)
    if series.empty:
        render_unavailable_card(asset, f"{participant} data unavailable")
        return
    d = filter_range(series, range_choice)
    latest = series.dropna(subset=["Date"]).tail(1).iloc[0]
    render_card_metrics(latest, cftc_contract_status(master, asset, source_error=source_error))
    fig = build_cftc_chart(d, asset, participant)
    st.plotly_chart(fig, use_container_width=True, config=CFTC_PLOTLY_CONFIG)


def render_card_metrics(row: pd.Series, contract_status: dict[str, Any] | None = None) -> None:
    contract_status = contract_status or {"status": "CURRENT", "latest_report_date": None, "present_in_latest_report": True}
    current = contract_status.get("status") == "CURRENT"
    history = safe_float(row.get("History_Weeks"))
    three_year_status = "READY" if np.isfinite(history) and history >= CFTC_3Y_PERCENTILE_MIN_PERIODS else "N/A (<156W)"
    five_year_status = "READY" if np.isfinite(history) and history >= CFTC_5Y_PERCENTILE_MIN_PERIODS else "N/A (<260W)"
    updated = fmt_date(row.get("Date"))
    date_value = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.isna(date_value):
        freshness = "DATE UNKNOWN"
    else:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        freshness = "STALE" if (today - date_value.normalize()).days > CFTC_STALE_DAYS else "CURRENT"
    metrics = [
        ("Net % OI", fmt_signed_pct_points(row.get("NetPctOI")) if current else "N/A"),
        ("3Y Percentile", f"{fmt_score(row.get('NetPctOI_3Y_Percentile'))} ({three_year_status})" if current else "N/A"),
        ("5Y Percentile", f"{fmt_score(row.get('NetPctOI_5Y_Percentile'))} ({five_year_status})" if current else "N/A"),
        ("4W", fmt_signed_pp(row.get("NetPctOI_4W_Change")) if current else "N/A"),
        ("13W", fmt_signed_pp(row.get("NetPctOI_13W_Change")) if current else "N/A"),
        ("CFTC Status", str(contract_status.get("status", "N/A"))),
        ("Latest official", str(contract_status.get("latest_report_date") or "N/A")),
        ("Last available", f"{updated} ({freshness})"),
    ]
    text = " | ".join(f"{label}: {value}" for label, value in metrics)
    st.markdown(f"<div style='font-size:0.70rem;color:#cbd5e1;line-height:1.25;margin-bottom:0.35rem;'>{html.escape(text)}</div>", unsafe_allow_html=True)


def render_unavailable_card(asset: str, reason: str) -> None:
    st.markdown(f"**{html.escape(asset)}**")
    st.warning(f"DATA UNAVAILABLE: {reason}")


def build_cftc_chart(data: pd.DataFrame, asset: str, participant: str) -> go.Figure:
    d = data.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["NetPctOI"] = pd.to_numeric(d["NetPctOI"], errors="coerce")
    d["NetPctOI_3Y_Percentile"] = pd.to_numeric(d["NetPctOI_3Y_Percentile"], errors="coerce")
    if "NetPctOI_5Y_Percentile" not in d:
        d["NetPctOI_5Y_Percentile"] = np.nan
    d["NetPctOI_5Y_Percentile"] = pd.to_numeric(d["NetPctOI_5Y_Percentile"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["NetPctOI_5Y_Percentile"],
            mode="lines",
            name="5Y Percentile",
            yaxis="y2",
            line={"color": "#c084fc", "width": 1.5, "dash": "dot"},
            hovertemplate="Week: %{x|%Y-%m-%d}<br>5Y Percentile: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["NetPctOI"],
            mode="lines",
            name="Net % OI",
            line={"color": "#60a5fa", "width": 1.6},
            hovertemplate="Week: %{x|%Y-%m-%d}<br>Net % OI: %{y:+.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["NetPctOI_3Y_Percentile"],
            mode="lines",
            name="3Y Percentile",
            yaxis="y2",
            line={"color": "#facc15", "width": 1.6, "dash": "dash"},
            hovertemplate="Week: %{x|%Y-%m-%d}<br>3Y Percentile: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1}, opacity=0.55)
    fig.update_layout(
        title={"text": f"{asset} COT — {participant} Net % OI and Trailing 3Y / 5Y Percentiles", "font": {"size": 12}},
        template="plotly_dark",
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#f8fafc", "size": 10},
        height=285,
        margin={"l": 38, "r": 38, "t": 52, "b": 42},
        legend={"orientation": "h", "y": -0.22, "x": 0.0},
        yaxis={"title": "Net % OI", "gridcolor": "#263241", "zeroline": False},
        yaxis2={"title": "Percentile", "overlaying": "y", "side": "right", "range": [0, 100], "gridcolor": "#263241"},
        xaxis={"gridcolor": "#1f2937"},
    )
    return fig


def filter_range(frame: pd.DataFrame, range_choice: str) -> pd.DataFrame:
    d = frame.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    if d.empty or range_choice == "MAX":
        return d
    years = {"3Y": 3, "5Y": 5, "10Y": 10}.get(range_choice, 5)
    cutoff = d["Date"].max() - pd.DateOffset(years=years)
    return d.loc[d["Date"] >= cutoff].copy()


def source_timestamp(status: dict[str, Any], key: str) -> str:
    value = status.get(key, {}).get("last_updated_utc") if isinstance(status, dict) else None
    if not value:
        return "n/a"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def source_state(status: dict[str, Any], key: str) -> str:
    if not isinstance(status, dict):
        return "n/a"
    return str(status.get(key, {}).get("status", "n/a"))


def cftc_source_detail(status: dict[str, Any], key: str, freshness: Any) -> str:
    source = status.get(key, {}) if isinstance(status, dict) else {}
    frequency = source.get("update_frequency", "Weekly")
    update_day = source.get("scheduled_update_day", "Saturday")
    return f"{freshness} | updates {frequency.lower()} on {update_day}"


def fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).date().isoformat()


def fmt_score(value: Any) -> str:
    numeric = safe_float(value)
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.0f}"


def fmt_signed_pct_points(value: Any) -> str:
    numeric = safe_float(value)
    return "n/a" if not np.isfinite(numeric) else f"{numeric:+.1f}%"


def fmt_signed_pp(value: Any) -> str:
    numeric = safe_float(value)
    return "n/a" if not np.isfinite(numeric) else f"{numeric:+.1f} pp"


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan

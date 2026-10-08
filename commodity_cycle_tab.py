from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from commodity_cycle.data import (
    FRED_SERIES,
    PRICE_TICKERS,
    build_commodity_cycle_history,
    build_market_confirmation,
    latest_complete_commodity_cycle_row,
    load_cftc_snapshot,
    load_fred_history,
    load_monthly_prices,
    load_tradingview_performance_prices,
    load_commodity_term_structure,
)
from commodity_cycle.model import CORE_SERIES, CONFIRMATION_SERIES
from commodity_cycle.export import commodity_tables_to_xlsx
from commodity_cycle.ppi_led_etf import (
    HORIZONS as PPI_LED_HORIZONS,
    build_performance_table as build_ppi_led_performance_table,
    download_market_prices as download_ppi_led_market_prices,
    relative_median_frame,
)

TTL_SECONDS = 21600
RANGE_OPTIONS = ("1Y", "3Y", "5Y", "10Y", "20Y", "Full")
MOMENTUM_WEEKS = {"1M": 4, "3M": 13, "6M": 26, "12M": 52}
CORE_COLORS = {
    "Neutral": "#e2e8f0", "Early Broadening": "#fde047", "Confirmed Broadening": "#fb923c",
    "Systemic Broadening": "#f43f5e", "Mature": "#c084fc", "Early Easing": "#4ade80",
    "Confirmed Easing": "#38bdf8", "DATA INCOMPLETE": "#64748b",
}
FINAL_COLORS = {
    "Low Inflation / Neutral": "#38bdf8", "Reflation / Early Inflation": "#fde047",
    "Inflation Expansion": "#f43f5e", "Late Cycle / Peak Risk": "#c084fc",
    "Disinflation Transition": "#4ade80", "Confirmed Disinflation": "#22d3ee",
    "Broad Inflation": "#f43f5e", "N/A": "#64748b",
}
STATE_BAND_OPACITY = 0.30
PRIMARY_COMMODITY_COLUMNS = (
    "Sector", "Commodity", "Price", "Return 1M", "Return 3M", "Return 6M", "Return 12M", "Price State",
    "Curve Spread", "Annualized Curve Spread", "Raw Curve State", "Seasonal Percentile 10Y", "Seasonal Relative State",
    "MM Net % OI", "4W Change", "13W Change", "5Y Percentile", "CFTC Relative State", "Price × Seasonal Curve",
    "Price Date", "Term Structure As Of",
)
PERCENT_COLUMNS = {
    "Return 1M", "Return 3M", "Return 6M", "Return 12M", "Curve Spread", "Annualized Curve Spread", "MTD Average Spread",
}
DISPLAY_COLUMN_NAMES = {
    "3Y Percentile": "COT 3Y Percentile",
    "5Y Percentile": "COT 5Y Percentile",
    "Seasonal Percentile 5Y": "Spread 5Y Seasonal Percentile",
    "Seasonal Percentile 10Y": "Spread 10Y Seasonal Percentile",
    "Price Date": "Price As Of",
    "Term Structure As Of": "Curve As Of",
}


@st.cache_data(show_spinner="Loading Commodity Cycle data…", ttl=TTL_SECONDS)
def load_commodity_cycle_snapshot(api_key: str | None, refresh_nonce: int = 0) -> dict[str, Any]:
    _ = refresh_nonce
    fred, fred_status = load_fred_history(api_key)
    prices, price_status = load_monthly_prices()
    performance_prices, performance_status, performance_source = load_tradingview_performance_prices(prices)
    term_history, term_current, term_diagnostics = load_commodity_term_structure()
    cftc, cftc_status = load_cftc_snapshot()
    history, capex, term_history_rt = build_commodity_cycle_history(fred, prices, term_history)
    ppiaco = fred.get("PPIACO", pd.Series(dtype=float)) if isinstance(fred, pd.DataFrame) else pd.Series(dtype=float)
    ppi_led_table = pd.DataFrame()
    ppi_led_as_of = None
    ppi_led_status: dict[str, str] = {}
    try:
        ppi_series = pd.to_numeric(ppiaco, errors="coerce").dropna()
        if not ppi_series.empty:
            ppi_led_prices, ppi_led_status = download_ppi_led_market_prices(pd.Timestamp(ppi_series.index.max()))
            ppi_led_table, ppi_led_as_of = build_ppi_led_performance_table(ppi_series, ppi_led_prices)
        else:
            ppi_led_status = {"PPIACO": "FRED PPIACO unavailable; common calculation date is not available"}
    except Exception as exc:
        ppi_led_status = {"Module": f"PPI-led ETF data unavailable: {exc}"}
    commodity, sector = build_market_confirmation(
        prices, term_history_rt, term_current, cftc, performance_prices=performance_prices,
        performance_sources=performance_source,
    )
    if not commodity.empty:
        commodity["Price Status"] = commodity["Commodity"].map(price_status)
        commodity["Performance Status"] = commodity["Commodity"].map(performance_status).fillna("MISSING")
        term_freshness: dict[str, str] = {}
        for _, row in term_current.iterrows():
            key = str(row["Asset"])
            status = str(row.get("Status", "MISSING"))
            # Market confirmation uses the preferred primary structure only.
            preferred = {"WTI": "F1/F3", "Natural Gas": "F1/F3", "RBOB": "F1/F3", "Corn": "Dec/Mar", "Wheat": "Dec/Mar", "Soybeans": "Nov/Jan", "Copper": "Cash/3M", "Aluminum": "Cash/3M"}.get(key)
            if row.get("Structure") == preferred or key not in term_freshness:
                term_freshness[key] = status
        commodity["Term Structure Status"] = commodity["Commodity"].map(term_freshness).fillna("MISSING")
    return {
        "fred": fred, "fred_status": fred_status, "prices": prices, "price_status": price_status,
        "performance_prices": performance_prices, "performance_status": performance_status,
        "performance_source": performance_source,
        "term_history": term_history_rt, "term_current": term_current, "term_diagnostics": term_diagnostics, "cftc": cftc,
        "cftc_status": cftc_status, "history": history, "capex": capex,
        "ppi_led_table": ppi_led_table, "ppi_led_as_of": ppi_led_as_of,
        "ppi_led_status": ppi_led_status,
        "commodity": commodity, "sector": sector,
    }


def render_commodity_cycle_tab(api_key: str | None) -> None:
    st.subheader("Commodity Cycle")
    left, middle, right = st.columns([1.15, 1.0, 7.85])
    with left:
        refresh = st.button("Refresh Commodity Cycle", key="commodity_cycle_refresh", use_container_width=True)
    with middle:
        st.caption("Refresh interval: 6 hours")
    if refresh:
        st.session_state["commodity_cycle_refresh_nonce"] = int(pd.Timestamp.now(tz="UTC").value)
        load_commodity_cycle_snapshot.clear()
        st.rerun()
    try:
        data = load_commodity_cycle_snapshot(api_key, int(st.session_state.get("commodity_cycle_refresh_nonce", 0)))
    except Exception as exc:
        st.error(f"Commodity Cycle data load failed: {exc}")
        return

    history = data["history"]
    current = latest_complete_commodity_cycle_row(history)
    capex = data["capex"]
    capex_current = capex.dropna(subset=["CAPEX Intensity"]).iloc[-1] if not capex.empty else pd.Series(dtype=object)
    commodity = data["commodity"]
    sectors = data["sector"]

    st.markdown(
        """
        <style>
        .st-key-commodity-cycle-summary [data-testid="stMetricValue"],
        .st-key-commodity-cycle-summary [data-testid="stMetricValue"] > div,
        .st-key-commodity-cycle-sector-dashboard [data-testid="stMetricValue"],
        .st-key-commodity-cycle-sector-dashboard [data-testid="stMetricValue"] > div {
            font-size: clamp(0.95rem, 1.15vw, 1.4rem) !important;
            line-height: 1.2 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    summary = [
        ("Final State 2", current.get("Final State 2", "N/A")),
        ("Core State", current.get("Core State", "N/A")),
        ("PPI Confirmation", current.get("PPI Confirmation", "N/A")),
        (
            "CAPEX Full-History Percentile",
            f"{_fmt(capex_current.get('CAPEX Current Percentile'), '%')} ({_fmt(capex_current.get('CAPEX 24M Change'), '%')})",
        ),
    ]
    with st.container(key="commodity-cycle-summary"):
        cards = st.columns(4)
        for idx, (label, value) in enumerate(summary):
            with cards[idx % 4]:
                st.metric(label, _format_state(value))
                if label == "PPI Confirmation":
                    st.caption(_ppi_sector_tightening_easing(current))

    _range_picker("commodity_cycle_overview_range")
    selected_range = st.session_state["commodity_cycle_overview_range"]
    if history.empty:
        st.info("The FRED Inventory/Sales model is unavailable. Check Diagnostics / Data below; market and positioning sections remain available where their feeds loaded.")
    else:
        _render_ppi_core_chart(history, selected_range)
        _render_ppi_cpi_roc_chart(history, selected_range)
        view = st.radio("Regime shading", ["Final State", "Final State 2"], index=1, horizontal=True,
                        key="commodity_cycle_final_view_v2")
        _render_final_state_chart(history, selected_range, view)
    if not history.empty:
        st.markdown("#### FRED Inventory / Sales Regime")
        _render_fred_heatmap(history)
        with st.expander("Core stress and breadth diagnostics", expanded=False):
            cols = [c for c in history if c.endswith("Rolling Stress") or c.endswith("Seasonal Stress")]
            st.dataframe(history[cols].tail(24).round(1), use_container_width=True)

    _render_market_section(
        commodity,
        sectors,
        data["performance_prices"],
        selected_range,
        capex,
        history,
        data["term_history"],
        data.get("term_diagnostics", pd.DataFrame()),
    )

    _render_ppi_led_etf(data.get("ppi_led_table", pd.DataFrame()), data.get("ppi_led_as_of"), data.get("ppi_led_status", {}))

    with st.expander("Diagnostics / Data", expanded=False):
        if not capex.empty:
            st.markdown("#### CAPEX Intensity history")
            history_view = capex[[
                "CAPEX Intensity", "CAPEX Expanding Percentile", "CAPEX Vulnerability RT",
                "CAPEX 24M Change", "CAPEX Direction",
            ]].tail(20).copy()
            st.dataframe(history_view.round(3), use_container_width=True)
        _render_diagnostics_section(data, commodity)


def _render_ppi_led_etf(table: pd.DataFrame, as_of: Any, statuses: dict[str, str]) -> None:
    st.markdown("#### PPI led ETF")
    if table.empty or as_of is None:
        st.info("PPI-led ETF performance is unavailable because the PPIACO calculation date or price data could not be loaded.")
        failed = [f"{asset}: {status}" for asset, status in statuses.items() if "unavailable" in status.lower() or "missing" in status.lower()]
        if failed:
            st.caption(" · ".join(failed))
        return

    st.caption(
        f"As of: {pd.Timestamp(as_of):%d %b %Y} · PPIACO: FRED · RTSI: TradingView MCP · "
        "ECH, ENOR, EWC, EWA, EZA, EPU, EWZ, GUNR, GNR, PICK, MXI, IXC, VEGI, WOOD, COPX, URNM: "
        "yfinance adjusted closes"
    )
    shown = table.copy()
    formatters = {f"{month}M": "{:+.1%}" for month in PPI_LED_HORIZONS}

    def semantic_return(value: Any) -> str:
        if pd.isna(value):
            return "color: #94a3b8"
        if float(value) > 0.001:
            return "background-color: #14532d; color: #f8fafc"
        if float(value) < -0.001:
            return "background-color: #7f1d1d; color: #f8fafc"
        return "background-color: #334155; color: #f8fafc"

    separators = {
        "PPIACO", "RTSI — Russia", "GUNR — Global Natural Resources", "Average",
    }

    def row_separation(row: pd.Series) -> list[str]:
        label = str(row.get("Asset", ""))
        border = "border-top: 2px solid #64748b;" if label in separators else ""
        weight = "font-weight: 700;" if label in {"Average", "Median"} else ""
        return [border + weight for _ in row]

    styled = shown.style.format(formatters, na_rep="N/A").apply(
        lambda column: [semantic_return(value) for value in column], subset=list(formatters), axis=0
    ).apply(row_separation, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=max(470, 35 * (len(shown) + 1)))

    period = st.selectbox(
        "Relative performance period",
        options=tuple(f"{month}M" for month in PPI_LED_HORIZONS),
        index=4,
        key="ppi_led_etf_relative_period",
        horizontal=True,
    )
    median_rows = shown.loc[shown["Asset"].eq("Median"), period]
    median = pd.to_numeric(median_rows, errors="coerce").iloc[0] if not median_rows.empty else np.nan
    median_label = "N/A" if pd.isna(median) else f"{median:+.1%}"
    st.caption(f"Median {period} Return: {median_label} · bars show asset return minus equity-universe median, in percentage points")
    chart_data = relative_median_frame(shown, period)
    if chart_data.empty:
        st.info(f"No assets have enough history to calculate relative performance for {period}.")
        return
    chart_data["Relative pp"] = chart_data["Relative"] * 100
    chart_data["Label"] = chart_data["Relative pp"].map(lambda value: f"{value:+.1f} pp")
    chart_data["Color"] = chart_data["Relative"].map(
        lambda value: "#22c55e" if value > 0.001 else "#ef4444" if value < -0.001 else "#94a3b8"
    )
    custom = np.column_stack([
        chart_data["Asset Return"], chart_data["Median Return"],
        chart_data["Relative pp"], chart_data["Rank"],
    ])
    fig = go.Figure(go.Bar(
        x=chart_data["Relative pp"], y=chart_data["Ticker"], orientation="h",
        marker_color=chart_data["Color"], text=chart_data["Label"], textposition="outside",
        customdata=custom,
        hovertemplate=(
            "Asset: %{y}<br>Period: " + period + "<br>Asset return: %{customdata[0]:+.1%}"
            "<br>Universe median: %{customdata[1]:+.1%}<br>Relative to median: %{customdata[2]:+.1f} pp"
            "<br>Rank: %{customdata[3]:.0f} / " + str(len(chart_data)) + "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="#e2e8f0", line_width=1.5)
    fig.update_layout(
        template="plotly_dark", title="Relative to Median Performance", height=max(470, 30 * len(chart_data) + 130),
        xaxis_title="Return vs Median, percentage points", yaxis_title="Ticker",
        yaxis=dict(autorange="reversed"), margin=dict(l=35, r=80, t=55, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    failures = [f"{asset}: {status}" for asset, status in statuses.items() if "unavailable" in status.lower() or "missing" in status.lower()]
    if failures:
        st.caption("Unavailable sources (shown as N/A): " + " · ".join(failures))


def _ppi_sector_tightening_easing(current: pd.Series) -> str:
    """Summarize current seasonal stress breadth across the three core and five confirmation sectors."""
    columns = [f"{sector} Seasonal Delta 6M" for sector in (*CORE_SERIES, *CONFIRMATION_SERIES)]
    values = pd.to_numeric(current.reindex(columns), errors="coerce")
    if values.isna().any():
        return "Sectors Tightening: N/A · Sectors Easing: N/A"
    tightening = int(values.gt(5).sum())
    easing = int(values.lt(-5).sum())
    return f"Sectors Tightening: {tightening}/8 · Sectors Easing: {easing}/8"


def _render_market_section(
    commodity: pd.DataFrame,
    sectors: pd.DataFrame,
    performance_prices: pd.DataFrame,
    selected_range: str,
    capex: pd.DataFrame,
    history: pd.DataFrame,
    term_history: pd.DataFrame,
    term_diagnostics: pd.DataFrame,
) -> None:
    with st.container(key="commodity-cycle-sector-dashboard"):
        st.markdown("#### Sector dashboard")
        if sectors.empty:
            st.info("Market confirmation data is unavailable.")
        else:
            cols = st.columns(len(sectors))
            for col, (_, row) in zip(cols, sectors.iterrows()):
                with col:
                    st.markdown(
                        f"<div style='font-size:2rem;font-weight:700;line-height:1.15'>{row['Sector']}</div>",
                        unsafe_allow_html=True,
                    )
                    avg_cftc = row.get("CFTC Average 5Y Percentile", np.nan)
                    avg_cftc_text = "N/A" if pd.isna(avg_cftc) else f"{float(avg_cftc):.1f}"
                    st.markdown(
                        "<div style='font-size:1.05rem;line-height:1.4;margin-top:0.25rem'>"
                        f"<div>Price: {row['Price State']} · Bullish {row['Bullish Count']} / Bearish {row['Bearish Count']}</div>"
                        f"<div>Curve tight breadth: {_fmt(row['Curve Tight Breadth'], '%')}</div>"
                        f"<div>Avg CFTC: {avg_cftc_text}</div>"
                        f"<div>CFTC Dispersion: {_fmt(row['CFTC Dispersion'], ' pts')}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
    st.markdown("#### Commodity confirmation table")
    if commodity.empty:
        st.info("No commodity market observations are currently available.")
    else:
        primary, auxiliary = _commodity_confirmation_frames(commodity)
        st.download_button(
            "Download Commodity Tables + Seasonal History (.xlsx)",
            data=commodity_tables_to_xlsx(primary, auxiliary, term_history, term_diagnostics),
            file_name=f"commodity_cycle_tables_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="commodity_cycle_tables_download",
        )
        st.dataframe(
            _style_commodity_table(primary, highlight_primary=True),
            use_container_width=True,
            hide_index=True,
            height=max(400, 46 * (len(primary) + 1)),
        )
        if not auxiliary.empty:
            with st.expander("Additional commodity diagnostics", expanded=False):
                st.dataframe(_style_commodity_table(auxiliary), use_container_width=True, hide_index=True)
    momentum_period = st.selectbox(
        "Price Momentum",
        options=("1M", "3M", "6M", "12M"),
        index=2,
        key="commodity_cycle_price_momentum_period",
    )
    st.markdown("#### Price momentum × CFTC positioning")
    if not commodity.empty:
        momentum_column = f"Return {momentum_period}"
        fig = go.Figure()
        for _, row in commodity.iterrows():
            x, y = row.get(momentum_column), row.get("5Y Percentile")
            if pd.notna(x) and pd.notna(y):
                raw_curve_state = row.get("Raw Curve State", "N/A")
                fig.add_trace(go.Scatter(
                    x=[x * 100], y=[y], mode="markers+text", text=[row["Commodity"]],
                    textposition="top center", name=row["Commodity"],
                    marker=_curve_scatter_marker(raw_curve_state),
                    hovertemplate=(
                        f"{row['Commodity']}<br>{momentum_period} price return: %{{x:.1f}}%"
                        f"<br>CFTC 5Y pctl: %{{y:.0f}}<br>Raw curve: {raw_curve_state}<extra></extra>"
                    ),
                ))
        fig.update_layout(template="plotly_dark", height=380, xaxis_title=f"Price return {momentum_period} (%)", yaxis_title="CFTC 5Y percentile", showlegend=False, margin=dict(l=30, r=20, t=20, b=30))
        fig.add_hline(y=75, line_dash="dot", line_color="#f59e0b")
        fig.add_hline(y=25, line_dash="dot", line_color="#38bdf8")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("#### Commodity price and momentum history")
    for row_start in range(0, len(PRICE_TICKERS), 3):
        columns = st.columns(3)
        for column, asset in zip(columns, list(PRICE_TICKERS)[row_start:row_start + 3]):
            with column:
                _render_asset_price_momentum(
                    asset,
                    performance_prices.get(asset, pd.Series(dtype=float)),
                    momentum_period,
                    selected_range,
                )
    if not capex.empty:
        _render_capex_intensity_chart(capex, history, selected_range)


def _commodity_confirmation_frames(commodity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_columns = [column for column in PRIMARY_COMMODITY_COLUMNS if column in commodity]
    primary = commodity.loc[:, primary_columns].copy()
    identifiers = [column for column in ("Sector", "Commodity") if column in commodity]
    diagnostic_columns = [column for column in commodity if column not in PRIMARY_COMMODITY_COLUMNS]
    auxiliary_columns = identifiers + [column for column in diagnostic_columns if column not in identifiers]
    auxiliary = commodity.loc[:, auxiliary_columns].copy() if auxiliary_columns else pd.DataFrame(index=commodity.index)
    primary = primary.rename(columns=DISPLAY_COLUMN_NAMES)
    auxiliary = auxiliary.rename(columns=DISPLAY_COLUMN_NAMES)
    return primary, auxiliary


def _commodity_numeric_formatters(frame: pd.DataFrame) -> dict[str, str]:
    formatters: dict[str, str] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        formatters[column] = "{:.2%}" if column in PERCENT_COLUMNS else "{:.2f}"
    return formatters


def _return_gradient_styles(values: pd.Series) -> list[str]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return ["" for _ in values]
    low, high = float(valid.min()), float(valid.max())
    styles = []
    for value in numeric:
        if pd.isna(value):
            styles.append("")
            continue
        ratio = 0.5 if high == low else (float(value) - low) / (high - low)
        if ratio <= 0.5:
            start, end, blend = (127, 29, 29), (133, 77, 14), ratio * 2
        else:
            start, end, blend = (133, 77, 14), (20, 83, 45), (ratio - 0.5) * 2
        red, green, blue = (round(start[i] + (end[i] - start[i]) * blend) for i in range(3))
        styles.append(f"background-color: rgb({red}, {green}, {blue}); color: #ffffff")
    return styles


def _raw_curve_state_style(value: Any) -> str:
    state = str(value).strip().lower()
    if state == "contango":
        return "background-color: #14532d; color: #ffffff"
    if state == "backwardation":
        return "background-color: #7f1d1d; color: #ffffff"
    return ""


def _curve_scatter_marker(value: Any) -> dict[str, Any]:
    """Use a prominent marker whose color represents absolute curve structure."""
    state = str(value).strip().lower()
    color = "#ef4444" if state == "backwardation" else "#22c55e" if state == "contango" else "#94a3b8"
    return {"size": 14, "color": color, "line": {"color": "#f8fafc", "width": 1}}


def _style_commodity_table(frame: pd.DataFrame, *, highlight_primary: bool = False) -> pd.io.formats.style.Styler:
    styled = frame.style.format(_commodity_numeric_formatters(frame), na_rep="N/A")
    if highlight_primary:
        returns = [column for column in ("Return 1M", "Return 3M", "Return 6M", "Return 12M") if column in frame]
        if returns:
            styled = styled.apply(_return_gradient_styles, subset=returns)
        if "Raw Curve State" in frame:
            styled = styled.map(_raw_curve_state_style, subset=["Raw Curve State"])
    return styled


def _price_momentum_series(prices: pd.Series, period: str) -> pd.Series:
    """Return weekly close-to-close momentum using the app's 4/13/26/52-week horizons."""
    weeks = MOMENTUM_WEEKS.get(period)
    if weeks is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if values.empty:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None).normalize()
    values.index = dates
    values = values.loc[~values.index.isna()].groupby(level=0).last().sort_index()
    weekly = values.resample("W-FRI").last().dropna()
    return weekly.div(weekly.shift(weeks)).sub(1).mul(100).rename(f"Return {period}")


def _render_asset_price_momentum(
    asset: str,
    prices: pd.Series,
    period: str,
    selected_range: str,
) -> None:
    values = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if values.empty:
        st.markdown(f"**{asset}**")
        st.info("No price history available.")
        return
    dates = pd.to_datetime(values.index, errors="coerce", utc=True).tz_localize(None).normalize()
    values.index = dates
    values = values.loc[~values.index.isna()].groupby(level=0).last().sort_index()
    weekly = values.resample("W-FRI").last().dropna()
    momentum = _price_momentum_series(values, period)
    visible_price = _slice_range(weekly.to_frame("Price"), selected_range)["Price"]
    visible_momentum = _slice_range(momentum.to_frame("Momentum"), selected_range)["Momentum"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.62, 0.38], subplot_titles=("Price", f"{period} price momentum"))
    fig.add_trace(go.Scatter(x=visible_price.index, y=visible_price, name=f"{asset} price",
                             line=dict(color="#38bdf8", width=1.6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=visible_momentum.index, y=visible_momentum, name=f"{period} return",
                             line=dict(color="#c084fc", width=1.6), showlegend=False,
                             connectgaps=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=350, title=dict(text=asset, x=0.02, xanchor="left"),
                      margin=dict(l=35, r=12, t=45, b=25), showlegend=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_ppi_cpi_roc_chart(history: pd.DataFrame, selected_range: str) -> None:
    view = _slice_range(history, selected_range)
    ppi_level = pd.to_numeric(history.get("PPIACO", pd.Series(index=history.index, dtype=float)), errors="coerce").dropna()
    cpi_level = pd.to_numeric(history.get("CPIAUCSL", pd.Series(index=history.index, dtype=float)), errors="coerce").dropna()
    ppi_roc = ppi_level.pct_change(periods=12, fill_method=None).mul(100).reindex(view.index)
    cpi_roc = cpi_level.pct_change(periods=12, fill_method=None).mul(100).reindex(view.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=ppi_roc, name="PPIACO 12M ROC",
                             line=dict(color="#38bdf8", width=2), connectgaps=False))
    fig.add_trace(go.Scatter(x=view.index, y=cpi_roc, name="CPIAUCSL 12M ROC",
                             line=dict(color="#f97316", width=2), connectgaps=False))

    observed_ppi = ppi_roc.dropna()
    if len(observed_ppi) >= 2:
        mean = float(observed_ppi.mean())
        std = float(observed_ppi.std(ddof=1))
        if np.isfinite(std):
            for sigma, color in ((1, "#fde047"), (2, "#fb7185")):
                for direction in (1, -1):
                    level = mean + direction * sigma * std
                    label = f"PPIACO mean {direction * sigma:+d}σ"
                    fig.add_trace(go.Scatter(
                        x=view.index, y=np.full(len(view), level), name=label,
                        line=dict(color=color, width=1.2, dash="dot"),
                        hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
                    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig.update_layout(template="plotly_dark", title="PPIACO 12M ROC + CPIAUCSL 12M ROC",
                      height=360, margin=dict(l=35, r=20, t=50, b=85),
                      yaxis_title="12M ROC (%)",
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if cpi_roc.notna().sum() == 0:
        st.caption("CPIAUCSL history is unavailable for the selected range.")


def _render_ppi_core_chart(history: pd.DataFrame, selected_range: str) -> None:
    view = _slice_range(history, selected_range)
    fig = go.Figure()
    _add_state_bands(fig, view, "Core State", CORE_COLORS)
    fig.add_trace(go.Scatter(x=view.index, y=view["PPIACO"], name="PPIACO", showlegend=False, line=dict(color="white", width=2), customdata=np.column_stack([view["PPI12M"], view["PPI Impulse 3M pp"], view["Core Median Stress"], view["Seasonal Tightening 5"], view["Rolling Tightening 5"], view["Confirmation Net 5"]]), hovertemplate="%{x|%Y-%m}<br>PPIACO %{y:.2f}<br>PPI12M %{customdata[0]:.2f}%<br>PPI impulse %{customdata[1]:.2f} pp<br>Core stress %{customdata[2]:.1f}<br>Seasonal tightening %{customdata[3]:.0f}/3<br>Rolling tightening %{customdata[4]:.0f}/3<br>Confirmation net %{customdata[5]:.0%}<extra></extra>"))
    _add_state_legend(fig, view, "Core State", CORE_COLORS)
    fig.update_layout(template="plotly_dark", title="PPIACO + Core State", height=470,
                      margin=dict(l=35, r=25, t=55, b=105), yaxis_title="PPIACO index",
                      legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, title_text="State"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_final_state_chart(history: pd.DataFrame, selected_range: str, view: str) -> None:
    data_col = "Final State" if view == "Final State" else "Final State 2"
    view_data = _slice_range(history, selected_range)
    fig = go.Figure()
    _add_state_bands(fig, view_data, data_col, FINAL_COLORS)
    custom = np.column_stack([view_data["PPI12M"], view_data["PPI Impulse 3M pp"], view_data["Core State"], view_data["Final State"], view_data["Final State 2"]])
    fig.add_trace(go.Scatter(x=view_data.index, y=view_data["PPIACO"], name="PPIACO", showlegend=False, line=dict(color="white", width=2), customdata=custom,
                             hovertemplate="%{x|%Y-%m}<br>PPIACO %{y:.2f}<br>PPI12M %{customdata[0]:.2f}%<br>3M impulse %{customdata[1]:.2f} pp<br>Core %{customdata[2]}<br>Final State %{customdata[3]}<br>Final State 2 %{customdata[4]}<extra></extra>"))
    _add_state_legend(fig, view_data, data_col, FINAL_COLORS)
    fig.update_layout(template="plotly_dark", title=f"PPIACO + {view}", height=450,
                      margin=dict(l=35, r=25, t=55, b=105), yaxis_title="PPIACO index",
                      legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, title_text="State"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_fred_heatmap(history: pd.DataFrame) -> None:
    row = latest_complete_commodity_cycle_row(history)
    if row.empty:
        st.info("No fully classified month yet; stress windows require sufficient prior monthly data.")
        return
    table = []
    for label in FRED_SERIES:
        if label in {"PPIACO", "CPIAUCSL", "CAPEX", "FPI"}:
            continue
        rolling = row.get(f"{label} Rolling Stress")
        seasonal = row.get(f"{label} Seasonal Stress")
        droll = row.get(f"{label} Rolling Delta 6M")
        dseason = row.get(f"{label} Seasonal Delta 6M")
        table.append({"Series": label, "Inventory / Sales": _fmt(row.get(label)), "Seasonal Stress": _fmt(seasonal), "Rolling Stress": _fmt(rolling), "Δ Seasonal 6M": _fmt(dseason), "Δ Rolling 6M": _fmt(droll), "Direction": _direction_label(droll)})
    frame = pd.DataFrame(table)
    st.dataframe(frame, use_container_width=True, hide_index=True, height=max(400, 46 * (len(frame) + 1)))


def _render_capex_intensity_chart(capex: pd.DataFrame, history: pd.DataFrame, selected_range: str) -> None:
    recent = _slice_range(capex, selected_range)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=recent.index, y=recent["CAPEX Intensity"], name="CAPEX / FPI", line=dict(color="#facc15")), row=1, col=1)
    if not history.empty:
        ppi = history["PPIACO"].dropna()
        ppi = _slice_range(ppi.to_frame(), selected_range)["PPIACO"]
        fig.add_trace(go.Scatter(x=ppi.index, y=ppi, name="PPIACO", line=dict(color="#38bdf8")), row=1, col=1, secondary_y=True)
    fig.update_layout(template="plotly_dark", height=350, title="CAPEX Intensity vs FRED PPIACO",
                      margin=dict(l=35, r=30, t=50, b=30), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="CAPEX / FPI", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PPIACO index", row=1, col=1, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_diagnostics_section(data: dict[str, Any], commodity: pd.DataFrame) -> None:
    _render_data_status(data)
    st.caption("The bundled workbook remains the immutable baseline for Energy and Metals. The downloadable workbook below is generated from the live engine and includes reconstructed Agriculture seasonal history when available.")
    if not commodity.empty:
        st.markdown("#### Input provenance")
        provenance_columns = [
            "Commodity", "Price Date", "Price Status", "Performance As Of", "Analytics Return Source",
            "Performance Status", "Term Structure As Of", "Term Structure Status", "As Of Alignment",
            "Price Data Quality", "Contract Selection Quality", "CurveDataQuality", "Leg 1", "Leg 2",
            "Curve Spread", "Annualized Curve Spread", "Raw Curve State", "MTD Average Spread",
            "MTD Daily Observations", "Current Seasonal Status", "Seasonal Percentile 5Y",
            "Seasonal Percentile As Of", "Seasonal Percentile Status", "Seasonal Percentile 5Y HistoryN",
            "Seasonal Percentile 5Y HistoryStatus", "Seasonal Percentile 10Y",
            "Seasonal Percentile 10Y HistoryN", "Seasonal Percentile 10Y HistoryStatus",
            "Seasonal Percentile 10Y Explanation", "Current Curve Vendor", "Seasonal History Vendor",
            "Vendor Consistency", "Seasonal History Source", "Rollover Method", "Rollover Date",
            "Days To Expiry", "CFTC As Of", "CFTC Status", "Latest Official CFTC Report Date",
            "Series Present In Latest Report", "CFTC Contract Market Code", "CFTC Market Name",
            "CFTC Open Interest", "MM Long", "MM Short", "MM Spreading", "MM Net", "Net Direction",
            "CFTC Relative State", "History Weeks", "Last Available Date", "Last Available MM Net % OI",
            "Last Available COT 5Y Percentile", "Reason Current Signal Missing",
        ]
        available_columns = [column for column in provenance_columns if column in commodity]
        st.dataframe(commodity[available_columns], use_container_width=True)

    diagnostics = data.get("term_diagnostics", pd.DataFrame())
    if not diagnostics.empty:
        seasonal = diagnostics.loc[
            diagnostics.get("Diagnostic Type", pd.Series(index=diagnostics.index, dtype=object)).eq("Agriculture Seasonal Structure")
        ].copy()
        contracts = diagnostics.drop(seasonal.index)
        if not seasonal.empty:
            st.markdown("#### Agriculture seasonal structure")
            seasonal_columns = [
                "Commodity", "Season", "Near Contract", "Deferred Contract",
                "Near Contract Symbol", "Deferred Contract Symbol", "Current Near DTE",
                "Matched DTE", "DTE Window", "Valid N", "Median Seasonal Spread",
                "Current Spread", "5Y HistoryN", "5Y History Start", "5Y History End",
                "5Y Seasonal Percentile", "10Y HistoryN", "10Y History Start",
                "10Y History End", "10Y Seasonal Percentile", "Seasonal Relative State",
                "Source", "Last Update", "Data Quality", "Missing Contracts", "Rejection Reason",
            ]
            st.dataframe(
                seasonal[[column for column in seasonal_columns if column in seasonal]],
                use_container_width=True,
                hide_index=True,
            )
        if not contracts.empty:
            st.markdown("#### Contract discovery / selection diagnostics")
            st.dataframe(contracts, use_container_width=True, hide_index=True)


def _render_data_status(data: dict[str, Any]) -> None:
    rows = []
    for name, status in data["fred_status"].items():
        rows.append({"Data": name, "Provider": "FRED", "Status": status})
    for name, status in data["price_status"].items():
        rows.append({"Data": f"{name} price", "Provider": "Yahoo Finance", "Status": status})
    for name, status in data.get("performance_status", {}).items():
        provider = data.get("performance_source", {}).get(name, "TradingView MCP")
        rows.append({"Data": f"{name} performance", "Provider": provider, "Status": status})
    for name, status in data["cftc_status"].items():
        rows.append({"Data": name, "Provider": "Shared CFTC positioning service", "Status": status})
    term = data["term_current"]
    seen = set()
    for _, row in term.iterrows():
        seen.add(str(row["Asset"]))
        rows.append({"Data": f"{row['Asset']} {row.get('Structure', 'term structure')}", "Provider": row.get("Source", "N/A"), "Status": f"{row.get('Status', 'MISSING')} — as of {row.get('As Of', 'N/A')}; observed {row.get('Observed At', 'N/A')}"})
    for asset in PRICE_TICKERS:
        if asset not in seen:
            provider = "Westmetall LME Cash/3M" if asset in {"Copper", "Aluminum"} else "Yahoo Finance individual futures (TradingView MCP fallback)"
            rows.append({"Data": f"{asset} term structure", "Provider": provider, "Status": "MISSING — no successful current observation stored"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _add_state_bands(fig: go.Figure, frame: pd.DataFrame, column: str, colors: dict[str, str]) -> None:
    if frame.empty or column not in frame:
        return
    dates = pd.to_datetime(frame.index)
    states = frame[column].astype(str).to_numpy()
    start = 0
    for end in range(1, len(frame) + 1):
        if end == len(frame) or states[end] != states[start]:
            state = states[start]
            if state in colors:
                left = dates[start].to_period("M").to_timestamp()
                right = (dates[end - 1].to_period("M") + 1).to_timestamp() if end == len(frame) else dates[end].to_period("M").to_timestamp()
                fig.add_vrect(x0=left, x1=right, fillcolor=colors[state], opacity=STATE_BAND_OPACITY,
                              line_width=0, layer="below")
            start = end


def _add_state_legend(fig: go.Figure, frame: pd.DataFrame, column: str, colors: dict[str, str]) -> None:
    if frame.empty or column not in frame:
        return
    present = set(frame[column].dropna().astype(str))
    for state, color in colors.items():
        if state not in present:
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=state,
            marker=dict(symbol="square", size=12, color=color), hoverinfo="skip",
        ))


def _range_picker(key: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = "10Y"
    st.radio("History range", RANGE_OPTIONS, horizontal=True, key=key)


def _slice_range(frame: pd.DataFrame, option: str) -> pd.DataFrame:
    if frame.empty or option == "Full":
        return frame
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20}.get(option)
    if years is None:
        return frame
    last = pd.to_datetime(frame.index.max())
    return frame.loc[pd.to_datetime(frame.index) >= last - pd.DateOffset(years=years)]


def _format_state(value: Any) -> str:
    return str(value) if value is not None and not pd.isna(value) else "N/A"


def _fmt(value: Any, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        number = float(value)
        if suffix == "%" and abs(number) <= 1:
            return f"{number:.1%}"
        return f"{number:.1f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _direction_label(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if value > 5:
        return "Tightening"
    if value < -5:
        return "Easing"
    return "Stable"

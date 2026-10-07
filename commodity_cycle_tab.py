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
    load_commodity_term_structure,
)
from commodity_cycle.export import commodity_tables_to_xlsx

TTL_SECONDS = 21600
RANGE_OPTIONS = ("1Y", "3Y", "5Y", "10Y", "20Y", "Full")
CORE_COLORS = {
    "Neutral": "#e2e8f0", "Early Broadening": "#fde047", "Confirmed Broadening": "#fb923c",
    "Systemic Broadening": "#f43f5e", "Mature": "#c084fc", "Early Easing": "#4ade80",
    "Confirmed Easing": "#38bdf8", "DATA INCOMPLETE": "#64748b",
}
FINAL_COLORS = {
    "Low Inflation / Neutral": "#38bdf8", "Reflation / Early Inflation": "#fde047",
    "Inflation Expansion": "#f43f5e", "Late Cycle / Peak Risk": "#c084fc",
    "Disinflation Transition": "#4ade80", "Confirmed Disinflation": "#22d3ee",
    "Broad Inflation": "#fb923c", "N/A": "#64748b",
}
STATE_BAND_OPACITY = 0.30
PRIMARY_COMMODITY_COLUMNS = (
    "Sector", "Commodity", "Price", "Return 3M", "Return 6M", "Return 12M", "Price State",
    "Curve Spread", "Raw Curve State", "MM Net % OI", "4W Change", "13W Change", "5Y Percentile",
    "Seasonal Percentile 5Y", "CFTC Relative State", "Seasonal Curve State", "Price × Curve",
    "Price Date", "Term Structure As Of",
)
PERCENT_COLUMNS = {
    "Return 3M", "Return 6M", "Return 12M", "Curve Spread", "MTD Average Spread",
}
DISPLAY_COLUMN_NAMES = {
    "3Y Percentile": "COT 3Y Percentile",
    "5Y Percentile": "COT 5Y Percentile",
    "Seasonal Percentile 5Y": "Spread 5Y Seasonal Percentile",
    "Seasonal Percentile 10Y": "Spread 10Y Seasonal Percentile",
}


@st.cache_data(show_spinner="Loading Commodity Cycle data…", ttl=TTL_SECONDS)
def load_commodity_cycle_snapshot(api_key: str | None, refresh_nonce: int = 0) -> dict[str, Any]:
    _ = refresh_nonce
    fred, fred_status = load_fred_history(api_key)
    prices, price_status = load_monthly_prices()
    term_history, term_current, term_diagnostics = load_commodity_term_structure()
    cftc, cftc_status = load_cftc_snapshot()
    history, capex, term_history_rt = build_commodity_cycle_history(fred, prices, term_history)
    commodity, sector = build_market_confirmation(prices, term_history_rt, term_current, cftc)
    if not commodity.empty:
        commodity["Price Status"] = commodity["Commodity"].map(price_status)
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
        "term_history": term_history_rt, "term_current": term_current, "term_diagnostics": term_diagnostics, "cftc": cftc,
        "cftc_status": cftc_status, "history": history, "capex": capex,
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
    sector_states = {str(r["Sector"]): str(r["Market Confirmation"]) for _, r in sectors.iterrows()}

    summary = [
        ("Core State", current.get("Core State", "N/A")),
        ("PPI Confirmation", current.get("PPI Confirmation", "N/A")),
        ("Final State", current.get("Final State", "N/A")),
        ("Final State 2", current.get("Final State 2", "N/A")),
        ("CAPEX Vulnerability", capex_current.get("CAPEX State", "N/A")),
        ("Energy", sector_states.get("Energy", "N/A")),
        ("Metals", sector_states.get("Metals", "N/A")),
        ("Agriculture", sector_states.get("Agriculture", "N/A")),
    ]
    cards = st.columns(4)
    for idx, (label, value) in enumerate(summary):
        with cards[idx % 4]:
            st.metric(label, _format_state(value))

    overview, physical, market, capex_tab, diagnostics = st.tabs(
        ["Overview", "FRED Inventory/Sales Regime", "Market Confirmation", "CAPEX Vulnerability", "Diagnostics / Data"]
    )
    with overview:
        if history.empty:
            st.info("The FRED Inventory/Sales model is unavailable. Check the data diagnostics below; market and positioning sections remain available where their feeds loaded.")
        else:
            _range_picker("commodity_cycle_overview_range")
            _render_ppi_core_chart(history, st.session_state["commodity_cycle_overview_range"])
            view = st.radio("Regime shading", ["Final State", "Final State 2"], index=1, horizontal=True,
                            key="commodity_cycle_final_view_v2")
            _render_final_state_chart(history, st.session_state["commodity_cycle_overview_range"], view)
    with physical:
        if history.empty:
            st.info("No FRED history is currently available.")
        else:
            _render_fred_heatmap(history)
            st.markdown("#### Core stress and breadth diagnostics")
            cols = [c for c in history if c.endswith("Rolling Stress") or c.endswith("Seasonal Stress")]
            st.dataframe(history[cols].tail(24).round(1), use_container_width=True)
    with market:
        _render_market_section(commodity, sectors, data["prices"], data["term_history"])
    with capex_tab:
        if capex.empty:
            st.info("CAPEX intensity requires both quarterly FRED series E318RC1Q027SBEA and FPI.")
        else:
            st.metric("CAPEX intensity", _fmt(capex_current.get("CAPEX Intensity")), help="E318RC1Q027SBEA / FPI; quarterly, not seasonally normalized.")
            st.metric("Full-history percentile", _fmt(capex_current.get("CAPEX Current Percentile"), "%"))
            st.metric("24M direction", f"{_fmt(capex_current.get('CAPEX 24M Change'), '%')} · {capex_current.get('CAPEX Direction', 'N/A')}")
            _render_capex_charts(capex, history)
            history_view = capex[["CAPEX Intensity", "CAPEX Expanding Percentile", "CAPEX Vulnerability RT", "CAPEX 24M Change", "CAPEX Direction"]].tail(20).copy()
            st.dataframe(history_view.round(3), use_container_width=True)
    with diagnostics:
        _render_data_status(data)
        st.caption("The workbook remains the immutable historical baseline for Energy and Metals. Agriculture seasonal history is reconstructed from cached individual expired CBOT contracts using DTE-aligned TradingView MCP daily closes.")
        if not commodity.empty:
            st.markdown("#### Input provenance")
            st.dataframe(commodity[[c for c in ["Commodity", "Price Date", "Price Status", "Term Structure As Of", "Term Structure Status", "CurveDataQuality", "Leg 1", "Leg 2", "Curve Spread", "Raw Curve State", "MTD Average Spread", "MTD Daily Observations", "Current Seasonal Status", "Seasonal Percentile 5Y", "5Y HistoryN", "5Y HistoryStartDate", "5Y HistoryEndDate", "5Y HistoryStatus", "Seasonal Percentile 10Y", "10Y HistoryN", "10Y HistoryStartDate", "10Y HistoryEndDate", "10Y HistoryStatus", "Rollover Method", "Rollover Date", "Days To Expiry", "Updated Date", "CFTC Status"] if c in commodity]], use_container_width=True)
        diagnostics = data.get("term_diagnostics", pd.DataFrame())
        if not diagnostics.empty:
            seasonal = diagnostics.loc[diagnostics.get("Diagnostic Type", pd.Series(index=diagnostics.index, dtype=object)).eq("Agriculture Seasonal Structure")].copy()
            contracts = diagnostics.drop(seasonal.index)
            if not seasonal.empty:
                st.markdown("#### Agriculture seasonal structure")
                seasonal_columns = [
                    "Commodity", "Season", "Near Contract", "Deferred Contract",
                    "Near Contract Symbol", "Deferred Contract Symbol", "Current Near DTE",
                    "Matched DTE", "DTE Window", "Valid N", "Median Seasonal Spread",
                    "Current Spread", "5Y HistoryN", "5Y History Start", "5Y History End",
                    "5Y Seasonal Percentile", "10Y HistoryN", "10Y History Start",
                    "10Y History End", "10Y Seasonal Percentile", "Seasonal Curve State",
                    "Source", "Last Update", "Data Quality", "Missing Contracts", "Rejection Reason",
                ]
                st.dataframe(seasonal[[column for column in seasonal_columns if column in seasonal]], use_container_width=True, hide_index=True)
            if not contracts.empty:
                st.markdown("#### Contract discovery / selection diagnostics")
                st.dataframe(contracts, use_container_width=True, hide_index=True)


def _render_market_section(commodity: pd.DataFrame, sectors: pd.DataFrame, prices: pd.DataFrame, term_history: pd.DataFrame) -> None:
    st.markdown("#### Sector dashboard")
    if sectors.empty:
        st.info("Market confirmation data is unavailable.")
    else:
        cols = st.columns(len(sectors))
        for col, (_, row) in zip(cols, sectors.iterrows()):
            with col:
                st.markdown(f"**{row['Sector']}**")
                st.metric("Market Confirmation", str(row["Market Confirmation"]))
                st.caption(f"Price: {row['Price State']} · Bullish {row['Bullish Count']} / Bearish {row['Bearish Count']}")
                st.caption(f"Curve tight breadth: {_fmt(row['Curve Tight Breadth'], '%')} · CFTC dispersion: {_fmt(row['CFTC Dispersion'], ' pts')}")
    st.markdown("#### Commodity confirmation table")
    if commodity.empty:
        st.info("No commodity market observations are currently available.")
    else:
        primary, auxiliary = _commodity_confirmation_frames(commodity)
        st.download_button(
            "Download Commodity Tables (.xlsx)",
            data=commodity_tables_to_xlsx(primary, auxiliary),
            file_name=f"commodity_cycle_tables_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="commodity_cycle_tables_download",
        )
        st.dataframe(_style_commodity_table(primary, highlight_primary=True), use_container_width=True, hide_index=True)
        if not auxiliary.empty:
            st.markdown("#### Additional commodity diagnostics")
            st.dataframe(_style_commodity_table(auxiliary), use_container_width=True, hide_index=True)
        st.markdown("#### Price momentum × CFTC positioning")
        fig = go.Figure()
        for _, row in commodity.iterrows():
            x, y = row.get("Return 12M"), row.get("5Y Percentile")
            if pd.notna(x) and pd.notna(y):
                fig.add_trace(go.Scatter(x=[x * 100], y=[y], mode="markers+text", text=[row["Commodity"]], textposition="top center", name=row["Commodity"], hovertemplate=f"{row['Commodity']}<br>12M price return: %{{x:.1f}}%<br>CFTC 5Y pctl: %{{y:.0f}}<extra></extra>"))
        fig.update_layout(template="plotly_dark", height=380, xaxis_title="Price return 12M (%)", yaxis_title="CFTC 5Y percentile", showlegend=False, margin=dict(l=30, r=20, t=20, b=30))
        fig.add_hline(y=75, line_dash="dot", line_color="#f59e0b")
        fig.add_hline(y=25, line_dash="dot", line_color="#38bdf8")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        asset = st.selectbox("Commodity drill-down", commodity["Commodity"].tolist(), key="commodity_cycle_drilldown")
        _render_commodity_drilldown(asset, prices, term_history)


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


def _style_commodity_table(frame: pd.DataFrame, *, highlight_primary: bool = False) -> pd.io.formats.style.Styler:
    styled = frame.style.format(_commodity_numeric_formatters(frame), na_rep="N/A")
    if highlight_primary:
        returns = [column for column in ("Return 3M", "Return 6M", "Return 12M") if column in frame]
        if returns:
            styled = styled.apply(_return_gradient_styles, subset=returns)
        if "Raw Curve State" in frame:
            styled = styled.map(_raw_curve_state_style, subset=["Raw Curve State"])
    return styled


def _render_commodity_drilldown(asset: str, prices: pd.DataFrame, term_history: pd.DataFrame) -> None:
    p = prices[asset].dropna() if asset in prices else pd.Series(dtype=float)
    curve = term_history.loc[(term_history["Asset"] == asset) & pd.to_numeric(term_history["Spread %"], errors="coerce").notna()].copy() if not term_history.empty else pd.DataFrame()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        specs=[[{}], [{}], [{"secondary_y": True}]],
                        subplot_titles=(f"{asset} monthly price (Yahoo Finance continuous future)", "3M price momentum", "Provided term-structure spread and seasonal percentile"))
    if len(p):
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name="Price", line=dict(color="#38bdf8")), row=1, col=1)
        momentum = p.div(p.shift(3)).sub(1) * 100
        fig.add_trace(go.Scatter(x=momentum.index, y=momentum, name="3M return", line=dict(color="#a78bfa"), connectgaps=False), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    if not curve.empty:
        fig.add_trace(go.Scatter(x=curve["Date"], y=curve["Spread %"] * 100, name="Spread %", line=dict(color="#facc15"), connectgaps=False), row=3, col=1, secondary_y=False)
        for pctl_col, color in (("Seasonal Pctl 5Y RT", "#fb7185"), ("Seasonal Pctl 10Y RT", "#c084fc")):
            if pctl_col in curve:
                fig.add_trace(go.Scatter(x=curve["Date"], y=pd.to_numeric(curve[pctl_col], errors="coerce"), name=pctl_col, line=dict(color=color, dash="dot"), connectgaps=False), row=3, col=1, secondary_y=True)
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=3, col=1)
    fig.update_layout(template="plotly_dark", height=690, margin=dict(l=35, r=20, t=45, b=25), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="3M return %", row=2, col=1)
    fig.update_yaxes(title_text="Spread %", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Seasonal percentile", range=[0, 100], row=3, col=1, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if curve.empty:
        st.caption("No supplied historical term-structure observations for this commodity; no synthetic curve is shown.")


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
        if label in {"PPIACO", "CAPEX", "FPI"}:
            continue
        rolling = row.get(f"{label} Rolling Stress")
        seasonal = row.get(f"{label} Seasonal Stress")
        droll = row.get(f"{label} Rolling Delta 6M")
        dseason = row.get(f"{label} Seasonal Delta 6M")
        table.append({"Series": label, "Inventory / Sales": _fmt(row.get(label)), "Seasonal Stress": _fmt(seasonal), "Rolling Stress": _fmt(rolling), "Δ Seasonal 6M": _fmt(dseason), "Δ Rolling 6M": _fmt(droll), "Direction": _direction_label(droll)})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def _render_capex_charts(capex: pd.DataFrame, history: pd.DataFrame) -> None:
    recent = _slice_range(capex, st.session_state.get("commodity_cycle_overview_range", "10Y"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        specs=[[{"secondary_y": True}], [{}]],
                        subplot_titles=("CAPEX Intensity vs FRED PPIACO", "CAPEX Vulnerability — expanding, no look-ahead percentile"))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["CAPEX Intensity"], name="CAPEX / FPI", line=dict(color="#facc15")), row=1, col=1)
    if not history.empty:
        ppi = history["PPIACO"].dropna()
        ppi = _slice_range(ppi.to_frame(), st.session_state.get("commodity_cycle_overview_range", "10Y"))["PPIACO"]
        fig.add_trace(go.Scatter(x=ppi.index, y=ppi, name="PPIACO", line=dict(color="#38bdf8")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["CAPEX Vulnerability RT"], name="Vulnerability", line=dict(color="#fb7185")), row=2, col=1)
    for low, high, color, label in (
        (0, 20, "#22c55e", "Very Low / Low"), (20, 40, "#38bdf8", "Neutral"),
        (40, 60, "#facc15", "Moderate-High"), (60, 80, "#f97316", "High"),
        (80, 100, "#ef4444", "Extreme"),
    ):
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.08, line_width=0, row=2, col=1,
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(template="plotly_dark", height=580, margin=dict(l=35, r=30, t=45, b=25), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="CAPEX / FPI", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PPIACO index", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Vulnerability (100 − expanding percentile)", range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if not history.empty:
        common = history[["Final State 2"]].join(capex[["CAPEX Vulnerability RT"]], how="inner").dropna()
        if not common.empty:
            overlay = go.Figure()
            _add_state_bands(overlay, common, "Final State 2", FINAL_COLORS)
            overlay.add_trace(go.Scatter(x=common.index, y=common["CAPEX Vulnerability RT"], name="CAPEX Vulnerability RT",
                                         showlegend=False, line=dict(color="white", width=2), connectgaps=False))
            _add_state_legend(overlay, common, "Final State 2", FINAL_COLORS)
            overlay.update_layout(template="plotly_dark", height=420, title="CAPEX Vulnerability + Final State 2",
                                  yaxis_title="Vulnerability", margin=dict(l=35, r=20, t=55, b=105),
                                  legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, title_text="State"))
            st.plotly_chart(overlay, use_container_width=True, config={"displayModeBar": False})


def _render_data_status(data: dict[str, Any]) -> None:
    rows = []
    for name, status in data["fred_status"].items():
        rows.append({"Data": name, "Provider": "FRED", "Status": status})
    for name, status in data["price_status"].items():
        rows.append({"Data": f"{name} price", "Provider": "Yahoo Finance", "Status": status})
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

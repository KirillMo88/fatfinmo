from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from finance_core import download_completed_ohlcv
from fred_client import FredApiError, download_fred_series
from global_liquidity import (
    GLOBAL_LIQUIDITY_STORAGE_DIR,
    read_global_liquidity,
    start_background_update_if_stale as start_global_liquidity_update_if_stale,
    update_global_liquidity,
)
from macro_research_export import (
    DEFAULT_START_DATE,
    EXPORT_CATEGORY_OPTIONS,
    EXPORT_LAYER_OPTIONS,
    EXPORT_MODEL_USAGE_OPTIONS,
    build_weekly_macro_research_workbook,
)


SLOW_REFRESH_SECONDS = 21600
HORIZON_WEEKS = {
    "1W": 1,
    "1M": 4,
    "3M": 13,
    "6M": 26,
    "12M": 52,
    "36M": 156,
}
CHANGE_COLUMNS = list(HORIZON_WEEKS.keys())
ChangeType = Literal["percent", "bps", "absolute", "state"]
CurrentFormat = Literal["number", "percent", "bps", "usd_tn", "usd_bn", "integer"]


@dataclass(frozen=True)
class MacroSeriesSpec:
    block: str
    instrument: str
    source: str
    change_type: ChangeType
    current_format: CurrentFormat
    frequency: str
    unit: str
    series: pd.Series
    data_status: str = "OK"


def render_global_macro_tab(api_key: str | None = None) -> None:
    st.subheader("Global Macro")
    st.caption(
        "Monitoring/data layer only. No combined Global Macro Score is calculated here; strategy tabs consume only the factors they need."
    )
    st.markdown(
        """
<style>
.global-macro-table-wrap {
    width: 100%;
    overflow-x: hidden;
}
.global-macro-table-wrap table.global-macro-table {
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse;
    font-size: 0.72rem;
}
.global-macro-table-wrap table.global-macro-table th,
.global-macro-table-wrap table.global-macro-table td {
    width: auto !important;
    max-width: none !important;
    padding: 0.38rem 0.42rem !important;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2;
    height: 1.9rem;
    max-height: 1.9rem;
}
.global-macro-table-wrap table.global-macro-table th {
    font-weight: 700;
}
</style>
""",
        unsafe_allow_html=True,
    )
    if "global_macro_refresh_nonce" not in st.session_state:
        st.session_state["global_macro_refresh_nonce"] = 0
    controls = st.columns([1.2, 1.0, 1.0, 1.4, 3.4])
    with controls[0]:
        force_refresh = st.button("Refresh Global Macro", use_container_width=True, key="global_macro_refresh")
    with controls[1]:
        export_start = st.date_input("Export start", value=pd.Timestamp(DEFAULT_START_DATE).date(), key="macro_research_export_start")
    with controls[2]:
        export_end = st.date_input("Export end", value=pd.Timestamp.now(tz="UTC").date(), key="macro_research_export_end")
    with controls[4]:
        st.markdown(
            f"<div style='padding-top:1.55rem; color:#94a3b8; font-size:0.78rem;'>"
            f"Liquidity storage: {GLOBAL_LIQUIDITY_STORAGE_DIR}</div>",
            unsafe_allow_html=True,
        )
    export_filters = st.columns([1.35, 2.35, 2.35, 2.0])
    with export_filters[0]:
        export_layers = st.multiselect(
            "Export Layer",
            options=EXPORT_LAYER_OPTIONS,
            default=["All"],
            help="All exports every layer; remove it to select individual layers.",
            key="macro_research_export_layers",
        )
    with export_filters[1]:
        export_model_usages = st.multiselect(
            "Export Model usage",
            options=EXPORT_MODEL_USAGE_OPTIONS,
            default=["All"],
            help="All exports every model usage; remove it to select individual usages.",
            key="macro_research_export_model_usages",
        )
    with export_filters[2]:
        export_categories = st.multiselect(
            "Export Category",
            options=EXPORT_CATEGORY_OPTIONS,
            default=["All"],
            help="All exports every category; remove it to select individual categories.",
            key="macro_research_export_categories",
        )
    with export_filters[3]:
        export_research = st.button("Export Weekly Macro Dataset", use_container_width=True, key="global_macro_research_export")
    _render_tradingview_mcp_panel()

    if force_refresh:
        st.session_state["global_macro_refresh_nonce"] += 1
        with st.spinner("Refreshing Global Liquidity sources..."):
            try:
                update_global_liquidity(api_key=api_key, force=True)
            except Exception as exc:
                st.warning(f"Liquidity refresh did not complete: {exc}")
    else:
        try:
            start_global_liquidity_update_if_stale(api_key=api_key)
        except Exception:
            pass

    refresh_nonce = st.session_state.get("slow_refresh_nonce", 0) + st.session_state["global_macro_refresh_nonce"]
    snapshot = load_global_macro_snapshot(api_key or "", refresh_nonce)
    if snapshot.empty:
        st.warning("Global Macro data is not available yet.")
        return
    if export_research:
        with st.spinner("Building weekly macro research workbook..."):
            try:
                workbook_bytes, filename = build_weekly_macro_research_workbook(
                    api_key=api_key,
                    start_date=pd.Timestamp(export_start),
                    end_date=pd.Timestamp(export_end),
                    layers=export_layers,
                    model_usages=export_model_usages,
                    categories=export_categories,
                )
                st.session_state["global_macro_research_export_payload"] = {"bytes": workbook_bytes, "filename": filename}
            except Exception as exc:
                st.warning(f"Weekly macro research export failed: {exc}")
    export_payload = st.session_state.get("global_macro_research_export_payload")
    if isinstance(export_payload, dict) and export_payload.get("bytes"):
        st.download_button(
            "Download Weekly Macro Dataset .xlsx",
            data=export_payload["bytes"],
            file_name=str(export_payload.get("filename") or "global_macro_research_weekly.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
            key="global_macro_research_download",
        )

    _render_global_macro_summary(snapshot)
    table_widths = {
        "instrument": _global_macro_column_width(snapshot, "Instrument"),
        "source": _global_macro_column_width(snapshot, "Source"),
    }
    for block in [
        "Markets",
        "Global Liquidity",
        "Funding Conditions",
        "FX / USD",
        "Inflation",
        "Rates & Curves",
        "Growth / Business Cycle",
        "Risk / Financial Conditions",
    ]:
        block_df = snapshot[snapshot["Block"].eq(block)].drop(columns=["Block"], errors="ignore")
        if block_df.empty:
            continue
        st.markdown(f"### {block}")
        st.markdown(_global_macro_table_html(block_df, block, table_widths), unsafe_allow_html=True)


def _render_tradingview_mcp_panel() -> None:
    with st.expander("TradingView MCP", expanded=False):
        try:
            import tradingview_mcp as tv_mcp
        except Exception as exc:
            st.warning(f"TradingView MCP module is not available: {exc}")
            return

        query = getattr(st, "query_params", {})
        code = _first_query_value(query, "code")
        state = _first_query_value(query, "state")
        if code and state:
            try:
                tv_mcp.exchange_code(code, state)
                _tradingview_mcp_status_snapshot.clear()
                st.success("TradingView OAuth authorization saved.")
                try:
                    st.query_params.clear()
                except Exception:
                    pass
            except Exception as exc:
                st.warning(f"TradingView OAuth callback failed: {exc}")

        status = _tradingview_mcp_status_snapshot()
        status_cols = st.columns(4)
        with status_cols[0]:
            st.metric("MCP Endpoint", "Reachable" if status.oauth_status != "ENDPOINT_ERROR" else "Error")
        with status_cols[1]:
            st.metric("OAuth", status.oauth_status)
        with status_cols[2]:
            st.metric("Tools", "OK" if status.connected else "n/a")
        with status_cols[3]:
            st.metric("Source Mode", "MCP_PRIMARY" if status.connected else "FALLBACK_SOURCE")
        st.caption(f"https://mcp.tradingview.com/mcp - {status.detail}")

        redirect_default = os.environ.get("TRADINGVIEW_MCP_REDIRECT_URI", "")
        redirect_uri = st.text_input(
            "TradingView OAuth redirect URL",
            value=redirect_default,
            placeholder="https://your-screener-url/",
            key="tradingview_mcp_redirect_uri",
        )
        auth_cols = st.columns([1.2, 1.2, 5.6])
        with auth_cols[0]:
            if st.button("Create Auth Link", key="tradingview_mcp_auth_link", use_container_width=True):
                if not redirect_uri:
                    st.warning("Set a public redirect URL first.")
                else:
                    try:
                        st.session_state["tradingview_mcp_auth_url"] = tv_mcp.build_authorization_url(redirect_uri)
                    except Exception as exc:
                        st.warning(f"Could not create TradingView auth link: {exc}")
        with auth_cols[1]:
            validate = st.button("Validate Symbols", key="tradingview_mcp_validate", use_container_width=True)
        auth_url = st.session_state.get("tradingview_mcp_auth_url")
        if auth_url:
            st.markdown(f"[Open TradingView authorization]({auth_url})")

        try:
            tools = tv_mcp.discover_tools(force=False) if status.connected else []
        except Exception:
            tools = []
        if tools:
            tool_names = ", ".join(str(tool.get("name", "")) for tool in tools)
            st.caption(f"Discovered tools: {tool_names}")

        if validate:
            _render_tradingview_mcp_validation(tv_mcp)


def _render_tradingview_mcp_validation(tv_mcp: Any) -> None:
    symbols = [
        "ECONOMICS:CNM2",
        "ECONOMICS:CNCBBS",
        "ECONOMICS:USBCOI",
        "TVC:DE10Y",
        "TVC:FR10Y",
        "TVC:CN10Y",
        "TVC:JP10Y",
    ]
    reports = []
    previews = []
    with st.spinner("Validating TradingView MCP symbols..."):
        for symbol in symbols:
            try:
                if symbol.startswith("TVC:"):
                    frame = tv_mcp.get_ohlcv_data(symbol, interval="1D", count=5000)
                    dates = pd.to_datetime(frame["date"], errors="coerce")
                    values = pd.to_numeric(frame["close"], errors="coerce")
                    reports.append(
                        {
                            "Symbol": symbol,
                            "Description": "TradingView bond yield OHLCV",
                            "Source": "TradingView MCP / get_ohlcv",
                            "Unit": "percentage points",
                            "Scale": "close",
                            "Frequency": "daily",
                            "First available date": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else "n/a",
                            "Last available date": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else "n/a",
                            "Latest value": values.iloc[-1] if not values.dropna().empty else np.nan,
                            "Number of observations": int(values.notna().sum()),
                            "Null count": int(values.isna().sum()),
                            "Duplicate date count": int(dates.duplicated().sum()),
                            "Data Status": "OK" if not values.dropna().empty else "MISSING",
                            "Validation": "OK" if len(values.dropna()) >= 24 else "INSUFFICIENT_HISTORY",
                        }
                    )
                    sample = frame.head(5).copy()
                    sample = pd.concat([sample, frame.tail(5)], ignore_index=True)
                    sample.insert(0, "symbol", symbol)
                    previews.append(sample)
                    continue
                result = tv_mcp.get_economic_data(symbol, date_from="2010-01-01", force=True)
                valid, status = tv_mcp.validate_economic_result(result, min_observations=24, max_stale_days=120)
                summary = tv_mcp.validation_summary(result)
                summary["Validation"] = "OK" if valid else status
                reports.append(summary)
                sample = pd.concat([result.frame.head(5), result.frame.tail(5)], ignore_index=True)
                sample.insert(0, "symbol", symbol)
                previews.append(sample)
            except Exception as exc:
                reports.append(
                    {
                        "Symbol": symbol,
                        "Description": "n/a",
                        "Source": "TradingView MCP",
                        "Unit": "n/a",
                        "Scale": "n/a",
                        "Frequency": "n/a",
                        "First available date": "n/a",
                        "Last available date": "n/a",
                        "Latest value": np.nan,
                        "Number of observations": 0,
                        "Null count": 0,
                        "Duplicate date count": 0,
                        "Data Status": "ERROR",
                        "Validation": str(exc),
                    }
                )
    st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
    if previews:
        preview = pd.concat(previews, ignore_index=True)
        st.dataframe(preview, use_container_width=True, hide_index=True)


def _first_query_value(query: Any, key: str) -> str:
    try:
        value = query.get(key)
    except Exception:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


@st.cache_data(show_spinner=False, ttl=300)
def _tradingview_mcp_status_snapshot() -> Any:
    import tradingview_mcp as tv_mcp

    return tv_mcp.connection_status()


@st.cache_data(show_spinner=True, ttl=SLOW_REFRESH_SECONDS)
def load_global_macro_snapshot(api_key_signature: str = "", refresh_nonce: int = 0) -> pd.DataFrame:
    _ = refresh_nonce
    api_key = api_key_signature or None
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    raw_liquidity, monthly_liquidity, weekly_liquidity = read_global_liquidity()
    fred = _download_fred_macro_series(api_key)
    market = _download_market_macro_series()
    specs: list[MacroSeriesSpec] = []
    specs.extend(_market_specs(market))
    specs.extend(_global_liquidity_specs(raw_liquidity, monthly_liquidity, weekly_liquidity))
    specs.extend(_fx_specs(raw_liquidity, market))
    specs.extend(_inflation_specs(fred))
    specs.extend(_rates_specs(fred, weekly_liquidity))
    specs.extend(_growth_specs(fred, market))
    specs.extend(_risk_specs(fred, market))
    rows = [_series_row(spec, now) for spec in specs]
    return pd.DataFrame(rows)


def _download_fred_macro_series(api_key: str | None) -> dict[str, pd.Series]:
    series_ids = [
        "T5YIE",
        "T10YIE",
        "MICH",
        "DFII10",
        "FEDFUNDS",
        "DGS2",
        "DGS10",
        "DGS3MO",
        "IRLTLT01DEM156N",
        "IRLTLT01FRM156N",
        "IRLTLT01CNM156N",
        "IRLTLT01JPM156N",
        "NAPM",
        "RSAFS",
        "NMFCI",
        "ICSA",
        "CFNAI",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "ANFCI",
    ]
    out: dict[str, pd.Series] = {}
    if not api_key:
        return {series_id: pd.Series(dtype="float64") for series_id in series_ids}
    for series_id in series_ids:
        try:
            frame = download_fred_series(series_id, api_key=api_key, observation_start="2010-01-01")
            out[series_id] = _series_from_frame(frame, "Date", "Value")
        except (FredApiError, Exception):
            out[series_id] = pd.Series(dtype="float64")
    return out


def _download_market_macro_series() -> dict[str, pd.Series]:
    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "GLD": "GLD",
        "BTC-USD": "BTC-USD",
        "DXY": "DX-Y.NYB",
        "WTI": "CL=F",
        "Copper": "HG=F",
        "VIX": "^VIX",
        "MOVE": "^MOVE",
    }
    out: dict[str, pd.Series] = {}
    for key, ticker in tickers.items():
        try:
            frame = download_completed_ohlcv(ticker, period="max")
            close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            out[key] = close.sort_index()
        except Exception:
            out[key] = pd.Series(dtype="float64")
    return out


def _market_specs(market: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    return [
        _market_spec("S&P 500 ETF", "SPY", market.get("SPY")),
        _market_spec("Nasdaq-100 ETF", "QQQ", market.get("QQQ")),
        _market_spec("Gold ETF", "GLD", market.get("GLD")),
        _market_spec("Bitcoin", "BTC-USD", market.get("BTC-USD")),
    ]


def _market_spec(instrument: str, ticker: str, series: pd.Series | None) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Markets",
        instrument=instrument,
        source=ticker,
        change_type="percent",
        current_format="number",
        frequency="daily",
        unit="price",
        series=_clean_series(series),
    )


def _global_liquidity_specs(raw: pd.DataFrame, monthly: pd.DataFrame, weekly: pd.DataFrame) -> list[MacroSeriesSpec]:
    monthly = _prepare_date_frame(monthly)
    weekly = _prepare_date_frame(weekly)
    raw = raw.copy() if raw is not None else pd.DataFrame()
    specs = [
        _liquidity_spec("Global M2 Money Supply", "Derived / US + EA + China + Japan", monthly, "global_m2_usd_bn", "monthly"),
        _liquidity_spec("United States M2 Money Supply", "FRED / M2SL", monthly, "us_m2_usd_bn", "monthly"),
        _liquidity_spec("Euro Area M2 Money Supply", "ECB Data API", monthly, "ea_m2_usd_bn", "monthly"),
        _liquidity_spec("China M2 Money Supply", _source_label(raw, "Money & Quasi-money (M2)", "Fallback / China M2"), monthly, "china_m2_usd_bn", "monthly"),
        _liquidity_spec("Japan M2 Money Supply", "BoJ Time-Series API", monthly, "japan_m2_usd_bn", "monthly"),
    ]
    regime = _global_liquidity_monitor_frame(monthly, weekly)
    specs.extend(
        [
            _absolute_spec("Global Liquidity Direction 13W", "Derived / score delta 13W", regime, "direction_13w", "score points"),
            _score_spec("Global Liquidity Score", "Derived / M2 + CB + USNL impulses", regime, "global_liquidity_score"),
            _liquidity_spec("Global Central Bank Assets", "Derived / Fed + ECB + BoJ + PBoC", monthly, "global_cb_assets_usd_bn", "monthly"),
            _liquidity_spec("Federal Reserve Total Assets", "FRED / WALCL", monthly, "fed_assets_usd_bn", "monthly"),
            _liquidity_spec("European Central Bank Total Assets", "ECB Data API / ILM", monthly, "ecb_assets_usd_bn", "monthly"),
            _liquidity_spec("Bank of Japan Total Assets", "BoJ Time-Series API", monthly, "boj_assets_usd_bn", "monthly"),
            _liquidity_spec("People's Bank of China Total Assets", _source_label(raw, "PBOC_TOTAL_ASSETS", "PBoC / TradingView fallback"), monthly, "pboc_assets_usd_bn", "monthly"),
            _liquidity_spec("US Net Liquidity", "FRED / WALCL - WTREGEN - RRPONTSYD", weekly, "us_net_liquidity_usd_bn", "weekly"),
            _liquidity_spec("U.S. Bank Reserves", "FRED / WRESBAL", weekly, "US_BankReserves", "weekly"),
            _absolute_spec("SOFR", "FRED / SOFR", weekly, "SOFR", "%", block="Funding Conditions"),
            _absolute_spec("EFFR", "FRED / EFFR", weekly, "EFFR", "%", block="Funding Conditions"),
            _absolute_spec("SOFR-EFFR Spread", "Derived / SOFR - EFFR", weekly, "SOFR_EFFR_Spread", "percentage points", block="Funding Conditions"),
            _state_spec("Global M2 Trend", "Derived / 13W + 26W + 52W trend", regime, "trend_state"),
            _state_spec("Long Liquidity Cycle", "Derived / 65M reference cycle", regime, "long_cycle_phase"),
        ]
    )
    return specs


def _liquidity_spec(instrument: str, source: str, frame: pd.DataFrame, column: str, frequency: str) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Global Liquidity",
        instrument=instrument,
        source=source,
        change_type="percent",
        current_format="usd_tn",
        frequency=frequency,
        unit="bn USD",
        series=_frame_series(frame, column),
        data_status=_frame_status(frame, column),
    )


def _score_spec(instrument: str, source: str, frame: pd.DataFrame, column: str) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Global Liquidity",
        instrument=instrument,
        source=source,
        change_type="absolute",
        current_format="number",
        frequency="weekly",
        unit="0-100",
        series=_frame_series(frame, column),
        data_status=_frame_status(frame, column),
    )


def _absolute_spec(
    instrument: str,
    source: str,
    frame: pd.DataFrame,
    column: str,
    unit: str,
    block: str = "Global Liquidity",
) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block=block,
        instrument=instrument,
        source=source,
        change_type="absolute",
        current_format="number",
        frequency="weekly",
        unit=unit,
        series=_frame_series(frame, column),
        data_status=_frame_status(frame, column),
    )


def _state_spec(instrument: str, source: str, frame: pd.DataFrame, column: str) -> MacroSeriesSpec:
    if frame.empty or column not in frame.columns:
        series = pd.Series(dtype="object")
    else:
        series = frame.set_index("date")[column].dropna().sort_index()
    return MacroSeriesSpec(
        block="Global Liquidity",
        instrument=instrument,
        source=source,
        change_type="state",
        current_format="number",
        frequency="weekly",
        unit="state",
        series=series,
        data_status="OK" if not series.empty else "MISSING",
    )


def _fx_specs(raw: pd.DataFrame, market: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    return [
        MacroSeriesSpec("FX / USD", "U.S. Dollar Index", "DXY / DX-Y.NYB", "percent", "number", "daily", "index", _clean_series(market.get("DXY"))),
        MacroSeriesSpec("FX / USD", "EUR/USD", "FRED / DEXUSEU", "percent", "number", "daily", "rate", _raw_series(raw, "DEXUSEU")),
        MacroSeriesSpec("FX / USD", "USD/JPY", "FRED / DEXJPUS", "percent", "number", "daily", "rate", _raw_series(raw, "DEXJPUS")),
        MacroSeriesSpec("FX / USD", "USD/CNY", "FRED / DEXCHUS", "percent", "number", "daily", "rate", _raw_series(raw, "DEXCHUS")),
    ]


def _inflation_specs(fred: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    return [
        _fred_bps_spec("Inflation", "U.S. 5-Year Breakeven Inflation Rate", "FRED / T5YIE", fred.get("T5YIE")),
        _fred_bps_spec("Inflation", "U.S. 10-Year Breakeven Inflation Rate", "FRED / T10YIE", fred.get("T10YIE")),
        _fred_bps_spec("Inflation", "U.S. 1-Year Inflation Expectations", "FRED / MICH", fred.get("MICH")),
        _fred_bps_spec("Inflation", "U.S. 10-Year Real Yield", "FRED / DFII10", fred.get("DFII10")),
    ]


def _rates_specs(fred: dict[str, pd.Series], weekly: pd.DataFrame | None = None) -> list[MacroSeriesSpec]:
    us2y = _clean_series(fred.get("DGS2"))
    us10y = _clean_series(fred.get("DGS10"))
    us3m = _clean_series(fred.get("DGS3MO"))
    international_rates = [
        ("Germany 10-Year Government Bond Yield", "GE10Y", "TVC:DE10Y", "FRED / IRLTLT01DEM156N", fred.get("IRLTLT01DEM156N")),
        ("France 10-Year Government Bond Yield", "FR10Y", "TVC:FR10Y", "FRED / IRLTLT01FRM156N", fred.get("IRLTLT01FRM156N")),
        ("China 10-Year Government Bond Yield", "CN10Y", "TVC:CN10Y", "FRED / IRLTLT01CNM156N", fred.get("IRLTLT01CNM156N")),
        ("Japan 10-Year Government Bond Yield", "JP10Y", "TVC:JP10Y", "FRED / IRLTLT01JPM156N", fred.get("IRLTLT01JPM156N")),
    ]
    international_specs: list[MacroSeriesSpec] = []
    for instrument, alias, symbol, fallback_source, fallback_series in international_rates:
        series, source, status = _tradingview_rate_series_or_fallback(symbol, alias, fallback_source, fallback_series)
        international_specs.append(_rate_spec(instrument, source, series, status))
    return [
        _fred_bps_spec("Rates & Curves", "Federal Funds Effective Rate", "FRED / FEDFUNDS", fred.get("FEDFUNDS")),
        _fred_bps_spec("Rates & Curves", "U.S. 2-Year Treasury Yield", "FRED / DGS2", us2y),
        _fred_bps_spec("Rates & Curves", "U.S. 10-Year Treasury Yield", "FRED / DGS10", us10y),
        _absolute_spec("U.S. 10Y Term Premium", "FRED / THREEFYTP10", weekly if weekly is not None else pd.DataFrame(), "US10Y_TermPremium", "percentage points", block="Rates & Curves"),
        _spread_spec("U.S. 2Y-10Y Treasury Curve", "Derived / DGS10 - DGS2", us10y - us2y),
        _spread_spec("U.S. 3M-10Y Treasury Curve", "Derived / DGS10 - DGS3MO", us10y - us3m),
        *international_specs,
    ]


def _rate_spec(instrument: str, source: str, series: pd.Series | None, status: str) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Rates & Curves",
        instrument=instrument,
        source=source,
        change_type="bps",
        current_format="percent",
        frequency="daily/monthly",
        unit="percentage points",
        series=_clean_series(series),
        data_status=status,
    )


def _tradingview_rate_series_or_fallback(
    symbol: str,
    alias: str,
    fallback_source: str,
    fallback_series: pd.Series | None,
) -> tuple[pd.Series, str, str]:
    try:
        from tradingview_mcp import get_ohlcv_data

        frame = get_ohlcv_data(symbol, interval="1D", count=5000)
        series = _clean_series(
            pd.Series(
                pd.to_numeric(frame["close"], errors="coerce").values,
                index=pd.to_datetime(frame["date"], errors="coerce"),
            )
        )
        latest = _latest_index(series)
        age_days = (pd.Timestamp.now(tz="UTC").tz_localize(None) - latest).days if latest is not None else 9999
        if len(series) < 24 or age_days > 10:
            raise RuntimeError(f"insufficient or stale OHLCV data: n={len(series)}, age_days={age_days}")
        return series, f"TradingView MCP / {symbol} ({alias})", "OK"
    except Exception:
        series = _clean_series(fallback_series)
        return series, f"{fallback_source} fallback", "FALLBACK_SOURCE" if not series.empty else "MISSING"


def _growth_specs(fred: dict[str, pd.Series], market: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    ism_manufacturing, ism_manufacturing_source, ism_manufacturing_status = _tradingview_economic_series_or_fallback(
        "ECONOMICS:USBCOI",
        "FRED / NAPM",
        fred.get("NAPM"),
    )
    ism_services, ism_services_source, ism_services_status = _investing_ism_services_series_or_fallback(fred.get("NMFCI"))
    return [
        MacroSeriesSpec("Growth / Business Cycle", "U.S. ISM Manufacturing PMI", ism_manufacturing_source, "absolute", "number", "monthly", "index", ism_manufacturing, ism_manufacturing_status),
        MacroSeriesSpec("Growth / Business Cycle", "U.S. ISM Services PMI", ism_services_source, "absolute", "number", "monthly", "index", ism_services, ism_services_status),
        MacroSeriesSpec("Growth / Business Cycle", "U.S. Initial Jobless Claims", "FRED / ICSA", "percent", "integer", "weekly", "claims", _clean_series(fred.get("ICSA"))),
        MacroSeriesSpec("Growth / Business Cycle", "Chicago Fed National Activity Index", "FRED / CFNAI", "absolute", "number", "monthly", "index", _clean_series(fred.get("CFNAI"))),
        MacroSeriesSpec("Growth / Business Cycle", "WTI Crude Oil", "Yahoo / CL=F", "percent", "number", "daily", "price", _clean_series(market.get("WTI"))),
        MacroSeriesSpec("Growth / Business Cycle", "Copper", "Yahoo / HG=F", "percent", "number", "daily", "price", _clean_series(market.get("Copper"))),
    ]


def _investing_ism_series_or_fallback(fallback_series: pd.Series | None) -> tuple[pd.Series, str, str]:
    try:
        from business_cycle import load_investing_pmi_releases, load_pmi_release_fallback

        history = load_pmi_release_fallback("2010-01-01")
        live = load_investing_pmi_releases("2010-01-01")
        history_series = pd.Series(
            pd.to_numeric(history["Value"], errors="coerce").values,
            index=pd.to_datetime(history["Date"], errors="coerce"),
        ).dropna()
        live_series = pd.Series(
            pd.to_numeric(live["Value"], errors="coerce").values,
            index=pd.to_datetime(live["Date"], errors="coerce"),
        ).dropna()
        history_series.index = pd.to_datetime(history_series.index).tz_localize(None)
        live_series.index = pd.to_datetime(live_series.index).tz_localize(None)
        combined = pd.concat([_clean_series(fallback_series), _clean_series(history_series), _clean_series(live_series)]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if not combined.empty:
            return combined, "Investing.com / ISM Manufacturing PMI + preserved history", "OK"
    except Exception:
        pass
    series = _clean_series(fallback_series)
    return series, "FRED / NAPM fallback", "FALLBACK_SOURCE" if not series.empty else "MISSING"


def _investing_ism_services_series_or_fallback(fallback_series: pd.Series | None) -> tuple[pd.Series, str, str]:
    try:
        from business_cycle import load_investing_ism_services_releases

        live = load_investing_ism_services_releases("2010-01-01")
        live_series = pd.Series(
            pd.to_numeric(live["Value"], errors="coerce").values,
            index=pd.to_datetime(live["Date"], errors="coerce"),
        ).dropna()
        live_series.index = pd.to_datetime(live_series.index).tz_localize(None)
        combined = pd.concat([_clean_series(fallback_series), _clean_series(live_series)]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if not combined.empty:
            return combined, "Investing.com / ISM Services PMI + preserved history", "OK"
    except Exception:
        pass
    series = _clean_series(fallback_series)
    return series, "FRED / NMFCI fallback", "FALLBACK_SOURCE" if not series.empty else "MISSING"


def _tradingview_economic_series_or_fallback(
    symbol: str,
    fallback_source: str,
    fallback_series: pd.Series | None,
) -> tuple[pd.Series, str, str]:
    try:
        from tradingview_mcp import get_economic_data, validate_economic_result

        result = get_economic_data(symbol, date_from="2010-01-01")
        valid, status = validate_economic_result(result, min_observations=24, max_stale_days=120)
        if valid and not result.frame.empty:
            series = _clean_series(
                pd.Series(
                    pd.to_numeric(result.frame["value"], errors="coerce").values,
                    index=pd.to_datetime(result.frame["date"], errors="coerce"),
                )
            )
            return series, f"TradingView MCP / {symbol}", "OK"
        raise RuntimeError(status)
    except Exception:
        series = _clean_series(fallback_series)
        return series, f"{fallback_source} fallback", "FALLBACK_SOURCE" if not series.empty else "MISSING"


def _risk_specs(fred: dict[str, pd.Series], market: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    return [
        MacroSeriesSpec("Risk / Financial Conditions", "CBOE Volatility Index", "Yahoo / ^VIX", "percent", "number", "daily", "index", _clean_series(market.get("VIX"))),
        MacroSeriesSpec("Risk / Financial Conditions", "ICE BofA MOVE Index", "Yahoo / ^MOVE", "percent", "number", "daily", "index", _clean_series(market.get("MOVE"))),
        _fred_bps_spec("Risk / Financial Conditions", "U.S. High Yield Option-Adjusted Spread", "FRED / BAMLH0A0HYM2", fred.get("BAMLH0A0HYM2"), current_format="bps"),
        _fred_bps_spec("Risk / Financial Conditions", "U.S. Investment Grade Option-Adjusted Spread", "FRED / BAMLC0A0CM", fred.get("BAMLC0A0CM"), current_format="bps"),
        MacroSeriesSpec("Risk / Financial Conditions", "Chicago Fed Adjusted National Financial Conditions Index", "FRED / ANFCI", "absolute", "number", "weekly", "index", _clean_series(fred.get("ANFCI"))),
    ]


def _fred_bps_spec(
    block: str,
    instrument: str,
    source: str,
    series: pd.Series | None,
    current_format: CurrentFormat = "percent",
) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block=block,
        instrument=instrument,
        source=source,
        change_type="bps",
        current_format=current_format,
        frequency="daily/monthly",
        unit="percentage points",
        series=_clean_series(series),
    )


def _spread_spec(instrument: str, source: str, series: pd.Series) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Rates & Curves",
        instrument=instrument,
        source=source,
        change_type="bps",
        current_format="bps",
        frequency="daily",
        unit="percentage points",
        series=_clean_series(series),
    )


def _series_row(spec: MacroSeriesSpec, now: pd.Timestamp) -> dict[str, str]:
    series = _clean_series(spec.series)
    latest_date = _latest_index(series)
    latest_value = _latest_value(series)
    status = _series_status(spec, latest_date, now)
    row = {
        "Block": spec.block,
        "Instrument": spec.instrument,
        "Current": _format_state(latest_value) if spec.change_type == "state" else _format_current(latest_value, spec.current_format),
        "1W": "n/a",
        "1M": "n/a",
        "3M": "n/a",
        "6M": "n/a",
        "12M": "n/a",
        "36M": "n/a",
        "Source": spec.source,
        "Frequency": spec.frequency,
        "Unit": spec.unit,
        "Last Updated": latest_date.strftime("%Y-%m-%d") if latest_date is not None else "n/a",
        "Data Status": status,
    }
    for label, weeks in HORIZON_WEEKS.items():
        if spec.change_type == "state":
            row[label] = _format_state(_value_on_or_before(series, latest_date - pd.DateOffset(weeks=weeks) if latest_date is not None else None))
        else:
            previous = _value_on_or_before(series, latest_date - pd.DateOffset(weeks=weeks) if latest_date is not None else None)
            row[label] = _format_change(latest_value, previous, spec.change_type)
    return row


def _render_global_macro_summary(snapshot: pd.DataFrame) -> None:
    wanted = [
        "Global M2 Money Supply",
        "Global Central Bank Assets",
        "Global Liquidity Score",
        "U.S. Dollar Index",
        "U.S. 2-Year Treasury Yield",
        "U.S. 10-Year Treasury Yield",
        "U.S. 10-Year Real Yield",
        "U.S. ISM Manufacturing PMI",
        "U.S. ISM Services PMI",
        "U.S. High Yield Option-Adjusted Spread",
        "CBOE Volatility Index",
    ]
    rows = snapshot[snapshot["Instrument"].isin(wanted)].copy()
    rows["_order"] = rows["Instrument"].map({name: idx for idx, name in enumerate(wanted)})
    rows = rows.sort_values("_order").head(12)
    if rows.empty:
        return
    st.markdown("### Summary")
    cols = st.columns(4)
    for idx, row in enumerate(rows.to_dict("records")):
        with cols[idx % 4]:
            st.markdown(
                f"""
<div style="padding:0.55rem 0; line-height:1.15;">
  <div style="font-size:0.70rem; color:#94a3b8; font-weight:700;">{row['Instrument']}</div>
  <div style="font-size:1.05rem; color:#f8fafc; font-weight:800;">{row['Current']}</div>
  <div style="font-size:0.70rem; color:#cbd5e1;">1M {row['1M']} | {row['Data Status']}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def _style_global_macro_table(frame: pd.DataFrame, block: str = "") -> Any:
    return frame.style.set_properties(
        **{
            "background-color": "#0f131a",
            "color": "#e5e7eb",
            "border-color": "#263241",
        }
    ).apply(lambda row: _macro_change_color_row(row, block), axis=1)


def _global_macro_table_html(frame: pd.DataFrame, block: str = "", widths: dict[str, int] | None = None) -> str:
    widths = widths or {
        "instrument": _global_macro_column_width(frame, "Instrument"),
        "source": _global_macro_column_width(frame, "Source"),
    }
    instrument_width = widths["instrument"]
    source_width = widths["source"]
    styled = _style_global_macro_table(frame, block).hide(axis="index").set_table_attributes('class="global-macro-table"')
    table = styled.to_html()
    return (
        f'<div class="global-macro-table-wrap" '
        f'style="--instrument-width:{instrument_width}px; --source-width:{source_width}px;">'
        f'<style>'
        f'.global-macro-table-wrap table.global-macro-table th:nth-child(1), '
        f'.global-macro-table-wrap table.global-macro-table td:nth-child(1) '
        f'{{width:var(--instrument-width) !important;}} '
        f'.global-macro-table-wrap table.global-macro-table th:nth-child(9), '
        f'.global-macro-table-wrap table.global-macro-table td:nth-child(9) '
        f'{{width:var(--source-width) !important;}}'
        f'</style>{table}</div>'
    )


def _global_macro_column_width(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 110
    values = [str(column)] + frame[column].fillna("n/a").astype(str).tolist()
    longest = max((len(value) for value in values), default=12)
    return max(110, longest * 7 + 22)


def _macro_change_color_row(row: pd.Series, block: str = "") -> list[str]:
    direction = _macro_direction(row, block)
    styles = [""] * len(row)
    instrument = str(row.get("Instrument", ""))
    if instrument in {
        "United States M2 Money Supply",
        "Euro Area M2 Money Supply",
        "China M2 Money Supply",
        "Japan M2 Money Supply",
        "Federal Reserve Total Assets",
        "European Central Bank Total Assets",
        "Bank of Japan Total Assets",
        "People's Bank of China Total Assets",
    } and "Instrument" in row.index:
        styles[row.index.get_loc("Instrument")] = "text-align: right;"
    if direction == 0:
        return styles
    for idx, column in enumerate(row.index):
        if column not in CHANGE_COLUMNS:
            continue
        value = _parse_change_value(row.get(column))
        if not np.isfinite(value):
            continue
        score = value * direction
        styles[idx] = _heatmap_style(score)
    return styles


def _macro_direction(row: pd.Series, block: str = "") -> int:
    block = str(block or row.get("Block", ""))
    instrument = str(row.get("Instrument", ""))
    if block in {"Markets", "Global Liquidity"}:
        return 1
    if "PMI" in instrument or "Chicago Fed National Activity Index" in instrument or instrument == "Copper":
        return 1
    if (
        block in {"Inflation", "Rates & Curves", "Risk / Financial Conditions"}
        or instrument in {"U.S. Dollar Index", "WTI Crude Oil", "U.S. Initial Jobless Claims"}
    ):
        return -1
    return 0


def _parse_change_value(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return math.nan
    text = text.replace(",", "").replace("%", "").replace("bp", "").strip()
    try:
        return float(text)
    except Exception:
        return math.nan


def _heatmap_style(score: float) -> str:
    if not np.isfinite(score):
        return ""
    if score >= 5.0:
        return "background-color: #16a34a; color: #ffffff"
    if score >= 2.0:
        return "background-color: #4ade80; color: #111827"
    if score > 0.0:
        return "background-color: #86efac; color: #111827"
    if score == 0.0:
        return "background-color: #f8fafc; color: #111827"
    if score > -2.0:
        return "background-color: #fdba74; color: #111827"
    return "background-color: #ef4444; color: #ffffff"


def _series_from_frame(frame: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    if frame is None or frame.empty or date_col not in frame.columns or value_col not in frame.columns:
        return pd.Series(dtype="float64")
    return _clean_series(pd.Series(pd.to_numeric(frame[value_col], errors="coerce").values, index=pd.to_datetime(frame[date_col], errors="coerce")))


def _prepare_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _frame_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or "date" not in frame.columns or column not in frame.columns:
        return pd.Series(dtype="float64")
    return _clean_series(pd.Series(pd.to_numeric(frame[column], errors="coerce").values, index=frame["date"]))


def _frame_status(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "MISSING"
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return "MISSING"
    status = str(frame["data_status"].dropna().iloc[-1]) if "data_status" in frame.columns and frame["data_status"].notna().any() else "OK"
    return "PARTIAL" if "PARTIAL" in status.upper() else "OK"


def _raw_series(raw: pd.DataFrame, series_id: str) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype="float64")
    d = raw[raw["series_id"].eq(series_id)].dropna(subset=["observation_date", "raw_value"]).copy()
    if d.empty:
        return pd.Series(dtype="float64")
    return _clean_series(pd.Series(pd.to_numeric(d["raw_value"], errors="coerce").values, index=pd.to_datetime(d["observation_date"], errors="coerce")))


def _source_label(raw: pd.DataFrame, series_id: str, fallback: str) -> str:
    if raw is None or raw.empty:
        return fallback
    source = raw[raw["series_id"].eq(series_id)].dropna(subset=["source_name"]).copy()
    if source.empty:
        return fallback
    return str(source.sort_values("observation_date")["source_name"].iloc[-1])


def _global_liquidity_monitor_frame(monthly: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty and weekly.empty:
        return pd.DataFrame(columns=["date", "global_liquidity_score", "direction_13w", "trend_state", "long_cycle_phase"])
    index = pd.DatetimeIndex([])
    if not weekly.empty:
        index = index.union(pd.DatetimeIndex(weekly["date"]))
    if not monthly.empty:
        month_dates = pd.DatetimeIndex(monthly["date"])
        end_date = index.max() if len(index) else month_dates.max()
        index = index.union(pd.date_range(month_dates.min(), end_date, freq="W-FRI"))
    index = index.sort_values()
    frame = pd.DataFrame(index=index)
    if not monthly.empty:
        monthly_indexed = monthly.set_index("date").sort_index()
        for source_col, target_col in [
            ("global_m2_usd_bn", "global_m2_usd_bn"),
            ("global_m2_impulse_3y_percentile", "m2_impulse"),
        ]:
            if source_col in monthly_indexed.columns:
                frame[target_col] = pd.to_numeric(monthly_indexed[source_col], errors="coerce").reindex(index).ffill()
    if not weekly.empty:
        weekly_indexed = weekly.set_index("date").sort_index()
        for source_col, target_col in [
            ("global_cb_impulse_3y_percentile", "cb_impulse"),
            ("us_net_liquidity_impulse_3y_percentile", "usnl_impulse"),
        ]:
            if source_col in weekly_indexed.columns:
                frame[target_col] = pd.to_numeric(weekly_indexed[source_col], errors="coerce").reindex(index).ffill()
    components = frame[["m2_impulse", "cb_impulse", "usnl_impulse"]].copy() if set(["m2_impulse", "cb_impulse", "usnl_impulse"]).issubset(frame.columns) else pd.DataFrame(index=index)
    if not components.empty:
        frame["global_liquidity_score"] = 0.50 * components["m2_impulse"] + 0.25 * components["cb_impulse"] + 0.25 * components["usnl_impulse"]
        frame["direction_13w"] = frame["global_liquidity_score"] - frame["global_liquidity_score"].shift(13)
    if "global_m2_usd_bn" in frame.columns:
        m2 = pd.to_numeric(frame["global_m2_usd_bn"], errors="coerce")
        trend_score = (
            0.50 * (m2.pct_change(13, fill_method=None) > 0).astype(float)
            + 0.30 * (m2.pct_change(26, fill_method=None) > 0).astype(float)
            + 0.20 * (m2.pct_change(52, fill_method=None) > 0).astype(float)
        )
        frame["trend_state"] = trend_score.map(_trend_state)
    frame["long_cycle_phase"] = [_long_cycle_phase(date) for date in frame.index]
    frame = frame.reset_index().rename(columns={"index": "date"})
    return frame


def _trend_state(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "DATA_INCOMPLETE"
    if not np.isfinite(numeric):
        return "DATA_INCOMPLETE"
    if numeric >= 0.8:
        return "BROAD_EXPANSION"
    if numeric >= 0.5:
        return "FLAT_MIXED"
    if numeric >= 0.2:
        return "BROAD_DECELERATION"
    return "CONTRACTION"


def _long_cycle_phase(value: Any) -> str:
    date = pd.Timestamp(value).to_period("M").to_timestamp()
    anchor = pd.Timestamp("2022-10-01")
    months = (date.year - anchor.year) * 12 + date.month - anchor.month
    phase = months % 65
    if phase < 16:
        return "RECOVERY_REACCELERATION"
    if phase < 33:
        return "ACCELERATING_EXPANSION"
    if phase < 49:
        return "DECELERATING_EXPANSION"
    return "CONTRACTION"


def _clean_series(series: pd.Series | None) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")
    out = series.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[pd.notna(out.index)]
    out = out.sort_index()
    if out.dtype == object:
        return out.dropna()
    return pd.to_numeric(out, errors="coerce").dropna()


def _latest_index(series: pd.Series) -> pd.Timestamp | None:
    if series.empty:
        return None
    return pd.Timestamp(series.index[-1])


def _latest_value(series: pd.Series) -> Any:
    if series.empty:
        return np.nan
    return series.iloc[-1]


def _value_on_or_before(series: pd.Series, date: pd.Timestamp | None) -> Any:
    if series.empty or date is None:
        return np.nan
    values = series.loc[:date].dropna()
    if values.empty:
        return np.nan
    return values.iloc[-1]


def _format_current(value: Any, current_format: CurrentFormat) -> str:
    number = _to_float(value)
    if current_format == "number":
        return "n/a" if not np.isfinite(number) else f"{number:,.2f}"
    if current_format == "percent":
        return "n/a" if not np.isfinite(number) else f"{number:.2f}%"
    if current_format == "bps":
        return "n/a" if not np.isfinite(number) else f"{number * 100:.0f} bp"
    if current_format == "usd_tn":
        return "n/a" if not np.isfinite(number) else f"${number / 1000.0:,.2f}T"
    if current_format == "usd_bn":
        return "n/a" if not np.isfinite(number) else f"${number:,.0f}B"
    if current_format == "integer":
        return "n/a" if not np.isfinite(number) else f"{number:,.0f}"
    return str(value)


def _format_change(current: Any, previous: Any, change_type: ChangeType) -> str:
    cur = _to_float(current)
    prev = _to_float(previous)
    if not np.isfinite(cur) or not np.isfinite(prev):
        return "n/a"
    if change_type == "percent":
        if prev == 0:
            return "n/a"
        return f"{(cur / prev - 1.0) * 100.0:+.1f}%"
    if change_type == "bps":
        return f"{cur - prev:+.2f}%"
    if change_type == "absolute":
        return f"{cur - prev:+.2f}"
    return "n/a"


def _format_state(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value if value else "n/a"
    if pd.isna(value):
        return "n/a"
    return str(value)


def _series_status(spec: MacroSeriesSpec, latest_date: pd.Timestamp | None, now: pd.Timestamp) -> str:
    if spec.data_status not in {"OK", "CURRENT"}:
        return spec.data_status
    if latest_date is None:
        return "MISSING"
    age_days = max(0, (now - latest_date).days)
    stale_limit = {
        "daily": 10,
        "weekly": 24,
        "monthly": 75,
        "daily/monthly": 75,
    }.get(spec.frequency, 75)
    status = "STALE" if age_days > stale_limit else "OK"
    if "TradingView" in spec.source or "fallback" in spec.source or "Fallback" in spec.source:
        status = "FALLBACK_SOURCE" if status == "OK" else status
    return status


def _to_float(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except Exception:
        return math.nan

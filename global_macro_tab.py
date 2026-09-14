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
    if "global_macro_refresh_nonce" not in st.session_state:
        st.session_state["global_macro_refresh_nonce"] = 0
    controls = st.columns([1.2, 5.8])
    with controls[0]:
        force_refresh = st.button("Refresh Global Macro", use_container_width=True, key="global_macro_refresh")
    with controls[1]:
        st.markdown(
            f"<div style='padding-top:1.55rem; color:#94a3b8; font-size:0.78rem;'>"
            f"Liquidity storage: {GLOBAL_LIQUIDITY_STORAGE_DIR}</div>",
            unsafe_allow_html=True,
        )
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

    _render_global_macro_summary(snapshot)
    for block in [
        "Markets",
        "Global Liquidity",
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
        st.dataframe(_style_global_macro_table(block_df, block), use_container_width=True, hide_index=True)


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
    symbols = ["ECONOMICS:CNM2", "ECONOMICS:CNCBBS", "ECONOMICS:USBCOI", "ECONOMICS:USNMPMI"]
    reports = []
    previews = []
    with st.spinner("Validating TradingView economic symbols..."):
        for symbol in symbols:
            try:
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
    specs.extend(_rates_specs(fred))
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
        _liquidity_spec("Global Central Bank Assets", "Derived / Fed + ECB + BoJ + PBoC", monthly, "global_cb_assets_usd_bn", "monthly"),
        _liquidity_spec("Federal Reserve Total Assets", "FRED / WALCL", monthly, "fed_assets_usd_bn", "monthly"),
        _liquidity_spec("European Central Bank Total Assets", "ECB Data API / ILM", monthly, "ecb_assets_usd_bn", "monthly"),
        _liquidity_spec("Bank of Japan Total Assets", "BoJ Time-Series API", monthly, "boj_assets_usd_bn", "monthly"),
        _liquidity_spec("People's Bank of China Total Assets", _source_label(raw, "PBOC_TOTAL_ASSETS", "PBoC / TradingView fallback"), monthly, "pboc_assets_usd_bn", "monthly"),
        _liquidity_spec("US Net Liquidity", "FRED / WALCL - WTREGEN - RRPONTSYD", weekly, "us_net_liquidity_usd_bn", "weekly"),
    ]
    regime = _global_liquidity_monitor_frame(monthly, weekly)
    specs.extend(
        [
            _score_spec("Global Liquidity Score", "Derived / M2 + CB + USNL impulses", regime, "global_liquidity_score"),
            _absolute_spec("Global Liquidity Direction 13W", "Derived / score delta 13W", regime, "direction_13w", "score points"),
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


def _absolute_spec(instrument: str, source: str, frame: pd.DataFrame, column: str, unit: str) -> MacroSeriesSpec:
    return MacroSeriesSpec(
        block="Global Liquidity",
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


def _rates_specs(fred: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    us2y = _clean_series(fred.get("DGS2"))
    us10y = _clean_series(fred.get("DGS10"))
    us3m = _clean_series(fred.get("DGS3MO"))
    return [
        _fred_bps_spec("Rates & Curves", "Federal Funds Effective Rate", "FRED / FEDFUNDS", fred.get("FEDFUNDS")),
        _fred_bps_spec("Rates & Curves", "U.S. 2-Year Treasury Yield", "FRED / DGS2", us2y),
        _fred_bps_spec("Rates & Curves", "U.S. 10-Year Treasury Yield", "FRED / DGS10", us10y),
        _spread_spec("U.S. 2Y-10Y Treasury Curve", "Derived / DGS10 - DGS2", us10y - us2y),
        _spread_spec("U.S. 3M-10Y Treasury Curve", "Derived / DGS10 - DGS3MO", us10y - us3m),
        _fred_bps_spec("Rates & Curves", "Germany 10-Year Government Bond Yield", "FRED / IRLTLT01DEM156N", fred.get("IRLTLT01DEM156N")),
        _fred_bps_spec("Rates & Curves", "France 10-Year Government Bond Yield", "FRED / IRLTLT01FRM156N", fred.get("IRLTLT01FRM156N")),
        _fred_bps_spec("Rates & Curves", "China 10-Year Government Bond Yield", "FRED / IRLTLT01CNM156N", fred.get("IRLTLT01CNM156N")),
        _fred_bps_spec("Rates & Curves", "Japan 10-Year Government Bond Yield", "FRED / IRLTLT01JPM156N", fred.get("IRLTLT01JPM156N")),
    ]


def _growth_specs(fred: dict[str, pd.Series], market: dict[str, pd.Series]) -> list[MacroSeriesSpec]:
    ism_manufacturing, ism_manufacturing_source, ism_manufacturing_status = _tradingview_economic_series_or_fallback(
        "ECONOMICS:USBCOI",
        "FRED / NAPM",
        fred.get("NAPM"),
    )
    ism_services, ism_services_source, ism_services_status = _tradingview_economic_series_or_fallback(
        "ECONOMICS:USNMPMI",
        "FRED / NMFCI fallback",
        fred.get("NMFCI"),
    )
    return [
        MacroSeriesSpec("Growth / Business Cycle", "U.S. ISM Manufacturing PMI", ism_manufacturing_source, "absolute", "number", "monthly", "index", ism_manufacturing, ism_manufacturing_status),
        MacroSeriesSpec("Growth / Business Cycle", "U.S. ISM Services PMI", ism_services_source, "absolute", "number", "monthly", "index", ism_services, ism_services_status),
        MacroSeriesSpec("Growth / Business Cycle", "U.S. Initial Jobless Claims", "FRED / ICSA", "percent", "integer", "weekly", "claims", _clean_series(fred.get("ICSA"))),
        MacroSeriesSpec("Growth / Business Cycle", "Chicago Fed National Activity Index", "FRED / CFNAI", "absolute", "number", "monthly", "index", _clean_series(fred.get("CFNAI"))),
        MacroSeriesSpec("Growth / Business Cycle", "WTI Crude Oil", "Yahoo / CL=F", "percent", "number", "daily", "price", _clean_series(market.get("WTI"))),
        MacroSeriesSpec("Growth / Business Cycle", "Copper", "Yahoo / HG=F", "percent", "number", "daily", "price", _clean_series(market.get("Copper"))),
    ]


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


def _macro_change_color_row(row: pd.Series, block: str = "") -> list[str]:
    direction = _macro_direction(row, block)
    styles = [""] * len(row)
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

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest.engine import BacktestConfig, ConditionRow, StrategyRules, run_backtest
from backtest.io import (
    detect_date_candidates,
    ensure_columns_numeric,
    find_numeric_columns,
    prepare_timeseries_df,
    read_uploaded_file,
)
from backtest.metrics import drawdown_series, headline_metrics, yearly_table

st.set_page_config(page_title="Strategy Backtester MVP", layout="wide")
st.title("Trading Strategy Backtesting MVP")

CONDITIONS = ["=", ">", "<", ">=", "<=", "bear divergence", "bull divergence"]
ROWS_PER_STRATEGY = 5
STORE_PATH = Path("data/saved_strategies.json")
APP_MODES = ["Excel Data", "Live Data (Yahoo Finance)"]
TIMEFRAMES = ["Daily", "Weekly"]
LIVE_INDICATOR_VARS = [
    "Low vs MA50",
    "High vs MA50",
    "Low vs MA200",
    "High vs MA200",
    "RSI",
    "MACD",
    "ROC",
]
LIVE_PRICE_VARS = ["Open", "High", "Low", "Close"]
LIVE_VARIABLES = LIVE_PRICE_VARS + LIVE_INDICATOR_VARS

try:
    import yfinance as yf
except Exception:
    yf = None


def load_strategy_store() -> dict[str, dict[str, Any]]:
    if not STORE_PATH.exists():
        return {}
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def save_strategy_store(store: dict[str, dict[str, Any]]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=True, indent=2)


def parse_saved_date(value: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    try:
        return pd.Timestamp(value)
    except Exception:
        return fallback


def clamp_date(ts: pd.Timestamp, min_date, max_date):
    d = ts.date()
    if d < min_date:
        return min_date
    if d > max_date:
        return max_date
    return d


def normalize_row_for_ui(row: dict[str, Any], numeric_columns: list[str], min_date, max_date, row_id: int) -> dict[str, Any]:
    default_col = numeric_columns[0]
    variable = row.get("variable", default_col)
    if variable not in numeric_columns:
        variable = default_col

    condition = row.get("condition", "=")
    if condition not in CONDITIONS:
        condition = "="

    return {
        "use": bool(row.get("use", row_id == 1)),
        "row_id": row_id,
        "variable": variable,
        "condition": condition,
        "input_value": float(row.get("input_value", 0.0) or 0.0),
        "start_date": clamp_date(parse_saved_date(row.get("start_date"), pd.Timestamp(min_date)), min_date, max_date),
        "end_date": clamp_date(parse_saved_date(row.get("end_date"), pd.Timestamp(max_date)), min_date, max_date),
    }


def apply_loaded_strategy_to_session(strategy: dict[str, Any], numeric_columns: list[str], min_date, max_date) -> None:
    config = strategy.get("configuration", {})

    g_start = parse_saved_date(config.get("global_start_date"), pd.Timestamp(min_date))
    g_end = parse_saved_date(config.get("global_end_date"), pd.Timestamp(max_date))
    st.session_state["global_start_date"] = clamp_date(g_start, min_date, max_date)
    st.session_state["global_end_date"] = clamp_date(g_end, min_date, max_date)

    enter_ticker = config.get("enter_ticker", numeric_columns[0])
    exit_ticker = config.get("exit_ticker", enter_ticker)
    if enter_ticker not in numeric_columns:
        enter_ticker = numeric_columns[0]
    if exit_ticker not in numeric_columns:
        exit_ticker = enter_ticker
    st.session_state["enter_ticker"] = enter_ticker
    st.session_state["exit_ticker"] = exit_ticker

    enter_rows = config.get("enter_rows", [])
    exit_rows = config.get("exit_rows", [])

    for row_id in range(1, ROWS_PER_STRATEGY + 1):
        eraw = enter_rows[row_id - 1] if row_id - 1 < len(enter_rows) else {}
        e = normalize_row_for_ui(eraw, numeric_columns, min_date, max_date, row_id)
        st.session_state[f"enter_use_{row_id}"] = e["use"]
        st.session_state[f"enter_var_{row_id}"] = e["variable"]
        st.session_state[f"enter_src_{row_id}"] = eraw.get("ticker_source", "Underlying")
        st.session_state[f"enter_cond_{row_id}"] = e["condition"]
        st.session_state[f"enter_input_{row_id}"] = e["input_value"]
        st.session_state[f"enter_start_{row_id}"] = e["start_date"]
        st.session_state[f"enter_end_{row_id}"] = e["end_date"]

        xraw = exit_rows[row_id - 1] if row_id - 1 < len(exit_rows) else {}
        x = normalize_row_for_ui(xraw, numeric_columns, min_date, max_date, row_id)
        st.session_state[f"exit_use_{row_id}"] = x["use"]
        st.session_state[f"exit_var_{row_id}"] = x["variable"]
        st.session_state[f"exit_src_{row_id}"] = xraw.get("ticker_source", "Underlying")
        st.session_state[f"exit_cond_{row_id}"] = x["condition"]
        st.session_state[f"exit_input_{row_id}"] = x["input_value"]
        st.session_state[f"exit_start_{row_id}"] = x["start_date"]
        st.session_state[f"exit_end_{row_id}"] = x["end_date"]

    raw_bm = config.get("benchmarks", [])
    st.session_state["benchmarks"] = [b for b in raw_bm if b in numeric_columns][:3]


def apply_loaded_sidebar_settings(strategy: dict[str, Any]) -> None:
    config = strategy.get("configuration", {})
    settings = config.get("settings", {})
    st.session_state["initial_deposit"] = float(settings.get("initial_deposit", 10000.0))
    st.session_state["risk_free"] = float(settings.get("risk_free", 0.0))
    st.session_state["commission_pct"] = float(settings.get("commission_pct", 0.0))
    st.session_state["slippage_pct"] = float(settings.get("slippage_pct", 0.0))
    st.session_state["divergence_pivot_window"] = int(settings.get("divergence_pivot_window", 3))
    st.session_state["divergence_lookback"] = int(settings.get("divergence_lookback", 60))
    if config.get("live_underlying"):
        st.session_state["live_underlying"] = config.get("live_underlying")
    if config.get("live_traded"):
        st.session_state["live_traded"] = config.get("live_traded")


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.where(pd.notna(out), None).to_dict(orient="records")


@st.cache_data(show_spinner=False)
def fetch_live_ohlc(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    if yf is None:
        raise ValueError(
            "Live mode requires yfinance. Install it with: pip install yfinance "
            "or pip install -r requirements.txt"
        )
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns depending on version/settings.
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        if symbol in lvl0:
            df = df[symbol]
        elif symbol in lvl1:
            df = df.xs(symbol, axis=1, level=1)
        else:
            # Fallback: flatten with the first level names.
            df = df.copy()
            df.columns = [str(c[0]) for c in df.columns]

    col_map = {str(c).strip().lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(
            f"Yahoo data for '{symbol}' is missing OHLC columns. "
            f"Available: {list(map(str, df.columns))}"
        )

    out = df[[col_map["open"], col_map["high"], col_map["low"], col_map["close"]]].copy()
    out.columns = ["Open", "High", "Low", "Close"]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    return out


def aggregate_weekly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(how="any")
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
    return (close / close.shift(period) - 1.0) * 100.0


def build_live_dataset(
    underlying_symbol: str,
    traded_symbol: str,
    benchmark_symbols: list[str],
    start_date,
    end_date,
    timeframe: str,
) -> tuple[pd.DataFrame, str, list[str], dict[str, str]]:
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    u = fetch_live_ohlc(underlying_symbol, start_str, end_str)
    t = fetch_live_ohlc(traded_symbol, start_str, end_str)
    if u.empty:
        raise ValueError(f"No data returned for underlying ticker '{underlying_symbol}'.")
    if t.empty:
        raise ValueError(f"No data returned for traded ticker '{traded_symbol}'.")

    if timeframe == "Weekly":
        u = aggregate_weekly_ohlc(u)
        t = aggregate_weekly_ohlc(t)

    common_idx = u.index.intersection(t.index)
    if common_idx.empty:
        raise ValueError("No overlapping timestamps between underlying and traded assets.")

    u = u.loc[common_idx].copy()
    t = t.loc[common_idx].copy()
    out = pd.DataFrame(index=common_idx)

    for col in ["Open", "High", "Low", "Close"]:
        out[f"Underlying_{col}"] = u[col]
        out[f"Traded_{col}"] = t[col]

    sma50 = out["Underlying_Close"].rolling(50).mean()
    out["SMA50"] = sma50
    out["High vs MA50"] = (out["Underlying_High"] / sma50 - 1.0) * 100.0
    out["Low vs MA50"] = (out["Underlying_Low"] / sma50 - 1.0) * 100.0

    sma200 = out["Underlying_Close"].rolling(200).mean()
    out["SMA200"] = sma200
    out["High vs MA200"] = (out["Underlying_High"] / sma200 - 1.0) * 100.0
    out["Low vs MA200"] = (out["Underlying_Low"] / sma200 - 1.0) * 100.0
    out["RSI"] = compute_rsi(out["Underlying_Close"], period=14)
    out["MACD"] = compute_macd_hist(out["Underlying_Close"], fast=12, slow=26, signal=9)
    out["ROC"] = compute_roc(out["Underlying_Close"], period=12)

    benchmark_map: dict[str, str] = {}
    for sym in benchmark_symbols:
        b = fetch_live_ohlc(sym, start_str, end_str)
        if b.empty:
            continue
        if timeframe == "Weekly":
            b = aggregate_weekly_ohlc(b)
        b = b.loc[b.index.intersection(out.index)]
        if b.empty:
            continue
        col = f"BMK_{sym}_Close"
        out[col] = b["Close"]
        benchmark_map[sym] = col

    out = out.reset_index().rename(columns={"index": "Date"})
    return out, "Date", [f"Missing/empty benchmark skipped: {s}" for s in benchmark_symbols if s not in benchmark_map], benchmark_map


def resolve_live_variable_column(variable: str, ticker_source: str) -> str:
    if variable in LIVE_INDICATOR_VARS:
        return variable
    if variable in LIVE_PRICE_VARS:
        prefix = "Underlying" if ticker_source == "Underlying" else "Traded"
        return f"{prefix}_{variable}"
    raise ValueError(f"Unsupported live-mode variable '{variable}'.")


def render_strategy_conditions(
    section_key: str,
    title: str,
    mode: str,
    variable_options: list[str],
    execution_ticker_options: list[str],
    min_date,
    max_date,
    ticker_on_change=None,
) -> tuple[str, list[dict]]:
    st.markdown(f"### {title}")
    ticker = st.selectbox(
        f"{title} ticker",
        options=execution_ticker_options,
        key=f"{section_key}_ticker",
        on_change=ticker_on_change,
    )

    header = st.columns([0.7, 0.8, 1.4, 1.0, 1.3, 1.3, 1.4, 1.4])
    header[0].markdown("**Use**")
    header[1].markdown("**Row**")
    header[2].markdown("**Variable**")
    header[3].markdown("**Ticker**")
    header[4].markdown("**Condition**")
    header[5].markdown("**Input**")
    header[6].markdown("**Start date**")
    header[7].markdown("**End date**")

    out: list[dict] = []
    for row_id in range(1, ROWS_PER_STRATEGY + 1):
        cols = st.columns([0.7, 0.8, 1.4, 1.0, 1.3, 1.3, 1.4, 1.4])
        use_row = cols[0].checkbox(
            f"{title} Use {row_id}",
            key=f"{section_key}_use_{row_id}",
            value=(row_id == 1),
            label_visibility="collapsed",
        )
        cols[1].markdown(f"`{row_id}`")
        variable = cols[2].selectbox(
            f"{title} Variable {row_id}",
            options=variable_options,
            key=f"{section_key}_var_{row_id}",
            label_visibility="collapsed",
        )
        ticker_options = ["Underlying", "Traded"] if mode == "Live Data (Yahoo Finance)" else execution_ticker_options
        force_underlying = mode == "Live Data (Yahoo Finance)" and variable in LIVE_INDICATOR_VARS
        if force_underlying:
            st.session_state[f"{section_key}_src_{row_id}"] = "Underlying"
        ticker_source = cols[3].selectbox(
            f"{title} Source {row_id}",
            options=ticker_options,
            key=f"{section_key}_src_{row_id}",
            disabled=force_underlying,
            label_visibility="collapsed",
        )
        condition = cols[4].selectbox(
            f"{title} Condition {row_id}",
            options=CONDITIONS,
            key=f"{section_key}_cond_{row_id}",
            label_visibility="collapsed",
        )
        needs_input = condition not in {"bear divergence", "bull divergence"}
        input_value = cols[5].number_input(
            f"{title} Input {row_id}",
            value=0.0,
            key=f"{section_key}_input_{row_id}",
            disabled=not needs_input,
            label_visibility="collapsed",
        )
        start_date = cols[6].date_input(
            f"{title} Start {row_id}",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{section_key}_start_{row_id}",
            label_visibility="collapsed",
        )
        end_date = cols[7].date_input(
            f"{title} End {row_id}",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{section_key}_end_{row_id}",
            label_visibility="collapsed",
        )

        out.append(
            {
                "use": use_row,
                "row_id": row_id,
                "variable": variable,
                "ticker_source": ticker_source,
                "condition": condition,
                "input_value": float(input_value) if needs_input else None,
                "start_date": pd.Timestamp(start_date),
                "end_date": pd.Timestamp(end_date),
            }
        )

    return ticker, out


def format_numeric_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(decimals)
    return out


def format_numeric_df_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    for col in num_cols:
        out[col] = out[col].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    return out


def sync_exit_ticker_from_enter() -> None:
    if "enter_ticker" in st.session_state:
        st.session_state["exit_ticker"] = st.session_state["enter_ticker"]


strategy_store = load_strategy_store()

pending_name = st.session_state.get("pending_strategy_load")
if pending_name and pending_name in strategy_store:
    cfg = strategy_store[pending_name].get("configuration", {})
    saved_mode = cfg.get("app_mode")
    saved_tf = cfg.get("timeframe")
    if saved_mode in APP_MODES:
        st.session_state["app_mode"] = saved_mode
    if saved_tf in TIMEFRAMES:
        st.session_state["timeframe"] = saved_tf

mode_cols = st.columns(2)
app_mode = mode_cols[0].radio("Mode", APP_MODES, horizontal=True, key="app_mode")
timeframe = mode_cols[1].radio("Timeframe", TIMEFRAMES, horizontal=True, key="timeframe")

st.sidebar.subheader("Saved Strategies")
saved_names = sorted(strategy_store.keys())
select_col, trash_col = st.sidebar.columns([0.84, 0.16])
selected_saved_name = select_col.selectbox(
    "Choose strategy",
    options=["(None)"] + saved_names,
    key="saved_strategy_select",
)
# Vertical spacer to align trash control with selectbox input row.
trash_col.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
if trash_col.button("🗑️", help="Delete selected strategy", type="tertiary"):
    if selected_saved_name == "(None)":
        st.sidebar.warning("Choose a strategy to delete.")
    else:
        strategy_store = load_strategy_store()
        if selected_saved_name in strategy_store:
            del strategy_store[selected_saved_name]
            save_strategy_store(strategy_store)
        if st.session_state.get("loaded_strategy_name") == selected_saved_name:
            st.session_state["loaded_strategy_name"] = ""
        st.sidebar.success(f"Deleted strategy '{selected_saved_name}'.")
        st.rerun()
if st.sidebar.button("Load Selected Strategy"):
    if selected_saved_name == "(None)":
        st.sidebar.warning("Choose a strategy name first.")
    else:
        strategy = strategy_store.get(selected_saved_name)
        if strategy is not None:
            # Apply sidebar keys before their widgets are instantiated in this rerun.
            apply_loaded_sidebar_settings(strategy)
        st.session_state["auto_run_after_load"] = True
        st.session_state["pending_strategy_load"] = selected_saved_name
        st.rerun()

st.sidebar.header("Backtest Settings")
initial_deposit = st.sidebar.number_input("Initial Deposit", min_value=1.0, value=10000.0, step=100.0, key="initial_deposit")
risk_free = st.sidebar.number_input("Risk-free annual rate (%)", value=0.0, step=0.1, key="risk_free")
commission_pct = st.sidebar.number_input("Commission per trade (%)", min_value=0.0, value=0.0, step=0.01, key="commission_pct")
slippage_pct = st.sidebar.number_input("Slippage per trade (%)", min_value=0.0, value=0.0, step=0.01, key="slippage_pct")
with st.sidebar.expander("Divergence Settings", expanded=False):
    divergence_pivot_window = st.number_input("Pivot window", min_value=1, value=3, step=1, key="divergence_pivot_window")
    divergence_lookback = st.number_input("Lookback bars", min_value=5, value=60, step=1, key="divergence_lookback")

parse_warnings: list[str] = []
raw_df: pd.DataFrame | None = None
source_file_name = ""
fallback_name: str | None = None
date_col_fixed: str | None = None
live_benchmark_map: dict[str, str] = {}

if app_mode == "Excel Data":
    uploaded_file = st.file_uploader("Upload data (.csv or .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            raw_df, parse_warnings = read_uploaded_file(uploaded_file)
            source_file_name = uploaded_file.name
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    else:
        # Fallback: use saved strategy full input table when no file is uploaded.
        fallback_name = st.session_state.get("pending_strategy_load")
        if not fallback_name or fallback_name not in strategy_store:
            selected_name = st.session_state.get("saved_strategy_select")
            if selected_name and selected_name != "(None)" and selected_name in strategy_store:
                fallback_name = selected_name
            else:
                loaded_name = st.session_state.get("loaded_strategy_name")
                if loaded_name and loaded_name in strategy_store:
                    fallback_name = loaded_name

        if fallback_name and fallback_name in strategy_store:
            saved = strategy_store[fallback_name]
            data_rows = saved.get("full_input_table", {}).get("data", [])
            if not data_rows:
                data_rows = saved.get("underlying_inputs", {}).get("data", [])
            if not data_rows:
                st.error("Saved strategy has no underlying input data. Upload a file or save strategy again.")
                st.stop()
            raw_df = pd.DataFrame(data_rows)
            source_file_name = f"saved:{fallback_name}"
            parse_warnings.append(
                f"Using saved input data from strategy '{fallback_name}' (no file uploaded)."
            )
        else:
            st.info("Upload a CSV/XLSX file to configure rules and run the backtest.")
            st.stop()
else:
    st.subheader("Live Data Inputs")
    if yf is None:
        st.error(
            "yfinance is not installed. Live mode is unavailable until you install dependencies:\n"
            "`pip install -r requirements.txt`"
        )
        st.stop()
    lc = st.columns(4)
    underlying_symbol = lc[0].text_input("Underlying ticker", value=st.session_state.get("live_underlying", "^NDX"), key="live_underlying")
    traded_symbol = lc[1].text_input("Traded ticker", value=st.session_state.get("live_traded", "TQQQ"), key="live_traded")
    live_end = lc[3].date_input("End date (dd/mm/yyyy)", value=st.session_state.get("live_end", pd.Timestamp.today().date()), key="live_end")
    # Auto-start is determined from first bar where:
    # 1) Underlying SMA200 is available, and
    # 2) Traded asset price exists on the same timestamp.
    auto_start_date = (pd.Timestamp(live_end) - pd.DateOffset(years=5)).date()
    try:
        auto_fetch_start = (pd.Timestamp(live_end) - pd.DateOffset(years=15)).strftime("%Y-%m-%d")
        auto_fetch_end = (pd.Timestamp(live_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        u_auto = fetch_live_ohlc(underlying_symbol, auto_fetch_start, auto_fetch_end)
        t_auto = fetch_live_ohlc(traded_symbol, auto_fetch_start, auto_fetch_end)
        if not u_auto.empty and not t_auto.empty:
            if timeframe == "Weekly":
                u_auto = aggregate_weekly_ohlc(u_auto)
                t_auto = aggregate_weekly_ohlc(t_auto)
            common_idx = u_auto.index.intersection(t_auto.index)
            if len(common_idx) > 0:
                u_auto = u_auto.loc[common_idx]
                t_auto = t_auto.loc[common_idx]
                sma = u_auto["Close"].rolling(200).mean()
                ready_mask = sma.notna() & t_auto["Close"].notna()
                ready_idx = ready_mask[ready_mask].index
                if len(ready_idx) > 0:
                    auto_start_date = pd.Timestamp(ready_idx[0]).date()
    except Exception:
        pass
    lc[2].date_input(
        "Start date (dd/mm/yyyy)",
        value=auto_start_date,
        disabled=True,
        help="Automatically set to first date where Underlying SMA200 is available.",
    )
    bc = st.columns(3)
    bc[0].text_input(
        "Benchmark 1 (Yahoo ticker)",
        value=traded_symbol,
        disabled=True,
        help="Benchmark 1 is automatically set to Traded ticker.",
    )
    b2 = bc[1].text_input("Benchmark 2 (Yahoo ticker)", value=st.session_state.get("live_bmk_2", ""), key="live_bmk_2").strip()
    b3 = bc[2].text_input("Benchmark 3 (Yahoo ticker)", value=st.session_state.get("live_bmk_3", ""), key="live_bmk_3").strip()
    benchmark_syms = [s for s in [traded_symbol, b2, b3] if s]
    try:
        fetch_start = (pd.Timestamp(live_end) - pd.DateOffset(years=15)).date()
        raw_df, date_col_fixed, live_warns, live_benchmark_map = build_live_dataset(
            underlying_symbol=underlying_symbol,
            traded_symbol=traded_symbol,
            benchmark_symbols=benchmark_syms,
            start_date=fetch_start,
            end_date=live_end,
            timeframe=timeframe,
        )
        sma_ready = raw_df.loc[raw_df["SMA200"].notna() & raw_df["Traded_Close"].notna(), "Date"]
        if sma_ready.empty:
            raise ValueError(
                "Insufficient history: no bar where Underlying SMA200 is available and Traded price exists."
            )
        sma_start = pd.Timestamp(sma_ready.iloc[0])
        raw_df = raw_df.loc[raw_df["Date"] >= sma_start].copy().reset_index(drop=True)
        parse_warnings.append(
            f"Live mode start date auto-set to {sma_start.date()} (first bar with Underlying SMA200 and Traded price available)."
        )
        parse_warnings.extend(live_warns)
        source_file_name = f"live:{underlying_symbol}->{traded_symbol}"
    except Exception as exc:
        st.error(str(exc))
        st.stop()

if raw_df.empty:
    st.error("Data source returned no rows.")
    st.stop()

date_candidates = detect_date_candidates(raw_df)
if not date_candidates:
    st.error("No valid date column detected. Please provide at least one parseable date column.")
    st.stop()

# Prefer date column from selected/pending saved strategy when available.
preferred_date_col = None
for strategy_name in [
    st.session_state.get("pending_strategy_load"),
    st.session_state.get("saved_strategy_select"),
    st.session_state.get("loaded_strategy_name"),
]:
    if not strategy_name or strategy_name == "(None)" or strategy_name not in strategy_store:
        continue
    saved_date_col = strategy_store[strategy_name].get("configuration", {}).get("date_column")
    if saved_date_col in date_candidates:
        preferred_date_col = saved_date_col
        break
if preferred_date_col is not None:
    st.session_state["date_col"] = preferred_date_col

if date_col_fixed is not None:
    st.session_state["date_col"] = date_col_fixed
    date_col = date_col_fixed
    st.caption(f"Date column: {date_col_fixed}")
else:
    if "date_col" not in st.session_state or st.session_state["date_col"] not in date_candidates:
        st.session_state["date_col"] = date_candidates[0]
    date_col = st.selectbox("Date column", date_candidates, key="date_col")

try:
    ts_df, prep_warnings = prepare_timeseries_df(raw_df, date_col)
except Exception as exc:
    st.error(str(exc))
    st.stop()

for w in prep_warnings:
    st.warning(w)
for w in parse_warnings:
    st.warning(w)

numeric_columns = find_numeric_columns(ts_df)
if not numeric_columns:
    st.error("No numeric columns available for Variable/Ticker/Benchmark selection.")
    st.stop()

st.caption(f"Rows: {len(ts_df)} | Numeric columns available: {len(numeric_columns)}")

min_date = ts_df.index.min().date()
max_date = ts_df.index.max().date()

pending = st.session_state.get("pending_strategy_load")
if pending:
    strategy = strategy_store.get(pending)
    if strategy is None:
        st.warning(f"Saved strategy '{pending}' not found.")
    else:
        apply_loaded_strategy_to_session(strategy, numeric_columns, min_date, max_date)
        st.session_state["loaded_strategy_name"] = pending
    st.session_state["pending_strategy_load"] = None
    st.rerun()

st.subheader("Global Backtest Window")
window_cols = st.columns(2)
global_start_date = window_cols[0].date_input(
    "Global start date (dd/mm/yyyy)",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key="global_start_date",
)
global_end_date = window_cols[1].date_input(
    "Global end date (dd/mm/yyyy)",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="global_end_date",
)
if global_start_date > global_end_date:
    st.error("Global start date cannot be after global end date.")
    st.stop()
global_start_ts = pd.Timestamp(global_start_date)
global_end_ts = pd.Timestamp(global_end_date)

enter_ticker, enter_inputs = render_strategy_conditions(
    "enter",
    "Enter Strategy",
    app_mode,
    LIVE_VARIABLES if app_mode == "Live Data (Yahoo Finance)" else numeric_columns,
    ["Traded_Close"] if app_mode == "Live Data (Yahoo Finance)" else numeric_columns,
    min_date,
    max_date,
    ticker_on_change=sync_exit_ticker_from_enter,
)
if "exit_ticker" not in st.session_state:
    st.session_state["exit_ticker"] = enter_ticker

exit_ticker, exit_inputs = render_strategy_conditions(
    "exit",
    "Exit Strategy",
    app_mode,
    LIVE_VARIABLES if app_mode == "Live Data (Yahoo Finance)" else numeric_columns,
    ["Traded_Close"] if app_mode == "Live Data (Yahoo Finance)" else numeric_columns,
    min_date,
    max_date,
)

st.subheader("Benchmarks")
if app_mode == "Live Data (Yahoo Finance)":
    benchmarks = list(live_benchmark_map.keys())[:3]
    if benchmarks:
        st.caption("Loaded live benchmarks: " + ", ".join(benchmarks))
    else:
        st.caption("No live benchmarks loaded.")
else:
    benchmarks = st.multiselect(
        "Select up to 3 benchmark columns",
        options=numeric_columns,
        max_selections=3,
        key="benchmarks",
    )

manual_run = st.button("Run Backtest", type="primary")
run = manual_run or bool(st.session_state.pop("auto_run_after_load", False))

if run:
    try:
        active_enter = [r for r in enter_inputs if r["use"]]
        active_exit = [r for r in exit_inputs if r["use"]]

        if not active_enter:
            raise ValueError("Select at least one active row in Enter Strategy.")
        if not active_exit:
            raise ValueError("Select at least one active row in Exit Strategy.")

        for r in active_enter:
            if r["start_date"] > r["end_date"]:
                raise ValueError(f"Enter Strategy row {r['row_id']}: start date cannot be after end date.")
        for r in active_exit:
            if r["start_date"] > r["end_date"]:
                raise ValueError(f"Exit Strategy row {r['row_id']}: start date cannot be after end date.")

        if app_mode == "Live Data (Yahoo Finance)":
            for side, rows in [("Enter", active_enter), ("Exit", active_exit)]:
                for r in rows:
                    if r["condition"] in {"bear divergence", "bull divergence"} and r["variable"] not in LIVE_INDICATOR_VARS:
                        raise ValueError(
                            f"{side} row {r['row_id']}: divergence requires an underlying indicator variable "
                            f"({ ' / '.join(LIVE_INDICATOR_VARS) })."
                        )

        numeric_needed = sorted(
            set(
                [
                    enter_ticker,
                    exit_ticker,
                    *[
                        resolve_live_variable_column(r["variable"], r["ticker_source"])
                        if app_mode == "Live Data (Yahoo Finance)"
                        else r["variable"]
                        for r in active_enter
                    ],
                    *[
                        resolve_live_variable_column(r["variable"], r["ticker_source"])
                        if app_mode == "Live Data (Yahoo Finance)"
                        else r["variable"]
                        for r in active_exit
                    ],
                    *(
                        [live_benchmark_map[b] for b in benchmarks if b in live_benchmark_map]
                        if app_mode == "Live Data (Yahoo Finance)"
                        else benchmarks
                    ),
                ]
            )
        )
        ts_df_numeric, numeric_warnings = ensure_columns_numeric(ts_df, numeric_needed)

        if app_mode == "Live Data (Yahoo Finance)":
            required_indicators = sorted(
                {
                    resolve_live_variable_column(r["variable"], r["ticker_source"])
                    for r in active_enter + active_exit
                    if r["variable"] in LIVE_INDICATOR_VARS
                }
            )
            if required_indicators:
                valid_mask = ts_df_numeric[required_indicators].notna().all(axis=1)
                if not valid_mask.any():
                    raise ValueError(
                        "Insufficient data for selected indicator warm-up in the chosen date range."
                    )
                warmup_start = valid_mask[valid_mask].index[0]
                if warmup_start > global_start_ts:
                    numeric_warnings.append(
                        f"Backtest start shifted to {warmup_start.date()} due to indicator warm-up."
                    )
                    global_start_ts = warmup_start

        strategy_rules = StrategyRules(
            enter_ticker=enter_ticker,
            exit_ticker=exit_ticker,
            enter_conditions=[
                ConditionRow(
                    row_id=r["row_id"],
                    variable=(
                        resolve_live_variable_column(r["variable"], r["ticker_source"])
                        if app_mode == "Live Data (Yahoo Finance)"
                        else r["variable"]
                    ),
                    condition=r["condition"],
                    input_value=r["input_value"],
                    start_date=r["start_date"],
                    end_date=r["end_date"],
                    divergence_price_col=("Underlying_Close" if app_mode == "Live Data (Yahoo Finance)" else None),
                )
                for r in active_enter
            ],
            exit_conditions=[
                ConditionRow(
                    row_id=r["row_id"],
                    variable=(
                        resolve_live_variable_column(r["variable"], r["ticker_source"])
                        if app_mode == "Live Data (Yahoo Finance)"
                        else r["variable"]
                    ),
                    condition=r["condition"],
                    input_value=r["input_value"],
                    start_date=r["start_date"],
                    end_date=r["end_date"],
                    divergence_price_col=("Underlying_Close" if app_mode == "Live Data (Yahoo Finance)" else None),
                )
                for r in active_exit
            ],
        )

        bt_cfg = BacktestConfig(
            initial_deposit=float(initial_deposit),
            commission_pct=float(commission_pct),
            slippage_pct=float(slippage_pct),
            divergence_pivot_window=int(divergence_pivot_window),
            divergence_lookback=int(divergence_lookback),
        )

        result = run_backtest(
            df=ts_df_numeric,
            strategy_rules=strategy_rules,
            config=bt_cfg,
            global_start=global_start_ts,
            global_end=global_end_ts,
        )

        equity_curves = pd.DataFrame(index=result.equity.index)
        equity_curves["Strategy"] = result.equity

        for b in benchmarks:
            b_col = live_benchmark_map[b] if app_mode == "Live Data (Yahoo Finance)" else b
            s = ts_df_numeric.loc[equity_curves.index, b_col].copy().ffill()
            if pd.isna(s.iloc[0]) or s.iloc[0] <= 0:
                raise ValueError(f"Benchmark '{b}' has invalid start value in selected window.")
            equity_curves[b] = float(initial_deposit) * (s / float(s.iloc[0]))

        metric_rows = []
        warnings = list(result.warnings)
        warnings.extend(numeric_warnings)

        for col in equity_curves.columns:
            m, m_warn = headline_metrics(
                equity_curves[col],
                risk_free_annual_pct=float(risk_free),
                periods_per_year=(252 if timeframe == "Daily" else 52),
            )
            warnings.extend([f"{col}: {w}" for w in m_warn])
            metric_rows.append({"Series": col, **m})

        metrics_df = pd.DataFrame(metric_rows)
        yearly_df = yearly_table(equity_curves["Strategy"])
        trades_df = result.trades.copy()
        if not trades_df.empty:
            trades_df["gross_value"] = trades_df["price"] * trades_df["shares"]

        metrics_display = format_numeric_df(metrics_df, 2)
        yearly_display = format_numeric_df(yearly_df, 2)
        trades_display = format_numeric_df(trades_df, 2)

        st.session_state["last_backtest"] = {
            "source_file": source_file_name,
            "date_col": date_col,
            "global_start_date": str(global_start_date),
            "global_end_date": str(global_end_date),
            "enter_ticker": enter_ticker,
            "exit_ticker": exit_ticker,
            "benchmarks": list(benchmarks),
            "settings": {
                "initial_deposit": float(initial_deposit),
                "risk_free": float(risk_free),
                "commission_pct": float(commission_pct),
                "slippage_pct": float(slippage_pct),
                "divergence_pivot_window": int(divergence_pivot_window),
                "divergence_lookback": int(divergence_lookback),
            },
            "enter_rows": enter_inputs,
            "exit_rows": exit_inputs,
            "numeric_needed": numeric_needed,
            "equity_curves": equity_curves,
            "metrics_display": metrics_display,
            "yearly_display": yearly_display,
            "trades_display": trades_display,
            "input_slice": ts_df_numeric.loc[equity_curves.index, numeric_needed].copy().reset_index(),
            "full_input_table": raw_df.copy(),
            "warnings": sorted(set(warnings)),
        }
    except Exception as exc:
        st.error(str(exc))

bundle = st.session_state.get("last_backtest")
if bundle is None:
    st.info("Configure strategy and click 'Run Backtest' to see results.")
    st.stop()

metrics_display = bundle["metrics_display"]
yearly_display = bundle["yearly_display"]
trades_display = bundle["trades_display"]
equity_curves = bundle["equity_curves"]
warnings = bundle["warnings"]

metrics_display_ui = format_numeric_df_for_display(metrics_display, 2)
yearly_display_ui = format_numeric_df_for_display(yearly_display, 2)
trades_display_ui = format_numeric_df_for_display(trades_display, 2)
if "Year" in yearly_display_ui.columns:
    yearly_display_ui["Year"] = yearly_display["Year"].astype("Int64").astype(str)

st.subheader("Headline Metrics (Strategy)")
srow = metrics_display[metrics_display["Series"] == "Strategy"].iloc[0]
mcols = st.columns(4)
mcols[0].metric("CAGR %", f"{srow['CAGR %']:.2f}" if pd.notna(srow["CAGR %"]) else "NaN")
mcols[1].metric("Max Drawdown %", f"{srow['Max Drawdown %']:.2f}")
mcols[2].metric("Average Drawdown %", f"{srow['Average Drawdown %']:.2f}")
mcols[3].metric("Sharpe Ratio", f"{srow['Sharpe Ratio']:.2f}" if pd.notna(srow["Sharpe Ratio"]) else "NaN")

st.subheader("Metrics (Strategy + Benchmarks)")
st.dataframe(metrics_display_ui, use_container_width=True)

st.subheader("Yearly Results (Strategy)")
st.dataframe(yearly_display_ui, use_container_width=True)

st.subheader("Executed Transactions")
if trades_display_ui.empty:
    st.info("No transactions were executed in this backtest window.")
else:
    trade_cols = ["date", "signal_date", "action", "reason", "ticker", "price", "shares", "gross_value", "commission"]
    available_cols = [c for c in trade_cols if c in trades_display_ui.columns]
    st.dataframe(trades_display_ui[available_cols], use_container_width=True)

st.subheader("Weekly Capitalisation (Log Scale)")
weekly = equity_curves.resample("W-FRI").last().dropna(how="all")
weekly_dd_pct = drawdown_series(weekly["Strategy"]).fillna(0.0) * 100.0

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.75, 0.25],
    subplot_titles=("Equity (Log Scale)", "Strategy Drawdown %"),
)
for col in weekly.columns:
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly[col], mode="lines", name=col), row=1, col=1)
fig.add_trace(
    go.Bar(
        x=weekly_dd_pct.index,
        y=weekly_dd_pct.values,
        name="Strategy Drawdown %",
        marker_color="#ff0000",
        opacity=0.85,
    ),
    row=2,
    col=1,
)
fig.update_layout(
    xaxis2_title="Date",
    yaxis_title="Equity",
    yaxis2_title="Drawdown %",
    template="plotly_white",
    height=780,
    barmode="overlay",
)
fig.update_yaxes(type="log", row=1, col=1)
st.plotly_chart(fig, use_container_width=True)

if warnings:
    st.subheader("Warnings")
    for w in warnings:
        st.warning(w)

st.subheader("Assumptions")
st.markdown(
    "\n".join(
        [
            "- Execution timing: signals computed on day t and executed on day t+1 close.",
            "- Entry signal: all active Enter Strategy conditions must be true simultaneously.",
            "- Exit signal: all active Exit Strategy conditions must be true simultaneously.",
            "- Costs: commission and slippage applied on each filled trade using configured percentages.",
            "- Portfolio constraints: long-only, single holding at a time, otherwise fully in cash.",
            "- Average drawdown: arithmetic mean of the full drawdown time series.",
            "- Backtest window: user-selected global date range.",
        ]
    )
)

st.subheader("Export")
equity_csv = (
    format_numeric_df(equity_curves.reset_index().rename(columns={"index": "date"}), 2)
    .to_csv(index=False, float_format="%.2f")
)
st.download_button(
    "Download Equity Curves CSV",
    data=equity_csv,
    file_name="equity_curves.csv",
    mime="text/csv",
)

buffer = StringIO()
metrics_display.to_csv(buffer, index=False, float_format="%.2f")
buffer.write("\nYearly Results (Strategy)\n")
yearly_display.to_csv(buffer, index=False, float_format="%.2f")
if not trades_display.empty:
    buffer.write("\nExecuted Transactions\n")
    trades_display.to_csv(buffer, index=False, float_format="%.2f")
st.download_button(
    "Download Metrics + Yearly CSV",
    data=buffer.getvalue(),
    file_name="metrics_and_yearly.csv",
    mime="text/csv",
)

st.subheader("Save Strategy")
default_save_name = st.session_state.get("loaded_strategy_name", "")
strategy_name = st.text_input("Strategy name", value=default_save_name, key="save_strategy_name")

if st.button("Save", type="secondary"):
    clean_name = strategy_name.strip()
    if not clean_name:
        st.error("Please provide a strategy name before saving.")
    else:
        payload = {
            "name": clean_name,
            "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source_file": bundle["source_file"],
            "configuration": {
                "app_mode": app_mode,
                "timeframe": timeframe,
                "date_column": bundle["date_col"],
                "global_start_date": bundle["global_start_date"],
                "global_end_date": bundle["global_end_date"],
                "enter_ticker": bundle["enter_ticker"],
                "exit_ticker": bundle["exit_ticker"],
                "benchmarks": bundle["benchmarks"],
                "live_underlying": st.session_state.get("live_underlying"),
                "live_traded": st.session_state.get("live_traded"),
                "settings": bundle["settings"],
                "enter_rows": [
                    {
                        "use": bool(r["use"]),
                        "row_id": int(r["row_id"]),
                        "variable": r["variable"],
                        "ticker_source": r.get("ticker_source", "Underlying"),
                        "condition": r["condition"],
                        "input_value": r["input_value"],
                        "start_date": str(pd.Timestamp(r["start_date"]).date()),
                        "end_date": str(pd.Timestamp(r["end_date"]).date()),
                    }
                    for r in bundle["enter_rows"]
                ],
                "exit_rows": [
                    {
                        "use": bool(r["use"]),
                        "row_id": int(r["row_id"]),
                        "variable": r["variable"],
                        "ticker_source": r.get("ticker_source", "Underlying"),
                        "condition": r["condition"],
                        "input_value": r["input_value"],
                        "start_date": str(pd.Timestamp(r["start_date"]).date()),
                        "end_date": str(pd.Timestamp(r["end_date"]).date()),
                    }
                    for r in bundle["exit_rows"]
                ],
            },
            "results": {
                "metrics": df_to_records(bundle["metrics_display"]),
                "yearly": df_to_records(bundle["yearly_display"]),
                "trades": df_to_records(bundle["trades_display"]),
                "equity_curves": df_to_records(format_numeric_df(bundle["equity_curves"].reset_index().rename(columns={"index": "date"}), 2)),
            },
            "underlying_inputs": {
                "columns": bundle["numeric_needed"],
                "data": df_to_records(format_numeric_df(bundle["input_slice"].rename(columns={"index": "date"}), 2)),
            },
            "full_input_table": {
                "columns": list(bundle["full_input_table"].columns),
                "data": df_to_records(bundle["full_input_table"]),
            },
        }

        strategy_store = load_strategy_store()
        strategy_store[clean_name] = payload
        save_strategy_store(strategy_store)
        st.session_state["loaded_strategy_name"] = clean_name
        st.success(f"Strategy '{clean_name}' saved.")
        st.rerun()

# pip install streamlit yfinance ta pandas numpy

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
import time
import os
import json
import html
import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from streamlit.errors import StreamlitSecretNotFoundError

from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from alpha_engine import (
    alpha_config,
    calculate_alpha_engine,
    calculate_sma200d_robust_z_36m,
    sort_by_alpha,
)
from finance_core import (
    USD_KRW_TICKER,
    convert_krw_ohlcv_to_usd,
    download_completed_ohlcv,
    download_latest_ohlcv,
    extract_ohlcv_frame,
    is_krw_quoted_ticker,
)
from fund_flows import FundFlowCache, default_fund_flow_cache_path, get_fund_flow_metrics
from market_model import (
    YAHOO_MARKET_TICKERS,
    calculate_confirmations_history,
    calculate_credit_stress_confirmation_history,
    calculate_fast_transition_risk_history,
    calculate_macro_transition_risk_history,
    calculate_market_model,
    calculate_overall_transition_status,
    calculate_positioning_risk_history,
    calculate_tail_risk_history,
    classify_global_liquidity_backdrop,
    download_fred_market_data,
    market_model_config,
    weekly_close,
)
from table_export import dataframe_to_excel_xls_bytes
from ai_dashboard import AI_GROUP_LABELS, AI_UNIVERSE, canonical_ai_group_label, is_ai_group_label
from ai_dashboard_tab import render_ai_dashboard_tab
from business_cycle_tab import render_business_cycle_tab
from cio_view import render_cio_view_tab
from cftc_cot_tab import render_cftc_cot_tab
from global_macro_tab import render_global_macro_tab
from global_dashboard_tab import render_global_dashboard_tab
from liquidity_forecast import read_forecast_snapshot
from liquidity_forecast_tab import render_liquidity_forecast
from rates_financial_conditions_tab import render_rates_financial_conditions_tab
from funding_conditions_tab import render_funding_conditions_tab
from financial_fragility_tab import render_financial_fragility_tab
from treasury_fiscal_regime_tab import render_treasury_fiscal_regime_tab
from treasury_funding_policy import read_snapshot as read_treasury_funding_policy_snapshot
from gold_regime_tab import render_gold_regime_tab
from market_cycle_tab import render_market_cycle_tab
from global_liquidity import (
    GLOBAL_LIQUIDITY_STORAGE_DIR,
    freshness as global_liquidity_freshness,
    global_liquidity_update_in_progress,
    read_global_liquidity,
    start_background_update_if_stale as start_global_liquidity_update_if_stale,
    update_global_liquidity,
)
from bybit_derivatives import (
    BYBIT_ASSET_MAP,
    BYBIT_STORAGE_PATH,
    bybit_update_in_progress,
    latest_states as bybit_latest_states,
    read_bybit_storage,
    start_background_update_if_stale,
)
from screener_metrics import (
    correction_risk_from_percentile_analogs,
    historical_momentum_52w_metrics,
    sma200w_distance_percentile,
)

# ============================================================
# 1) ETF Universe (exactly as provided)
# ============================================================
ETF_UNIVERSE_FULL = {
    "Crypto": {
        "BTC": ["BTC-USD"],
        "Other": ["ETH-USD", "SOL-USD", "BITW"],
    },
    "Gold": {
        "Gold": ["GLD", "SLV", "GDX", "GDXJ", "SIL"],
    },
    "Equities": {
        "US": ["SPY", "QQQ", "IWM", "MTUM", "VLU", "VUG", "SPYD"],
        "US Sectors": ["XLK", "XLC", "XLF", "XLI", "XLY", "XLE", "XLB", "XLP", "XLU", "XLV", "IBB"],
        "DM": ["VEA"],
        "EM": ["EEM"],
        "EU": ["EWG", "EWQ", "EWI", "EWU"],
        "Asia": ["EWJ", "EWY", "INDA", "ASEA", "EIDO", "EPHE", "EWM", "EWT", "THD", "TUR", "VNM"],
        "China": ["MCHI", "KWEB"],
        "Pacific": ["EWA"],
        "LATAM": ["ILF", "EWZ", "ECH", "EPU", "EWW"],
        "Tech": ["BLOK", "BUG", "DTCR", "SKYY", "SOXX", "TINY"],
        "Transport": ["BOAT"],
    },
    "Equity": {
        "AI": AI_GROUP_LABELS,
    },
    "Bonds": {
        "Bonds": ["SHY", "IEF", "TLT", "TIP", "EMB", "HYG", "JNK"],
    },
    "Commodities": {
        "Commodities": ["DBC"],
        "Energy": ["DBO", "DBE", "FCG", "URNM", "FAN", "TAN"],
        "Agriculture": ["DBA"],
        "Metals": ["DBP", "DBB", "PALL", "PPLT", "CPER", "COPX", "LIT", "REMX"],
    },
    "Real Estate": {
        "US": ["VNQ"],
        "ex US": ["VNQI"]
    }
}

ETF_UNIVERSE_SHORT = {
    "Crypto": {
        "Crypto": ["IBIT"],
    },
    "Gold": {
        "Gold": ["GLD", "GDX", "SLV"],
    },
    "Equities": {
        "US": ["SPY", "QQQ", "IWM", "MTUM", "VLUE"],
        "DM": ["VEA", "EWG", "EWI"],
        "EM": ["EEM", "MCHI", "FXI", "KWEB", "INDA", "ASEA", "EWZ", "ILF"],
    },
    "Bonds": {
        "Bonds": ["TLT", "TIP", "EMB", "HYG"],
    },
    "Commodities": {
        "Commodities": ["DBC", "DBA", "DBO", "DBP", "DBB"],
    },
    "Real Estate": {
        "RE": ["VNQ", "VNQI"],
    },
}

ETF_UNIVERSE_CRYPTO = {
    "Crypto": {
        "BTC": ["BTC-USD"],
        "L1": [
            "ETH-USD", "SOL-USD", "SUI-USD", "APT-USD", "NEAR-USD", "TRX-USD", "ADA-USD",
            "AVAX-USD", "TON-USD", "HBAR-USD", "VET-USD", "INJ-USD", "TIA-USD", "DOT-USD",
        ],
        "CEX": ["BNB-USD", "BGB-USD", "OKB-USD", "CRO-USD"],
        "PAYMENT": ["XRP-USDT", "XLM-USDT", "LTC-USD", "BCH-USD", "XMR-USD", "DASH-USD", "CELO-USD"],
        "Oracles": ["LINK-USD", "PYTH-USD", "BAND-USD"],
        "DEFI": ["AAVE-USD", "JUP-USD", "HYPE32196-USD", "UNI-USD", "RAY-USD"],
        "L2": ["MATIC-USD", "ARB-USD", "OP-USD", "ZK-USD"],
        "MEMES": ["DOGE-USD", "SHIB-USD", "TRUMP-USD"],
        "SHARED COMPUTE": ["TAO-USD", "RENDER-USD", "FET-USD"],
    }
}

ETF_UNIVERSE_MAP = {
    "Full list": ETF_UNIVERSE_FULL,
    "Short List": ETF_UNIVERSE_SHORT,
    "Crypto list": ETF_UNIVERSE_CRYPTO,
}
UNIVERSE_STORAGE_PATH = Path(__file__).with_name("custom_universe_lists.json")
AUTO_REFRESH_SECONDS = 600
SLOW_REFRESH_SECONDS = 21600

GRAPH_PERIOD_OPTIONS = ["Daily", "Weekly", "Monthly", "Full history"]

ALPHA_CORE_COLUMNS = [
    "Alpha_Score",
    "Alpha_State",
    "Momentum_Score",
    "Trend_Quality_Score",
    "Persistence_Score",
    "Alpha_Confidence",
    "Entry_Risk_Score",
    "Opportunity_State",
    "Opportunity_Score",
]

TABLE_ALPHA_COLUMNS = [col for col in ALPHA_CORE_COLUMNS if col != "Alpha_Confidence"]

ALPHA_TECHNICAL_COLUMNS = [
    "ADX_DI_Trend_Score",
    "DI_Balance",
    "DI_Plus_14",
    "DI_Minus_14",
    "SMA_Regime_Score",
    "Absolute_SMA_Score",
    "Relative_SMA_Score",
    "SMA200d_Robust_Z_36M",
    "Regime_Dependent_SMAZ_Risk",
    "Perf12M_Extreme_Risk",
    "Base_Alpha",
    "SPY_vs_SMA40W_%",
    "SPY_Drawdown_52W_%",
    "SPY_Volatility_13W_%",
    "SPY_Volatility_Percentile",
    "Alpha_Data_Complete",
]

DISPLAY_COLUMNS = [
    "Group", "Subgroup", "Ticker",
    "CurrentPrice",
    *TABLE_ALPHA_COLUMNS,
    *ALPHA_TECHNICAL_COLUMNS,
    "Perf_1D_%", "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%", "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%", "Perf_10Y_%",
    "SMA200W_Distance_Percentile", "Perf_12M_Percentile", "Avg_Forward_Return_6M_%", "Correction_Risk_%",
    "FundFlows_1M_%", "FundFlows_3M_%",
    "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14", "RSI_14W", "ADX_14", "BB_Position",
    "SMA50w_vs_SMA200w_Spread_%", "SMA_Spread_%_Change_6M_%", "SMA_Trend",
    "Div_6M_vs_RSI", "Div_6M_vs_MACD", "Div_6M_vs_ROC",
    "Divergence_Bull_Count", "Divergence_Bear_Count",
]

TABLE_HEADER_NAMES = {
    "Group": "Group",
    "Subgroup": "Sub\ngroup",
    "Ticker": "Ticker",
    "CurrentPrice": "Current\nPrice",
    "Alpha_Score": "Alpha\nScore",
    "Alpha_State": "Alpha\nState",
    "Momentum_Score": "Momentum\nScore",
    "Trend_Quality_Score": "Trend Quality\nScore",
    "Persistence_Score": "Persistence\nScore",
    "Market_Regime": "Market\nRegime",
    "Alpha_Confidence": "Alpha\nConfidence",
    "Entry_Risk_Score": "Entry Risk\nScore",
    "Entry_Risk": "Entry\nRisk",
    "Opportunity_State": "Opportunity\nState",
    "Opportunity_Score": "Opportunity\nScore",
    "ADX_DI_Trend_Score": "ADX DI\nTrend Score",
    "DI_Balance": "DI\nBalance",
    "DI_Plus_14": "+DI\n14",
    "DI_Minus_14": "-DI\n14",
    "SMA_Regime_Score": "SMA Regime\nScore",
    "Absolute_SMA_Score": "Absolute SMA\nScore",
    "Relative_SMA_Score": "Relative SMA\nScore",
    "SMA200d_Robust_Z_36M": "SMA200d Robust\nZ 36M",
    "Regime_Dependent_SMAZ_Risk": "Regime SMAZ\nRisk",
    "Perf12M_Extreme_Risk": "Perf12M Extreme\nRisk",
    "Base_Alpha": "Base\nAlpha",
    "SPY_vs_SMA40W_%": "SPY vs\nSMA40W %",
    "SPY_Drawdown_52W_%": "SPY Drawdown\n52W %",
    "SPY_Volatility_13W_%": "SPY Volatility\n13W %",
    "SPY_Volatility_Percentile": "SPY Volatility\nPercentile",
    "Alpha_Data_Complete": "Alpha Data\nComplete",
    "Perf_1D_%": "Perf\n1D %",
    "Perf_1W_%": "Perf\n1W %",
    "Perf_1M_%": "Perf\n1M %",
    "Perf_3M_%": "Perf\n3M %",
    "Perf_6M_%": "Perf\n6M %",
    "Perf_12M_%": "Perf\n12M %",
    "Perf_12M_Percentile": "Perf 12M\nPercentile",
    "Avg_Forward_Return_6M_%": "Avg forward\nreturn 6M %",
    "Correction_Risk_%": "Correction\nrisk %",
    "Perf_3Y_%": "Perf\n3Y %",
    "Perf_5Y_%": "Perf\n5Y %",
    "Perf_10Y_%": "Perf\n10Y %",
    "FundFlows_1M_%": "FundFlows\n1M %",
    "FundFlows_3M_%": "FundFlows\n3M %",
    "Price_vs_52W_High_%": "Price vs\n52W High %",
    "Price_vs_ATH_%": "Price vs\nATH %",
    "RSI_14": "RSI\n14D",
    "RSI_14W": "RSI\n14W",
    "ADX_14": "ADX\n14",
    "BB_Position": "BB\nPosition",
    "SMA200W_Distance_Percentile": "SMA200W\nPercentile",
    "SMA50w_vs_SMA200w_Spread_%": "Price vs\nSMA200d %",
    "SMA_Spread_%_Change_6M_%": "SMA Spread\nChange 6M %",
    "SMA_Trend": "SMA\nTrend",
    "Divergence_Bull_Count": "Divergence\nBull\nCount",
    "Divergence_Bear_Count": "Divergence\nBear\nCount",
    "Div_6M_vs_RSI": "Div\nRSI",
    "Div_6M_vs_MACD": "Div\nMACD",
    "Div_6M_vs_ROC": "Div\nROC",
}

TABLE_PERMANENTLY_HIDDEN_COLUMNS = {
    *ALPHA_TECHNICAL_COLUMNS,
    "SMA50w_vs_SMA200w_Spread_%",
    "SMA_Spread_%_Change_6M_%",
    "BB_Mid",
    "BB_Upper",
    "BB_Lower",
    "BB_StepUp",
    "BB_StepDown",
    "WeeklyClose_Last",
}

NUMERIC_COLUMNS = [
    "CurrentPrice",
    "Perf_1D_%", "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%", "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%", "Perf_10Y_%",
    "SMA200W_Distance_Percentile", "Perf_12M_Percentile", "Avg_Forward_Return_6M_%", "Correction_Risk_%",
    "FundFlows_1M_%", "FundFlows_3M_%",
    "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14", "RSI_14W",
    "BB_Position",
    "BB_Mid", "BB_Upper", "BB_Lower", "BB_StepUp", "BB_StepDown", "WeeklyClose_Last",
    "SMA50w_vs_SMA200w_Spread_%", "SMA50w_vs_SMA200w_Spread_Avg_36M_%", "SMA_Spread_%_Change_6M_%",
    "ADX_14", "DI_Plus_14", "DI_Minus_14", "DI_Plus_14_Delta2", "DI_Minus_14_Delta2",
    "Divergence_Bull_Count", "Divergence_Bear_Count",
    "Alpha_Rank", "Alpha_Score", "Momentum_Score", "Trend_Quality_Score", "Persistence_Score",
    "Alpha_Confidence", "Entry_Risk_Score", "Opportunity_Score", "ADX_DI_Trend_Score", "DI_Balance", "SMA_Regime_Score",
    "Absolute_SMA_Score", "Relative_SMA_Score", "SMA200d_Robust_Z_36M",
    "Regime_Dependent_SMAZ_Risk", "Perf12M_Extreme_Risk", "Base_Alpha",
    "SPY_vs_SMA40W_%", "SPY_Drawdown_52W_%", "SPY_Volatility_13W_%", "SPY_Volatility_Percentile",
]

PERFORMANCE_COLUMNS = [
    "Perf_1D_%",
    "Perf_1W_%",
    "Perf_1M_%",
    "Perf_3M_%",
    "Perf_6M_%",
    "Perf_12M_%",
    "Perf_3Y_%",
    "Perf_5Y_%",
    "Perf_10Y_%",
    "Avg_Forward_Return_6M_%",
]

FAST_PERFORMANCE_COLUMNS = [
    "CurrentPrice",
    "Perf_1D_%",
    "Perf_1W_%",
    "Perf_1M_%",
    "Perf_3M_%",
    "Perf_6M_%",
    "Perf_12M_%",
]

PERFORMANCE_REFERENCE_COLUMNS = [
    "PerfRef_1D",
    "PerfRef_1W",
    "PerfRef_1M",
    "PerfRef_3M",
    "PerfRef_6M",
    "PerfRef_12M",
]

SNAPSHOT_DIR = Path(__file__).with_name("persistent") / "snapshots"
JOB_STATUS_DIR = Path(__file__).with_name("persistent") / "job_status"
SCREENER_SNAPSHOT_MODEL_VERSION = "screener-snapshot-v1"

INVERSE_PERFORMANCE_COLUMNS = [
    "Perf_12M_Percentile",
    "SMA200W_Distance_Percentile",
    "Correction_Risk_%",
]

PERF_TOP5_MAP = {
    "Off": None,
    "Week": "Perf_1W_%",
    "Month": "Perf_1M_%",
    "3 months": "Perf_3M_%",
    "6 months": "Perf_6M_%",
    "12 months": "Perf_12M_%",
    "3 years": "Perf_3Y_%",
    "5 years": "Perf_5Y_%",
}

DIVERGENCE_DEFAULTS = {
    "pivot_window": 3,
    "lookback_bars": 60,
    "alignment_tolerance": 2,
    "min_price_move": 0.005,  # 0.5%
    "min_ind_move_rsi": 4.0,
    "min_ind_move_macd_std": 0.5,
    "min_ind_move_roc_std": 0.7,
    "rolling_std_n": 20,
    "max_span": 40,
    "wp": 0.35,
    "wi": 0.40,
    "wt": 0.15,
    "wz": 0.10,
    "eps": 1e-9,
}

DIVERGENCE_PROFILE_DEFAULTS = {
    "Weekly": {
        "pivot_window": 3,
        "lookback_bars": 60,
    },
    "Monthly": {
        "pivot_window": 5,
        "lookback_bars": 120,
    },
}


def clone_universe_map(universe_map: dict) -> dict:
    return deepcopy(universe_map)


def _clean_universe_block(block: dict) -> dict:
    cleaned = {}
    if not isinstance(block, dict):
        return cleaned
    for group, subgroups in block.items():
        group_name = str(group).strip()
        if not group_name or not isinstance(subgroups, dict):
            continue
        cleaned_subgroups = {}
        for subgroup, tickers in subgroups.items():
            subgroup_name = str(subgroup).strip()
            if not subgroup_name or not isinstance(tickers, (list, tuple)):
                continue
            unique = []
            for ticker in tickers:
                t = canonical_ai_group_label(ticker) if is_ai_group_label(ticker) else str(ticker).strip().upper()
                if t and t not in unique:
                    unique.append(t)
            if unique:
                cleaned_subgroups[subgroup_name] = unique
        if cleaned_subgroups:
            cleaned[group_name] = cleaned_subgroups
    return cleaned


def load_universe_map() -> dict:
    base = clone_universe_map(ETF_UNIVERSE_MAP)
    if not UNIVERSE_STORAGE_PATH.exists():
        return base
    try:
        raw = json.loads(UNIVERSE_STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return base
    if not isinstance(raw, dict):
        return base
    for list_name, block in raw.items():
        if not isinstance(list_name, str):
            continue
        cleaned_block = _clean_universe_block(block)
        if cleaned_block:
            base[list_name] = cleaned_block
    return base


def save_universe_map(universe_map: dict) -> None:
    payload = {}
    for list_name, block in universe_map.items():
        if isinstance(list_name, str):
            payload[list_name] = _clean_universe_block(block)
    UNIVERSE_STORAGE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def flatten_universe_for_editor(universe: dict) -> pd.DataFrame:
    rows = []
    for group, subgroups in universe.items():
        for subgroup, tickers in subgroups.items():
            for ticker in tickers:
                rows.append({"Group": group, "Subgroup": subgroup, "Ticker": ticker})
    return pd.DataFrame(rows, columns=["Group", "Subgroup", "Ticker"])


def snapshot_hash(*parts: Any) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def snapshot_paths(kind: str, key: str) -> tuple[Path, Path]:
    base = SNAPSHOT_DIR / f"{kind}_{key}"
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def read_snapshot_frame(kind: str, key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path, meta_path = snapshot_paths(kind, key)
    frame = pd.DataFrame()
    if data_path.exists():
        try:
            frame = pd.read_parquet(data_path)
        except Exception:
            frame = pd.DataFrame()
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return frame, metadata


def atomic_write_snapshot(kind: str, key: str, frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = snapshot_paths(kind, key)
    tmp_data = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    tmp_meta = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp_data, index=False)
    tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp_data, data_path)
    os.replace(tmp_meta, meta_path)


def snapshot_metadata(status: str, rows: int, data_as_of: Any = None, error: str | None = None) -> dict[str, Any]:
    metadata = {
        "Status": status,
        "Rows": int(rows),
        "CalculatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "Data_AsOf": None if data_as_of is None else str(data_as_of),
        "ModelVersion": SCREENER_SNAPSHOT_MODEL_VERSION,
    }
    if error:
        metadata["Error"] = str(error)
    return metadata


def background_job_running(job_name: str) -> bool:
    return (JOB_STATUS_DIR / f"{job_name}.lock").exists()


def start_refresh_job(job: str) -> tuple[bool, str]:
    job_name = job.replace("-", "_")
    if background_job_running(job_name):
        return False, f"{job_name} is already running"
    JOB_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).with_name("refresh_jobs.py")
    kwargs: dict[str, Any] = {
        "cwd": str(Path(__file__).parent),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([sys.executable, str(script_path), job], **kwargs)
    except Exception as exc:
        return False, str(exc)
    return True, f"{job_name} started"


def build_universe_from_editor_df(editor_df: pd.DataFrame) -> tuple[dict, list]:
    out = {}
    errors = []
    seen = set()
    for idx, row in editor_df.iterrows():
        group = str(row.get("Group", "")).strip()
        subgroup = str(row.get("Subgroup", "")).strip()
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not group and not subgroup and not ticker:
            continue
        if not group or not subgroup or not ticker:
            errors.append(f"Row {idx + 1}: Group, Subgroup and Ticker are required.")
            continue
        key = (group, subgroup, ticker)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(group, {}).setdefault(subgroup, []).append(ticker)
    return out, errors


# ============================================================
# 2) Robust Close extractor (always returns 1D Series)
# ============================================================
def extract_close_series(px: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Returns a 1D Series of closes from yfinance output.
    Handles:
      - single-level columns with 'Close'
      - MultiIndex columns: ('Close','SPY') or ('SPY','Close')
    """
    if px is None or px.empty:
        return pd.Series(dtype="float64")

    if isinstance(px.columns, pd.MultiIndex):
        cols = px.columns

        # (Field, Ticker)
        if "Close" in cols.get_level_values(0):
            c = px["Close"]
            if isinstance(c, pd.DataFrame):
                if ticker in c.columns:
                    return c[ticker].dropna()
                return c.iloc[:, 0].dropna()
            return c.dropna()

        # (Ticker, Field)
        if ticker in cols.get_level_values(0):
            sub = px[ticker]
            if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                return sub["Close"].dropna()
            if isinstance(sub, pd.Series):
                return sub.dropna()

        return pd.Series(dtype="float64")

    # single-level columns
    if "Close" in px.columns:
        c = px["Close"]
        if isinstance(c, pd.DataFrame):
            return c.iloc[:, 0].dropna()
        return c.dropna()

    return pd.Series(dtype="float64")


# ============================================================
# 3) Scalar-safe helpers
# ============================================================
def safe_value_on_or_before(series: pd.Series, dt: pd.Timestamp) -> float:
    s = series.dropna().sort_index()
    s = s.loc[:dt]
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def safe_last(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def safe_perf(close: pd.Series, end_dt: pd.Timestamp, days: int) -> float:
    start_dt = end_dt - pd.DateOffset(days=int(days))
    v0 = safe_value_on_or_before(close, start_dt)
    v1 = safe_value_on_or_before(close, end_dt)
    if np.isnan(v0) or np.isnan(v1) or v0 == 0.0:
        return np.nan
    return (v1 / v0 - 1) * 100.0


def pct_spread(sma_fast: float, sma_slow: float) -> float:
    if np.isnan(sma_fast) or np.isnan(sma_slow) or sma_slow == 0.0:
        return np.nan
    return (sma_fast / sma_slow - 1) * 100.0


def pct_change(v0: float, v1: float) -> float:
    if np.isnan(v0) or np.isnan(v1) or v0 == 0.0:
        return np.nan
    return (v1 / v0 - 1) * 100.0


def delta_last_n_bars(series: pd.Series, n: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - n])


def _wilder_smooth_avg(series: pd.Series, n: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype="float64")
    if len(vals) < n:
        return out
    first_n = vals.iloc[:n]
    if first_n.isna().any():
        return out
    out.iloc[n - 1] = float(first_n.mean())
    for i in range(n, len(vals)):
        cur = float(vals.iloc[i])
        prev = float(out.iloc[i - 1])
        out.iloc[i] = ((prev * (n - 1)) + cur) / n
    return out


def compute_adx_dmi_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    idx = c.index

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    up_move = h.diff()
    down_move = l.shift(1) - l

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=idx,
        dtype="float64",
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=idx,
        dtype="float64",
    )

    sm_tr = _wilder_smooth_avg(tr, period)
    sm_plus_dm = _wilder_smooth_avg(plus_dm, period)
    sm_minus_dm = _wilder_smooth_avg(minus_dm, period)

    plus_di = 100.0 * (sm_plus_dm / sm_tr.replace(0.0, np.nan))
    minus_di = 100.0 * (sm_minus_dm / sm_tr.replace(0.0, np.nan))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))

    adx = pd.Series(np.nan, index=idx, dtype="float64")
    start = period - 1
    if len(dx) >= start + period:
        first_window = dx.iloc[start:start + period].dropna()
        if len(first_window) == period:
            first_adx_idx = start + period - 1
            adx.iloc[first_adx_idx] = float(first_window.mean())
            for i in range(first_adx_idx + 1, len(dx)):
                cur_dx = float(dx.iloc[i]) if np.isfinite(dx.iloc[i]) else np.nan
                prev_adx = float(adx.iloc[i - 1]) if np.isfinite(adx.iloc[i - 1]) else np.nan
                if np.isfinite(cur_dx) and np.isfinite(prev_adx):
                    adx.iloc[i] = ((prev_adx * (period - 1)) + cur_dx) / period

    return adx, plus_di, minus_di


def build_weekly_ohlcv_from_daily(ohlcv: pd.DataFrame, include_partial_last_week: bool = False) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    wk = (
        ohlcv.resample("W-FRI")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Close"])
    )
    if wk.empty:
        return wk

    # Optional exclusion of partial current week to avoid look-ahead.
    if not include_partial_last_week:
        last_daily_ts = pd.Timestamp(ohlcv.index.max())
        if pd.Timestamp(wk.index[-1]) > last_daily_ts:
            wk = wk.iloc[:-1]
    return wk


def detect_recent_sma_crossover(sma50: pd.Series, sma200: pd.Series, lookback_bars: int = 14) -> tuple[bool, bool]:
    pair = pd.concat([sma50, sma200], axis=1, keys=["sma50", "sma200"]).dropna()
    if len(pair) < 2:
        return False, False

    cur = pair["sma50"] > pair["sma200"]
    prev = pair["sma50"].shift(1) > pair["sma200"].shift(1)
    golden_events = cur & (~prev.fillna(False))
    death_events = (~cur) & prev.fillna(False)

    window = max(1, int(lookback_bars))
    golden_recent = bool(golden_events.tail(window).any())
    death_recent = bool(death_events.tail(window).any())
    return golden_recent, death_recent


def compute_weekly_bb_position(
    weekly_close: pd.Series,
    period: int = 50,
    std_mult: float = 2.0,
) -> tuple[float, float, float, float, float, float, float]:
    wc = pd.to_numeric(weekly_close, errors="coerce").dropna()
    if len(wc) < period:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mid = wc.rolling(window=period, min_periods=period).mean()
    std = wc.rolling(window=period, min_periods=period).std()
    bb = pd.concat([wc.rename("close"), mid.rename("mid"), std.rename("std")], axis=1).dropna()
    if bb.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    last = bb.iloc[-1]
    c = float(last["close"])
    m = float(last["mid"])
    s = float(last["std"])
    if (not np.isfinite(s)) or s <= 0:
        return 0, m, np.nan, np.nan, np.nan, np.nan, c

    upper = m + std_mult * s
    lower = m - std_mult * s

    eps = 1e-12
    step_up = (upper - m) / 10.0
    step_dn = (m - lower) / 10.0
    if step_up <= eps or step_dn <= eps:
        return 0, m, upper, lower, step_up, step_dn, c

    if c >= upper:
        return 10, m, upper, lower, step_up, step_dn, c
    if c <= lower:
        return -10, m, upper, lower, step_up, step_dn, c

    if c > m:
        pos = int(np.ceil((c - m) / step_up))
        return int(clamp(pos, 1, 9)), m, upper, lower, step_up, step_dn, c
    if c < m:
        pos = int(np.ceil((m - c) / step_dn))
        return -int(clamp(pos, 1, 9)), m, upper, lower, step_up, step_dn, c
    return 0, m, upper, lower, step_up, step_dn, c


def classify_divergence(price_ret_pct: float, ind_delta: float, eps: float = 0.0) -> str:
    """
    bull divergence: price falling and indicator rising
    bear divergence: price rising and indicator falling
    otherwise: nothing
    """
    if np.isnan(price_ret_pct) or np.isnan(ind_delta):
        return np.nan
    if (price_ret_pct < -eps) and (ind_delta > eps):
        return "bull"
    if (price_ret_pct > eps) and (ind_delta < -eps):
        return "bear"
    return "nothing"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _find_pivot_indices(series: pd.Series, left: int, right: int, mode: str) -> list:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(vals)
    pivots = []
    if n == 0:
        return pivots
    left = max(1, int(left))
    right = max(1, int(right))
    for i in range(left, n - right):
        v = vals[i]
        if not np.isfinite(v):
            continue
        window = vals[i - left:i + right + 1]
        if np.isnan(window).any():
            continue
        if mode == "low":
            if v == np.min(window):
                pivots.append(i)
        else:
            if v == np.max(window):
                pivots.append(i)
    return pivots


def _map_price_pivots_to_indicator(
    price_series: pd.Series,
    ind_series: pd.Series,
    pivot_window: int,
    k_align: int,
    mode: str,
) -> list:
    price_pivots = _find_pivot_indices(price_series, pivot_window, pivot_window, mode)
    ind_pivots = _find_pivot_indices(ind_series, pivot_window, pivot_window, mode)
    if not price_pivots or not ind_pivots:
        return []

    ind_vals = pd.to_numeric(ind_series, errors="coerce").to_numpy(dtype=float)
    price_vals = pd.to_numeric(price_series, errors="coerce").to_numpy(dtype=float)
    mapped = []
    for p_idx in price_pivots:
        candidates = [j for j in ind_pivots if abs(j - p_idx) <= k_align]
        if not candidates:
            continue
        i_idx = min(candidates, key=lambda x: abs(x - p_idx))
        p_val = price_vals[p_idx]
        i_val = ind_vals[i_idx]
        if not np.isfinite(p_val) or not np.isfinite(i_val):
            continue
        mapped.append(
            {
                "t_price": int(p_idx),
                "t_ind": int(i_idx),
                "price": float(p_val),
                "ind": float(i_val),
            }
        )
    mapped.sort(key=lambda x: x["t_price"])
    return mapped


def _indicator_zone_bonus(name: str, ind1: float, ind2: float, ind_std: float, direction: str) -> float:
    if name == "RSI":
        if direction == "bull" and min(ind1, ind2) < 30:
            return 1.0
        if direction == "bear" and max(ind1, ind2) > 70:
            return 1.0
        return 0.0

    opposite_side_zero = ind1 * ind2 < 0
    near_extreme = (ind_std > 0.0) and (max(abs(ind1), abs(ind2)) >= 1.5 * ind_std)
    return 1.0 if (opposite_side_zero or near_extreme) else 0.0


def detect_divergence_for_indicator(
    low: pd.Series,
    high: pd.Series,
    indicator: pd.Series,
    indicator_name: str,
    cfg: dict,
) -> tuple:
    pv = int(cfg["pivot_window"])
    k = int(cfg["alignment_tolerance"])
    lookback_bars = int(cfg["lookback_bars"])
    max_span = int(cfg["max_span"])
    min_price_move = float(cfg["min_price_move"])
    std_n = int(cfg["rolling_std_n"])
    eps = float(cfg["eps"])
    wp = float(cfg["wp"])
    wi = float(cfg["wi"])
    wt = float(cfg["wt"])
    wz = float(cfg["wz"])

    ind_std_series = pd.to_numeric(indicator, errors="coerce").rolling(std_n, min_periods=5).std()
    low_pairs = _map_price_pivots_to_indicator(low, indicator, pv, k, mode="low")
    high_pairs = _map_price_pivots_to_indicator(high, indicator, pv, k, mode="high")

    n = len(indicator)
    min_t = max(0, n - lookback_bars)
    candidates = []

    def add_candidate(pair1: dict, pair2: dict, side: str):
        t1 = int(pair1["t_price"])
        t2 = int(pair2["t_price"])
        if t2 <= t1:
            return
        if t2 < min_t:
            return
        if (t2 - t1) > max_span:
            return

        p1 = float(pair1["price"])
        p2 = float(pair2["price"])
        i1 = float(pair1["ind"])
        i2 = float(pair2["ind"])
        if p1 == 0.0:
            return

        if indicator_name == "RSI":
            min_ind_move = float(cfg["min_ind_move_rsi"])
        elif indicator_name == "MACD_HIST":
            std_here = float(ind_std_series.iloc[t2]) if t2 < len(ind_std_series) else np.nan
            min_ind_move = float(cfg["min_ind_move_macd_std"]) * (std_here + eps) if np.isfinite(std_here) else np.inf
        else:  # ROC
            std_here = float(ind_std_series.iloc[t2]) if t2 < len(ind_std_series) else np.nan
            min_ind_move = float(cfg["min_ind_move_roc_std"]) * (std_here + eps) if np.isfinite(std_here) else np.inf

        if abs((p2 - p1) / p1) < min_price_move:
            return
        if abs(i2 - i1) < min_ind_move:
            return

        div_type = None
        direction = None
        if side == "low":
            # Bullish divergences use price lows.
            if (p2 < p1) and (i2 > i1):
                div_type = "regular_bull"
                direction = "bull"
            elif (p2 > p1) and (i2 < i1):
                div_type = "hidden_bull"
                direction = "bull"
        else:
            # Bearish divergences use price highs.
            if (p2 > p1) and (i2 < i1):
                div_type = "regular_bear"
                direction = "bear"
            elif (p2 < p1) and (i2 > i1):
                div_type = "hidden_bear"
                direction = "bear"

        if div_type is None:
            return

        mp = abs((p2 - p1) / p1)
        std_here = float(ind_std_series.iloc[t2]) if t2 < len(ind_std_series) else np.nan
        std_here = std_here if np.isfinite(std_here) else 0.0
        mi = abs(i2 - i1) / (std_here + eps)
        ts = clamp((t2 - t1) / max_span, 0.0, 1.0)
        zone_bonus = _indicator_zone_bonus(indicator_name, i1, i2, std_here, direction)
        score = 100.0 * clamp(wp * mp + wi * mi + wt * ts + wz * zone_bonus, 0.0, 1.0)

        candidates.append(
            {
                "direction": direction,
                "type": div_type,
                "score": float(score),
                "t1": t1,
                "t2": t2,
            }
        )

    for i in range(1, len(low_pairs)):
        add_candidate(low_pairs[i - 1], low_pairs[i], side="low")
    for i in range(1, len(high_pairs)):
        add_candidate(high_pairs[i - 1], high_pairs[i], side="high")

    if not candidates:
        return "nothing", np.nan, "none"

    best = max(candidates, key=lambda x: x["score"])
    return best["direction"], best["score"], best["type"]


# ============================================================
# 4) Metrics function (same logic)
# ============================================================
@st.cache_data(show_spinner=False, ttl=SLOW_REFRESH_SECONDS)
def build_ai_group_ohlcv(group_label: str, period: str = "10y") -> pd.DataFrame:
    canonical = canonical_ai_group_label(group_label)
    members = AI_UNIVERSE.get(canonical, [])
    normalized_frames: dict[str, pd.DataFrame] = {}
    for member in members:
        frame = download_completed_ohlcv(member, period=period)
        if frame.empty:
            continue
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        if close.empty:
            continue
        base = float(close.iloc[0])
        if not np.isfinite(base) or base == 0.0:
            continue
        normalized = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        for column in ["Open", "High", "Low", "Close"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce") / base * 100.0
        normalized["Volume"] = pd.to_numeric(normalized["Volume"], errors="coerce")
        normalized_frames[member] = normalized

    if not normalized_frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    combined = pd.concat(normalized_frames, axis=1).sort_index()
    out = pd.DataFrame(index=combined.index)
    for column in ["Open", "High", "Low", "Close"]:
        out[column] = combined.xs(column, axis=1, level=1).mean(axis=1, skipna=True)
    out["Volume"] = combined.xs("Volume", axis=1, level=1).sum(axis=1, min_count=1)
    return out.dropna(subset=["Close"]).sort_index()


@st.cache_data(show_spinner=False, ttl=AUTO_REFRESH_SECONDS)
def build_ai_group_latest_ohlcv(group_label: str, period: str = "10y", refresh_bucket: int = 0) -> pd.DataFrame:
    _ = refresh_bucket
    canonical = canonical_ai_group_label(group_label)
    members = AI_UNIVERSE.get(canonical, [])
    normalized_frames: dict[str, pd.DataFrame] = {}
    for member in members:
        frame = download_latest_ohlcv(member, period=period)
        if frame.empty:
            continue
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
        if close.empty:
            continue
        base = float(close.iloc[0])
        if not np.isfinite(base) or base == 0.0:
            continue
        normalized = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        for column in ["Open", "High", "Low", "Close"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce") / base * 100.0
        normalized["Volume"] = pd.to_numeric(normalized["Volume"], errors="coerce")
        normalized_frames[member] = normalized

    if not normalized_frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    combined = pd.concat(normalized_frames, axis=1).sort_index()
    out = pd.DataFrame(index=combined.index)
    for column in ["Open", "High", "Low", "Close"]:
        out[column] = combined.xs(column, axis=1, level=1).mean(axis=1, skipna=True)
    out["Volume"] = combined.xs("Volume", axis=1, level=1).sum(axis=1, min_count=1)
    return out.dropna(subset=["Close"]).sort_index()


def download_metrics_ohlcv(ticker: str, period: str = "10y") -> pd.DataFrame:
    if is_ai_group_label(ticker):
        return build_ai_group_ohlcv(canonical_ai_group_label(ticker), period=period)
    return download_completed_ohlcv(ticker, period=period)


def download_performance_ohlcv(ticker: str, period: str = "10y", refresh_bucket: int = 0) -> pd.DataFrame:
    if is_ai_group_label(ticker):
        return build_ai_group_latest_ohlcv(canonical_ai_group_label(ticker), period=period, refresh_bucket=refresh_bucket)
    return download_latest_ohlcv(ticker, period=period)


def performance_reference_prices(close: pd.Series, today: pd.Timestamp) -> dict[str, float]:
    windows = {
        "PerfRef_1D": 1,
        "PerfRef_1W": 7,
        "PerfRef_1M": 30,
        "PerfRef_3M": 90,
        "PerfRef_6M": 182,
        "PerfRef_12M": 365,
    }
    return {key: safe_value_on_or_before(close, today - pd.Timedelta(days=days)) for key, days in windows.items()}


def lightweight_current_price(ticker: str, refresh_bucket: int = 0) -> float:
    _ = refresh_bucket
    if is_ai_group_label(ticker):
        return np.nan
    for attempt in range(2):
        try:
            raw = yf.download(
                ticker,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            frame = extract_ohlcv_frame(raw, ticker)
            if is_krw_quoted_ticker(ticker):
                fx_raw = yf.download(
                    USD_KRW_TICKER,
                    period="5d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                frame = convert_krw_ohlcv_to_usd(frame, extract_ohlcv_frame(fx_raw, USD_KRW_TICKER))
            close = pd.to_numeric(frame.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])
        except Exception:
            pass
        time.sleep(0.25 * (attempt + 1))
    return np.nan


def overlay_performance_from_refs(row: pd.Series, current_price: float) -> dict[str, float]:
    out = {"CurrentPrice": current_price}
    mapping = {
        "Perf_1D_%": "PerfRef_1D",
        "Perf_1W_%": "PerfRef_1W",
        "Perf_1M_%": "PerfRef_1M",
        "Perf_3M_%": "PerfRef_3M",
        "Perf_6M_%": "PerfRef_6M",
        "Perf_12M_%": "PerfRef_12M",
    }
    for perf_col, ref_col in mapping.items():
        ref = pd.to_numeric(pd.Series([row.get(ref_col)]), errors="coerce").iloc[0]
        if np.isfinite(current_price) and np.isfinite(ref) and ref != 0.0:
            out[perf_col] = (current_price / ref - 1.0) * 100.0
        else:
            out[perf_col] = row.get(perf_col, np.nan)
    return out


def get_metrics(ticker: str, divergence_cfg: dict):
    ohlcv = download_metrics_ohlcv(ticker)
    if ohlcv.empty:
        return None

    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    low = pd.to_numeric(ohlcv["Low"], errors="coerce").dropna()
    high = pd.to_numeric(ohlcv["High"], errors="coerce").dropna()
    base = pd.concat([close, low, high], axis=1, join="inner").dropna()
    if base.empty:
        return None
    close = base["Close"]
    low = base["Low"]
    high = base["High"]
    if close.empty or len(close) < 60:
        return None

    try:
        today = close.index[-1]
        cur_px = float(close.iloc[-1])

        # Performance (calendar-day approximations)
        perf_1d = safe_perf(close, today, 1)
        perf_1w = safe_perf(close, today, 7)
        perf_1m = safe_perf(close, today, 30)
        perf_3m = safe_perf(close, today, 90)
        perf_6m = safe_perf(close, today, 182)
        perf_12m = safe_perf(close, today, 365)
        perf_12m_percentile, avg_forward_return_6m = historical_momentum_52w_metrics(close)
        perf_3y = safe_perf(close, today, 365 * 3)
        perf_5y = safe_perf(close, today, 365 * 5)
        perf_10y = safe_perf(close, today, 365 * 10)
        perf_refs = performance_reference_prices(close, today)

        # 52W high distance
        last_52w = close.loc[today - pd.DateOffset(days=int(365 * 1.1)):]
        if last_52w.empty:
            vs_52w = np.nan
        else:
            high_52w = float(last_52w.max())
            vs_52w = np.nan if high_52w == 0.0 else (cur_px / high_52w - 1) * 100.0
        ath = float(high.max()) if not high.empty else np.nan
        vs_ath = np.nan if np.isnan(ath) or ath == 0.0 else (cur_px / ath - 1.0) * 100.0

        # Indicators
        rsi14 = RSIIndicator(close=close, window=14).rsi()
        macd_hist = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
        roc12 = ROCIndicator(close=close, window=12).roc()
        adx14, di_plus14, di_minus14 = compute_adx_dmi_wilder(high=high, low=low, close=close, period=14)

        cur_rsi = safe_last(rsi14)
        cur_adx14 = safe_last(adx14)
        cur_di_plus14 = safe_last(di_plus14)
        cur_di_minus14 = safe_last(di_minus14)
        cur_di_plus14_delta2 = delta_last_n_bars(di_plus14, 2)
        cur_di_minus14_delta2 = delta_last_n_bars(di_minus14, 2)
        div_rsi, _, _ = detect_divergence_for_indicator(low, high, rsi14, "RSI", divergence_cfg)
        div_macd, _, _ = detect_divergence_for_indicator(low, high, macd_hist, "MACD_HIST", divergence_cfg)
        div_roc, _, _ = detect_divergence_for_indicator(low, high, roc12, "ROC", divergence_cfg)

        # Price vs SMA200d (%) and 6M % change
        cutoff_6m = today - pd.DateOffset(days=182)
        sma200d = close.rolling(window=200, min_periods=200).mean()
        sma200d_now = safe_last(sma200d)
        spread_pct_now = pct_spread(cur_px, sma200d_now)
        spread_series_pct = (close / sma200d - 1.0) * 100.0
        spread_valid = spread_series_pct.dropna()
        spread_avg_36m = np.nan
        if not spread_valid.empty:
            # 36 months ~= 756 trading days; use shorter available history when needed.
            spread_avg_36m = float(spread_valid.tail(756).mean())
        sma200d_robust_z_36m = calculate_sma200d_robust_z_36m(close)

        spread_pct_6m_ago = safe_value_on_or_before(spread_series_pct, cutoff_6m)

        spread_pct_change_6m = pct_change(spread_pct_6m_ago, spread_pct_now)

        sma_trend = np.nan
        if not np.isnan(spread_pct_now) and not np.isnan(spread_pct_6m_ago):
            sma_trend = "bull" if (spread_pct_now - spread_pct_6m_ago) > 0 else "bear"

        # SMA50/200 crossover detection (no look-ahead).
        sma50d = close.rolling(window=50, min_periods=50).mean()
        golden_cross_d1 = False
        death_cross_d1 = False
        if len(close) >= 250:
            golden_cross_d1, death_cross_d1 = detect_recent_sma_crossover(sma50d, sma200d, lookback_bars=14)

        wk_source = ohlcv[["Open", "High", "Low", "Close", "Volume"]]
        # Use completed weekly bars only (no look-ahead).
        wk_ohlcv = build_weekly_ohlcv_from_daily(wk_source, include_partial_last_week=False)
        golden_cross_w1 = False
        death_cross_w1 = False
        wk_close = pd.to_numeric(wk_ohlcv["Close"], errors="coerce").dropna()
        rsi14w = RSIIndicator(close=wk_close, window=14).rsi() if len(wk_close) >= 14 else pd.Series(dtype="float64")
        cur_rsi14w = safe_last(rsi14w)
        (
            bb_position,
            bb_mid,
            bb_upper,
            bb_lower,
            bb_step_up,
            bb_step_down,
            wk_close_last,
        ) = compute_weekly_bb_position(wk_close, period=50, std_mult=2.0)
        sma200w_percentile = sma200w_distance_percentile(wk_close)
        correction_risk, _, _ = correction_risk_from_percentile_analogs(close, wk_close)
        if len(wk_close) >= 260:
            sma50w = wk_close.rolling(window=50, min_periods=50).mean()
            sma200w = wk_close.rolling(window=200, min_periods=200).mean()
            golden_cross_w1, death_cross_w1 = detect_recent_sma_crossover(sma50w, sma200w, lookback_bars=14)

        try:
            fund_flow_metrics = None if is_ai_group_label(ticker) else get_fund_flow_metrics(ticker)
        except Exception:
            fund_flow_metrics = None
        flows_1m = np.nan if fund_flow_metrics is None else fund_flow_metrics.flow_1m_pct
        flows_3m = np.nan if fund_flow_metrics is None else fund_flow_metrics.flow_3m_pct

        return [
            cur_px,
            perf_1d, perf_1w, perf_1m, perf_3m, perf_6m,
            perf_12m, perf_3y, perf_5y, perf_10y,
            *[perf_refs.get(col, np.nan) for col in PERFORMANCE_REFERENCE_COLUMNS],
            sma200w_percentile, perf_12m_percentile, avg_forward_return_6m, correction_risk,
            flows_1m, flows_3m,
            vs_52w, vs_ath, cur_rsi, cur_rsi14w,
            spread_pct_now, spread_avg_36m, spread_pct_change_6m, sma200d_robust_z_36m, sma_trend,
            cur_adx14, cur_di_plus14, cur_di_minus14, cur_di_plus14_delta2, cur_di_minus14_delta2,
            bb_position, bb_mid, bb_upper, bb_lower, bb_step_up, bb_step_down, wk_close_last,
            int(golden_cross_d1), int(death_cross_d1), int(golden_cross_w1), int(death_cross_w1),
            div_rsi, div_macd, div_roc
        ]
    except Exception:
        return None


def get_performance_metrics(ticker: str, refresh_bucket: int = 0) -> list[float] | None:
    if is_ai_group_label(ticker):
        return get_ai_group_performance_metrics(canonical_ai_group_label(ticker), refresh_bucket=refresh_bucket)
    ohlcv = download_performance_ohlcv(ticker, period="15mo", refresh_bucket=refresh_bucket)
    if ohlcv.empty:
        return None
    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    today = close.index[-1]
    return [
        float(close.iloc[-1]),
        safe_perf(close, today, 1),
        safe_perf(close, today, 7),
        safe_perf(close, today, 30),
        safe_perf(close, today, 90),
        safe_perf(close, today, 182),
        safe_perf(close, today, 365),
    ]


def get_ai_group_performance_metrics(group_label: str, refresh_bucket: int = 0) -> list[float] | None:
    windows = [1, 7, 30, 90, 182, 365]
    member_returns: list[list[float]] = []
    current_prices: list[float] = []
    for member in AI_UNIVERSE.get(canonical_ai_group_label(group_label), []):
        ohlcv = download_performance_ohlcv(member, period="15mo", refresh_bucket=refresh_bucket)
        if ohlcv.empty:
            continue
        close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        end = pd.Timestamp(close.index[-1])
        current_prices.append(float(close.iloc[-1]))
        member_returns.append([safe_perf(close, end, days) for days in windows])
    if not member_returns:
        return None
    frame = pd.DataFrame(member_returns, columns=[col for col in FAST_PERFORMANCE_COLUMNS if col != "CurrentPrice"])
    frame.insert(0, "CurrentPrice", np.nan if not current_prices else float(np.nanmean(current_prices)))
    return [float(pd.to_numeric(frame[col], errors="coerce").mean(skipna=True)) for col in FAST_PERFORMANCE_COLUMNS]


def compute_performance_table(slow_df: pd.DataFrame, overlay_key: str, refresh_nonce: int = 0) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    previous_overlay, previous_meta = read_snapshot_frame("market_performance_latest", overlay_key)
    previous_by_key: dict[tuple[str, str, str], pd.Series] = {}
    if not previous_overlay.empty:
        for _, prev_row in previous_overlay.iterrows():
            previous_by_key[(str(prev_row.get("Group")), str(prev_row.get("Subgroup")), str(prev_row.get("Ticker")))] = prev_row

    refresh_bucket = int(refresh_nonce)
    previous_bucket = int(previous_meta.get("RefreshBucket", -1)) if str(previous_meta.get("RefreshBucket", "")).lstrip("-").isdigit() else -1
    if not previous_overlay.empty and previous_bucket == refresh_bucket:
        return previous_overlay, str(previous_meta.get("CalculatedAt", "")), previous_meta

    columns = ["Group", "Subgroup", "Ticker", *FAST_PERFORMANCE_COLUMNS]
    rows = []
    for _, row in slow_df.iterrows():
        group = str(row.get("Group", ""))
        subgroup = str(row.get("Subgroup", ""))
        ticker = str(row.get("Ticker", ""))
        key = (group, subgroup, ticker)
        previous = previous_by_key.get(key)
        current_price = lightweight_current_price(ticker, refresh_bucket=refresh_bucket)
        if not np.isfinite(current_price):
            current_price = pd.to_numeric(pd.Series([previous.get("CurrentPrice") if previous is not None else row.get("CurrentPrice")]), errors="coerce").iloc[0]
        values = overlay_performance_from_refs(row, current_price)
        if previous is not None:
            for col in FAST_PERFORMANCE_COLUMNS:
                if not np.isfinite(pd.to_numeric(pd.Series([values.get(col)]), errors="coerce").iloc[0]):
                    values[col] = previous.get(col, row.get(col, np.nan))
        rows.append([group, subgroup, ticker] + [values.get(col, np.nan) for col in FAST_PERFORMANCE_COLUMNS])
    fetched_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    overlay = pd.DataFrame(rows, columns=columns)
    meta = snapshot_metadata("CURRENT", len(overlay), error=None)
    meta["RefreshBucket"] = int(refresh_bucket)
    try:
        atomic_write_snapshot("market_performance_latest", overlay_key, overlay, meta)
    except Exception as exc:
        if not previous_overlay.empty:
            fallback = snapshot_metadata("UPDATE_FAILED_USING_PREVIOUS", len(previous_overlay), error=str(exc))
            fallback["RefreshBucket"] = previous_bucket
            return previous_overlay, str(previous_meta.get("CalculatedAt", "")), fallback
        meta = snapshot_metadata("WRITE_FAILED", len(overlay), error=str(exc))
    return overlay, fetched_at_utc, meta


@st.cache_data(show_spinner=True, ttl=SLOW_REFRESH_SECONDS)
def compute_slow_metrics_table(
    universe: dict,
    universe_signature: str,
    divergence_cfg: dict,
    divergence_signature: str,
    refresh_nonce: int = 0,
) -> tuple[pd.DataFrame, str]:
    _ = universe_signature
    _ = divergence_signature
    _ = refresh_nonce
    columns = [
        "Group", "Subgroup", "Ticker",
        "CurrentPrice",
        "Perf_1D_%", "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%",
        "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%", "Perf_10Y_%",
        *PERFORMANCE_REFERENCE_COLUMNS,
        "SMA200W_Distance_Percentile", "Perf_12M_Percentile", "Avg_Forward_Return_6M_%", "Correction_Risk_%",
        "FundFlows_1M_%", "FundFlows_3M_%",
        "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14", "RSI_14W",
        "SMA50w_vs_SMA200w_Spread_%", "SMA50w_vs_SMA200w_Spread_Avg_36M_%", "SMA_Spread_%_Change_6M_%",
        "SMA200d_Robust_Z_36M", "SMA_Trend",
        "ADX_14", "DI_Plus_14", "DI_Minus_14", "DI_Plus_14_Delta2", "DI_Minus_14_Delta2",
        "BB_Position", "BB_Mid", "BB_Upper", "BB_Lower", "BB_StepUp", "BB_StepDown", "WeeklyClose_Last",
        "GoldenCross_D1", "DeathCross_D1", "GoldenCross_W1", "DeathCross_W1",
        "Div_6M_vs_RSI", "Div_6M_vs_MACD", "Div_6M_vs_ROC",
    ]
    rows = []
    for group, subgroups in universe.items():
        for subgroup, tickers in subgroups.items():
            for ticker in tickers:
                res = get_metrics(ticker, divergence_cfg)
                if res is None:
                    rows.append([group, subgroup, ticker] + [np.nan] * (len(columns) - 3))
                else:
                    rows.append([group, subgroup, ticker] + res)

    df = pd.DataFrame(rows, columns=columns)

    df["Divergence_Bull_Count"] = (
        (df["Div_6M_vs_RSI"] == "bull").astype(int)
        + (df["Div_6M_vs_MACD"] == "bull").astype(int)
        + (df["Div_6M_vs_ROC"] == "bull").astype(int)
    )
    df["Divergence_Bear_Count"] = (
        (df["Div_6M_vs_RSI"] == "bear").astype(int)
        + (df["Div_6M_vs_MACD"] == "bear").astype(int)
        + (df["Div_6M_vs_ROC"] == "bear").astype(int)
    )
    market = load_market_model_snapshot(f"market-model:{universe_signature}:{divergence_signature}")
    df = calculate_alpha_engine(df, market_regime=market)

    fetched_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    return df, fetched_at_utc


@st.cache_data(show_spinner=True, ttl=AUTO_REFRESH_SECONDS)
def compute_metrics_table(
    universe: dict,
    universe_signature: str,
    divergence_cfg: dict,
    divergence_signature: str,
    performance_refresh_nonce: int = 0,
    slow_refresh_nonce: int = 0,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    snapshot_key = snapshot_hash(universe_signature, divergence_signature)
    slow_df, slow_meta = read_snapshot_frame("screener_snapshot_latest", snapshot_key)
    previous_refresh_nonce = int(slow_meta.get("RefreshNonce", -1)) if str(slow_meta.get("RefreshNonce", "")).lstrip("-").isdigit() else -1
    requested_inline_refresh = int(slow_refresh_nonce) > 0 and int(slow_refresh_nonce) > previous_refresh_nonce
    allow_inline_refresh = os.getenv("SCREENER_ALLOW_INLINE_ANALYTICS", "0").strip().lower() in {"1", "true", "yes"}
    should_rebuild_snapshot = (slow_df.empty or requested_inline_refresh) and allow_inline_refresh
    if should_rebuild_snapshot:
        try:
            slow_df, slow_fetched_at_utc = compute_slow_metrics_table(
                universe,
                universe_signature,
                divergence_cfg,
                divergence_signature,
                slow_refresh_nonce,
            )
            if not slow_df.empty:
                data_as_of = None
                if "Ticker" in slow_df.columns:
                    data_as_of = f"{slow_df['Ticker'].nunique()} tickers"
                slow_meta = snapshot_metadata("CURRENT", len(slow_df), data_as_of=data_as_of)
                slow_meta["RefreshNonce"] = int(slow_refresh_nonce)
                slow_meta["SourceCalculatedAt"] = slow_fetched_at_utc
                atomic_write_snapshot("screener_snapshot_latest", snapshot_key, slow_df, slow_meta)
        except Exception as exc:
            fallback, fallback_meta = read_snapshot_frame("screener_snapshot_latest", snapshot_key)
            if not fallback.empty:
                slow_df = fallback
                slow_meta = dict(fallback_meta)
                slow_meta["Status"] = "NIGHTLY_UPDATE_FAILED_USING_PREVIOUS"
                slow_meta["Error"] = str(exc)
            else:
                raise
    elif slow_df.empty:
        slow_meta = snapshot_metadata(
            "SNAPSHOT_MISSING_WAITING_FOR_BACKGROUND_JOB",
            0,
            error="No screener snapshot found. Run the background nightly analytics job.",
        )
    elif requested_inline_refresh:
        slow_meta = dict(slow_meta)
        slow_meta["Status"] = "BACKGROUND_REFRESH_REQUESTED"

    perf_df, performance_fetched_at_utc, perf_meta = compute_performance_table(
        slow_df,
        snapshot_key,
        performance_refresh_nonce,
    )
    status_payload = {
        "analytics": slow_meta,
        "prices": perf_meta,
    }
    if slow_df.empty:
        return perf_df, performance_fetched_at_utc, status_payload
    if perf_df.empty:
        return slow_df, str(slow_meta.get("CalculatedAt", "")), status_payload

    keys = ["Group", "Subgroup", "Ticker"]
    merged = slow_df.drop(columns=FAST_PERFORMANCE_COLUMNS, errors="ignore").merge(
        perf_df,
        on=keys,
        how="left",
    )
    ordered_columns = [col for col in slow_df.columns if col in merged.columns] + [
        col for col in merged.columns if col not in slow_df.columns
    ]
    return merged[ordered_columns], performance_fetched_at_utc, status_payload


def apply_filters(df: pd.DataFrame):
    groups_all = sorted(df["Group"].dropna().unique().tolist())
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = st.columns(
        [1.05, 1.05, 1.0, 0.85, 1.0, 1.1, 1.1, 1.2, 1.2, 1.0, 0.95, 0.9]
    )

    with c1:
        selected_group = st.selectbox("Group", options=["All"] + groups_all, index=0)

    df_group = df if selected_group == "All" else df[df["Group"] == selected_group]

    subgroups_all = sorted(df_group["Subgroup"].dropna().unique().tolist())
    with c2:
        selected_subgroup = st.selectbox("Subgroup", options=["All"] + subgroups_all, index=0)

    filtered = df_group if selected_subgroup == "All" else df_group[df_group["Subgroup"] == selected_subgroup]

    with c3:
        rsi_option = st.selectbox(
            "RSI filter",
            options=["Off", "RSI < 30", "RSI > 70", "RSI < 30 OR RSI > 70"],
            index=0,
        )

    with c4:
        sma_option = st.selectbox("SMA Trend", options=["Off", "Bull", "Bear", "Both"], index=0)

    with c5:
        divergence_option = st.selectbox(
            "Momentum divergence",
            options=["Off", "2 bull", "3 bull", "2 bear", "3 bear"],
            index=0,
        )

    with c6:
        perf_top5_label = st.selectbox("Top 10 performing ETFs", options=list(PERF_TOP5_MAP.keys()), index=0)

    with c7:
        flow_top5_enabled = st.selectbox("Top 5 by Fund Flows (3M)", options=["Off", "On"], index=0) == "On"

    with c8:
        adx_trend_filter = st.selectbox(
            "ADX / DI Trend",
            options=["Off", "ADX & +DI (Bullish Trend)", "ADX & -DI (Bearish Trend)"],
            index=0,
        )
    with c9:
        cross_filter = st.selectbox(
            "SMA Crossovers",
            options=[
                "Off",
                "Daily Golden Cross (D1)",
                "Daily Death Cross (D1)",
                "Weekly Golden Cross (W1)",
                "Weekly Death Cross (W1)",
            ],
            index=0,
        )
    with c10:
        bb_filter = st.selectbox(
            "BB filter",
            options=["Off", "Overbought (>= +5)", "Oversold (<= -5)"],
            index=0,
        )
    with c11:
        alpha_sort = st.selectbox(
            "Sort by",
            options=["Off", "Alpha Score", "Opportunity State", "Entry Risk", "Alpha Confidence", "Opportunity Score"],
            index=0,
        )
    with c12:
        alpha_top = st.selectbox("Top Alpha", options=["All", "Top 5", "Top 10", "Top 20"], index=0)

    # Apply non-top performance filters next
    if rsi_option == "RSI < 30":
        filtered = filtered[filtered["RSI_14"] < 30]
    elif rsi_option == "RSI > 70":
        filtered = filtered[filtered["RSI_14"] > 70]
    elif rsi_option == "RSI < 30 OR RSI > 70":
        filtered = filtered[(filtered["RSI_14"] < 30) | (filtered["RSI_14"] > 70)]

    if sma_option == "Bull":
        filtered = filtered[filtered["SMA_Trend"] == "bull"]
    elif sma_option == "Bear":
        filtered = filtered[filtered["SMA_Trend"] == "bear"]
    elif sma_option == "Both":
        filtered = filtered[filtered["SMA_Trend"].isin(["bull", "bear"])]

    if divergence_option == "2 bull":
        filtered = filtered[filtered["Divergence_Bull_Count"] >= 2]
    elif divergence_option == "3 bull":
        filtered = filtered[filtered["Divergence_Bull_Count"] == 3]
    elif divergence_option == "2 bear":
        filtered = filtered[filtered["Divergence_Bear_Count"] >= 2]
    elif divergence_option == "3 bear":
        filtered = filtered[filtered["Divergence_Bear_Count"] == 3]

    if adx_trend_filter == "ADX & +DI (Bullish Trend)":
        filtered = filtered[(filtered["ADX_14"] > 25) & (filtered["DI_Plus_14"] > filtered["DI_Minus_14"])]
    elif adx_trend_filter == "ADX & -DI (Bearish Trend)":
        filtered = filtered[(filtered["ADX_14"] > 25) & (filtered["DI_Minus_14"] > filtered["DI_Plus_14"])]

    if cross_filter == "Daily Golden Cross (D1)":
        filtered = filtered[filtered["GoldenCross_D1"] == 1]
    elif cross_filter == "Daily Death Cross (D1)":
        filtered = filtered[filtered["DeathCross_D1"] == 1]
    elif cross_filter == "Weekly Golden Cross (W1)":
        filtered = filtered[filtered["GoldenCross_W1"] == 1]
    elif cross_filter == "Weekly Death Cross (W1)":
        filtered = filtered[filtered["DeathCross_W1"] == 1]

    if bb_filter == "Overbought (>= +5)":
        filtered = filtered[filtered["BB_Position"] >= 5]
    elif bb_filter == "Oversold (<= -5)":
        filtered = filtered[filtered["BB_Position"] <= -5]

    # Top performance filters last
    if perf_top5_label != "Off":
        metric = PERF_TOP5_MAP[perf_top5_label]
        filtered = filtered.sort_values(by=metric, ascending=False, na_position="last").head(10)

    flow_unavailable = False
    if flow_top5_enabled:
        flow_df = filtered.dropna(subset=["FundFlows_3M_%"])
        if flow_df.empty:
            filtered = filtered.iloc[0:0].copy()
            flow_unavailable = True
        else:
            filtered = flow_df.sort_values(by="FundFlows_3M_%", ascending=False).head(5)

    if alpha_sort != "Off" or alpha_top != "All":
        filtered = sort_by_alpha(filtered, sort_by=alpha_sort)
    if alpha_top != "All":
        top_n = int(alpha_top.split()[-1])
        filtered = filtered.dropna(subset=["Alpha_Score"]).head(top_n)

    return filtered, flow_unavailable


def render_inputs_tab(universe_map: dict):
    st.caption("Edit ticker inputs for Full list or Short List. Use + to add rows. Mark X and save to delete rows.")
    list_names = list(universe_map.keys())
    if not list_names:
        st.error("No universe lists are available.")
        return

    selected_list = st.selectbox("List to edit", options=list_names, index=0, key="inputs_selected_list")
    current_df = flatten_universe_for_editor(universe_map.get(selected_list, {}))
    current_df.insert(0, "X", False)

    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"inputs_editor_{selected_list}",
        column_config={
            "X": st.column_config.CheckboxColumn("X", help="Mark row for deletion", default=False, width="small"),
            "Group": st.column_config.TextColumn("Group", required=True, width="medium"),
            "Subgroup": st.column_config.TextColumn("Subgroup", required=True, width="medium"),
            "Ticker": st.column_config.TextColumn("Ticker", required=True, width="medium"),
        },
    )

    if st.button("Save Inputs", type="primary", key=f"inputs_save_{selected_list}"):
        work_df = edited_df.copy()
        work_df["X"] = work_df["X"].fillna(False).astype(bool)
        work_df = work_df[~work_df["X"]].drop(columns=["X"], errors="ignore")

        new_universe_block, errors = build_universe_from_editor_df(work_df)
        if errors:
            st.error("\n".join(errors[:8]))
            return
        if not new_universe_block:
            st.error("At least one valid ticker row is required.")
            return

        updated_map = clone_universe_map(universe_map)
        updated_map[selected_list] = new_universe_block
        save_universe_map(updated_map)
        st.session_state["universe_map"] = updated_map
        st.cache_data.clear()
        st.success(f"Saved {selected_list}. Screener will refresh now.")
        st.rerun()


def render_tester_tab() -> None:
    tester_url = os.environ.get("TESTER_APP_URL", "").strip()
    if not tester_url:
        try:
            tester_url = str(st.secrets["TESTER_APP_URL"]).strip() if "TESTER_APP_URL" in st.secrets else ""
        except StreamlitSecretNotFoundError:
            tester_url = ""
    if not tester_url:
        st.info(
            "Tester app URL is not configured. "
            "Set `TESTER_APP_URL` in Streamlit app Secrets to embed Tester here."
        )
        st.code("TESTER_APP_URL = \"https://your-tester-app.streamlit.app\"")
        return
    st.caption(f"Tester mounted from {tester_url}")
    components.iframe(tester_url, height=2100, scrolling=True)


def render_crypto_derivatives_tab() -> None:
    st.subheader("Crypto Derivatives - Bybit")
    st.caption("Public market data only. No API key, no API secret, no account access.")

    col_asset, col_refresh, col_path = st.columns([1.4, 1.0, 4.5])
    with col_asset:
        selected_asset = st.selectbox("Asset", options=list(BYBIT_ASSET_MAP.keys()), index=0, key="bybit_asset_selector")
    with col_refresh:
        force_refresh = st.button("Refresh Bybit Data", use_container_width=True, key="bybit_force_refresh")
    with col_path:
        st.markdown(
            f"<div style='padding-top:1.5rem; color:#94a3b8; font-size:0.78rem;'>Storage: {html.escape(str(BYBIT_STORAGE_PATH))}</div>",
            unsafe_allow_html=True,
        )

    update_started = start_background_update_if_stale(path=BYBIT_STORAGE_PATH, force=force_refresh)
    if bybit_update_in_progress():
        st.info("Bybit public market data update is running in the background.")
    elif update_started:
        st.success("Bybit public market data update completed.")
    derivatives_df = read_bybit_storage(BYBIT_STORAGE_PATH)

    states = bybit_latest_states(derivatives_df)
    state = states.get(selected_asset)
    if state is None:
        st.warning("No Bybit state is available for this asset.")
        return

    st.markdown("#### Current Summary")
    render_key_value_table(
        [
            ("Asset", state.asset),
            ("Bybit Symbol", state.exchange_symbol),
            ("Price", fmt_plain_number(state.price, 2)),
            ("Open Interest USD", fmt_money_compact(state.open_interest_usd)),
            ("OI Change 1W", fmt_plain_percent(state.oi_change_1w_pct)),
            ("OI Change 4W", fmt_plain_percent(state.oi_change_4w_pct)),
            ("OI Change 13W", fmt_plain_percent(state.oi_change_13w_pct)),
            ("OI 4W Percentile", fmt_plain_number(state.oi_change_4w_percentile, 0)),
            ("Funding Current", fmt_plain_percent(state.funding_current)),
            ("Funding 7D", fmt_plain_percent(state.funding_7d)),
            ("Funding 28D", fmt_plain_percent(state.funding_28d)),
            ("Funding Percentile", fmt_plain_number(state.funding_percentile, 0)),
            ("Perpetual Premium", fmt_plain_percent_from_pct(state.perp_premium_pct)),
            ("Premium 28D", fmt_plain_percent_from_pct(state.premium_28d_avg)),
            ("Premium Percentile", fmt_plain_number(state.premium_percentile, 0)),
            ("OI / Price Regime", state.oi_price_regime),
            ("History Start", state.history_start_date),
            ("Last Updated", state.last_updated),
            ("Data Status", state.data_status),
        ]
    )

    asset_history = derivatives_df[derivatives_df["asset"].eq(selected_asset)].copy() if not derivatives_df.empty else pd.DataFrame()
    if asset_history.empty:
        st.info("No stored history for the selected asset yet.")
        return

    st.markdown("#### Weekly History")
    history_cols = [
        "date",
        "price",
        "open_interest_usd",
        "oi_change_1w_pct",
        "oi_change_4w_pct",
        "oi_change_13w_pct",
        "oi_change_4w_percentile",
        "funding_rate",
        "funding_7d",
        "funding_28d",
        "funding_28d_percentile",
        "perp_premium_pct",
        "perp_premium_28d_avg",
        "perp_premium_percentile",
        "oi_price_regime",
        "data_status",
        "outlier_flags",
    ]
    display_history = asset_history.sort_values("date", ascending=False)
    display_history = display_history[[col for col in history_cols if col in display_history.columns]].head(104)
    st.dataframe(display_history, use_container_width=True, hide_index=True)


def render_key_value_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)


def _liquidity_latest_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def _liquidity_ordinary_roc(frame: pd.DataFrame, column: str, periods: int) -> float:
    if frame.empty or column not in frame.columns or periods < 1:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) <= periods:
        return np.nan
    current = float(values.iloc[-1])
    previous = float(values.iloc[-(periods + 1)])
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return np.nan
    return (current / previous - 1.0) * 100.0


def _liquidity_roc_momentum(frame: pd.DataFrame, column: str, momentum_months: int) -> float:
    """Return the change in the 52-week ROC over the requested month lag."""
    if frame.empty or column not in frame.columns or momentum_months < 1:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    roc_52w = values.pct_change(12, fill_method=None) * 100.0
    momentum = roc_52w.diff(momentum_months).dropna()
    return float(momentum.iloc[-1]) if not momentum.empty else np.nan


def _liquidity_cycle_maturity_pct(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return _liquidity_cycle_months_since_anchor(pd.Timestamp(value)) / 65.0 * 100.0


def _liquidity_fmt_roc(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.1f}%"


def _liquidity_display_state(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "n/a"
    return text.replace("_", " ").title()


def _liquidity_percentile_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    percentile = float(value)
    if not np.isfinite(percentile):
        return "n/a"
    if percentile >= 90.0:
        return "EXTREME ACCELERATION"
    if percentile >= 70.0:
        return "ACCELERATION"
    if percentile >= 40.0:
        return "NEUTRAL"
    if percentile >= 10.0:
        return "DECELERATION"
    return "EXTREME DECELERATION"


def _render_liquidity_summary_block(title: str, value: str, rows: list[tuple[str, str]]) -> None:
    safe_title = html.escape(title)
    safe_value = html.escape(value)
    body = "".join(
        "<div style='display:flex;align-items:baseline;gap:0.7rem;padding:0.12rem 0;background:transparent !important;'>"
        f"<span style='white-space:nowrap;color:#cbd5e1;font-size:1rem;'>{html.escape(label)}</span>"
        f"<span style='white-space:nowrap;color:#f8fafc;font-size:1.15rem;font-weight:800;text-align:left;'>{html.escape(row_value)}</span>"
        "</div>"
        for label, row_value in rows
    )
    st.markdown(
        f"""
<div style="padding:0.2rem 0 0.65rem 0;line-height:1.25;">
  <div style="color:#94a3b8;font-size:0.68rem;font-weight:700;margin-bottom:0.12rem;white-space:nowrap;">{safe_title}</div>
  <div style="color:#f8fafc;font-size:1.4rem;font-weight:800;line-height:1.15;margin-bottom:0.3rem;white-space:nowrap;">{safe_value}</div>
  <div style="width:100%;background:transparent !important;">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_liquidity_regime_summary(monthly: pd.DataFrame, regime: pd.DataFrame, latest: dict[str, Any]) -> None:
    forecast_frame, _ = read_forecast_snapshot()
    valid_forecast = (
        forecast_frame.loc[forecast_frame.get("LiquidityForwardSignal", pd.Series(dtype="object")).notna()]
        if not forecast_frame.empty and "LiquidityForwardSignal" in forecast_frame.columns
        else pd.DataFrame()
    )
    forecast_signal = valid_forecast.iloc[-1].get("LiquidityForwardSignal", "n/a") if not valid_forecast.empty else "n/a"

    treasury = read_treasury_funding_policy_snapshot()
    refinancing = _liquidity_latest_numeric(treasury.monthly, "near_term_refinancing_pressure")
    latest_date = pd.to_datetime(latest.get("date"), errors="coerce")
    current_date = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if pd.notna(latest_date):
        latest_date = min(pd.Timestamp(latest_date).normalize(), current_date)

    m2_rows = [
        (f"ROC {months}m", _liquidity_fmt_roc(_liquidity_ordinary_roc(monthly, "global_m2_usd_bn", months)))
        for months in (1, 3, 6, 12)
    ]
    m2_rows.extend(
        [
            ("Growth Percentile Status (52w)", _liquidity_percentile_status(latest.get("m2_growth_pctl"))),
            ("Fast Impulse Percentile Status (13w)", _liquidity_percentile_status(latest.get("m2_fast_impulse_pctl"))),
            ("Medium Impulse Percentile Status (26w)", _liquidity_percentile_status(latest.get("m2_medium_impulse_pctl"))),
            ("Slow Impulse Percentile Status (39w)", _liquidity_percentile_status(latest.get("m2_slow_impulse_pctl"))),
        ]
    )
    cb_rows = [
        (f"ROC {months}m", _liquidity_fmt_roc(_liquidity_ordinary_roc(monthly, "global_cb_assets_usd_bn", months)))
        for months in (1, 3, 6, 12)
    ]
    cb_rows.extend(
        [
            ("Growth Percentile Status (52w)", _liquidity_percentile_status(latest.get("cb_growth_pctl"))),
            ("Fast Impulse Percentile Status (13w)", _liquidity_percentile_status(latest.get("cb_fast_impulse_pctl"))),
            ("Medium Impulse Percentile Status (26w)", _liquidity_percentile_status(latest.get("cb_medium_impulse_pctl"))),
            ("Slow Impulse Percentile Status (39w)", _liquidity_percentile_status(latest.get("cb_slow_impulse_pctl"))),
        ]
    )
    usnl_rows = [
        (f"ROC {months}m", _liquidity_fmt_roc(_liquidity_ordinary_roc(regime, "us_net_liquidity_usd_bn", weeks)))
        for months, weeks in ((1, 4), (3, 13), (6, 26), (12, 52))
    ]
    usnl_rows.extend(
        [
            ("Growth Percentile Status (52w)", _liquidity_percentile_status(latest.get("usnl_growth_pctl"))),
            ("Fast Impulse Percentile Status (13w)", _liquidity_percentile_status(latest.get("usnl_fast_impulse_pctl"))),
            ("Medium Impulse Percentile Status (26w)", _liquidity_percentile_status(latest.get("usnl_medium_impulse_pctl"))),
            ("Slow Impulse Percentile Status (39w)", _liquidity_percentile_status(latest.get("usnl_slow_impulse_pctl"))),
        ]
    )
    maturity = _liquidity_cycle_maturity_pct(latest_date)
    score_rows = [
        (f"ROC {months}m", _liquidity_fmt_roc(_liquidity_ordinary_roc(regime, "global_liquidity_score", weeks)))
        for months, weeks in ((1, 4), (3, 13), (6, 26), (12, 52))
    ]
    score_rows.extend(
        [
            ("Status", _liquidity_direction_state(latest.get("direction_13w"))),
            ("Final Regime", str(latest.get("final_regime_label", "n/a"))),
            ("65M cycle Maturity", "n/a" if pd.isna(maturity) else f"{maturity:.0f}%"),
            ("Liquidity Forecast Signal", _liquidity_display_state(forecast_signal)),
            ("Near-Term Treasury Refinancing", "n/a" if pd.isna(refinancing) else f"{refinancing:.1f}"),
            ("Data Status", _liquidity_display_state(latest.get("data_status", "n/a"))),
        ]
    )

    columns = st.columns(4, gap="medium")
    with columns[0]:
        _render_liquidity_summary_block(
            "Global Liquidity Score",
            _liquidity_fmt_number(latest.get("global_liquidity_score"), 1),
            score_rows,
        )
    with columns[1]:
        _render_liquidity_summary_block(
            "Global M2",
            _liquidity_fmt_trillions(_liquidity_latest_numeric(monthly, "global_m2_usd_bn")),
            m2_rows,
        )
    with columns[2]:
        _render_liquidity_summary_block(
            "Global CB Assets",
            _liquidity_fmt_trillions(_liquidity_latest_numeric(monthly, "global_cb_assets_usd_bn")),
            cb_rows,
        )
    with columns[3]:
        _render_liquidity_summary_block(
            "US Net Liquidity",
            _liquidity_fmt_trillions(latest.get("us_net_liquidity_usd_bn")),
            usnl_rows,
        )
def render_global_liquidity_dashboard_tab() -> None:
    st.subheader("Global Liquidity Regime")

    control_cols = st.columns([1.2, 4.8])
    with control_cols[0]:
        force_refresh = st.button("Refresh Liquidity Data", use_container_width=True, key="global_liquidity_force_refresh")
    with control_cols[1]:
        st.markdown(
            f"<div style='padding-top:1.6rem; color:#94a3b8; font-size:0.78rem;'>Storage: {html.escape(str(GLOBAL_LIQUIDITY_STORAGE_DIR))}</div>",
            unsafe_allow_html=True,
        )

    if force_refresh:
        with st.spinner("Refreshing official liquidity sources..."):
            raw, monthly, weekly = update_global_liquidity(api_key=get_fred_api_key_for_app(), force=True)
    else:
        update_started = start_global_liquidity_update_if_stale(api_key=get_fred_api_key_for_app())
        raw, monthly, weekly = read_global_liquidity()
        if raw.empty and monthly.empty and weekly.empty and not global_liquidity_update_in_progress():
            with st.spinner("Initial liquidity backfill is running..."):
                raw, monthly, weekly = update_global_liquidity(api_key=get_fred_api_key_for_app(), force=True)
        elif update_started or global_liquidity_update_in_progress():
            st.info("Liquidity data refresh is running in the background. Reload the tab in a moment to see new observations.")

    if raw.empty and monthly.empty and weekly.empty:
        st.warning("No Global Liquidity data is stored yet.")
        return

    monthly = _liquidity_prepare_dates(monthly)
    weekly = _liquidity_prepare_dates(weekly)
    regime = _build_global_liquidity_regime_frame(monthly, weekly)
    if regime.empty:
        st.warning("Global Liquidity Regime cannot be calculated from the current storage.")
        return
    latest = _liquidity_latest_row_with_value(regime, "global_liquidity_score")
    if not latest:
        latest = _liquidity_latest_row(regime)

    range_choice = st.radio(
        "Time range",
        ["1Y", "3Y", "5Y", "10Y", "MAX"],
        index=2,
        horizontal=True,
        key="global_liquidity_regime_range",
    )
    chart_frame = _liquidity_filter_range(regime, range_choice)

    st.markdown("### Regime Summary")
    _render_liquidity_regime_summary(monthly, regime, latest)

    st.markdown("### Global M2")
    _render_global_m2_level_growth(chart_frame, regime)
    with st.expander("Global M2 Momentum vs 65M Liquidity Cycle", expanded=True):
        _render_long_cycle_chart(chart_frame, regime)
    _render_liquidity_impulse_percentile_chart(chart_frame, "Global M2", "m2")
    _render_liquidity_impulse_percentile_chart(chart_frame, "Global CB Assets", "cb")
    _render_liquidity_impulse_percentile_chart(chart_frame, "US Net Liquidity", "usnl")

    _render_global_liquidity_components_chart(chart_frame)
    _render_liquidity_contribution_chart(chart_frame, "m2")
    _render_liquidity_contribution_chart(chart_frame, "cb")
    _render_us_net_liquidity_chart(chart_frame)

    st.markdown("### What Global Liquidity Regime Means for Markets")
    st.dataframe(_build_liquidity_asset_impact_table(latest), use_container_width=True, hide_index=True)

    range_start = pd.to_datetime(chart_frame["date"], errors="coerce").min() if not chart_frame.empty else None
    render_liquidity_forecast(range_start=range_start)

    st.markdown("### Data Quality")
    st.dataframe(_build_liquidity_regime_quality_summary(regime), use_container_width=True, hide_index=True)
    freshness_rows = [
        {
            "Block": item.block,
            "Last Observation": item.last_observation_date,
            "Last Release": item.last_release_date,
            "Last Updated": item.last_updated,
            "Status": item.data_status,
        }
        for item in global_liquidity_freshness(raw, monthly, weekly)
    ]
    st.dataframe(pd.DataFrame(freshness_rows), use_container_width=True, hide_index=True)
    with st.expander("Source Details", expanded=False):
        st.markdown("#### Regional M2")
        st.dataframe(_build_liquidity_regional_table(raw, monthly), use_container_width=True, hide_index=True)
        st.markdown("#### Central Bank Assets")
        st.dataframe(_build_liquidity_cb_table(raw, monthly), use_container_width=True, hide_index=True)
        st.markdown("#### Raw Sources")
        st.dataframe(_build_liquidity_quality_table(raw), use_container_width=True, hide_index=True)

    st.markdown("### Methodology")
    render_market_formula(
        "Global Liquidity Regime formulas",
        "Global_M2 = US_M2 + EA_M2 + China_M2 + Japan_M2\n"
        "TrendScore = 0.50 * I(M2_13W > 0) + 0.30 * I(M2_26W > 0) + 0.20 * I(M2_52W > 0)\n"
        "M2Impulse = 0.50 * Pctl(M2_13W) + 0.30 * Pctl(M2_26W) + 0.20 * Pctl(M2_52W)\n"
        "Global_CB_Assets = Fed_Assets + ECB_Assets + BoJ_Assets + PBoC_Assets\n"
        "CBImpulse = 0.50 * Pctl(CB_13W) + 0.30 * Pctl(CB_26W) + 0.20 * Pctl(CB_52W)\n"
        "US_Net_Liquidity = WALCL - TGA - RRP\n"
        "USNLImpulse = 0.50 * Pctl(USNL_4W) + 0.30 * Pctl(USNL_13W) + 0.20 * Pctl(USNL_26W)\n"
        "GlobalLiquidityScore = 0.50 * M2Impulse + 0.25 * CBImpulse + 0.25 * USNLImpulse\n"
        "Direction13W = GlobalLiquidityScore_t - GlobalLiquidityScore_t-13W\n\n"
        "GlobalM2YoYGrowth = GlobalM2_t / GlobalM2_t-52W - 1\n"
        "FastImpulse = GlobalM2YoYGrowth_t - GlobalM2YoYGrowth_t-13W\n"
        "MediumImpulse = GlobalM2YoYGrowth_t - GlobalM2YoYGrowth_t-26W\n"
        "SlowImpulse = GlobalM2YoYGrowth_t - GlobalM2YoYGrowth_t-39W\n"
        "GrowthPercentile and Fast/Medium/SlowImpulsePercentile = trailing 3Y percentile (156W)\n\n"
        "The same Growth and Impulse Percentile calculations are applied to Global CB Assets and US Net Liquidity.\n\n"
        "Percentile status: >=90 extreme acceleration; >=70 acceleration; >=40 neutral; >=10 deceleration; <10 extreme deceleration.\n\n"
        "All M2 and central-bank balance-sheet series are already normalized to bn USD. Missing major data is not treated as zero. "
        "Percentiles are trailing 3Y point-in-time windows. The 65M long cycle is structural context and is not included in the score.",
    )


def _build_global_liquidity_regime_frame(monthly: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty and weekly.empty:
        return pd.DataFrame()
    monthly_source = _liquidity_weekly_from_monthly(
        monthly,
        [
            "us_m2_usd_bn",
            "ea_m2_usd_bn",
            "china_m2_usd_bn",
            "japan_m2_usd_bn",
            "fed_assets_usd_bn",
            "ecb_assets_usd_bn",
            "boj_assets_usd_bn",
            "pboc_assets_usd_bn",
            "last_updated",
        ],
    )
    weekly_source = weekly.set_index("date").sort_index() if not weekly.empty and "date" in weekly.columns else pd.DataFrame()
    index = monthly_source.index.union(weekly_source.index).sort_values()
    if index.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(index=index)
    for column in monthly_source.columns:
        frame[column] = monthly_source[column].reindex(index).ffill()
    weekly_cols = [
        "us_net_liquidity_usd_bn",
        "fed_assets_usd_bn",
        "tga_usd_bn",
        "rrp_usd_bn",
        "last_updated",
    ]
    for column in weekly_cols:
        if column in weekly_source.columns:
            frame[f"weekly_{column}" if column in frame.columns else column] = weekly_source[column].reindex(index).ffill()

    m2_components = ["us_m2_usd_bn", "ea_m2_usd_bn", "china_m2_usd_bn", "japan_m2_usd_bn"]
    cb_components = ["fed_assets_usd_bn", "ecb_assets_usd_bn", "boj_assets_usd_bn", "pboc_assets_usd_bn"]
    frame["global_m2_usd_bn"] = frame[m2_components].sum(axis=1, min_count=4)
    frame["global_cb_assets_usd_bn"] = frame[cb_components].sum(axis=1, min_count=4)
    frame["global_m2_partial_usd_bn"] = frame[m2_components].sum(axis=1, min_count=1)
    frame["global_cb_assets_partial_usd_bn"] = frame[cb_components].sum(axis=1, min_count=1)
    if "weekly_us_net_liquidity_usd_bn" in frame.columns:
        frame["us_net_liquidity_usd_bn"] = frame["weekly_us_net_liquidity_usd_bn"]

    for weeks in [4, 13, 26, 52]:
        frame[f"m2_{weeks}w"] = frame["global_m2_usd_bn"].pct_change(weeks, fill_method=None)
        frame[f"cb_{weeks}w"] = frame["global_cb_assets_usd_bn"].pct_change(weeks, fill_method=None)
    for weeks in [4, 13, 26]:
        frame[f"usnl_{weeks}w"] = frame["us_net_liquidity_usd_bn"].pct_change(weeks, fill_method=None)

    frame["trend_score"] = (
        0.50 * (frame["m2_13w"] > 0).astype(float)
        + 0.30 * (frame["m2_26w"] > 0).astype(float)
        + 0.20 * (frame["m2_52w"] > 0).astype(float)
    )
    frame.loc[frame[["m2_13w", "m2_26w", "m2_52w"]].isna().any(axis=1), "trend_score"] = np.nan
    frame["trend_state"] = frame["trend_score"].map(_liquidity_trend_state)

    frame["m2_13w_pctl"] = _liquidity_trailing_percentile(frame["m2_13w"])
    frame["m2_26w_pctl"] = _liquidity_trailing_percentile(frame["m2_26w"])
    frame["m2_52w_pctl"] = _liquidity_trailing_percentile(frame["m2_52w"])
    for prefix, source_column in (
        ("m2", "global_m2_usd_bn"),
        ("cb", "global_cb_assets_usd_bn"),
        ("usnl", "us_net_liquidity_usd_bn"),
    ):
        frame[f"{prefix}_growth"] = frame[source_column].div(frame[source_column].shift(52)).sub(1.0)
        frame[f"{prefix}_fast_impulse"] = frame[f"{prefix}_growth"].sub(frame[f"{prefix}_growth"].shift(13))
        frame[f"{prefix}_medium_impulse"] = frame[f"{prefix}_growth"].sub(frame[f"{prefix}_growth"].shift(26))
        frame[f"{prefix}_slow_impulse"] = frame[f"{prefix}_growth"].sub(frame[f"{prefix}_growth"].shift(39))
        frame[f"{prefix}_growth_pctl"] = _liquidity_trailing_percentile(frame[f"{prefix}_growth"])
        frame[f"{prefix}_fast_impulse_pctl"] = _liquidity_trailing_percentile(frame[f"{prefix}_fast_impulse"])
        frame[f"{prefix}_medium_impulse_pctl"] = _liquidity_trailing_percentile(frame[f"{prefix}_medium_impulse"])
        frame[f"{prefix}_slow_impulse_pctl"] = _liquidity_trailing_percentile(frame[f"{prefix}_slow_impulse"])
    frame["cb_13w_pctl"] = _liquidity_trailing_percentile(frame["cb_13w"])
    frame["cb_26w_pctl"] = _liquidity_trailing_percentile(frame["cb_26w"])
    frame["cb_52w_pctl"] = _liquidity_trailing_percentile(frame["cb_52w"])
    frame["usnl_4w_pctl"] = _liquidity_trailing_percentile(frame["usnl_4w"])
    frame["usnl_13w_pctl"] = _liquidity_trailing_percentile(frame["usnl_13w"])
    frame["usnl_26w_pctl"] = _liquidity_trailing_percentile(frame["usnl_26w"])

    frame["m2_impulse"] = 0.50 * frame["m2_13w_pctl"] + 0.30 * frame["m2_26w_pctl"] + 0.20 * frame["m2_52w_pctl"]
    frame["cb_impulse"] = 0.50 * frame["cb_13w_pctl"] + 0.30 * frame["cb_26w_pctl"] + 0.20 * frame["cb_52w_pctl"]
    frame["usnl_impulse"] = 0.50 * frame["usnl_4w_pctl"] + 0.30 * frame["usnl_13w_pctl"] + 0.20 * frame["usnl_26w_pctl"]
    frame["global_liquidity_score"] = 0.50 * frame["m2_impulse"] + 0.25 * frame["cb_impulse"] + 0.25 * frame["usnl_impulse"]
    frame["impulse_state"] = frame["global_liquidity_score"].map(_liquidity_score_state)
    frame["direction_13w"] = frame["global_liquidity_score"] - frame["global_liquidity_score"].shift(13)
    frame["direction_13w_state"] = frame["direction_13w"].map(_liquidity_direction_state)
    frame["long_cycle_phase"] = [ _liquidity_long_cycle_phase(date) for date in frame.index ]
    frame["long_cycle_value"] = [ _liquidity_long_cycle_value(date) for date in frame.index ]
    frame["cycle_confirmation"] = _liquidity_cycle_confirmation(frame)
    frame["final_regime_label"] = frame.apply(_liquidity_final_label, axis=1)
    frame["data_status"] = np.where(
        frame[["global_m2_usd_bn", "global_cb_assets_usd_bn", "us_net_liquidity_usd_bn", "global_liquidity_score"]].notna().all(axis=1),
        "CURRENT",
        "PARTIAL_DATA",
    )
    if "last_updated" not in frame.columns or frame["last_updated"].isna().all():
        frame["last_updated"] = frame.get("weekly_last_updated", "n/a")
    frame = frame.reset_index().rename(columns={"index": "date"})
    return frame


def _liquidity_weekly_from_monthly(monthly: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if monthly.empty or "date" not in monthly.columns:
        return pd.DataFrame()
    use_cols = [column for column in columns if column in monthly.columns]
    if not use_cols:
        return pd.DataFrame()
    source = monthly[["date"] + use_cols].dropna(subset=["date"]).copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.dropna(subset=["date"]).sort_values("date")
    source = source.set_index("date")
    weekly_index = pd.date_range(source.index.min(), source.index.max() + pd.offsets.MonthEnd(1), freq="W-FRI")
    combined_index = source.index.union(weekly_index).sort_values()
    return source.reindex(combined_index).ffill().reindex(weekly_index).ffill()


def _liquidity_trailing_percentile(series: pd.Series, window: int = 156, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def rank_last(window_values: np.ndarray) -> float:
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods or not np.isfinite(window_values[-1]):
            return np.nan
        return float((clean <= window_values[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _liquidity_trend_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number >= 0.80:
        return "STRONGLY_EXPANDING"
    if number >= 0.60:
        return "EXPANDING"
    if number >= 0.40:
        return "FLAT_MIXED"
    if number >= 0.20:
        return "CONTRACTING"
    return "STRONGLY_CONTRACTING"


def _liquidity_score_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number >= 80:
        return "VERY_STRONG"
    if number >= 60:
        return "STRONG"
    if number >= 40:
        return "NEUTRAL"
    if number >= 20:
        return "WEAK"
    return "VERY_WEAK"


def _liquidity_direction_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    number = float(value)
    if number > 10:
        return "ACCELERATING"
    if number >= 5:
        return "IMPROVING"
    if number > -5:
        return "STABLE"
    if number >= -10:
        return "DETERIORATING"
    return "DETERIORATING_FAST"


def _liquidity_long_cycle_phase(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    date = pd.Timestamp(value)
    months = _liquidity_cycle_months_since_anchor(date)
    phase_pos = months % 65.0
    if phase_pos < 16.25:
        return "RECOVERY_REACCELERATION"
    if phase_pos < 32.50:
        return "ACCELERATING_EXPANSION"
    if phase_pos < 48.75:
        return "DECELERATING_EXPANSION"
    return "CONTRACTION"


def _liquidity_long_cycle_value(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    months = _liquidity_cycle_months_since_anchor(pd.Timestamp(value))
    return float(-np.cos((months / 65.0) * 2.0 * np.pi) * 100.0)


def _liquidity_cycle_months_since_anchor(value: pd.Timestamp) -> float:
    anchor = pd.Timestamp("2022-10-01")
    date = pd.Timestamp(value)
    return (date.year - anchor.year) * 12 + (date.month - anchor.month) + (date.day - 1) / 30.4375


def _liquidity_cycle_peak_date(cycle_number: int) -> pd.Timestamp:
    return pd.Timestamp("2022-10-01") + pd.DateOffset(months=32 + (65 * cycle_number), days=15)


def _liquidity_cycle_confirmation(frame: pd.DataFrame) -> pd.Series:
    roc = pd.to_numeric(frame.get("m2_52w", np.nan), errors="coerce")
    roc_direction = roc.diff(13)
    cycle_direction = pd.Series(frame.get("long_cycle_value", np.nan), index=frame.index).diff(13)
    confirmed = np.sign(roc_direction) == np.sign(cycle_direction)
    return pd.Series(np.where(confirmed, "CYCLE_CONFIRMED", "LIQUIDITY_CYCLE_DIVERGENCE"), index=frame.index).where(
        roc_direction.notna() & cycle_direction.notna(),
        "DATA_INCOMPLETE",
    )


def _liquidity_final_label(row: pd.Series) -> str:
    score = row.get("global_liquidity_score")
    direction = row.get("direction_13w")
    trend = str(row.get("trend_state", ""))
    if score is None or direction is None or pd.isna(score) or pd.isna(direction):
        return "DATA_INCOMPLETE"
    score = float(score)
    direction = float(direction)
    if score >= 80 and direction >= 5:
        return "STRONG_EXPANSION"
    if score >= 60 and direction >= -5:
        return "EXPANSION"
    if score >= 40 and direction > 5:
        return "REACCELERATION"
    if score >= 40 and direction < -10:
        return "LIQUIDITY_WARNING"
    if score >= 40:
        return "DECELERATING_EXPANSION" if "EXPANDING" in trend else "NEUTRAL"
    if score < 20:
        return "STRONG_CONTRACTION"
    return "CONTRACTION"


def _liquidity_filter_range(frame: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if frame.empty or range_key == "MAX":
        return frame
    years = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.get(range_key)
    if years is None or "date" not in frame.columns:
        return frame
    max_date = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(max_date):
        return frame
    return frame[pd.to_datetime(frame["date"], errors="coerce") >= max_date - pd.DateOffset(years=years)].copy()


def _liquidity_fmt_score_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1f} / {_liquidity_score_state(value)}"


def _liquidity_fmt_score_delta(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.1f} score pts"


def _liquidity_history_start(frame: pd.DataFrame) -> str:
    if frame.empty or "date" not in frame.columns:
        return "History Start: n/a"
    rows = frame.dropna(subset=["global_liquidity_score"])
    if rows.empty:
        rows = frame.dropna(subset=["global_m2_usd_bn", "global_cb_assets_usd_bn", "us_net_liquidity_usd_bn"], how="all")
    if rows.empty:
        return "History Start: n/a"
    return f"History Start: {pd.Timestamp(rows['date'].iloc[0]).strftime('%Y-%m-%d')}"


LIQUIDITY_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _style_liquidity_plotly(fig: go.Figure, height: int, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 72, "t": 62, "b": 42},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(
        tickformat="%b'%y",
        showgrid=False,
        zeroline=False,
        color="#cbd5e1",
        linecolor="#475569",
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#263241",
        zeroline=False,
        color="#cbd5e1",
        linecolor="#475569",
        ticks="outside",
    )
    return fig


def _render_global_m2_level_growth(frame: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    growth_choice = st.selectbox("Global M2 Growth", ["13W ROC", "26W ROC", "52W / YoY"], index=2, key="global_m2_growth_selector")
    growth_col = {"13W ROC": "m2_13w", "26W ROC": "m2_26w", "52W / YoY": "m2_52w"}[growth_choice]
    if frame.empty:
        st.info("No data for Global M2 - Level and Growth.")
        return
    chart_df = frame[["date", "global_m2_usd_bn", growth_col, "long_cycle_phase"]].copy()
    chart_df["level"] = pd.to_numeric(chart_df["global_m2_usd_bn"], errors="coerce") / 1000.0
    chart_df["growth"] = pd.to_numeric(chart_df[growth_col], errors="coerce") * 100.0
    chart_df = chart_df.dropna(subset=["date"])
    if chart_df.empty:
        st.info("No data for Global M2 - Level and Growth.")
        return
    bands = _liquidity_phase_bands(full_frame, frame)
    fig = go.Figure()
    phase_colors = {
        "RECOVERY_REACCELERATION": "#22c55e",
        "ACCELERATING_EXPANSION": "#84cc16",
        "DECELERATING_EXPANSION": "#facc15",
        "CONTRACTION": "#ef4444",
    }
    for _, band in bands.iterrows():
        fig.add_vrect(
            x0=band["start"],
            x1=band["end"],
            fillcolor=phase_colors.get(str(band["Phase"]), "#64748b"),
            opacity=0.16,
            line_width=0,
        )
    level_df = chart_df.dropna(subset=["level"])
    growth_df = chart_df.dropna(subset=["growth"])
    fig.add_trace(
        go.Scatter(
            x=level_df["date"],
            y=level_df["level"],
            mode="lines",
            name="Global M2",
            line={"color": "#38bdf8", "width": 2.2},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Global M2: %{y:.2f}T<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=growth_df["date"],
            y=growth_df["growth"],
            mode="lines",
            name=growth_choice,
            yaxis="y2",
            line={"color": "#f97316", "width": 1.8},
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{growth_choice}: %{{y:.2f}}%<extra></extra>",
        )
    )
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=0, y1=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(
        yaxis={"title": "Global M2, USD tn"},
        yaxis2={"title": f"{growth_choice}, %", "overlaying": "y", "side": "right", "showgrid": False},
    )
    st.plotly_chart(_style_liquidity_plotly(fig, 360, "Global M2 - Level and Growth"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _liquidity_phase_bands(full_frame: pd.DataFrame, visible_frame: pd.DataFrame) -> pd.DataFrame:
    if full_frame.empty or visible_frame.empty:
        return pd.DataFrame(columns=["start", "end", "Phase"])
    dates = pd.to_datetime(full_frame["date"], errors="coerce")
    phases = full_frame["long_cycle_phase"].astype(str)
    rows = []
    start = None
    current = None
    last_date = None
    min_visible = pd.to_datetime(visible_frame["date"], errors="coerce").min()
    max_visible = pd.to_datetime(visible_frame["date"], errors="coerce").max()
    for date, phase in zip(dates, phases):
        if pd.isna(date):
            continue
        if current is None:
            start = date
            current = phase
        elif phase != current:
            rows.append({"start": max(start, min_visible), "end": min(last_date, max_visible), "Phase": current})
            start = date
            current = phase
        last_date = date
    if current is not None and start is not None and last_date is not None:
        rows.append({"start": max(start, min_visible), "end": min(last_date, max_visible), "Phase": current})
    bands = pd.DataFrame(rows)
    if bands.empty:
        return bands
    return bands[bands["end"] >= bands["start"]]


def _render_global_liquidity_components_chart(frame: pd.DataFrame) -> None:
    mapping = {
        "GlobalLiquidityScore": "global_liquidity_score",
        "M2Impulse": "m2_impulse",
        "CBImpulse": "cb_impulse",
        "USNLImpulse": "usnl_impulse",
    }
    fig = go.Figure()
    colors = {
        "GlobalLiquidityScore": "#f8fafc",
        "M2Impulse": "#38bdf8",
        "CBImpulse": "#a78bfa",
        "USNLImpulse": "#22c55e",
    }
    added = False
    for label, column in mapping.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        d = pd.DataFrame({"date": frame["date"], "value": values}).dropna()
        if d.empty:
            continue
        added = True
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["value"],
                mode="lines",
                name=label,
                line={"color": colors.get(label, "#cbd5e1"), "width": 1.9},
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{label}: %{{y:.1f}}<extra></extra>",
            )
        )
    if not added:
        st.info("No data for Global Liquidity Score - Components.")
        return
    for level in [20, 40, 60, 80]:
        fig.add_hline(y=level, line={"color": "#94a3b8", "dash": "dot", "width": 1}, opacity=0.55)
    fig.update_layout(yaxis={"title": "Score", "range": [0, 100]})
    st.plotly_chart(_style_liquidity_plotly(fig, 320, "Global Liquidity Score - Components"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_liquidity_contribution_chart(frame: pd.DataFrame, block: str) -> None:
    horizon = st.selectbox(
        "Regional M2 horizon" if block == "m2" else "Central Bank horizon",
        ["13W", "26W", "52W"],
        index=0,
        key=f"global_liquidity_{block}_contribution_horizon",
    )
    weeks = int(horizon.replace("W", ""))
    if block == "m2":
        title = "Global M2 - Regional Contribution"
        components = {
            "US": "us_m2_usd_bn",
            "Euro Area": "ea_m2_usd_bn",
            "China": "china_m2_usd_bn",
            "Japan": "japan_m2_usd_bn",
        }
        total_col = "global_m2_usd_bn"
    else:
        title = "Global Central Bank Assets - Regional Impulse"
        components = {
            "Fed": "fed_assets_usd_bn",
            "ECB": "ecb_assets_usd_bn",
            "BoJ": "boj_assets_usd_bn",
            "PBoC": "pboc_assets_usd_bn",
        }
        total_col = "global_cb_assets_usd_bn"
    rows = []
    for label, column in components.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").diff(weeks)
        for date, value in zip(frame["date"], values):
            if pd.notna(date) and np.isfinite(value):
                rows.append({"Date": date, "Component": label, "Change": float(value)})
    chart_df = pd.DataFrame(rows)
    total = pd.DataFrame()
    if total_col in frame.columns:
        total = pd.DataFrame({"Date": frame["date"], "Total Change": pd.to_numeric(frame[total_col], errors="coerce").diff(weeks)})
        total = total.dropna(subset=["Date", "Total Change"])
    if chart_df.empty:
        st.info(f"No data for {title}.")
        return
    fig = go.Figure()
    palette = ["#38bdf8", "#a78bfa", "#22c55e", "#facc15"]
    for idx, component in enumerate(chart_df["Component"].drop_duplicates()):
        d = chart_df[chart_df["Component"].eq(component)]
        fig.add_trace(
            go.Bar(
                x=d["Date"],
                y=d["Change"],
                name=component,
                marker_color=palette[idx % len(palette)],
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{component}: %{{y:,.0f}}B<extra></extra>",
            )
        )
    if not total.empty:
        fig.add_trace(
            go.Scatter(
                x=total["Date"],
                y=total["Total Change"],
                mode="lines",
                name="Global total change",
                line={"color": "#f8fafc", "width": 1.8},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: %{y:,.0f}B<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(barmode="relative", yaxis={"title": f"{horizon} change, USD bn"})
    st.plotly_chart(_style_liquidity_plotly(fig, 320, title), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_us_net_liquidity_chart(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No data for US Net Liquidity - Funding Impulse.")
        return
    horizon = st.selectbox(
        "US Net Liquidity horizon",
        ["13W", "26W", "52W"],
        index=0,
        key="global_liquidity_usnl_contribution_horizon",
    )
    weeks = int(horizon.replace("W", ""))
    rows = []
    fed_column = "weekly_fed_assets_usd_bn" if "weekly_fed_assets_usd_bn" in frame.columns else "fed_assets_usd_bn"
    mapping = {
        "Fed Assets": (fed_column, 1.0),
        "-TGA": ("tga_usd_bn", -1.0),
        "-RRP": ("rrp_usd_bn", -1.0),
    }
    for label, (column, sign) in mapping.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce") * sign
        values = values.diff(weeks)
        for date, value in zip(frame["date"], values):
            if pd.notna(date) and np.isfinite(value):
                rows.append({"Date": date, "Component": label, "Change": float(value)})
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info("No data for US Net Liquidity - Funding Impulse.")
        return
    total = pd.DataFrame(
        {
            "Date": frame["date"],
            "Total Change": pd.to_numeric(frame.get("us_net_liquidity_usd_bn", np.nan), errors="coerce").diff(weeks),
        }
    ).dropna(subset=["Date", "Total Change"])
    fig = go.Figure()
    colors = {"Fed Assets": "#38bdf8", "-TGA": "#f97316", "-RRP": "#a78bfa"}
    for component in chart_df["Component"].drop_duplicates():
        d = chart_df[chart_df["Component"].eq(component)]
        fig.add_trace(
            go.Bar(
                x=d["Date"],
                y=d["Change"],
                name=component,
                marker_color=colors.get(component, "#cbd5e1"),
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{component}: %{{y:,.0f}}B<extra></extra>",
            )
        )
    if not total.empty:
        fig.add_trace(
            go.Scatter(
                x=total["Date"],
                y=total["Total Change"],
                mode="lines",
                name="Global total change",
                line={"color": "#f8fafc", "width": 1.8},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: %{y:,.0f}B<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(barmode="relative", yaxis={"title": f"{horizon} change, USD bn"})
    st.plotly_chart(_style_liquidity_plotly(fig, 320, "US Net Liquidity - Funding Impulse"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_long_cycle_chart(frame: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No data for Global M2 Momentum vs 65M Liquidity Cycle.")
        return
    m2_roc = pd.to_numeric(full_frame.get("m2_52w", np.nan), errors="coerce")
    normalized = (_liquidity_rolling_zscore(m2_roc, 156, 104).clip(-2, 2) * 50.0).reindex(frame.index)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    min_date = dates.min()
    max_date = dates.max()
    current_peak = _liquidity_cycle_peak_date(0)
    next_peak = _liquidity_cycle_peak_date(1)
    cycle_end = max(max_date, next_peak) if pd.notna(max_date) else next_peak
    cycle_dates = pd.date_range(min_date, cycle_end, freq="W-FRI") if pd.notna(min_date) else pd.DatetimeIndex([])
    cycle = pd.Series([_liquidity_long_cycle_value(date) for date in cycle_dates], index=cycle_dates)
    rows = []
    for date, value in zip(frame["date"], normalized):
        if pd.notna(date) and np.isfinite(value):
            rows.append({"Date": date, "Series": "Normalized Global M2 ROC", "Value": float(value)})
    for date, value in cycle.items():
        if pd.notna(date) and np.isfinite(value):
            rows.append({"Date": date, "Series": "65M Reference Cycle", "Value": float(value)})
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info("No data for Global M2 Momentum vs 65M Liquidity Cycle.")
        return
    latest = _liquidity_latest_row(frame)
    st.caption(f"Informational status: {latest.get('cycle_confirmation', 'n/a')}")
    fig = go.Figure()
    colors = {"Normalized Global M2 ROC": "#38bdf8", "65M Reference Cycle": "#facc15"}
    for series in chart_df["Series"].drop_duplicates():
        d = chart_df[chart_df["Series"].eq(series)]
        fig.add_trace(
            go.Scatter(
                x=d["Date"],
                y=d["Value"],
                mode="lines",
                name=series,
                line={"color": colors.get(series, "#cbd5e1"), "width": 1.8},
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{series}: %{{y:.1f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    trough = pd.Timestamp("2022-10-01")
    for marker_date, label, color in [
        (trough, "Cycle trough Oct 2022", "#ef4444"),
        (current_peak, "Current cycle peak", "#22c55e"),
        (next_peak, "Next cycle peak", "#22c55e"),
    ]:
        if pd.notna(min_date) and marker_date >= min_date and marker_date <= cycle_end:
            marker_x = marker_date.strftime("%Y-%m-%d")
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=marker_x,
                x1=marker_x,
                y0=0,
                y1=1,
                line={"color": color, "dash": "dot", "width": 1},
            )
            fig.add_annotation(
                x=marker_x,
                y=1.03,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font={"color": color, "size": 10},
                xanchor="left",
            )
    fig.update_layout(yaxis={"title": "-100 to +100", "range": [-100, 100]})
    if pd.notna(min_date) and pd.notna(max_date):
        fig.update_xaxes(range=[min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")])
    st.plotly_chart(_style_liquidity_plotly(fig, 300, "Global M2 Momentum vs 65M Liquidity Cycle"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_liquidity_impulse_percentile_chart(frame: pd.DataFrame, asset_label: str, prefix: str) -> None:
    title = f"{asset_label} Growth and Impulse Percentiles"
    if frame.empty:
        st.info(f"No data for {title}.")
        return
    mapping = {
        "Growth Percentile": (f"{prefix}_growth_pctl", "#f8fafc"),
        "Fast Impulse Percentile": (f"{prefix}_fast_impulse_pctl", "#38bdf8"),
        "Medium Impulse Percentile": (f"{prefix}_medium_impulse_pctl", "#a78bfa"),
        "Slow Impulse Percentile": (f"{prefix}_slow_impulse_pctl", "#facc15"),
    }
    fig = go.Figure()
    added = False
    for label, (column, color) in mapping.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        chart_df = pd.DataFrame({"date": frame["date"], "value": values}).dropna()
        if chart_df.empty:
            continue
        added = True
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["value"],
                mode="lines",
                name=label,
                line={"color": color, "width": 1.8},
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{label}: %{{y:.1f}}<extra></extra>",
            )
        )
    if not added:
        st.info(f"No data for {title}.")
        return
    for level in [20, 40, 60, 80]:
        fig.add_hline(y=level, line={"color": "#64748b", "dash": "dot", "width": 1}, opacity=0.55)
    fig.update_layout(
        yaxis={"title": "Trailing 3Y percentile", "range": [0, 100]},
        hovermode="x unified",
    )
    st.plotly_chart(_style_liquidity_plotly(fig, 320, title), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _liquidity_rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std(ddof=0)
    return (values - mean) / std.replace(0.0, np.nan)


def _build_liquidity_asset_impact_table(latest: dict[str, Any]) -> pd.DataFrame:
    score = latest.get("global_liquidity_score")
    direction = latest.get("direction_13w")
    long_cycle = str(latest.get("long_cycle_phase", "n/a"))
    final_label = str(latest.get("final_regime_label", "n/a"))
    score_value = float(score) if score is not None and not pd.isna(score) else np.nan
    direction_value = float(direction) if direction is not None and not pd.isna(direction) else np.nan

    spy_bias = _liquidity_asset_bias(score_value, direction_value, sensitivity=1.0)
    qqq_bias = _liquidity_asset_bias(score_value, direction_value, sensitivity=1.25)
    btc_bias = _liquidity_asset_bias(score_value, direction_value, sensitivity=1.15)
    gld_bias = "Neutral" if long_cycle != "CONTRACTION" else "Neutral / weaker structural backdrop"

    return pd.DataFrame(
        [
            {
                "Asset": "SPY",
                "Liquidity Bias": spy_bias,
                "Horizon": "8-12W",
                "Long-Cycle Context": long_cycle,
                "Historical Regime Context": "Strong/improving liquidity is historically favorable; weak/deteriorating liquidity is a correction-risk modifier.",
                "Confidence": "MEDIUM",
                "Current Implication": _liquidity_asset_text("SPY", spy_bias, final_label),
            },
            {
                "Asset": "QQQ",
                "Liquidity Bias": qqq_bias,
                "Horizon": "8-12W",
                "Long-Cycle Context": long_cycle,
                "Historical Regime Context": "More sensitive than SPY; accelerating expansion was strongest and contraction was weakest in long-cycle tests.",
                "Confidence": "MEDIUM_HIGH",
                "Current Implication": _liquidity_asset_text("QQQ", qqq_bias, final_label),
            },
            {
                "Asset": "GLD",
                "Liquidity Bias": gld_bias,
                "Horizon": "Structural",
                "Long-Cycle Context": long_cycle,
                "Historical Regime Context": "Global Liquidity Score is not a core short-term gold signal; long-cycle context is more relevant.",
                "Confidence": "MEDIUM",
                "Current Implication": "Gold remains driven primarily by DXY, real yields, US2Y, Gold Tactical Flow and Gold Alpha.",
            },
            {
                "Asset": "BTC-USD",
                "Liquidity Bias": btc_bias,
                "Horizon": "4-12W",
                "Long-Cycle Context": "Informational only",
                "Historical Regime Context": "Global Liquidity Score is a meaningful macro modifier, but its BTC relationship is unstable across subperiods.",
                "Confidence": "MEDIUM_LOW",
                "Current Implication": "Macro liquidity can be overridden by trend, ETF flows, open interest, funding, basis and crypto-specific liquidity.",
            },
        ]
    )


def _liquidity_asset_bias(score: float, direction: float, sensitivity: float = 1.0) -> str:
    if not np.isfinite(score) or not np.isfinite(direction):
        return "Data incomplete"
    adjusted = (score - 50.0) + direction * sensitivity
    if score >= 60 and direction > 5:
        return "Positive"
    if adjusted <= -20 or (score < 40 and direction < -5):
        return "Caution / negative liquidity modifier"
    if adjusted <= -8:
        return "Neutral / caution"
    if adjusted >= 12:
        return "Constructive"
    return "Neutral"


def _liquidity_asset_text(asset: str, bias: str, final_label: str) -> str:
    if "negative" in bias.lower() or "caution" in bias.lower():
        return f"{asset} has a weaker liquidity backdrop under {final_label}; this is a risk modifier, not an automatic sell signal."
    if bias in {"Positive", "Constructive"}:
        return f"{asset} has a historically more favorable liquidity backdrop under {final_label}."
    return f"{asset} has a mixed liquidity backdrop under {final_label}."


def _build_liquidity_regime_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Global M2", "global_m2_usd_bn"),
        ("Global CB Assets", "global_cb_assets_usd_bn"),
        ("US Net Liquidity", "us_net_liquidity_usd_bn"),
    ]
    for label, column in specs:
        if frame.empty or column not in frame.columns:
            rows.append({"Block": label, "History Start": "n/a", "Last Observation": "n/a", "Last Updated": "n/a", "Status": "MISSING"})
            continue
        rows_with_values = frame.dropna(subset=[column])
        status = "CURRENT" if not rows_with_values.empty else "MISSING"
        rows.append(
            {
                "Block": label,
                "History Start": _liquidity_fmt_date(rows_with_values["date"].min() if not rows_with_values.empty else None),
                "Last Observation": _liquidity_fmt_date(rows_with_values["date"].max() if not rows_with_values.empty else None),
                "Last Updated": str(frame["last_updated"].dropna().iloc[-1]) if "last_updated" in frame.columns and frame["last_updated"].notna().any() else "n/a",
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


BTC_SPOT_ETF_FLOW_TICKERS = ("IBIT", "FBTC", "GBTC", "ARKB", "BITB", "BTCO", "EZBC", "HODL", "BRRR", "BTCW")
BTC_HALVING_EVENTS = [
    {"date": pd.Timestamp("2016-07-09"), "label": "2016 Halving", "price": 650.0},
    {"date": pd.Timestamp("2020-05-11"), "label": "2020 Halving", "price": 8600.0},
    {"date": pd.Timestamp("2024-04-20"), "label": "2024 Halving", "price": 63800.0},
]
BTC_CYCLE_TOPS = [
    {"date": pd.Timestamp("2017-12-17"), "label": "2017 Top", "price": 19666.0},
    {"date": pd.Timestamp("2021-11-10"), "label": "2021 Top", "price": 69000.0},
    {"date": pd.Timestamp("2025-10-01"), "label": "2025 Top", "price": 126000.0},
]
BTC_CYCLE_BOTTOMS = [
    {"date": pd.Timestamp("2015-01-14"), "label": "2015 Bottom", "price": 172.0},
    {"date": pd.Timestamp("2018-12-15"), "label": "2018 Bottom", "price": 3200.0},
    {"date": pd.Timestamp("2022-11-21"), "label": "2022 Bottom", "price": 15600.0},
]
BTC_CURRENT_CYCLE_TOP_DATE = pd.Timestamp("2025-10-01")
BTC_CURRENT_CYCLE_TOP_PRICE = 126000.0


def render_btc_regime_tab(table_df: pd.DataFrame, market_snapshot: dict) -> None:
    st.subheader("BTC Regime")
    btc_price = load_btc_weekly_price()
    raw, monthly, weekly = read_global_liquidity()
    liquidity = _build_global_liquidity_regime_frame(_liquidity_prepare_dates(monthly), _liquidity_prepare_dates(weekly))
    liquidity_latest = _liquidity_latest_row_with_value(liquidity, "global_liquidity_score")
    btc_row = _btc_alpha_row(table_df)
    bybit = _btc_bybit_history()
    etf = _btc_etf_flow_history()
    snapshot = _build_btc_regime_snapshot(btc_price, liquidity_latest, btc_row, bybit, etf, market_snapshot)

    st.markdown("### BTC Regime Summary")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_market_metric("Halving Phase", snapshot["halving_phase"], f"{snapshot['months_since_halving']:.1f}M since halving")
    with summary_cols[1]:
        render_market_metric("Cycle Status", snapshot["cycle_bottom_status"], snapshot["bottom_note"])
    with summary_cols[2]:
        render_market_metric("Global Liquidity Regime", snapshot["global_liquidity_label"], f"{snapshot['liquidity_direction']} | {snapshot['long_cycle_phase']}")
    with summary_cols[3]:
        render_market_metric("Tactical Flow State", snapshot["tactical_flow_state"], snapshot["tactical_note"])

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_market_metric("BTC REGIME", snapshot["final_state"], snapshot["final_note"])
    with summary_cols[1]:
        render_market_metric("BTC Structural Macro", _liquidity_fmt_score_state(snapshot["structural_macro"]), "Global M2 13W 40% + DXY 40% + US2Y 20%")
    with summary_cols[2]:
        render_market_metric("BTC Forward Macro Risk", f"{_liquidity_fmt_number(snapshot['forward_macro_risk'], 1)} / {snapshot['forward_macro_risk_state']}", "M2 35% + DXY 30% + US2Y 20% + Credit 15%")
    with summary_cols[3]:
        render_market_metric("BTC Alpha", _liquidity_fmt_score_state(snapshot["btc_alpha"]), "existing Alpha Engine")

    st.markdown("### BTC Regime Interpretation")
    render_market_formula("Current interpretation", _btc_interpretation_text(snapshot))

    st.markdown("### BTC Price - Halving Cycle")
    btc_range = st.radio("BTC chart range", ["3Y", "5Y", "10Y", "MAX"], index=1, horizontal=True, key="btc_regime_price_range")
    btc_price_chart = _filter_date_range(btc_price, btc_range)
    btc_x_range = _btc_x_range(btc_price_chart)
    _render_btc_price_halving_chart(btc_price_chart, liquidity, btc_x_range)
    _render_btc_etf_flow_intensity_chart(etf, btc_x_range)

    st.markdown("### Structural Matrix")
    st.dataframe(_btc_halving_liquidity_matrix(snapshot), use_container_width=True, hide_index=True)

    st.markdown("### BTC Macro Regime")
    market_history = load_market_transition_history("btc-regime-market-transition-history")
    btc_macro_frame = _build_btc_macro_frame(liquidity, market_snapshot, market_history, btc_x_range)
    _render_btc_macro_score_chart(btc_macro_frame, "BTCStructuralMacro", "BTC Structural Macro", "#22c55e")
    _render_btc_macro_score_chart(btc_macro_frame, "BTCForwardMacroRisk", "BTC Forward Macro Risk", "#ef4444")
    st.markdown("### BTC Trend / Alpha")
    st.dataframe(_btc_alpha_table(btc_row), use_container_width=True, hide_index=True)

    st.markdown("### BTC Tactical Flows")
    st.dataframe(_btc_tactical_table(snapshot, etf, bybit), use_container_width=True, hide_index=True)

    st.markdown("### BTC Cycle Bottom Monitor")
    st.dataframe(_btc_bottom_monitor(snapshot), use_container_width=True, hide_index=True)

    st.markdown("### BTC Mature Cycle Multiples")
    _render_btc_cycle_multiples_chart()
    st.markdown("### BTC Cycle Valuation")
    st.dataframe(_btc_valuation_table(snapshot), use_container_width=True, hide_index=True)
    st.markdown("### Historical Cycle Table")
    st.dataframe(_btc_cycle_table(), use_container_width=True, hide_index=True)
    st.markdown("### Data Quality")
    st.dataframe(_btc_data_quality_table(btc_price, etf, bybit, liquidity_latest), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False, ttl=SLOW_REFRESH_SECONDS)
def load_btc_weekly_price() -> pd.DataFrame:
    daily = download_completed_ohlcv("BTC-USD", period="max")
    if daily.empty:
        return pd.DataFrame(columns=["date", "Open", "High", "Low", "Close", "Volume"])
    weekly = daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    weekly = weekly[weekly.index <= today]
    return weekly.reset_index().rename(columns={"index": "date", "Date": "date"})


def _btc_alpha_row(table_df: pd.DataFrame) -> dict[str, Any]:
    if table_df.empty or "Ticker" not in table_df.columns:
        return {}
    rows = table_df[table_df["Ticker"].astype(str).str.upper().eq("BTC-USD")]
    return rows.tail(1).to_dict("records")[0] if not rows.empty else {}


def _btc_bybit_history() -> pd.DataFrame:
    try:
        data = read_bybit_storage(BYBIT_STORAGE_PATH)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return data
    out = data[data["asset"].astype(str).str.upper().eq("BTC-USD")].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    out = out.dropna(subset=["date"]).loc[lambda frame: frame["date"] <= today].sort_values("date")
    if "oi_change_4w_pct" in out.columns:
        computed_percentile = _liquidity_trailing_percentile(
            pd.to_numeric(out["oi_change_4w_pct"], errors="coerce"),
            156,
            52,
        )
        if "oi_change_4w_percentile" not in out.columns:
            out["oi_change_4w_percentile"] = computed_percentile
        else:
            out["oi_change_4w_percentile"] = pd.to_numeric(out["oi_change_4w_percentile"], errors="coerce").fillna(computed_percentile)
    return out


def _btc_etf_flow_history() -> pd.DataFrame:
    cache = FundFlowCache(default_fund_flow_cache_path())
    frames = []
    for ticker in BTC_SPOT_ETF_FLOW_TICKERS:
        observations = cache.load_observations(ticker, pd.Timestamp("2024-01-01").date())
        if not observations:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime([obs.date for obs in observations]),
                    "ticker": ticker,
                    "net_flow": [obs.net_flow for obs in observations],
                    "aum": [obs.aum for obs in observations],
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ETF_Flow_1W",
                "ETF_Flow_4W",
                "ETF_Flow_13W",
                "ETF_Flow_Intensity_4W",
                "ETF_Flow_3Y_Pctl",
                "ETF_Flow_13W_Pctl",
                "ETF_Total_AUM",
                "ETF_Coverage_Count",
            ]
        )
    daily = pd.concat(frames, ignore_index=True)
    daily["net_flow"] = pd.to_numeric(daily["net_flow"], errors="coerce")
    daily["aum"] = pd.to_numeric(daily["aum"], errors="coerce")
    weekly_by_ticker = (
        daily.dropna(subset=["date", "net_flow"])
        .set_index("date")
        .groupby("ticker")
        .resample("W-FRI")
        .agg(net_flow=("net_flow", "sum"), aum=("aum", "last"))
        .dropna(subset=["net_flow"])
        .reset_index()
    )
    weekly = (
        weekly_by_ticker.groupby("date")
        .agg(
            ETF_Flow_1W=("net_flow", "sum"),
            ETF_Total_AUM=("aum", "sum"),
            ETF_Coverage_Count=("ticker", "nunique"),
        )
        .sort_index()
    )
    weekly["ETF_Flow_4W"] = weekly["ETF_Flow_1W"].rolling(4, min_periods=1).sum()
    weekly["ETF_Flow_13W"] = weekly["ETF_Flow_1W"].rolling(13, min_periods=1).sum()
    aum_ref = pd.to_numeric(weekly["ETF_Total_AUM"], errors="coerce").replace(0.0, np.nan)
    weekly["ETF_Flow_Intensity_4W"] = (weekly["ETF_Flow_4W"] / aum_ref) * 100.0
    fallback_scale = weekly["ETF_Flow_1W"].abs().rolling(156, min_periods=52).median()
    fallback_intensity = weekly["ETF_Flow_4W"] / (4.0 * fallback_scale.replace(0.0, np.nan))
    weekly["ETF_Flow_Intensity_4W"] = weekly["ETF_Flow_Intensity_4W"].where(weekly["ETF_Flow_Intensity_4W"].notna(), fallback_intensity)
    weekly["ETF_Flow_3Y_Pctl"] = _liquidity_trailing_percentile(weekly["ETF_Flow_Intensity_4W"], 156, 52)
    weekly["ETF_Flow_13W_Pctl"] = _liquidity_trailing_percentile(weekly["ETF_Flow_13W"], 156, 104)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    weekly = weekly[weekly.index <= today]
    return weekly.reset_index()


def _build_btc_regime_snapshot(
    price: pd.DataFrame,
    liquidity: dict[str, Any],
    btc_row: dict[str, Any],
    bybit: pd.DataFrame,
    etf: pd.DataFrame,
    market: dict,
) -> dict[str, Any]:
    current_date = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    latest_price = _btc_latest_price(price)
    halving = max([event for event in BTC_HALVING_EVENTS if event["date"] <= current_date], key=lambda item: item["date"])
    months_since_halving = _months_between(halving["date"], current_date)
    months_since_top = _months_between(BTC_CURRENT_CYCLE_TOP_DATE, current_date) if current_date >= BTC_CURRENT_CYCLE_TOP_DATE else np.nan
    phase = _btc_halving_phase(months_since_halving)
    liq_score = _safe_float(liquidity.get("global_liquidity_score"))
    liq_direction = str(liquidity.get("direction_13w_state", "DATA_INCOMPLETE"))
    btc_m2_growth_13w = _safe_float(liquidity.get("m2_13w"))
    btc_m2_bull = _safe_float(liquidity.get("m2_13w_pctl"))
    btc_m2_risk = 100.0 - btc_m2_bull if np.isfinite(btc_m2_bull) else np.nan
    dxy_risk = _safe_float(market.get("Macro_DXY_Risk", market.get("DXY_Risk")))
    us2y_risk = _safe_float(market.get("US2Y_Risk"))
    credit_risk = _safe_float(market.get("Credit_Risk"))
    credit_state = str(market.get("Credit_State", "DATA_INCOMPLETE"))
    structural_macro = _weighted_mean([btc_m2_bull, 100.0 - dxy_risk, 100.0 - us2y_risk], [0.40, 0.40, 0.20])
    forward_macro_risk = _weighted_mean([btc_m2_risk, dxy_risk, us2y_risk, credit_risk], [0.35, 0.30, 0.20, 0.15])
    alpha = _safe_float(btc_row.get("Alpha_Score"))
    latest_etf = _liquidity_latest_row(etf)
    latest_bybit = _liquidity_latest_row(bybit)
    tactical_state, tactical_note = _btc_tactical_state(latest_etf, latest_bybit, alpha)
    bottom_status, bottom_note = _btc_bottom_status(phase, liq_direction, alpha, latest_etf, latest_bybit, forward_macro_risk, credit_state)
    final_state, final_note = _btc_final_state(phase, liq_score, liq_direction, alpha, structural_macro, forward_macro_risk, tactical_state, bottom_status, btc_m2_bull, credit_state)
    multiple = _btc_expected_multiple(liq_score, liq_direction)
    candidate_bottom = _btc_candidate_bottom(price)
    drawdown = (latest_price / BTC_CURRENT_CYCLE_TOP_PRICE - 1.0) if np.isfinite(latest_price) else np.nan
    return {
        "date": current_date,
        "btc_price": latest_price,
        "halving_date": halving["date"],
        "halving_price": halving["price"],
        "halving_phase": phase,
        "months_since_halving": months_since_halving,
        "months_since_cycle_top": months_since_top,
        "global_liquidity_label": str(liquidity.get("final_regime_label", "n/a")),
        "global_liquidity_score": liq_score,
        "liquidity_direction": liq_direction,
        "btc_global_m2_growth_13w": btc_m2_growth_13w,
        "btc_global_m2_bull": btc_m2_bull,
        "btc_global_m2_risk": btc_m2_risk,
        "long_cycle_phase": str(liquidity.get("long_cycle_phase", "n/a")),
        "structural_macro": structural_macro,
        "forward_macro_risk": forward_macro_risk,
        "forward_macro_risk_state": _btc_risk_state(forward_macro_risk),
        "credit_risk": credit_risk,
        "credit_state": credit_state,
        "btc_alpha": alpha,
        "tactical_flow_state": tactical_state,
        "tactical_note": tactical_note,
        "cycle_bottom_status": bottom_status,
        "bottom_note": bottom_note,
        "final_state": final_state,
        "final_note": final_note,
        "expected_multiple": multiple,
        "expected_multiple_label": f"{multiple[0]:.1f}x-{multiple[2]:.1f}x",
        "projected_top_range": _btc_projected_top_range(candidate_bottom, multiple, bottom_status),
        "projected_top_note": "Cycle Bottom Not Confirmed" if bottom_status not in {"CANDIDATE_BOTTOM", "BOTTOM_CONFIRMED"} else "candidate bottom x multiple range",
        "candidate_bottom": candidate_bottom,
        "drawdown_from_top": drawdown,
        "last_updated": _btc_latest_observation_date(price, etf, bybit, liquidity.get("date")),
    }


def _btc_latest_price(price: pd.DataFrame) -> float:
    if price.empty or "Close" not in price.columns:
        return np.nan
    return _safe_float(pd.to_numeric(price["Close"], errors="coerce").dropna().iloc[-1])


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return ((end - start).days / 30.4375) if pd.notna(start) and pd.notna(end) else np.nan


def _btc_halving_phase(months_since_halving: float) -> str:
    if not np.isfinite(months_since_halving):
        return "DATA_INCOMPLETE"
    if months_since_halving < 6:
        return "POST_HALVING_EARLY"
    if months_since_halving < 12:
        return "BULL_EXPANSION"
    if months_since_halving < 18:
        return "LATE_BULL_PEAK_WINDOW"
    if months_since_halving < 30:
        return "POST_PEAK_BEAR"
    return "ACCUMULATION_PRE_HALVING"


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    value_arr = np.array(values, dtype="float64")
    weight_arr = np.array(weights, dtype="float64")
    mask = np.isfinite(value_arr)
    if not mask.any():
        return np.nan
    return float(np.average(value_arr[mask], weights=weight_arr[mask]))


def _btc_risk_state(value: float) -> str:
    if not np.isfinite(value):
        return "DATA_INCOMPLETE"
    if value < 20:
        return "LOW"
    if value < 40:
        return "MODERATE"
    if value < 60:
        return "ELEVATED"
    if value < 80:
        return "HIGH"
    return "EXTREME"


def _btc_tactical_state(etf: dict[str, Any], bybit: dict[str, Any], alpha: float) -> tuple[str, str]:
    flow_4w = _safe_float(etf.get("ETF_Flow_4W"))
    flow_13w = _safe_float(etf.get("ETF_Flow_13W"))
    oi_4w = _safe_float(bybit.get("oi_change_4w_pct"))
    oi_pctl = _safe_float(bybit.get("oi_change_4w_percentile"))
    funding_pctl = _safe_float(bybit.get("funding_28d_percentile"))
    funding_28d = _safe_float(bybit.get("funding_28d"))
    etf_supportive = np.isfinite(flow_4w) and flow_4w > 0 and np.isfinite(flow_13w) and flow_13w > 0
    etf_weak = np.isfinite(flow_4w) and flow_4w < 0
    oi_extreme = (np.isfinite(oi_pctl) and oi_pctl >= 80) or (np.isfinite(oi_4w) and oi_4w > 0.15)
    funding_extreme = (np.isfinite(funding_pctl) and funding_pctl >= 80) or (np.isfinite(funding_28d) and funding_28d > 0.02)
    if etf_supportive and np.isfinite(alpha) and alpha >= 60 and not oi_extreme and not funding_extreme:
        return "STRONG_CONFIRMATION", "ETF demand positive; leverage not extreme"
    if etf_weak and np.isfinite(oi_4w) and oi_4w < 0 and (not np.isfinite(funding_pctl) or funding_pctl < 60):
        return "DELEVERAGING", "ETF weak; OI falling; funding normalized"
    if (np.isfinite(alpha) and alpha >= 60 and etf_weak) or (oi_extreme and funding_extreme):
        return "FLOW_DIVERGENCE", "price/alpha strength not fully confirmed by flows or leverage"
    if oi_extreme or funding_extreme:
        return "OVERHEATED", "OI or funding is elevated"
    if etf_supportive:
        return "SUPPORTIVE", "ETF flows positive"
    return "NEUTRAL", "mixed or partial tactical data"


def _btc_bottom_status(
    phase: str,
    liq_direction: str,
    alpha: float,
    etf: dict[str, Any],
    bybit: dict[str, Any],
    macro_risk: float,
    credit_state: str = "",
) -> tuple[str, str]:
    if phase not in {"POST_PEAK_BEAR", "ACCUMULATION_PRE_HALVING"}:
        return "NO_BOTTOM_SIGNAL", "bottom module inactive outside bear/accumulation phases"
    signals = []
    credit_widening = str(credit_state or "").upper() in {"WIDENING", "SEVERE_WIDENING"}
    if liq_direction in {"IMPROVING", "ACCELERATING", "STABLE"}:
        signals.append("liquidity improving/stable")
    if credit_state and not credit_widening:
        signals.append("credit not widening")
    if np.isfinite(alpha) and alpha >= 50:
        signals.append("BTC Alpha stabilizing")
    if _safe_float(etf.get("ETF_Flow_4W")) > 0:
        signals.append("ETF flows positive")
    if _safe_float(bybit.get("oi_change_4w_pct")) < 0:
        signals.append("OI deleveraging")
    if abs(_safe_float(bybit.get("funding_28d"))) < 0.01:
        signals.append("funding normalized")
    if np.isfinite(macro_risk) and macro_risk <= 40:
        signals.append("macro risk contained")
    count = len(signals)
    if credit_widening and count >= 4:
        return "BOTTOMING_NOT_CONFIRMED", ", ".join(signals + [f"credit {credit_state}"])
    if count >= 5:
        return "BOTTOM_CONFIRMED", ", ".join(signals)
    if count >= 4:
        return "CANDIDATE_BOTTOM", ", ".join(signals)
    if count >= 3:
        return "BOTTOMING_WATCH", ", ".join(signals)
    if count >= 2:
        return "EARLY_BOTTOMING_SIGNS", ", ".join(signals)
    return "NO_BOTTOM_SIGNAL", "insufficient confirmation"


def _btc_final_state(
    phase: str,
    liq_score: float,
    liq_direction: str,
    alpha: float,
    structural_macro: float,
    forward_risk: float,
    tactical: str,
    bottom: str,
    btc_m2_bull: float = np.nan,
    credit_state: str = "",
) -> tuple[str, str]:
    deteriorating = liq_direction in {"DETERIORATING", "DETERIORATING_FAST"}
    m2_weak = np.isfinite(btc_m2_bull) and btc_m2_bull < 40.0
    m2_improving = np.isfinite(btc_m2_bull) and btc_m2_bull >= 60.0
    credit_widening = str(credit_state or "").upper() in {"WIDENING", "SEVERE_WIDENING"}
    tactical_weak = tactical in {"FLOW_DIVERGENCE", "OVERHEATED", "DELEVERAGING"}
    if phase == "LATE_BULL_PEAK_WINDOW" and sum([deteriorating or m2_weak, np.isfinite(alpha) and alpha < 60, tactical_weak, np.isfinite(forward_risk) and forward_risk > 60, credit_widening]) >= 2:
        return "TOP_RISK_HIGH", "late bull window with multiple deterioration warnings"
    if (deteriorating or m2_weak) and credit_widening:
        return "MACRO_CREDIT_STRESS", "Global M2 momentum is weak/deteriorating and credit is widening"
    if deteriorating or m2_weak:
        return "MACRO_HEADWIND", "Global M2 momentum is weak/deteriorating, but credit is not confirming stress"
    if phase == "POST_PEAK_BEAR" and np.isfinite(alpha) and alpha < 50 and liq_direction not in {"IMPROVING", "ACCELERATING"}:
        return "POST_CYCLE_BEAR", "post-peak phase; trend/alpha weak; liquidity not improving"
    if bottom in {"BOTTOMING_WATCH", "CANDIDATE_BOTTOM", "BOTTOM_CONFIRMED"}:
        return "BOTTOMING_WATCH", "bottom module has multiple confirmations"
    if bottom == "BOTTOMING_NOT_CONFIRMED":
        return "BOTTOMING_WATCH", "bottom setup exists but credit widening prevents confirmation"
    if phase == "ACCUMULATION_PRE_HALVING" and (liq_direction in {"IMPROVING", "ACCELERATING"} or m2_improving) and not credit_widening and np.isfinite(alpha) and alpha >= 60 and tactical in {"SUPPORTIVE", "STRONG_CONFIRMATION"}:
        return "STRONG_EARLY_BULL_SETUP", "accumulation timing confirmed by liquidity, alpha and flows"
    if phase in {"POST_HALVING_EARLY", "BULL_EXPANSION"} and np.isfinite(alpha) and alpha >= 70 and np.isfinite(liq_score) and liq_score >= 60 and not deteriorating and np.isfinite(structural_macro) and structural_macro >= 60 and np.isfinite(forward_risk) and forward_risk <= 40 and tactical in {"SUPPORTIVE", "STRONG_CONFIRMATION"}:
        return "HIGH_CONVICTION_BULL", "cycle, liquidity, macro, trend and flows aligned"
    if phase == "LATE_BULL_PEAK_WINDOW" and np.isfinite(alpha) and alpha >= 70 and not deteriorating and tactical in {"SUPPORTIVE", "STRONG_CONFIRMATION"}:
        return "LATE_CYCLE_BULL", "late-cycle trend remains confirmed"
    if np.isfinite(alpha) and alpha >= 60 and ((np.isfinite(forward_risk) and forward_risk > 60) or liq_direction == "DETERIORATING_FAST"):
        return "MACRO_WARNING", "high alpha is overridden by macro/liquidity warning"
    if np.isfinite(alpha) and alpha >= 60 and tactical in {"FLOW_DIVERGENCE", "OVERHEATED"}:
        return "FLOW_DIVERGENCE_WARNING", "price strength is not confirmed by flows/leverage"
    if np.isfinite(alpha) and alpha >= 60 and np.isfinite(liq_score) and liq_score >= 40 and not deteriorating:
        return "BULLISH", "trend positive with broadly supportive macro"
    if phase == "ACCUMULATION_PRE_HALVING":
        return "ACCUMULATION", "pre-halving accumulation phase"
    return "NEUTRAL", "no stronger rule matched"


def _btc_expected_multiple(score: float, direction: str) -> tuple[float, float, float]:
    if (np.isfinite(score) and score < 40) or direction in {"DETERIORATING", "DETERIORATING_FAST"}:
        return 2.0, 2.25, 2.5
    if np.isfinite(score) and score >= 60 and direction in {"IMPROVING", "ACCELERATING"}:
        return 4.0, 4.5, 5.0
    return 2.5, 3.0, 4.0


def _btc_candidate_bottom(price: pd.DataFrame) -> float:
    if price.empty or "date" not in price.columns or "Close" not in price.columns:
        return np.nan
    rows = price[pd.to_datetime(price["date"], errors="coerce") >= BTC_CURRENT_CYCLE_TOP_DATE].copy()
    if rows.empty:
        return np.nan
    return _safe_float(pd.to_numeric(rows["Close"], errors="coerce").min())


def _btc_projected_top_range(bottom: float, multiple: tuple[float, float, float], bottom_status: str) -> str:
    if bottom_status not in {"CANDIDATE_BOTTOM", "BOTTOM_CONFIRMED"} or not np.isfinite(bottom):
        return "N/A"
    return f"{fmt_money_compact(bottom * multiple[0])} - {fmt_money_compact(bottom * multiple[2])}"


def _btc_latest_observation_date(*items: Any) -> str:
    dates: list[pd.Timestamp] = []
    for item in items:
        if isinstance(item, pd.DataFrame):
            if item.empty or "date" not in item.columns:
                continue
            value = pd.to_datetime(item["date"], errors="coerce").max()
        else:
            value = pd.to_datetime(item, errors="coerce")
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        if pd.notna(value) and pd.Timestamp(value).tz_localize(None) <= today:
            dates.append(pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value))
    return _liquidity_fmt_date(max(dates)) if dates else "n/a"


def _btc_interpretation_text(s: dict[str, Any]) -> str:
    top_text = "n/a" if not np.isfinite(s["months_since_cycle_top"]) else f"{s['months_since_cycle_top']:.1f}"
    return (
        f"BTC is in {s['halving_phase']}, about {s['months_since_halving']:.1f} months after the {pd.Timestamp(s['halving_date']).strftime('%Y-%m-%d')} halving "
        f"and {top_text} months after the fixed October 2025 cycle top.\n\n"
        f"Global Liquidity is {s['global_liquidity_label']} with score {fmt_plain_number(s['global_liquidity_score'], 1)} "
        f"and 13W direction {s['liquidity_direction']}. The long liquidity cycle is {s['long_cycle_phase']}.\n\n"
        f"BTC-specific liquidity uses Global M2 13W momentum: bull score {fmt_plain_number(s.get('btc_global_m2_bull'), 1)} "
        f"and 13W growth {fmt_plain_percent(s.get('btc_global_m2_growth_13w'))}. Credit state is {s.get('credit_state', 'n/a')}.\n\n"
        f"BTC Structural Macro is {fmt_plain_number(s['structural_macro'], 1)}, Forward Macro Risk is "
        f"{fmt_plain_number(s['forward_macro_risk'], 1)} / {s['forward_macro_risk_state']}, and BTC Alpha is {fmt_plain_number(s['btc_alpha'], 1)}.\n\n"
        f"Tactical flows are {s['tactical_flow_state']} ({s['tactical_note']}). Cycle Bottom Status is {s['cycle_bottom_status']} ({s['bottom_note']}).\n\n"
        f"Final state is {s['final_state']}: {s['final_note']}. This is a regime/risk framework, not a deterministic price forecast."
    )


def _filter_date_range(frame: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or range_key == "MAX":
        return frame
    years = {"3Y": 3, "5Y": 5, "10Y": 10}.get(range_key)
    if years is None:
        return frame
    max_date = pd.to_datetime(frame["date"], errors="coerce").max()
    return frame[pd.to_datetime(frame["date"], errors="coerce") >= max_date - pd.DateOffset(years=years)].copy()


def _render_btc_price_halving_chart(price: pd.DataFrame, liquidity: pd.DataFrame, x_range: list[pd.Timestamp] | None = None) -> None:
    if price.empty:
        st.info("No BTC price history.")
        return
    d = price.dropna(subset=["date", "Close"]).copy()
    liquidity_frame = _btc_filter_to_x_range(liquidity, x_range)
    fig = go.Figure()
    for _, band in _btc_phase_bands(d).iterrows():
        fig.add_vrect(x0=band["start"], x1=band["end"], fillcolor=band["color"], opacity=0.14, line_width=0)
    fig.add_trace(go.Scatter(x=d["date"], y=d["Close"], mode="lines", name="BTC", line={"color": "#f7931a", "width": 2.0}))
    if not liquidity_frame.empty and "global_liquidity_score" in liquidity_frame.columns:
        fig.add_trace(
            go.Scatter(
                x=liquidity_frame["date"],
                y=pd.to_numeric(liquidity_frame["global_liquidity_score"], errors="coerce"),
                mode="lines",
                name="Global Liquidity Score",
                yaxis="y2",
                line={"color": "#38bdf8", "width": 1.7},
                hovertemplate="Week: %{x|%Y-%m-%d}<br>Global Liquidity Score: %{y:.1f}<extra></extra>",
            )
        )
    for event in BTC_HALVING_EVENTS:
        _btc_add_marker(fig, event["date"], event["label"], "#38bdf8")
    for event in BTC_CYCLE_TOPS:
        _btc_add_marker(fig, event["date"], event["label"], "#ef4444")
    for event in BTC_CYCLE_BOTTOMS:
        _btc_add_marker(fig, event["date"], event["label"], "#22c55e")
    fig.update_layout(
        yaxis={"title": "BTC USD", "type": "log"},
        yaxis2={
            "title": "Global Liquidity Score",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
            "color": "#cbd5e1",
            "linecolor": "#475569",
        },
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(_style_liquidity_plotly(fig, 390, "BTC Price - Halving Cycle"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _btc_phase_bands(price: pd.DataFrame) -> pd.DataFrame:
    min_date = pd.to_datetime(price["date"], errors="coerce").min()
    max_date = pd.to_datetime(price["date"], errors="coerce").max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame()
    starts = [event["date"] for event in BTC_HALVING_EVENTS if event["date"] <= max_date]
    rows = []
    colors = {
        "POST_HALVING_EARLY": "#22c55e",
        "BULL_EXPANSION": "#84cc16",
        "LATE_BULL_PEAK_WINDOW": "#facc15",
        "POST_PEAK_BEAR": "#ef4444",
        "ACCUMULATION_PRE_HALVING": "#38bdf8",
    }
    for start in starts:
        for m0, m1 in [(0, 6), (6, 12), (12, 18), (18, 30), (30, 48)]:
            phase = _btc_halving_phase(m0 + 0.1)
            band_start = max(start + pd.DateOffset(months=m0), min_date)
            band_end = min(start + pd.DateOffset(months=m1), max_date)
            if band_end >= band_start:
                rows.append({"start": band_start, "end": band_end, "phase": phase, "color": colors.get(phase, "#64748b")})
    return pd.DataFrame(rows)


def _btc_add_marker(fig: go.Figure, date: pd.Timestamp, label: str, color: str) -> None:
    marker_x = pd.Timestamp(date).strftime("%Y-%m-%d")
    fig.add_shape(type="line", xref="x", yref="paper", x0=marker_x, x1=marker_x, y0=0, y1=1, line={"color": color, "dash": "dot", "width": 1})
    fig.add_annotation(x=marker_x, y=1.03, xref="x", yref="paper", text=label, showarrow=False, font={"color": color, "size": 10}, xanchor="left")


def _btc_halving_liquidity_matrix(snapshot: dict[str, Any]) -> pd.DataFrame:
    phase = snapshot["halving_phase"]
    direction = snapshot["liquidity_direction"]
    score = snapshot["global_liquidity_score"]
    btc_m2_bull = snapshot.get("btc_global_m2_bull")
    credit_state = snapshot.get("credit_state", "n/a")
    strong = np.isfinite(score) and score >= 60
    weak = np.isfinite(score) and score < 40
    improving = direction in {"IMPROVING", "ACCELERATING"}
    deteriorating = direction in {"DETERIORATING", "DETERIORATING_FAST"}
    if phase == "POST_HALVING_EARLY":
        state = "HIGHLY_SUPPORTIVE" if strong or improving else "MIXED_DELAYED_EXPANSION"
    elif phase == "BULL_EXPANSION":
        state = "STRONG_BULL" if strong or improving else "BULLISH_BUT_FRAGILE"
    elif phase == "LATE_BULL_PEAK_WINDOW":
        state = "LATE_BULL_CYCLE_MAY_EXTEND" if improving else "TOP_RISK_HIGH" if deteriorating else "LATE_CYCLE_NEUTRAL"
    elif phase == "POST_PEAK_BEAR":
        state = "BOTTOMING_WATCH" if improving else "BEARISH_DELEVERAGING" if deteriorating or weak else "POST_PEAK_NEUTRAL"
    else:
        state = "STRONG_EARLY_BULL_SETUP" if improving else "WEAK_ACCUMULATION" if deteriorating else "ACCUMULATION"
    return pd.DataFrame(
        [
            {
                "Halving Phase": phase,
                "Global Liquidity Score": fmt_plain_number(score, 1),
                "BTC Global M2 13W Bull": fmt_plain_number(btc_m2_bull, 1),
                "Direction": direction,
                "Credit State": credit_state,
                "Structural Interpretation": state,
            }
        ]
    )


def _btc_x_range(frame: pd.DataFrame) -> list[pd.Timestamp] | None:
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return [pd.Timestamp(dates.min()), pd.Timestamp(dates.max())]


def _btc_filter_to_x_range(frame: pd.DataFrame, x_range: list[pd.Timestamp] | None) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or not x_range:
        return frame.copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).loc[lambda data: (data["date"] >= x_range[0]) & (data["date"] <= x_range[1])].copy()


def _build_btc_macro_frame(
    liquidity: pd.DataFrame,
    market: dict,
    market_history: pd.DataFrame | None,
    x_range: list[pd.Timestamp] | None,
) -> pd.DataFrame:
    frame = _btc_filter_to_x_range(liquidity, x_range)
    if frame.empty:
        return frame
    frame = frame.copy()
    if market_history is not None and not market_history.empty and "Date" in market_history.columns:
        market_cols = [column for column in ["Date", "Macro_DXY_Risk", "US2Y_Risk", "Credit_Risk"] if column in market_history.columns]
        mh = market_history[market_cols].copy()
        mh["date"] = pd.to_datetime(mh["Date"], errors="coerce")
        mh = mh.dropna(subset=["date"]).drop(columns=["Date"], errors="ignore").sort_values("date")
        frame = pd.merge_asof(
            frame.sort_values("date"),
            mh,
            on="date",
            direction="backward",
        )
    for column, fallback_key in [
        ("Macro_DXY_Risk", "Macro_DXY_Risk"),
        ("US2Y_Risk", "US2Y_Risk"),
        ("Credit_Risk", "Credit_Risk"),
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(_safe_float(market.get(fallback_key)))
    m2_source = frame["m2_13w_pctl"] if "m2_13w_pctl" in frame.columns else pd.Series(np.nan, index=frame.index)
    m2_bull = pd.to_numeric(m2_source, errors="coerce")
    m2_risk = 100.0 - m2_bull
    dxy_risk = pd.to_numeric(frame["Macro_DXY_Risk"], errors="coerce")
    us2y_risk = pd.to_numeric(frame["US2Y_Risk"], errors="coerce")
    credit_risk = pd.to_numeric(frame["Credit_Risk"], errors="coerce")
    frame["BTCGlobalM2Bull13W"] = m2_bull
    frame["BTCGlobalM2Risk13W"] = m2_risk
    frame["BTCStructuralMacro"] = [
        _weighted_mean([m2, 100.0 - dxy, 100.0 - us2y], [0.40, 0.40, 0.20])
        for m2, dxy, us2y in zip(m2_bull, dxy_risk, us2y_risk)
    ]
    frame["BTCForwardMacroRisk"] = [
        _weighted_mean([m2, dxy, us2y, credit], [0.35, 0.30, 0.20, 0.15])
        for m2, dxy, us2y, credit in zip(m2_risk, dxy_risk, us2y_risk, credit_risk)
    ]
    return frame


def _render_btc_macro_score_chart(frame: pd.DataFrame, metric: str, title: str, color: str) -> None:
    if frame.empty or "date" not in frame.columns or metric not in frame.columns:
        st.info(f"No data available for {title}.")
        return
    d = frame.dropna(subset=["date", metric]).copy()
    if d.empty:
        st.info(f"No data available for {title}.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=pd.to_numeric(d[metric], errors="coerce"), mode="lines", name=title, line={"color": color, "width": 1.9}))
    for level in [20, 40, 60, 80]:
        fig.add_hline(y=level, line={"color": "#94a3b8", "dash": "dot", "width": 1}, opacity=0.5)
    fig.update_layout(yaxis={"title": "Score", "range": _btc_adaptive_y_range(d[metric])})
    x_range = _btc_x_range(frame)
    if x_range:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(_style_liquidity_plotly(fig, 250, title), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _btc_adaptive_y_range(values: pd.Series, lower_bound: float = 0.0, upper_bound: float = 100.0) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [lower_bound, upper_bound]
    vmin = float(numeric.min())
    vmax = float(numeric.max())
    span = max(vmax - vmin, 5.0)
    padding = max(span * 0.18, 3.0)
    return [max(lower_bound, vmin - padding), min(upper_bound, vmax + padding)]


def _btc_alpha_table(row: dict[str, Any]) -> pd.DataFrame:
    cols = ["Alpha_Score", "Momentum_Score", "Trend_Quality_Score", "Persistence_Score", "Alpha_State", "Opportunity_State", "Entry_Risk"]
    return pd.DataFrame([{"Metric": col, "Value": row.get(col, "n/a")} for col in cols])


def _btc_tactical_table(snapshot: dict[str, Any], etf: pd.DataFrame, bybit: pd.DataFrame) -> pd.DataFrame:
    latest_etf = _liquidity_latest_row(etf)
    latest_bybit = _liquidity_latest_row(bybit)
    return pd.DataFrame(
        [
            {"Metric": "Tactical Flow State", "Value": snapshot["tactical_flow_state"]},
            {"Metric": "ETF Flow 1W", "Value": fmt_money_compact(latest_etf.get("ETF_Flow_1W"))},
            {"Metric": "ETF Flow 4W", "Value": fmt_money_compact(latest_etf.get("ETF_Flow_4W"))},
            {"Metric": "ETF Flow 13W", "Value": fmt_money_compact(latest_etf.get("ETF_Flow_13W"))},
            {"Metric": "ETF Flow Intensity 4W", "Value": fmt_plain_number(latest_etf.get("ETF_Flow_Intensity_4W"), 2)},
            {"Metric": "ETF Flow Intensity Percentile", "Value": fmt_plain_number(latest_etf.get("ETF_Flow_3Y_Pctl"), 0)},
            {"Metric": "OI USD", "Value": fmt_money_compact(latest_bybit.get("open_interest_usd"))},
            {"Metric": "OI Change 4W", "Value": fmt_plain_percent(latest_bybit.get("oi_change_4w_pct"))},
            {"Metric": "Funding 28D", "Value": fmt_plain_percent(latest_bybit.get("funding_28d"))},
            {"Metric": "Basis / Premium", "Value": fmt_plain_percent_from_pct(latest_bybit.get("perp_premium_pct"))},
        ]
    )


def _render_btc_etf_flow_intensity_chart(etf: pd.DataFrame, x_range: list[pd.Timestamp] | None = None) -> None:
    _render_btc_dual_axis_percentile_chart(
        etf,
        left_metric="ETF_Flow_Intensity_4W",
        right_metric="ETF_Flow_3Y_Pctl",
        title="BTC ETF Fund Flows - 4W Flow Intensity and Trailing 3Y Percentile",
        left_name="4W Flow Intensity",
        right_name="3Y Percentile",
        left_color="#22d3ee",
        right_color="#facc15",
        left_axis_title="4W Flow Intensity, normalized",
        right_axis_title="Trailing 3Y Percentile",
        x_range=x_range,
    )


def _render_btc_bybit_positioning_percentile_chart(bybit: pd.DataFrame) -> None:
    _render_btc_dual_axis_percentile_chart(
        bybit,
        left_metric="oi_change_4w_pct",
        right_metric="oi_change_4w_percentile",
        title="BTC Bybit Positioning - OI Change 4W and Trailing Percentile",
        left_name="OI Change 4W",
        right_name="OI Percentile",
        left_color="#60a5fa",
        right_color="#facc15",
        left_axis_title="OI Change 4W",
        right_axis_title="Trailing Percentile",
        left_multiplier=100.0,
        left_suffix="%",
    )


def _render_btc_dual_axis_percentile_chart(
    data: pd.DataFrame,
    left_metric: str,
    right_metric: str,
    title: str,
    left_name: str,
    right_name: str,
    left_color: str,
    right_color: str,
    left_axis_title: str,
    right_axis_title: str,
    left_multiplier: float = 1.0,
    left_suffix: str = "",
    x_range: list[pd.Timestamp] | None = None,
) -> None:
    if data.empty or not all(column in data.columns for column in ["date", left_metric, right_metric]):
        st.info(f"No data available for {title}.")
        return
    d = _btc_filter_to_x_range(data, x_range)
    if d.empty:
        d = data.dropna(subset=["date"]).copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d[left_metric] = pd.to_numeric(d[left_metric], errors="coerce") * left_multiplier
    d[right_metric] = pd.to_numeric(d[right_metric], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    if d[[left_metric, right_metric]].dropna(how="all").empty:
        st.info(f"No data available for {title}.")
        return
    latest_date = _frame_date_max(d)
    status = "CURRENT" if latest_date != "n/a" else "PARTIAL_DATA"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d[left_metric],
            mode="lines",
            name=left_name,
            line={"color": left_color, "width": 1.8},
            hovertemplate=f"Week: %{{x|%Y-%m-%d}}<br>{left_name}: %{{y:.2f}}{left_suffix}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d[right_metric],
            mode="lines",
            name=right_name,
            yaxis="y2",
            line={"color": right_color, "width": 1.8, "dash": "dash"},
            hovertemplate=f"Week: %{{x|%Y-%m-%d}}<br>{right_name}: %{{y:.0f}}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(
        yaxis={"title": left_axis_title},
        yaxis2={
            "title": right_axis_title,
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
            "color": "#cbd5e1",
            "linecolor": "#475569",
        },
        annotations=[
            {
                "text": f"Last Updated: {latest_date} - {status}",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.12,
                "showarrow": False,
                "font": {"color": "#cbd5e1", "size": 11},
                "xanchor": "left",
            }
        ],
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(_style_liquidity_plotly(fig, 320, title), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_btc_etf_flows_chart(etf: pd.DataFrame) -> None:
    if etf.empty:
        st.info("No BTC ETF flow history.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=etf["date"], y=etf["ETF_Flow_1W"], name="1W", marker_color="#38bdf8"))
    fig.add_trace(go.Scatter(x=etf["date"], y=etf["ETF_Flow_4W"], mode="lines", name="4W", line={"color": "#facc15", "width": 1.7}))
    fig.add_trace(go.Scatter(x=etf["date"], y=etf["ETF_Flow_13W"], mode="lines", name="13W", line={"color": "#22c55e", "width": 1.7}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(yaxis={"title": "Net flow, USD"})
    st.plotly_chart(_style_liquidity_plotly(fig, 300, "BTC ETF Flows"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_btc_open_interest_chart(bybit: pd.DataFrame) -> None:
    if bybit.empty:
        st.info("No BTC Open Interest history.")
        return
    d = bybit.copy()
    fig = go.Figure()
    if "open_interest_usd" in d.columns:
        fig.add_trace(go.Scatter(x=d["date"], y=pd.to_numeric(d["open_interest_usd"], errors="coerce"), mode="lines", name="OI USD", line={"color": "#38bdf8", "width": 1.8}))
    for col, label, color in [("oi_change_1w_pct", "OI Change 1W", "#facc15"), ("oi_change_4w_pct", "OI Change 4W", "#f97316"), ("oi_change_13w_pct", "OI Change 13W", "#ef4444")]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d["date"], y=pd.to_numeric(d[col], errors="coerce") * 100.0, mode="lines", name=label, yaxis="y2", line={"color": color, "width": 1.4}))
    fig.update_layout(yaxis={"title": "OI USD"}, yaxis2={"title": "Change, %", "overlaying": "y", "side": "right", "showgrid": False})
    st.plotly_chart(_style_liquidity_plotly(fig, 300, "BTC Open Interest"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_btc_funding_chart(bybit: pd.DataFrame) -> None:
    if bybit.empty:
        st.info("No BTC Funding history.")
        return
    d = bybit.copy()
    fig = go.Figure()
    for col, label, color in [("funding_1d", "Funding 1D", "#38bdf8"), ("funding_7d", "Funding 7D", "#facc15"), ("funding_28d", "Funding 28D", "#22c55e")]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d["date"], y=pd.to_numeric(d[col], errors="coerce") * 100.0, mode="lines", name=label, line={"color": color, "width": 1.5}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(yaxis={"title": "Funding, %"})
    st.plotly_chart(_style_liquidity_plotly(fig, 260, "BTC Funding"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _render_btc_basis_chart(bybit: pd.DataFrame) -> None:
    if bybit.empty:
        st.info("No BTC Basis / Premium history.")
        return
    d = bybit.copy()
    fig = go.Figure()
    for col, label, color in [("perp_premium_pct", "Current", "#38bdf8"), ("perp_premium_7d_avg", "1W avg", "#facc15"), ("perp_premium_28d_avg", "4W avg", "#22c55e")]:
        if col in d.columns:
            values = pd.to_numeric(d[col], errors="coerce")
            if values.notna().any():
                fig.add_trace(go.Scatter(x=d["date"], y=values, mode="lines", name=label, line={"color": color, "width": 1.5}))
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.update_layout(yaxis={"title": "Premium, %"})
    st.plotly_chart(_style_liquidity_plotly(fig, 250, "BTC Perpetual Basis / Premium"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _btc_bottom_monitor(snapshot: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Signal": "Drawdown from Oct 2025 top", "State": fmt_plain_percent(snapshot["drawdown_from_top"])},
            {"Signal": "BTC Alpha", "State": _liquidity_fmt_score_state(snapshot["btc_alpha"])},
            {"Signal": "Global Liquidity Direction", "State": snapshot["liquidity_direction"]},
            {"Signal": "ETF / Derivatives", "State": snapshot["tactical_flow_state"]},
            {"Signal": "Macro Risk", "State": f"{_liquidity_fmt_number(snapshot['forward_macro_risk'], 1)} / {snapshot['forward_macro_risk_state']}"},
            {"Signal": "Cycle Bottom Status", "State": snapshot["cycle_bottom_status"]},
        ]
    )


def _render_btc_cycle_multiples_chart() -> None:
    data = pd.DataFrame(
        [
            {"Cycle": "2015-2017", "Metric": "Bottom->Top", "Multiple": 114.0},
            {"Cycle": "2015-2017", "Metric": "Bottom->Halving", "Multiple": 3.8},
            {"Cycle": "2015-2017", "Metric": "Halving->Top", "Multiple": 30.0},
            {"Cycle": "2018-2021", "Metric": "Bottom->Top", "Multiple": 21.5},
            {"Cycle": "2018-2021", "Metric": "Bottom->Halving", "Multiple": 2.7},
            {"Cycle": "2018-2021", "Metric": "Halving->Top", "Multiple": 7.9},
            {"Cycle": "2022-2025", "Metric": "Bottom->Top", "Multiple": 8.0},
            {"Cycle": "2022-2025", "Metric": "Bottom->Halving", "Multiple": 4.1},
            {"Cycle": "2022-2025", "Metric": "Halving->Top", "Multiple": 2.0},
        ]
    )
    fig = go.Figure()
    colors = {"Bottom->Halving": "#22c55e", "Halving->Top": "#facc15", "Bottom->Top": "#38bdf8"}
    for metric in data["Metric"].drop_duplicates():
        d = data[data["Metric"].eq(metric)]
        fig.add_trace(go.Bar(x=d["Cycle"], y=d["Multiple"], name=metric, marker_color=colors.get(metric, "#cbd5e1")))
    fig.update_layout(barmode="group", yaxis={"title": "Multiple, x"})
    st.plotly_chart(_style_liquidity_plotly(fig, 300, "BTC Mature Cycle Multiples"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def _btc_valuation_table(snapshot: dict[str, Any]) -> pd.DataFrame:
    bottom_ok = snapshot["cycle_bottom_status"] in {"CANDIDATE_BOTTOM", "BOTTOM_CONFIRMED"}
    low, base, high = snapshot["expected_multiple"]
    bottom = snapshot["candidate_bottom"]
    return pd.DataFrame(
        [
            {"Metric": "Current Price", "Value": fmt_money_compact(snapshot["btc_price"])},
            {"Metric": "Current Drawdown from Cycle Top", "Value": fmt_plain_percent(snapshot["drawdown_from_top"])},
            {"Metric": "Candidate Cycle Bottom", "Value": fmt_money_compact(bottom) if np.isfinite(bottom) else "n/a"},
            {"Metric": "Bottom Confidence", "Value": snapshot["cycle_bottom_status"]},
            {"Metric": "Expected Bottom->Top Multiple", "Value": f"{low:.1f}x / {base:.1f}x / {high:.1f}x"},
            {"Metric": "Projected Top Low", "Value": fmt_money_compact(bottom * low) if bottom_ok and np.isfinite(bottom) else "N/A - Cycle Bottom Not Confirmed"},
            {"Metric": "Projected Top Base", "Value": fmt_money_compact(bottom * base) if bottom_ok and np.isfinite(bottom) else "N/A - Cycle Bottom Not Confirmed"},
            {"Metric": "Projected Top High", "Value": fmt_money_compact(bottom * high) if bottom_ok and np.isfinite(bottom) else "N/A - Cycle Bottom Not Confirmed"},
        ]
    )


def _btc_cycle_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Cycle": "2015-2017", "Halving Date": "2016-07-09", "Halving Price": "$650", "Cycle Top Date": "2017-12-17", "Cycle Top Price": "$19,666", "Days Halving->Top": 526, "Halving->Top Multiple": "30.3x", "Cycle Bottom": "2015-01-14", "Bottom->Top Multiple": "114x"},
            {"Cycle": "2018-2021", "Halving Date": "2020-05-11", "Halving Price": "$8,600", "Cycle Top Date": "2021-11-10", "Cycle Top Price": "$69,000", "Days Halving->Top": 548, "Halving->Top Multiple": "8.0x", "Cycle Bottom": "2018-12-15", "Bottom->Top Multiple": "21.5x"},
            {"Cycle": "2022-2025", "Halving Date": "2024-04-20", "Halving Price": "$63,800", "Cycle Top Date": "2025-10-01", "Cycle Top Price": "$126,000", "Days Halving->Top": 529, "Halving->Top Multiple": "2.0x", "Cycle Bottom": "2022-11-21", "Bottom->Top Multiple": "8.0x"},
        ]
    )


def _btc_data_quality_table(price: pd.DataFrame, etf: pd.DataFrame, bybit: pd.DataFrame, liquidity: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Block": "BTC Price", "History Start": _frame_date_min(price), "Last Observation": _frame_date_max(price), "Status": "CURRENT" if not price.empty else "MISSING"},
            {"Block": "BTC ETF Flow", "History Start": _frame_date_min(etf), "Last Observation": _frame_date_max(etf), "Status": "CURRENT" if not etf.empty else "PARTIAL_DATA"},
            {"Block": "Bybit OI", "History Start": _frame_date_min(bybit.dropna(subset=["open_interest_usd"]) if "open_interest_usd" in bybit.columns else pd.DataFrame()), "Last Observation": _frame_date_max(bybit), "Status": "CURRENT" if not bybit.empty else "PARTIAL_DATA"},
            {"Block": "Funding", "History Start": _frame_date_min(bybit.dropna(subset=["funding_28d"]) if "funding_28d" in bybit.columns else pd.DataFrame()), "Last Observation": _frame_date_max(bybit), "Status": "CURRENT" if not bybit.empty else "PARTIAL_DATA"},
            {"Block": "Basis / Premium", "History Start": _frame_date_min(bybit.dropna(subset=["perp_premium_pct"]) if "perp_premium_pct" in bybit.columns else pd.DataFrame()), "Last Observation": _frame_date_max(bybit.dropna(subset=["perp_premium_pct"]) if "perp_premium_pct" in bybit.columns else pd.DataFrame()), "Status": "CURRENT" if "perp_premium_pct" in bybit.columns and bybit["perp_premium_pct"].notna().any() else "PARTIAL_DATA"},
            {"Block": "Global Liquidity", "History Start": "see Global Liquidity Regime", "Last Observation": str(liquidity.get("date", "n/a")), "Status": str(liquidity.get("data_status", "n/a"))},
        ]
    )


def _frame_date_min(frame: pd.DataFrame) -> str:
    if frame.empty or "date" not in frame.columns:
        return "n/a"
    return _liquidity_fmt_date(pd.to_datetime(frame["date"], errors="coerce").min())


def _frame_date_max(frame: pd.DataFrame) -> str:
    if frame.empty or "date" not in frame.columns:
        return "n/a"
    return _liquidity_fmt_date(pd.to_datetime(frame["date"], errors="coerce").max())


def _liquidity_prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _liquidity_latest_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.dropna(how="all").tail(1).to_dict("records")[0]


def _liquidity_latest_row_with_value(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    rows = frame.dropna(subset=[column]).dropna(how="all")
    if rows.empty:
        return {}
    return rows.tail(1).to_dict("records")[0]


def _liquidity_fmt_number(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:,.{decimals}f}"
    except Exception:
        return "n/a"


def _liquidity_fmt_trillions(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"${number / 1000.0:,.2f}T"
    except Exception:
        return "n/a"


def _liquidity_fmt_bn(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"${number:,.0f}B"
    except Exception:
        return "n/a"


def _liquidity_fmt_pct(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number * 100.0:+.1f}%"
    except Exception:
        return "n/a"


def _liquidity_line_chart(
    frame: pd.DataFrame,
    title: str,
    series_map: dict[str, tuple[str, float]],
    y_title: str,
    height: int = 260,
    zero_line: bool = False,
) -> None:
    if frame.empty or "date" not in frame.columns:
        st.info(f"No data for {title}.")
        return
    rows = []
    for label, (column, divisor) in series_map.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce") / divisor
        for date, value in zip(frame["date"], values):
            if pd.notna(date) and np.isfinite(value):
                rows.append({"Date": date, "Series": label, "Value": float(value)})
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info(f"No data for {title}.")
        return
    base = alt.Chart(chart_df).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b'%y", labelFontSize=9, labelOverlap=True)),
        y=alt.Y("Value:Q", axis=alt.Axis(title=y_title)),
        color=alt.Color("Series:N", legend=alt.Legend(title=None)),
        tooltip=[
            alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("Series:N"),
            alt.Tooltip("Value:Q", format=",.2f"),
        ],
    )
    line = base.mark_line(strokeWidth=1.7)
    points = base.mark_circle(size=18, opacity=0.45)
    layers = [line, points]
    if zero_line:
        layers.insert(0, alt.Chart(chart_df).mark_rule(color="#64748b").encode(y=alt.datum(0)))
    st.altair_chart(alt.layer(*layers).properties(height=height, title=title), use_container_width=True)


def _render_liquidity_regional_chart(monthly: pd.DataFrame) -> None:
    mode = st.radio(
        "Regional M2 Contribution",
        ["LEVEL", "12M CHANGE CONTRIBUTION"],
        horizontal=True,
        key="global_liquidity_regional_mode",
    )
    if mode == "LEVEL":
        mapping = {
            "US": ("us_m2_usd_bn", 1000.0),
            "Euro Area": ("ea_m2_usd_bn", 1000.0),
            "China": ("china_m2_usd_bn", 1000.0),
            "Japan": ("japan_m2_usd_bn", 1000.0),
        }
        y_title = "USD trillions"
    else:
        mapping = {
            "US": ("us_contribution_12m", 0.01),
            "Euro Area": ("ea_contribution_12m", 0.01),
            "China": ("china_contribution_12m", 0.01),
            "Japan": ("japan_contribution_12m", 0.01),
        }
        y_title = "% of 12M change"
    rows = []
    for region, (column, divisor) in mapping.items():
        if column not in monthly.columns:
            continue
        values = pd.to_numeric(monthly[column], errors="coerce") / divisor
        for date, value in zip(monthly["date"], values):
            if pd.notna(date) and np.isfinite(value):
                rows.append({"Date": date, "Region": region, "Value": float(value)})
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        st.info("No regional Global M2 data.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_area(opacity=0.8)
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b'%y", labelFontSize=9, labelOverlap=True)),
            y=alt.Y("Value:Q", stack="zero", axis=alt.Axis(title=y_title)),
            color=alt.Color("Region:N", legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("Region:N"),
                alt.Tooltip("Value:Q", format=",.2f"),
            ],
        )
        .properties(height=300, title=f"Regional M2 Contribution - {mode}")
    )
    st.altair_chart(chart, use_container_width=True)


def _build_liquidity_regional_table(raw: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    region_config = [
        ("US", "M2SL", None, "us_m2_usd_bn", "us_m2_share"),
        ("Euro Area", "M.U2.Y.V.M20.X.1.U2.2300.Z01.E", "DEXUSEU", "ea_m2_usd_bn", "ea_m2_share"),
        ("China", "Money & Quasi-money (M2)", "DEXCHUS", "china_m2_usd_bn", "china_m2_share"),
        ("Japan", "MD02'MAM1NAM2M2MO", "DEXJPUS", "japan_m2_usd_bn", "japan_m2_share"),
    ]
    rows = []
    for region, series_id, fx_id, usd_col, share_col in region_config:
        source = raw[raw["series_id"].eq(series_id)].copy() if not raw.empty else pd.DataFrame()
        source = source.dropna(subset=["raw_value", "observation_date"]) if not source.empty else source
        latest_source = source.sort_values("observation_date").tail(1).to_dict("records")[0] if not source.empty else {}
        latest_region = _liquidity_latest_row_with_value(monthly, usd_col)
        latest_share = _liquidity_latest_row_with_value(monthly, share_col)
        fx_value = "1.00" if fx_id is None else _liquidity_latest_raw_value(raw, fx_id)
        rows.append(
            {
                "Region": region,
                "M2 Local": _liquidity_fmt_number(latest_source.get("raw_value"), 2),
                "FX": fx_value,
                "M2 USD": _liquidity_fmt_trillions(latest_region.get(usd_col)),
                "Share of Global M2": _liquidity_fmt_pct(latest_share.get(share_col)),
                "1M": _liquidity_fmt_region_growth(monthly, usd_col, 1),
                "3M": _liquidity_fmt_region_growth(monthly, usd_col, 3),
                "6M": _liquidity_fmt_region_growth(monthly, usd_col, 6),
                "12M": _liquidity_fmt_region_growth(monthly, usd_col, 12),
                "Last Update": str(latest_source.get("download_timestamp", "n/a")),
                "Status": str(latest_source.get("data_status", "ERROR")),
            }
        )
    return pd.DataFrame(rows)


def _liquidity_latest_raw_value(raw: pd.DataFrame, series_id: str) -> str:
    if raw.empty:
        return "n/a"
    source = raw[raw["series_id"].eq(series_id)].dropna(subset=["raw_value", "observation_date"]).copy()
    if source.empty:
        return "n/a"
    return _liquidity_fmt_number(source.sort_values("observation_date")["raw_value"].iloc[-1], 4)


def _liquidity_fmt_region_growth(monthly: pd.DataFrame, column: str, periods: int) -> str:
    if monthly.empty or column not in monthly.columns:
        return "n/a"
    values = pd.to_numeric(monthly[column], errors="coerce").dropna()
    if len(values) <= periods:
        return "n/a"
    return _liquidity_fmt_pct(values.iloc[-1] / values.iloc[-periods - 1] - 1.0)


def _build_liquidity_cb_table(raw: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    cb_config = [
        ("Fed", "WALCL", "FRED / WALCL", "fed_assets_usd_bn", "fed_cb_share"),
        ("ECB", "ILM.W.U2.C.T000000.Z5.Z01", "ECB Data API / ILM.W.U2.C.T000000.Z5.Z01", "ecb_assets_usd_bn", "ecb_cb_share"),
        ("BoJ", "BS01'MABJMTA", "BOJ Time-Series API / BS01 MABJMTA", "boj_assets_usd_bn", "boj_cb_share"),
        ("PBoC", "PBOC_TOTAL_ASSETS", "PBoC / Balance Sheet of Monetary Authority", "pboc_assets_usd_bn", "pboc_cb_share"),
    ]
    rows = []
    for central_bank, series_id, source_label, usd_col, share_col in cb_config:
        source = raw[raw["series_id"].eq(series_id)].copy() if not raw.empty else pd.DataFrame()
        source = source.dropna(subset=["observation_date"]) if not source.empty else source
        latest_source = source.sort_values("observation_date").tail(1).to_dict("records")[0] if not source.empty else {}
        latest_value = _liquidity_latest_row_with_value(monthly, usd_col)
        latest_share = _liquidity_latest_row_with_value(monthly, share_col)
        rows.append(
            {
                "Central Bank": central_bank,
                "Source": source_label,
                "Source Series": series_id,
                "Native Frequency": str(latest_source.get("frequency", "n/a")),
                "Source Unit": str(latest_source.get("unit", "n/a")),
                "Raw Value": _liquidity_fmt_number(latest_source.get("raw_value"), 2),
                "USD Value": _liquidity_fmt_trillions(latest_value.get(usd_col)),
                "Share of Global CB": _liquidity_fmt_pct(latest_share.get(share_col)),
                "1M": _liquidity_fmt_region_growth(monthly, usd_col, 1),
                "3M": _liquidity_fmt_region_growth(monthly, usd_col, 3),
                "6M": _liquidity_fmt_region_growth(monthly, usd_col, 6),
                "12M": _liquidity_fmt_region_growth(monthly, usd_col, 12),
                "Last Observation": _liquidity_fmt_date(latest_source.get("observation_date")),
                "Last Release": _liquidity_fmt_date(latest_source.get("release_date")),
                "Last Update": str(latest_source.get("download_timestamp", "n/a")),
                "Status": str(latest_source.get("data_status", "ERROR")),
            }
        )
    return pd.DataFrame(rows)


def _build_liquidity_quality_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["Source", "Series ID", "Native Frequency", "History Start", "Last Observation", "Last Release", "Missing Data Count", "Status"])
    quality = []
    for (source, series_id), group in raw.groupby(["source", "series_id"], dropna=False):
        observation_dates = pd.to_datetime(group["observation_date"], errors="coerce")
        release_dates = pd.to_datetime(group["release_date"], errors="coerce")
        quality.append(
            {
                "Source": source,
                "Series ID": series_id,
                "Native Frequency": str(group["frequency"].dropna().iloc[-1]) if group["frequency"].notna().any() else "n/a",
                "History Start": _liquidity_fmt_date(observation_dates.min()),
                "Last Observation": _liquidity_fmt_date(observation_dates.max()),
                "Last Release": _liquidity_fmt_date(release_dates.max()),
                "Missing Data Count": int(pd.to_numeric(group["raw_value"], errors="coerce").isna().sum()),
                "Status": str(group["data_status"].dropna().iloc[-1]) if group["data_status"].notna().any() else "ERROR",
            }
        )
    return pd.DataFrame(quality).sort_values(["Source", "Series ID"])


def _liquidity_fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fmt_plain_number(value: Any, decimals: int = 1) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:,.{decimals}f}"
    except Exception:
        return "n/a"


def fmt_plain_percent(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number * 100:.2f}%"
    except Exception:
        return "n/a"


def fmt_plain_percent_from_pct(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:.3f}%"
    except Exception:
        return "n/a"


def fmt_money_compact(value: Any) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        sign = "-" if number < 0 else ""
        number = abs(number)
        if number >= 1_000_000_000:
            return f"{sign}${number / 1_000_000_000:.2f}B"
        if number >= 1_000_000:
            return f"{sign}${number / 1_000_000:.2f}M"
        return f"{sign}${number:,.0f}"
    except Exception:
        return "n/a"


def render_description_tab() -> None:
    st.subheader("Metric Calculation Description")
    st.markdown(
        """
### Alpha Engine
- Alpha Engine is a cross-sectional scoring model. It compares all assets in the selected universe, converts raw metrics into normalized `0..100` component scores, then combines them into `Alpha Score`, `Entry Risk`, `Opportunity State`, and `Opportunity Score`.
- If one of the required core components is missing, `Alpha Score` and `Opportunity Score` are set to missing.

#### Market Regime and Alpha Confidence
- Market regime is calculated from weekly `SPY` data.
- `SPY_SMA40W = SMA(SPY weekly close, 40)`
- `SPY_Drawdown_52W = SPY_Close / 52W_High - 1`
- `SPY_Volatility_13W = Std(weekly returns, 13) * sqrt(52)`
- `SPY_Volatility_Percentile = percentile rank of current 13W volatility`
- Structural bull:
  - `SPY_Close > SPY_SMA40W`
  - `SPY_Drawdown_52W > -10%`
- High volatility:
  - `SPY_Volatility_Percentile >= 75`
- Regime classification:
  - Structural bull and normal volatility -> `BULL`
  - Structural bull and high volatility -> `BULL_HIGH_VOL`
  - Not structural bull and normal volatility -> `CORRECTION`
  - Not structural bull and high volatility -> `STRESS`
- `Alpha Confidence`:
  - `BULL`: `100`
  - `BULL_HIGH_VOL`: `100`
  - `CORRECTION`: `50`
  - `STRESS`: `20`

#### Momentum Score
- Uses cross-sectional percentile ranks across the selected universe.
- `Perf1M_Rank = percentile_rank(Perf1M %)`
- `Perf3M_Rank = percentile_rank(Perf3M %)`
- `Perf6M_Rank = percentile_rank(Perf6M %)`
- Formula:
  - `Momentum Score = 0.20*Perf1M_Rank + 0.45*Perf3M_Rank + 0.35*Perf6M_Rank`

#### Trend Quality Score
- `Trend Quality Score` combines ADX/DI trend, SMA regime, and distance to the 52-week high.
- ADX/DI block:
  - `DI_Balance = (DI_Plus_14 - DI_Minus_14) / (DI_Plus_14 + DI_Minus_14)`
  - `DI_Bull_Score = 100 * clamp(DI_Balance / 0.25, 0..1)`
  - `ADX_Strength_Score` is piecewise-scored:
    - `ADX 15 -> 0`
    - `ADX 20 -> 40`
    - `ADX 25 -> 70`
    - `ADX 30 -> 90`
    - `ADX 35 -> 100`
  - `ADX_DI_Trend_Score = ADX_Strength_Score * DI_Bull_Score / 100`
- SMA regime block:
  - `Absolute_SMA_Score` uses `SMA50w_vs_SMA200w_Spread_%`:
    - `-5 -> 0`
    - `0 -> 50`
    - `5 -> 100`
  - `SMA200d_Robust_Z_36M` compares the current daily `Close/SMA200d - 1` spread with its last 36 months using median and MAD:
    - `robust_sigma = 1.4826 * MAD`
    - `SMA200d_Robust_Z_36M = (latest_spread - median_spread) / robust_sigma`
  - `Relative_SMA_Score` uses `SMA200d_Robust_Z_36M`:
    - `-2 -> 0`
    - `-1 -> 25`
    - `0 -> 50`
    - `1 -> 70`
    - `2 -> 90`
    - `3 -> 100`
  - If relative SMA data is not available, `Relative_SMA_Score` falls back to `50`.
  - `SMA_Regime_Score = 0.65*Absolute_SMA_Score + 0.35*Relative_SMA_Score`
- 52-week high block:
  - `High52W_Score` uses `Price_vs_52W_High_%`:
    - `-25 -> 0`
    - `-15 -> 50`
    - `-5 -> 90`
    - `0 -> 100`
- Formula:
  - `Trend Quality Score = 0.55*ADX_DI_Trend_Score + 0.30*SMA_Regime_Score + 0.15*High52W_Score`

#### Persistence Score
- Uses cross-sectional percentile ranks and historical consistency.
- `Perf12M_Rank = percentile_rank(Perf12M %)`
- `Median_Rank = median(Perf1M_Rank, Perf3M_Rank, Perf6M_Rank, Perf12M_Rank)`
- `Min_Core_Rank = min(Perf3M_Rank, Perf6M_Rank, Perf12M_Rank)`
- `Positive_Breadth_Score = count(positive Perf1M/3M/6M/12M) * 25`
- `Historical_12M_Score = 100 * clamp(Perf12M_Percentile / 85, 0..1)`
- Formula:
  - `Persistence Score = 0.35*Median_Rank + 0.25*Min_Core_Rank + 0.20*Positive_Breadth_Score + 0.20*Historical_12M_Score`

#### Alpha Score and Alpha State
- `Base Alpha = 0.35*Momentum Score + 0.30*Trend Quality Score + 0.35*Persistence Score`
- `Alpha Score = clamp(Base Alpha, 0..100)`
- `Alpha State`:
  - `>= 80`: `Strong Alpha`
  - `>= 70`: `Positive Alpha`
  - `>= 60`: `Moderate Alpha`
  - `>= 50`: `Neutral`
  - `< 50`: `Weak`

#### Entry Risk
- `Entry Risk Score` is a penalty score from `0..100`, where a lower value means a cleaner entry.
- It combines regime-dependent `SMA200d_Robust_Z_36M` risk and extreme 12-month percentile risk.
- Regime-dependent SMAZ risk:
  - `BULL`: `0` if `SMAZ <= 3.0`, otherwise `10`
  - `BULL_HIGH_VOL`: `0` if `SMAZ <= 2.5`, `5` if `<= 3.0`, otherwise `10`
  - `CORRECTION`: `10` if `SMAZ <= 1.5`, `20` if `<= 2.5`, `30` if `<= 3.0`, otherwise `50`
  - `STRESS`: `20` if `SMAZ <= 1.5`, `30` if `<= 2.5`, `50` if `<= 3.0`, otherwise `80`
- 12-month percentile risk:
  - `Perf12M_Percentile <= 98`: `0`
  - `98 < Perf12M_Percentile <= 99`: `5`
  - `Perf12M_Percentile > 99`: `10`
- Base formula:
  - `Entry Risk Score = clamp(Regime_Dependent_SMAZ_Risk + Perf12M_Extreme_Risk, 0..100)`
- Stress override:
  - If `Market_Regime == STRESS`, `Momentum Score >= 70`, and `SMA200d_Robust_Z_36M > 3`, then `Entry Risk Score` is at least `80`.
- `Entry Risk` label:
  - `<= 20`: `Low`
  - `<= 40`: `Moderate`
  - `<= 60`: `Elevated`
  - `<= 80`: `High`
  - `> 80`: `Extreme`

#### Opportunity Score and Opportunity State
- `Opportunity Score` adjusts `Alpha Score` by current entry risk:
  - `Opportunity Score = Alpha Score * (1 - Entry Risk Score / 100)`
- `Opportunity State`:
  - Missing inputs or unknown regime -> `Missing Data`
  - `STRESS` and `Entry Risk Score >= 60` -> `STRESS_AVOID_CHASING`
  - `Alpha Score < 50` -> `WEAK`
  - `Alpha Score >= 75`, `Entry Risk Score <= 30`, and regime is `BULL` or `BULL_HIGH_VOL` -> `HIGH_CONVICTION`
  - `Alpha Score >= 65` and regime is `CORRECTION` -> `LOW_CONFIDENCE`
  - `Alpha Score >= 70` and `Entry Risk Score > 40` -> `STRONG_BUT_EXTENDED`
  - `Alpha Score >= 65` and `Entry Risk Score <= 40` -> `ATTRACTIVE`
  - Otherwise -> `NEUTRAL`

#### Alpha Sorting
- `Off`: keeps the current table order.
- `Alpha Score`: sorts by `Alpha Score`, then `Persistence Score`, `Trend Quality Score`, `Momentum Score`, and lower `Entry Risk Score`.
- `Opportunity State`: sorts by opportunity bucket, then `Opportunity Score`, `Alpha Score`, and lower `Entry Risk Score`.
- `Entry Risk`: sorts by lower `Entry Risk Score`, then higher `Alpha Score` and `Opportunity Score`.
- `Alpha Confidence`: sorts by higher `Alpha Confidence`, then higher `Alpha Score` and lower `Entry Risk Score`.
- `Opportunity Score`: sorts by higher `Opportunity Score`, then higher `Alpha Score` and lower `Entry Risk Score`.

### Performance Ratios
- `Perf1D %`  
  Formula: `(Close_now / Close_(now-1 calendar day on-or-before) - 1) * 100`
- `Perf1W %`  
  Formula: `(Close_now / Close_(now-7 calendar days on-or-before) - 1) * 100`
- `Perf1M %`  
  Formula: `(Close_now / Close_(now-30 calendar days on-or-before) - 1) * 100`
- `Perf3M %`  
  Formula: `(Close_now / Close_(now-90 calendar days on-or-before) - 1) * 100`
- `Perf6M %`  
  Formula: `(Close_now / Close_(now-182 calendar days on-or-before) - 1) * 100`
- `Perf12M %`  
  Formula: `(Close_now / Close_(now-365 calendar days on-or-before) - 1) * 100`
- `Perf3Y %`  
  Formula: `(Close_now / Close_(now-3*365 calendar days on-or-before) - 1) * 100`
- `Perf5Y %`  
  Formula: `(Close_now / Close_(now-5*365 calendar days on-or-before) - 1) * 100`
- `Perf10Y %`  
  Formula: `(Close_now / Close_(now-10*365 calendar days on-or-before) - 1) * 100`

### `BBPosition`
- Calculated on completed **weekly** bars.
- Midline: `Mid = SMA(weekly close, 50)`
- Bands: `Upper = Mid + 2*Std(50)`, `Lower = Mid - 2*Std(50)`
- Score range: `-10..+10`
  - `>= Upper` -> `+10`
  - `<= Lower` -> `-10`
  - Between `Mid` and `Upper` -> `+1..+9` (10 equal zones)
  - Between `Lower` and `Mid` -> `-1..-9` (10 equal zones)
  - At `Mid` -> `0`

### RSI
- `RSI 14D` is calculated on daily close prices with a 14-bar window.
- `RSI 14W` is calculated on completed weekly close prices with a 14-bar window.

### `SMATrend`
- Uses daily `SMA200`.
- `spread_now = (Close_now / SMA200_now - 1) * 100`
- `spread_6m_ago` computed on/just before ~182 days ago.
- Trend:
  - `bull` if `spread_now - spread_6m_ago > 0`
  - `bear` otherwise

### Divergences (`DivRSI`, `DivMACD`, `DivROC`)
- Indicators:
  - `DivRSI` uses `RSI(14)`
  - `DivMACD` uses `MACD histogram (12,26,9)`
  - `DivROC` uses `ROC(12)`
- Pivot detection:
  - Price pivots are found on `Low` (for bull checks) and `High` (for bear checks).
  - Indicator pivots are found the same way (`low`/`high` mode).
  - A pivot at index `i` is valid if it is min/max inside a symmetric window:
    `i - pivot_window ... i + pivot_window`.
- Pivot alignment:
  - Each price pivot is matched to the nearest indicator pivot within
    `± alignment_tolerance` bars.
  - Unmatched pivots are ignored.
- Candidate construction:
  - Candidates are built from consecutive matched pivot pairs `(t1, t2)`.
  - Hard filters:
    - `t2` must be inside latest `lookback_bars`
    - `(t2 - t1) <= max_span`
    - `abs((p2-p1)/p1) >= min_price_move`
    - `abs(i2-i1) >= min_ind_move`
- `min_ind_move`:
  - RSI: fixed threshold `min_ind_move_rsi`
  - MACD/ROC: volatility-scaled threshold
    `k * (rolling_std(indicator, rolling_std_n) + eps)`
    where `k = min_ind_move_macd_std` or `min_ind_move_roc_std`
- Divergence type rules:
  - On price lows:
    - `regular_bull`: `p2 < p1` and `i2 > i1`
    - `hidden_bull`: `p2 > p1` and `i2 < i1`
  - On price highs:
    - `regular_bear`: `p2 > p1` and `i2 < i1`
    - `hidden_bear`: `p2 < p1` and `i2 > i1`
- Scoring:
  - `mp = abs((p2-p1)/p1)` (price move magnitude)
  - `mi = abs(i2-i1)/(std_here + eps)` (indicator move normalized by volatility)
  - `ts = clamp((t2-t1)/max_span, 0..1)` (time-span contribution)
  - `zone_bonus`:
    - RSI: `1` if bull and RSI below 30, or bear and RSI above 70
    - MACD/ROC: `1` if crossing zero or at strong extreme (`>= 1.5*std`)
  - Final score:
    `score = 100 * clamp(wp*mp + wi*mi + wt*ts + wz*zone_bonus, 0..1)`
- Final output per indicator:
  - Candidate with highest score is selected.
  - Output column value is:
    - `bull`
    - `bear`
    - `nothing` (if no candidate passes all filters)
"""
    )


def render_alpha_engine_tab(df: pd.DataFrame) -> None:
    available = df.dropna(subset=["Alpha_Score"]).copy()
    if available.empty:
        st.info("Alpha data unavailable")
        return

    available = sort_by_alpha(available).reset_index(drop=True)
    alpha_columns = ["Group", "Subgroup", "Ticker", *ALPHA_CORE_COLUMNS]
    alpha_view = available[[col for col in alpha_columns if col in available.columns]].copy()

    st.dataframe(
        alpha_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Alpha_Score": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f"),
            "Alpha_State": st.column_config.TextColumn(width="large"),
            "Momentum_Score": st.column_config.NumberColumn(format="%.1f"),
            "Trend_Quality_Score": st.column_config.NumberColumn(format="%.1f"),
            "Persistence_Score": st.column_config.NumberColumn(format="%.1f"),
            "Alpha_Confidence": st.column_config.NumberColumn(format="%.0f"),
            "Entry_Risk_Score": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f"),
            "Opportunity_Score": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f"),
        },
    )

    detail_left, detail_right = st.columns([1.1, 1.4])
    with detail_left:
        tickers = available["Ticker"].astype(str).tolist()
        selected_ticker = st.selectbox("Ticker", options=tickers, key="alpha_engine_ticker")
    row = available[available["Ticker"].astype(str) == selected_ticker].iloc[0]

    with detail_right:
        st.markdown(
            f"""
**Alpha Score:** {row["Alpha_Score"]:.1f}

| Component | Value |
| --- | ---: |
| Base Alpha | {row["Base_Alpha"]:.1f} |
| Momentum | {row["Momentum_Score"]:.1f} |
| Trend Quality | {row["Trend_Quality_Score"]:.1f} |
| Persistence | {row["Persistence_Score"]:.1f} |
| Market Regime | {row["Market_Regime"]} |
| Alpha Confidence | {row["Alpha_Confidence"]:.0f} |
| Entry Risk Score | {row["Entry_Risk_Score"]:.1f} |
| Entry Risk | {row["Entry_Risk"]} |
| Opportunity State | {row["Opportunity_State"]} |
| Opportunity Score | {row["Opportunity_Score"]:.1f} |
| SMA Z36M | {row["SMA200d_Robust_Z_36M"]:.2f} |
| Perf12M Percentile | {row["Perf_12M_Percentile"]:.1f} |
| Alpha Data Complete | {bool(row["Alpha_Data_Complete"])} |
"""
        )

    technical_columns = [
        "Ticker",
        "ADX_DI_Trend_Score",
        "DI_Balance",
        "DI_Plus_14",
        "DI_Minus_14",
        "SMA_Regime_Score",
        "Absolute_SMA_Score",
        "Relative_SMA_Score",
        "SMA200d_Robust_Z_36M",
        "Regime_Dependent_SMAZ_Risk",
        "Perf12M_Extreme_Risk",
        "Base_Alpha",
        "SPY_vs_SMA40W_%",
        "SPY_Drawdown_52W_%",
        "SPY_Volatility_13W_%",
        "SPY_Volatility_Percentile",
        "Alpha_Data_Complete",
    ]
    technical_view = available[[col for col in technical_columns if col in available.columns]].copy()
    with st.expander("Technical Alpha Fields", expanded=False):
        st.dataframe(
            technical_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ADX_DI_Trend_Score": st.column_config.NumberColumn(format="%.1f"),
                "DI_Balance": st.column_config.NumberColumn(format="%.2f"),
                "DI_Plus_14": st.column_config.NumberColumn(format="%.1f"),
                "DI_Minus_14": st.column_config.NumberColumn(format="%.1f"),
                "SMA_Regime_Score": st.column_config.NumberColumn(format="%.1f"),
                "Absolute_SMA_Score": st.column_config.NumberColumn(format="%.1f"),
                "Relative_SMA_Score": st.column_config.NumberColumn(format="%.1f"),
                "SMA200d_Robust_Z_36M": st.column_config.NumberColumn(format="%.2f"),
                "Perf_12M_Percentile": st.column_config.NumberColumn(format="%.1f"),
                "Regime_Dependent_SMAZ_Risk": st.column_config.NumberColumn(format="%.1f"),
                "Perf12M_Extreme_Risk": st.column_config.NumberColumn(format="%.1f"),
                "Base_Alpha": st.column_config.NumberColumn(format="%.1f"),
                "SPY_vs_SMA40W_%": st.column_config.NumberColumn(format="%.1f"),
                "SPY_Drawdown_52W_%": st.column_config.NumberColumn(format="%.1f"),
                "SPY_Volatility_13W_%": st.column_config.NumberColumn(format="%.1f"),
                "SPY_Volatility_Percentile": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def render_top_alpha_status(
    market_slot,
    fast_transition_slot,
    macro_transition_slot,
    overall_status_slot,
    entry_risk_slot,
    confidence_slot,
    market: dict,
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> None:
    market_source = df.dropna(subset=["Market_Regime"]) if "Market_Regime" in df.columns else pd.DataFrame()
    if market:
        regime = format_market_value(market.get("Market_Regime"))
        confidence = format_market_value(market.get("Alpha_Confidence"), "score")
    elif not market_source.empty:
        regime = str(market_source["Market_Regime"].iloc[0])
        confidence_value = pd.to_numeric(market_source["Alpha_Confidence"], errors="coerce").dropna()
        confidence = "n/a" if confidence_value.empty else f"{confidence_value.iloc[0]:.0f}"
    else:
        regime = "n/a"
        confidence = "n/a"

    fast_value = (
        f"{format_market_value(market.get('Fast_Transition_Risk'), 'score')} / "
        f"{format_market_value(market.get('Fast_Transition_State'))}"
        if market
        else "n/a"
    )
    macro_value = (
        f"{format_market_value(market.get('Macro_Transition_Risk'), 'score')} / "
        f"{format_market_value(market.get('Macro_Transition_State'))}"
        if market
        else "n/a"
    )
    overall_value = format_market_value(market.get("Final_Market_State", market.get("Overall_Transition_Status"))) if market else "n/a"

    entry_scores = pd.to_numeric(filtered_df.get("Entry_Risk_Score", pd.Series(dtype="float64")), errors="coerce").dropna()
    if entry_scores.empty:
        entry_summary = "n/a"
        entry_detail = "No scored assets"
    else:
        avg_risk = float(entry_scores.mean())
        high_count = int((entry_scores > 60.0).sum())
        entry_summary = f"{avg_risk:.1f}"
        entry_detail = f"{entry_risk_category(avg_risk)} avg | {high_count} high+"

    with market_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Market Regime</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{regime}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with fast_transition_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Fast Transition Risk</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{fast_value}</div>
  <div style="font-size: 0.68rem; color: #cbd5e1;">VIX 70% + DXY 30%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with macro_transition_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Macro Transition Risk</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{macro_value}</div>
  <div style="font-size: 0.68rem; color: #cbd5e1;">DXY 40% + US2Y 30% + Global M2 20% + Fed liquidity 10%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with overall_status_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Final Market State</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{overall_value}</div>
  <div style="font-size: 0.68rem; color: #cbd5e1;">structural + risk layers</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with confidence_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Alpha Confidence</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{confidence}</div>
  <div style="font-size: 0.68rem; color: #cbd5e1;">Market regime confidence</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with entry_risk_slot.container():
        st.markdown(
            f"""
<div style="padding-top: 1.35rem; line-height: 1.1;">
  <div style="font-size: 0.68rem; color: #94a3b8; font-weight: 700;">Entry Risk</div>
  <div style="font-size: 0.9rem; color: #f8fafc; font-weight: 800;">{entry_summary}</div>
  <div style="font-size: 0.68rem; color: #cbd5e1;">{entry_detail}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def entry_risk_category(score: float) -> str:
    if not np.isfinite(score):
        return "n/a"
    if score <= 20.0:
        return "Low"
    if score <= 40.0:
        return "Moderate"
    if score <= 60.0:
        return "Elevated"
    if score <= 80.0:
        return "High"
    return "Extreme"


def extract_ohlcv_frame(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if px is None or px.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    try:
        if isinstance(px.columns, pd.MultiIndex):
            cols = px.columns
            sub = None

            # (Ticker, Field)
            if ticker in cols.get_level_values(0):
                sub = px[ticker]
            # (Field, Ticker)
            elif ticker in cols.get_level_values(1):
                sub = px.xs(ticker, axis=1, level=1)

            if sub is None or sub.empty:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            out = sub.copy()
        else:
            out = px.copy()

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
        if not keep or "Close" not in keep:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        out = out[keep].copy().dropna(subset=["Close"]).sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in out.columns:
                out[col] = np.nan
        return out[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def extract_market_ohlcv_frame(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = extract_ohlcv_frame(px, ticker)
    if out.empty:
        return out
    try:
        raw = px
        if isinstance(px.columns, pd.MultiIndex):
            if ticker in px.columns.get_level_values(0):
                raw = px[ticker]
            elif ticker in px.columns.get_level_values(1):
                raw = px.xs(ticker, axis=1, level=1)
        if ticker in {"SPY", "IWM", "XLI", "XLP"} and "Adj Close" in raw.columns:
            adj_close = pd.to_numeric(raw["Adj Close"], errors="coerce").reindex(out.index)
            out.loc[adj_close.notna(), "Close"] = adj_close.loc[adj_close.notna()]
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False, ttl=SLOW_REFRESH_SECONDS)
def load_ohlcv_history(tickers: list, universe_signature: str) -> dict:
    _ = universe_signature
    data = {}
    if not tickers:
        return data

    unique_tickers = list(dict.fromkeys(tickers))
    try:
        px = yf.download(
            unique_tickers,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        px = pd.DataFrame()

    missing_tickers = []
    for ticker in unique_tickers:
        frame = extract_ohlcv_frame(px, ticker)
        data[ticker] = frame
        if frame.empty:
            missing_tickers.append(ticker)

    # Bulk download can intermittently miss individual tickers; retry them one-by-one.
    for ticker in missing_tickers:
        try:
            px_single = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            px_single = pd.DataFrame()
        single_frame = extract_ohlcv_frame(px_single, ticker)
        if not single_frame.empty:
            data[ticker] = single_frame
    return data


def get_fred_api_key_for_app() -> str | None:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["FRED_API_KEY"]).strip() if "FRED_API_KEY" in st.secrets else None
    except StreamlitSecretNotFoundError:
        return None


def enrich_market_snapshot_with_global_liquidity(market: dict) -> dict:
    enriched = dict(market or {})
    try:
        _, monthly, weekly = read_global_liquidity()
        liquidity = _build_global_liquidity_regime_frame(
            _liquidity_prepare_dates(monthly),
            _liquidity_prepare_dates(weekly),
        )
        latest = _liquidity_latest_row_with_value(liquidity, "global_liquidity_score")
    except Exception:
        latest = {}

    score = _safe_float(latest.get("global_liquidity_score"))
    direction = _safe_float(latest.get("direction_13w"))
    direction_state = str(latest.get("direction_13w_state", "DATA_INCOMPLETE"))
    backdrop = classify_global_liquidity_backdrop(score, direction, direction_state)
    final_state = calculate_overall_transition_status(
        enriched.get("Fast_Transition_Risk"),
        enriched.get("Macro_Transition_Risk"),
        enriched.get("Negative_Confirmation_Count"),
        structural_regime=str(enriched.get("Market_Regime", "UNKNOWN")),
        global_liquidity_backdrop=backdrop,
        global_liquidity_score=score,
        global_liquidity_direction_13w=direction,
        global_liquidity_direction_state=direction_state,
        credit_state=enriched.get("Credit_State", ""),
    )
    enriched.update(
        {
            "Global_Liquidity_Backdrop": backdrop,
            "Global_Liquidity_Score": score,
            "Global_Liquidity_Direction_13W": direction,
            "Global_Liquidity_Direction_13W_State": direction_state,
            "Long_Liquidity_Cycle": str(latest.get("long_cycle_phase", "DATA_INCOMPLETE")),
            "Global_Liquidity_Last_Updated": str(latest.get("last_updated", "n/a")),
            "Global_Liquidity_Data_Status": str(latest.get("data_status", "DATA_INCOMPLETE")),
            "Overall_Transition_Status": final_state,
            "Final_Market_State": final_state,
        }
    )
    return enrich_market_snapshot_with_positioning_tail_risk(enriched)


def load_positioning_history_for_market_regime() -> pd.DataFrame:
    try:
        from positioning import read_processed

        master = read_processed("cftc_master")
        aaii = read_processed("aaii")
        return calculate_positioning_risk_history(aaii, master)
    except Exception:
        return pd.DataFrame()


def enrich_market_snapshot_with_positioning_tail_risk(market: dict) -> dict:
    enriched = dict(market or {})
    history = pd.DataFrame([{"Date": pd.Timestamp.now(tz="UTC").tz_localize(None).normalize(), **enriched}])
    positioning_history = load_positioning_history_for_market_regime()
    tail = calculate_tail_risk_history(history, positioning_history)
    if tail.empty:
        return enriched
    row = tail.iloc[-1]
    for key in [
        "AAII_Bearish_3Y_Percentile",
        "VIX_AssetManager_NetPctOI",
        "VIX_AssetManager_NetPctOI_3Y_Percentile",
        "PositioningRisk",
        "PositioningState",
        "LiquidityWarning",
        "CreditWarning",
        "FastWarning",
        "MacroWarning",
        "TailRiskFlag",
        "TailRiskReason",
        "PositioningModel_Version",
        "TailRiskModel_Version",
    ]:
        if key in row.index:
            enriched[key] = row.get(key)
    return enriched


def global_m2_weekly_series_for_market() -> pd.Series:
    try:
        _, monthly, weekly = read_global_liquidity()
        liquidity = _build_global_liquidity_regime_frame(
            _liquidity_prepare_dates(monthly),
            _liquidity_prepare_dates(weekly),
        )
    except Exception:
        return pd.Series(dtype="float64")
    if liquidity.empty or "date" not in liquidity.columns or "global_m2_usd_bn" not in liquidity.columns:
        return pd.Series(dtype="float64")
    values = liquidity[["date", "global_m2_usd_bn"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["global_m2_usd_bn"] = pd.to_numeric(values["global_m2_usd_bn"], errors="coerce")
    values = values.dropna(subset=["date", "global_m2_usd_bn"]).sort_values("date")
    if values.empty:
        return pd.Series(dtype="float64")
    return values.set_index("date")["global_m2_usd_bn"]


@st.cache_data(show_spinner=True, ttl=SLOW_REFRESH_SECONDS)
def load_market_model_snapshot(cache_signature: str) -> dict:
    _ = cache_signature
    try:
        px = yf.download(
            list(YAHOO_MARKET_TICKERS),
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        px = pd.DataFrame()

    weekly = {}
    for ticker in YAHOO_MARKET_TICKERS:
        daily = extract_market_ohlcv_frame(px, ticker)
        weekly[ticker] = build_weekly_ohlcv_from_daily(
            daily[["Open", "High", "Low", "Close", "Volume"]],
            include_partial_last_week=False,
        ) if not daily.empty else pd.DataFrame()

    fred_data = download_fred_market_data(api_key=get_fred_api_key_for_app())
    cfg = market_model_config()
    global_m2 = global_m2_weekly_series_for_market()
    market = calculate_market_model(weekly, fred_data, config=cfg, global_m2=global_m2)
    fast_history = calculate_fast_transition_risk_history(
        weekly_close(weekly.get("^VIX", pd.DataFrame())),
        weekly_close(weekly.get("DX-Y.NYB", pd.DataFrame())),
        cfg,
    )
    if not fast_history.empty:
        latest_fast = fast_history.dropna(subset=["Fast_Transition_Risk"]).tail(1)
        if not latest_fast.empty:
            market["Fast_Risk_Direction_4W"] = latest_fast["Fast_Risk_Direction_4W"].iloc[0]
            market["Fast_Risk_Direction_4W_State"] = latest_fast["Fast_Risk_Direction_4W_State"].iloc[0]
    return enrich_market_snapshot_with_global_liquidity(market)


@st.cache_data(show_spinner=True, ttl=SLOW_REFRESH_SECONDS)
def load_market_transition_history(cache_signature: str) -> pd.DataFrame:
    _ = cache_signature
    try:
        px = yf.download(
            list(YAHOO_MARKET_TICKERS),
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        px = pd.DataFrame()

    weekly = {}
    for ticker in YAHOO_MARKET_TICKERS:
        daily = extract_market_ohlcv_frame(px, ticker)
        weekly[ticker] = build_weekly_ohlcv_from_daily(
            daily[["Open", "High", "Low", "Close", "Volume"]],
            include_partial_last_week=False,
        ) if not daily.empty else pd.DataFrame()

    cfg = market_model_config()
    fred_data = download_fred_market_data(api_key=get_fred_api_key_for_app())
    global_m2 = global_m2_weekly_series_for_market()
    fast = calculate_fast_transition_risk_history(
        weekly_close(weekly.get("^VIX", pd.DataFrame())),
        weekly_close(weekly.get("DX-Y.NYB", pd.DataFrame())),
        cfg,
    )
    macro = calculate_macro_transition_risk_history(
        weekly_close(weekly.get("DX-Y.NYB", pd.DataFrame())),
        fred_data,
        cfg,
        global_m2=global_m2,
    )
    credit = calculate_credit_stress_confirmation_history(fred_data, cfg)
    confirmations = calculate_confirmations_history(weekly, fred_data, cfg)
    if fast.empty and macro.empty and confirmations.empty and credit.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Fast_Transition_Risk",
                "Fast_Transition_State",
                "Macro_Transition_Risk",
                "Macro_Transition_State",
                "Fast_Risk_Direction_4W",
                "Fast_Risk_Direction_4W_State",
                "Market_Regime",
                "Global_Liquidity_Backdrop",
                "Global_Liquidity_Score",
                "Global_Liquidity_Direction_13W",
                "Global_Liquidity_Direction_13W_State",
                "Long_Liquidity_Cycle",
                "Credit_State",
                "Credit_Risk",
                "Credit_Level_State",
                "Macro_DXY_Risk",
                "US2Y_Risk",
                "Global_M2_Bull_Score_26W",
                "Global_M2_Risk_26W",
                "Negative_Confirmation_Count",
                "Overall_Transition_Status",
                "Final_Market_State",
                "AAII_Bearish_3Y_Percentile",
                "VIX_AssetManager_NetPctOI",
                "VIX_AssetManager_NetPctOI_3Y_Percentile",
                "PositioningRisk",
                "PositioningState",
                "LiquidityWarning",
                "CreditWarning",
                "FastWarning",
                "MacroWarning",
                "TailRiskFlag",
                "TailRiskReason",
                "PositioningModel_Version",
                "TailRiskModel_Version",
            ]
        )

    history = pd.merge(
        fast[["Date", "Fast_Transition_Risk", "Fast_Transition_State", "Fast_Risk_Direction_4W", "Fast_Risk_Direction_4W_State"]] if not fast.empty else pd.DataFrame(columns=["Date"]),
        macro[
            [
                "Date",
                "Macro_Transition_Risk",
                "Macro_Transition_State",
                "Global_M2_26W",
                "Global_M2_Bull_Score_26W",
                "Global_M2_Risk_26W",
                "Macro_DXY_Risk",
                "US2Y_Risk",
            ]
        ] if not macro.empty else pd.DataFrame(columns=["Date"]),
        on="Date",
        how="outer",
    ).sort_values("Date")
    history = pd.merge(
        history,
        credit[["Date", "Credit_Risk", "Credit_State", "HY_OAS", "HY_OAS_Change_13W", "HY_Level_Percentile", "Credit_Level_State"]] if not credit.empty else pd.DataFrame(columns=["Date"]),
        on="Date",
        how="outer",
    ).sort_values("Date")
    history = pd.merge(
        history,
        confirmations[["Date", "Negative_Confirmation_Count"]] if not confirmations.empty else pd.DataFrame(columns=["Date", "Negative_Confirmation_Count"]),
        on="Date",
        how="outer",
    ).sort_values("Date")
    history["Date"] = pd.to_datetime(history["Date"])
    structural = _prepare_spy_weekly_regime_frame()
    if not structural.empty:
        structural_slice = structural[["Date", "Market_Regime"]].copy()
        structural_slice["Date"] = pd.to_datetime(structural_slice["Date"], errors="coerce")
        history = pd.merge(history, structural_slice.dropna(subset=["Date"]), on="Date", how="left").sort_values("Date")
        history["Market_Regime"] = history["Market_Regime"].ffill()
    try:
        _, monthly, weekly_liquidity = read_global_liquidity()
        liquidity = _build_global_liquidity_regime_frame(
            _liquidity_prepare_dates(monthly),
            _liquidity_prepare_dates(weekly_liquidity),
        )
    except Exception:
        liquidity = pd.DataFrame()
    if not liquidity.empty:
        liquidity_slice = liquidity[
            [
                "date",
                "global_liquidity_score",
                "direction_13w",
                "direction_13w_state",
                "long_cycle_phase",
            ]
        ].copy()
        liquidity_slice["Date"] = pd.to_datetime(liquidity_slice["date"], errors="coerce")
        liquidity_slice = liquidity_slice.rename(
            columns={
                "global_liquidity_score": "Global_Liquidity_Score",
                "direction_13w": "Global_Liquidity_Direction_13W",
                "direction_13w_state": "Global_Liquidity_Direction_13W_State",
                "long_cycle_phase": "Long_Liquidity_Cycle",
            }
        ).drop(columns=["date"], errors="ignore")
        history = pd.merge(history, liquidity_slice.dropna(subset=["Date"]), on="Date", how="left").sort_values("Date")
        for col in [
            "Global_Liquidity_Score",
            "Global_Liquidity_Direction_13W",
            "Global_Liquidity_Direction_13W_State",
            "Long_Liquidity_Cycle",
        ]:
            history[col] = history[col].ffill()
    history["Global_Liquidity_Backdrop"] = history.apply(
        lambda row: classify_global_liquidity_backdrop(
            row.get("Global_Liquidity_Score"),
            row.get("Global_Liquidity_Direction_13W"),
            row.get("Global_Liquidity_Direction_13W_State"),
        ),
        axis=1,
    )
    history["Overall_Transition_Status"] = history.apply(
        lambda row: calculate_overall_transition_status(
            row.get("Fast_Transition_Risk"),
            row.get("Macro_Transition_Risk"),
            row.get("Negative_Confirmation_Count"),
            structural_regime=row.get("Market_Regime", "UNKNOWN"),
            global_liquidity_backdrop=row.get("Global_Liquidity_Backdrop"),
            global_liquidity_score=row.get("Global_Liquidity_Score"),
            global_liquidity_direction_13w=row.get("Global_Liquidity_Direction_13W"),
            global_liquidity_direction_state=row.get("Global_Liquidity_Direction_13W_State"),
            credit_state=row.get("Credit_State", ""),
        ),
        axis=1,
    )
    history["Final_Market_State"] = history["Overall_Transition_Status"]
    positioning_history = load_positioning_history_for_market_regime()
    history = calculate_tail_risk_history(history, positioning_history)
    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    return history.loc[history["Date"] <= today_utc].reset_index(drop=True)


def prepare_candle_data(ohlcv: pd.DataFrame, period_mode: str, max_candles: int) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "SMA50", "SMA200", "BB_Mid", "BB_Upper", "BB_Lower", "Up"]
        )

    d = ohlcv.copy().sort_index()

    if period_mode == "Weekly":
        d = d.resample("W-FRI").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    elif period_mode in {"Monthly", "Full history"}:
        d = d.resample("ME").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    else:  # Daily / Full history
        d = d.copy()

    if d.empty:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "SMA50", "SMA200", "BB_Mid", "BB_Upper", "BB_Lower", "Up"]
        )

    # Compute MAs on full history first (TradingView-like), then slice visible bars.
    d["SMA50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["SMA200"] = d["Close"].rolling(200, min_periods=200).mean()
    bb_std = d["Close"].rolling(50, min_periods=50).std()
    d["BB_Mid"] = d["SMA50"]
    d["BB_Upper"] = d["BB_Mid"] + (2.0 * bb_std)
    d["BB_Lower"] = d["BB_Mid"] - (2.0 * bb_std)
    prev_close = d["Close"].shift(1)
    d["Up"] = np.where(d["Close"] >= prev_close, "up", "down")
    if len(d) > 0:
        d.iloc[0, d.columns.get_loc("Up")] = "up"

    if max_candles > 0:
        d = d.tail(max_candles)

    d = d.reset_index().rename(columns={"index": "Date"})
    return d


def render_ticker_candle_tile(ticker: str, subgroup: str, ohlcv: pd.DataFrame, period_mode: str) -> None:
    max_candles = 0 if period_mode == "Full history" else 150
    d = prepare_candle_data(ohlcv, period_mode, max_candles=max_candles)
    if d.empty or len(d) < 20:
        st.info(f"{ticker}: no chart data")
        return

    h1, h2 = st.columns([10, 1])
    with h1:
        st.markdown(f"### {ticker} ({subgroup})")
    with h2:
        if st.button("⤢", key=f"fs_{ticker}", help="Fullscreen", type="tertiary"):
            st.session_state["graphs_focus_ticker"] = ticker
            st.rerun()

    close = d["Close"]
    d = d.copy()
    d["RSI"] = RSIIndicator(close=close, window=14).rsi()
    d["MACD_HIST"] = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    d["ROC"] = ROCIndicator(close=close, window=12).roc()
    d["PPO200"] = np.where(d["SMA200"] > 0, (d["Close"] / d["SMA200"] - 1.0) * 100.0, np.nan)
    d["MACD_POS"] = np.where(d["MACD_HIST"] >= 0, "pos", "neg")

    n = len(d)
    if n > 1200:
        candle_size = 0.7
    elif n > 700:
        candle_size = 1.0
    elif n > 350:
        candle_size = 1.6
    else:
        candle_size = 2.2

    price_vals = pd.concat([d["Low"], d["High"], d["SMA50"], d["SMA200"], d["BB_Upper"], d["BB_Lower"]], axis=1).stack().dropna()
    if price_vals.empty:
        y_domain = None
    else:
        ymin = float(price_vals.min())
        ymax = float(price_vals.max())
        span = ymax - ymin
        pad = max(span * 0.06, max(abs(ymax), 1.0) * 0.01)
        y_domain = [ymin - pad, ymax + pad]

    price_scale = alt.Scale(zero=False, domain=y_domain) if y_domain is not None else alt.Scale(zero=False)

    base = alt.Chart(d).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=8))
    )

    wick = base.mark_rule().encode(
        y=alt.Y("Low:Q", axis=alt.Axis(title=None, labelFontSize=8), scale=price_scale),
        y2="High:Q",
        color=alt.Color("Up:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )

    body = base.mark_bar(size=candle_size).encode(
        y=alt.Y("Open:Q", scale=price_scale),
        y2="Close:Q",
        color=alt.Color("Up:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )

    bb_band = base.mark_area(opacity=0.08, color="#94a3b8").encode(
        y=alt.Y("BB_Lower:Q", scale=price_scale),
        y2="BB_Upper:Q",
    )
    bb_upper = base.mark_line(color="#ef4444", strokeWidth=1.0, opacity=0.9).encode(y=alt.Y("BB_Upper:Q", scale=price_scale))
    bb_lower = base.mark_line(color="#10b981", strokeWidth=1.0, opacity=0.9).encode(y=alt.Y("BB_Lower:Q", scale=price_scale))
    sma50 = base.mark_line(color="#f59e0b", strokeWidth=1.4).encode(y=alt.Y("SMA50:Q", scale=price_scale))
    sma200 = base.mark_line(color="#60a5fa", strokeWidth=1.4).encode(y=alt.Y("SMA200:Q", scale=price_scale))

    price_chart = (bb_band + wick + body + bb_upper + bb_lower + sma50 + sma200).properties(height=170)

    rsi_base = alt.Chart(d).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, labels=False, ticks=False))
    )
    rsi_line = rsi_base.mark_line(color="#8b5cf6", strokeWidth=1.2).encode(
        y=alt.Y("RSI:Q", axis=alt.Axis(title="RSI", labelFontSize=8, titleFontSize=8))
    )
    rsi_70 = alt.Chart(d).mark_rule(color="#ef4444", strokeDash=[4, 3]).encode(y=alt.datum(70))
    rsi_30 = alt.Chart(d).mark_rule(color="#22c55e", strokeDash=[4, 3]).encode(y=alt.datum(30))
    rsi_chart = (rsi_30 + rsi_70 + rsi_line).properties(height=58)

    macd_base = alt.Chart(d).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, labels=False, ticks=False))
    )
    macd_hist_chart = macd_base.mark_bar(size=2).encode(
        y=alt.Y("MACD_HIST:Q", axis=alt.Axis(title="MACD", labelFontSize=8, titleFontSize=8)),
        color=alt.Color("MACD_POS:N", scale=alt.Scale(domain=["pos", "neg"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    macd_zero = alt.Chart(d).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    macd_chart = (macd_zero + macd_hist_chart).properties(height=58)

    roc_base = alt.Chart(d).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=7))
    )
    roc_line = roc_base.mark_line(color="#38bdf8", strokeWidth=1.2).encode(
        y=alt.Y("ROC:Q", axis=alt.Axis(title="ROC", labelFontSize=8, titleFontSize=8))
    )
    roc_zero = alt.Chart(d).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    roc_chart = (roc_zero + roc_line).properties(height=58)

    ppo_base = alt.Chart(d).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=7))
    )
    ppo_line = ppo_base.mark_line(color="#14b8a6", strokeWidth=1.2).encode(
        y=alt.Y("PPO200:Q", axis=alt.Axis(title="PPO200", labelFontSize=8, titleFontSize=8))
    )
    ppo_zero = alt.Chart(d).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    ppo_chart = (ppo_zero + ppo_line).properties(height=58)

    chart = alt.vconcat(price_chart, rsi_chart, macd_chart, roc_chart, ppo_chart, spacing=4).resolve_scale(x="shared")
    st.altair_chart(chart, use_container_width=True)


def render_zoom_chart_with_indicators(ticker: str, subgroup: str, ohlcv: pd.DataFrame, period_mode: str) -> None:
    max_candles = 0 if period_mode == "Full history" else 150
    d = prepare_candle_data(ohlcv, period_mode, max_candles=max_candles)
    if d.empty or len(d) < 20:
        st.info(f"{ticker}: not enough data for zoom chart.")
        return

    close = d["Close"]
    rsi = RSIIndicator(close=close, window=14).rsi()
    macd_hist = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    roc = ROCIndicator(close=close, window=12).roc()

    ind = d.copy()
    ind["RSI"] = rsi
    ind["MACD_HIST"] = macd_hist
    ind["ROC"] = roc
    ind["PPO200"] = np.where(ind["SMA200"] > 0, (ind["Close"] / ind["SMA200"] - 1.0) * 100.0, np.nan)
    ind["Up"] = np.where(ind["Close"] >= ind["Close"].shift(1), "up", "down")
    if len(ind) > 0:
        ind.iloc[0, ind.columns.get_loc("Up")] = "up"
    ind["MACD_POS"] = np.where(ind["MACD_HIST"] >= 0, "pos", "neg")

    price_vals = pd.concat(
        [ind["Low"], ind["High"], ind["SMA50"], ind["SMA200"], ind["BB_Upper"], ind["BB_Lower"]],
        axis=1,
    ).stack().dropna()
    if price_vals.empty:
        y_domain = None
    else:
        ymin = float(price_vals.min())
        ymax = float(price_vals.max())
        span = ymax - ymin
        pad = max(span * 0.06, max(abs(ymax), 1.0) * 0.01)
        y_domain = [ymin - pad, ymax + pad]

    price_scale = alt.Scale(zero=False, domain=y_domain) if y_domain is not None else alt.Scale(zero=False)

    base_price = alt.Chart(ind).encode(
        x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=9))
    )
    wick = base_price.mark_rule().encode(
        y=alt.Y("Low:Q", axis=alt.Axis(title=None), scale=price_scale),
        y2="High:Q",
        color=alt.Color("Up:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    body = base_price.mark_bar(size=4).encode(
        y=alt.Y("Open:Q", scale=price_scale),
        y2="Close:Q",
        color=alt.Color("Up:N", scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    bb_band = base_price.mark_area(opacity=0.08, color="#94a3b8").encode(
        y=alt.Y("BB_Lower:Q", scale=price_scale),
        y2="BB_Upper:Q",
    )
    bb_upper = base_price.mark_line(color="#ef4444", strokeWidth=1.0, opacity=0.9).encode(y=alt.Y("BB_Upper:Q", scale=price_scale))
    bb_lower = base_price.mark_line(color="#10b981", strokeWidth=1.0, opacity=0.9).encode(y=alt.Y("BB_Lower:Q", scale=price_scale))
    sma50 = base_price.mark_line(color="#f59e0b", strokeWidth=1.5).encode(y=alt.Y("SMA50:Q", scale=price_scale))
    sma200 = base_price.mark_line(color="#60a5fa", strokeWidth=1.5).encode(y=alt.Y("SMA200:Q", scale=price_scale))
    price_chart = (bb_band + wick + body + bb_upper + bb_lower + sma50 + sma200).properties(
        height=210, title=f"{ticker} ({subgroup}) - {period_mode}"
    )

    rsi_base = alt.Chart(ind).encode(x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=9)))
    rsi_line = rsi_base.mark_line(color="#8b5cf6", strokeWidth=1.5).encode(y=alt.Y("RSI:Q", axis=alt.Axis(title="RSI")))
    rsi_70 = alt.Chart(ind).mark_rule(color="#ef4444", strokeDash=[5, 4]).encode(y=alt.datum(70))
    rsi_30 = alt.Chart(ind).mark_rule(color="#22c55e", strokeDash=[5, 4]).encode(y=alt.datum(30))
    rsi_chart = (rsi_30 + rsi_70 + rsi_line).properties(height=65)

    macd_base = alt.Chart(ind).encode(x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=9)))
    macd_hist_chart = macd_base.mark_bar(size=4).encode(
        y=alt.Y("MACD_HIST:Q", axis=alt.Axis(title="MACD Hist")),
        color=alt.Color("MACD_POS:N", scale=alt.Scale(domain=["pos", "neg"], range=["#22c55e", "#ef4444"]), legend=None),
    )
    macd_zero = alt.Chart(ind).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    macd_chart = (macd_zero + macd_hist_chart).properties(height=65)

    roc_base = alt.Chart(ind).encode(x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=9)))
    roc_line = roc_base.mark_line(color="#38bdf8", strokeWidth=1.5).encode(y=alt.Y("ROC:Q", axis=alt.Axis(title="ROC")))
    roc_zero = alt.Chart(ind).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    roc_chart = (roc_zero + roc_line).properties(height=65)

    ppo_base = alt.Chart(ind).encode(x=alt.X("Date:T", axis=alt.Axis(title=None, format="%b %y", labelFontSize=9)))
    ppo_line = ppo_base.mark_line(color="#14b8a6", strokeWidth=1.5).encode(
        y=alt.Y("PPO200:Q", axis=alt.Axis(title="PPO200"))
    )
    ppo_zero = alt.Chart(ind).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    ppo_chart = (ppo_zero + ppo_line).properties(height=65)

    st.altair_chart(
        alt.vconcat(price_chart, rsi_chart, macd_chart, roc_chart, ppo_chart, spacing=2).resolve_scale(x="shared"),
        use_container_width=True,
    )


def render_graphs_tab(filtered_df: pd.DataFrame, selected_universe: dict, selected_universe_name: str) -> None:
    g1, g2 = st.columns([1, 8])
    with g1:
        period_mode = st.selectbox("Period", GRAPH_PERIOD_OPTIONS, index=1, key="graphs_period")
    with g2:
        st.caption("Type: Candle | Green = close above previous close, Red = close below previous close | SMA50 + SMA200")

    if filtered_df.empty:
        st.info("No rows to chart for current filters.")
        return

    tickers = filtered_df["Ticker"].dropna().astype(str).drop_duplicates().tolist()
    subgroup_map = (
        filtered_df.dropna(subset=["Ticker"])
        .drop_duplicates(subset=["Ticker"])
        .set_index("Ticker")["Subgroup"]
        .to_dict()
    )

    ohlcv_map = load_ohlcv_history(tickers, f"{selected_universe_name}:{str(selected_universe)}")

    focus_ticker = st.session_state.get("graphs_focus_ticker")
    if focus_ticker and focus_ticker in ohlcv_map:
        focus_subgroup = str(subgroup_map.get(focus_ticker, ""))
        f1, f2 = st.columns([12, 1])
        with f1:
            st.markdown(f"## {focus_ticker} ({focus_subgroup}) - Focus View")
        with f2:
            if st.button("⤡", key="close_focus_chart", help="Close fullscreen", type="tertiary"):
                st.session_state.pop("graphs_focus_ticker", None)
                st.rerun()

        render_zoom_chart_with_indicators(
            ticker=focus_ticker,
            subgroup=focus_subgroup,
            ohlcv=ohlcv_map.get(focus_ticker, pd.DataFrame()),
            period_mode=period_mode,
        )
        return

    cols = st.columns(3)
    for i, ticker in enumerate(tickers):
        with cols[i % 3]:
            render_ticker_candle_tile(
                ticker=ticker,
                subgroup=str(subgroup_map.get(ticker, "")),
                ohlcv=ohlcv_map.get(ticker, pd.DataFrame()),
                period_mode=period_mode,
            )


def _build_chart_frame(df: pd.DataFrame) -> pd.DataFrame:
    chart_df = df.copy().reset_index(drop=True)
    chart_df["TickerAxis"] = chart_df["Ticker"].astype(str) + " (" + chart_df["Subgroup"].astype(str) + ")"
    chart_df["Order"] = np.arange(len(chart_df))
    for col in NUMERIC_COLUMNS:
        if col in chart_df.columns:
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
    return chart_df


def _render_bar_chart(
    chart_df: pd.DataFrame,
    metric: str,
    title: str,
    height: int = 220,
    dot_metric: str = None,
) -> None:
    cols = ["TickerAxis", "Ticker", "Group", "Subgroup", metric]
    if dot_metric:
        cols.append(dot_metric)
    d = chart_df[cols].dropna(subset=[metric]).copy()
    if d.empty:
        st.info(f"No data for {title}.")
        return

    x_sort = d["TickerAxis"].tolist()

    base = alt.Chart(d).encode(
        x=alt.X(
            "TickerAxis:N",
            sort=x_sort,
            axis=alt.Axis(title=None, labelAngle=-90, labelFontSize=9, labelLimit=180),
        ),
        y=alt.Y(f"{metric}:Q", axis=alt.Axis(title="%", format=".0f")),
        tooltip=[
            alt.Tooltip("Ticker:N"),
            alt.Tooltip("Group:N"),
            alt.Tooltip("Subgroup:N"),
            alt.Tooltip(f"{metric}:Q", format=".2f"),
        ],
    )

    bars = base.mark_bar(color="#0b7285", size=14)
    baseline = alt.Chart(d).mark_rule(color="#6b7280").encode(y=alt.datum(0))
    labels = (
        base.transform_calculate(label=f"format(datum['{metric}'], '.0f') + '%'")
        .mark_text(fontSize=9, dy=-6, color="#9ca3af")
        .encode(text="label:N")
    )
    layers = [baseline, bars, labels]
    if dot_metric and dot_metric in d.columns:
        dots = (
            alt.Chart(d.dropna(subset=[dot_metric]))
            .mark_circle(color="#ef4444", size=58)
            .encode(
                x=alt.X("TickerAxis:N", sort=x_sort),
                y=alt.Y(f"{dot_metric}:Q"),
                tooltip=[
                    alt.Tooltip("Ticker:N"),
                    alt.Tooltip("Group:N"),
                    alt.Tooltip("Subgroup:N"),
                    alt.Tooltip(f"{dot_metric}:Q", title="Avg Spread 36M %", format=".2f"),
                ],
            )
        )
        layers.append(dots)

    chart = alt.layer(*layers).properties(height=height, title=title)
    st.altair_chart(chart, use_container_width=True)


def _render_rsi_chart(chart_df: pd.DataFrame, height: int = 250) -> None:
    d = chart_df[["TickerAxis", "Ticker", "Group", "Subgroup", "RSI_14"]].dropna(subset=["RSI_14"]).copy()
    if d.empty:
        st.info("No RSI data to chart.")
        return

    x_sort = d["TickerAxis"].tolist()
    points = alt.Chart(d).mark_circle(color="#0b7285", size=55).encode(
        x=alt.X(
            "TickerAxis:N",
            sort=x_sort,
            axis=alt.Axis(title=None, labelAngle=-90, labelFontSize=9, labelLimit=180),
        ),
        y=alt.Y("RSI_14:Q", axis=alt.Axis(title="RSI", format=".0f")),
        tooltip=[
            alt.Tooltip("Ticker:N"),
            alt.Tooltip("Group:N"),
            alt.Tooltip("Subgroup:N"),
            alt.Tooltip("RSI_14:Q", format=".2f"),
        ],
    )
    labels = points.transform_calculate(label="format(datum.RSI_14, '.0f')").mark_text(
        dx=10, dy=0, fontSize=9, color="#9ca3af"
    ).encode(text="label:N")

    line_70 = alt.Chart(d).mark_rule(color="#dc2626", size=2).encode(y=alt.datum(70))
    line_30 = alt.Chart(d).mark_rule(color="#22c55e", size=2).encode(y=alt.datum(30))

    chart = (line_30 + line_70 + points + labels).properties(height=height, title="RSI")
    st.altair_chart(chart, use_container_width=True)


PERFORMANCE_BUBBLE_X_OPTIONS = {
    "Performance 1W": "Perf_1W_%",
    "Performance 1M": "Perf_1M_%",
    "Performance 3M": "Perf_3M_%",
}


def _prepare_performance_sma200w_bubble_frame(df: pd.DataFrame, x_metric: str) -> pd.DataFrame:
    required_cols = ["Ticker", x_metric, "SMA200W_Distance_Percentile", "Avg_Forward_Return_6M_%"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return pd.DataFrame()

    d = df[required_cols].copy()
    d[x_metric] = pd.to_numeric(d[x_metric], errors="coerce")
    d["SMA200W_Distance_Percentile"] = pd.to_numeric(d["SMA200W_Distance_Percentile"], errors="coerce")
    d["Avg_Forward_Return_6M_%"] = pd.to_numeric(d["Avg_Forward_Return_6M_%"], errors="coerce")
    d = d.dropna(subset=[x_metric, "SMA200W_Distance_Percentile", "Avg_Forward_Return_6M_%"])
    d = d[d[x_metric] >= 0.0].copy()
    if d.empty:
        return d

    d["Selected_Performance"] = d[x_metric]
    d["SMA200W_Percentile"] = d["SMA200W_Distance_Percentile"].clip(lower=0.0, upper=100.0)
    d["Avg_Forward_Return_6M"] = d["Avg_Forward_Return_6M_%"]
    d["Forward_Return_Sign"] = np.where(d["Avg_Forward_Return_6M"] >= 0.0, "Positive", "Negative")
    d["Bubble_Size_Value"] = d["Avg_Forward_Return_6M"].abs()
    size_cap = float(d["Bubble_Size_Value"].quantile(0.95))
    if not np.isfinite(size_cap) or size_cap <= 0.0:
        size_cap = 1.0
    d["Bubble_Size"] = d["Bubble_Size_Value"].clip(lower=0.0, upper=size_cap)
    return d


def _render_performance_sma200w_bubble_chart(chart_df: pd.DataFrame) -> None:
    st.subheader("Performance vs SMA200W Percentile")
    x_label = st.radio(
        "X-axis metric",
        options=list(PERFORMANCE_BUBBLE_X_OPTIONS.keys()),
        index=1,
        horizontal=True,
        key="performance_sma200w_bubble_x_metric",
    )
    x_metric = PERFORMANCE_BUBBLE_X_OPTIONS[x_label]
    d = _prepare_performance_sma200w_bubble_frame(chart_df, x_metric)
    if d.empty:
        st.info(f"No assets with non-negative {x_label} and complete SMA200W / forward return data.")
        return

    x_max = float(d["Selected_Performance"].max())
    x_domain_max = max(1.0, x_max * 1.12)

    base = alt.Chart(d).encode(
        x=alt.X(
            "Selected_Performance:Q",
            scale=alt.Scale(domain=[0.0, x_domain_max], zero=True),
            axis=alt.Axis(title=f"{x_label} %", format=".0f"),
        ),
        y=alt.Y(
            "SMA200W_Percentile:Q",
            scale=alt.Scale(domain=[100.0, 0.0]),
            axis=alt.Axis(title="SMA200W Percentile", format=".0f"),
        ),
        tooltip=[
            alt.Tooltip("Ticker:N", title="Asset / Ticker"),
            alt.Tooltip("Selected_Performance:Q", title=f"{x_label} %", format="+.2f"),
            alt.Tooltip("SMA200W_Percentile:Q", title="SMA200W Percentile", format=".0f"),
            alt.Tooltip("Avg_Forward_Return_6M:Q", title="Avg Forward Return 6M %", format="+.2f"),
        ],
    )

    bubbles = base.mark_circle(opacity=0.72, stroke="#111827", strokeWidth=0.6).encode(
        color=alt.Color(
            "Forward_Return_Sign:N",
            scale=alt.Scale(domain=["Positive", "Negative"], range=["#22c55e", "#ef4444"]),
            legend=alt.Legend(title="Bubble color", labelExpr="datum.label === 'Positive' ? 'Positive Avg Forward Return 6M' : 'Negative Avg Forward Return 6M'"),
        ),
        size=alt.Size(
            "Bubble_Size:Q",
            scale=alt.Scale(range=[80, 850], zero=True),
            legend=alt.Legend(title="Bubble size: |Avg Forward Return 6M|"),
        ),
    )
    labels = base.mark_text(dx=9, dy=0, align="left", baseline="middle", fontSize=10, color="#e5e7eb").encode(
        text="Ticker:N"
    )

    chart = (bubbles + labels).properties(height=520)
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Green = positive Avg Forward Return 6M, red = negative. "
        "Larger bubble = larger absolute Avg Forward Return 6M. "
        "Assets with negative selected performance are excluded."
    )


def _prepare_spy_weekly_regime_frame() -> pd.DataFrame:
    ohlcv = download_metrics_ohlcv("SPY", period="max")
    if ohlcv.empty:
        return pd.DataFrame()

    weekly = build_weekly_ohlcv_from_daily(
        ohlcv[["Open", "High", "Low", "Close", "Volume"]],
        include_partial_last_week=False,
    )
    if weekly.empty:
        return pd.DataFrame()

    cfg = alpha_config()["market_regime"]
    weekly = weekly.sort_index().copy()
    close = pd.to_numeric(weekly["Close"], errors="coerce")
    high = pd.to_numeric(weekly["High"], errors="coerce").ffill()
    sma40 = close.rolling(int(cfg["spy_sma_weeks"]), min_periods=int(cfg["spy_sma_weeks"])).mean()
    high52w = high.rolling(52, min_periods=52).max()
    vol13w = close.pct_change().rolling(
        int(cfg["vol_window_weeks"]),
        min_periods=int(cfg["vol_window_weeks"]),
    ).std() * np.sqrt(52.0)

    def expanding_percentile_rank(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan
        current = finite[-1]
        return float((finite <= current).sum() / len(finite) * 100.0)

    vol_percentile = vol13w.expanding(min_periods=int(cfg["vol_window_weeks"])).apply(
        expanding_percentile_rank,
        raw=True,
    )

    frame = weekly.assign(
        SPY_SMA40W=sma40,
        SPY_Drawdown_52W=close / high52w - 1.0,
        SPY_Volatility_13W=vol13w,
        SPY_Volatility_Percentile=vol_percentile,
    )
    structural_bull = (close > frame["SPY_SMA40W"]) & (
        frame["SPY_Drawdown_52W"] > float(cfg["drawdown_threshold"])
    )
    high_vol = frame["SPY_Volatility_Percentile"] >= float(cfg["high_vol_percentile"])
    frame["Market_Regime"] = np.select(
        [
            structural_bull & ~high_vol,
            structural_bull & high_vol,
            ~structural_bull & ~high_vol,
            ~structural_bull & high_vol,
        ],
        ["BULL", "BULL_HIGH_VOL", "CORRECTION", "STRESS"],
        default="UNKNOWN",
    )
    frame = frame.loc["2016-01-01":"2026-12-31"].dropna(subset=["Open", "High", "Low", "Close"])
    frame = frame[frame["Market_Regime"] != "UNKNOWN"]
    if frame.empty:
        return pd.DataFrame()

    out = frame.reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out["NextDate"] = out["Date"].shift(-1)
    out.loc[out["NextDate"].isna(), "NextDate"] = out.loc[out["NextDate"].isna(), "Date"] + pd.Timedelta(days=7)
    prev_close = pd.to_numeric(out["Close"], errors="coerce").shift(1)
    out["Direction"] = np.where(pd.to_numeric(out["Close"], errors="coerce") >= prev_close, "up", "down")
    if not out.empty:
        out.loc[out.index[0], "Direction"] = "up"
    return out


def _render_spy_weekly_market_regime_chart() -> None:
    st.subheader("SPY Weekly Market Regime")
    d = _prepare_spy_weekly_regime_frame()
    if d.empty:
        st.info("No SPY weekly market regime data for 2016-2026.")
        return

    price_values = pd.concat(
        [
            pd.to_numeric(d["Low"], errors="coerce"),
            pd.to_numeric(d["High"], errors="coerce"),
            pd.to_numeric(d["SPY_SMA40W"], errors="coerce"),
        ],
        axis=1,
    ).stack().dropna()
    if price_values.empty:
        st.info("No SPY price data for the market regime chart.")
        return

    ymin = float(price_values.min())
    ymax = float(price_values.max())
    span = ymax - ymin
    pad = max(span * 0.06, max(abs(ymax), 1.0) * 0.01)
    y_domain = [ymin - pad, ymax + pad]
    d = d.copy()
    d["YMin"] = y_domain[0]
    d["YMax"] = y_domain[1]

    price_scale = alt.Scale(zero=False, domain=y_domain)
    x_axis = alt.Axis(title=None, format="%Y", labelFontSize=9)
    regime_colors = alt.Scale(
        domain=["BULL", "BULL_HIGH_VOL", "CORRECTION", "STRESS"],
        range=["#00ff66", "#b7ff4a", "#ffd400", "#ff1744"],
    )

    base = alt.Chart(d).encode(x=alt.X("Date:T", axis=x_axis))
    zones = alt.Chart(d).mark_rect(opacity=0.34).encode(
        x=alt.X("Date:T", axis=x_axis),
        x2="NextDate:T",
        y=alt.Y("YMin:Q", scale=price_scale, axis=alt.Axis(title="SPY")),
        y2="YMax:Q",
        color=alt.Color("Market_Regime:N", scale=regime_colors, legend=alt.Legend(title="Market Regime")),
        tooltip=[
            alt.Tooltip("Date:T", title="Week", format="%Y-%m-%d"),
            alt.Tooltip("Market_Regime:N", title="Regime"),
            alt.Tooltip("SPY_Volatility_Percentile:Q", title="Vol Percentile", format=".0f"),
        ],
    )
    wick = base.mark_rule().encode(
        y=alt.Y("Low:Q", scale=price_scale, axis=alt.Axis(title="SPY")),
        y2="High:Q",
        color=alt.Color(
            "Direction:N",
            scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Date:T", title="Week", format="%Y-%m-%d"),
            alt.Tooltip("Open:Q", format=".2f"),
            alt.Tooltip("High:Q", format=".2f"),
            alt.Tooltip("Low:Q", format=".2f"),
            alt.Tooltip("Close:Q", format=".2f"),
            alt.Tooltip("Market_Regime:N", title="Regime"),
        ],
    )
    body = base.mark_bar(size=3).encode(
        y=alt.Y("Open:Q", scale=price_scale),
        y2="Close:Q",
        color=alt.Color(
            "Direction:N",
            scale=alt.Scale(domain=["up", "down"], range=["#22c55e", "#ef4444"]),
            legend=None,
        ),
    )
    sma40 = base.mark_line(color="#e5e7eb", strokeWidth=1.1, opacity=0.8).encode(
        y=alt.Y("SPY_SMA40W:Q", scale=price_scale),
        tooltip=[
            alt.Tooltip("Date:T", title="Week", format="%Y-%m-%d"),
            alt.Tooltip("SPY_SMA40W:Q", title="SMA40W", format=".2f"),
        ],
    )
    price_chart = (zones + wick + body + sma40).properties(height=360, title="SPY Weekly Bars with Market Regime Zones")

    transition_history = load_market_transition_history("spy-weekly-transition-history")
    transition_charts = []
    if not transition_history.empty:
        transition_history = transition_history.copy()
        transition_history["NextDate"] = transition_history["Date"].shift(-1)
        transition_history.loc[transition_history["NextDate"].isna(), "NextDate"] = (
            transition_history.loc[transition_history["NextDate"].isna(), "Date"] + pd.Timedelta(days=7)
        )
        risk_color = alt.Scale(
            domain=[0, 50, 100],
            range=["#00ff66", "#ffd400", "#ff1744"],
            clamp=True,
        )
        for metric, state_col, title in [
            ("Fast_Transition_Risk", "Fast_Transition_State", "Fast Transition Risk"),
            ("Macro_Transition_Risk", "Macro_Transition_State", "Macro Transition Risk"),
        ]:
            risk_data = transition_history.dropna(subset=[metric]).copy()
            if risk_data.empty:
                continue
            risk_bar = (
                alt.Chart(risk_data)
                .mark_bar(opacity=0.92)
                .encode(
                    x=alt.X("Date:T", axis=alt.Axis(title=None, labels=False, ticks=False)),
                    x2="NextDate:T",
                    y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title=title, format=".0f")),
                    color=alt.Color(f"{metric}:Q", scale=risk_color, legend=None),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Week", format="%Y-%m-%d"),
                        alt.Tooltip(f"{metric}:Q", title=title, format=".0f"),
                        alt.Tooltip(f"{state_col}:N", title="State"),
                    ],
                )
                .properties(height=105, title=title)
            )
            levels = [10, 20, 40, 60] if metric == "Fast_Transition_Risk" else [20, 40, 60, 80]
            threshold_layers = [
                alt.Chart(risk_data).mark_rule(color=color, strokeDash=[4, 3], opacity=0.85).encode(y=alt.datum(level))
                for level, color in zip(levels, ["#22c55e", "#facc15", "#f97316", "#ef4444"])
            ]
            transition_charts.append(risk_bar + threshold_layers[0] + threshold_layers[1] + threshold_layers[2] + threshold_layers[3])

        status_data = transition_history.dropna(subset=["Overall_Transition_Status"]).copy()
        if not status_data.empty:
            status_data["StatusBand"] = "Final Market State"
            status_color = alt.Scale(
                domain=[
                    "BULL",
                    "RECOVERY",
                    "BULL_LIQUIDITY_WARNING",
                    "BULL_WITH_WARNING",
                    "DETERIORATING",
                    "CORRECTION",
                    "STRESS",
                    "DATA_INCOMPLETE",
                ],
                range=[
                    "#00ff66",
                    "#8cff3d",
                    "#ffd400",
                    "#f59e0b",
                    "#fb923c",
                    "#ff1744",
                    "#be123c",
                    "#64748b",
                ],
            )
            status_band = (
                alt.Chart(status_data)
                .mark_rect(opacity=0.92)
                .encode(
                    x=alt.X("Date:T", axis=alt.Axis(title=None, labels=False, ticks=False)),
                    x2="NextDate:T",
                    y=alt.Y("StatusBand:N", axis=alt.Axis(title=None, labelAngle=0)),
                    color=alt.Color(
                        "Overall_Transition_Status:N",
                        scale=status_color,
                        legend=alt.Legend(title="Overall Status"),
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Week", format="%Y-%m-%d"),
                        alt.Tooltip("Overall_Transition_Status:N", title="Overall Status"),
                        alt.Tooltip("Fast_Transition_Risk:Q", title="Fast Risk", format=".0f"),
                        alt.Tooltip("Macro_Transition_Risk:Q", title="Macro Risk", format=".0f"),
                        alt.Tooltip("Global_Liquidity_Backdrop:N", title="Liquidity Backdrop"),
                        alt.Tooltip("Negative_Confirmation_Count:Q", title="Negative Confirmations", format=".1f"),
                    ],
                )
                .properties(height=54, title="Final Market State")
            )
            transition_charts.append(status_band)

    chart = alt.vconcat(price_chart, *transition_charts, spacing=8).resolve_scale(x="shared")
    st.altair_chart(chart, use_container_width=True)


def _render_tail_risk_validation_report() -> None:
    with st.expander("Tail Risk Validation", expanded=False):
        transition_history = load_market_transition_history("spy-weekly-transition-history")
        spy = _prepare_spy_weekly_regime_frame()
        if transition_history.empty or spy.empty or "TailRiskFlag" not in transition_history.columns:
            st.info("Tail risk validation data is not available yet.")
            return

        corrections = _build_spy_correction_validation(spy, transition_history)
        distribution = _build_tail_risk_distribution_validation(corrections, transition_history)
        incidence = _build_tail_risk_forward_incidence(spy, transition_history)

        st.markdown("#### SPY weekly-close corrections greater than 15%")
        if corrections.empty:
            st.info("No weekly-close SPY corrections greater than 15% were detected in the available sample.")
        else:
            st.dataframe(corrections, use_container_width=True, hide_index=True)

        st.markdown("#### TailRiskFlag distribution")
        if distribution.empty:
            st.info("No TailRiskFlag distribution is available.")
        else:
            st.dataframe(distribution, use_container_width=True, hide_index=True)

        st.markdown("#### Forward drawdown incidence by TailRiskFlag")
        if incidence.empty:
            st.info("Forward incidence cannot be calculated for the available sample.")
        else:
            st.dataframe(incidence, use_container_width=True, hide_index=True)


def _build_spy_correction_validation(spy: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    price = spy[["Date", "Close"]].dropna().copy()
    if price.empty:
        return pd.DataFrame()
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price = price.dropna(subset=["Date"]).sort_values("Date")
    close = pd.to_numeric(price["Close"], errors="coerce")
    price = price.loc[close.notna()].copy()
    price["Close"] = close.loc[price.index].astype(float)
    price = price.loc[(price["Date"] >= "2016-01-01") & (price["Date"] <= "2026-12-31")]
    if len(price) < 2:
        return pd.DataFrame()

    corrections = []
    peak_date = price["Date"].iloc[0]
    peak_value = float(price["Close"].iloc[0])
    trough_date = peak_date
    trough_value = peak_value
    in_correction = False

    for row in price.iloc[1:].itertuples(index=False):
        current_date = row.Date
        current_value = float(row.Close)
        drawdown = current_value / peak_value - 1.0 if peak_value else np.nan
        if not in_correction:
            if current_value >= peak_value:
                peak_date = current_date
                peak_value = current_value
                trough_date = current_date
                trough_value = current_value
            elif np.isfinite(drawdown) and drawdown <= -0.15:
                in_correction = True
                trough_date = current_date
                trough_value = current_value
        else:
            if current_value < trough_value:
                trough_date = current_date
                trough_value = current_value
            if current_value >= peak_value:
                corrections.append(
                    {
                        "Peak Date": peak_date,
                        "Trough Date": trough_date,
                        "Max Drawdown %": (trough_value / peak_value - 1.0) * 100.0,
                    }
                )
                peak_date = current_date
                peak_value = current_value
                trough_date = current_date
                trough_value = current_value
                in_correction = False

    if in_correction:
        corrections.append(
            {
                "Peak Date": peak_date,
                "Trough Date": trough_date,
                "Max Drawdown %": (trough_value / peak_value - 1.0) * 100.0,
            }
        )
    if not corrections:
        return pd.DataFrame()

    lookup_columns = [
        "Date",
        "PositioningRisk",
        "PositioningState",
        "Global_Liquidity_Score",
        "Global_Liquidity_Direction_13W",
        "LiquidityWarning",
        "Credit_Risk",
        "Fast_Transition_Risk",
        "Macro_Transition_Risk",
        "TailRiskFlag",
        "TailRiskReason",
    ]
    lookup = history[[col for col in lookup_columns if col in history.columns]].copy()
    lookup["Date"] = pd.to_datetime(lookup["Date"], errors="coerce")
    lookup = lookup.dropna(subset=["Date"]).sort_values("Date")
    result = pd.DataFrame(corrections)
    points = pd.DataFrame({"Date": pd.to_datetime(result["Peak Date"]) - pd.Timedelta(days=7)})
    sampled = pd.merge_asof(points.sort_values("Date"), lookup, on="Date", direction="backward")
    sampled = sampled.drop(columns=["Date"], errors="ignore").reset_index(drop=True)
    return pd.concat([result.reset_index(drop=True), sampled], axis=1)


def _build_tail_risk_distribution_validation(corrections: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "TailRiskFlag" not in history.columns:
        return pd.DataFrame()
    all_flags = history["TailRiskFlag"].dropna().astype(str)
    correction_flags = corrections.get("TailRiskFlag", pd.Series(dtype="object")).dropna().astype(str)
    flags = sorted(set(all_flags.unique()).union(set(correction_flags.unique())))
    rows = []
    for flag in flags:
        rows.append(
            {
                "TailRiskFlag": flag,
                "All Weeks Count": int((all_flags == flag).sum()),
                "All Weeks %": float((all_flags == flag).mean() * 100.0) if len(all_flags) else np.nan,
                "Correction Peaks Count": int((correction_flags == flag).sum()),
                "Correction Peaks %": float((correction_flags == flag).mean() * 100.0) if len(correction_flags) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_tail_risk_forward_incidence(spy: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if spy.empty or history.empty or "TailRiskFlag" not in history.columns:
        return pd.DataFrame()
    price = spy[["Date", "Close"]].dropna().copy()
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price["Close"] = pd.to_numeric(price["Close"], errors="coerce")
    price = price.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    lookup = history[["Date", "TailRiskFlag"]].dropna().copy()
    lookup["Date"] = pd.to_datetime(lookup["Date"], errors="coerce")
    lookup = lookup.dropna(subset=["Date"]).sort_values("Date")
    aligned = pd.merge_asof(price, lookup, on="Date", direction="backward")
    rows = []
    for horizon in [4, 8, 12]:
        samples = []
        for idx, row in aligned.iloc[:-horizon].iterrows():
            flag = row.get("TailRiskFlag")
            if not isinstance(flag, str) or not flag:
                continue
            current = float(row["Close"])
            future = aligned["Close"].iloc[idx + 1 : idx + horizon + 1]
            if len(future) < horizon or current <= 0:
                continue
            samples.append(
                {
                    "Horizon": f"{horizon}W",
                    "TailRiskFlag": flag,
                    "Forward Return": future.iloc[-1] / current - 1.0,
                    "Future Min Return": future.min() / current - 1.0,
                }
            )
        sample = pd.DataFrame(samples)
        if sample.empty:
            continue
        for flag, group in sample.groupby("TailRiskFlag"):
            min_return = pd.to_numeric(group["Future Min Return"], errors="coerce")
            forward_return = pd.to_numeric(group["Forward Return"], errors="coerce")
            rows.append(
                {
                    "Horizon": f"{horizon}W",
                    "TailRiskFlag": flag,
                    "N": int(len(group)),
                    "Avg Forward Return %": float(forward_return.mean() * 100.0),
                    "Avg Future Min Return %": float(min_return.mean() * 100.0),
                    "Prob <= -5%": float((min_return <= -0.05).mean() * 100.0),
                    "Prob <= -10%": float((min_return <= -0.10).mean() * 100.0),
                    "Prob <= -15%": float((min_return <= -0.15).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def format_market_value(value, kind: str = "number") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value if value else "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    if kind == "percent":
        return f"{numeric * 100.0:.1f}%"
    if kind == "percent_points":
        return f"{numeric:.1f}%"
    if kind == "bp":
        return f"{numeric:.0f} bp"
    if kind == "score":
        return f"{numeric:.0f}"
    return f"{numeric:.2f}"


def render_market_metric(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
<div style="padding: 0.75rem 0; line-height: 1.15;">
  <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">{label}</div>
  <div style="font-size: 1.1rem; color: #f8fafc; font-weight: 800;">{value}</div>
  <div style="font-size: 0.72rem; color: #cbd5e1;">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_market_detail_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )


def render_market_formula(title: str, formula: str) -> None:
    safe_title = html.escape(title)
    safe_formula = html.escape(formula)
    st.markdown(
        f"""
<div style="margin-top: 0.35rem; margin-bottom: 1.4rem; color: #cbd5e1; font-size: 0.78rem; line-height: 1.35;">
  <div style="color: #94a3b8; font-weight: 800; margin-bottom: 0.25rem;">{safe_title}</div>
  <pre style="white-space: pre-wrap; background: #111827; border: 1px solid #293241; border-radius: 6px; padding: 0.7rem; margin: 0;">{safe_formula}</pre>
</div>
""",
        unsafe_allow_html=True,
    )


def overall_status_logic_text(market: dict) -> str:
    fast = market.get("Fast_Transition_Risk")
    macro = market.get("Macro_Transition_Risk")
    negative = market.get("Negative_Confirmation_Count")
    structural = format_market_value(market.get("Market_Regime"))
    backdrop = format_market_value(market.get("Global_Liquidity_Backdrop"))
    liquidity_score = market.get("Global_Liquidity_Score")
    liquidity_direction = market.get("Global_Liquidity_Direction_13W")
    liquidity_direction_state = format_market_value(market.get("Global_Liquidity_Direction_13W_State"))
    status = format_market_value(market.get("Final_Market_State", market.get("Overall_Transition_Status")))
    fast_text = format_market_value(fast, "score")
    macro_text = format_market_value(macro, "score")
    negative_text = format_market_value(negative)

    try:
        fast_value = float(fast)
        macro_value = float(macro)
        negative_value = float(negative)
    except Exception:
        fast_value = np.nan
        macro_value = np.nan
        negative_value = np.nan

    liquidity_warning = backdrop in {"LIQUIDITY_WARNING", "NEGATIVE", "STRONGLY_NEGATIVE"} or _safe_float(liquidity_score) < 40.0 or _safe_float(liquidity_direction) < -10.0
    credit_state = format_market_value(market.get("Credit_State"))
    credit_widening = str(market.get("Credit_State", "")).upper() in {"WIDENING", "SEVERE_WIDENING"}
    fast_warning = np.isfinite(fast_value) and fast_value >= 20.0
    macro_warning = np.isfinite(macro_value) and macro_value >= 20.0

    if structural == "STRESS":
        matched_rule = "STRESS: Structural Regime is STRESS."
    elif structural == "CORRECTION":
        matched_rule = "CORRECTION: Structural Regime is CORRECTION."
    elif not np.isfinite(fast_value) or not np.isfinite(macro_value) or not np.isfinite(negative_value):
        matched_rule = "DATA_INCOMPLETE: one or more required inputs are missing."
    elif sum([fast_warning, macro_warning, liquidity_warning]) >= 2:
        matched_rule = "DETERIORATING: at least two primary warning layers are active."
    elif (fast_warning or macro_warning) and negative_value >= 3.0:
        matched_rule = "DETERIORATING: one primary warning is amplified by 3+ negative confirmations."
    elif (fast_warning or macro_warning or liquidity_warning) and credit_widening:
        matched_rule = "DETERIORATING: one primary warning is confirmed by widening credit stress."
    elif fast_warning or macro_warning:
        matched_rule = "BULL_WITH_WARNING: structure remains bullish, but fast or macro risk is above 20."
    elif liquidity_warning:
        matched_rule = "BULL_LIQUIDITY_WARNING: structure remains bullish, but medium-term liquidity is weakening."
    else:
        matched_rule = "BULL: bullish structure without active primary warning layers."

    return (
        f"Current inputs:\n"
        f"Structural Regime = {structural}\n"
        f"Fast Transition Risk = {fast_text}\n"
        f"Macro Transition Risk = {macro_text}\n"
        f"Global Liquidity Backdrop = {backdrop}\n"
        f"Global Liquidity Score = {format_market_value(liquidity_score, 'score')}\n"
        f"Global Liquidity Direction 13W = {format_market_value(liquidity_direction, 'score')} / {liquidity_direction_state}\n"
        f"Credit State = {credit_state}\n"
        f"Negative Confirmations = {negative_text}\n"
        f"Current Final Market State = {status}\n\n"
        f"Matched rule:\n"
        f"{matched_rule}\n\n"
        f"Rule priority, first match wins:\n"
        f"1. STRESS if Structural Regime is STRESS.\n"
        f"2. CORRECTION if Structural Regime is CORRECTION.\n"
        f"3. DETERIORATING if two or more primary warning layers are active.\n"
        f"4. DETERIORATING if one primary warning is confirmed by 3+ negative confirmations.\n"
        f"5. DETERIORATING if one primary warning is confirmed by WIDENING / SEVERE_WIDENING credit.\n"
        f"6. BULL_WITH_WARNING if Fast or Macro Transition Risk >= 20.\n"
        f"7. BULL_LIQUIDITY_WARNING if liquidity is warning/negative or direction is rapidly deteriorating.\n"
        f"8. BULL otherwise.\n\n"
        f"Confirmations are an amplifier only; they are not a standalone regime trigger."
    )


def market_regime_interpretation_text(market: dict) -> str:
    structural = format_market_value(market.get("Market_Regime"))
    final_state = format_market_value(market.get("Final_Market_State", market.get("Overall_Transition_Status")))
    fast = f"{format_market_value(market.get('Fast_Transition_Risk'), 'score')} / {format_market_value(market.get('Fast_Transition_State'))}"
    fast_direction = f"{format_market_value(market.get('Fast_Risk_Direction_4W'), 'score')} / {format_market_value(market.get('Fast_Risk_Direction_4W_State'))}"
    macro = f"{format_market_value(market.get('Macro_Transition_Risk'), 'score')} / {format_market_value(market.get('Macro_Transition_State'))}"
    backdrop = format_market_value(market.get("Global_Liquidity_Backdrop"))
    liquidity = f"{format_market_value(market.get('Global_Liquidity_Score'), 'score')} / {format_market_value(market.get('Global_Liquidity_Direction_13W_State'))}"
    credit = f"{format_market_value(market.get('Credit_State'))} / {format_market_value(market.get('Credit_Level_State'))}"
    confirmations = format_market_value(market.get("Negative_Confirmation_Count"))
    long_cycle = format_market_value(market.get("Long_Liquidity_Cycle"))
    if final_state == "BULL_LIQUIDITY_WARNING":
        implication = "Market structure remains bullish, but medium-term liquidity support is weakening materially. This is not the same as technical deterioration."
    elif final_state == "DETERIORATING":
        implication = "Multiple primary warning layers are active, so forward risk/reward is deteriorating even if price structure has not yet broken."
    elif final_state in {"CORRECTION", "STRESS"}:
        implication = "Structural price action has already moved out of a bull regime, so structural regime has priority over liquidity or confirmation layers."
    elif final_state == "BULL_WITH_WARNING":
        implication = "Market structure remains bullish, but near-term or macro transition risk is above the watch threshold."
    else:
        implication = "Market structure remains bullish without a strong primary warning combination."
    return (
        f"The market is currently in a structural {structural} regime.\n\n"
        f"Fast Transition Risk is {fast}; 4W direction is {fast_direction}. This block is a 1-4 week stress detector.\n\n"
        f"Macro Transition Risk is {macro}. This block captures developing 4-12 week macro pressure from DXY, US2Y, Global M2 26W and Fed Net Liquidity.\n\n"
        f"Credit stress confirmation is {credit}. Credit only escalates an existing primary warning; it does not enter Fast Transition Risk.\n\n"
        f"Global Liquidity Backdrop is {backdrop}; score/direction is {liquidity}, with long-cycle context {long_cycle}. This layer is a medium-term 8-26W+ expected-return backdrop.\n\n"
        f"Negative confirmations count is {confirmations}. Confirmations can amplify an existing primary warning but do not trigger deterioration alone.\n\n"
        f"Current final state: {final_state}. {implication}"
    )


def _render_market_global_liquidity_backdrop_chart() -> None:
    try:
        _, monthly, weekly = read_global_liquidity()
        frame = _build_global_liquidity_regime_frame(
            _liquidity_prepare_dates(monthly),
            _liquidity_prepare_dates(weekly),
        )
    except Exception:
        frame = pd.DataFrame()
    if frame.empty:
        st.info("No Global Liquidity data available.")
        return
    d = _liquidity_filter_range(frame, "5Y")
    if d.empty:
        st.info("No Global Liquidity data available for the selected range.")
        return
    fig = go.Figure()
    series = [
        ("global_liquidity_score", "Global Liquidity Score", "#38bdf8", 2.0),
        ("m2_impulse", "M2 Impulse", "#22c55e", 1.3),
        ("cb_impulse", "CB Impulse", "#facc15", 1.3),
        ("usnl_impulse", "USNL Impulse", "#f97316", 1.3),
    ]
    for column, label, color, width in series:
        if column in d.columns:
            fig.add_trace(
                go.Scatter(
                    x=d["date"],
                    y=pd.to_numeric(d[column], errors="coerce"),
                    mode="lines",
                    name=label,
                    line={"color": color, "width": width},
                )
            )
    if "direction_13w" in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=pd.to_numeric(d["direction_13w"], errors="coerce"),
                mode="lines",
                name="Direction 13W",
                yaxis="y2",
                line={"color": "#e879f9", "width": 1.5, "dash": "dash"},
            )
        )
    for level in [20, 40, 60, 80]:
        fig.add_hline(y=level, line={"color": "#94a3b8", "dash": "dot", "width": 1}, opacity=0.45)
    fig.update_layout(
        yaxis={"title": "Score", "range": [0, 100]},
        yaxis2={
            "title": "Direction 13W",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "color": "#cbd5e1",
            "linecolor": "#475569",
        },
    )
    st.plotly_chart(_style_liquidity_plotly(fig, 320, "Global Liquidity Backdrop"), use_container_width=True, config=LIQUIDITY_PLOTLY_CONFIG)


def render_market_regime_tab(market: dict) -> None:
    st.subheader("Market Regime")

    summary_cols = st.columns(5)
    with summary_cols[0]:
        render_market_metric("Structural Regime", format_market_value(market.get("Market_Regime")), "SPY weekly")
    with summary_cols[1]:
        render_market_metric("Final Market State", format_market_value(market.get("Final_Market_State", market.get("Overall_Transition_Status"))), "rule hierarchy")
    with summary_cols[2]:
        render_market_metric(
            "Macro Transition Risk",
            f"{format_market_value(market.get('Macro_Transition_Risk'), 'score')} / {format_market_value(market.get('Macro_Transition_State'))}",
            "DXY 40% + US2Y 30% + Global M2 20% + Fed liquidity 10%",
        )
    with summary_cols[3]:
        render_market_metric(
            "Fast Transition Risk",
            f"{format_market_value(market.get('Fast_Transition_Risk'), 'score')} / {format_market_value(market.get('Fast_Transition_State'))}",
            "VIX 70% + DXY 30%",
        )
    with summary_cols[4]:
        render_market_metric(
            "Fast Risk Direction 4W",
            f"{format_market_value(market.get('Fast_Risk_Direction_4W'), 'score')} / {format_market_value(market.get('Fast_Risk_Direction_4W_State'))}",
            "acceleration only",
        )

    summary_cols = st.columns(3)
    with summary_cols[0]:
        render_market_metric("Long Liquidity Cycle", format_market_value(market.get("Long_Liquidity_Cycle")), "context only")
    with summary_cols[1]:
        render_market_metric("Global Liquidity Backdrop", format_market_value(market.get("Global_Liquidity_Backdrop")), "8-26W+ backdrop")
    with summary_cols[2]:
        render_market_metric("Credit Stress Confirmation", format_market_value(market.get("Credit_State")), format_market_value(market.get("Credit_Level_State")))

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_market_metric("Negative Confirmations", format_market_value(market.get("Negative_Confirmation_Count")), "amplifier only")
    with summary_cols[1]:
        render_market_metric("WTI Confirmation", format_market_value(market.get("WTI_Confirmation")), "oil pressure")
    with summary_cols[2]:
        render_market_metric("10Y Real Yield Confirmation", format_market_value(market.get("Real_Yield_10Y_Confirmation")), "real-rate pressure")
    with summary_cols[3]:
        render_market_metric("RSI Divergence", format_market_value(market.get("RSI_Divergence")), "SPY weekly RSI14")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_market_metric("Positioning Risk", f"{format_market_value(market.get('PositioningRisk'), 'score')} / {format_market_value(market.get('PositioningState'))}", "AAII bearish + VIX AM")
    with summary_cols[1]:
        render_market_metric("Tail Risk Flag", format_market_value(market.get("TailRiskFlag")), format_market_value(market.get("TailRiskReason")))
    with summary_cols[2]:
        render_market_metric("Liquidity Warning", format_market_value(market.get("LiquidityWarning")), "directional pressure")
    with summary_cols[3]:
        render_market_metric("Credit Warning", format_market_value(market.get("CreditWarning")), "HY OAS transmission")

    st.markdown("### Market Regime Interpretation")
    render_market_formula("Interpretation", market_regime_interpretation_text(market))

    _render_spy_weekly_market_regime_chart()
    _render_tail_risk_validation_report()

    st.markdown("### Current Final Market State Logic")
    render_market_formula("Logic", overall_status_logic_text(market))

    st.markdown("### Structural Market Regime")
    render_market_detail_table(
        [
            ("Structural Regime", format_market_value(market.get("Market_Regime"))),
            ("SPY vs SMA40W", format_market_value(market.get("SPY_vs_SMA40W_%"), "percent_points")),
            ("SPY Drawdown 52W", format_market_value(market.get("SPY_Drawdown_52W_%"), "percent_points")),
            ("SPY Volatility 13W", format_market_value(market.get("SPY_Volatility_13W_%"), "percent_points")),
            ("SPY Volatility Percentile", format_market_value(market.get("SPY_Volatility_Percentile"), "score")),
        ]
    )
    render_market_formula(
        "Formula",
        "SPY_SMA40W = SMA(SPY weekly close, 40)\n"
        "SPY_Drawdown_52W = SPY_Close / RollingHigh52W - 1\n"
        "SPY_Vol13W = StdDev(weekly returns, 13) * sqrt(52)\n"
        "HighVol = SPY_Vol13W percentile >= 75\n"
        "StructuralBull = SPY_Close > SPY_SMA40W and SPY_Drawdown_52W > -10%\n"
        "BULL/BULL_HIGH_VOL if StructuralBull; CORRECTION/STRESS otherwise, split by HighVol",
    )

    st.markdown("### Fast Transition Risk")
    render_market_detail_table(
        [
            ("Fast Transition Risk", f"{format_market_value(market.get('Fast_Transition_Risk'), 'score')} / {format_market_value(market.get('Fast_Transition_State'))}"),
            ("VIX Robust Z 26W", format_market_value(market.get("VIX_Z26"))),
            ("VIX Risk", format_market_value(market.get("VIX_Risk"), "score")),
            ("DXY Return 13W", format_market_value(market.get("DXY_Return_13W"), "percent")),
            ("DXY Return 26W", format_market_value(market.get("DXY_Return_26W"), "percent")),
            ("DXY Risk", format_market_value(market.get("DXY_Risk"), "score")),
        ]
    )
    render_market_formula(
        "Formula",
        "MedianVIX26 = Median(VIX, 26W)\n"
        "MADVIX26 = Median(abs(VIX - MedianVIX26), 26W)\n"
        "VIX_Z26 = (VIX - MedianVIX26) / (1.4826 * MADVIX26)\n"
        "VIX_Risk = piecewise_score(VIX_Z26)\n"
        "DXY_Risk = piecewise_score(DXY_26W_Return)\n"
        "FastTransitionRisk = clip(0.70 * VIX_Risk + 0.30 * DXY_Risk, 0, 100)\n"
        "FastRiskDirection4W = FastTransitionRisk_t - FastTransitionRisk_t_minus_4W\n"
        "Fast states: LOW <=10, NORMAL <=20, WATCH <=40, HIGH <=60, EXTREME >60",
    )

    st.markdown("### Macro Transition Risk")
    render_market_detail_table(
        [
            ("Macro Transition Risk", f"{format_market_value(market.get('Macro_Transition_Risk'), 'score')} / {format_market_value(market.get('Macro_Transition_State'))}"),
            ("Fed Liquidity 13W", format_market_value(market.get("Fed_Liquidity_13W"), "percent")),
            ("Fed Liquidity 26W", format_market_value(market.get("Fed_Liquidity_26W"), "percent")),
            ("Fed Liquidity Risk", format_market_value(market.get("Fed_Liquidity_Risk"), "score")),
            ("US2Y Change 13W", format_market_value(market.get("US2Y_Change_13W_bp"), "bp")),
            ("US2Y Risk", format_market_value(market.get("US2Y_Risk"), "score")),
            ("Global M2 26W", format_market_value(market.get("Global_M2_26W"), "percent")),
            ("Global M2 Bull Score 26W", format_market_value(market.get("Global_M2_Bull_Score_26W"), "score")),
            ("Global M2 Risk 26W", format_market_value(market.get("Global_M2_Risk_26W"), "score")),
            ("DXY Risk", format_market_value(market.get("Macro_DXY_Risk"), "score")),
        ]
    )
    render_market_formula(
        "Formula",
        "FedLiquidity = WALCL - RRPONTSYD - WTREGEN\n"
        "FedLiquidity13W = FedLiquidity / FedLiquidity.shift(13) - 1\n"
        "FedLiquidity26W = FedLiquidity / FedLiquidity.shift(26) - 1\n"
        "US2Y_Change13W_bp = (DGS2 - DGS2.shift(13)) * 100\n"
        "DXY_Risk = piecewise_score(DXY_26W_Return)\n"
        "GlobalM2Growth26W = GlobalM2 / GlobalM2.shift(26) - 1\n"
        "GlobalM2BullScore = trailing 3Y percentile(GlobalM2Growth26W)\n"
        "GlobalM2Risk = 100 - GlobalM2BullScore\n"
        "FedLiquidity_Risk = piecewise_score(FedLiquidity26W)\n"
        "US2Y_Risk = piecewise_score(US2Y_Change13W_bp)\n"
        "MacroTransitionRisk = clip(0.40 * DXY_Risk + 0.30 * US2Y_Risk + 0.20 * GlobalM2Risk + 0.10 * FedLiquidity_Risk, 0, 100)",
    )

    st.markdown("### Credit Stress Confirmation")
    render_market_detail_table(
        [
            ("HY OAS", format_market_value(market.get("HY_OAS"))),
            ("HY OAS Change 13W", format_market_value(market.get("HY_OAS_Change_13W"))),
            ("Credit Widening Percentile", format_market_value(market.get("Credit_Widening_Percentile", market.get("Credit_Risk")), "score")),
            ("Credit State", format_market_value(market.get("Credit_State"))),
            ("HY Level Percentile", format_market_value(market.get("HY_Level_Percentile"), "score")),
            ("Credit Level State", format_market_value(market.get("Credit_Level_State"))),
        ]
    )
    render_market_formula(
        "Formula",
        "HYOASChange13W = BAMLH0A0HYM2_t - BAMLH0A0HYM2_t_minus_13W\n"
        "CreditRisk = trailing 3Y percentile(HYOASChange13W)\n"
        "HYLevelPercentile = trailing 3Y percentile(HY OAS level)\n"
        "Credit confirms/escalates existing macro/liquidity warnings only; it is not included in Fast Risk.",
    )

    st.markdown("### Global Liquidity Backdrop")
    render_market_detail_table(
        [
            ("Global Liquidity Backdrop", format_market_value(market.get("Global_Liquidity_Backdrop"))),
            ("Global Liquidity Score", format_market_value(market.get("Global_Liquidity_Score"), "score")),
            ("Global Liquidity Direction 13W", f"{format_market_value(market.get('Global_Liquidity_Direction_13W'), 'score')} / {format_market_value(market.get('Global_Liquidity_Direction_13W_State'))}"),
            ("Long Liquidity Cycle", format_market_value(market.get("Long_Liquidity_Cycle"))),
            ("Data Status", format_market_value(market.get("Global_Liquidity_Data_Status"))),
            ("Last Updated", format_market_value(market.get("Global_Liquidity_Last_Updated"))),
        ]
    )
    render_market_formula(
        "Formula",
        "GlobalLiquidityScore = 0.50 * M2Impulse + 0.25 * CBImpulse + 0.25 * USNLImpulse\n"
        "GlobalLiquidityDirection13W = GlobalLiquidityScore_t - GlobalLiquidityScore_t_minus_13W\n"
        "GlobalLiquidityBackdrop is a separate medium-term backdrop and does not directly change Structural Regime.",
    )
    _render_market_global_liquidity_backdrop_chart()

    st.markdown("### Positioning and Tail Risk Overlay")
    render_market_detail_table(
        [
            ("Positioning Risk", f"{format_market_value(market.get('PositioningRisk'), 'score')} / {format_market_value(market.get('PositioningState'))}"),
            ("AAII Bearish 3Y Percentile", format_market_value(market.get("AAII_Bearish_3Y_Percentile"), "score")),
            ("VIX Asset Manager Net % OI", format_market_value(market.get("VIX_AssetManager_NetPctOI"), "score")),
            ("VIX Asset Manager Net % OI 3Y Percentile", format_market_value(market.get("VIX_AssetManager_NetPctOI_3Y_Percentile"), "score")),
            ("Liquidity Warning", format_market_value(market.get("LiquidityWarning"))),
            ("Credit Warning", format_market_value(market.get("CreditWarning"))),
            ("Fast Warning", format_market_value(market.get("FastWarning"))),
            ("Macro Warning", format_market_value(market.get("MacroWarning"))),
            ("Tail Risk Flag", format_market_value(market.get("TailRiskFlag"))),
            ("Tail Risk Reason", format_market_value(market.get("TailRiskReason"))),
            ("Positioning Model Version", format_market_value(market.get("PositioningModel_Version"))),
            ("Tail Risk Model Version", format_market_value(market.get("TailRiskModel_Version"))),
        ]
    )
    render_market_formula(
        "Formula",
        "PositioningRisk = 0.50 * AAII_Bearish_3Y_Percentile + 0.50 * VIX_AssetManager_NetPctOI_3Y_Percentile\n"
        "CFTC VIX Asset Manager positioning uses report date + 3 calendar days when publication date is unavailable.\n"
        "TailRiskFlag is rule-based interaction logic across Positioning, Liquidity, Credit, Fast Risk, Macro Risk and Structural Regime.\n"
        "TailRiskFlag is a separate overlay and does not change StructuralMarketRegime or FinalMarketState.",
    )

    st.markdown("### Confirmations")
    render_market_detail_table(
        [
            ("WTI 4W", format_market_value(market.get("WTI_4W_Return"), "percent")),
            ("WTI 13W", format_market_value(market.get("WTI_13W_Return"), "percent")),
            ("WTI 26W", format_market_value(market.get("WTI_26W_Return"), "percent")),
            ("WTI Confirmation", format_market_value(market.get("WTI_Confirmation"))),
            ("10Y Real Yield 13W", format_market_value(market.get("Real_Yield_10Y_Change_13W_bp"), "bp")),
            ("10Y Real Yield Confirmation", format_market_value(market.get("Real_Yield_10Y_Confirmation"))),
            ("IWM/SPY 13W", format_market_value(market.get("IWM_SPY_13W_Return"), "percent")),
            ("IWM/SPY Confirmation", format_market_value(market.get("IWM_SPY_Confirmation"))),
            ("XLI/XLP 13W", format_market_value(market.get("XLI_XLP_13W_Return"), "percent")),
            ("XLI/XLP Confirmation", format_market_value(market.get("XLI_XLP_Confirmation"))),
            ("RSI Divergence", format_market_value(market.get("RSI_Divergence"))),
            ("Negative Confirmation Count", format_market_value(market.get("Negative_Confirmation_Count"))),
            ("Confirmation Flag", format_market_value(market.get("Confirmation_Flag"))),
        ]
    )
    render_market_formula(
        "Formula",
        "WTI confirmation = classify(WTI_13W_Return)\n"
        "10Y Real Yield confirmation = classify((DFII10 - DFII10.shift(13)) * 100 bp)\n"
        "IWM/SPY confirmation = classify((IWM/SPY) / (IWM/SPY).shift(13) - 1)\n"
        "XLI/XLP confirmation = classify((XLI/XLP) / (XLI/XLP).shift(13) - 1)\n"
        "RSI Divergence = latest SPY weekly higher-high with lower RSI14 high\n"
        "NegativeConfirmationCount = NEGATIVE + STRONG_NEGATIVE confirmations; MILD RSI divergence counts as 0.5",
    )


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows to chart for current filters.")
        return

    chart_df = _build_chart_frame(df)

    _render_performance_sma200w_bubble_chart(chart_df)
    _render_rsi_chart(chart_df)
    _render_bar_chart(
        chart_df,
        "SMA50w_vs_SMA200w_Spread_%",
        "Price vs SMA200d (%)",
        dot_metric="SMA50w_vs_SMA200w_Spread_Avg_36M_%",
    )
    _render_bar_chart(chart_df, "Perf_1W_%", "Performance Week")
    _render_bar_chart(chart_df, "Perf_1M_%", "Performance 1m")
    _render_bar_chart(chart_df, "Perf_3M_%", "Performance 3m")
    _render_bar_chart(chart_df, "Perf_6M_%", "Performance 6m")
    _render_bar_chart(chart_df, "Perf_12M_%", "Performance 12m")
    _render_bar_chart(chart_df, "Perf_3Y_%", "Performance 3Y")
    _render_bar_chart(chart_df, "Perf_5Y_%", "Performance 5Y")


def _stepper_number(label: str, key: str, default: int, min_value: int = 1, max_value: int = 500, step: int = 1) -> int:
    if key not in st.session_state:
        st.session_state[key] = int(default)
    st.caption(label)
    c1, c2, c3 = st.columns([1.4, 0.8, 0.8])
    with c1:
        st.number_input(
            label=f"{label}_value",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=key,
            label_visibility="collapsed",
        )
    with c2:
        if st.button("-", key=f"{key}_minus", use_container_width=True):
            st.session_state[key] = max(min_value, int(st.session_state[key]) - step)
            st.rerun()
    with c3:
        if st.button("+", key=f"{key}_plus", use_container_width=True):
            st.session_state[key] = min(max_value, int(st.session_state[key]) + step)
            st.rerun()
    return int(st.session_state[key])


def render_divergence_settings() -> dict:
    with st.expander("Divergence Settings", expanded=False):
        profile = st.selectbox(
            "Divergence profile",
            options=["Weekly", "Monthly"],
            index=0,
            key="div_profile",
        )
        profile_defaults = DIVERGENCE_PROFILE_DEFAULTS[profile]
        profile_key = profile.lower()

        pivot_window = _stepper_number(
            label="Pivot window",
            key=f"div_{profile_key}_pivot_window",
            default=profile_defaults["pivot_window"],
            min_value=1,
            max_value=20,
            step=1,
        )
        lookback_bars = _stepper_number(
            label="Lookback bars",
            key=f"div_{profile_key}_lookback_bars",
            default=profile_defaults["lookback_bars"],
            min_value=20,
            max_value=300,
            step=5,
        )

    cfg = dict(DIVERGENCE_DEFAULTS)
    cfg["profile"] = profile
    cfg["pivot_window"] = int(pivot_window)
    cfg["lookback_bars"] = int(lookback_bars)
    return cfg


def _first_query_value(query: Any, key: str) -> str | None:
    try:
        value = query.get(key)
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def _handle_tradingview_oauth_callback() -> None:
    query = getattr(st, "query_params", {})
    code = _first_query_value(query, "code")
    state = _first_query_value(query, "state")
    if not code or not state:
        return
    try:
        import tradingview_mcp as tv_mcp

        tv_mcp.exchange_code(code, state)
        st.success("TradingView OAuth authorization saved.")
        try:
            st.query_params.clear()
        except Exception:
            pass
    except Exception as exc:
        st.warning(f"TradingView OAuth callback failed: {exc}")


def main():
    st.set_page_config(page_title="ETF Market Screener", layout="wide")
    _handle_tradingview_oauth_callback()
    if "universe_map" not in st.session_state:
        st.session_state["universe_map"] = load_universe_map()
    if "performance_refresh_nonce" not in st.session_state:
        st.session_state["performance_refresh_nonce"] = 0
    if "slow_refresh_nonce" not in st.session_state:
        st.session_state["slow_refresh_nonce"] = 0
    if "nightly_job_notice" not in st.session_state:
        st.session_state["nightly_job_notice"] = ""
    universe_map = st.session_state["universe_map"]

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.35rem !important;
            padding-bottom: 0.35rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            max-width: 100% !important;
        }
        [data-testid="stHeader"] {
            height: 1.35rem;
            background: transparent;
        }
        [data-testid="stToolbar"] {
            top: 0.25rem;
            right: 0.35rem;
        }
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            font-size: 0.74rem;
            padding: 0.2rem 0.5rem;
            min-height: 1.55rem;
        }
        div[data-baseweb="select"] > div {
            min-height: 1.62rem;
            font-size: 0.72rem;
        }
        label p {
            font-size: 0.69rem !important;
            line-height: 0.95rem !important;
        }
        div[data-testid="stSelectbox"] label p {
            min-height: 1.95rem !important;
            display: flex !important;
            align-items: flex-end !important;
        }
        [data-testid="stCaptionContainer"] p {
            margin-top: 0.1rem !important;
            margin-bottom: 0.2rem !important;
            font-size: 0.7rem !important;
        }
        [data-testid="stTabs"] {
            margin-top: -0.15rem !important;
        }
        div[data-testid="stRadio"] > div {
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            gap: 0.35rem !important;
            padding-bottom: 0.2rem !important;
        }
        div[data-testid="stRadio"] label {
            white-space: nowrap !important;
        }
        [data-testid="stDataFrame"] [role="columnheader"] {
            font-size: 0.77rem !important;
            min-width: 108px !important;
            width: 108px !important;
            max-width: 108px !important;
            white-space: normal !important;
            word-break: break-word !important;
            line-height: 1.05 !important;
            height: auto !important;
        }
        [data-testid="stDataFrame"] [role="gridcell"] {
            font-size: 0.75rem !important;
            min-width: 108px !important;
            width: 108px !important;
            max-width: 108px !important;
        }
        [data-testid="stDataFrame"] {
            margin-top: 0.1rem !important;
        }
        .ag-theme-streamlit .ag-header-cell-label {
            white-space: normal !important;
            line-height: 1.1 !important;
            align-items: center !important;
        }
        .ag-theme-streamlit .ag-header-cell-text {
            white-space: pre-line !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    (
        top_left,
        top_market_col,
        top_fast_col,
        top_macro_col,
        top_overall_col,
        top_entry_col,
        top_confidence_col,
        top_mid,
        top_export_col,
        top_refresh_col,
        top_hard_refresh_col,
    ) = st.columns(
        [1.85, 1.15, 1.45, 1.65, 1.15, 1.15, 1.15, 0.3, 1.15, 0.95, 1.35]
    )
    with top_left:
        selected_universe_name = st.selectbox("ETF Version", options=list(universe_map.keys()), index=0)
    with top_market_col:
        market_status_slot = st.empty()
    with top_fast_col:
        fast_transition_status_slot = st.empty()
    with top_macro_col:
        macro_transition_status_slot = st.empty()
    with top_overall_col:
        overall_status_slot = st.empty()
    with top_entry_col:
        entry_risk_status_slot = st.empty()
    with top_confidence_col:
        alpha_confidence_status_slot = st.empty()
    with top_export_col:
        table_export_slot = st.empty()
    with top_refresh_col:
        refresh = st.button("Refresh Prices", use_container_width=True)
    with top_hard_refresh_col:
        hard_refresh = st.button("Run Nightly Analytics Now", use_container_width=True)
    if refresh:
        st.session_state["performance_refresh_nonce"] += 1
        st.rerun()
    if hard_refresh:
        started, message = start_refresh_job("nightly-analytics")
        st.session_state["nightly_job_notice"] = (
            "Nightly analytics job started in background."
            if started
            else f"Nightly analytics job was not started: {message}"
        )
        st.rerun()

    divergence_cfg = render_divergence_settings()
    divergence_signature = (
        f"profile={divergence_cfg['profile']}:"
        f"pv={divergence_cfg['pivot_window']}:"
        f"lb={divergence_cfg['lookback_bars']}"
    )

    with st.spinner("Computing ETF metrics..."):
        selected_universe = universe_map[selected_universe_name]
        market_signature = f"market-model:{selected_universe_name}:{str(selected_universe)}:{divergence_signature}"
        performance_refresh_key = int(time.time() // AUTO_REFRESH_SECONDS) + (
            int(st.session_state["performance_refresh_nonce"]) * 10_000_000
        )
        df, app_refresh_utc, snapshot_status = compute_metrics_table(
            selected_universe,
            f"{selected_universe_name}:{str(selected_universe)}",
            divergence_cfg,
            divergence_signature,
            performance_refresh_key,
            st.session_state["slow_refresh_nonce"],
        )
        market_snapshot = load_market_model_snapshot(market_signature)
    refresh_text = "n/a"
    try:
        ts = pd.to_datetime(app_refresh_utc, utc=True)
        refresh_text = ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    status_payload = snapshot_status if isinstance(snapshot_status, dict) else {}
    analytics_meta = status_payload.get("analytics", {}) if isinstance(status_payload, dict) else {}
    analytics_text = "n/a"
    analytics_status = str(analytics_meta.get("Status", "n/a"))
    try:
        analytics_ts = pd.to_datetime(analytics_meta.get("CalculatedAt"), utc=True)
        analytics_text = analytics_ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    _, refresh_col = st.columns([8, 2])
    with refresh_col:
        st.markdown(
            (
                "<div style='text-align:right; font-size:0.76rem; color:#cbd5e1; line-height:1.25;'>"
                f"Prices updated: <b>{refresh_text}</b><br>"
                f"Analytics calculated: <b>{analytics_text}</b><br>"
                f"Status: <b>{html.escape(analytics_status)}</b>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    if background_job_running("nightly_analytics"):
        st.info("Nightly analytics is running in the background. Snapshots will appear when it finishes.")
    elif st.session_state.get("nightly_job_notice"):
        st.caption(st.session_state["nightly_job_notice"])

    filtered_df, flow_unavailable = apply_filters(df)
    render_top_alpha_status(
        market_status_slot,
        fast_transition_status_slot,
        macro_transition_status_slot,
        overall_status_slot,
        entry_risk_status_slot,
        alpha_confidence_status_slot,
        market_snapshot,
        df,
        filtered_df,
    )
    graph_ordered_df = filtered_df.copy()
    table_df = filtered_df.copy().reset_index(drop=True)
    for display_col in DISPLAY_COLUMNS:
        if display_col not in table_df.columns:
            table_df[display_col] = np.nan
    table_df["__row_id__"] = np.arange(len(table_df))
    table_display_df = table_df[["__row_id__"] + DISPLAY_COLUMNS].copy()
    table_col_labels = {
        col: TABLE_HEADER_NAMES.get(col, col.replace("_", " "))
        for col in DISPLAY_COLUMNS
    }
    table_display_df = table_display_df.rename(columns=table_col_labels)

    with table_export_slot:
        st.download_button(
            "Download .xls",
            data=dataframe_to_excel_xls_bytes(table_display_df),
            file_name="screener_table.xls",
            mime="application/vnd.ms-excel",
            key="table_xls_download",
            use_container_width=True,
        )

    st.caption(f"Rows: {len(filtered_df)}/{len(df)}")

    if flow_unavailable:
        st.warning("Fund flow data unavailable")

    view_options = [
        "Table",
        "Charts",
        "Graphs",
        "AI Dashboard",
        "Market Cycle",
        "Liquidity Cycle",
        "Business Cycle",
        "Rates & Financial Conditions",
        "Funding Conditions",
        "Treasury & Fiscal Regime",
        "Global Dashboard",
        "CIO View",
        "Global Macro",
        "CFTC COT",
        "Gold Regime",
        "BTC Regime",
        "Crypto Derivatives",
        "Alpha Engine",
        "Financial Fragility",
        "Market Regime",
        "Inputs",
        "Description",
        "Tester",
    ]
    active_view = st.radio(
        "View",
        options=view_options,
        horizontal=True,
        label_visibility="collapsed",
        key="active_main_view",
    )
    if active_view == "Table":
        gb = GridOptionsBuilder.from_dataframe(table_display_df)
        gb.configure_default_column(
            sortable=True,
            filter=False,
            resizable=True,
            minWidth=48,
            width=54,
            maxWidth=75,
            wrapHeaderText=True,
            autoHeaderHeight=True,
        )
        gb.configure_column("__row_id__", hide=True, sortable=False, filter=False)
        gb.configure_grid_options(rowHeight=24, domLayout="autoHeight")

        compact_widths = {
            "Group": 120,
            "Subgroup": 140,
            "Ticker": 95,
            "Alpha_Score": 82,
            "Alpha_State": 264,
            "Momentum_Score": 92,
            "Trend_Quality_Score": 108,
            "Persistence_Score": 100,
            "Market_Regime": 130,
            "Entry_Risk_Score": 94,
            "Entry_Risk": 92,
            "Opportunity_State": 190,
            "Opportunity_Score": 104,
            "Price_vs_52W_High_%": 130,
            "Price_vs_ATH_%": 110,
            "RSI_14": 39,
            "RSI_14W": 48,
            "Divergence_Bull_Count": 54,
            "Divergence_Bear_Count": 54,
        }
        hidden_labels = {
            table_col_labels[col]
            for col in TABLE_PERMANENTLY_HIDDEN_COLUMNS
            if col in table_col_labels
        }
        for col in DISPLAY_COLUMNS:
            header_label = table_col_labels.get(col, col.replace("_", " "))
            if header_label in table_display_df.columns:
                gb.configure_column(
                    header_label,
                    headerName=header_label,
                    width=compact_widths.get(col, 108),
                    minWidth=compact_widths.get(col, 108),
                    suppressSizeToFit=True,
                    hide=header_label in hidden_labels,
                )

        number_formatter = JsCode(
            """
            function(params) {
                if (params.value === null || params.value === undefined || isNaN(params.value)) {
                    return "NaN";
                }
                return Number(params.value).toFixed(0);
            }
            """
        )
        for col in NUMERIC_COLUMNS:
            col_label = table_col_labels.get(col, col.replace("_", " "))
            if col_label in table_display_df.columns:
                gb.configure_column(col_label, valueFormatter=number_formatter, type=["numericColumn"])

        perf_color_styles = {}
        for col in PERFORMANCE_COLUMNS:
            col_label = table_col_labels.get(col, col.replace("_", " "))
            if col_label not in table_display_df.columns:
                continue
            vals_source = table_display_df
            if col == "Perf_10Y_%" and "Ticker" in table_display_df.columns:
                vals_source = table_display_df[table_display_df["Ticker"] != "BTC-USD"]
            vals = pd.to_numeric(vals_source[col_label], errors="coerce").dropna()
            if vals.empty:
                continue
            vmin = float(vals.min())
            vmax = float(vals.max())
            btc_skip_rule = "if (params.data && params.data['Ticker'] === 'BTC-USD') { return {}; }" if col == "Perf_10Y_%" else ""
            perf_color_styles[col_label] = JsCode(
                f"""
                function(params) {{
                    {btc_skip_rule}
                    if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                        return {{}};
                    }}
                    const min = {vmin};
                    const max = {vmax};
                    if (max === min) {{
                        return {{backgroundColor: "#fff3bf", color: "#111827"}};
                    }}
                    const t = (Number(params.value) - min) / (max - min);
                    const r = Math.round(248 + t * (74 - 248));
                    const g = Math.round(113 + t * (222 - 113));
                    const b = Math.round(113 + t * (128 - 113));
                    return {{backgroundColor: `rgb(${{r}}, ${{g}}, ${{b}})`, color: "#111827"}};
                }}
                """
            )
            gb.configure_column(col_label, cellStyle=perf_color_styles[col_label])

        for col in INVERSE_PERFORMANCE_COLUMNS:
            col_label = table_col_labels.get(col, col.replace("_", " "))
            if col_label not in table_display_df.columns:
                continue
            vals = pd.to_numeric(table_display_df[col_label], errors="coerce").dropna()
            if vals.empty:
                continue
            vmin = float(vals.min())
            vmax = float(vals.max())
            perf_color_styles[col_label] = JsCode(
                f"""
                function(params) {{
                    if (params.value === null || params.value === undefined || isNaN(params.value)) {{
                        return {{}};
                    }}
                    const min = {vmin};
                    const max = {vmax};
                    if (max === min) {{
                        return {{backgroundColor: "#fff3bf", color: "#111827"}};
                    }}
                    const t = (Number(params.value) - min) / (max - min);
                    const r = Math.round(74 + t * (248 - 74));
                    const g = Math.round(222 + t * (113 - 222));
                    const b = Math.round(128 + t * (113 - 128));
                    return {{backgroundColor: `rgb(${{r}}, ${{g}}, ${{b}})`, color: "#111827"}};
                }}
                """
            )
            gb.configure_column(col_label, cellStyle=perf_color_styles[col_label])

        alpha_score_label = table_col_labels.get("Alpha_Score")
        if alpha_score_label in table_display_df.columns:
            gb.configure_column(
                alpha_score_label,
                cellStyle=JsCode(
                    """
                    function(params) {
                        if (params.value === null || params.value === undefined || isNaN(params.value)) {
                            return {};
                        }
                        const v = Number(params.value);
                        if (v >= 80) { return {backgroundColor: "#16a34a", color: "#ffffff"}; }
                        if (v >= 70) { return {backgroundColor: "#4ade80", color: "#111827"}; }
                        if (v >= 60) { return {backgroundColor: "#86efac", color: "#111827"}; }
                        if (v >= 50) { return {backgroundColor: "#fef3c7", color: "#111827"}; }
                        if (v >= 40) { return {backgroundColor: "#fdba74", color: "#111827"}; }
                        return {backgroundColor: "#f87171", color: "#111827"};
                    }
                    """
                ),
            )

        grid_options = gb.build()

        # The Streamlit component wrapper still needs an explicit height.
        # Size it to all rows to keep a single-page scroll (no nested grid scroll).
        table_height = max(520, 96 + (len(table_display_df) * 24))
        grid_response = AgGrid(
            table_display_df,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SORTING_CHANGED | GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True,
            height=table_height,
            key="table_aggrid",
        )

        sorted_grid_data = getattr(grid_response, "data", None)
        if isinstance(sorted_grid_data, pd.DataFrame) and "__row_id__" in sorted_grid_data.columns:
            ordered_ids = (
                pd.to_numeric(sorted_grid_data["__row_id__"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        else:
            ordered_ids = table_df["__row_id__"].tolist()

        st.session_state["table_graph_row_order"] = ordered_ids
        graph_ordered_df = (
            table_df.set_index("__row_id__")
            .reindex(ordered_ids)
            .dropna(how="all")
            .reset_index(drop=True)
        )
    elif active_view == "Charts":
        render_charts(filtered_df)
    elif active_view == "Graphs":
        row_order = st.session_state.get("table_graph_row_order", None)
        base_df = filtered_df.copy().reset_index(drop=True)
        base_df["__row_id__"] = np.arange(len(base_df))
        if row_order:
            id_set = set(base_df["__row_id__"].tolist())
            ordered = [rid for rid in row_order if rid in id_set]
            missing = [rid for rid in base_df["__row_id__"].tolist() if rid not in set(ordered)]
            final_order = ordered + missing
            graph_ordered_df = (
                base_df.set_index("__row_id__")
                .reindex(final_order)
                .dropna(how="all")
                .reset_index(drop=True)
            )
        else:
            graph_ordered_df = base_df
        render_graphs_tab(graph_ordered_df.drop(columns=["__row_id__"], errors="ignore"), selected_universe, selected_universe_name)
    elif active_view == "AI Dashboard":
        render_ai_dashboard_tab()
    elif active_view == "CIO View":
        render_cio_view_tab(table_df.drop(columns=["__row_id__"], errors="ignore"), market_snapshot, get_fred_api_key_for_app())
    elif active_view == "Global Dashboard":
        try:
            _, dashboard_monthly, dashboard_weekly = read_global_liquidity()
            dashboard_liquidity = _build_global_liquidity_regime_frame(
                _liquidity_prepare_dates(dashboard_monthly),
                _liquidity_prepare_dates(dashboard_weekly),
            )
        except Exception:
            dashboard_liquidity = pd.DataFrame()
        render_global_dashboard_tab(
            get_fred_api_key_for_app(),
            market_snapshot,
            dashboard_liquidity,
            transition_history_loader=lambda: load_market_transition_history(
                "financial-fragility-validation-export"
            ),
        )
    elif active_view == "Financial Fragility":
        try:
            _, fragility_monthly, fragility_weekly = read_global_liquidity()
            fragility_liquidity = _build_global_liquidity_regime_frame(
                _liquidity_prepare_dates(fragility_monthly),
                _liquidity_prepare_dates(fragility_weekly),
            )
        except Exception:
            fragility_liquidity = pd.DataFrame()
        render_financial_fragility_tab(get_fred_api_key_for_app(), fragility_liquidity)
    elif active_view == "Market Regime":
        render_market_regime_tab(market_snapshot)
    elif active_view == "Market Cycle":
        render_market_cycle_tab(get_fred_api_key_for_app())
    elif active_view == "Business Cycle":
        render_business_cycle_tab(get_fred_api_key_for_app())
    elif active_view == "Global Macro":
        render_global_macro_tab(get_fred_api_key_for_app())
    elif active_view == "CFTC COT":
        render_cftc_cot_tab()
    elif active_view == "Liquidity Cycle":
        render_global_liquidity_dashboard_tab()
    elif active_view == "Rates & Financial Conditions":
        render_rates_financial_conditions_tab(get_fred_api_key_for_app())
    elif active_view == "Funding Conditions":
        render_funding_conditions_tab(get_fred_api_key_for_app())
    elif active_view == "Treasury & Fiscal Regime":
        render_treasury_fiscal_regime_tab(get_fred_api_key_for_app())
    elif active_view == "Gold Regime":
        render_gold_regime_tab(table_df.drop(columns=["__row_id__"], errors="ignore"), get_fred_api_key_for_app())
    elif active_view == "BTC Regime":
        render_btc_regime_tab(table_df.drop(columns=["__row_id__"], errors="ignore"), market_snapshot)
    elif active_view == "Crypto Derivatives":
        render_crypto_derivatives_tab()
    elif active_view == "Alpha Engine":
        render_alpha_engine_tab(table_df.drop(columns=["__row_id__"], errors="ignore"))
    elif active_view == "Inputs":
        render_inputs_tab(universe_map)
    elif active_view == "Description":
        render_description_tab()
    elif active_view == "Tester":
        render_tester_tab()


if __name__ == "__main__":
    main()

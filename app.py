# pip install streamlit yfinance ta pandas numpy

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
import time
import os
import json
from copy import deepcopy
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
import streamlit.components.v1 as components

from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD

# ============================================================
# 1) ETF Universe (exactly as provided)
# ============================================================
ETF_UNIVERSE_FULL = {
    "Crypto": {
        "Crypto": ["IBIT"],
    },
    "Gold": {
        "Gold": ["GLD", "SLV", "GDX", "GDXJ", "SIL"],
    },
    "Equities": {
        "US": [
            "SPY", "QQQ", "IWM", "MTUM", "VLU", "VUG", "SPYD",
            "XLK", "ARKK", "XLC", "XLF", "XLI", "XLY", "XLE",
            "XLB", "XLP", "XLU", "XLV", "IBB"
        ],
        "DM": ["VEA", "EWG", "EWI", "EWU", "EWJ", "EPP", "EWY"],
        "EM": ["EEM", "MCHI", "FXI", "KWEB", "INDA", "ASEA", "EWZ", "ILF"],
    },
    "Bonds": {
        "Bonds": ["SGOV", "SHY", "IEF", "TLT", "TIP", "EMB", "HYG", "JNK"],
    },
    "Commodities": {
        "Energy": ["DBO", "URNM", "UGA"],
        "Commodities": ["DBC", "DBA", "DBO", "DBP", "DBB", "DBE"],
        "Metals": ["PALL", "CPER", "PPLT"],
    },
    "Real Estate": {
        "RE": ["VNQ", "VNQI"]
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

ETF_UNIVERSE_MAP = {
    "Full list": ETF_UNIVERSE_FULL,
    "Short List": ETF_UNIVERSE_SHORT,
}
UNIVERSE_STORAGE_PATH = Path(__file__).with_name("custom_universe_lists.json")

GRAPH_PERIOD_OPTIONS = ["Daily", "Weekly", "Monthly", "Full history"]

DISPLAY_COLUMNS = [
    "Group", "Subgroup", "Ticker",
    "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%", "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%",
    "FundFlows_1M_%", "FundFlows_3M_%",
    "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14", "ADX_14", "BB_Position",
    "SMA50w_vs_SMA200w_Spread_%", "SMA_Spread_%_Change_6M_%", "SMA_Trend",
    "Div_6M_vs_RSI", "Div_6M_vs_MACD", "Div_6M_vs_ROC",
    "Divergence_Bull_Count", "Divergence_Bear_Count",
]

TABLE_HEADER_NAMES = {
    "Group": "Group",
    "Subgroup": "Sub\ngroup",
    "Ticker": "Ticker",
    "Perf_1W_%": "Perf\n1W %",
    "Perf_1M_%": "Perf\n1M %",
    "Perf_3M_%": "Perf\n3M %",
    "Perf_6M_%": "Perf\n6M %",
    "Perf_12M_%": "Perf\n12M %",
    "Perf_3Y_%": "Perf\n3Y %",
    "Perf_5Y_%": "Perf\n5Y %",
    "FundFlows_1M_%": "FundFlows\n1M %",
    "FundFlows_3M_%": "FundFlows\n3M %",
    "Price_vs_52W_High_%": "Price vs\n52W High %",
    "Price_vs_ATH_%": "Price vs\nATH %",
    "RSI_14": "RSI\n14",
    "ADX_14": "ADX\n14",
    "BB_Position": "BB\nPosition",
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
    "FundFlows_1M_%",
    "FundFlows_3M_%",
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
    "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%", "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%",
    "FundFlows_1M_%", "FundFlows_3M_%",
    "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14",
    "BB_Position",
    "BB_Mid", "BB_Upper", "BB_Lower", "BB_StepUp", "BB_StepDown", "WeeklyClose_Last",
    "SMA50w_vs_SMA200w_Spread_%", "SMA50w_vs_SMA200w_Spread_Avg_36M_%", "SMA_Spread_%_Change_6M_%",
    "ADX_14", "DI_Plus_14", "DI_Minus_14", "DI_Plus_14_Delta2", "DI_Minus_14_Delta2",
    "Divergence_Bull_Count", "Divergence_Bear_Count",
]

PERFORMANCE_COLUMNS = [
    "Perf_1W_%",
    "Perf_1M_%",
    "Perf_3M_%",
    "Perf_6M_%",
    "Perf_12M_%",
    "Perf_3Y_%",
    "Perf_5Y_%",
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
                t = str(ticker).strip().upper()
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
    start_dt = end_dt - pd.Timedelta(days=days)
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
def download_metrics_ohlcv(ticker: str) -> pd.DataFrame:
    # yfinance can intermittently fail per-symbol; retry before giving up.
    attempts = 3
    for i in range(attempts):
        try:
            px = yf.download(
                ticker,
                period="10y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            frame = extract_ohlcv_frame(px, ticker)
            if not frame.empty:
                return frame
        except Exception:
            pass
        time.sleep(0.25 * (i + 1))

    # Fallback on max history in case period-specific query fails.
    try:
        px = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        frame = extract_ohlcv_frame(px, ticker)
        if not frame.empty:
            return frame
    except Exception:
        pass

    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


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
        perf_1w = safe_perf(close, today, 7)
        perf_1m = safe_perf(close, today, 30)
        perf_3m = safe_perf(close, today, 90)
        perf_6m = safe_perf(close, today, 182)
        perf_12m = safe_perf(close, today, 365)
        perf_3y = safe_perf(close, today, 365 * 3)
        perf_5y = safe_perf(close, today, 365 * 5)

        # 52W high distance
        last_52w = close.loc[today - pd.Timedelta(days=365 * 1.1):]
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
        cutoff_6m = today - pd.Timedelta(days=182)
        sma200d = close.rolling(window=200, min_periods=200).mean()
        sma200d_now = safe_last(sma200d)
        spread_pct_now = pct_spread(cur_px, sma200d_now)
        spread_series_pct = (close / sma200d - 1.0) * 100.0
        spread_valid = spread_series_pct.dropna()
        spread_avg_36m = np.nan
        if not spread_valid.empty:
            # 36 months ~= 756 trading days; use shorter available history when needed.
            spread_avg_36m = float(spread_valid.tail(756).mean())

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
        (
            bb_position,
            bb_mid,
            bb_upper,
            bb_lower,
            bb_step_up,
            bb_step_down,
            wk_close_last,
        ) = compute_weekly_bb_position(wk_close, period=50, std_mult=2.0)
        if len(wk_close) >= 260:
            sma50w = wk_close.rolling(window=50, min_periods=50).mean()
            sma200w = wk_close.rolling(window=200, min_periods=200).mean()
            golden_cross_w1, death_cross_w1 = detect_recent_sma_crossover(sma50w, sma200w, lookback_bars=14)

        # Fund flows placeholders (data source needed)
        flows_1m = np.nan
        flows_3m = np.nan

        return [
            perf_1w, perf_1m, perf_3m, perf_6m,
            perf_12m, perf_3y, perf_5y,
            flows_1m, flows_3m,
            vs_52w, vs_ath, cur_rsi,
            spread_pct_now, spread_avg_36m, spread_pct_change_6m, sma_trend,
            cur_adx14, cur_di_plus14, cur_di_minus14, cur_di_plus14_delta2, cur_di_minus14_delta2,
            bb_position, bb_mid, bb_upper, bb_lower, bb_step_up, bb_step_down, wk_close_last,
            int(golden_cross_d1), int(death_cross_d1), int(golden_cross_w1), int(death_cross_w1),
            div_rsi, div_macd, div_roc
        ]
    except Exception:
        return None


@st.cache_data(show_spinner=True)
def compute_metrics_table(universe: dict, universe_signature: str, divergence_cfg: dict, divergence_signature: str) -> pd.DataFrame:
    _ = universe_signature
    _ = divergence_signature
    rows = []
    for group, subgroups in universe.items():
        for subgroup, tickers in subgroups.items():
            for ticker in tickers:
                res = get_metrics(ticker, divergence_cfg)
                if res is None:
                    rows.append([group, subgroup, ticker] + [np.nan] * 35)
                else:
                    rows.append([group, subgroup, ticker] + res)

    columns = [
        "Group", "Subgroup", "Ticker",
        "Perf_1W_%", "Perf_1M_%", "Perf_3M_%", "Perf_6M_%",
        "Perf_12M_%", "Perf_3Y_%", "Perf_5Y_%",
        "FundFlows_1M_%", "FundFlows_3M_%",
        "Price_vs_52W_High_%", "Price_vs_ATH_%", "RSI_14",
        "SMA50w_vs_SMA200w_Spread_%", "SMA50w_vs_SMA200w_Spread_Avg_36M_%", "SMA_Spread_%_Change_6M_%", "SMA_Trend",
        "ADX_14", "DI_Plus_14", "DI_Minus_14", "DI_Plus_14_Delta2", "DI_Minus_14_Delta2",
        "BB_Position", "BB_Mid", "BB_Upper", "BB_Lower", "BB_StepUp", "BB_StepDown", "WeeklyClose_Last",
        "GoldenCross_D1", "DeathCross_D1", "GoldenCross_W1", "DeathCross_W1",
        "Div_6M_vs_RSI", "Div_6M_vs_MACD", "Div_6M_vs_ROC",
    ]

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

    return df


def apply_filters(df: pd.DataFrame):
    groups_all = sorted(df["Group"].dropna().unique().tolist())
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns([1.1, 1.1, 1.1, 0.9, 1.0, 1.2, 1.2, 1.3, 1.3, 1.1])

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
        perf_top5_label = st.selectbox("Top 5 performing ETFs", options=list(PERF_TOP5_MAP.keys()), index=0)

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

    # Apply non-top-5 filters next
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

    # Top-5 filters last
    if perf_top5_label != "Off":
        metric = PERF_TOP5_MAP[perf_top5_label]
        filtered = filtered.sort_values(by=metric, ascending=False, na_position="last").head(5)

    flow_unavailable = False
    if flow_top5_enabled:
        flow_df = filtered.dropna(subset=["FundFlows_3M_%"])
        if flow_df.empty:
            filtered = filtered.iloc[0:0].copy()
            flow_unavailable = True
        else:
            filtered = flow_df.sort_values(by="FundFlows_3M_%", ascending=False).head(5)

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
    tester_url = st.secrets.get("TESTER_APP_URL", os.environ.get("TESTER_APP_URL", "")).strip()
    if not tester_url:
        st.info(
            "Tester app URL is not configured. "
            "Set `TESTER_APP_URL` in Streamlit app Secrets to embed Tester here."
        )
        st.code("TESTER_APP_URL = \"https://your-tester-app.streamlit.app\"")
        return
    st.caption(f"Tester mounted from {tester_url}")
    components.iframe(tester_url, height=2100, scrolling=True)


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


@st.cache_data(show_spinner=False)
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


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows to chart for current filters.")
        return

    chart_df = _build_chart_frame(df)

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


def main():
    st.set_page_config(page_title="ETF Market Screener", layout="wide")
    if "universe_map" not in st.session_state:
        st.session_state["universe_map"] = load_universe_map()
    universe_map = st.session_state["universe_map"]

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.2rem !important;
            padding-bottom: 0.35rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            max-width: 100% !important;
        }
        [data-testid="stHeader"] {
            height: 0rem;
        }
        [data-testid="stToolbar"] {
            top: 0.15rem;
            right: 0.35rem;
        }
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        .stButton > button {
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

    top_left, top_mid, top_refresh_col, top_hard_refresh_col = st.columns([2, 8, 1, 1.4])
    with top_left:
        selected_universe_name = st.selectbox("ETF Version", options=list(universe_map.keys()), index=0)
    with top_refresh_col:
        refresh = st.button("Refresh", use_container_width=True)
    with top_hard_refresh_col:
        hard_refresh = st.button("Hard Refresh Data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()
    if hard_refresh:
        st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
        st.rerun()

    divergence_cfg = render_divergence_settings()
    divergence_signature = (
        f"profile={divergence_cfg['profile']}:"
        f"pv={divergence_cfg['pivot_window']}:"
        f"lb={divergence_cfg['lookback_bars']}"
    )

    with st.spinner("Computing ETF metrics..."):
        selected_universe = universe_map[selected_universe_name]
        df = compute_metrics_table(
            selected_universe,
            f"{selected_universe_name}:{str(selected_universe)}",
            divergence_cfg,
            divergence_signature,
        )

    filtered_df, flow_unavailable = apply_filters(df)
    graph_ordered_df = filtered_df.copy()

    st.caption(f"Rows: {len(filtered_df)}/{len(df)}")

    if flow_unavailable:
        st.warning("Fund flow data unavailable")

    table_tab, charts_tab, graphs_tab, inputs_tab, tester_tab = st.tabs(["Table", "Charts", "Graphs", "Inputs", "Tester"])
    with table_tab:
        table_df = filtered_df.copy().reset_index(drop=True)
        table_df["__row_id__"] = np.arange(len(table_df))
        table_display_df = table_df[["__row_id__"] + DISPLAY_COLUMNS].copy()
        table_col_labels = {
            col: TABLE_HEADER_NAMES.get(col, col.replace("_", " "))
            for col in DISPLAY_COLUMNS
        }
        table_display_df = table_display_df.rename(columns=table_col_labels)

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
            "Price_vs_52W_High_%": 130,
            "Price_vs_ATH_%": 110,
            "RSI_14": 39,
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
                    const r = Math.round(248 + t * (74 - 248));
                    const g = Math.round(113 + t * (222 - 113));
                    const b = Math.round(113 + t * (128 - 113));
                    return {{backgroundColor: `rgb(${{r}}, ${{g}}, ${{b}})`, color: "#111827"}};
                }}
                """
            )
            gb.configure_column(col_label, cellStyle=perf_color_styles[col_label])

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
    with charts_tab:
        render_charts(filtered_df)
    with graphs_tab:
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
    with inputs_tab:
        render_inputs_tab(universe_map)
    with tester_tab:
        render_tester_tab()


if __name__ == "__main__":
    main()

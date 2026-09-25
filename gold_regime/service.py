from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from finance_core import drop_incomplete_daily_bar
from fred_client import download_fred_series_batch

from .config import GOLD_REGIME_CONFIG
from .cot import calculate_cot_momentum_score, download_cftc_cot, extract_comex_gold_cot, load_comex_gold_cot_from_positioning
from .etf_flows import aggregate_gold_etf_flows, load_gold_etf_flows
from .macro import calculate_gold_macro_from_fred
from .models import Freshness, GoldRegimeSnapshot
from .regime import (
    additional_structural_demand_context,
    apply_gold_regime_history,
    determine_flow_flags,
    generate_gold_regime_explanation,
)
from .utils import freshness_from_date, weekly_close


YAHOO_GOLD_REGIME_TICKERS = ["GLD", "DX-Y.NYB", "CL=F"]
FRED_GOLD_REGIME_SERIES = ["DFII10", "DGS2"]


def build_gold_regime_snapshot(
    gold_alpha: float | None,
    fred_api_key: str | None = None,
    config: dict | None = None,
    cache_dir: Path | None = None,
) -> GoldRegimeSnapshot:
    cfg = config or GOLD_REGIME_CONFIG
    cache = cache_dir or Path("persistent") / "finance_cache" / "gold_regime"
    cache.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    start_date = date(2016, 1, 1)

    yahoo_weekly = load_yahoo_weekly(YAHOO_GOLD_REGIME_TICKERS)
    try:
        fred_data = download_fred_series_batch(
            FRED_GOLD_REGIME_SERIES,
            api_key=fred_api_key,
            observation_start="2010-01-01",
        )
    except Exception:
        fred_data = pd.DataFrame(columns=["Series_ID", "Date", "Value"])

    macro_history = calculate_gold_macro_from_fred(
        dxy=weekly_close(yahoo_weekly.get("DX-Y.NYB", pd.DataFrame())),
        wti=weekly_close(yahoo_weekly.get("CL=F", pd.DataFrame())),
        fred_data=fred_data,
        config=cfg,
    )
    gold_price = weekly_close(yahoo_weekly.get("GLD", pd.DataFrame())).rename("gold_price")
    price_history = pd.DataFrame({"date": gold_price.index, "gold_price": gold_price.values})

    etf_daily, available_etfs, unavailable_etfs = load_gold_etf_flows(start_date, today, cfg)
    etf_history = aggregate_gold_etf_flows(etf_daily, cfg["etf_tickers"], cfg)

    try:
        cot_history, contract_name = load_comex_gold_cot_from_positioning()
        if cot_history.empty:
            cot_raw = download_cftc_cot(cfg, cache)
            cot_frame, contract_name = extract_comex_gold_cot(cot_raw, cache)
            cot_history = calculate_cot_momentum_score(cot_frame, cfg)
    except Exception:
        contract_name = None
        cot_history = pd.DataFrame()

    history = merge_gold_histories(price_history, macro_history, etf_history, cot_history)
    history = carry_forward_cot_history(history)
    if not history.empty and "date" in history.columns:
        history["long_liquidity_cycle"] = history["date"].map(long_liquidity_cycle_phase)
    history = apply_gold_regime_history(history, gold_alpha)
    history = history.loc[(history["date"] >= pd.Timestamp("2016-01-01")) & (history["date"] <= pd.Timestamp(today))]
    history = history.sort_values("date").reset_index(drop=True)
    current = latest_gold_regime_row(history)
    enrich_current(current, cfg)
    freshness = calculate_freshness(current, history, cfg)
    if "Structural Demand" in freshness:
        current["structural_demand_status"] = freshness["Structural Demand"].status
    if "Long Liquidity Cycle" in freshness:
        current["long_liquidity_cycle_status"] = freshness["Long Liquidity Cycle"].status
    current["explanation"] = generate_gold_regime_explanation(current)
    current["additional_structural_demand_context"] = additional_structural_demand_context(
        current.get("gold_alpha", np.nan),
        current.get("forward_macro_risk", np.nan),
        current.get("structural_demand_score"),
    )

    return GoldRegimeSnapshot(
        current=current,
        history=history,
        etf_unavailable_tickers=unavailable_etfs,
        etf_available_tickers=available_etfs,
        cot_contract_market_name=contract_name,
        freshness=freshness,
    )


def load_yahoo_weekly(tickers: list[str]) -> dict[str, pd.DataFrame]:
    try:
        px = yf.download(
            tickers,
            period="max",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        px = pd.DataFrame()

    out = {}
    for ticker in tickers:
        daily = drop_incomplete_daily_bar(extract_ohlcv(px, ticker))
        out[ticker] = build_weekly_ohlcv(daily) if not daily.empty else pd.DataFrame()
    return out


def extract_ohlcv(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if px is None or px.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    try:
        if isinstance(px.columns, pd.MultiIndex):
            if ticker in px.columns.get_level_values(0):
                frame = px[ticker].copy()
            elif ticker in px.columns.get_level_values(1):
                frame = px.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        else:
            frame = px.copy()
        keep = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in frame.columns]
        frame = frame[keep].copy()
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def build_weekly_ohlcv(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    values = daily.copy()
    values.index = pd.to_datetime(values.index)
    weekly = values.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    weekly = weekly.loc[weekly.index <= today]
    if not weekly.empty and pd.Timestamp(weekly.index[-1]) > pd.Timestamp(values.index[-1]).normalize():
        weekly = weekly.iloc[:-1]
    return weekly.dropna(subset=["Close"])


def merge_gold_histories(*frames: pd.DataFrame) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        current = frame.copy()
        current["date"] = pd.to_datetime(current["date"], errors="coerce")
        current = current.dropna(subset=["date"]).sort_values("date")
        merged = current if merged is None else pd.merge(merged, current, on="date", how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("date").reset_index(drop=True)


def latest_gold_regime_row(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {}
    useful = history.dropna(subset=["gold_regime"]) if "gold_regime" in history.columns else history
    if useful.empty:
        useful = history
    row = useful.sort_values("date").iloc[-1]
    return {key: row[key] for key in useful.columns}


def carry_forward_cot_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    cot_columns = [column for column in history.columns if column.startswith("cot_")]
    if not cot_columns:
        return history
    out = history.sort_values("date").copy()
    out[cot_columns] = out[cot_columns].ffill()
    return out


def enrich_current(current: dict[str, Any], cfg: dict) -> None:
    demand = cfg.get("structural_demand", {})
    current["structural_demand_score"] = demand.get("score")
    current["monetary_demand_share"] = demand.get("monetary_demand_share")
    current["central_bank_4q_purchases"] = demand.get("central_bank_4q_purchases")
    current["demand_rotation_yoy"] = demand.get("demand_rotation_yoy")
    current["structural_demand_last_updated"] = demand.get("last_updated")
    current["structural_demand_mode"] = "MANUAL / INFORMATIONAL"
    if "flow_state" not in current:
        current["flow_state"] = "PARTIAL_DATA"

    flags = determine_flow_flags(
        current.get("gold_alpha", np.nan),
        current.get("tactical_flow_score", np.nan),
        current.get("etf_flow_score", np.nan),
        current.get("cot_momentum_score", np.nan),
        current.get("structural_macro_score", np.nan),
        current.get("forward_macro_risk", np.nan),
        current.get("long_liquidity_cycle"),
    )
    current.update(flags)


def calculate_freshness(current: dict[str, Any], history: pd.DataFrame | None, cfg: dict) -> dict[str, Freshness]:
    limits = cfg["freshness_days"]
    fields = {
        "Market prices": ("date", "gold_price", limits["market_prices"]),
        "Gold macro series": ("date", "structural_macro_score", limits["fred_yields"]),
        "ETF flows": ("date", "etf_flow_1w", limits["etf_flows"]),
        "COT": ("cot_report_date", "cot_report_date", limits["cot"]),
    }
    out = {}
    for label, (date_field, data_field, stale_after) in fields.items():
        value = latest_available_date(current, history, date_field, data_field)
        last_updated, age, status = freshness_from_date(value, stale_after)
        out[label] = Freshness(last_updated=last_updated, data_age_days=age, status=status)

    demand_date = current.get("structural_demand_last_updated")
    if demand_date is None or pd.isna(demand_date):
        out["Structural Demand"] = Freshness(last_updated=None, data_age_days=None, status="NOT_CONFIGURED")
    else:
        last_updated, age, status = freshness_from_date(demand_date, limits["structural_demand"])
        out["Structural Demand"] = Freshness(last_updated=last_updated, data_age_days=age, status=status)
    cycle_date = latest_available_date(current, history, "date", "long_liquidity_cycle")
    last_updated, age, status = freshness_from_date(cycle_date, stale_after_days=7)
    out["Long Liquidity Cycle"] = Freshness(last_updated=last_updated, data_age_days=age, status=status)
    return out


def latest_available_date(current: dict[str, Any], history: pd.DataFrame | None, date_field: str, data_field: str) -> Any:
    current_value = current.get(date_field)
    current_data = current.get(data_field)
    if current_value is not None and not pd.isna(current_value) and current_data is not None and not pd.isna(current_data):
        return current_value
    if history is None or history.empty or date_field not in history.columns or data_field not in history.columns:
        return current_value
    available = history.dropna(subset=[date_field, data_field]).sort_values("date" if "date" in history.columns else date_field)
    if available.empty:
        return current_value
    return available.iloc[-1][date_field]


def long_liquidity_cycle_phase(value: Any) -> str:
    if value is None or pd.isna(value):
        return "DATA_INCOMPLETE"
    date_value = pd.Timestamp(value)
    anchor = pd.Timestamp("2022-10-01")
    months = (date_value.year - anchor.year) * 12 + (date_value.month - anchor.month) + (date_value.day - 1) / 30.4375
    phase_pos = months % 65.0
    if phase_pos < 16.25:
        return "RECOVERY_REACCELERATION"
    if phase_pos < 32.50:
        return "ACCELERATING_EXPANSION"
    if phase_pos < 48.75:
        return "DECELERATING_EXPANSION"
    return "CONTRACTION"

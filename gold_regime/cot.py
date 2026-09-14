from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GOLD_REGIME_CONFIG
from .utils import cache_file_is_fresh, classify_score, rolling_percentile_rank


COT_MOMENTUM_LABELS = ("STRONG_BULLISH_COT", "BULLISH_COT", "NEUTRAL", "BEARISH_COT", "STRONG_BEARISH_COT")


def download_cftc_cot(config: dict | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    cfg = (config or GOLD_REGIME_CONFIG)["cot"]
    directory = cache_dir or Path("persistent") / "finance_cache" / "gold_regime"
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / "cftc_disaggregated_futures_only.csv"
    if cache_file_is_fresh(cache_path, int(cfg["cache_ttl_seconds"])):
        return pd.read_csv(cache_path)
    try:
        frame = pd.read_csv(str(cfg["url"]), low_memory=False)
        frame.to_csv(cache_path, index=False)
        return frame
    except Exception:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise


def extract_comex_gold_cot(cot_data: pd.DataFrame, cache_dir: Path | None = None) -> tuple[pd.DataFrame, str | None]:
    if cot_data.empty:
        return pd.DataFrame(columns=cot_columns()), None
    frame = normalize_cot_columns(cot_data)
    required = {
        "report_date_as_yyyy_mm_dd",
        "market_and_exchange_names",
        "contract_market_name",
        "commodity_name",
        "open_interest_all",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
    }
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=cot_columns()), None

    directory = cache_dir or Path("persistent") / "finance_cache" / "gold_regime"
    directory.mkdir(parents=True, exist_ok=True)
    identifier_path = directory / "cftc_comex_gold_contract.txt"
    saved_identifier = identifier_path.read_text().strip() if identifier_path.exists() else ""
    if saved_identifier:
        selected = frame[frame["contract_market_name"].astype(str) == saved_identifier].copy()
    else:
        selected = infer_comex_gold_contract(frame)
        if not selected.empty:
            identifier_path.write_text(str(selected["contract_market_name"].iloc[0]))

    if selected.empty:
        return pd.DataFrame(columns=cot_columns()), None

    report_dates = parse_cot_report_dates(selected["report_date_as_yyyy_mm_dd"])
    out = pd.DataFrame(
        {
            "report_date": report_dates,
            "market_and_exchange_names": selected["market_and_exchange_names"].astype(str),
            "contract_market_name": selected["contract_market_name"].astype(str),
            "commodity_name": selected["commodity_name"].astype(str),
            "cot_open_interest": parse_numeric(selected["open_interest_all"]),
            "cot_mm_long": parse_numeric(selected["m_money_positions_long_all"]),
            "cot_mm_short": parse_numeric(selected["m_money_positions_short_all"]),
            "cot_mm_spread": parse_numeric(selected.get("m_money_positions_spread_all", pd.Series(index=selected.index, dtype="float64"))),
        }
    )
    out = out.dropna(subset=["report_date", "cot_open_interest", "cot_mm_long", "cot_mm_short"]).sort_values("report_date")
    out["cot_report_date"] = out["report_date"].dt.date
    out["date"] = out["report_date"] + pd.Timedelta(days=3)
    out["date"] = out["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    out["cot_mm_net"] = out["cot_mm_long"] - out["cot_mm_short"]
    out["cot_mm_net_pct_oi"] = out["cot_mm_net"] / out["cot_open_interest"].replace(0.0, np.nan)
    contract = str(out["contract_market_name"].iloc[-1]) if not out.empty else None
    return out[cot_columns()], contract


def infer_comex_gold_contract(frame: pd.DataFrame) -> pd.DataFrame:
    commodity = frame["commodity_name"].astype(str).str.upper()
    market = frame["market_and_exchange_names"].astype(str).str.upper()
    contract = frame["contract_market_name"].astype(str).str.upper()
    mask = (
        commodity.eq("GOLD")
        & market.str.contains("COMMODITY EXCHANGE", na=False)
        & contract.str.contains("GOLD", na=False)
        & ~contract.str.contains("MICRO|MINI|OPTION", na=False)
    )
    selected = frame.loc[mask].copy()
    if selected.empty:
        selected = frame.loc[
            commodity.eq("GOLD") & contract.str.contains("GOLD", na=False) & ~contract.str.contains("MICRO|MINI|OPTION", na=False)
        ].copy()
    return selected


def calculate_cot_momentum_score(cot_frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or GOLD_REGIME_CONFIG
    percentile_cfg = cfg["percentile"]
    cot_window = int(cfg["tactical_flow"]["cot_change_window_weeks"])
    percentile_window = int(percentile_cfg["window_weeks"])
    percentile_min = int(percentile_cfg["minimum_weeks"])
    if cot_frame.empty:
        return pd.DataFrame(columns=cot_score_columns())

    out = cot_frame.copy().sort_values("date")
    out["cot_change_4w"] = out["cot_mm_net_pct_oi"] - out["cot_mm_net_pct_oi"].shift(cot_window)
    out["cot_momentum_score"] = rolling_percentile_rank(out["cot_change_4w"], percentile_window, percentile_min)
    out["cot_momentum_state"] = out["cot_momentum_score"].map(classify_cot_momentum_state)
    out["cot_mm_net_pct_oi_percentile"] = rolling_percentile_rank(out["cot_mm_net_pct_oi"], percentile_window, percentile_min)
    out["cot_net_position_state"] = out["cot_mm_net_pct_oi_percentile"].map(classify_cot_position_state)
    return out[cot_score_columns()]


def classify_cot_momentum_state(score: float) -> str:
    return classify_score(float(score), COT_MOMENTUM_LABELS) if np.isfinite(score) else "DATA_INCOMPLETE"


def classify_cot_position_state(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "DATA_INCOMPLETE"
    if percentile >= 90.0:
        return "EXTREME_LONG"
    if percentile >= 75.0:
        return "HIGH"
    if percentile <= 10.0:
        return "EXTREME_SHORT"
    if percentile <= 25.0:
        return "LOW"
    return "NORMAL"


def normalize_cot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower().replace(" ", "_").replace("-", "_") for column in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def parse_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")


def parse_cot_report_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y %b %d %I:%M:%S %p", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def cot_columns() -> list[str]:
    return [
        "date",
        "cot_report_date",
        "market_and_exchange_names",
        "contract_market_name",
        "commodity_name",
        "cot_open_interest",
        "cot_mm_long",
        "cot_mm_short",
        "cot_mm_spread",
        "cot_mm_net",
        "cot_mm_net_pct_oi",
    ]


def cot_score_columns() -> list[str]:
    return cot_columns() + [
        "cot_mm_net_pct_oi_percentile",
        "cot_net_position_state",
        "cot_change_4w",
        "cot_momentum_score",
        "cot_momentum_state",
    ]

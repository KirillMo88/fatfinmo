from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from fund_flows import FundFlowObservation, fetch_etf_com_fund_flow_history

from .config import GOLD_REGIME_CONFIG
from .utils import classify_score, rolling_percentile_rank


ETF_FLOW_LABELS = ("STRONG_INFLOW", "INFLOW", "NEUTRAL", "OUTFLOW", "STRONG_OUTFLOW")


def load_gold_etf_flows(
    start_date: date,
    end_date: date | None = None,
    config: dict | None = None,
    fetcher: Callable[[str, date, date], list[FundFlowObservation]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cfg = config or GOLD_REGIME_CONFIG
    tickers = list(cfg["etf_tickers"])
    end = end_date or datetime.now(timezone.utc).date()
    fetch = fetcher or fetch_etf_com_fund_flow_history
    frames: list[pd.DataFrame] = []
    available: list[str] = []
    unavailable: list[str] = []

    for ticker in tickers:
        try:
            observations = fetch(ticker, start_date, end)
        except Exception:
            observations = []
        if not observations:
            unavailable.append(ticker)
            continue
        available.append(ticker)
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime([obs.date for obs in observations]),
                    "ticker": ticker,
                    "net_flow": [obs.net_flow for obs in observations],
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "net_flow"]), available, unavailable
    return pd.concat(frames, ignore_index=True), available, unavailable


def aggregate_gold_etf_flows(
    daily_flows: pd.DataFrame,
    etf_tickers: Iterable[str],
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = config or GOLD_REGIME_CONFIG
    flow_window = int(cfg["tactical_flow"]["etf_flow_window_weeks"])
    percentile_window = int(cfg["percentile"]["window_weeks"])
    percentile_min = int(cfg["percentile"]["minimum_weeks"])
    required_coverage = 5

    columns = [
        "date",
        "etf_flow_1w",
        "etf_flow_4w",
        "etf_flow_13w",
        "etf_flow_intensity_4w",
        "etf_flow_score",
        "etf_flow_state",
        "etf_coverage_count",
        "etf_coverage_total",
        "etf_available_tickers",
        "etf_unavailable_tickers",
    ]
    if daily_flows.empty:
        return pd.DataFrame(columns=columns)

    frame = daily_flows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["net_flow"] = pd.to_numeric(frame["net_flow"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.dropna(subset=["date", "net_flow"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    weekly_by_ticker = (
        frame.set_index("date")
        .groupby("ticker")
        .resample("W-FRI")["net_flow"]
        .sum(min_count=1)
        .dropna()
        .reset_index()
    )
    weekly = (
        weekly_by_ticker.groupby("date")
        .agg(
            etf_flow_1w=("net_flow", "sum"),
            etf_coverage_count=("ticker", "nunique"),
            etf_available_tickers=("ticker", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_index()
        .reset_index()
    )
    requested = {str(ticker).upper() for ticker in etf_tickers}
    weekly["etf_coverage_total"] = len(requested)
    weekly["etf_unavailable_tickers"] = weekly["etf_available_tickers"].map(
        lambda value: ",".join(sorted(requested - set(str(value).split(",")))) if value else ",".join(sorted(requested))
    )
    weekly["etf_flow_4w"] = weekly["etf_flow_1w"].rolling(flow_window, min_periods=flow_window).sum()
    weekly["etf_flow_13w"] = weekly["etf_flow_1w"].rolling(13, min_periods=13).sum()

    flow_scale = weekly["etf_flow_1w"].abs().rolling(percentile_window, min_periods=percentile_min).median()
    weekly["etf_flow_intensity_4w"] = weekly["etf_flow_4w"] / (flow_window * flow_scale.replace(0.0, np.nan))
    weekly["etf_flow_score"] = rolling_percentile_rank(
        weekly["etf_flow_intensity_4w"],
        percentile_window,
        percentile_min,
    )
    weekly.loc[weekly["etf_coverage_count"] < required_coverage, "etf_flow_score"] = np.nan
    weekly["etf_flow_state"] = weekly["etf_flow_score"].map(classify_etf_flow_state)
    return weekly[columns]


def classify_etf_flow_state(score: float) -> str:
    return classify_score(float(score), ETF_FLOW_LABELS) if np.isfinite(score) else "DATA_INCOMPLETE"

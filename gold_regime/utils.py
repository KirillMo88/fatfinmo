from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def rolling_percentile_rank(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def percentile(window_values: np.ndarray) -> float:
        current = window_values[-1]
        if not np.isfinite(current):
            return np.nan
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods:
            return np.nan
        less = np.sum(clean < current)
        equal = np.sum(clean == current)
        return float(((less + 0.5 * equal) / len(clean)) * 100.0)

    return numeric.rolling(window=window, min_periods=min_periods).apply(percentile, raw=True)


def union_index(series_list: Iterable[pd.Series]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for series in series_list:
        if series is None or series.empty:
            continue
        index = index.union(pd.DatetimeIndex(pd.to_datetime(series.dropna().index)))
    return index.sort_values()


def weekly_close(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame["Close"], errors="coerce").dropna().sort_index()


def fred_series_weekly(fred_data: pd.DataFrame | None) -> dict[str, pd.Series]:
    if fred_data is None or fred_data.empty:
        return {}
    frame = fred_data.copy()
    frame["Series_ID"] = frame["Series_ID"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    out = {}
    for series_id, group in frame.dropna(subset=["Date"]).groupby("Series_ID"):
        values = group.sort_values("Date").set_index("Date")["Value"].dropna()
        if not values.empty:
            out[str(series_id)] = values.resample("W-FRI").last().ffill().dropna()
    return out


def classify_score(score: float, labels: tuple[str, str, str, str, str]) -> str:
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score >= 80.0:
        return labels[0]
    if score >= 60.0:
        return labels[1]
    if score >= 40.0:
        return labels[2]
    if score >= 20.0:
        return labels[3]
    return labels[4]


def classify_risk(score: float) -> str:
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score <= 20.0:
        return "LOW"
    if score <= 40.0:
        return "MODERATE"
    if score <= 60.0:
        return "ELEVATED"
    if score <= 80.0:
        return "HIGH"
    return "EXTREME"


def cache_file_is_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - modified
    return age.total_seconds() <= ttl_seconds


def freshness_from_date(value: date | pd.Timestamp | None, stale_after_days: int, delayed_after_days: int | None = None) -> tuple[date | None, int | None, str]:
    if value is None or pd.isna(value):
        return None, None, "ERROR"
    current = pd.Timestamp.now(tz="UTC").date()
    last_updated = pd.Timestamp(value).date()
    age = max((current - last_updated).days, 0)
    delayed = stale_after_days if delayed_after_days is None else delayed_after_days
    if age > stale_after_days:
        status = "STALE"
    elif age > delayed:
        status = "DELAYED"
    else:
        status = "CURRENT"
    return last_updated, age, status

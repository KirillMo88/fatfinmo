from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd


HY_OAS_COLUMNS = ["Date", "HY_OAS", "HYOASSourceFrequency"]
HY_OAS_SOURCE_PRIORITY = {
    "WEEKLY_ARCHIVE": 1,
    "DAILY_TRADINGVIEW": 2,
    "DAILY_FRED": 3,
}
CREDIT_ARCHIVE_START = pd.Timestamp("2015-01-01")
CREDIT_ARCHIVE_END = pd.Timestamp("2025-12-31")
CREDIT_ARCHIVE_PATH = Path(__file__).resolve().parent / "data" / "credit" / "credit_oas_weekly_2015_2025.csv"


def frozen_credit_archive_frame(series_id: str | None = None) -> pd.DataFrame:
    """Read the immutable 2015–2025 weekly credit archive.

    The archive is deliberately separate from the refreshable runtime cache so
    current-source outages cannot rewrite the historical model inputs.
    """
    columns = ["Series_ID", "Date", "Value"]
    try:
        if not CREDIT_ARCHIVE_PATH.exists():
            return pd.DataFrame(columns=columns)
        frame = pd.read_csv(CREDIT_ARCHIVE_PATH, usecols=columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)
    frame["Series_ID"] = frame["Series_ID"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["Series_ID", "Date", "Value"])
    frame = frame.loc[frame["Date"].between(CREDIT_ARCHIVE_START, CREDIT_ARCHIVE_END)]
    if series_id is not None:
        frame = frame.loc[frame["Series_ID"].eq(str(series_id).upper())]
    return frame.sort_values(["Series_ID", "Date"]).reset_index(drop=True)


def restrict_credit_live_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only observations after the frozen archive cutoff."""
    if frame is None or frame.empty or "Date" not in frame:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(columns=HY_OAS_COLUMNS)
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return result.loc[result["Date"].gt(CREDIT_ARCHIVE_END)].reset_index(drop=True)


def weekly_archive_available_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Expose Monday-labelled weekly bars only after that week has completed."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=HY_OAS_COLUMNS)
    observed_date = pd.to_datetime(raw.get("time"), errors="coerce").dt.tz_localize(None).dt.normalize()
    value = pd.to_numeric(raw.get("close"), errors="coerce")
    frame = pd.DataFrame(
        {
            # Archive bars are labelled with the start of the weekly period.
            # The value becomes available on the following Monday.
            "Date": observed_date + pd.Timedelta(days=7),
            "HY_OAS": value,
            "HYOASSourceFrequency": "WEEKLY_ARCHIVE",
        }
    )
    return frame.dropna(subset=["Date", "HY_OAS"]).sort_values("Date").reset_index(drop=True)


def combine_hy_oas_sources(
    weekly_archive: pd.DataFrame,
    tradingview_daily: pd.DataFrame,
    fred_daily: pd.DataFrame,
    end_date: Any = None,
) -> pd.DataFrame:
    """Combine point-in-time observations with deterministic daily-source priority."""
    parts: list[pd.DataFrame] = []
    for frame in (weekly_archive, tradingview_daily, fred_daily):
        if frame is None or frame.empty:
            continue
        part = frame.copy()
        for column in HY_OAS_COLUMNS:
            if column not in part:
                part[column] = pd.NA
        part["Date"] = pd.to_datetime(part["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        part["HY_OAS"] = pd.to_numeric(part["HY_OAS"], errors="coerce")
        part["_source_priority"] = part["HYOASSourceFrequency"].map(HY_OAS_SOURCE_PRIORITY).fillna(0)
        parts.append(part[HY_OAS_COLUMNS + ["_source_priority"]].dropna(subset=["Date", "HY_OAS"]))
    if not parts:
        return pd.DataFrame(columns=HY_OAS_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["Date", "_source_priority"])
    combined = combined.drop_duplicates("Date", keep="last").sort_values("Date")
    if end_date is not None and pd.notna(end_date):
        end = pd.Timestamp(end_date).tz_localize(None).normalize()
        combined = combined.loc[combined["Date"].le(end)]
    return combined[HY_OAS_COLUMNS].reset_index(drop=True)

from __future__ import annotations

from typing import Any

import pandas as pd


HY_OAS_COLUMNS = ["Date", "HY_OAS", "HYOASSourceFrequency"]
HY_OAS_SOURCE_PRIORITY = {
    "WEEKLY_ARCHIVE": 1,
    "DAILY_FRED": 2,
    "DAILY_TRADINGVIEW": 3,
}


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

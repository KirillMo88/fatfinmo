from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SPX_SEASONALITY_START_YEAR = 2010
SPX_SEASONALITY_MODEL_WEEKS = 52
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class SPXSeasonality:
    weekly_model: pd.DataFrame
    monthly_statistics: pd.DataFrame
    current_year_actual: pd.DataFrame
    metadata: dict[str, Any]


def _clean_price_frame(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame is None or frame.empty or not {"Date", value_column}.issubset(frame.columns):
        return pd.DataFrame(columns=["Date", value_column])
    result = frame[["Date", value_column]].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    return (
        result.dropna(subset=["Date", value_column])
        .loc[lambda data: data[value_column] > 0]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def _year_position(dates: pd.Series, year: int) -> np.ndarray:
    dates = pd.to_datetime(dates)
    days_in_year = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    return ((dates.dt.dayofyear.to_numpy(dtype=float) - 1.0) / days_in_year) * (SPX_SEASONALITY_MODEL_WEEKS - 1)


def _expected_last_market_session(year: int) -> pd.Timestamp:
    expected = pd.Timestamp(year=year, month=12, day=31)
    while expected.weekday() >= 5:
        expected -= pd.Timedelta(days=1)
    if pd.Timestamp(year=year + 1, month=1, day=1).weekday() == 5 and expected.day == 31:
        expected -= pd.offsets.BDay(1)
    return expected


def weekly_closes_from_daily(daily_spx: pd.DataFrame) -> pd.DataFrame:
    if daily_spx is None or daily_spx.empty or "SPX_Close" not in daily_spx:
        return pd.DataFrame(columns=["Date", "SPX_Close"])
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily_spx.index, errors="coerce", utc=True).tz_convert(None).normalize(),
            "SPX_Close": pd.to_numeric(daily_spx["SPX_Close"], errors="coerce").to_numpy(),
        }
    ).dropna(subset=["Date", "SPX_Close"])
    if daily.empty:
        return pd.DataFrame(columns=["Date", "SPX_Close"])
    weeks = daily["Date"].dt.to_period("W-FRI")
    weekly = (
        daily.assign(_week=weeks)
        .groupby("_week", sort=True, as_index=False)
        .tail(1)[["Date", "SPX_Close"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    latest_date = daily["Date"].max()
    latest_week = latest_date.to_period("W-FRI")
    week_end = latest_week.end_time.normalize()
    final_year_close = (
        latest_date == _expected_last_market_session(latest_date.year)
        and latest_date.month == 12
    )
    if week_end > latest_date and not final_year_close:
        weekly = weekly.loc[weekly["Date"].dt.to_period("W-FRI").ne(latest_week)].reset_index(drop=True)
    return weekly


def monthly_closes_from_daily(daily_spx: pd.DataFrame) -> pd.DataFrame:
    if daily_spx is None or daily_spx.empty or "SPX_Close" not in daily_spx:
        return pd.DataFrame(columns=["Date", "SPX_Close"])
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily_spx.index, errors="coerce", utc=True).tz_convert(None).normalize(),
            "SPX_Close": pd.to_numeric(daily_spx["SPX_Close"], errors="coerce").to_numpy(),
        }
    ).dropna(subset=["Date", "SPX_Close"])
    if daily.empty:
        return pd.DataFrame(columns=["Date", "SPX_Close"])
    monthly = (
        daily.assign(_month=daily["Date"].dt.to_period("M"))
        .groupby("_month", sort=True, as_index=False)
        .tail(1)
        .copy()
    )
    monthly["Date"] = monthly["_month"].dt.to_timestamp(how="end").dt.normalize()
    return monthly[["Date", "SPX_Close"]].sort_values("Date").reset_index(drop=True)


def _interpolate_year(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    coordinates = np.linspace(0.0, SPX_SEASONALITY_MODEL_WEEKS - 1.0, len(numeric))
    valid = np.isfinite(numeric) & np.isfinite(coordinates)
    if valid.sum() == 0:
        return np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)
    order = np.argsort(coordinates[valid])
    return np.interp(
        np.arange(SPX_SEASONALITY_MODEL_WEEKS, dtype=float),
        coordinates[valid][order],
        numeric[valid][order],
        left=np.nan,
        right=np.nan,
    )


def _completed_year_end(
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    latest_observation_date: pd.Timestamp,
) -> int:
    latest_year = int(latest_observation_date.year)
    completed_end = latest_year - 1
    expected_last_session = _expected_last_market_session(latest_year)
    december_bar_present = monthly["Date"].dt.year.eq(latest_year) & monthly["Date"].dt.month.eq(12)
    if latest_observation_date >= expected_last_session and december_bar_present.any():
        latest_weekly_date = weekly.loc[weekly["Date"].dt.year.eq(latest_year), "Date"].max()
        if pd.notna(latest_weekly_date) and latest_weekly_date >= expected_last_session:
            completed_end = latest_year
    return completed_end


def calculate_spx_seasonality(
    weekly_spx: pd.DataFrame,
    monthly_spx: pd.DataFrame,
    latest_observation_date: pd.Timestamp | str | None = None,
    historical_start_year: int = SPX_SEASONALITY_START_YEAR,
) -> SPXSeasonality:
    weekly = _clean_price_frame(weekly_spx, "SPX_Close")
    monthly = _clean_price_frame(monthly_spx, "SPX_Close")
    if weekly.empty:
        empty_weekly = pd.DataFrame(
            columns=["Model Week", "Approx Month", "Mean Weekly Return", "Median Weekly Return", "Mean Cycle", "Median Cycle", "P25 Level", "P75 Level"]
        )
        empty_monthly = pd.DataFrame(
            columns=["Month", "Average Monthly Return", "Median Monthly Return", "P25 Monthly Return", "P75 Monthly Return", "Observation Count"]
        )
        return SPXSeasonality(empty_weekly, empty_monthly, pd.DataFrame(columns=["Model Week", "Date", "Actual Level"]), {"Status": "INSUFFICIENT_DATA"})

    latest_date = pd.to_datetime(latest_observation_date, errors="coerce", utc=True)
    if pd.isna(latest_date):
        latest_date = weekly["Date"].max()
    latest_date = latest_date.tz_convert(None).normalize()
    completed_end = _completed_year_end(weekly, monthly, latest_date)
    years = [year for year in range(historical_start_year, completed_end + 1) if weekly["Date"].dt.year.eq(year).any()]
    model_weeks = np.arange(1, SPX_SEASONALITY_MODEL_WEEKS + 1)
    model_coordinates = np.arange(SPX_SEASONALITY_MODEL_WEEKS, dtype=float)
    weekly["Weekly Return"] = weekly["SPX_Close"].pct_change(fill_method=None)

    annual_returns: list[np.ndarray] = []
    annual_levels: list[np.ndarray] = []
    for year in years:
        annual = weekly.loc[weekly["Date"].dt.year.eq(year)].copy()
        if annual.empty:
            continue
        annual_returns.append(_interpolate_year(annual["Weekly Return"]))
        normalized = annual["SPX_Close"] / annual["SPX_Close"].iloc[0] * 100.0
        annual_levels.append(_interpolate_year(normalized))

    if annual_returns:
        return_matrix = np.vstack(annual_returns)
        level_matrix = np.vstack(annual_levels)
        with np.errstate(invalid="ignore"):
            mean_returns = np.nanmean(return_matrix, axis=0)
            median_returns = np.nanmedian(return_matrix, axis=0)
            p25_levels = np.nanquantile(level_matrix, 0.25, axis=0)
            p75_levels = np.nanquantile(level_matrix, 0.75, axis=0)
        mean_returns[np.isfinite(mean_returns) & (mean_returns <= -1.0)] = np.nan
        median_returns[np.isfinite(median_returns) & (median_returns <= -1.0)] = np.nan
        mean_cycle = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)
        median_cycle = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)
        mean_cycle[0] = 100.0
        median_cycle[0] = 100.0
        for index in range(1, SPX_SEASONALITY_MODEL_WEEKS):
            mean_cycle[index] = mean_cycle[index - 1] * (1.0 + mean_returns[index]) if np.isfinite(mean_returns[index]) else np.nan
            median_cycle[index] = median_cycle[index - 1] * (1.0 + median_returns[index]) if np.isfinite(median_returns[index]) else np.nan
    else:
        mean_returns = median_returns = p25_levels = p75_levels = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)
        mean_cycle = median_cycle = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)

    approx_dates = pd.date_range("2001-01-01", periods=365, freq="D")
    approx_month = [MONTH_LABELS[approx_dates[min(364, int(round(position / 51.0 * 364)))].month - 1] for position in model_coordinates]
    weekly_model = pd.DataFrame(
        {
            "Model Week": model_weeks,
            "Approx Month": approx_month,
            "Mean Weekly Return": mean_returns,
            "Median Weekly Return": median_returns,
            "Mean Cycle": mean_cycle,
            "Median Cycle": median_cycle,
            "P25 Level": p25_levels,
            "P75 Level": p75_levels,
            "Model Position": model_coordinates + 1.0,
        }
    )

    actual_year = max(completed_end + 1, int(latest_date.year))
    actual = weekly.loc[weekly["Date"].dt.year.eq(actual_year)].copy()
    if not actual.empty:
        actual["Actual Level"] = actual["SPX_Close"] / actual["SPX_Close"].iloc[0] * 100.0
        actual["Model Week"] = _year_position(actual["Date"], actual_year) + 1.0
        actual = actual[["Model Week", "Date", "Actual Level"]].reset_index(drop=True)
        in_actual_range = (model_weeks >= actual["Model Week"].min()) & (model_weeks <= actual["Model Week"].max())
        actual_on_model_weeks = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)
        actual_on_model_weeks[in_actual_range] = np.interp(
            model_weeks[in_actual_range],
            actual["Model Week"],
            actual["Actual Level"],
        )
    else:
        actual = pd.DataFrame(columns=["Model Week", "Date", "Actual Level"])
        actual_on_model_weeks = np.full(SPX_SEASONALITY_MODEL_WEEKS, np.nan)

    weekly_model["Current Year Actual"] = actual_on_model_weeks

    monthly["Year"] = monthly["Date"].dt.year
    monthly["Month Number"] = monthly["Date"].dt.month
    monthly["Month Ordinal"] = monthly["Date"].dt.to_period("M").astype(int)
    monthly["Monthly Return"] = monthly["SPX_Close"] / monthly["SPX_Close"].shift(1) - 1.0
    monthly["Consecutive Month"] = monthly["Month Ordinal"].diff().eq(1)
    monthly_returns = monthly.loc[
        monthly["Year"].between(historical_start_year, completed_end) & monthly["Consecutive Month"]
    ].copy()
    monthly_rows = []
    for month_number, month_name in enumerate(MONTH_LABELS, start=1):
        sample = monthly_returns.loc[monthly_returns["Month Number"].eq(month_number), "Monthly Return"].dropna()
        monthly_rows.append(
            {
                "Month": month_name,
                "Month Number": month_number,
                "Average Monthly Return": sample.mean() if not sample.empty else np.nan,
                "Median Monthly Return": sample.median() if not sample.empty else np.nan,
                "P25 Monthly Return": sample.quantile(0.25) if not sample.empty else np.nan,
                "P75 Monthly Return": sample.quantile(0.75) if not sample.empty else np.nan,
                "Observation Count": int(sample.count()),
                "Historical Start Year": historical_start_year,
                "Historical End Year": completed_end,
            }
        )
    monthly_statistics = pd.DataFrame(monthly_rows)
    weekly_model["Historical Start Year"] = historical_start_year
    weekly_model["Historical End Year"] = completed_end
    weekly_model["Number of Historical Years"] = len(years)
    metadata = {
        "Status": "OK" if years else "INSUFFICIENT_HISTORY",
        "Historical Start Year": historical_start_year,
        "Historical End Year": completed_end,
        "Number of Historical Years": len(years),
        "Historical Years": years,
        "Current Year": actual_year,
        "Latest Observation": latest_date,
    }
    return SPXSeasonality(weekly_model, monthly_statistics, actual, metadata)

"""Point-in-time BTC macro-cycle model built from canonical Screener inputs."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


BTC_CYCLE_HORIZONS = {"3M": 3, "6M": 6, "9M": 9, "12M": 12}
BTC_ACTUAL_HALVINGS = pd.DatetimeIndex(
    ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
)
BTC_PROJECTED_HALVING = pd.Timestamp("2028-04-01")
BTC_LIQUIDITY_TROUGH_ANCHORS = pd.DatetimeIndex(
    ["2010-06-18", "2015-03-06", "2019-03-08", "2022-10-28"]
)
BTC_LIQUIDITY_CYCLE_MONTHS = 52.7
BTC_SECONDARY_PERCENTILE_WINDOW = 156
BTC_SECONDARY_PERCENTILE_MINIMUM = 104
BTC_RANGE_OPTIONS = ("1Y", "3Y", "5Y", "10Y", "MAX", "Next Cycle")
BTC_MODULAR_CYCLE_START = pd.Timestamp("2026-09-27")
BTC_MODULAR_HISTORICAL_MODULES = (
    (pd.Timestamp("2018-11-02"), pd.Timestamp("2022-11-25")),
    (pd.Timestamp("2022-09-27"), pd.Timestamp("2026-06-26")),
)

HALVING_PHASES = (
    "EARLY_EXPANSION",
    "LATE_EXPANSION",
    "CYCLE_TOP_RISK",
    "BEAR_DELEVERAGING",
    "BOTTOMING_TRANSITION",
    "ACCUMULATION_PRE_HALVING",
)
LIQUIDITY_PHASES = (
    "RECOVERY_ACCELERATION",
    "ACCELERATING_EXPANSION",
    "DECELERATING_EXPANSION",
    "ACCELERATING_CONTRACTION",
)

HALVING_BASES = {
    "CYCLE_TOP_RISK": {"3M": 15.0, "6M": 10.0, "9M": 10.0, "12M": 5.0},
    "BEAR_DELEVERAGING": {"3M": 20.0, "6M": 15.0, "9M": 15.0, "12M": 20.0},
    "BOTTOMING_TRANSITION": {"3M": 60.0, "6M": 70.0, "9M": 70.0, "12M": 75.0},
    "ACCUMULATION_PRE_HALVING": {"3M": 60.0, "6M": 65.0, "9M": 75.0, "12M": 90.0},
    "EARLY_EXPANSION": {"3M": 85.0, "6M": 90.0, "9M": 95.0, "12M": 90.0},
    "LATE_EXPANSION": {"3M": 45.0, "6M": 45.0, "9M": 30.0, "12M": 25.0},
}
LIQUIDITY_MODIFIERS = {
    "RECOVERY_ACCELERATION": 10.0,
    "ACCELERATING_EXPANSION": 20.0,
    "DECELERATING_EXPANSION": 0.0,
    "ACCELERATING_CONTRACTION": -20.0,
}
SECONDARY_WEIGHTS = {
    "3M": (0.60, 0.30, 0.10),
    "6M": (0.50, 0.40, 0.10),
    "9M": (0.40, 0.50, 0.10),
    "12M": (0.30, 0.60, 0.10),
}
SECONDARY_MODIFIER_LIMITS = {"3M": 10.0, "6M": 10.0, "9M": 5.0, "12M": 5.0}


def halving_phase(progress: float) -> str:
    if not np.isfinite(progress):
        return "DATA_INCOMPLETE"
    if progress < 20.0:
        return "EARLY_EXPANSION"
    if progress < 35.0:
        return "LATE_EXPANSION"
    if progress < 42.0:
        return "CYCLE_TOP_RISK"
    if progress < 55.0:
        return "BEAR_DELEVERAGING"
    if progress < 70.0:
        return "BOTTOMING_TRANSITION"
    return "ACCUMULATION_PRE_HALVING"


def halving_cycle_position(at: pd.Timestamp | str) -> dict[str, Any]:
    target = pd.Timestamp(at).normalize()
    schedule = list(BTC_ACTUAL_HALVINGS) + [BTC_PROJECTED_HALVING]
    if target >= BTC_PROJECTED_HALVING:
        next_projected = BTC_PROJECTED_HALVING
        while next_projected <= target:
            next_projected += pd.DateOffset(months=48)
        schedule.extend(pd.date_range(BTC_PROJECTED_HALVING, next_projected, freq=pd.DateOffset(months=48))[1:])
    schedule = pd.DatetimeIndex(sorted(set(pd.Timestamp(value).normalize() for value in schedule)))
    previous = schedule[schedule <= target]
    following = schedule[schedule > target]
    if previous.empty or following.empty:
        return {
            "progress": np.nan,
            "phase": "DATA_INCOMPLETE",
            "previous_halving": pd.NaT,
            "next_halving": pd.NaT,
            "next_halving_projected": False,
        }
    previous_date = previous[-1]
    next_date = following[0]
    progress = float((target - previous_date).days / (next_date - previous_date).days * 100.0)
    is_projected = next_date not in BTC_ACTUAL_HALVINGS
    return {
        "progress": progress,
        "phase": halving_phase(progress),
        "previous_halving": previous_date,
        "next_halving": next_date,
        "next_halving_projected": bool(is_projected),
    }


def add_fractional_months(value: pd.Timestamp | str, months: float) -> pd.Timestamp:
    days = int(round(months * 365.2425 / 12.0))
    return pd.Timestamp(value).normalize() + pd.Timedelta(days=days, unit="D")


def next_accumulation_pre_halving_start(as_of_date: pd.Timestamp | str) -> pd.Timestamp:
    """Return the start of the accumulation phase in the cycle after the next halving."""
    next_halving = halving_cycle_position(as_of_date)["next_halving"]
    following_halving = next_halving + pd.DateOffset(months=48)
    accumulation_start_days = int(np.ceil((following_halving - next_halving).days * 0.70))
    return (next_halving + pd.Timedelta(accumulation_start_days, unit="D")).normalize()


def btc_cycle_time_range(history: pd.DataFrame, choice: str) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    observed = history.loc[~history["Projected"].astype(bool), "Date"]
    first_date = pd.Timestamp(observed.min()).normalize()
    latest_date = pd.Timestamp(observed.max()).normalize()
    if choice == "Next Cycle":
        return latest_date - pd.DateOffset(months=12), next_accumulation_pre_halving_start(latest_date), True
    if choice == "MAX":
        return first_date, latest_date, False
    years = int(choice.removesuffix("Y"))
    return max(first_date, latest_date - pd.DateOffset(years=years)), latest_date, False


def liquidity_cycle_position(
    target_date: pd.Timestamp | str,
    as_of_date: pd.Timestamp | str,
) -> dict[str, Any]:
    """Project cycle position from the latest fixed trough known at as_of_date."""
    target = pd.Timestamp(target_date).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    known = BTC_LIQUIDITY_TROUGH_ANCHORS[BTC_LIQUIDITY_TROUGH_ANCHORS <= as_of]
    if known.empty:
        return {"anchor": pd.NaT, "progress": np.nan, "phase": "DATA_INCOMPLETE", "next_trough": pd.NaT}

    anchor = pd.Timestamp(known[-1])
    cycle_days = int(round(BTC_LIQUIDITY_CYCLE_MONTHS * 365.2425 / 12.0))
    elapsed_days = max(0, (target - anchor).days)
    elapsed_cycles = elapsed_days // cycle_days
    anchor += pd.Timedelta(elapsed_cycles * cycle_days, unit="D")
    next_trough = anchor + pd.Timedelta(cycle_days, unit="D")
    progress = float((target - anchor).total_seconds() / (next_trough - anchor).total_seconds() * 100.0)
    progress = min(max(progress, 0.0), 100.0)
    if progress < 25.0:
        phase = "RECOVERY_ACCELERATION"
    elif progress < 50.0:
        phase = "ACCELERATING_EXPANSION"
    elif progress < 75.0:
        phase = "DECELERATING_EXPANSION"
    else:
        phase = "ACCELERATING_CONTRACTION"
    return {"anchor": anchor, "progress": progress, "phase": phase, "next_trough": next_trough}


def _project_global_m2_cycle(
    source: pd.DataFrame,
    dates: pd.Series,
    next_trough: pd.Timestamp,
) -> pd.Series:
    values = pd.to_numeric(source["PrimaryMarketCycle"], errors="coerce").dropna()
    dates = pd.to_datetime(dates, errors="coerce")
    if values.empty or dates.empty:
        return pd.Series(np.nan, index=dates.index, dtype="float64")

    source_dates = pd.to_datetime(source.loc[values.index, "Date"], errors="coerce")
    last_date = pd.Timestamp(source_dates.iloc[-1]).normalize()
    last_value = float(values.iloc[-1])
    recent = pd.DataFrame({"Date": source_dates, "Value": values}).tail(7)
    amplitude = float(recent["Value"].abs().quantile(0.90))
    if not np.isfinite(amplitude) or amplitude < 0.25:
        amplitude = max(float(values.abs().quantile(0.90)), 1.0)

    period_days = BTC_LIQUIDITY_CYCLE_MONTHS * 365.2425 / 12.0
    trough = pd.Timestamp(next_trough).normalize()
    interval_days = max((trough - last_date).total_seconds() / 86400.0, 1.0)
    if len(recent) >= 2:
        slope = (float(recent["Value"].iloc[-1]) - float(recent["Value"].iloc[0])) / max(
            (pd.Timestamp(recent["Date"].iloc[-1]) - pd.Timestamp(recent["Date"].iloc[0])).total_seconds() / 86400.0,
            1.0,
        )
    else:
        slope = 0.0
    slope = float(np.clip(slope, -2 * np.pi * amplitude / period_days, 2 * np.pi * amplitude / period_days))

    result = pd.Series(np.nan, index=dates.index, dtype="float64")
    forecast_mask = dates.gt(last_date)
    for index, target in dates.loc[forecast_mask].items():
        target = pd.Timestamp(target).normalize()
        if target <= trough:
            t = float(np.clip((target - last_date).total_seconds() / 86400.0 / interval_days, 0.0, 1.0))
            t2, t3 = t * t, t * t * t
            h00, h10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t
            h01, h11 = -2 * t3 + 3 * t2, t3 - t2
            result.loc[index] = h00 * last_value + h10 * interval_days * slope + h01 * -amplitude
        else:
            phase = 2 * np.pi * ((target - trough).total_seconds() / 86400.0) / period_days
            result.loc[index] = -amplitude * np.cos(phase)
    return result


def point_in_time_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def rank(window: np.ndarray) -> float:
        valid = window[np.isfinite(window)]
        current = window[-1]
        if len(valid) < BTC_SECONDARY_PERCENTILE_MINIMUM or not np.isfinite(current):
            return np.nan
        return float(np.count_nonzero(valid <= current) / len(valid) * 100.0)

    return values.rolling(BTC_SECONDARY_PERCENTILE_WINDOW, min_periods=1).apply(rank, raw=True)


def _normalize_price_history(btc_weekly: pd.DataFrame) -> pd.DataFrame:
    if btc_weekly is None or btc_weekly.empty:
        return pd.DataFrame(columns=["Date", "BTC_Price"])
    date_column = "Date" if "Date" in btc_weekly.columns else "date" if "date" in btc_weekly.columns else None
    price_column = "Close" if "Close" in btc_weekly.columns else "close" if "close" in btc_weekly.columns else None
    if date_column is None or price_column is None:
        return pd.DataFrame(columns=["Date", "BTC_Price"])
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(btc_weekly[date_column], errors="coerce").dt.tz_localize(None),
            "BTC_Price": pd.to_numeric(btc_weekly[price_column], errors="coerce"),
        }
    )
    return frame.dropna().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def build_btc_modular_cycle_forecast(
    btc_weekly: pd.DataFrame,
    target_peak: float,
    scenario: str = "Base",
    start_date: pd.Timestamp | str = BTC_MODULAR_CYCLE_START,
    points_per_week: int = 1,
) -> pd.DataFrame:
    """Build one canonical, amplitude-scaled BTC modular-cycle price path."""
    prices = _normalize_price_history(btc_weekly)
    if prices.empty:
        raise ValueError("Canonical BTC weekly prices are unavailable")
    target_peak = float(target_peak)
    if not np.isfinite(target_peak) or target_peak <= 0:
        raise ValueError("BTC modular-cycle target peak must be a positive finite price")
    if not isinstance(points_per_week, int) or points_per_week < 1:
        raise ValueError("points_per_week must be a positive integer")

    start_date = pd.Timestamp(start_date).tz_localize(None).normalize()
    start_index = (prices["Date"] - start_date).abs().idxmin()
    start_price = float(prices.loc[start_index, "BTC_Price"])
    if target_peak <= start_price:
        raise ValueError("BTC modular-cycle target peak must be above the cycle start price")

    normalized_progress = np.linspace(0.0, 1.0, 1001)
    module_shapes = []
    module_durations = []

    for module_start, module_end in BTC_MODULAR_HISTORICAL_MODULES:
        start_row = (prices["Date"] - module_start).abs().idxmin()
        end_row = (prices["Date"] - module_end).abs().idxmin()
        if abs((pd.Timestamp(prices.loc[start_row, "Date"]) - module_start).days) > 7:
            raise ValueError(f"BTC price history does not cover modular-cycle start {module_start:%Y-%m-%d}")
        if abs((pd.Timestamp(prices.loc[end_row, "Date"]) - module_end).days) > 7:
            raise ValueError(f"BTC price history does not cover modular-cycle end {module_end:%Y-%m-%d}")
        module_start_price = float(prices.loc[start_row, "BTC_Price"])
        module_end_price = float(prices.loc[end_row, "BTC_Price"])
        if module_start_price <= 0 or module_end_price <= 0:
            raise ValueError("Historical modular-cycle prices must be positive")
        observed_start = pd.Timestamp(prices.loc[start_row, "Date"])
        observed_end = pd.Timestamp(prices.loc[end_row, "Date"])
        module_durations.append((observed_end - observed_start).total_seconds() / (7 * 86400.0))

        within_module = prices.loc[prices["Date"].between(module_start, module_end)].copy()
        module_dates = pd.DatetimeIndex([module_start, *within_module["Date"].tolist(), module_end])
        module_prices = np.concatenate(
            ([module_start_price], within_module["BTC_Price"].to_numpy(dtype="float64"), [module_end_price])
        )
        progress = (module_dates - module_start).total_seconds().to_numpy() / (module_end - module_start).total_seconds()
        order = np.argsort(progress, kind="stable")
        progress, module_prices = progress[order], module_prices[order]
        progress, unique_indices = np.unique(progress, return_index=True)
        module_prices = module_prices[unique_indices]
        log_shape = np.log(module_prices / module_start_price)
        module_shapes.append(np.interp(normalized_progress, progress, log_shape))

    duration_weeks = float(np.mean(module_durations))
    end_date = start_date + pd.to_timedelta(duration_weeks * 7.0, unit="D")
    raw_log_shape = np.mean(np.vstack(module_shapes), axis=0)
    shape_peak_index = int(np.argmax(raw_log_shape))
    max_raw_log_shape = float(raw_log_shape[shape_peak_index])
    if not np.isfinite(max_raw_log_shape) or max_raw_log_shape <= 0:
        raise ValueError("Average historical modular-cycle log shape has no positive peak")

    interval_days = (end_date - start_date).total_seconds() / 86400.0
    step_days = 7.0 / points_per_week
    elapsed_days = np.arange(0.0, interval_days, step_days, dtype="float64")
    elapsed_days = np.append(elapsed_days, interval_days)
    dates = pd.DatetimeIndex(start_date + pd.to_timedelta(elapsed_days, unit="D"))
    progress_pct = elapsed_days / interval_days * 100.0
    scaled_shape = np.interp(progress_pct / 100.0, normalized_progress, raw_log_shape)
    max_sampled_log_shape = float(np.max(scaled_shape))
    if not np.isfinite(max_sampled_log_shape) or max_sampled_log_shape <= 0:
        raise ValueError("Weekly average modular-cycle log shape has no positive peak")
    amplitude_scale = float(np.log(target_peak / start_price) / max_sampled_log_shape)
    multiples = np.exp(amplitude_scale * scaled_shape)
    model_prices = start_price * multiples
    peak_index = int(np.argmax(model_prices))
    peak_date = pd.Timestamp(dates[peak_index])
    peak_price = float(model_prices[peak_index])
    if not np.isclose(peak_price, target_peak, rtol=0, atol=max(1e-7, target_peak * 1e-10)):
        raise ArithmeticError("BTC modular-cycle peak does not match the selected target")

    halving_positions = [halving_cycle_position(date) for date in dates]
    distance_to_halving = [(BTC_PROJECTED_HALVING - date).total_seconds() / 86400.0 for date in dates]
    distance_to_peak = [(peak_date - date).total_seconds() / 86400.0 for date in dates]
    progress_since_halving = [
        item["progress"] if date >= BTC_PROJECTED_HALVING else np.nan
        for date, item in zip(dates, halving_positions, strict=False)
    ]
    module1_start, module1_end = BTC_MODULAR_HISTORICAL_MODULES[0]
    module2_start, module2_end = BTC_MODULAR_HISTORICAL_MODULES[1]
    result = pd.DataFrame(
        {
            "Date": dates,
            "NextCycle_StartDate": start_date,
            "NextCycle_StartPrice": start_price,
            "NextCycle_EndDate": end_date,
            "NextCycle_DurationWeeks": duration_weeks,
            "NextCycle_ProgressPct": progress_pct,
            "NextCycle_ModelMultiple": multiples,
            "NextCycle_ModelPrice": model_prices,
            "NextCycle_TargetPeak": target_peak,
            "NextCycle_Scenario": scenario,
            "NextCycle_PeakDate": peak_date,
            "NextCycle_PeakPrice": peak_price,
            "NextCycle_HistoricalModule1_Start": module1_start,
            "NextCycle_HistoricalModule1_End": module1_end,
            "NextCycle_HistoricalModule2_Start": module2_start,
            "NextCycle_HistoricalModule2_End": module2_end,
            "HalvingPhase": [item["phase"] for item in halving_positions],
            "DistanceToProjectedHalvingDays": distance_to_halving,
            "DistanceToModelCycleTopDays": distance_to_peak,
            "ProgressSince2028HalvingPct": progress_since_halving,
            "ProjectedFlag": "PROJECTED",
            "HistoricalProjectedFlag": "PROJECTED",
            "ProgressSince2028HalvingTooltip": [
                f"{value:.1f}%" if np.isfinite(value) else "" for value in progress_since_halving
            ],
        }
    )
    validate_btc_modular_cycle_forecast(result, target_peak)
    return result


def validate_btc_modular_cycle_forecast(forecast: pd.DataFrame, target_peak: float | None = None) -> list[str]:
    errors: list[str] = []
    if forecast is None or forecast.empty:
        return ["BTC modular-cycle forecast is empty"]
    first = forecast.iloc[0]
    if pd.Timestamp(first["NextCycle_StartDate"]) != BTC_MODULAR_CYCLE_START:
        errors.append("BTC modular-cycle forecast has an unexpected start date")
    if not np.isclose(float(forecast["NextCycle_ProgressPct"].iloc[0]), 0.0, atol=1e-12):
        errors.append("BTC modular-cycle progress must begin at zero")
    if not np.isclose(float(forecast["NextCycle_ProgressPct"].iloc[-1]), 100.0, atol=1e-12):
        errors.append("BTC modular-cycle progress must end at 100")
    if forecast["Date"].iloc[0] != first["NextCycle_StartDate"] or forecast["Date"].iloc[-1] != first["NextCycle_EndDate"]:
        errors.append("BTC modular-cycle forecast dates do not match its model window")
    actual_peak = float(pd.to_numeric(forecast["NextCycle_ModelPrice"], errors="coerce").max())
    expected_peak = float(first["NextCycle_TargetPeak"] if target_peak is None else target_peak)
    if not np.isclose(actual_peak, expected_peak, rtol=0, atol=max(1e-7, expected_peak * 1e-10)):
        errors.append("BTC modular-cycle peak does not match the selected target")
    if forecast["NextCycle_ModelMultiple"].lt(0).any() or not np.isfinite(forecast["NextCycle_ModelPrice"]).all():
        errors.append("BTC modular-cycle model contains invalid values")
    return errors


def _asof_column(
    base_dates: pd.Series,
    source: pd.DataFrame,
    source_date_col: str,
    source_value_col: str,
    output_col: str,
) -> pd.Series:
    if source is None or source.empty or source_date_col not in source or source_value_col not in source:
        return pd.Series(np.nan, index=base_dates.index, dtype="float64", name=output_col)
    values = pd.DataFrame(
        {
            "Date": pd.to_datetime(source[source_date_col], errors="coerce").dt.tz_localize(None),
            output_col: pd.to_numeric(source[source_value_col], errors="coerce"),
        }
    ).dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    if values.empty:
        return pd.Series(np.nan, index=base_dates.index, dtype="float64", name=output_col)
    aligned = pd.merge_asof(
        pd.DataFrame({"Date": pd.to_datetime(base_dates, errors="coerce")}).sort_values("Date"),
        values,
        on="Date",
        direction="backward",
    )
    return pd.Series(aligned[output_col].to_numpy(), index=base_dates.index, name=output_col)


def _secondary_components(frame: pd.DataFrame) -> None:
    dxy = pd.to_numeric(frame["DXY"], errors="coerce")
    us2y = pd.to_numeric(frame["US2Y"], errors="coerce")
    real_yield = pd.to_numeric(frame["RealYield"], errors="coerce")
    frame["DXY_13W_Return"] = dxy.pct_change(13, fill_method=None)
    frame["DXY_13W_Pct"] = point_in_time_percentile(frame["DXY_13W_Return"])
    frame["DXYBull"] = 100.0 - frame["DXY_13W_Pct"]
    frame["US2Y_13W_Change"] = us2y - us2y.shift(13)
    frame["US2Y_13W_Pct"] = point_in_time_percentile(frame["US2Y_13W_Change"])
    frame["US2YBull"] = 100.0 - frame["US2Y_13W_Pct"]
    frame["RealYield_13W_Change"] = real_yield - real_yield.shift(13)
    frame["RealYield_13W_Pct"] = point_in_time_percentile(frame["RealYield_13W_Change"])
    frame["RealYieldBull"] = 100.0 - frame["RealYield_13W_Pct"]


def build_btc_cycle_history(
    btc_weekly: pd.DataFrame,
    global_m2_cycle: pd.DataFrame,
    macro_weekly: pd.DataFrame,
    projection_end: pd.Timestamp | None = None,
    global_liquidity_score: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build weekly history plus phase and Global M2 cycle projections."""
    observed = _normalize_price_history(btc_weekly)
    if observed.empty:
        return pd.DataFrame()
    last_observed = pd.Timestamp(observed["Date"].max()).normalize()
    projection_end = pd.Timestamp(projection_end).normalize() if projection_end is not None else next_accumulation_pre_halving_start(last_observed)

    future_dates = pd.date_range(last_observed + pd.Timedelta(1, unit="D"), projection_end, freq="W-FRI")
    projected = pd.DataFrame({"Date": future_dates, "BTC_Price": np.nan})
    frame = pd.concat([observed, projected], ignore_index=True).sort_values("Date").reset_index(drop=True)
    frame["Projected"] = frame["Date"].gt(last_observed)
    frame["DXY"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_DXY_Close", "DXY")
    frame["US2Y"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_US2Y", "US2Y")
    frame["RealYield"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_RealYield", "RealYield")
    frame["GlobalM2PrimaryCycle"] = _asof_column(frame["Date"], global_m2_cycle, "Date", "PrimaryMarketCycle", "GlobalM2PrimaryCycle")
    frame["GlobalLiquidityScore"] = _asof_column(
        frame["Date"], global_liquidity_score, "date", "global_liquidity_score", "GlobalLiquidityScore"
    )
    frame["GlobalM2CycleProjected"] = False
    if global_m2_cycle is not None and not global_m2_cycle.empty:
        cycle_source = global_m2_cycle.copy()
        date_column = "Date" if "Date" in cycle_source.columns else "date"
        cycle_source["Date"] = pd.to_datetime(cycle_source[date_column], errors="coerce")
        cycle_source["PrimaryMarketCycle"] = pd.to_numeric(cycle_source["PrimaryMarketCycle"], errors="coerce")
        cycle_source = cycle_source[["Date", "PrimaryMarketCycle"]].dropna().sort_values("Date")
        if not cycle_source.empty:
            cycle_last_date = pd.Timestamp(cycle_source["Date"].iloc[-1]).normalize()
            if cycle_last_date < last_observed:
                latest_cycle_value = frame.loc[frame["Date"].eq(last_observed), "GlobalM2PrimaryCycle"].iloc[0]
                cycle_source.loc[len(cycle_source)] = [last_observed, latest_cycle_value]
            next_trough = liquidity_cycle_position(last_observed, last_observed)["next_trough"]
            cycle_forecast_dates = frame["Date"].where(frame["Projected"])
            cycle_forecast = _project_global_m2_cycle(cycle_source, cycle_forecast_dates, next_trough)
            forecast_mask = cycle_forecast.notna()
            frame.loc[forecast_mask, "GlobalM2PrimaryCycle"] = cycle_forecast.loc[forecast_mask]
            frame.loc[forecast_mask, "GlobalM2CycleProjected"] = True
    _secondary_components(frame)

    halving_positions = [halving_cycle_position(value) for value in frame["Date"]]
    frame["Current_Halving_Progress"] = [item["progress"] for item in halving_positions]
    frame["Current_Halving_Phase"] = [item["phase"] for item in halving_positions]
    frame["Previous_Halving"] = [item["previous_halving"] for item in halving_positions]
    frame["Next_Halving"] = [item["next_halving"] for item in halving_positions]
    frame["Next_Halving_Projected"] = [item["next_halving_projected"] for item in halving_positions]

    known_through = last_observed
    liquidity_positions = [liquidity_cycle_position(date, known_through if projected else date) for date, projected in zip(frame["Date"], frame["Projected"], strict=False)]
    frame["Current_Liquidity_Progress"] = [item["progress"] for item in liquidity_positions]
    frame["Current_Liquidity_Phase"] = [item["phase"] for item in liquidity_positions]
    frame["Liquidity_Trough_Anchor"] = [item["anchor"] for item in liquidity_positions]
    frame["Projected_Liquidity_Next_Trough"] = [item["next_trough"] for item in liquidity_positions]

    for horizon, months in BTC_CYCLE_HORIZONS.items():
        targets = frame["Date"] + pd.DateOffset(months=months)
        target_halving = [halving_cycle_position(target) for target in targets]
        target_liquidity = [liquidity_cycle_position(target, date if not is_projected else known_through) for target, date, is_projected in zip(targets, frame["Date"], frame["Projected"], strict=False)]
        phase_col = f"HalvingPhase_{horizon}"
        frame[phase_col] = [item["phase"] for item in target_halving]
        frame[f"HalvingProgress_{horizon}"] = [item["progress"] for item in target_halving]
        frame[f"HalvingBase_{horizon}"] = [HALVING_BASES.get(str(phase), {}).get(horizon, np.nan) for phase in frame[phase_col]]
        frame[f"LiquidityPhase_{horizon}"] = [item["phase"] for item in target_liquidity]
        frame[f"LiquidityProgress_{horizon}"] = [item["progress"] for item in target_liquidity]
        frame[f"LiquidityCycleModifier_{horizon}"] = frame[f"LiquidityPhase_{horizon}"].map(LIQUIDITY_MODIFIERS)
        weights = SECONDARY_WEIGHTS[horizon]
        macro_values = frame[["DXYBull", "US2YBull", "RealYieldBull"]].to_numpy(dtype="float64")
        secondary_macro = np.full(len(frame), np.nan)
        for idx, values in enumerate(macro_values):
            if np.isfinite(values).all():
                secondary_macro[idx] = float(np.dot(values, weights))
        frame[f"SecondaryMacro_{horizon}"] = secondary_macro
        raw_modifier = (frame[f"SecondaryMacro_{horizon}"] - 50.0) * (0.20 if months <= 6 else 0.10)
        limit = SECONDARY_MODIFIER_LIMITS[horizon]
        frame[f"SecondaryMacroModifier_{horizon}"] = raw_modifier.clip(-limit, limit)
        frame.loc[frame["Projected"], f"SecondaryMacro_{horizon}"] = np.nan
        frame.loc[frame["Projected"], f"SecondaryMacroModifier_{horizon}"] = 0.0
        before_clamp = frame[f"HalvingBase_{horizon}"] + frame[f"LiquidityCycleModifier_{horizon}"] + frame[f"SecondaryMacroModifier_{horizon}"]
        frame[f"BTC_MACRO_{horizon}_BeforeClamp"] = before_clamp
        frame[f"BTC_MACRO_{horizon}"] = before_clamp.clip(0.0, 95.0)
        frame[f"BTC_MACRO_{horizon}_State"] = frame[f"BTC_MACRO_{horizon}"].map(btc_macro_state)
        frame[f"HalvingPhase_{horizon}"] = frame[phase_col]

    frame["Historical_Projected_Flag"] = np.where(frame["Projected"], "PROJECTED", "HISTORICAL")
    return frame


def btc_macro_state(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "DATA_INCOMPLETE"
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score < 20.0:
        return "STRONGLY_UNFAVORABLE"
    if score < 40.0:
        return "UNFAVORABLE"
    if score < 60.0:
        return "NEUTRAL_MIXED"
    if score < 80.0:
        return "SUPPORTIVE"
    return "STRONGLY_SUPPORTIVE"


def btc_cycle_validation(history: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if history is None or history.empty:
        return ["BTC Cycle history is empty"]
    for horizon in BTC_CYCLE_HORIZONS:
        score = pd.to_numeric(history[f"BTC_MACRO_{horizon}"], errors="coerce").dropna()
        if score.lt(0).any() or score.gt(95).any():
            errors.append(f"{horizon} score outside 0-95")
        projected = history["Projected"].astype(bool)
        if history.loc[projected, "BTC_Price"].notna().any():
            errors.append("Projected BTC price must remain empty")
        if not pd.to_numeric(history.loc[projected, f"SecondaryMacroModifier_{horizon}"], errors="coerce").eq(0).all():
            errors.append(f"Projected {horizon} secondary modifier must be zero")
        calculated = (
            pd.to_numeric(history[f"HalvingBase_{horizon}"], errors="coerce")
            + pd.to_numeric(history[f"LiquidityCycleModifier_{horizon}"], errors="coerce")
            + pd.to_numeric(history[f"SecondaryMacroModifier_{horizon}"], errors="coerce")
        ).clip(0.0, 95.0)
        actual = pd.to_numeric(history[f"BTC_MACRO_{horizon}"], errors="coerce")
        matched = actual.notna() & calculated.notna()
        if not np.allclose(actual.loc[matched], calculated.loc[matched], atol=1e-9):
            errors.append(f"{horizon} score does not reconcile")
    return errors


def btc_cycle_export_xlsx(
    history: pd.DataFrame,
    next_cycle_forecast: pd.DataFrame | None = None,
) -> bytes:
    output = io.BytesIO()
    export = history.copy()
    export.insert(0, "Date", pd.to_datetime(export.pop("Date"), errors="coerce"))
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        export.to_excel(writer, sheet_name="BTC Cycle", index=False)
        worksheet = writer.sheets["BTC Cycle"]
        worksheet.freeze_panes = "B2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.column_dimensions["A"].width = 14
        if next_cycle_forecast is not None and not next_cycle_forecast.empty:
            forecast_columns = [
                "Date",
                "NextCycle_ModelPrice",
                "NextCycle_ModelMultiple",
                "NextCycle_ProgressPct",
                "NextCycle_TargetPeak",
                "NextCycle_Scenario",
                "NextCycle_PeakDate",
                "NextCycle_EndDate",
                "HalvingPhase",
                "ProjectedFlag",
                "NextCycle_StartDate",
                "NextCycle_StartPrice",
                "NextCycle_DurationWeeks",
                "NextCycle_PeakPrice",
                "NextCycle_HistoricalModule1_Start",
                "NextCycle_HistoricalModule1_End",
                "NextCycle_HistoricalModule2_Start",
                "NextCycle_HistoricalModule2_End",
                "DistanceToProjectedHalvingDays",
                "DistanceToModelCycleTopDays",
                "ProgressSince2028HalvingPct",
            ]
            model_export = next_cycle_forecast[forecast_columns].copy()
            model_export["Date"] = pd.to_datetime(model_export["Date"], errors="coerce")
            model_export.to_excel(writer, sheet_name="Next Cycle Forecast", index=False)
            forecast_sheet = writer.sheets["Next Cycle Forecast"]
            forecast_sheet.freeze_panes = "B2"
            forecast_sheet.auto_filter.ref = forecast_sheet.dimensions
            forecast_sheet.column_dimensions["A"].width = 14
    return output.getvalue()

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
BTC_LIQUIDITY_CYCLE_MONTHS = 49
BTC_SECONDARY_PERCENTILE_WINDOW = 156
BTC_SECONDARY_PERCENTILE_MINIMUM = 104
BTC_DEFAULT_PROJECTION_END = pd.Timestamp("2028-04-30")

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
    while anchor + pd.DateOffset(months=BTC_LIQUIDITY_CYCLE_MONTHS) <= target:
        anchor += pd.DateOffset(months=BTC_LIQUIDITY_CYCLE_MONTHS)
    next_trough = anchor + pd.DateOffset(months=BTC_LIQUIDITY_CYCLE_MONTHS)
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
    projection_end: pd.Timestamp = BTC_DEFAULT_PROJECTION_END,
) -> pd.DataFrame:
    """Build weekly historical and phase-only projected BTC macro-cycle rows."""
    observed = _normalize_price_history(btc_weekly)
    if observed.empty:
        return pd.DataFrame()
    last_observed = pd.Timestamp(observed["Date"].max()).normalize()
    projection_end = pd.Timestamp(projection_end).normalize()

    future_dates = pd.date_range(last_observed + pd.Timedelta(days=1), projection_end, freq="W-FRI")
    projected = pd.DataFrame({"Date": future_dates, "BTC_Price": np.nan})
    frame = pd.concat([observed, projected], ignore_index=True).sort_values("Date").reset_index(drop=True)
    frame["Projected"] = frame["Date"].gt(last_observed)
    frame["DXY"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_DXY_Close", "DXY")
    frame["US2Y"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_US2Y", "US2Y")
    frame["RealYield"] = _asof_column(frame["Date"], macro_weekly, "Date", "BTC_RealYield", "RealYield")
    frame["GlobalM2PrimaryCycle"] = _asof_column(frame["Date"], global_m2_cycle, "Date", "PrimaryMarketCycle", "GlobalM2PrimaryCycle")
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


def btc_cycle_export_xlsx(history: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    export = history.copy()
    export.insert(0, "Date", pd.to_datetime(export.pop("Date"), errors="coerce"))
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        export.to_excel(writer, sheet_name="BTC Cycle", index=False)
        worksheet = writer.sheets["BTC Cycle"]
        worksheet.freeze_panes = "B2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.column_dimensions["A"].width = 14
    return output.getvalue()

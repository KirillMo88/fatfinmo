from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CORRECTION_BOTTOM_MODEL_VERSION = "CORRECTION_BOTTOM_V2"
EXTREME_THRESHOLDS = {
    "VIX_Stress": 97.6,
    "RSI_Stress": 89.3,
    "SPY5D_DownsideStress": 90.5,
    "Breadth50_Stress": 93.2,
    "Breadth200_Stress": 94.4,
}
EXTREME_CAPITULATION_2_DRAWDOWN = -0.20
EXTREME_CAPITULATION_2_VIX_STRESS = 99.8


def calculate_correction_bottom_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the point-in-time daily Correction Bottom model on SPY data."""
    if frame.empty or "Date" not in frame or "SPY_Close" not in frame:
        return frame.copy()

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    close = numeric(out, "SPY_Close")
    low = numeric(out, "SPY_Low").fillna(close)
    volume = numeric(out, "SPY_Volume")
    vix = numeric(out, "VIX")
    breadth50 = numeric(out, "SPXAboveSMA50D")
    breadth200 = numeric(out, "SPXAboveSMA200D")
    hy_oas = numeric(out, "HY_OAS")

    breadth50_observed = observed_mask(out, "SPXAboveSMA50D")
    breadth200_observed = observed_mask(out, "SPXAboveSMA200D")
    breadth_frequency = classify_breadth_frequency(breadth50_observed, breadth200_observed)

    rsi = wilder_rsi(close, 14)
    spy_5d_return = close.pct_change(5, fill_method=None)
    sma200 = close.rolling(200, min_periods=200).mean()
    sma200_slope20 = sma200 - sma200.shift(20)
    roc3m = close.pct_change(63, fill_method=None)
    roc6m = close.pct_change(126, fill_method=None)
    previous_10d_high = close.shift(1).rolling(10, min_periods=10).max()

    vix_peak20 = vix.rolling(20, min_periods=5).max()
    vix_retreat20 = safe_retreat(vix, vix_peak20)
    breadth50_change5 = valid_observation_change(breadth50, breadth50_observed, 5)
    breadth200_change5 = valid_observation_change(breadth200, breadth200_observed, 5)

    hy_change20 = hy_oas - hy_oas.shift(20)
    hy_peak20 = hy_oas.rolling(20, min_periods=5).max()
    hy_retreat20 = safe_retreat(hy_oas, hy_peak20)
    hy_pct3y = rolling_percentile_pit(hy_oas, 756, 252)

    volume_sma20 = volume.rolling(20, min_periods=10).mean()
    volume_ratio20 = volume / volume_sma20.replace(0.0, np.nan)
    volume_pct1y = rolling_percentile_pit(volume, 252, 126)
    volume_max_pct5 = volume_pct1y.rolling(5, min_periods=1).max()
    volume_max_ratio5 = volume_ratio20.rolling(5, min_periods=1).max()

    vix_stress = rolling_percentile_pit(vix, 252, 126)
    rsi_stress = 100.0 - rolling_percentile_pit(rsi, 252, 126)
    downside_stress = 100.0 - rolling_percentile_pit(spy_5d_return, 252, 126)
    breadth50_stress = 100.0 - rolling_percentile_masked(
        breadth50, breadth50_observed, window=252, min_periods=40
    )
    breadth200_stress = 100.0 - rolling_percentile_masked(
        breadth200, breadth200_observed, window=252, min_periods=40
    )

    out["SPY_RSI14"] = rsi
    out["SPY_5D_Return"] = spy_5d_return
    out["SPY_SMA200D"] = sma200
    out["SMA200D_Slope20D"] = sma200_slope20
    out["SPY_ROC3M"] = roc3m
    out["SPY_ROC6M"] = roc6m
    out["Previous10DHigh"] = previous_10d_high
    out["VIX_20D_Peak"] = vix_peak20
    out["VIXRetreat20D"] = vix_retreat20
    out["Breadth50_5D_Change"] = breadth50_change5
    out["Breadth200_5D_Change"] = breadth200_change5
    out["HYOAS20DChange"] = hy_change20
    out["HYOAS20DPeak"] = hy_peak20
    out["HYOASRetreat20D"] = hy_retreat20
    out["HYOASPercentile3Y"] = hy_pct3y
    out["SPYVolumePct1Y"] = volume_pct1y
    out["SPYVolumeRatio20"] = volume_ratio20
    out["SPYVolumeMaxPct5"] = volume_max_pct5
    out["SPYVolumeMaxRatio5"] = volume_max_ratio5
    out["VIX_Stress"] = vix_stress
    out["RSI_Stress"] = rsi_stress
    out["SPY5D_DownsideStress"] = downside_stress
    out["Breadth50_Stress"] = breadth50_stress
    out["Breadth200_Stress"] = breadth200_stress
    out["ExtremeStressMean5"] = pd.concat(
        [vix_stress, rsi_stress, downside_stress, breadth50_stress, breadth200_stress], axis=1
    ).mean(axis=1, skipna=False)
    extreme_component_values = out[list(EXTREME_THRESHOLDS)].apply(pd.to_numeric, errors="coerce")
    extreme_component_pass = pd.DataFrame(
        {
            column: extreme_component_values[column].ge(threshold)
            for column, threshold in EXTREME_THRESHOLDS.items()
        },
        index=out.index,
    )
    extreme_conditions_met = extreme_component_pass.sum(axis=1).astype("Int64")
    extreme_conditions_met.loc[extreme_component_values.isna().any(axis=1)] = pd.NA
    out["ExtremeConditionsMet"] = extreme_conditions_met
    out["BreadthSourceFrequency"] = breadth_frequency
    out["CFTCDataAvailable"] = False
    out["CFTCUnavailableReason"] = "CFTC is not an input to Correction Bottom V2"
    out["CorrectionModelVersion"] = CORRECTION_BOTTOM_MODEL_VERSION
    out["CorrectionFrequency"] = "DAILY"

    add_correction_lifecycle(out)
    add_extreme_capitulation(out)
    add_historical_correction_metadata(out)
    add_compatibility_columns(out)
    return out


def add_correction_lifecycle(out: pd.DataFrame) -> None:
    close = numeric(out, "SPY_Close")
    low = numeric(out, "SPY_Low").fillna(close)
    dates = pd.to_datetime(out["Date"], errors="coerce")
    vix_retreat = numeric(out, "VIXRetreat20D")
    breadth50_change = numeric(out, "Breadth50_5D_Change")
    breadth200_change = numeric(out, "Breadth200_5D_Change")
    previous_10d_high = numeric(out, "Previous10DHigh")
    sma200 = numeric(out, "SPY_SMA200D")
    sma200_slope = numeric(out, "SMA200D_Slope20D")
    roc3m = numeric(out, "SPY_ROC3M")
    roc6m = numeric(out, "SPY_ROC6M")
    hy_oas = numeric(out, "HY_OAS")
    hy_change = numeric(out, "HYOAS20DChange")
    hy_retreat = numeric(out, "HYOASRetreat20D")
    hy_pct = numeric(out, "HYOASPercentile3Y")

    result: dict[str, list[Any]] = {name: [] for name in lifecycle_columns()}

    running_peak = np.nan
    running_peak_date = pd.NaT
    active = False
    cycle_number = 0
    cycle_id: str | None = None
    cycle_peak = np.nan
    cycle_peak_date = pd.NaT
    cycle_start_date = pd.NaT
    cycle_worst = np.nan
    cycle_low = np.nan
    cycle_low_date = pd.NaT
    wave_number = 0
    wave_id: str | None = None
    wave_start_date = pd.NaT
    wave_reference_high = np.nan
    wave_worst = np.nan
    tactical_eligible = False
    post_tactical_high = np.nan
    wave_recovery_high = np.nan
    latest_durable_date = pd.NaT
    latest_durable_price = np.nan
    latest_durable_route = ""
    latest_bear_score = np.nan
    latest_bear_category = "DATA INCOMPLETE"

    for idx in range(len(out)):
        date = dates.iloc[idx]
        price = safe_float(close.iloc[idx])
        day_low = safe_float(low.iloc[idx])
        if not np.isfinite(price):
            append_unavailable_lifecycle(result)
            continue

        if not active and (not np.isfinite(running_peak) or price > running_peak):
            running_peak = price
            running_peak_date = date

        drawdown_to_peak = price / running_peak - 1.0 if np.isfinite(running_peak) and running_peak > 0 else np.nan
        recovered_today = bool(active and price >= cycle_peak)

        if recovered_today:
            append_lifecycle_row(
                result,
                cycle_id=cycle_id,
                wave_id=wave_id,
                active=False,
                peak_date=cycle_peak_date,
                peak_price=cycle_peak,
                start_date=cycle_start_date,
                drawdown=0.0,
                worst_drawdown=cycle_worst,
                low_date=cycle_low_date,
                low_price=cycle_low,
                status="NORMAL",
                correction_type=correction_severity(cycle_worst),
                wave_start_date=wave_start_date,
                wave_recovery_high=wave_recovery_high,
                wave_worst=wave_worst,
                tactical_eligible=False,
                tactical_event=False,
                post_tactical_high=np.nan,
                durable=False,
                trend_route=False,
                credit_route=False,
                route="",
                bear_values=bear_risk_values(price, sma200.iloc[idx], sma200_slope.iloc[idx], roc3m.iloc[idx], roc6m.iloc[idx], hy_pct.iloc[idx], hy_change.iloc[idx]),
                latest_durable=(latest_durable_date, latest_durable_price, latest_durable_route, latest_bear_score, latest_bear_category),
                recovery_date=date,
            )
            active = False
            running_peak = price
            running_peak_date = date
            cycle_id = None
            wave_id = None
            tactical_eligible = False
            post_tactical_high = np.nan
            wave_recovery_high = np.nan
            latest_durable_date = pd.NaT
            latest_durable_price = np.nan
            latest_durable_route = ""
            latest_bear_score = np.nan
            latest_bear_category = "DATA INCOMPLETE"
            continue

        if not active and np.isfinite(drawdown_to_peak) and drawdown_to_peak <= -0.05:
            active = True
            cycle_number += 1
            cycle_id = f"C{cycle_number:03d}"
            cycle_peak = running_peak
            cycle_peak_date = running_peak_date
            cycle_start_date = date
            cycle_worst = drawdown_to_peak
            cycle_low = day_low if np.isfinite(day_low) else price
            cycle_low_date = date
            wave_number = 1
            wave_id = f"{cycle_id}-W{wave_number:02d}"
            wave_start_date = date
            wave_reference_high = cycle_peak
            wave_worst = price / wave_reference_high - 1.0
            tactical_eligible = True
            post_tactical_high = np.nan
            wave_recovery_high = np.nan
            latest_durable_date = pd.NaT
            latest_durable_price = np.nan
            latest_durable_route = ""
            latest_bear_score = np.nan
            latest_bear_category = "DATA INCOMPLETE"

        if not active:
            bear_values = bear_risk_values(
                price, sma200.iloc[idx], sma200_slope.iloc[idx], roc3m.iloc[idx], roc6m.iloc[idx], hy_pct.iloc[idx], hy_change.iloc[idx]
            )
            append_lifecycle_row(
                result,
                cycle_id=None,
                wave_id=None,
                active=False,
                peak_date=running_peak_date,
                peak_price=running_peak,
                start_date=pd.NaT,
                drawdown=drawdown_to_peak,
                worst_drawdown=np.nan,
                low_date=pd.NaT,
                low_price=np.nan,
                status="NORMAL",
                correction_type="NORMAL",
                wave_start_date=pd.NaT,
                wave_recovery_high=np.nan,
                wave_worst=np.nan,
                tactical_eligible=False,
                tactical_event=False,
                post_tactical_high=np.nan,
                durable=False,
                trend_route=False,
                credit_route=False,
                route="",
                bear_values=bear_values,
                latest_durable=(pd.NaT, np.nan, "", np.nan, "DATA INCOMPLETE"),
                recovery_date=pd.NaT,
            )
            continue

        current_drawdown = price / cycle_peak - 1.0
        cycle_worst = min(cycle_worst, current_drawdown)
        if np.isfinite(day_low) and (not np.isfinite(cycle_low) or day_low < cycle_low):
            cycle_low = day_low
            cycle_low_date = date

        if not tactical_eligible and np.isfinite(post_tactical_high):
            post_tactical_high = max(post_tactical_high, price)
            wave_recovery_high = post_tactical_high
            if price / post_tactical_high - 1.0 <= -0.05:
                wave_number += 1
                wave_id = f"{cycle_id}-W{wave_number:02d}"
                wave_start_date = date
                wave_reference_high = post_tactical_high
                wave_worst = price / wave_reference_high - 1.0
                tactical_eligible = True
                post_tactical_high = np.nan
                wave_recovery_high = np.nan
                latest_durable_date = pd.NaT
                latest_durable_price = np.nan
                latest_durable_route = ""
                latest_bear_score = np.nan
                latest_bear_category = "DATA INCOMPLETE"

        if np.isfinite(wave_reference_high) and wave_reference_high > 0:
            wave_worst = min(wave_worst, price / wave_reference_high - 1.0)

        tactical_route_a = all_true(vix_retreat.iloc[idx] >= 0.25, breadth50_change.iloc[idx] >= 15.0)
        tactical_route_b = all_true(vix_retreat.iloc[idx] >= 0.30, breadth50_change.iloc[idx] >= 5.0)
        price_breakout = all_true(price > previous_10d_high.iloc[idx])
        tactical_event = bool(tactical_eligible and price_breakout and (tactical_route_a or tactical_route_b))

        trend_route = all_true(
            price > sma200.iloc[idx],
            sma200_slope.iloc[idx] > 0,
            roc3m.iloc[idx] >= 0,
            roc6m.iloc[idx] >= 0,
            breadth200_change.iloc[idx] >= 3.0,
        )
        credit_route = any_true(
            hy_change.iloc[idx] < 0,
            hy_pct.iloc[idx] >= 85.0,
            all_true(vix_retreat.iloc[idx] >= 0.40, hy_retreat.iloc[idx] >= 0.10),
        )
        durable = bool(tactical_event and (trend_route or credit_route))
        route = "BOTH" if trend_route and credit_route else "TREND_BREADTH" if trend_route else "CREDIT_STRESS" if credit_route else ""
        bear_values = bear_risk_values(
            price, sma200.iloc[idx], sma200_slope.iloc[idx], roc3m.iloc[idx], roc6m.iloc[idx], hy_pct.iloc[idx], hy_change.iloc[idx]
        )

        if tactical_event:
            tactical_eligible = False
            post_tactical_high = price
            wave_recovery_high = price
        if durable:
            latest_durable_date = date
            latest_durable_price = price
            latest_durable_route = route
            latest_bear_score = bear_values[0]
            latest_bear_category = bear_values[1]

        append_lifecycle_row(
            result,
            cycle_id=cycle_id,
            wave_id=wave_id,
            active=True,
            peak_date=cycle_peak_date,
            peak_price=cycle_peak,
            start_date=cycle_start_date,
            drawdown=current_drawdown,
            worst_drawdown=cycle_worst,
            low_date=cycle_low_date,
            low_price=cycle_low,
            status=correction_status(current_drawdown),
            correction_type=correction_severity(cycle_worst),
            wave_start_date=wave_start_date,
            wave_recovery_high=wave_recovery_high,
            wave_worst=wave_worst,
            tactical_eligible=tactical_eligible,
            tactical_event=tactical_event,
            post_tactical_high=post_tactical_high,
            durable=durable,
            trend_route=trend_route if tactical_event else False,
            credit_route=credit_route if tactical_event else False,
            route=route if durable else "",
            bear_values=bear_values,
            latest_durable=(latest_durable_date, latest_durable_price, latest_durable_route, latest_bear_score, latest_bear_category),
            recovery_date=pd.NaT,
        )

    for column, values in result.items():
        out[column] = values

    core_inputs = [
        "SPY_Close",
        "VIX",
        "SPY_RSI14",
        "SPY_5D_Return",
        "SPXAboveSMA50D",
        "SPXAboveSMA200D",
        "HY_OAS",
        "SPY_Volume",
    ]
    out["CorrectionDataCoverage"] = out[[col for col in core_inputs if col in out]].notna().mean(axis=1) * 100.0


def add_extreme_capitulation(out: pd.DataFrame) -> None:
    drawdown = numeric(out, "CorrectionCurrentDrawdown")
    required = list(EXTREME_THRESHOLDS)
    available = out[required].notna().all(axis=1)
    threshold_pass = pd.Series(True, index=out.index)
    for column, threshold in EXTREME_THRESHOLDS.items():
        threshold_pass &= numeric(out, column).ge(threshold)
    raw = pd.Series(pd.NA, index=out.index, dtype="boolean")
    raw.loc[available] = (drawdown.loc[available] <= -0.15) & threshold_pass.loc[available]
    out["ExtremeCapRaw"] = raw
    out["ExtremeEpisodeID"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["ExtremeEpisodeMarker"] = False
    out["ExtremeEpisodeStart"] = pd.NaT
    out["ExtremeEpisodeEnd"] = pd.NaT

    raw_indices = [int(idx) for idx in out.index[raw.fillna(False)]]
    episodes: list[list[int]] = []
    for idx in raw_indices:
        if not episodes or idx - episodes[-1][-1] > 10:
            episodes.append([idx])
        else:
            episodes[-1].append(idx)
    for episode_number, indices in enumerate(episodes, start=1):
        episode_id = f"E{episode_number:03d}"
        scores = numeric(out.loc[indices], "ExtremeStressMean5")
        representative = int(scores.idxmax()) if scores.notna().any() else indices[0]
        out.loc[indices, "ExtremeEpisodeID"] = episode_id
        out.loc[indices, "ExtremeEpisodeStart"] = out.loc[indices[0], "Date"]
        out.loc[indices, "ExtremeEpisodeEnd"] = out.loc[indices[-1], "Date"]
        out.loc[representative, "ExtremeEpisodeMarker"] = True

    out["ExtremeCapitulationStatus"] = np.where(raw.fillna(False), "ACTIVE", "OFF")

    extreme_2_raw = pd.Series(pd.NA, index=out.index, dtype="boolean")
    extreme_2_raw.loc[available] = (
        raw.loc[available].fillna(False)
        & (
            drawdown.loc[available].le(EXTREME_CAPITULATION_2_DRAWDOWN)
            | numeric(out, "VIX_Stress").loc[available].ge(EXTREME_CAPITULATION_2_VIX_STRESS)
        )
    )
    out["ExtremeCapitulation2Raw"] = extreme_2_raw
    out["ExtremeCapitulation2EpisodeID"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["ExtremeCapitulation2EpisodeMarker"] = False
    out["ExtremeCapitulation2EpisodeStart"] = pd.NaT
    out["ExtremeCapitulation2EpisodeEnd"] = pd.NaT

    extreme_2_indices = [int(idx) for idx in out.index[extreme_2_raw.fillna(False)]]
    extreme_2_episodes: list[list[int]] = []
    for idx in extreme_2_indices:
        if not extreme_2_episodes or idx - extreme_2_episodes[-1][-1] > 10:
            extreme_2_episodes.append([idx])
        else:
            extreme_2_episodes[-1].append(idx)
    for episode_number, indices in enumerate(extreme_2_episodes, start=1):
        episode_id = f"E2-{episode_number:03d}"
        scores = numeric(out.loc[indices], "ExtremeStressMean5")
        representative = int(scores.idxmax()) if scores.notna().any() else indices[0]
        out.loc[indices, "ExtremeCapitulation2EpisodeID"] = episode_id
        out.loc[indices, "ExtremeCapitulation2EpisodeStart"] = out.loc[indices[0], "Date"]
        out.loc[indices, "ExtremeCapitulation2EpisodeEnd"] = out.loc[indices[-1], "Date"]
        out.loc[representative, "ExtremeCapitulation2EpisodeMarker"] = True

    classic_ids_with_level_2 = set(
        out.loc[extreme_2_raw.fillna(False), "ExtremeEpisodeID"].dropna().astype(str)
    )
    out["ExtremeEpisodeHasLevel2"] = out["ExtremeEpisodeID"].astype(str).isin(classic_ids_with_level_2)
    out["ExtremeEpisodeDisplayMarker"] = out["ExtremeEpisodeMarker"] & ~out["ExtremeEpisodeHasLevel2"]
    out["ExtremeCapitulation2Status"] = np.where(extreme_2_raw.fillna(False), "ACTIVE", "OFF")

    out["extreme_capitulation"] = out["ExtremeCapRaw"]
    out["extreme_capitulation_event"] = out["ExtremeEpisodeMarker"]
    out["extreme_capitulation_episode_id"] = out["ExtremeEpisodeID"]
    out["extreme_capitulation_episode_start"] = out["ExtremeEpisodeStart"]
    out["extreme_capitulation_episode_end"] = out["ExtremeEpisodeEnd"]
    out["extreme_capitulation_2"] = out["ExtremeCapitulation2Raw"]
    out["extreme_capitulation_2_event"] = out["ExtremeCapitulation2EpisodeMarker"]
    out["extreme_capitulation_2_episode_id"] = out["ExtremeCapitulation2EpisodeID"]
    out["extreme_capitulation_2_episode_start"] = out["ExtremeCapitulation2EpisodeStart"]
    out["extreme_capitulation_2_episode_end"] = out["ExtremeCapitulation2EpisodeEnd"]

    volume_extreme = numeric(out, "SPYVolumeMaxPct5").ge(95.0) | numeric(out, "SPYVolumeMaxRatio5").ge(1.5)
    volume_elevated = numeric(out, "SPYVolumeMaxPct5").ge(80.0) | numeric(out, "SPYVolumeMaxRatio5").ge(1.2)
    out["VolumeConfirmation"] = np.select(
        [volume_extreme, volume_elevated], ["EXTREME", "ELEVATED"], default="NORMAL"
    )


def add_historical_correction_metadata(out: pd.DataFrame) -> None:
    out["HistoricalCorrectionFinalLowDate"] = pd.NaT
    out["HistoricalCorrectionFinalLowPrice"] = np.nan
    out["HistoricalCorrectionRecoveryDate"] = pd.NaT
    out["HistoricalCorrectionDurationDays"] = np.nan
    out["HistoricalCorrectionMaxDrawdown"] = np.nan
    cycle_ids = out["CorrectionCycleID"].dropna().unique().tolist()
    for cycle_id in cycle_ids:
        mask = out["CorrectionCycleID"].eq(cycle_id)
        rows = out.loc[mask]
        if rows.empty:
            continue
        low_idx = numeric(rows, "SPY_Low").idxmin()
        recovery_dates = pd.to_datetime(rows.loc[rows["CorrectionRecoveryDate"].notna(), "CorrectionRecoveryDate"], errors="coerce")
        recovery_date = recovery_dates.iloc[-1] if not recovery_dates.empty else pd.NaT
        start_date = pd.to_datetime(rows["CorrectionStartDate"], errors="coerce").dropna().min()
        end_date = recovery_date if pd.notna(recovery_date) else pd.to_datetime(rows["Date"], errors="coerce").max()
        out.loc[mask, "HistoricalCorrectionFinalLowDate"] = out.loc[low_idx, "Date"]
        out.loc[mask, "HistoricalCorrectionFinalLowPrice"] = safe_float(out.loc[low_idx, "SPY_Low"])
        out.loc[mask, "HistoricalCorrectionRecoveryDate"] = recovery_date
        out.loc[mask, "HistoricalCorrectionDurationDays"] = (end_date - start_date).days if pd.notna(start_date) else np.nan
        out.loc[mask, "HistoricalCorrectionMaxDrawdown"] = numeric(rows, "CorrectionCurrentDrawdown").min()


def add_compatibility_columns(out: pd.DataFrame) -> None:
    out["CorrectionDrawdown"] = out["CorrectionCurrentDrawdown"]
    out["CorrectionWorstDrawdown"] = out["CorrectionWorstDrawdown"]
    out["CorrectionPeak"] = out["CorrectionPeakPrice"]
    out["CorrectionState"] = out["CorrectionStatus"]
    out["CorrectionType"] = out["CorrectionSeverity"]
    event = np.where(out["DurableBottomConfirmed"], "DURABLE BOTTOM CONFIRMED", "")
    event = np.where(out["ExtremeEpisodeDisplayMarker"], "EXTREME CAPITULATION", event)
    event = np.where(out["ExtremeCapitulation2EpisodeMarker"], "EXTREME CAPITULATION 2", event)
    both = out["DurableBottomConfirmed"] & out["ExtremeEpisodeDisplayMarker"]
    event = np.where(both, "EXTREME CAPITULATION + DURABLE BOTTOM CONFIRMED", event)
    both_2 = out["DurableBottomConfirmed"] & out["ExtremeCapitulation2EpisodeMarker"]
    event = np.where(both_2, "EXTREME CAPITULATION 2 + DURABLE BOTTOM CONFIRMED", event)
    out["CorrectionEventType"] = event


def lifecycle_columns() -> list[str]:
    return [
        "CorrectionCycleID",
        "CorrectionWaveID",
        "CorrectionActive",
        "CorrectionPeakDate",
        "CorrectionPeakPrice",
        "CorrectionStartDate",
        "CorrectionCurrentDrawdown",
        "CorrectionWorstDrawdown",
        "CorrectionLowDate",
        "CorrectionLowPrice",
        "CorrectionStatus",
        "CorrectionSeverity",
        "CorrectionRecoveryDate",
        "WaveStartDate",
        "WaveRecoveryHigh",
        "WaveWorstDrawdown",
        "TacticalEligible",
        "TacticalInternalEvent",
        "PostTacticalRecoveryHigh",
        "DurableBottomConfirmed",
        "DurableRouteTrendBreadth",
        "DurableRouteCreditStress",
        "DurableRoute",
        "BearRiskScore",
        "BearRiskCategory",
        "BearRisk_SPYBelowSMA200",
        "BearRisk_SMA200SlopeNegative",
        "BearRisk_ROC3MNegative",
        "BearRisk_ROC6MNegative",
        "BearRisk_HYOASPct90",
        "BearRisk_HY20DNonImproving",
        "LatestDurableBottomDate",
        "LatestDurableBottomPrice",
        "LatestDurableRoute",
        "LatestDurableBearRiskScore",
        "LatestDurableBearRiskCategory",
    ]


def append_lifecycle_row(
    result: dict[str, list[Any]],
    *,
    cycle_id: str | None,
    wave_id: str | None,
    active: bool,
    peak_date: Any,
    peak_price: Any,
    start_date: Any,
    drawdown: Any,
    worst_drawdown: Any,
    low_date: Any,
    low_price: Any,
    status: str,
    correction_type: str,
    wave_start_date: Any,
    wave_recovery_high: Any,
    wave_worst: Any,
    tactical_eligible: bool,
    tactical_event: bool,
    post_tactical_high: Any,
    durable: bool,
    trend_route: bool,
    credit_route: bool,
    route: str,
    bear_values: tuple[Any, ...],
    latest_durable: tuple[Any, Any, str, Any, str],
    recovery_date: Any,
) -> None:
    bear_score, bear_category, *bear_flags = bear_values
    latest_date, latest_price, latest_route, latest_score, latest_category = latest_durable
    values = [
        cycle_id,
        wave_id,
        active,
        peak_date,
        peak_price,
        start_date,
        drawdown,
        worst_drawdown,
        low_date,
        low_price,
        status,
        correction_type,
        recovery_date,
        wave_start_date,
        wave_recovery_high,
        wave_worst,
        tactical_eligible,
        tactical_event,
        post_tactical_high,
        durable,
        trend_route,
        credit_route,
        route,
        bear_score,
        bear_category,
        *bear_flags,
        latest_date,
        latest_price,
        latest_route,
        latest_score,
        latest_category,
    ]
    for column, value in zip(lifecycle_columns(), values):
        result[column].append(value)


def append_unavailable_lifecycle(result: dict[str, list[Any]]) -> None:
    for column in lifecycle_columns():
        if column in {"CorrectionActive", "TacticalEligible", "TacticalInternalEvent", "DurableBottomConfirmed"}:
            result[column].append(False)
        elif column.endswith("Date"):
            result[column].append(pd.NaT)
        elif column in {"CorrectionStatus", "CorrectionSeverity", "BearRiskCategory", "LatestDurableBearRiskCategory"}:
            result[column].append("DATA INCOMPLETE")
        elif column in {"CorrectionCycleID", "CorrectionWaveID", "DurableRoute", "LatestDurableRoute"}:
            result[column].append(None if column.endswith("ID") else "")
        else:
            result[column].append(np.nan)


def bear_risk_values(
    price: Any,
    sma200: Any,
    slope20: Any,
    roc3m: Any,
    roc6m: Any,
    hy_pct: Any,
    hy_change: Any,
) -> tuple[Any, ...]:
    raw = [price, sma200, slope20, roc3m, roc6m, hy_pct, hy_change]
    if not all(np.isfinite(safe_float(value)) for value in raw):
        return (np.nan, "DATA INCOMPLETE", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA)
    flags = (
        bool(price < sma200),
        bool(slope20 < 0),
        bool(roc3m < 0),
        bool(roc6m < 0),
        bool(hy_pct >= 90.0),
        bool(hy_change >= 0),
    )
    score = int(sum(flags))
    category = "LOW BEAR RISK" if score <= 2 else "MODERATE BEAR RISK" if score == 3 else "HIGH BEAR RISK"
    return (score, category, *flags)


def rolling_percentile_pit(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = []
    for idx, value in enumerate(values):
        history = values.iloc[max(0, idx - window + 1) : idx + 1].dropna()
        if not np.isfinite(safe_float(value)) or len(history) < min_periods:
            result.append(np.nan)
        else:
            result.append(float((history <= value).mean() * 100.0))
    return pd.Series(result, index=series.index, dtype="float64")


def rolling_percentile_masked(
    series: pd.Series, observed: pd.Series, window: int, min_periods: int
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid_mask = observed.fillna(False).astype(bool)
    result = []
    for idx, value in enumerate(values):
        start = max(0, idx - window + 1)
        history = values.iloc[start : idx + 1][valid_mask.iloc[start : idx + 1]].dropna()
        if not np.isfinite(safe_float(value)) or len(history) < min_periods:
            result.append(np.nan)
        else:
            result.append(float((history <= value).mean() * 100.0))
    return pd.Series(result, index=series.index, dtype="float64")


def valid_observation_change(series: pd.Series, observed: pd.Series, periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.loc[observed.fillna(False).astype(bool)].dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    change = valid.diff(periods)
    return change.reindex(series.index).ffill()


def observed_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    marker = f"{column}_Observed"
    if marker in frame:
        return frame[marker].fillna(False).astype(bool)
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce").notna()


def classify_breadth_frequency(mask50: pd.Series, mask200: pd.Series) -> str:
    ratios = [float(mask.tail(252).mean()) for mask in (mask50, mask200) if len(mask)]
    ratio = min(ratios) if ratios else 0.0
    if ratio >= 0.80:
        return "DAILY"
    if ratio >= 0.12:
        return "IRREGULAR / VALID-OBSERVATION"
    return "UNAVAILABLE"


def wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss.ne(0.0), 100.0)


def safe_retreat(current: pd.Series, peak: pd.Series) -> pd.Series:
    current_values = pd.to_numeric(current, errors="coerce")
    peak_values = pd.to_numeric(peak, errors="coerce").replace(0.0, np.nan)
    return (peak_values - current_values) / peak_values


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def all_true(*conditions: Any) -> bool:
    return bool(conditions) and all(condition is not pd.NA and bool(condition) for condition in conditions)


def any_true(*conditions: Any) -> bool:
    return any(condition is not pd.NA and bool(condition) for condition in conditions)


def correction_status(drawdown: Any) -> str:
    value = safe_float(drawdown)
    if not np.isfinite(value) or value > -0.05:
        return "NORMAL"
    if value <= -0.20:
        return "BEAR / CRASH"
    if value <= -0.10:
        return "DEEP CORRECTION"
    return "ACTIVE CORRECTION"


def correction_severity(drawdown: Any) -> str:
    value = safe_float(drawdown)
    if not np.isfinite(value) or value > -0.05:
        return "NORMAL"
    if value <= -0.20:
        return "BEAR / CRASH"
    if value <= -0.15:
        return "DEEP CORRECTION"
    if value <= -0.10:
        return "SIGNIFICANT CORRECTION"
    return "PULLBACK / CORRECTION"

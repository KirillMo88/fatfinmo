from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_cycle import build_market_cycle_snapshot


OUTPUT_DIR = Path("/tmp/correction_bottom_v2_data")
REGRESSION_WINDOWS = [
    ("2007-2009", "2007-07-01", "2009-06-30"),
    ("2011", "2011-06-01", "2012-01-31"),
    ("2015-2016", "2015-07-01", "2016-04-30"),
    ("2018-2019", "2018-09-01", "2019-03-31"),
    ("2020", "2020-02-01", "2020-06-30"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2025", "2025-02-01", "2025-06-30"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = build_market_cycle_snapshot()
    history = snapshot.correction_daily.copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history = history.loc[history["Date"].ge(pd.Timestamp("2005-01-01"))].sort_values("Date").reset_index(drop=True)

    daily_columns = [
        "Date",
        "SPY_Close",
        "SPY_Low",
        "SPY_Volume",
        "VIX",
        "SPXAboveSMA50D",
        "SPXAboveSMA200D",
        "BreadthSourceFrequency",
        "HY_OAS",
        "HYOASSourceFrequency",
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
        "WaveStartDate",
        "WaveRecoveryHigh",
        "WaveWorstDrawdown",
        "ExtremeCapRaw",
        "ExtremeEpisodeID",
        "ExtremeEpisodeMarker",
        "ExtremeEpisodeDisplayMarker",
        "ExtremeEpisodeHasLevel2",
        "ExtremeCapitulation2Raw",
        "ExtremeCapitulation2EpisodeID",
        "ExtremeCapitulation2EpisodeMarker",
        "ExtremeCapitulation2EpisodeStart",
        "ExtremeCapitulation2EpisodeEnd",
        "extreme_capitulation",
        "extreme_capitulation_2",
        "ExtremeStressMean5",
        "ExtremeConditionsMet",
        "VIX_Stress",
        "RSI_Stress",
        "SPY5D_DownsideStress",
        "Breadth50_Stress",
        "Breadth200_Stress",
        "SPYVolumePct1Y",
        "SPYVolumeRatio20",
        "SPYVolumeMaxPct5",
        "SPYVolumeMaxRatio5",
        "VolumeConfirmation",
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
        "SPY_RSI14",
        "SPY_5D_Return",
        "SPY_SMA200D",
        "SMA200D_Slope20D",
        "SPY_ROC3M",
        "SPY_ROC6M",
        "Previous10DHigh",
        "VIX_20D_Peak",
        "VIXRetreat20D",
        "Breadth50_5D_Change",
        "Breadth200_5D_Change",
        "HYOAS20DChange",
        "HYOAS20DPeak",
        "HYOASRetreat20D",
        "HYOASPercentile3Y",
        "CorrectionDataCoverage",
        "CorrectionModelVersion",
    ]
    daily = history[[column for column in daily_columns if column in history.columns]].copy()
    write_csv(daily, "daily.csv")

    signal_mask = (
        history["ExtremeEpisodeDisplayMarker"].fillna(False)
        | history["ExtremeCapitulation2EpisodeMarker"].fillna(False)
        | history["DurableBottomConfirmed"].fillna(False)
    )
    signals = daily.loc[signal_mask].copy()
    signals.insert(
        1,
        "Event",
        np.select(
            [
                signals["ExtremeCapitulation2EpisodeMarker"].fillna(False) & signals["DurableBottomConfirmed"].fillna(False),
                signals["ExtremeEpisodeDisplayMarker"].fillna(False) & signals["DurableBottomConfirmed"].fillna(False),
                signals["ExtremeCapitulation2EpisodeMarker"].fillna(False),
                signals["ExtremeEpisodeDisplayMarker"].fillna(False),
                signals["DurableBottomConfirmed"].fillna(False),
            ],
            [
                "EXTREME CAPITULATION 2 + DURABLE",
                "EXTREME CAPITULATION + DURABLE",
                "EXTREME CAPITULATION 2",
                "EXTREME CAPITULATION",
                "DURABLE BOTTOM CONFIRMED",
            ],
            default="",
        ),
    )
    write_csv(signals, "signals.csv")

    debug_parts = []
    debug_columns = [
        "Date",
        "SPY_Close",
        "CorrectionCurrentDrawdown",
        "CorrectionCycleID",
        "CorrectionWaveID",
        "TacticalInternalEvent",
        "TacticalEligible",
        "ExtremeCapRaw",
        "ExtremeEpisodeMarker",
        "ExtremeCapitulation2Raw",
        "ExtremeCapitulation2EpisodeMarker",
        "DurableBottomConfirmed",
        "DurableRoute",
        "BearRiskScore",
        "BearRiskCategory",
        "VIXRetreat20D",
        "Breadth50_5D_Change",
        "Breadth200_5D_Change",
        "HYOAS20DChange",
        "HYOASPercentile3Y",
    ]
    for label, start, end in REGRESSION_WINDOWS:
        part = history.loc[history["Date"].between(pd.Timestamp(start), pd.Timestamp(end)), debug_columns].copy()
        part.insert(0, "Window", label)
        debug_parts.append(part)
    debug = pd.concat(debug_parts, ignore_index=True)
    write_csv(debug, "regression_windows.csv")

    episode_rows = []
    for cycle_id, rows in history.dropna(subset=["CorrectionCycleID"]).groupby("CorrectionCycleID", sort=False):
        rows = rows.sort_values("Date")
        episode_rows.append(
            {
                "CorrectionCycleID": cycle_id,
                "PeakDate": first_value(rows, "CorrectionPeakDate"),
                "PeakPrice": first_value(rows, "CorrectionPeakPrice"),
                "StartDate": first_value(rows, "CorrectionStartDate"),
                "FinalLowDate": first_value(rows, "HistoricalCorrectionFinalLowDate"),
                "FinalLowPrice": first_value(rows, "HistoricalCorrectionFinalLowPrice"),
                "RecoveryDate": first_value(rows, "HistoricalCorrectionRecoveryDate"),
                "MaxDrawdown": pd.to_numeric(rows["CorrectionCurrentDrawdown"], errors="coerce").min(),
                "Waves": rows["CorrectionWaveID"].nunique(),
                "ExtremeEpisodes": int(rows["ExtremeEpisodeMarker"].fillna(False).sum()),
                "Extreme2Episodes": int(rows["ExtremeCapitulation2EpisodeMarker"].fillna(False).sum()),
                "DurableSignals": int(rows["DurableBottomConfirmed"].fillna(False).sum()),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    write_csv(episodes, "episodes.csv")

    invariants = {
        "Tactical outside active correction": int((history["TacticalInternalEvent"] & ~history["CorrectionActive"]).sum()),
        "Durable without same-day Tactical": int((history["DurableBottomConfirmed"] & ~history["TacticalInternalEvent"]).sum()),
        "Missing breadth interpreted as zero": int((history["SPXAboveSMA50D"].isna() & history["Breadth50_Stress"].eq(0)).sum()),
        "Missing HY interpreted as Bear Risk zero": int((history["HY_OAS"].isna() & history["BearRiskScore"].eq(0)).sum()),
    }
    summary = {
        "model_version": str(history["CorrectionModelVersion"].iloc[-1]),
        "history_start": str(history["Date"].min().date()),
        "history_end": str(history["Date"].max().date()),
        "daily_rows": int(len(history)),
        "master_corrections": int(history["CorrectionCycleID"].nunique()),
        "correction_waves": int(history["CorrectionWaveID"].nunique()),
        "extreme_episodes": int(history["ExtremeEpisodeMarker"].sum()),
        "extreme_2_episodes": int(history["ExtremeCapitulation2EpisodeMarker"].sum()),
        "durable_signals": int(history["DurableBottomConfirmed"].sum()),
        "breadth_frequency": str(history["BreadthSourceFrequency"].iloc[-1]),
        "hy_coverage_pct": float(history["HY_OAS"].notna().mean() * 100.0),
        "hy_sources": {str(key): int(value) for key, value in history["HYOASSourceFrequency"].value_counts(dropna=False).items()},
        "invariants": invariants,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def first_value(frame: pd.DataFrame, column: str):
    if column not in frame:
        return np.nan
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else np.nan


def write_csv(frame: pd.DataFrame, name: str) -> None:
    copy = frame.copy()
    for column in copy.columns:
        if pd.api.types.is_datetime64_any_dtype(copy[column]):
            copy[column] = copy[column].dt.strftime("%Y-%m-%d")
    copy.to_csv(OUTPUT_DIR / name, index=False)


if __name__ == "__main__":
    main()

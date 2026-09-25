from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_cycle import build_market_cycle_snapshot


OUTPUT_DIR = Path("/tmp/current_risk_v1_export")
START_DATE = pd.Timestamp("2015-01-01")
END_DATE = pd.Timestamp("2026-12-31")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = build_market_cycle_snapshot()
    data = snapshot.daily.copy()
    data.index.name = None
    data["Date"] = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data["Date"].between(START_DATE, END_DATE)].sort_values("Date").reset_index(drop=True)

    columns = [
        "Date",
        "CurrentRiskModelVersion",
        "SPY_Close",
        "SPY_RawClose",
        "QQQ",
        "QQQ_RawClose",
        "VIX",
        "SPXAboveSMA50D",
        "HY_OAS",
        "HYOASSourceFrequency",
        "SPY_SMA200W",
        "SPY_WeeklyRSI14",
        "SPY_26W_High",
        "SPY_WeeklyRSI26WMax",
        "SPX_1M_DD20_Probability",
        "CurrentRiskSPY20DHigh",
        "CurrentRiskROC20D",
        "CurrentRiskROC20DMax",
        "CurrentRiskROC12",
        "CurrentRiskROC12Max20D",
        "CurrentRiskROCDivergenceGap",
        "CurrentRiskBreadth20DMax",
        "CurrentRiskBreadth5DChange",
        "CurrentRiskDailyRSI14",
        "CurrentRiskDailyRSI20DMax",
        "CurrentRiskQQQSPY",
        "CurrentRiskQQQSPY20DHigh",
        "CurrentRiskQQQDivergenceDepth",
        "CurrentRiskHY5DChange",
        "CurrentRiskHY10YPercentile",
        "CurrentRiskSPYExtension200W",
        "CurrentRiskSPYRawClose",
        "CurrentRiskQQQRawClose",
        "CurrentRiskSPYSMA200WRaw",
        "CurrentRiskPriceBasis",
        "PVC_ExtensionCondition",
        "PVC_NearHighCondition",
        "PVC_ROCCondition",
        "PVC_BreadthCondition",
        "PVC_DailyRSIDivergence",
        "PVC_WeeklyRSIDivergence",
        "PVC_BlowoffRSI",
        "PVC_V2",
        "PVC_MAX10D",
        "CurrentRiskVIXPrevious3DHigh",
        "CurrentRiskVIX3DChange",
        "CurrentRiskActivation",
        "CurrentRiskNewEvent",
        "CurrentRiskCalmVIXBase",
        "CurrentRiskHYLevelConfirmation",
        "CurrentRiskHYWidening",
        "CurrentRiskWeakBreadth",
        "CurrentRiskBreadthCollapse",
        "CurrentRiskTopConfirmation",
        "CurrentRiskCreditConfirmation",
        "CurrentRiskEscalationScoreRaw",
        "CurrentRiskEscalationScoreV2",
        "CurrentRiskSignalClass",
        "CurrentRiskDrawdownRisk",
        "CurrentRiskDrawdownRiskState",
        "CurrentRiskPriceCycleVulnerabilityRisk",
        "CurrentRiskPriceCycleVulnerabilityRiskState",
        "CurrentRiskBreadthRisk",
        "CurrentRiskBreadthRiskState",
        "CurrentRiskRSIDivergenceRisk",
        "CurrentRiskRSIDivergenceRiskState",
        "CurrentRiskVIXRisk",
        "CurrentRiskVIXRiskState",
        "CurrentRiskHighBetaRisk",
        "CurrentRiskHighBetaRiskState",
        "CurrentRiskHYRisk",
        "CurrentRiskHYRiskState",
        "CurrentRiskComponentAverage",
        "CurrentMarketRiskState",
        "CurrentMarketRiskDirection",
        "ActiveStressChannels",
        "CurrentRiskDataCoverage",
    ]
    export = data[[column for column in columns if column in data]].copy()
    export.to_csv(OUTPUT_DIR / "current_risk_daily_2015_2026.csv", index=False)
    raw_activations = export.loc[export["CurrentRiskActivation"].fillna(False).astype(bool)].copy()
    raw_activations.to_csv(OUTPUT_DIR / "current_risk_activation_days_2015_2026.csv", index=False)
    events = export.loc[export["CurrentRiskNewEvent"].fillna(False).astype(bool)].copy()
    events.to_csv(OUTPUT_DIR / "current_risk_events_2015_2026.csv", index=False)

    audit = pd.DataFrame(
        {
            "Date": export["Date"],
            "SPY_Close": export["CurrentRiskSPYRawClose"],
            "SPY raw Close": export["CurrentRiskSPYRawClose"],
            "SMA200W raw": export["CurrentRiskSPYSMA200WRaw"],
            "SPY Extension %": 100.0 * pd.to_numeric(export["CurrentRiskSPYExtension200W"], errors="coerce"),
            "CurrentRiskROC12": export["CurrentRiskROC12"],
            "CurrentRiskROC12Max20D": export["CurrentRiskROC12Max20D"],
            "CurrentRiskROCDivergenceGap": export["CurrentRiskROCDivergenceGap"],
            "PVC_ROCCondition": export["PVC_ROCCondition"],
            "PVC_V2": export["PVC_V2"],
            "PVC_MAX10D": export["PVC_MAX10D"],
            "CurrentRiskActivation": export["CurrentRiskActivation"],
            "CurrentRiskNewEvent": export["CurrentRiskNewEvent"],
            "EscalationScoreV2": export["CurrentRiskEscalationScoreV2"],
            "SignalClass": export["CurrentRiskSignalClass"],
            "TopConfirmation": export["CurrentRiskTopConfirmation"],
            "CreditConfirmation": export["CurrentRiskCreditConfirmation"],
        }
    )
    audit.to_csv(OUTPUT_DIR / "current_risk_audit_2015_2026.csv", index=False)

    summary = {
        "model": str(export["CurrentRiskModelVersion"].dropna().iloc[-1]),
        "start": str(export["Date"].min().date()),
        "end": str(export["Date"].max().date()),
        "rows": int(len(export)),
        "raw_activation_days": int(len(raw_activations)),
        "independent_events": int(len(events)),
        "signal_classes": {str(k): int(v) for k, v in events["CurrentRiskSignalClass"].value_counts().items()},
        "hy_sources": {str(k): int(v) for k, v in export["HYOASSourceFrequency"].value_counts(dropna=False).items()},
        "latest_state": str(export["CurrentMarketRiskState"].iloc[-1]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_entry_risk(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["entry_risk"]
    out = pd.DataFrame(index=df.index)
    out["Regime_Dependent_SMAZ_Risk"] = [
        regime_dependent_smaz_risk(regime, smaz)
        for regime, smaz in zip(df["Market_Regime"], pd.to_numeric(df["SMA200d_Robust_Z_36M"], errors="coerce"))
    ]
    out["Perf12M_Extreme_Risk"] = perf12m_extreme_risk(
        pd.to_numeric(df["Perf_12M_Percentile"], errors="coerce"),
        soft_threshold=float(cfg["perf12_soft_threshold"]),
        extreme_threshold=float(cfg["perf12_extreme_threshold"]),
        soft_risk=float(cfg["perf12_soft_risk"]),
        extreme_risk=float(cfg["perf12_extreme_risk"]),
    )
    out["Entry_Risk_Score"] = (
        out["Regime_Dependent_SMAZ_Risk"] + out["Perf12M_Extreme_Risk"]
    ).clip(0.0, 100.0)

    stress_mask = (
        df["Market_Regime"].eq("STRESS")
        & (pd.to_numeric(df["Momentum_Score"], errors="coerce") >= float(cfg["stress_momentum_threshold"]))
        & (pd.to_numeric(df["SMA200d_Robust_Z_36M"], errors="coerce") > float(cfg["stress_smaz_extreme"]))
    )
    out.loc[stress_mask, "Entry_Risk_Score"] = np.maximum(
        out.loc[stress_mask, "Entry_Risk_Score"],
        float(cfg["stress_min_entry_risk"]),
    )
    out["Entry_Risk"] = out["Entry_Risk_Score"].apply(classify_entry_risk)
    out["Opportunity_Score"] = (
        pd.to_numeric(df["Alpha_Score"], errors="coerce") * (1.0 - (out["Entry_Risk_Score"] / 100.0))
    ).clip(0.0, 100.0)
    return out


def regime_dependent_smaz_risk(regime: str, smaz: float) -> float:
    if not np.isfinite(smaz):
        return np.nan
    if regime == "BULL":
        return 0.0 if smaz <= 3.0 else 10.0
    if regime == "BULL_HIGH_VOL":
        if smaz <= 2.5:
            return 0.0
        if smaz <= 3.0:
            return 5.0
        return 10.0
    if regime == "CORRECTION":
        if smaz <= 1.5:
            return 10.0
        if smaz <= 2.5:
            return 20.0
        if smaz <= 3.0:
            return 30.0
        return 50.0
    if regime == "STRESS":
        if smaz <= 1.5:
            return 20.0
        if smaz <= 2.5:
            return 30.0
        if smaz <= 3.0:
            return 50.0
        return 80.0
    return np.nan


def perf12m_extreme_risk(
    perf12m_percentile: pd.Series,
    soft_threshold: float,
    extreme_threshold: float,
    soft_risk: float,
    extreme_risk: float,
) -> pd.Series:
    values = pd.to_numeric(perf12m_percentile, errors="coerce").replace([np.inf, -np.inf], np.nan)
    risk = pd.Series(np.nan, index=values.index, dtype="float64")
    risk.loc[values.notna() & (values <= soft_threshold)] = 0.0
    risk.loc[values.notna() & (values > soft_threshold) & (values <= extreme_threshold)] = soft_risk
    risk.loc[values.notna() & (values > extreme_threshold)] = extreme_risk
    return risk


def classify_entry_risk(score: float) -> str:
    if not np.isfinite(score):
        return "Missing Data"
    if score <= 20.0:
        return "Low"
    if score <= 40.0:
        return "Moderate"
    if score <= 60.0:
        return "Elevated"
    if score <= 80.0:
        return "High"
    return "Extreme"

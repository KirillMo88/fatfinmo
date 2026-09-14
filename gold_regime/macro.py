from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GOLD_REGIME_CONFIG
from .utils import classify_risk, classify_score, fred_series_weekly, rolling_percentile_rank, union_index


STRUCTURAL_MACRO_LABELS = ("STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH")


def calculate_gold_macro_history(
    dxy: pd.Series,
    real_yield: pd.Series,
    us2y: pd.Series,
    wti: pd.Series,
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = config or GOLD_REGIME_CONFIG
    percentile_cfg = cfg["percentile"]
    structural_cfg = cfg["structural_macro"]
    forward_cfg = cfg["forward_macro_risk"]

    index = union_index([dxy, real_yield, us2y, wti])
    if index.empty:
        return pd.DataFrame(columns=gold_macro_columns())

    dxy_values = pd.to_numeric(dxy, errors="coerce").sort_index().reindex(index).ffill()
    real_yield_values = pd.to_numeric(real_yield, errors="coerce").sort_index().reindex(index).ffill()
    us2y_values = pd.to_numeric(us2y, errors="coerce").sort_index().reindex(index).ffill()
    wti_values = pd.to_numeric(wti, errors="coerce").sort_index().reindex(index).ffill()

    percentile_window = int(percentile_cfg["window_weeks"])
    percentile_min = int(percentile_cfg["minimum_weeks"])
    macro_window = int(structural_cfg["window_weeks"])

    dxy_13w = dxy_values.pct_change(macro_window)
    dxy_percentile = rolling_percentile_rank(dxy_13w, percentile_window, percentile_min)
    dxy_bull_score = 100.0 - dxy_percentile

    real_yield_change_13w_bp = (real_yield_values - real_yield_values.shift(macro_window)) * 100.0
    real_yield_percentile = rolling_percentile_rank(real_yield_change_13w_bp, percentile_window, percentile_min)
    real_yield_bull_score = 100.0 - real_yield_percentile

    us2y_change_13w_bp = (us2y_values - us2y_values.shift(int(forward_cfg["us2y_window_weeks"]))) * 100.0
    us2y_percentile = rolling_percentile_rank(us2y_change_13w_bp, percentile_window, percentile_min)
    us2y_bull_score = 100.0 - us2y_percentile
    us2y_risk_score = us2y_percentile

    wti_26w = wti_values.pct_change(int(forward_cfg["wti_window_weeks"]))
    wti_percentile = rolling_percentile_rank(wti_26w, percentile_window, percentile_min)
    wti_risk_score = wti_percentile

    structural_macro_score = (
        float(structural_cfg["dxy_weight"]) * dxy_bull_score
        + float(structural_cfg["real_yield_weight"]) * real_yield_bull_score
        + float(structural_cfg["us2y_weight"]) * us2y_bull_score
    ).clip(0.0, 100.0)
    forward_macro_risk = (
        float(forward_cfg["us2y_weight"]) * us2y_risk_score
        + float(forward_cfg["wti_weight"]) * wti_risk_score
    ).clip(0.0, 100.0)

    out = pd.DataFrame(
        {
            "date": index,
            "dxy": dxy_values.values,
            "dxy_13w": dxy_13w.values,
            "dxy_percentile": dxy_percentile.values,
            "dxy_score": dxy_bull_score.values,
            "real_yield": real_yield_values.values,
            "real_yield_change_13w": real_yield_change_13w_bp.values,
            "real_yield_score": real_yield_bull_score.values,
            "real_yield_percentile": real_yield_percentile.values,
            "us2y": us2y_values.values,
            "us2y_change_13w": us2y_change_13w_bp.values,
            "us2y_bull_score": us2y_bull_score.values,
            "us2y_risk_score": us2y_risk_score.values,
            "us2y_percentile": us2y_percentile.values,
            "wti": wti_values.values,
            "wti_26w": wti_26w.values,
            "wti_percentile": wti_percentile.values,
            "wti_risk_score": wti_risk_score.values,
            "structural_macro_score": structural_macro_score.values,
            "forward_macro_risk": forward_macro_risk.values,
        }
    )
    out["structural_macro_state"] = out["structural_macro_score"].map(classify_structural_macro_state)
    out["forward_macro_risk_state"] = out["forward_macro_risk"].map(classify_risk)
    return out


def calculate_gold_macro_from_fred(
    dxy: pd.Series,
    wti: pd.Series,
    fred_data: pd.DataFrame | None,
    config: dict | None = None,
) -> pd.DataFrame:
    fred_weekly = fred_series_weekly(fred_data)
    return calculate_gold_macro_history(
        dxy=dxy,
        real_yield=fred_weekly.get("DFII10", pd.Series(dtype="float64")),
        us2y=fred_weekly.get("DGS2", pd.Series(dtype="float64")),
        wti=wti,
        config=config,
    )


def classify_structural_macro_state(score: float) -> str:
    return classify_score(float(score), STRUCTURAL_MACRO_LABELS) if np.isfinite(score) else "DATA_INCOMPLETE"


def gold_macro_columns() -> list[str]:
    return [
        "date",
        "dxy",
        "dxy_13w",
        "dxy_percentile",
        "dxy_score",
        "real_yield",
        "real_yield_change_13w",
        "real_yield_score",
        "real_yield_percentile",
        "us2y",
        "us2y_change_13w",
        "us2y_bull_score",
        "us2y_risk_score",
        "us2y_percentile",
        "wti",
        "wti_26w",
        "wti_percentile",
        "wti_risk_score",
        "structural_macro_score",
        "forward_macro_risk",
        "structural_macro_state",
        "forward_macro_risk_state",
    ]

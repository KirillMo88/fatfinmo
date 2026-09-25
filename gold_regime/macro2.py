from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


PERCENTILE_WINDOW_WEEKS = 156
PERCENTILE_MINIMUM_WEEKS = 104
HORIZONS = ("3M", "6M", "9M", "12M")

FINAL_SCORE_LABELS = (
    "STRONGLY_SUPPORTIVE",
    "SUPPORTIVE",
    "NEUTRAL_MIXED",
    "UNFAVORABLE",
    "STRONGLY_UNFAVORABLE",
)

BUSINESS_CYCLE_MODIFIERS = {
    "EARLY RECOVERY": {"3M": 0.0, "6M": 3.0, "9M": 7.0, "12M": 10.0},
    "DETERIORATING CONTRACTION": {"3M": 0.0, "6M": 2.0, "9M": 5.0, "12M": 7.0},
    "LATE / SLOWING EXPANSION": {"3M": 0.0, "6M": 0.0, "9M": 0.0, "12M": 0.0},
    "STRONG EXPANSION": {"3M": 0.0, "6M": -3.0, "9M": -7.0, "12M": -10.0},
}


def inclusive_trailing_percentile(
    values: pd.Series,
    window: int = PERCENTILE_WINDOW_WEEKS,
    minimum: int = PERCENTILE_MINIMUM_WEEKS,
) -> pd.Series:
    """Point-in-time percentile using the current value in its trailing window."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def percentile(window_values: np.ndarray) -> float:
        current = window_values[-1]
        if not np.isfinite(current):
            return np.nan
        valid = window_values[np.isfinite(window_values)]
        if len(valid) < minimum:
            return np.nan
        return float((np.sum(valid <= current) / len(valid)) * 100.0)

    return numeric.rolling(window=window, min_periods=minimum).apply(percentile, raw=True)


def _union_index(series_list: Iterable[pd.Series]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for series in series_list:
        if series is None or series.empty:
            continue
        values = pd.to_datetime(series.dropna().index, errors="coerce")
        values = values[~values.isna()]
        index = index.union(pd.DatetimeIndex(values))
    return index.sort_values()


def _aligned_numeric(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    values.index = pd.to_datetime(values.index, errors="coerce")
    values = values[~values.index.isna()].sort_index()
    return values.reindex(index).ffill()


def _aligned_state(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="object")
    values = series.astype("object")
    values.index = pd.to_datetime(values.index, errors="coerce")
    values = values[~values.index.isna()].sort_index()
    return values.reindex(index).ffill()


def _clip(series: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)


def _score_state(score: float) -> str:
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score >= 80.0:
        return FINAL_SCORE_LABELS[0]
    if score >= 60.0:
        return FINAL_SCORE_LABELS[1]
    if score >= 40.0:
        return FINAL_SCORE_LABELS[2]
    if score >= 20.0:
        return FINAL_SCORE_LABELS[3]
    return FINAL_SCORE_LABELS[4]


def _liquidity_block(values: pd.Series, prefix: str) -> dict[str, pd.Series]:
    growth = values.pct_change(52, fill_method=None)
    fast = growth - growth.shift(13)
    medium = growth - growth.shift(26)
    slow = growth - growth.shift(39)
    growth_pct = inclusive_trailing_percentile(growth)
    fast_pct = inclusive_trailing_percentile(fast)
    medium_pct = inclusive_trailing_percentile(medium)
    slow_pct = inclusive_trailing_percentile(slow)
    impulse_3m = 0.67 * fast_pct + 0.33 * medium_pct
    impulse_6m = 0.33 * fast_pct + 0.34 * medium_pct + 0.33 * slow_pct
    factor_3m = _clip(0.40 * growth_pct + 0.60 * impulse_3m)
    factor_6m = _clip(0.30 * growth_pct + 0.70 * impulse_6m)
    return {
        f"{prefix}_Growth": growth,
        f"{prefix}_FastImpulse": fast,
        f"{prefix}_MediumImpulse": medium,
        f"{prefix}_SlowImpulse": slow,
        f"{prefix}_GrowthPct": growth_pct,
        f"{prefix}_FastPct": fast_pct,
        f"{prefix}_MediumPct": medium_pct,
        f"{prefix}_SlowPct": slow_pct,
        f"{prefix}_Impulse_3M": impulse_3m,
        f"{prefix}_Impulse_6M": impulse_6m,
        f"{prefix}_Factor_3M": factor_3m,
        f"{prefix}_Factor_6M": factor_6m,
    }


def _business_cycle_modifier(state: Any, horizon: str) -> float:
    label = str(state).strip().upper() if state is not None and not pd.isna(state) else ""
    return BUSINESS_CYCLE_MODIFIERS.get(label, {}).get(horizon, np.nan)


def _rate_regime(us2y_change: float, us10y_change: float, curve_change: float, jp_shock: float) -> tuple[str, float]:
    if not all(np.isfinite(value) for value in (us2y_change, us10y_change, curve_change)):
        return "MIXED", 0.0
    if us2y_change < 0.0 and us10y_change < 0.0 and curve_change > 0.0:
        return "BULL STEEPENING", 15.0
    if us2y_change > 0.0 and us10y_change > 0.0 and curve_change > 0.0:
        if np.isfinite(jp_shock) and jp_shock >= 80.0:
            return "BEAR STEEPENING + SOVEREIGN STRESS", 10.0
        if np.isfinite(jp_shock) and jp_shock < 80.0:
            return "BEAR STEEPENING / CONVENTIONAL TIGHTENING", -10.0
        return "MIXED", 0.0
    if us2y_change > 0.0 and us10y_change > 0.0 and curve_change <= 0.0:
        return "BEAR FLATTENING", -5.0
    if us2y_change < 0.0 and us10y_change < 0.0 and curve_change <= 0.0:
        return "BULL FLATTENING", 0.0
    return "MIXED", 0.0


def calculate_gold_structural_macro2_history(
    gold_price: pd.Series | None,
    dxy: pd.Series | None,
    real_yield: pd.Series | None,
    us2y: pd.Series | None,
    us10y: pd.Series | None,
    jp10y: pd.Series | None,
    global_m2: pd.Series | None,
    global_cb_assets: pd.Series | None,
    us_net_liquidity: pd.Series | None,
    t5yie: pd.Series | None,
    t10yie: pd.Series | None,
    business_cycle_state: pd.Series | None,
) -> pd.DataFrame:
    sources = [
        gold_price, dxy, real_yield, us2y, us10y, jp10y,
        global_m2, global_cb_assets, us_net_liquidity, t5yie,
        t10yie, business_cycle_state,
    ]
    index = _union_index([series for series in sources if series is not None])
    if index.empty:
        return pd.DataFrame()

    aligned = {
        "gold_price": _aligned_numeric(gold_price, index),
        "dxy": _aligned_numeric(dxy, index),
        "real_yield": _aligned_numeric(real_yield, index),
        "us2y": _aligned_numeric(us2y, index),
        "us10y": _aligned_numeric(us10y, index),
        "jp10y": _aligned_numeric(jp10y, index),
        "global_m2": _aligned_numeric(global_m2, index),
        "global_cb_assets": _aligned_numeric(global_cb_assets, index),
        "us_net_liquidity": _aligned_numeric(us_net_liquidity, index),
        "t5yie": _aligned_numeric(t5yie, index),
        "t10yie": _aligned_numeric(t10yie, index),
        "business_cycle_state": _aligned_state(business_cycle_state, index),
    }
    out = pd.DataFrame(aligned, index=index).rename_axis("date").reset_index()

    out["DXY_13W_Return"] = out["dxy"].pct_change(13, fill_method=None)
    out["DXY_13W_Pct"] = inclusive_trailing_percentile(out["DXY_13W_Return"])
    out["DXYBull"] = _clip(100.0 - out["DXY_13W_Pct"])
    out["RealYield_13W_Change"] = out["real_yield"] - out["real_yield"].shift(13)
    out["RealYield_13W_Pct"] = inclusive_trailing_percentile(out["RealYield_13W_Change"])
    out["RealYieldBull"] = _clip(100.0 - out["RealYield_13W_Pct"])
    out["US2Y_13W_Change"] = out["us2y"] - out["us2y"].shift(13)
    out["US2Y_13W_Pct"] = inclusive_trailing_percentile(out["US2Y_13W_Change"])
    out["US2YBull"] = _clip(100.0 - out["US2Y_13W_Pct"])
    out["StructuralMacro"] = _clip(0.35 * out["DXYBull"] + 0.55 * out["RealYieldBull"] + 0.10 * out["US2YBull"])

    for prefix, column in (
        ("Global_M2", "global_m2"),
        ("Global_CB_Assets", "global_cb_assets"),
        ("US_Net_Liquidity", "us_net_liquidity"),
    ):
        block = _liquidity_block(out[column], prefix)
        for name, values in block.items():
            out[name] = values.to_numpy()
    out["GoldLiquidity_3M"] = _clip(
        0.20 * out["Global_M2_Factor_3M"]
        + 0.20 * out["Global_CB_Assets_Factor_3M"]
        + 0.60 * out["US_Net_Liquidity_Factor_3M"]
    )
    out["GoldLiquidity_6M"] = _clip(
        0.20 * out["Global_M2_Factor_6M"]
        + 0.20 * out["Global_CB_Assets_Factor_6M"]
        + 0.60 * out["US_Net_Liquidity_Factor_6M"]
    )

    t5_13 = out["t5yie"] - out["t5yie"].shift(13)
    t5_26 = out["t5yie"] - out["t5yie"].shift(26)
    t5_39 = out["t5yie"] - out["t5yie"].shift(39)
    t10_26 = out["t10yie"] - out["t10yie"].shift(26)
    t10_39 = out["t10yie"] - out["t10yie"].shift(39)
    for name, values in {
        "T5YIE_13W_Change": t5_13,
        "T5YIE_26W_Change": t5_26,
        "T5YIE_39W_Change": t5_39,
        "T10YIE_26W_Change": t10_26,
        "T10YIE_39W_Change": t10_39,
    }.items():
        out[name] = values
        out[f"{name}_Pct"] = inclusive_trailing_percentile(values)
    out["InflationRelief_3M"] = _clip(100.0 - out["T5YIE_13W_Change_Pct"])
    out["InflationRelief_6M"] = _clip(
        0.70 * (100.0 - out["T5YIE_26W_Change_Pct"])
        + 0.30 * (100.0 - out["T10YIE_26W_Change_Pct"])
    )
    out["InflationRelief_9M"] = _clip(
        0.50 * (100.0 - out["T5YIE_26W_Change_Pct"])
        + 0.50 * (100.0 - out["T10YIE_39W_Change_Pct"])
    )
    out["InflationRelief_12M"] = _clip(
        0.40 * (100.0 - out["T5YIE_39W_Change_Pct"])
        + 0.60 * (100.0 - out["T10YIE_39W_Change_Pct"])
    )

    out["US10Y_13W_Change"] = out["us10y"] - out["us10y"].shift(13)
    curve = out["us10y"] - out["us2y"]
    out["Curve_13W_Change"] = curve - curve.shift(13)
    out["JP10Y_13W_Change"] = out["jp10y"] - out["jp10y"].shift(13)
    out["JP10Y_Shock_Pct"] = inclusive_trailing_percentile(out["JP10Y_13W_Change"])
    regime_and_overlay = out.apply(
        lambda row: _rate_regime(
            row["US2Y_13W_Change"],
            row["US10Y_13W_Change"],
            row["Curve_13W_Change"],
            row["JP10Y_Shock_Pct"],
        ),
        axis=1,
    )
    out["Gold_Rate_Regime"] = [value[0] for value in regime_and_overlay]
    out["SovereignStressOverlay"] = [value[1] for value in regime_and_overlay]

    out["BusinessCycleState"] = out["business_cycle_state"]
    for horizon in HORIZONS:
        out[f"Gold_BC_Modifier_{horizon}"] = [
            _business_cycle_modifier(value, horizon) for value in out["BusinessCycleState"]
        ]
    out["CoreMacro_3M"] = _clip(0.60 * out["GoldLiquidity_3M"] + 0.30 * out["StructuralMacro"] + 0.10 * out["InflationRelief_3M"])
    out["CoreMacro_6M"] = _clip(0.30 * out["GoldLiquidity_6M"] + 0.20 * out["StructuralMacro"] + 0.50 * out["InflationRelief_6M"])
    out["CoreMacro_9M"] = _clip(0.50 * out["StructuralMacro"] + 0.50 * out["InflationRelief_9M"])
    out["CoreMacro_12M"] = _clip(0.60 * out["StructuralMacro"] + 0.40 * out["InflationRelief_12M"])
    for horizon in HORIZONS:
        final = out[f"CoreMacro_{horizon}"] + out["SovereignStressOverlay"] + out[f"Gold_BC_Modifier_{horizon}"]
        out[f"GLD_MACRO_{horizon}_BeforeClamp"] = final
        out[f"GLD_MACRO_{horizon}"] = _clip(final)
        out[f"GLD_MACRO_{horizon}_State"] = out[f"GLD_MACRO_{horizon}"].map(_score_state)

    return out


def macro2_export_columns() -> list[str]:
    return [
        "date", "gold_price", "GLD_MACRO_3M", "GLD_MACRO_6M", "GLD_MACRO_9M", "GLD_MACRO_12M",
        "GLD_MACRO_3M_State", "GLD_MACRO_6M_State", "GLD_MACRO_9M_State", "GLD_MACRO_12M_State",
        "GLD_MACRO_3M_BeforeClamp", "GLD_MACRO_6M_BeforeClamp", "GLD_MACRO_9M_BeforeClamp", "GLD_MACRO_12M_BeforeClamp",
        "CoreMacro_3M", "CoreMacro_6M", "CoreMacro_9M", "CoreMacro_12M", "StructuralMacro", "DXYBull", "RealYieldBull", "US2YBull",
        "DXY_13W_Return", "DXY_13W_Pct", "RealYield_13W_Change", "RealYield_13W_Pct", "US2Y_13W_Change", "US2Y_13W_Pct",
        "GoldLiquidity_3M", "GoldLiquidity_6M",
        "Global_M2_Growth", "Global_M2_GrowthPct", "Global_M2_FastImpulse", "Global_M2_FastPct", "Global_M2_MediumImpulse", "Global_M2_MediumPct", "Global_M2_SlowImpulse", "Global_M2_SlowPct", "Global_M2_Impulse_3M", "Global_M2_Impulse_6M", "Global_M2_Factor_3M", "Global_M2_Factor_6M",
        "Global_CB_Assets_Growth", "Global_CB_Assets_GrowthPct", "Global_CB_Assets_FastImpulse", "Global_CB_Assets_FastPct", "Global_CB_Assets_MediumImpulse", "Global_CB_Assets_MediumPct", "Global_CB_Assets_SlowImpulse", "Global_CB_Assets_SlowPct", "Global_CB_Assets_Impulse_3M", "Global_CB_Assets_Impulse_6M", "Global_CB_Assets_Factor_3M", "Global_CB_Assets_Factor_6M",
        "US_Net_Liquidity_Growth", "US_Net_Liquidity_GrowthPct", "US_Net_Liquidity_FastImpulse", "US_Net_Liquidity_FastPct", "US_Net_Liquidity_MediumImpulse", "US_Net_Liquidity_MediumPct", "US_Net_Liquidity_SlowImpulse", "US_Net_Liquidity_SlowPct", "US_Net_Liquidity_Impulse_3M", "US_Net_Liquidity_Impulse_6M", "US_Net_Liquidity_Factor_3M", "US_Net_Liquidity_Factor_6M",
        "InflationRelief_3M", "InflationRelief_6M", "InflationRelief_9M", "InflationRelief_12M",
        "T5YIE_13W_Change", "T5YIE_13W_Change_Pct", "T5YIE_26W_Change", "T5YIE_26W_Change_Pct", "T5YIE_39W_Change", "T5YIE_39W_Change_Pct", "T10YIE_26W_Change", "T10YIE_26W_Change_Pct", "T10YIE_39W_Change", "T10YIE_39W_Change_Pct",
        "Gold_Rate_Regime", "US10Y_13W_Change", "Curve_13W_Change", "JP10Y_13W_Change", "JP10Y_Shock_Pct",
        "SovereignStressOverlay", "BusinessCycleState", "Gold_BC_Modifier_3M", "Gold_BC_Modifier_6M", "Gold_BC_Modifier_9M", "Gold_BC_Modifier_12M",
    ]

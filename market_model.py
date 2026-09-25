from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from fred_client import (
    FED_LIQUIDITY_SERIES_ID,
    FRED_DEFAULT_OBSERVATION_START,
    FredApiError,
    calculate_fed_liquidity,
    download_fred_series_batch,
)
from market_regime import calculate_market_regime


MARKET_MODEL_CONFIG = {
    "structural": {
        "spy_sma_weeks": 40,
        "drawdown_threshold": -0.10,
        "vol_window_weeks": 13,
        "high_vol_percentile": 75.0,
        "alpha_confidence": {
            "BULL": 100.0,
            "BULL_HIGH_VOL": 100.0,
            "CORRECTION": 50.0,
            "STRESS": 20.0,
        },
    },
    "fast_transition": {
        "vix_weight": 0.70,
        "dxy_weight": 0.30,
        "vix_z_window": 26,
        "dxy_window": 26,
        "robust_sigma_epsilon": 1e-9,
        "vix_risk_points": [(-0.5, 5.0), (0.0, 10.0), (0.5, 25.0), (1.0, 40.0), (1.5, 55.0), (2.0, 70.0), (3.0, 85.0), (4.0, 95.0)],
        "dxy_risk_points": [(-0.05, 0.0), (-0.02, 5.0), (0.0, 10.0), (0.02, 20.0), (0.05, 40.0), (0.08, 60.0), (0.12, 80.0)],
    },
    "macro_transition": {
        "dxy_weight": 0.40,
        "us2y_weight": 0.30,
        "global_m2_weight": 0.20,
        "fed_liquidity_weight": 0.10,
        "dxy_window": 26,
        "global_m2_window": 26,
        "global_m2_percentile_window": 156,
        "global_m2_percentile_min_periods": 104,
        "liquidity_window": 26,
        "liquidity_short_window": 13,
        "us2y_window": 13,
        "fed_liquidity_risk_points": [(-0.12, 80.0), (-0.08, 65.0), (-0.05, 45.0), (-0.02, 25.0), (0.0, 10.0), (0.02, 5.0), (0.05, 0.0)],
        "us2y_risk_points": [(-75.0, 0.0), (-25.0, 5.0), (0.0, 10.0), (25.0, 25.0), (50.0, 45.0), (75.0, 65.0), (100.0, 85.0)],
    },
    "credit": {
        "series_id": "BAMLH0A0HYM2",
        "change_window": 13,
        "percentile_window": 156,
        "percentile_min_periods": 104,
    },
    "confirmations": {
        "wti_short_window": 4,
        "wti_window": 13,
        "wti_long_window": 26,
        "real_yield_window": 13,
        "iwm_spy_window": 13,
        "xli_xlp_window": 13,
        "rsi_period": 14,
        "rsi_swing_distance": 3,
        "rsi_swing_max_distance": 16,
    },
}

FRED_MARKET_SERIES_IDS = ("WALCL", "RRPONTSYD", "WTREGEN", "DGS2", "DFII10", "BAMLH0A0HYM2")
YAHOO_MARKET_TICKERS = ("SPY", "IWM", "XLI", "XLP", "^VIX", "DX-Y.NYB", "CL=F")
POSITIONING_MODEL_VERSION = "POSITIONING_V1"
TAIL_RISK_MODEL_VERSION = "TAILRISK_V1"


def market_model_config() -> dict:
    return deepcopy(MARKET_MODEL_CONFIG)


def calculate_positioning_risk_history(aaii: pd.DataFrame | None, cftc_master: pd.DataFrame | None) -> pd.DataFrame:
    aaii_component = positioning_aaii_bearish_percentile(aaii)
    vix_component = positioning_vix_asset_manager_percentile(cftc_master)
    index = union_series_index([aaii_component, vix_component])
    if index.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "AAII_Bearish_3Y_Percentile",
                "VIX_AssetManager_NetPctOI",
                "VIX_AssetManager_NetPctOI_3Y_Percentile",
                "PositioningRisk",
                "PositioningState",
                "PositioningModel_Version",
            ]
        )
    frame = pd.DataFrame(index=index)
    frame["AAII_Bearish_3Y_Percentile"] = aaii_component.reindex(index).ffill()
    frame["VIX_AssetManager_NetPctOI_3Y_Percentile"] = vix_component.reindex(index).ffill()
    vix_net = positioning_vix_asset_manager_net_pct_oi(cftc_master).reindex(index).ffill()
    frame["VIX_AssetManager_NetPctOI"] = vix_net
    components = frame[["AAII_Bearish_3Y_Percentile", "VIX_AssetManager_NetPctOI_3Y_Percentile"]]
    weights = pd.DataFrame(
        {
            "AAII_Bearish_3Y_Percentile": 0.50,
            "VIX_AssetManager_NetPctOI_3Y_Percentile": 0.50,
        },
        index=frame.index,
    ).where(components.notna(), 0.0)
    weight_sum = weights.sum(axis=1).replace(0.0, np.nan)
    frame["PositioningRisk"] = (components.fillna(0.0).mul(weights).sum(axis=1) / weight_sum).clip(0.0, 100.0)
    frame["PositioningState"] = frame["PositioningRisk"].map(classify_positioning_state)
    frame["PositioningModel_Version"] = POSITIONING_MODEL_VERSION
    return frame.reset_index().rename(columns={"index": "Date"})


def calculate_tail_risk_history(history: pd.DataFrame, positioning_history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None or history.empty or "Date" not in history.columns:
        return pd.DataFrame(
            columns=[
                "Date",
                "PositioningRisk",
                "PositioningState",
                "LiquidityWarning",
                "CreditWarning",
                "FastWarning",
                "MacroWarning",
                "TailRiskFlag",
                "TailRiskReason",
                "TailRiskModel_Version",
            ]
        )
    frame = history.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    if positioning_history is not None and not positioning_history.empty:
        pos = positioning_history.copy()
        pos["Date"] = pd.to_datetime(pos["Date"], errors="coerce")
        pos = pos.dropna(subset=["Date"]).sort_values("Date")
        keep = [
            "Date",
            "AAII_Bearish_3Y_Percentile",
            "VIX_AssetManager_NetPctOI",
            "VIX_AssetManager_NetPctOI_3Y_Percentile",
            "PositioningRisk",
            "PositioningState",
            "PositioningModel_Version",
        ]
        frame = pd.merge_asof(frame, pos[[col for col in keep if col in pos.columns]], on="Date", direction="backward")
    if "PositioningRisk" not in frame.columns:
        frame["PositioningRisk"] = np.nan
    if "PositioningState" not in frame.columns:
        frame["PositioningState"] = frame["PositioningRisk"].map(classify_positioning_state)
    frame["LiquidityWarning"] = frame.apply(
        lambda row: liquidity_warning(row.get("Global_Liquidity_Score"), row.get("Global_Liquidity_Direction_13W"), row.get("Global_Liquidity_Direction_13W_State"), row.get("Global_Liquidity_Backdrop")),
        axis=1,
    )
    credit_risk = numeric_column(frame, "Credit_Risk")
    fast_risk = numeric_column(frame, "Fast_Transition_Risk")
    macro_risk = numeric_column(frame, "Macro_Transition_Risk")
    frame["CreditWarning"] = credit_risk >= 60.0
    frame["CreditHigh"] = credit_risk >= 75.0
    frame["CreditExtreme"] = credit_risk >= 90.0
    fast_direction_state = frame.get("Fast_Risk_Direction_4W_State", pd.Series("", index=frame.index)).astype(str).str.upper()
    frame["FastWarning"] = (fast_risk >= 40.0) | fast_direction_state.eq("RAPID_DETERIORATION")
    frame["FastExtreme"] = fast_risk >= 60.0
    frame["MacroWarning"] = macro_risk >= 40.0
    frame["MacroHigh"] = macro_risk >= 60.0
    classified = frame.apply(classify_tail_risk_row, axis=1, result_type="expand")
    frame["TailRiskFlag"] = classified[0]
    frame["TailRiskReason"] = classified[1]
    frame["TailRiskModel_Version"] = TAIL_RISK_MODEL_VERSION
    return frame


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def positioning_aaii_bearish_percentile(aaii: pd.DataFrame | None) -> pd.Series:
    if aaii is None or aaii.empty or "Date" not in aaii.columns:
        return pd.Series(dtype="float64")
    frame = aaii.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    if "AAII_Bearish_3Y_Percentile" in frame.columns:
        values = pd.to_numeric(frame["AAII_Bearish_3Y_Percentile"], errors="coerce")
    elif "AAII_Bearish" in frame.columns:
        values = trailing_percentile(pd.to_numeric(frame["AAII_Bearish"], errors="coerce"), 156, 52)
    else:
        return pd.Series(dtype="float64")
    dates = frame["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return weekly_last_series(pd.Series(values.values, index=dates)).dropna().sort_index()


def positioning_vix_asset_manager_percentile(cftc_master: pd.DataFrame | None) -> pd.Series:
    selected = select_vix_asset_manager(cftc_master)
    if selected.empty:
        return pd.Series(dtype="float64")
    if "NetPctOI_3Y_Percentile" in selected.columns:
        values = pd.to_numeric(selected["NetPctOI_3Y_Percentile"], errors="coerce")
    else:
        values = trailing_percentile(pd.to_numeric(selected["NetPctOI"], errors="coerce"), 156, 52)
    dates = cftc_publication_week_dates(selected)
    return weekly_last_series(pd.Series(values.values, index=dates)).dropna().sort_index()


def positioning_vix_asset_manager_net_pct_oi(cftc_master: pd.DataFrame | None) -> pd.Series:
    selected = select_vix_asset_manager(cftc_master)
    if selected.empty or "NetPctOI" not in selected.columns:
        return pd.Series(dtype="float64")
    dates = cftc_publication_week_dates(selected)
    return weekly_last_series(pd.Series(pd.to_numeric(selected["NetPctOI"], errors="coerce").values, index=dates)).dropna().sort_index()


def weekly_last_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.sort_index().groupby(level=0).last()


def select_vix_asset_manager(cftc_master: pd.DataFrame | None) -> pd.DataFrame:
    if cftc_master is None or cftc_master.empty:
        return pd.DataFrame()
    frame = cftc_master.copy()
    if {"Canonical_Asset", "Participant_Category"}.issubset(frame.columns):
        mask = frame["Canonical_Asset"].astype(str).eq("VIX") & frame["Participant_Category"].astype(str).eq("Asset Manager")
        if "Preferred_For_Dashboard" in frame.columns:
            mask &= frame["Preferred_For_Dashboard"].astype(bool)
        selected = frame.loc[mask].copy()
    else:
        selected = pd.DataFrame()
    if selected.empty and "Raw_Contract_Name" in frame.columns:
        selected = frame.loc[
            frame["Raw_Contract_Name"].astype(str).str.upper().str.contains("VIX", na=False)
            & frame["Participant_Category"].astype(str).eq("Asset Manager")
        ].copy()
    return selected.sort_values("Date") if "Date" in selected.columns else selected


def cftc_publication_week_dates(frame: pd.DataFrame) -> pd.Series:
    if "Publication_Date" in frame.columns:
        publication = pd.to_datetime(frame["Publication_Date"], errors="coerce")
    else:
        publication = pd.to_datetime(frame["Date"], errors="coerce") + pd.Timedelta(days=3)
    return publication.dt.to_period("W-FRI").dt.end_time.dt.normalize()


def classify_positioning_state(value: float) -> str:
    risk = safe_numeric(value)
    if not np.isfinite(risk):
        return "DATA_INCOMPLETE"
    if risk < 40.0:
        return "BENIGN"
    if risk < 60.0:
        return "NORMAL"
    if risk < 75.0:
        return "ELEVATED"
    if risk < 90.0:
        return "CROWDED"
    return "EXTREME"


def liquidity_warning(score: float, direction_13w: float, direction_state: str = "", backdrop: str = "") -> bool:
    value = safe_numeric(score)
    direction = safe_numeric(direction_13w)
    state = str(direction_state or "").upper()
    back = str(backdrop or "").upper()
    if back in {"LIQUIDITY_WARNING", "NEGATIVE", "STRONGLY_NEGATIVE", "SUPPORTIVE_BUT_WEAKENING"}:
        return True
    if state in {"DETERIORATING", "DETERIORATING_FAST"} and (not np.isfinite(value) or value < 80.0):
        return True
    return bool(np.isfinite(direction) and direction < 0.0 and (not np.isfinite(value) or value < 80.0))


def classify_tail_risk_row(row: pd.Series) -> tuple[str, str]:
    positioning = safe_numeric(row.get("PositioningRisk"))
    credit = safe_numeric(row.get("Credit_Risk"))
    fast = safe_numeric(row.get("Fast_Transition_Risk"))
    macro = safe_numeric(row.get("Macro_Transition_Risk"))
    liquidity = bool(row.get("LiquidityWarning"))
    structural = str(row.get("Market_Regime", "")).upper()
    p60 = np.isfinite(positioning) and positioning >= 60.0
    p75 = np.isfinite(positioning) and positioning >= 75.0
    p90 = np.isfinite(positioning) and positioning >= 90.0
    c60 = np.isfinite(credit) and credit >= 60.0
    c75 = np.isfinite(credit) and credit >= 75.0
    c90 = np.isfinite(credit) and credit >= 90.0
    f40 = np.isfinite(fast) and fast >= 40.0
    f60 = np.isfinite(fast) and fast >= 60.0
    m40 = np.isfinite(macro) and macro >= 40.0

    if (
        (p75 and liquidity and c75)
        or (p90 and (m40 or c75 or f40))
        or (f60 and c75)
        or (structural == "STRESS" and (c75 or f60))
        or (liquidity and c90)
    ):
        return "EXTREME", tail_risk_reason(positioning, liquidity, credit, fast, macro)
    if (
        (p60 and liquidity and c60)
        or (p60 and c75)
        or (liquidity and c75)
        or (p75 and f40)
        or (p75 and m40)
        or (f60 and c60)
    ):
        return "HIGH", tail_risk_reason(positioning, liquidity, credit, fast, macro)
    if p60 or liquidity or c60 or f40 or m40:
        return "WATCH", tail_risk_reason(positioning, liquidity, credit, fast, macro)
    return "NORMAL", ""


def tail_risk_reason(positioning: float, liquidity: bool, credit: float, fast: float, macro: float) -> str:
    reasons = []
    if np.isfinite(positioning) and positioning >= 60.0:
        reasons.append("POSITIONING")
    if liquidity:
        reasons.append("LIQUIDITY")
    if np.isfinite(credit) and credit >= 60.0:
        reasons.append("CREDIT")
    if np.isfinite(fast) and fast >= 40.0:
        reasons.append("FAST")
    if np.isfinite(macro) and macro >= 40.0:
        reasons.append("MACRO")
    return "|".join(reasons)


def calculate_market_model(
    yahoo_weekly: dict[str, pd.DataFrame],
    fred_data: pd.DataFrame | None,
    config: dict | None = None,
    global_m2: pd.Series | None = None,
) -> dict:
    cfg = config or MARKET_MODEL_CONFIG
    structural = calculate_structural_regime(yahoo_weekly.get("SPY", pd.DataFrame()), cfg)
    fast = calculate_fast_transition_risk(
        weekly_close(yahoo_weekly.get("^VIX", pd.DataFrame())),
        weekly_close(yahoo_weekly.get("DX-Y.NYB", pd.DataFrame())),
        cfg,
    )
    macro = calculate_macro_transition_risk(
        weekly_close(yahoo_weekly.get("DX-Y.NYB", pd.DataFrame())),
        fred_data,
        cfg,
        global_m2=global_m2,
    )
    credit = calculate_credit_stress_confirmation(fred_data, cfg)
    confirmations = calculate_confirmations(yahoo_weekly, fred_data, cfg)
    overall_status = calculate_overall_transition_status(
        fast.get("Fast_Transition_Risk"),
        macro.get("Macro_Transition_Risk"),
        confirmations.get("Negative_Confirmation_Count"),
        structural_regime=structural.get("Market_Regime", "UNKNOWN"),
        credit_state=credit.get("Credit_State", ""),
    )
    alpha_confidence = calculate_alpha_confidence(
        structural.get("Market_Regime", "UNKNOWN"),
        fast.get("Fast_Transition_Risk"),
        macro.get("Macro_Transition_Risk"),
        cfg,
    )
    return {
        **structural,
        **fast,
        **macro,
        **credit,
        **confirmations,
        "Overall_Transition_Status": overall_status,
        "Final_Market_State": overall_status,
        "Alpha_Confidence": alpha_confidence,
    }


def calculate_structural_regime(spy_weekly: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = config or MARKET_MODEL_CONFIG
    return calculate_market_regime(spy_weekly, {"market_regime": cfg["structural"]})


def calculate_fast_transition_risk(vix: pd.Series, dxy: pd.Series, config: dict | None = None) -> dict:
    cfg = (config or MARKET_MODEL_CONFIG)["fast_transition"]
    vix_z = safe_last(calculate_vix_z(vix, int(cfg["vix_z_window"]), float(cfg["robust_sigma_epsilon"])))
    dxy_13w = safe_last(dxy.pct_change(13)) if not dxy.empty else np.nan
    dxy_26w = safe_last(dxy.pct_change(int(cfg["dxy_window"]))) if not dxy.empty else np.nan
    vix_risk = scalar_piecewise_score(vix_z, cfg["vix_risk_points"])
    dxy_risk = scalar_piecewise_score(dxy_26w, cfg["dxy_risk_points"])
    if not np.isfinite(vix_risk) or not np.isfinite(dxy_risk):
        risk = np.nan
        state = "DATA_INCOMPLETE"
    else:
        risk = float(np.clip(float(cfg["vix_weight"]) * vix_risk + float(cfg["dxy_weight"]) * dxy_risk, 0.0, 100.0))
        state = fast_transition_state(risk)
    return {
        "Fast_Transition_Risk": risk,
        "Fast_Transition_State": state,
        "Fast_Risk_Direction_4W": np.nan,
        "Fast_Risk_Direction_4W_State": "DATA_INCOMPLETE",
        "VIX_Z26": vix_z,
        "VIX_Risk": vix_risk,
        "DXY_Return_13W": dxy_13w,
        "DXY_Return_26W": dxy_26w,
        "DXY_Risk": dxy_risk,
    }


def calculate_fast_transition_risk_history(vix: pd.Series, dxy: pd.Series, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or MARKET_MODEL_CONFIG)["fast_transition"]
    index = union_series_index([vix, dxy])
    if index.empty:
        return pd.DataFrame(columns=["Date", "Fast_Transition_Risk", "Fast_Transition_State"])

    vix_values = pd.to_numeric(vix, errors="coerce").sort_index().reindex(index).ffill()
    dxy_values = pd.to_numeric(dxy, errors="coerce").sort_index().reindex(index).ffill()
    vix_z = calculate_vix_z(vix_values, int(cfg["vix_z_window"]), float(cfg["robust_sigma_epsilon"])).reindex(index)
    dxy_13w = dxy_values.pct_change(13)
    dxy_26w = dxy_values.pct_change(int(cfg["dxy_window"]))
    vix_risk = piecewise_score_series(vix_z, cfg["vix_risk_points"])
    dxy_risk = piecewise_score_series(dxy_26w, cfg["dxy_risk_points"])
    risk = (float(cfg["vix_weight"]) * vix_risk + float(cfg["dxy_weight"]) * dxy_risk).clip(0.0, 100.0)
    incomplete = vix_risk.isna() | dxy_risk.isna()
    state = risk.map(fast_transition_state)
    state.loc[incomplete] = "DATA_INCOMPLETE"
    risk.loc[incomplete] = np.nan
    direction_4w = risk - risk.shift(4)
    direction_state = direction_4w.map(fast_risk_direction_state)
    direction_state.loc[direction_4w.isna()] = "DATA_INCOMPLETE"
    return pd.DataFrame(
        {
            "Date": index,
            "Fast_Transition_Risk": risk.values,
            "Fast_Transition_State": state.values,
            "Fast_Risk_Direction_4W": direction_4w.values,
            "Fast_Risk_Direction_4W_State": direction_state.values,
            "VIX_Z26": vix_z.values,
            "VIX_Risk": vix_risk.values,
            "DXY_Return_13W": dxy_13w.values,
            "DXY_Return_26W": dxy_26w.values,
            "DXY_Risk": dxy_risk.values,
        }
    )


def calculate_vix_z(vix: pd.Series, window: int = 26, epsilon: float = 1e-9) -> pd.Series:
    values = pd.to_numeric(vix, errors="coerce").dropna().sort_index()
    if values.empty:
        return pd.Series(dtype="float64")
    rolling_median = values.rolling(window, min_periods=window).median()

    def mad(window_values: np.ndarray) -> float:
        median = np.nanmedian(window_values)
        return float(np.nanmedian(np.abs(window_values - median)))

    rolling_mad = values.rolling(window, min_periods=window).apply(mad, raw=True)
    sigma = (1.4826 * rolling_mad).where(lambda s: s.abs() > epsilon, epsilon)
    return ((values - rolling_median) / sigma).replace([np.inf, -np.inf], np.nan)


def calculate_macro_transition_risk(
    dxy: pd.Series,
    fred_data: pd.DataFrame | None,
    config: dict | None = None,
    global_m2: pd.Series | None = None,
) -> dict:
    full_cfg = config or MARKET_MODEL_CONFIG
    cfg = full_cfg["macro_transition"]
    dxy_26w = safe_last(dxy.pct_change(int(cfg["dxy_window"]))) if not dxy.empty else np.nan
    dxy_risk = scalar_piecewise_score(dxy_26w, full_cfg["fast_transition"]["dxy_risk_points"])

    fred_weekly = fred_series_weekly(fred_data)
    liquidity = fred_weekly.get(FED_LIQUIDITY_SERIES_ID, pd.Series(dtype="float64"))
    fed_liquidity_13w = safe_last(liquidity.pct_change(int(cfg["liquidity_short_window"]))) if not liquidity.empty else np.nan
    fed_liquidity_26w = safe_last(liquidity.pct_change(int(cfg["liquidity_window"]))) if not liquidity.empty else np.nan
    fed_liquidity_risk = scalar_piecewise_score(fed_liquidity_26w, cfg["fed_liquidity_risk_points"])

    us2y = fred_weekly.get("DGS2", pd.Series(dtype="float64"))
    us2y_change_13w_bp = safe_last((us2y - us2y.shift(int(cfg["us2y_window"]))) * 100.0) if not us2y.empty else np.nan
    us2y_risk = scalar_piecewise_score(us2y_change_13w_bp, cfg["us2y_risk_points"])

    global_m2_values = clean_weekly_series(global_m2)
    global_m2_26w = (
        safe_last(global_m2_values.pct_change(int(cfg["global_m2_window"]), fill_method=None))
        if not global_m2_values.empty
        else np.nan
    )
    global_m2_bull_score = safe_last(
        trailing_percentile(
            global_m2_values.pct_change(int(cfg["global_m2_window"]), fill_method=None),
            int(cfg["global_m2_percentile_window"]),
            int(cfg["global_m2_percentile_min_periods"]),
        )
    ) if not global_m2_values.empty else np.nan
    global_m2_risk = 100.0 - global_m2_bull_score if np.isfinite(global_m2_bull_score) else np.nan

    components = [dxy_risk, us2y_risk, global_m2_risk, fed_liquidity_risk]
    if sum(np.isfinite(component) for component in components) < 2:
        risk = np.nan
        state = "DATA_INCOMPLETE"
    else:
        weights = np.array(
            [
                float(cfg["dxy_weight"]),
                float(cfg["us2y_weight"]),
                float(cfg["global_m2_weight"]),
                float(cfg["fed_liquidity_weight"]),
            ]
        )
        values = np.array(components, dtype="float64")
        mask = np.isfinite(values)
        risk = float(np.clip(np.average(values[mask], weights=weights[mask]), 0.0, 100.0))
        state = transition_state(risk, alert_label="MACRO_ALERT")
    return {
        "Macro_Transition_Risk": risk,
        "Macro_Transition_State": state,
        "Fed_Liquidity_13W": fed_liquidity_13w,
        "Fed_Liquidity_26W": fed_liquidity_26w,
        "Fed_Liquidity_Risk": fed_liquidity_risk,
        "US2Y_Change_13W_bp": us2y_change_13w_bp,
        "US2Y_Risk": us2y_risk,
        "Global_M2_26W": global_m2_26w,
        "Global_M2_Bull_Score_26W": global_m2_bull_score,
        "Global_M2_Risk_26W": global_m2_risk,
        "Macro_DXY_Risk": dxy_risk,
    }


def calculate_macro_transition_risk_history(
    dxy: pd.Series,
    fred_data: pd.DataFrame | None,
    config: dict | None = None,
    global_m2: pd.Series | None = None,
) -> pd.DataFrame:
    full_cfg = config or MARKET_MODEL_CONFIG
    cfg = full_cfg["macro_transition"]
    fred_weekly = fred_series_weekly(fred_data)
    liquidity = fred_weekly.get(FED_LIQUIDITY_SERIES_ID, pd.Series(dtype="float64"))
    us2y = fred_weekly.get("DGS2", pd.Series(dtype="float64"))
    global_m2_values = clean_weekly_series(global_m2)
    index = union_series_index([dxy, liquidity, us2y, global_m2_values])
    if index.empty:
        return pd.DataFrame(columns=["Date", "Macro_Transition_Risk", "Macro_Transition_State"])

    dxy_values = pd.to_numeric(dxy, errors="coerce").sort_index().reindex(index).ffill()
    liquidity_values = pd.to_numeric(liquidity, errors="coerce").sort_index().reindex(index).ffill()
    us2y_values = pd.to_numeric(us2y, errors="coerce").sort_index().reindex(index).ffill()
    global_m2_values = global_m2_values.reindex(index).ffill()

    dxy_26w = dxy_values.pct_change(int(cfg["dxy_window"]))
    dxy_risk = piecewise_score_series(dxy_26w, full_cfg["fast_transition"]["dxy_risk_points"])
    fed_liquidity_13w = liquidity_values.pct_change(int(cfg["liquidity_short_window"]))
    fed_liquidity_26w = liquidity_values.pct_change(int(cfg["liquidity_window"]))
    fed_liquidity_risk = piecewise_score_series(fed_liquidity_26w, cfg["fed_liquidity_risk_points"])
    us2y_change_13w_bp = (us2y_values - us2y_values.shift(int(cfg["us2y_window"]))) * 100.0
    us2y_risk = piecewise_score_series(us2y_change_13w_bp, cfg["us2y_risk_points"])
    global_m2_26w = global_m2_values.pct_change(int(cfg["global_m2_window"]), fill_method=None)
    global_m2_bull_score = trailing_percentile(
        global_m2_26w,
        int(cfg["global_m2_percentile_window"]),
        int(cfg["global_m2_percentile_min_periods"]),
    )
    global_m2_risk = 100.0 - global_m2_bull_score

    components = pd.concat([dxy_risk, us2y_risk, global_m2_risk, fed_liquidity_risk], axis=1)
    components.columns = ["DXY_Risk", "US2Y_Risk", "Global_M2_Risk_26W", "Fed_Liquidity_Risk"]
    weights = pd.Series(
        [
            float(cfg["dxy_weight"]),
            float(cfg["us2y_weight"]),
            float(cfg["global_m2_weight"]),
            float(cfg["fed_liquidity_weight"]),
        ],
        index=components.columns,
    )
    finite_count = components.notna().sum(axis=1)
    weighted_sum = components.mul(weights, axis=1).sum(axis=1, min_count=1)
    active_weights = components.notna().mul(weights, axis=1).sum(axis=1)
    risk = (weighted_sum / active_weights.replace(0.0, np.nan)).clip(0.0, 100.0)
    risk.loc[finite_count < 2] = np.nan
    state = risk.map(lambda value: transition_state(value, alert_label="MACRO_ALERT"))
    state.loc[risk.isna()] = "DATA_INCOMPLETE"

    return pd.DataFrame(
        {
            "Date": index,
            "Macro_Transition_Risk": risk.values,
            "Macro_Transition_State": state.values,
            "Fed_Liquidity_13W": fed_liquidity_13w.values,
            "Fed_Liquidity_26W": fed_liquidity_26w.values,
            "Fed_Liquidity_Risk": fed_liquidity_risk.values,
            "US2Y_Change_13W_bp": us2y_change_13w_bp.values,
            "US2Y_Risk": us2y_risk.values,
            "Global_M2_26W": global_m2_26w.values,
            "Global_M2_Bull_Score_26W": global_m2_bull_score.values,
            "Global_M2_Risk_26W": global_m2_risk.values,
            "Macro_DXY_Risk": dxy_risk.values,
        }
    )


def calculate_credit_stress_confirmation(fred_data: pd.DataFrame | None, config: dict | None = None) -> dict:
    history = calculate_credit_stress_confirmation_history(fred_data, config)
    if history.empty:
        return {
            "HY_OAS": np.nan,
            "HY_OAS_Change_13W": np.nan,
            "Credit_Risk": np.nan,
            "Credit_Widening_Percentile": np.nan,
            "HY_Level_Percentile": np.nan,
            "Credit_State": "DATA_INCOMPLETE",
            "Credit_Level_State": "DATA_INCOMPLETE",
        }
    latest = history.dropna(subset=["Credit_Risk", "HY_OAS"], how="all").tail(1)
    if latest.empty:
        latest = history.tail(1)
    row = latest.iloc[0]
    return {
        "HY_OAS": safe_numeric(row.get("HY_OAS")),
        "HY_OAS_Change_13W": safe_numeric(row.get("HY_OAS_Change_13W")),
        "Credit_Risk": safe_numeric(row.get("Credit_Risk")),
        "Credit_Widening_Percentile": safe_numeric(row.get("Credit_Widening_Percentile")),
        "HY_Level_Percentile": safe_numeric(row.get("HY_Level_Percentile")),
        "Credit_State": str(row.get("Credit_State", "DATA_INCOMPLETE")),
        "Credit_Level_State": str(row.get("Credit_Level_State", "DATA_INCOMPLETE")),
    }


def calculate_credit_stress_confirmation_history(fred_data: pd.DataFrame | None, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or MARKET_MODEL_CONFIG)["credit"]
    fred_weekly = fred_series_weekly(fred_data)
    hy_oas = fred_weekly.get(str(cfg["series_id"]).upper(), pd.Series(dtype="float64"))
    if hy_oas.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "HY_OAS",
                "HY_OAS_Change_13W",
                "Credit_Risk",
                "Credit_Widening_Percentile",
                "HY_Level_Percentile",
                "Credit_State",
                "Credit_Level_State",
            ]
        )
    hy_oas = clean_weekly_series(hy_oas)
    change = hy_oas - hy_oas.shift(int(cfg["change_window"]))
    widening = trailing_percentile(change, int(cfg["percentile_window"]), int(cfg["percentile_min_periods"]))
    level = trailing_percentile(hy_oas, int(cfg["percentile_window"]), int(cfg["percentile_min_periods"]))
    return pd.DataFrame(
        {
            "Date": hy_oas.index,
            "HY_OAS": hy_oas.values,
            "HY_OAS_Change_13W": change.values,
            "Credit_Risk": widening.values,
            "Credit_Widening_Percentile": widening.values,
            "HY_Level_Percentile": level.values,
            "Credit_State": widening.map(classify_credit_state).values,
            "Credit_Level_State": level.map(classify_credit_level_state).values,
        }
    )


def calculate_confirmations(yahoo_weekly: dict[str, pd.DataFrame], fred_data: pd.DataFrame | None, config: dict | None = None) -> dict:
    cfg = (config or MARKET_MODEL_CONFIG)["confirmations"]
    wti = weekly_close(yahoo_weekly.get("CL=F", pd.DataFrame()))
    spy = weekly_close(yahoo_weekly.get("SPY", pd.DataFrame()))
    iwm = weekly_close(yahoo_weekly.get("IWM", pd.DataFrame()))
    xli = weekly_close(yahoo_weekly.get("XLI", pd.DataFrame()))
    xlp = weekly_close(yahoo_weekly.get("XLP", pd.DataFrame()))
    fred_weekly = fred_series_weekly(fred_data)

    wti_4w = safe_last(wti.pct_change(int(cfg["wti_short_window"]))) if not wti.empty else np.nan
    wti_13w = safe_last(wti.pct_change(int(cfg["wti_window"]))) if not wti.empty else np.nan
    wti_26w = safe_last(wti.pct_change(int(cfg["wti_long_window"]))) if not wti.empty else np.nan
    wti_confirmation = classify_wti_confirmation(wti_13w)

    real_yield = fred_weekly.get("DFII10", pd.Series(dtype="float64"))
    real_yield_change_13w_bp = safe_last((real_yield - real_yield.shift(int(cfg["real_yield_window"]))) * 100.0) if not real_yield.empty else np.nan
    real_yield_confirmation = classify_real_yield_confirmation(real_yield_change_13w_bp)

    iwm_spy_13w = ratio_return(iwm, spy, int(cfg["iwm_spy_window"]))
    iwm_spy_confirmation = classify_ratio_confirmation(iwm_spy_13w)
    xli_xlp_13w = ratio_return(xli, xlp, int(cfg["xli_xlp_window"]))
    xli_xlp_confirmation = classify_ratio_confirmation(xli_xlp_13w)
    rsi_divergence = calculate_rsi_divergence(spy, cfg)

    us2y = fred_weekly.get("DGS2", pd.Series(dtype="float64"))
    us2y_change_13w_bp = safe_last((us2y - us2y.shift(13)) * 100.0) if not us2y.empty else np.nan
    shock = (
        "INFLATION_TIGHTENING_SHOCK"
        if np.isfinite(wti_13w) and wti_13w > 0.10 and np.isfinite(us2y_change_13w_bp) and us2y_change_13w_bp > 0.0
        else ""
    )

    confirmation_values = [wti_confirmation, real_yield_confirmation, iwm_spy_confirmation, xli_xlp_confirmation]
    negative_count = float(sum(value in {"NEGATIVE", "STRONG_NEGATIVE"} for value in confirmation_values))
    if rsi_divergence in {"MODERATE", "STRONG"}:
        negative_count += 1.0
    elif rsi_divergence == "MILD":
        negative_count += 0.5

    return {
        "WTI_4W_Return": wti_4w,
        "WTI_13W_Return": wti_13w,
        "WTI_26W_Return": wti_26w,
        "WTI_Confirmation": wti_confirmation,
        "Real_Yield_10Y_Change_13W_bp": real_yield_change_13w_bp,
        "Real_Yield_10Y_Confirmation": real_yield_confirmation,
        "IWM_SPY_13W_Return": iwm_spy_13w,
        "IWM_SPY_Confirmation": iwm_spy_confirmation,
        "XLI_XLP_13W_Return": xli_xlp_13w,
        "XLI_XLP_Confirmation": xli_xlp_confirmation,
        "RSI_Divergence": rsi_divergence,
        "Confirmation_Flag": shock,
        "Negative_Confirmation_Count": negative_count,
    }


def calculate_confirmations_history(yahoo_weekly: dict[str, pd.DataFrame], fred_data: pd.DataFrame | None, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or MARKET_MODEL_CONFIG)["confirmations"]
    wti = weekly_close(yahoo_weekly.get("CL=F", pd.DataFrame()))
    spy = weekly_close(yahoo_weekly.get("SPY", pd.DataFrame()))
    iwm = weekly_close(yahoo_weekly.get("IWM", pd.DataFrame()))
    xli = weekly_close(yahoo_weekly.get("XLI", pd.DataFrame()))
    xlp = weekly_close(yahoo_weekly.get("XLP", pd.DataFrame()))
    fred_weekly = fred_series_weekly(fred_data)
    real_yield = fred_weekly.get("DFII10", pd.Series(dtype="float64"))

    wti_13w = wti.pct_change(int(cfg["wti_window"])) if not wti.empty else pd.Series(dtype="float64")
    real_yield_change_13w_bp = (
        (real_yield - real_yield.shift(int(cfg["real_yield_window"]))) * 100.0
        if not real_yield.empty
        else pd.Series(dtype="float64")
    )
    iwm_spy_13w = ratio_return_series(iwm, spy, int(cfg["iwm_spy_window"]))
    xli_xlp_13w = ratio_return_series(xli, xlp, int(cfg["xli_xlp_window"]))
    rsi_divergence = calculate_rsi_divergence_history(spy, cfg)

    index = union_series_index([wti_13w, real_yield_change_13w_bp, iwm_spy_13w, xli_xlp_13w, rsi_divergence])
    if index.empty:
        return pd.DataFrame(columns=["Date", "Negative_Confirmation_Count"])

    frame = pd.DataFrame(index=index)
    frame["WTI_13W_Return"] = pd.to_numeric(wti_13w, errors="coerce").reindex(index)
    frame["Real_Yield_10Y_Change_13W_bp"] = pd.to_numeric(real_yield_change_13w_bp, errors="coerce").reindex(index)
    frame["IWM_SPY_13W_Return"] = pd.to_numeric(iwm_spy_13w, errors="coerce").reindex(index)
    frame["XLI_XLP_13W_Return"] = pd.to_numeric(xli_xlp_13w, errors="coerce").reindex(index)
    frame["WTI_Confirmation"] = frame["WTI_13W_Return"].map(classify_wti_confirmation)
    frame["Real_Yield_10Y_Confirmation"] = frame["Real_Yield_10Y_Change_13W_bp"].map(classify_real_yield_confirmation)
    frame["IWM_SPY_Confirmation"] = frame["IWM_SPY_13W_Return"].map(classify_ratio_confirmation)
    frame["XLI_XLP_Confirmation"] = frame["XLI_XLP_13W_Return"].map(classify_ratio_confirmation)
    frame["RSI_Divergence"] = rsi_divergence.reindex(index).fillna("NONE")

    confirmation_cols = [
        "WTI_Confirmation",
        "Real_Yield_10Y_Confirmation",
        "IWM_SPY_Confirmation",
        "XLI_XLP_Confirmation",
    ]
    frame["Negative_Confirmation_Count"] = frame[confirmation_cols].isin({"NEGATIVE", "STRONG_NEGATIVE"}).sum(axis=1).astype(float)
    frame.loc[frame["RSI_Divergence"].isin({"MODERATE", "STRONG"}), "Negative_Confirmation_Count"] += 1.0
    frame.loc[frame["RSI_Divergence"] == "MILD", "Negative_Confirmation_Count"] += 0.5

    return frame.reset_index().rename(columns={"index": "Date"})


def calculate_overall_transition_status(
    fast_risk: float,
    macro_risk: float,
    negative_confirmations: float,
    structural_regime: str = "BULL",
    global_liquidity_backdrop: str = "NEUTRAL",
    global_liquidity_score: float = np.nan,
    global_liquidity_direction_13w: float = np.nan,
    global_liquidity_direction_state: str = "",
    credit_state: str = "",
) -> str:
    fast = safe_numeric(fast_risk)
    macro = safe_numeric(macro_risk)
    negative = safe_numeric(negative_confirmations)
    liquidity_score = safe_numeric(global_liquidity_score)
    liquidity_direction = safe_numeric(global_liquidity_direction_13w)
    liquidity_state = str(global_liquidity_direction_state or "").upper()
    credit = str(credit_state or "").upper()
    backdrop = str(global_liquidity_backdrop or "NEUTRAL").upper()
    structural = str(structural_regime or "UNKNOWN").upper()

    if structural == "STRESS":
        return "STRESS"
    if structural == "CORRECTION":
        return "CORRECTION"
    if structural not in {"BULL", "BULL_HIGH_VOL"}:
        return "DATA_INCOMPLETE"
    if not np.isfinite(fast) or not np.isfinite(macro) or not np.isfinite(negative):
        return "DATA_INCOMPLETE"

    liquidity_warning = (
        backdrop in {"LIQUIDITY_WARNING", "NEGATIVE", "STRONGLY_NEGATIVE"}
        or (np.isfinite(liquidity_score) and liquidity_score < 40.0)
        or (np.isfinite(liquidity_direction) and liquidity_direction < -10.0)
        or liquidity_state == "DETERIORATING_FAST"
    )
    fast_warning = fast >= 20.0
    macro_warning = macro >= 20.0
    primary_count = int(fast_warning) + int(macro_warning) + int(liquidity_warning)
    credit_widening = credit in {"WIDENING", "SEVERE_WIDENING"}
    if primary_count >= 2:
        return "DETERIORATING"
    if (fast_warning or macro_warning) and negative >= 3.0:
        return "DETERIORATING"
    if primary_count >= 1 and credit_widening:
        return "DETERIORATING"
    if fast_warning or macro_warning:
        return "BULL_WITH_WARNING"
    if liquidity_warning:
        return "BULL_LIQUIDITY_WARNING"
    return "BULL"


def classify_global_liquidity_backdrop(score: float, direction: float, direction_state: str = "") -> str:
    value = safe_numeric(score)
    delta = safe_numeric(direction)
    state = str(direction_state or "").upper()
    if not state and np.isfinite(delta):
        if delta > 10.0:
            state = "ACCELERATING"
        elif delta > 5.0:
            state = "IMPROVING"
        elif delta >= -5.0:
            state = "STABLE"
        elif delta >= -10.0:
            state = "DETERIORATING"
        else:
            state = "DETERIORATING_FAST"
    if not np.isfinite(value) or not state:
        return "DATA_INCOMPLETE"
    if value < 20.0 and state == "DETERIORATING_FAST":
        return "STRONGLY_NEGATIVE"
    if value > 60.0 and state in {"IMPROVING", "ACCELERATING"}:
        return "SUPPORTIVE"
    if value > 60.0 and state in {"DETERIORATING", "DETERIORATING_FAST"}:
        return "SUPPORTIVE_BUT_WEAKENING"
    if 40.0 <= value <= 60.0 and state == "STABLE":
        return "NEUTRAL"
    if 40.0 <= value <= 60.0 and state == "DETERIORATING_FAST":
        return "LIQUIDITY_WARNING"
    if value < 40.0 and state in {"IMPROVING", "ACCELERATING"}:
        return "EARLY_REACCELERATION"
    if value < 40.0 and state in {"DETERIORATING", "DETERIORATING_FAST"}:
        return "NEGATIVE"
    if value > 60.0:
        return "SUPPORTIVE"
    if value >= 40.0:
        return "LIQUIDITY_WARNING" if state in {"DETERIORATING", "DETERIORATING_FAST"} else "NEUTRAL"
    return "EARLY_REACCELERATION" if state in {"IMPROVING", "ACCELERATING"} else "NEGATIVE"


def classify_credit_state(value: float) -> str:
    if not np.isfinite(value):
        return "DATA_INCOMPLETE"
    if value < 40.0:
        return "BENIGN"
    if value < 60.0:
        return "NORMAL"
    if value < 75.0:
        return "WATCH"
    if value < 90.0:
        return "WIDENING"
    return "SEVERE_WIDENING"


def classify_credit_level_state(value: float) -> str:
    if not np.isfinite(value):
        return "DATA_INCOMPLETE"
    if value >= 80.0:
        return "STRESSED_LEVEL"
    if value >= 60.0:
        return "ELEVATED_LEVEL"
    return "BENIGN_LEVEL"


def calculate_alpha_confidence(regime: str, fast_risk: float, macro_risk: float, config: dict | None = None) -> float:
    cfg = (config or MARKET_MODEL_CONFIG)["structural"]
    base = float(cfg["alpha_confidence"].get(regime, np.nan))
    if not np.isfinite(base):
        return np.nan
    fast_modifier = transition_confidence_modifier(fast_risk, [(40.0, 0.0), (60.0, 15.0), (80.0, 30.0), (np.inf, 50.0)])
    macro_modifier = transition_confidence_modifier(macro_risk, [(40.0, 0.0), (60.0, 10.0), (80.0, 20.0), (np.inf, 30.0)])
    confidence = float(np.clip(base - fast_modifier - macro_modifier, 0.0, 100.0))
    if regime == "CORRECTION":
        return min(confidence, 50.0)
    if regime == "STRESS":
        return min(confidence, 20.0)
    return confidence


def transition_confidence_modifier(risk: float, bands: list[tuple[float, float]]) -> float:
    if not np.isfinite(risk):
        return 0.0
    for max_value, modifier in bands:
        if risk <= max_value:
            return float(modifier)
    return float(bands[-1][1])


def safe_numeric(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def transition_state(score: float, alert_label: str) -> str:
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score <= 20.0:
        return "LOW"
    if score <= 40.0:
        return "WATCH"
    if score <= 60.0:
        return "DETERIORATING"
    if score <= 80.0:
        return "HIGH_RISK"
    return alert_label


def fast_transition_state(score: float) -> str:
    if not np.isfinite(score):
        return "DATA_INCOMPLETE"
    if score <= 10.0:
        return "LOW"
    if score <= 20.0:
        return "NORMAL"
    if score <= 40.0:
        return "WATCH"
    if score <= 60.0:
        return "HIGH"
    return "EXTREME"


def fast_risk_direction_state(delta_4w: float) -> str:
    if not np.isfinite(delta_4w):
        return "DATA_INCOMPLETE"
    if delta_4w > 20.0:
        return "RAPID_DETERIORATION"
    if delta_4w > 10.0:
        return "DETERIORATING"
    if delta_4w < -10.0:
        return "IMPROVING"
    return "STABLE"


def weekly_close(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame["Close"], errors="coerce").dropna().sort_index()


def fred_series_weekly(fred_data: pd.DataFrame | None) -> dict[str, pd.Series]:
    if fred_data is None or fred_data.empty:
        return {}
    frame = fred_data.copy()
    frame["Series_ID"] = frame["Series_ID"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    out = {}
    for series_id, group in frame.dropna(subset=["Date"]).groupby("Series_ID"):
        values = group.sort_values("Date").set_index("Date")["Value"].dropna()
        if not values.empty:
            out[str(series_id)] = values.resample("W-FRI").last().ffill().dropna()
    return out


def clean_weekly_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if values.empty:
        return pd.Series(dtype="float64")
    values.index = pd.to_datetime(values.index, errors="coerce")
    values = values[values.index.notna()].sort_index()
    return values.resample("W-FRI").last().ffill().dropna()


def trailing_percentile(series: pd.Series, window: int = 156, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def rank_last(window_values: np.ndarray) -> float:
        clean = window_values[np.isfinite(window_values)]
        if len(clean) < min_periods or not np.isfinite(window_values[-1]):
            return np.nan
        return float((clean <= window_values[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def ratio_return(numerator: pd.Series, denominator: pd.Series, window: int) -> float:
    if numerator.empty or denominator.empty:
        return np.nan
    ratio = (numerator / denominator.reindex(numerator.index, method="ffill")).replace([np.inf, -np.inf], np.nan).dropna()
    return safe_last(ratio.pct_change(window)) if not ratio.empty else np.nan


def ratio_return_series(numerator: pd.Series, denominator: pd.Series, window: int) -> pd.Series:
    if numerator.empty or denominator.empty:
        return pd.Series(dtype="float64")
    index = union_series_index([numerator, denominator])
    if index.empty:
        return pd.Series(dtype="float64")
    numerator_values = pd.to_numeric(numerator, errors="coerce").sort_index().reindex(index).ffill()
    denominator_values = pd.to_numeric(denominator, errors="coerce").sort_index().reindex(index).ffill()
    ratio = (numerator_values / denominator_values).replace([np.inf, -np.inf], np.nan)
    return ratio.pct_change(window)


def calculate_rsi_divergence(close: pd.Series, cfg: dict) -> str:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if len(prices) < int(cfg["rsi_period"]) + 20:
        return "NONE"
    rsi = RSIIndicator(close=prices, window=int(cfg["rsi_period"])).rsi().dropna()
    common = prices.reindex(rsi.index).dropna()
    highs = []
    for i in range(1, len(common) - 1):
        if common.iloc[i] > common.iloc[i - 1] and common.iloc[i] >= common.iloc[i + 1]:
            highs.append(i)
    if len(highs) < 2:
        return "NONE"
    min_dist = int(cfg["rsi_swing_distance"])
    max_dist = int(cfg["rsi_swing_max_distance"])
    for i2 in reversed(highs):
        for i1 in reversed([idx for idx in highs if min_dist <= i2 - idx <= max_dist]):
            price_1 = float(common.iloc[i1])
            price_2 = float(common.iloc[i2])
            rsi_1 = float(rsi.iloc[i1])
            rsi_2 = float(rsi.iloc[i2])
            if price_2 <= price_1 or rsi_2 >= rsi_1:
                continue
            price_gain = price_2 / price_1 - 1.0
            rsi_drop = rsi_1 - rsi_2
            if price_gain >= 0.05 and rsi_drop >= 8.0:
                return "STRONG"
            if price_gain >= 0.03 and rsi_drop >= 5.0:
                return "MODERATE"
            if price_gain > 0.0 and rsi_drop >= 3.0:
                return "MILD"
    return "NONE"


def calculate_rsi_divergence_history(close: pd.Series, cfg: dict) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if prices.empty:
        return pd.Series(dtype="object")
    min_points = int(cfg["rsi_period"]) + 20
    values = []
    for current_date in prices.index:
        history = prices.loc[:current_date]
        values.append(calculate_rsi_divergence(history, cfg) if len(history) >= min_points else "NONE")
    return pd.Series(values, index=prices.index, dtype="object")


def classify_wti_confirmation(value: float) -> str:
    if not np.isfinite(value):
        return "DATA_INCOMPLETE"
    if value <= 0.05:
        return "POSITIVE"
    if value <= 0.10:
        return "MILD_NEGATIVE"
    if value <= 0.20:
        return "NEGATIVE"
    return "STRONG_NEGATIVE"


def classify_real_yield_confirmation(change_bp: float) -> str:
    if not np.isfinite(change_bp):
        return "DATA_INCOMPLETE"
    if change_bp <= 0.0:
        return "POSITIVE"
    if change_bp <= 20.0:
        return "NEUTRAL"
    if change_bp <= 40.0:
        return "NEGATIVE"
    return "STRONG_NEGATIVE"


def classify_ratio_confirmation(value: float) -> str:
    if not np.isfinite(value):
        return "DATA_INCOMPLETE"
    if value >= 0.0:
        return "POSITIVE"
    if value >= -0.02:
        return "NEUTRAL"
    if value >= -0.05:
        return "NEGATIVE"
    return "STRONG_NEGATIVE"


def scalar_piecewise_score(value: float, points: list[tuple[float, float]] | list[list[float]]) -> float:
    if not np.isfinite(value):
        return np.nan
    sorted_points = sorted(points, key=lambda point: float(point[0]))
    xp = np.array([float(point[0]) for point in sorted_points], dtype="float64")
    fp = np.array([float(point[1]) for point in sorted_points], dtype="float64")
    return float(np.clip(np.interp(float(value), xp, fp), 0.0, 100.0))


def piecewise_score_series(values: pd.Series, points: list[tuple[float, float]] | list[list[float]]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    sorted_points = sorted(points, key=lambda point: float(point[0]))
    xp = np.array([float(point[0]) for point in sorted_points], dtype="float64")
    fp = np.array([float(point[1]) for point in sorted_points], dtype="float64")
    out = pd.Series(np.nan, index=numeric.index, dtype="float64")
    mask = numeric.notna()
    if mask.any():
        out.loc[mask] = np.interp(numeric.loc[mask].astype(float), xp, fp)
    return out.clip(0.0, 100.0)


def union_series_index(series_list: list[pd.Series]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for series in series_list:
        if series is None or series.empty:
            continue
        index = index.union(pd.DatetimeIndex(pd.to_datetime(series.dropna().index)))
    return index.sort_values()


def safe_last(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def download_fred_market_data(api_key: str | None = None) -> pd.DataFrame:
    try:
        data = download_fred_series_batch(
            FRED_MARKET_SERIES_IDS,
            api_key=api_key,
            observation_start=FRED_DEFAULT_OBSERVATION_START,
        )
    except FredApiError:
        return pd.DataFrame(columns=["Series_ID", "Date", "Value"])
    if data.empty:
        return data
    if FED_LIQUIDITY_SERIES_ID not in set(data["Series_ID"].astype(str)):
        data = pd.concat([data, calculate_fed_liquidity(data)], ignore_index=True)
    return data

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
        "fed_liquidity_weight": 0.30,
        "us2y_weight": 0.30,
        "dxy_window": 26,
        "liquidity_window": 26,
        "liquidity_short_window": 13,
        "us2y_window": 13,
        "fed_liquidity_risk_points": [(-0.12, 80.0), (-0.08, 65.0), (-0.05, 45.0), (-0.02, 25.0), (0.0, 10.0), (0.02, 5.0), (0.05, 0.0)],
        "us2y_risk_points": [(-75.0, 0.0), (-25.0, 5.0), (0.0, 10.0), (25.0, 25.0), (50.0, 45.0), (75.0, 65.0), (100.0, 85.0)],
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

FRED_MARKET_SERIES_IDS = ("WALCL", "RRPONTSYD", "WTREGEN", "DGS2", "DFII10")
YAHOO_MARKET_TICKERS = ("SPY", "IWM", "XLI", "XLP", "^VIX", "DX-Y.NYB", "CL=F")


def market_model_config() -> dict:
    return deepcopy(MARKET_MODEL_CONFIG)


def calculate_market_model(
    yahoo_weekly: dict[str, pd.DataFrame],
    fred_data: pd.DataFrame | None,
    config: dict | None = None,
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
    )
    confirmations = calculate_confirmations(yahoo_weekly, fred_data, cfg)
    overall_status = calculate_overall_transition_status(
        fast.get("Fast_Transition_Risk"),
        macro.get("Macro_Transition_Risk"),
        confirmations.get("Negative_Confirmation_Count"),
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
        **confirmations,
        "Overall_Transition_Status": overall_status,
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
        state = transition_state(risk, alert_label="TRANSITION_ALERT")
    return {
        "Fast_Transition_Risk": risk,
        "Fast_Transition_State": state,
        "VIX_Z26": vix_z,
        "VIX_Risk": vix_risk,
        "DXY_Return_13W": dxy_13w,
        "DXY_Return_26W": dxy_26w,
        "DXY_Risk": dxy_risk,
    }


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


def calculate_macro_transition_risk(dxy: pd.Series, fred_data: pd.DataFrame | None, config: dict | None = None) -> dict:
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

    components = [dxy_risk, fed_liquidity_risk, us2y_risk]
    if sum(np.isfinite(component) for component in components) < 2:
        risk = np.nan
        state = "DATA_INCOMPLETE"
    else:
        weights = np.array([float(cfg["dxy_weight"]), float(cfg["fed_liquidity_weight"]), float(cfg["us2y_weight"])])
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
        "Macro_DXY_Risk": dxy_risk,
    }


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


def calculate_overall_transition_status(fast_risk: float, macro_risk: float, negative_confirmations: float) -> str:
    fast = float(fast_risk) if np.isfinite(fast_risk) else np.nan
    macro = float(macro_risk) if np.isfinite(macro_risk) else np.nan
    negative = float(negative_confirmations) if np.isfinite(negative_confirmations) else np.nan
    if not np.isfinite(fast) or not np.isfinite(macro) or not np.isfinite(negative):
        return "DATA_INCOMPLETE"
    if fast > 80.0 or (fast > 60.0 and macro > 60.0):
        return "TRANSITION_ALERT"
    if fast > 60.0 and macro > 40.0:
        return "HIGH_RISK"
    if fast > 40.0 or macro > 60.0 or negative >= 3.0:
        return "DETERIORATING"
    if (20.0 < fast <= 40.0) or macro > 40.0:
        return "TRANSITION_WATCH"
    if fast <= 40.0 and 20.0 < macro <= 40.0:
        return "MACRO_WATCH"
    if fast <= 20.0 and macro <= 20.0 and negative <= 1.0:
        return "STABLE"
    return "WATCH"


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


def ratio_return(numerator: pd.Series, denominator: pd.Series, window: int) -> float:
    if numerator.empty or denominator.empty:
        return np.nan
    ratio = (numerator / denominator.reindex(numerator.index, method="ffill")).replace([np.inf, -np.inf], np.nan).dropna()
    return safe_last(ratio.pct_change(window)) if not ratio.empty else np.nan


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
    xp = np.array([float(point[0]) for point in points], dtype="float64")
    fp = np.array([float(point[1]) for point in points], dtype="float64")
    return float(np.clip(np.interp(float(value), xp, fp), 0.0, 100.0))


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

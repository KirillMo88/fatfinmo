from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CURRENT_RISK_MODEL_VERSION = "CURRENT_RISK_V1"


def calculate_current_risk_v1(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the frozen daily Current Risk model without look-ahead."""
    if frame.empty or "Date" not in frame or "SPY_Close" not in frame:
        return frame.copy()

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Production carries explicit raw closes so the model never mixes
    # dividend-adjusted daily prices with a raw weekly price context.
    spy = numeric(out, "SPY_RawClose") if "SPY_RawClose" in out else numeric(out, "SPY_Close")
    qqq = numeric(out, "QQQ_RawClose") if "QQQ_RawClose" in out else numeric(out, "QQQ")
    vix = numeric(out, "VIX")
    breadth = numeric(out, "SPXAboveSMA50D")
    hy_oas = numeric(out, "HY_OAS")
    sma200w = numeric(out, "SPY_SMA200W")
    weekly_rsi = numeric(out, "SPY_WeeklyRSI14")
    weekly_high26 = numeric(out, "SPY_26W_High")
    weekly_rsi_max26 = numeric(out, "SPY_WeeklyRSI26WMax")
    drawdown_risk = numeric(out, "SPX_1M_DD20_Probability")

    high20 = spy.rolling(20, min_periods=20).max()
    roc20 = spy.pct_change(20, fill_method=None)
    roc20_max = roc20.rolling(20, min_periods=20).max()
    roc12 = 100.0 * spy.pct_change(12, fill_method=None)
    roc12_max20 = roc12.rolling(20, min_periods=20).max()
    roc_divergence_gap = roc12_max20 - roc12
    breadth20_max = breadth.rolling(20, min_periods=10).max()
    breadth5_change = breadth - breadth.shift(5)
    daily_rsi = wilder_rsi(spy, 14)
    daily_rsi20_max = daily_rsi.rolling(20, min_periods=10).max()
    qqq_spy = qqq / spy.replace(0.0, np.nan)
    qqq_spy20_high = qqq_spy.rolling(20, min_periods=20).max()
    qqq_divergence_depth = 1.0 - qqq_spy / qqq_spy20_high.replace(0.0, np.nan)
    hy_5d_change = hy_oas - hy_oas.shift(5)

    extension = spy / sma200w.replace(0.0, np.nan) - 1.0
    extension_condition = extension.ge(0.25)
    near_high = spy.ge(0.98 * high20)
    roc_condition = roc_divergence_gap.ge(2.0) | roc12.ge(3.0)
    breadth_condition = (breadth20_max - breadth).ge(10.0) | breadth.ge(75.0)
    daily_rsi_divergence = near_high & (daily_rsi20_max - daily_rsi).ge(15.0)
    weekly_rsi_divergence = spy.ge(0.98 * weekly_high26) & (weekly_rsi_max26 - weekly_rsi).ge(10.0)
    blowoff_rsi = daily_rsi.ge(78.0)

    pvc = (
        20 * extension_condition.astype(int)
        + 10 * near_high.astype(int)
        + 15 * roc_condition.astype(int)
        + 15 * breadth_condition.astype(int)
        + 15 * daily_rsi_divergence.astype(int)
        + 10 * weekly_rsi_divergence.astype(int)
        + 15 * blowoff_rsi.astype(int)
    ).astype(float)
    pvc_available = pd.concat(
        [extension, spy, high20, roc12, roc12_max20, breadth, breadth20_max, daily_rsi, weekly_rsi, weekly_high26, weekly_rsi_max26],
        axis=1,
    ).notna().all(axis=1)
    pvc = pvc.where(pvc_available)
    pvc_max10 = pvc.rolling(10, min_periods=1).max()

    vix_previous3_high = vix.shift(1).rolling(3, min_periods=3).max()
    vix_3d_change = vix.pct_change(3, fill_method=None)
    activation_available = pd.concat([pvc_max10, vix, vix_previous3_high, vix_3d_change, spy, high20], axis=1).notna().all(axis=1)
    activation = (
        pvc_max10.ge(70.0)
        & vix.gt(vix_previous3_high)
        & vix_3d_change.ge(0.10)
        & spy.ge(0.95 * high20)
        & activation_available
    )
    new_event = current_risk_new_event(activation, lookback_sessions=5)

    calm_vix = vix.shift(10).le(16.0)
    hy_level = hy_oas.ge(3.25)
    hy_widening = hy_5d_change.gt(0.0)
    weak_breadth = breadth.le(50.0)
    breadth_collapse = breadth5_change.le(-10.0)
    high_beta_divergence = qqq_spy.le(0.98 * qqq_spy20_high)
    escalation_inputs = pd.concat(
        [vix.shift(10), hy_oas, hy_5d_change, breadth, breadth5_change, qqq_spy, qqq_spy20_high], axis=1
    )
    escalation_available = escalation_inputs.notna().all(axis=1)
    escalation_raw = (
        calm_vix.astype(int)
        + hy_level.astype(int)
        + hy_widening.astype(int)
        + weak_breadth.astype(int)
        + breadth_collapse.astype(int)
        + high_beta_divergence.astype(int)
    ).astype(float).where(escalation_available)
    escalation_active = escalation_raw.where(activation, 0.0).where(activation_available)

    breadth_level_risk = ((75.0 - breadth) / 25.0 * 100.0).clip(0.0, 100.0)
    breadth_deterioration_risk = (-breadth5_change / 10.0 * 100.0).clip(0.0, 100.0)
    breadth_risk = pd.concat([breadth_level_risk, breadth_deterioration_risk], axis=1).max(axis=1, skipna=False)

    rsi_risk = (
        55.0 * daily_rsi_divergence.astype(float)
        + 45.0 * weekly_rsi_divergence.astype(float)
        + 55.0 * blowoff_rsi.astype(float)
    ).clip(0.0, 100.0).where(
        pd.concat([daily_rsi, daily_rsi20_max, weekly_rsi, weekly_high26, weekly_rsi_max26], axis=1)
        .notna()
        .all(axis=1)
    )

    vix_level_risk = ((vix - 12.0) / 28.0 * 100.0).clip(0.0, 100.0)
    vix_change_risk = (vix_3d_change / 0.20 * 100.0).clip(0.0, 100.0)
    vix_breakout = vix.gt(vix_previous3_high)
    vix_risk = (
        0.45 * vix_level_risk
        + 0.30 * vix_breakout.astype(float) * 100.0
        + 0.25 * vix_change_risk
    ).clip(0.0, 100.0).where(pd.concat([vix, vix_previous3_high, vix_3d_change], axis=1).notna().all(axis=1))

    high_beta_risk = (qqq_divergence_depth / 0.08 * 100.0).clip(0.0, 100.0)
    hy_level_risk = ((hy_oas - 2.50) / 2.00 * 100.0).clip(0.0, 100.0)
    hy_widening_risk = (hy_5d_change / 0.50 * 100.0).clip(0.0, 100.0)
    hy_risk = pd.concat([hy_level_risk, hy_widening_risk], axis=1).max(axis=1, skipna=False)
    hy_percentile10y = rolling_percentile_pit(hy_oas, window=2520, min_periods=1260)

    signal_class = [
        classify_signal(active, score, top, credit, available)
        for active, score, top, credit, available in zip(
            activation, escalation_active, high_beta_divergence, hy_level, activation_available & escalation_available
        )
    ]
    market_state = [
        classify_market_state(active, score, available)
        for active, score, available in zip(activation, escalation_active, activation_available & escalation_available)
    ]

    out["CurrentRiskSPY20DHigh"] = high20
    out["CurrentRiskROC20D"] = roc20
    out["CurrentRiskROC20DMax"] = roc20_max
    out["CurrentRiskROC12"] = roc12
    out["CurrentRiskROC12Max20D"] = roc12_max20
    out["CurrentRiskROCDivergenceGap"] = roc_divergence_gap
    out["CurrentRiskBreadth20DMax"] = breadth20_max
    out["CurrentRiskBreadth5DChange"] = breadth5_change
    out["CurrentRiskDailyRSI14"] = daily_rsi
    out["CurrentRiskDailyRSI20DMax"] = daily_rsi20_max
    out["CurrentRiskQQQSPY"] = qqq_spy
    out["CurrentRiskQQQSPY20DHigh"] = qqq_spy20_high
    out["CurrentRiskQQQDivergenceDepth"] = qqq_divergence_depth
    out["CurrentRiskHY5DChange"] = hy_5d_change
    out["CurrentRiskHY10YPercentile"] = hy_percentile10y
    out["CurrentRiskSPYExtension200W"] = extension
    out["CurrentRiskSPYRawClose"] = spy
    out["CurrentRiskQQQRawClose"] = qqq
    out["CurrentRiskSPYSMA200WRaw"] = sma200w
    out["CurrentRiskPriceBasis"] = "RAW_CLOSE" if "SPY_RawClose" in out and "QQQ_RawClose" in out else "LEGACY_CLOSE"

    out["PVC_ExtensionCondition"] = extension_condition
    out["PVC_NearHighCondition"] = near_high
    out["PVC_ROCCondition"] = roc_condition
    out["PVC_BreadthCondition"] = breadth_condition
    out["PVC_DailyRSIDivergence"] = daily_rsi_divergence
    out["PVC_WeeklyRSIDivergence"] = weekly_rsi_divergence
    out["PVC_BlowoffRSI"] = blowoff_rsi
    out["PVC_V2"] = pvc
    out["PVC_MAX10D"] = pvc_max10

    out["CurrentRiskVIXPrevious3DHigh"] = vix_previous3_high
    out["CurrentRiskVIX3DChange"] = vix_3d_change
    out["CurrentRiskActivation"] = activation
    out["CurrentRiskNewEvent"] = new_event
    out["CurrentRiskCalmVIXBase"] = calm_vix
    out["CurrentRiskHYLevelConfirmation"] = hy_level
    out["CurrentRiskHYWidening"] = hy_widening
    out["CurrentRiskWeakBreadth"] = weak_breadth
    out["CurrentRiskBreadthCollapse"] = breadth_collapse
    out["CurrentRiskTopConfirmation"] = high_beta_divergence
    out["CurrentRiskCreditConfirmation"] = hy_level
    out["CurrentRiskEscalationScoreRaw"] = escalation_raw
    out["CurrentRiskEscalationScoreV2"] = escalation_active
    out["CurrentRiskSignalClass"] = signal_class

    out["CurrentRiskDrawdownRisk"] = drawdown_risk.clip(0.0, 100.0)
    out["CurrentRiskPriceCycleVulnerabilityRisk"] = pvc_max10.clip(0.0, 100.0)
    out["CurrentRiskBreadthRisk"] = breadth_risk
    out["CurrentRiskRSIDivergenceRisk"] = rsi_risk
    out["CurrentRiskVIXRisk"] = vix_risk
    out["CurrentRiskHighBetaRisk"] = high_beta_risk
    out["CurrentRiskHYRisk"] = hy_risk
    for column in component_columns():
        out[f"{column}State"] = out[column].map(classify_component_state)

    out["CurrentRiskComponentAverage"] = out[component_columns()].mean(axis=1, skipna=False)
    out["CurrentMarketRiskState"] = market_state
    state_level = pd.Series(market_state, index=out.index).map(
        {"NORMAL": 0, "LOW CONFIRMATION": 1, "MODERATE": 2, "HIGH RISK": 3, "RED FLAG": 4}
    )
    out["CurrentMarketRiskDirection"] = np.select(
        [state_level.diff(5).gt(0), state_level.diff(5).lt(0)],
        ["RISING", "FALLING"],
        default="STABLE",
    )
    out["ActiveStressChannels"] = escalation_raw.where(activation, 0.0)
    out["CurrentRiskModelVersion"] = CURRENT_RISK_MODEL_VERSION
    out["CurrentRiskFrequency"] = "DAILY"
    out["CurrentRiskDataCoverage"] = pd.concat(
        [spy, vix, breadth, qqq, hy_oas, sma200w, weekly_rsi], axis=1
    ).notna().mean(axis=1) * 100.0

    out["HistoricalCurrentRiskState"] = out["CurrentMarketRiskState"]
    out["HistoricalCurrentRiskScore"] = out["CurrentRiskComponentAverage"]
    out["HistoricalCurrentRiskPrimaryDrivers"] = [primary_drivers(row) for row in out.to_dict("records")]
    out["HistoricalDrawdownRisk"] = out["CurrentRiskDrawdownRisk"]
    out["HistoricalHighBetaRisk"] = out["CurrentRiskHighBetaRisk"]
    out["HistoricalBreadthRisk"] = out["CurrentRiskBreadthRisk"]
    out["HistoricalRSIDivergenceRisk"] = out["CurrentRiskRSIDivergenceRisk"]
    out["HistoricalVIXLevelRisk"] = out["CurrentRiskVIXRisk"]
    out["HistoricalVIXMomentumRisk"] = out["CurrentRiskVIXRisk"]
    out["HistoricalVIXTermStructureRisk"] = out["CurrentRiskVIXRisk"]
    out["HistoricalCombinedVIXRisk"] = out["CurrentRiskVIXRisk"]
    return out


def current_risk_new_event(activation: pd.Series, lookback_sessions: int = 5) -> pd.Series:
    """Mark the first activation in an episode using only prior sessions."""
    active = activation.fillna(False).astype(bool)
    recently_active = (
        active.shift(1, fill_value=False)
        .rolling(lookback_sessions, min_periods=1)
        .max()
        .fillna(False)
        .astype(bool)
    )
    return active & ~recently_active


def component_columns() -> list[str]:
    return [
        "CurrentRiskDrawdownRisk",
        "CurrentRiskPriceCycleVulnerabilityRisk",
        "CurrentRiskBreadthRisk",
        "CurrentRiskRSIDivergenceRisk",
        "CurrentRiskVIXRisk",
        "CurrentRiskHighBetaRisk",
        "CurrentRiskHYRisk",
    ]


def classify_component_state(value: Any) -> str:
    score = safe_float(value)
    if not np.isfinite(score):
        return "DATA INCOMPLETE"
    if score <= 25.0:
        return "LOW"
    if score <= 50.0:
        return "MODERATE"
    if score <= 75.0:
        return "ELEVATED"
    return "HIGH"


def classify_market_state(active: Any, score: Any, available: Any) -> str:
    if not bool(available):
        return "DATA INCOMPLETE"
    if not bool(active):
        return "NORMAL"
    value = safe_float(score)
    if value <= 1:
        return "LOW CONFIRMATION"
    if value == 2:
        return "MODERATE"
    if value == 3:
        return "HIGH RISK"
    return "RED FLAG"


def classify_signal(active: Any, score: Any, top: Any, credit: Any, available: Any) -> str:
    state = classify_market_state(active, score, available)
    if state != "RED FLAG":
        return state if bool(active) else "INACTIVE"
    if bool(credit):
        return "RED FLAG + CREDIT CONFIRMATION"
    if bool(top):
        return "RED FLAG + TOP CONFIRMATION"
    return "RED FLAG"


def primary_drivers(row: dict[str, Any]) -> str:
    pairs = [
        ("Drawdown", row.get("CurrentRiskDrawdownRisk")),
        ("PVC", row.get("CurrentRiskPriceCycleVulnerabilityRisk")),
        ("Breadth", row.get("CurrentRiskBreadthRisk")),
        ("RSI", row.get("CurrentRiskRSIDivergenceRisk")),
        ("VIX", row.get("CurrentRiskVIXRisk")),
        ("QQQ", row.get("CurrentRiskHighBetaRisk")),
        ("HY", row.get("CurrentRiskHYRisk")),
    ]
    available = [(label, safe_float(value)) for label, value in pairs if np.isfinite(safe_float(value))]
    if not available:
        return "DATA INCOMPLETE"
    return ", ".join(label for label, _ in sorted(available, key=lambda item: item[1], reverse=True)[:3])


def rolling_percentile_pit(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result: list[float] = []
    for idx, value in enumerate(values):
        history = values.iloc[max(0, idx - window + 1) : idx + 1].dropna()
        if not np.isfinite(safe_float(value)) or len(history) < min_periods:
            result.append(np.nan)
        else:
            result.append(float((history <= value).mean() * 100.0))
    return pd.Series(result, index=series.index, dtype="float64")


def wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).where(avg_loss.ne(0.0), 100.0)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan

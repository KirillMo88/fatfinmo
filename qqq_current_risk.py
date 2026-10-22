from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


QQQ_CURRENT_RISK_MODEL_VERSION = "QQQ_CURRENT_RISK_V1_C"
QQQ_CURRENT_RISK_START = pd.Timestamp("2014-05-01")


@dataclass(frozen=True)
class QQQCurrentRiskConfig:
    volatility_source: str
    calm_volatility_threshold: float

    @property
    def identifier(self) -> str:
        return f"{QQQ_CURRENT_RISK_MODEL_VERSION}_{self.volatility_source}"


QQQ_CURRENT_RISK_VIX_CONFIG = QQQCurrentRiskConfig("VIX", 16.0)
QQQ_CURRENT_RISK_VXN_CONFIG = QQQCurrentRiskConfig("VXN", 18.0)
QQQ_CURRENT_RISK_CONFIGS = {
    "VIX": QQQ_CURRENT_RISK_VIX_CONFIG,
    "VXN": QQQ_CURRENT_RISK_VXN_CONFIG,
}


def calculate_qqq_current_risk_variants(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate both volatility configurations from one immutable shared snapshot."""
    shared = calculate_qqq_shared_indicators(frame)
    return {
        source: calculate_qqq_current_risk_variant(shared, config)
        for source, config in QQQ_CURRENT_RISK_CONFIGS.items()
    }


def calculate_qqq_shared_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the QQQ Variant C inputs that are identical for VIX and VXN."""
    if frame is None or frame.empty or "Date" not in frame or "QQQ_RawClose" not in frame:
        return pd.DataFrame() if frame is None else frame.copy()

    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    qqq = numeric(out, "QQQ_RawClose")
    iwm = numeric(out, "IWM_RawClose")
    ndfi = numeric(out, "NDFI")
    hy_oas = numeric(out, "HY_OAS")
    sma200w = numeric(out, "QQQ_SMA200W")
    weekly_rsi = numeric(out, "QQQ_WeeklyRSI14")
    weekly_high26 = numeric(out, "QQQ_26W_High")
    weekly_rsi_max26 = numeric(out, "QQQ_WeeklyRSI26WMax")

    high20 = qqq.rolling(20, min_periods=20).max()
    near_high_ratio = qqq / high20.replace(0.0, np.nan)
    roc12 = 100.0 * qqq.pct_change(12, fill_method=None)
    roc12_max20 = roc12.rolling(20, min_periods=20).max()
    roc_deterioration = roc12_max20 - roc12
    ndfi20_max = ndfi.rolling(20, min_periods=20).max()
    ndfi_deterioration = ndfi20_max - ndfi
    ndfi_5d_change = ndfi - ndfi.shift(5)
    daily_rsi = wilder_rsi(qqq, 14)
    daily_rsi20_max = daily_rsi.rolling(20, min_periods=20).max()
    daily_rsi_decline = daily_rsi20_max - daily_rsi
    weekly_rsi_decline = weekly_rsi_max26 - weekly_rsi
    relative_strength = iwm / qqq.replace(0.0, np.nan)
    relative_strength20_max = relative_strength.rolling(20, min_periods=20).max()
    iwm_qqq_divergence = 1.0 - relative_strength / relative_strength20_max.replace(0.0, np.nan)
    hy_5d_change = hy_oas - hy_oas.shift(5)

    extension = qqq / sma200w.replace(0.0, np.nan) - 1.0
    extension_flag = extension.ge(0.40)
    near_high_flag = near_high_ratio.ge(0.99)
    roc_flag = roc_deterioration.ge(5.0) | roc12.ge(6.0)
    breadth_flag = ndfi_deterioration.ge(12.0) | ndfi.ge(80.0)
    daily_rsi_divergence = near_high_flag & daily_rsi_decline.ge(12.0)
    weekly_rsi_divergence = qqq.ge(0.98 * weekly_high26) & weekly_rsi_decline.ge(8.0)
    rsi_blowoff = daily_rsi.ge(76.0)

    pvc_inputs = pd.concat(
        [
            extension,
            qqq,
            high20,
            roc12,
            roc12_max20,
            ndfi,
            ndfi20_max,
            daily_rsi,
            daily_rsi20_max,
            weekly_rsi,
            weekly_high26,
            weekly_rsi_max26,
        ],
        axis=1,
    )
    pvc_available = pvc_inputs.notna().all(axis=1)
    pvc = (
        20 * extension_flag.astype(int)
        + 10 * near_high_flag.astype(int)
        + 15 * roc_flag.astype(int)
        + 15 * breadth_flag.astype(int)
        + 15 * daily_rsi_divergence.astype(int)
        + 10 * weekly_rsi_divergence.astype(int)
        + 15 * rsi_blowoff.astype(int)
    ).astype(float).where(pvc_available)
    pvc_max10 = pvc.rolling(10, min_periods=1).max()

    out["QQQ_20D_High"] = high20
    out["QQQ_NearHighRatio"] = near_high_ratio
    out["QQQ_ROC12"] = roc12
    out["QQQ_ROC12Max20D"] = roc12_max20
    out["QQQ_ROCDeterioration"] = roc_deterioration
    out["NDFI20DMax"] = ndfi20_max
    out["NDFIDeterioration"] = ndfi_deterioration
    out["NDFI_5D_Change"] = ndfi_5d_change
    out["QQQ_DailyRSI14"] = daily_rsi
    out["QQQ_DailyRSI20DMax"] = daily_rsi20_max
    out["QQQ_DailyRSIDecline"] = daily_rsi_decline
    out["QQQ_WeeklyRSIDecline"] = weekly_rsi_decline
    out["IWM_QQQ_RelativeStrength"] = relative_strength
    out["IWM_QQQ_RelativeStrength20DMax"] = relative_strength20_max
    out["IWMQQQDivergence"] = iwm_qqq_divergence
    out["HY_OAS_5D_Change"] = hy_5d_change
    out["QQQ_Extension200W"] = extension

    out["PVC_ExtensionCondition"] = extension_flag
    out["PVC_NearHighCondition"] = near_high_flag
    out["PVC_ROCCondition"] = roc_flag
    out["PVC_BreadthCondition"] = breadth_flag
    out["PVC_DailyRSIDivergence"] = daily_rsi_divergence
    out["PVC_WeeklyRSIDivergence"] = weekly_rsi_divergence
    out["PVC_RSIBlowoff"] = rsi_blowoff
    out["PVC_ExtensionContribution"] = 20 * extension_flag.astype(int)
    out["PVC_NearHighContribution"] = 10 * near_high_flag.astype(int)
    out["PVC_ROCContribution"] = 15 * roc_flag.astype(int)
    out["PVC_BreadthContribution"] = 15 * breadth_flag.astype(int)
    out["PVC_DailyRSIDivergenceContribution"] = 15 * daily_rsi_divergence.astype(int)
    out["PVC_WeeklyRSIDivergenceContribution"] = 10 * weekly_rsi_divergence.astype(int)
    out["PVC_RSIBlowoffContribution"] = 15 * rsi_blowoff.astype(int)
    out["PVC_Complete"] = pvc_available
    out["PVC"] = pvc
    out["PVC_V2"] = pvc
    out["PVC_MAX10D"] = pvc_max10
    return out


def calculate_qqq_current_risk_variant(
    shared: pd.DataFrame,
    config: QQQCurrentRiskConfig,
) -> pd.DataFrame:
    """Apply one volatility configuration to precomputed QQQ shared indicators."""
    if shared is None or shared.empty:
        return pd.DataFrame() if shared is None else shared.copy()
    if config.volatility_source not in QQQ_CURRENT_RISK_CONFIGS:
        raise ValueError(f"unsupported volatility source: {config.volatility_source}")

    out = shared.copy()
    dates = pd.to_datetime(out["Date"], errors="coerce")
    qqq = numeric(out, "QQQ_RawClose")
    high20 = numeric(out, "QQQ_20D_High")
    pvc_max10 = numeric(out, "PVC_MAX10D")
    vol = numeric(out, config.volatility_source)
    ndfi = numeric(out, "NDFI")
    ndfi_5d_change = numeric(out, "NDFI_5D_Change")
    hy_oas = numeric(out, "HY_OAS")
    hy_5d_change = numeric(out, "HY_OAS_5D_Change")
    divergence = numeric(out, "IWMQQQDivergence")

    previous3_high = vol.shift(1).rolling(3, min_periods=3).max()
    volatility3d_change = vol.pct_change(3, fill_method=None)
    volatility_breakout = vol.gt(previous3_high)
    price_near_high = qqq.ge(0.95 * high20)
    eligible_date = dates.ge(QQQ_CURRENT_RISK_START)
    activation_available = pd.concat(
        [pvc_max10, vol, previous3_high, volatility3d_change, qqq, high20], axis=1
    ).notna().all(axis=1) & out["PVC_Complete"].fillna(False).astype(bool) & eligible_date
    activation = (
        pvc_max10.ge(50.0)
        & volatility_breakout
        & volatility3d_change.ge(0.10)
        & price_near_high
        & activation_available
    )
    calm_level = vol.shift(10)
    calm_volatility = calm_level.le(config.calm_volatility_threshold)
    hy_level = hy_oas.ge(3.25)
    hy_widening = hy_5d_change.gt(0.0)
    weak_breadth = ndfi.le(50.0)
    breadth_collapse = ndfi_5d_change.le(-12.0)
    top_confirmation = divergence.ge(0.03)
    escalation_available = pd.concat(
        [calm_level, hy_oas, hy_5d_change, ndfi, ndfi_5d_change, divergence], axis=1
    ).notna().all(axis=1) & eligible_date
    escalation_raw = (
        calm_volatility.astype(int)
        + hy_level.astype(int)
        + hy_widening.astype(int)
        + weak_breadth.astype(int)
        + breadth_collapse.astype(int)
        + top_confirmation.astype(int)
    ).astype(float).where(escalation_available)
    escalation = escalation_raw.where(activation, 0.0).where(activation_available & escalation_available)
    state_available = activation_available & escalation_available

    market_state = [
        classify_market_state(active, score, available)
        for active, score, available in zip(activation, escalation, state_available)
    ]
    display_class = [
        classify_signal(active, score, top, credit, available)
        for active, score, top, credit, available in zip(
            activation, escalation, top_confirmation, hy_level, state_available
        )
    ]
    confirmation = [
        confirmation_label(signal) for signal in display_class
    ]
    ordinal = pd.Series(market_state, index=out.index).map(
        {"NORMAL": 0, "LOW CONFIRMATION": 1, "MODERATE": 2, "HIGH RISK": 3, "RED FLAG": 4}
    )
    direction = np.select(
        [ordinal.diff(5).gt(0), ordinal.diff(5).lt(0)],
        ["RISING", "FALLING"],
        default="STABLE",
    )
    direction = pd.Series(direction, index=out.index).where(ordinal.notna(), "DATA INCOMPLETE")

    breadth_level_risk = ((75.0 - ndfi) / 25.0 * 100.0).clip(0.0, 100.0)
    breadth_deterioration_risk = (-ndfi_5d_change / 10.0 * 100.0).clip(0.0, 100.0)
    breadth_risk = pd.concat([breadth_level_risk, breadth_deterioration_risk], axis=1).max(axis=1, skipna=False)
    rsi_risk = (
        55.0 * out["PVC_DailyRSIDivergence"].astype(float)
        + 45.0 * out["PVC_WeeklyRSIDivergence"].astype(float)
        + 55.0 * out["PVC_RSIBlowoff"].astype(float)
    ).clip(0.0, 100.0).where(out["PVC_Complete"])
    volatility_level_risk = ((vol - 12.0) / 28.0 * 100.0).clip(0.0, 100.0)
    volatility_change_risk = (volatility3d_change / 0.20 * 100.0).clip(0.0, 100.0)
    volatility_risk = (
        0.45 * volatility_level_risk
        + 0.30 * volatility_breakout.astype(float) * 100.0
        + 0.25 * volatility_change_risk
    ).clip(0.0, 100.0).where(
        pd.concat([vol, previous3_high, volatility3d_change], axis=1).notna().all(axis=1)
    )
    high_beta_risk = (divergence / 0.08 * 100.0).clip(0.0, 100.0)
    hy_level_risk = ((hy_oas - 2.50) / 2.00 * 100.0).clip(0.0, 100.0)
    hy_widening_risk = (hy_5d_change / 0.50 * 100.0).clip(0.0, 100.0)
    hy_risk = pd.concat([hy_level_risk, hy_widening_risk], axis=1).max(axis=1, skipna=False)
    drawdown_risk = numeric(out, "QQQ_1M_DD20_Probability").clip(0.0, 100.0)

    out["VolatilitySource"] = config.volatility_source
    out["VolatilityLevel"] = vol
    out["VolatilityPrevious3DHigh"] = previous3_high
    out["Volatility3DChange"] = volatility3d_change
    out["VolatilityBreakout"] = volatility_breakout
    out["QQQActivationNearHigh"] = price_near_high
    out["CurrentRiskActivationAvailable"] = activation_available
    out["CurrentRiskActivation"] = activation
    out["CurrentRiskCalmVolatilityLevel"] = calm_level
    out["CurrentRiskCalmVolatility"] = calm_volatility
    out["CurrentRiskHYLevelConfirmation"] = hy_level
    out["CurrentRiskHYWidening"] = hy_widening
    out["CurrentRiskWeakBreadth"] = weak_breadth
    out["CurrentRiskBreadthCollapse"] = breadth_collapse
    out["CurrentRiskTopConfirmation"] = top_confirmation
    out["CurrentRiskCreditConfirmation"] = hy_level
    out["CurrentRiskEscalationAvailable"] = escalation_available
    out["CurrentRiskEscalationScoreRaw"] = escalation_raw
    out["CurrentRiskEscalationScore"] = escalation
    out["CurrentRiskEscalationScoreV2"] = escalation
    out["CurrentMarketRiskState"] = market_state
    out["CurrentRiskDisplayClass"] = display_class
    out["CurrentRiskSignalClass"] = display_class
    out["CurrentRiskConfirmation"] = confirmation
    out["CurrentMarketRiskDirection"] = direction

    out["CurrentRiskDrawdownRisk"] = drawdown_risk
    out["CurrentRiskPriceCycleVulnerabilityRisk"] = pvc_max10.clip(0.0, 100.0)
    out["CurrentRiskBreadthRisk"] = breadth_risk
    out["CurrentRiskRSIDivergenceRisk"] = rsi_risk
    out["CurrentRiskVolatilityRisk"] = volatility_risk
    out["CurrentRiskHighBetaRisk"] = high_beta_risk
    out["CurrentRiskHYRisk"] = hy_risk
    component_columns = qqq_component_columns()
    for column in component_columns:
        out[f"{column}State"] = out[column].map(classify_component_state)
    out["CurrentRiskComponentAverage"] = out[component_columns].mean(axis=1, skipna=False)
    out["HistoricalCurrentRiskPrimaryDrivers"] = [primary_drivers(row, config.volatility_source) for row in out.to_dict("records")]

    out["CurrentRiskModelVersion"] = QQQ_CURRENT_RISK_MODEL_VERSION
    out["CurrentRiskConfiguration"] = config.identifier
    out["CurrentRiskFrequency"] = "DAILY"
    out["CurrentRiskDataCoverage"] = pd.concat(
        [qqq, vol, ndfi, numeric(out, "IWM_RawClose"), hy_oas, numeric(out, "QQQ_SMA200W"), numeric(out, "QQQ_WeeklyRSI14")],
        axis=1,
    ).notna().mean(axis=1) * 100.0
    out["HistoricalCurrentRiskState"] = out["CurrentMarketRiskState"]
    out["HistoricalCurrentRiskScore"] = out["CurrentRiskComponentAverage"]
    out["HistoricalDrawdownRisk"] = out["CurrentRiskDrawdownRisk"]
    out["HistoricalHighBetaRisk"] = out["CurrentRiskHighBetaRisk"]
    out["HistoricalBreadthRisk"] = out["CurrentRiskBreadthRisk"]
    out["HistoricalRSIDivergenceRisk"] = out["CurrentRiskRSIDivergenceRisk"]
    out["HistoricalVolatilityRisk"] = out["CurrentRiskVolatilityRisk"]

    event_state = calculate_qqq_current_risk_event_engine(out)
    for column in event_state.columns:
        out[column] = event_state[column].to_numpy()
    out["CurrentRiskLastEventDate"] = out["Date"].where(out["CurrentRiskNewEvent"]).ffill()
    return out


def build_qqq_weekly_context(qqq_ohlcv: pd.DataFrame, now: Any = None) -> pd.DataFrame:
    """Build causal completed-week indicators and the QQQ historical drawdown analog."""
    weekly = completed_weekly_ohlc(qqq_ohlcv, now=now)
    if weekly.empty:
        return pd.DataFrame()
    close = numeric(weekly, "QQQ_WeeklyClose")
    low = numeric(weekly, "QQQ_WeeklyLow")
    weekly["QQQ_SMA200W"] = close.rolling(200, min_periods=200).mean()
    weekly["QQQ_WeeklyRSI14"] = wilder_rsi(close, 14)
    weekly["QQQ_26W_High"] = close.rolling(26, min_periods=26).max()
    weekly["QQQ_WeeklyRSI26WMax"] = weekly["QQQ_WeeklyRSI14"].rolling(26, min_periods=26).max()
    weekly["SMA200WExtensionPct"] = close / weekly["QQQ_SMA200W"].replace(0.0, np.nan) - 1.0
    weekly["SMA200WExtensionPercentile"] = expanding_percentile(weekly["SMA200WExtensionPct"], min_periods=200)
    weekly["ForwardMaxDrawdown_1M"] = forward_window_drawdown(close, low, weeks=4)

    monthly = completed_monthly_close(qqq_ohlcv)
    if not monthly.empty:
        monthly_close = numeric(monthly, "QQQ_MonthlyClose")
        monthly["QQQ_ROC12M"] = monthly_close.pct_change(12, fill_method=None)
        monthly["QQQ_ROC12M_3MMA"] = monthly["QQQ_ROC12M"].rolling(3, min_periods=2).mean()
        monthly["QQQ_ROC_Momentum_3M"] = monthly["QQQ_ROC12M_3MMA"] - monthly["QQQ_ROC12M_3MMA"].shift(3)
        candidate = pd.Series(
            [classify_momentum_phase(level, momentum) for level, momentum in zip(monthly["QQQ_ROC12M_3MMA"], monthly["QQQ_ROC_Momentum_3M"])],
            index=monthly.index,
        )
        monthly["MomentumCyclePhase"] = confirm_state(candidate, required=2)
        weekly = pd.merge_asof(
            weekly.sort_values("Date"),
            monthly[["Date", "MomentumCyclePhase"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )
    else:
        weekly["MomentumCyclePhase"] = "DATA INCOMPLETE"
    add_qqq_forward_drawdown_risk(weekly)
    return weekly


def completed_weekly_ohlc(frame: pd.DataFrame, now: Any = None) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame(columns=["Date", "QQQ_WeeklyClose", "QQQ_WeeklyLow"])
    source = frame.copy()
    source.index = pd.to_datetime(source.index, errors="coerce").tz_localize(None)
    source = source.loc[~source.index.isna()].sort_index()
    source["Close"] = pd.to_numeric(source["Close"], errors="coerce")
    source["Low"] = pd.to_numeric(source.get("Low", source["Close"]), errors="coerce")
    source = source.dropna(subset=["Close"])
    if source.empty:
        return pd.DataFrame(columns=["Date", "QQQ_WeeklyClose", "QQQ_WeeklyLow"])
    source["_period"] = source.index.to_period("W-FRI")
    rows = []
    for period, group in source.groupby("_period", sort=True):
        rows.append(
            {
                "Date": pd.Timestamp(group.index[-1]).normalize(),
                "PeriodEnd": period.end_time.normalize(),
                "QQQ_WeeklyClose": float(group["Close"].iloc[-1]),
                "QQQ_WeeklyLow": float(group["Low"].min()),
            }
        )
    out = pd.DataFrame(rows)
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    current = current.tz_localize(None) if current.tzinfo is not None else current
    if not out.empty:
        final = out.index[-1]
        actual = pd.Timestamp(out.loc[final, "Date"])
        period_end = pd.Timestamp(out.loc[final, "PeriodEnd"])
        if actual.weekday() != 4 and current.normalize() <= period_end:
            out = out.iloc[:-1]
    return out.drop(columns=["PeriodEnd"], errors="ignore").reset_index(drop=True)


def completed_monthly_close(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame(columns=["Date", "QQQ_MonthlyClose"])
    source = pd.to_numeric(frame["Close"], errors="coerce").dropna().copy()
    source.index = pd.to_datetime(source.index, errors="coerce").tz_localize(None)
    source = source.loc[~source.index.isna()].sort_index()
    rows = []
    for _, group in source.groupby(source.index.to_period("M"), sort=True):
        rows.append({"Date": pd.Timestamp(group.index[-1]).normalize(), "QQQ_MonthlyClose": float(group.iloc[-1])})
    return pd.DataFrame(rows)


def add_qqq_forward_drawdown_risk(weekly: pd.DataFrame, min_spacing_weeks: int = 8) -> None:
    dates = pd.to_datetime(weekly["Date"], errors="coerce")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    drawdowns = numeric(weekly, "ForwardMaxDrawdown_1M").to_numpy(dtype="float64")
    phases = weekly.get("MomentumCyclePhase", pd.Series("DATA INCOMPLETE", index=weekly.index)).astype(str).to_numpy()
    percentiles = numeric(weekly, "SMA200WExtensionPercentile").to_numpy(dtype="float64")
    all_indices = np.arange(len(weekly), dtype=int)
    min_spacing = np.timedelta64(int(min_spacing_weeks * 7), "D")

    def spaced(indices: np.ndarray) -> np.ndarray:
        kept: list[int] = []
        last_date: np.datetime64 | None = None
        for raw_index in indices:
            current_date = date_values[int(raw_index)]
            if last_date is None or current_date - last_date >= min_spacing:
                kept.append(int(raw_index))
                last_date = current_date
        return np.asarray(kept, dtype=int)

    probabilities: list[float] = []
    sample_sizes: list[int] = []
    sample_modes: list[str] = []
    confidences: list[str] = []
    for index in range(len(weekly)):
        phase = phases[index]
        percentile = safe_float(percentiles[index])
        if phase == "DATA INCOMPLETE" or not np.isfinite(percentile):
            probabilities.append(np.nan)
            sample_sizes.append(0)
            sample_modes.append("DATA INCOMPLETE")
            confidences.append("INSUFFICIENT SAMPLE")
            continue
        eligible = all_indices[
            (date_values >= QQQ_CURRENT_RISK_START.to_datetime64())
            & (date_values < date_values[index] - np.timedelta64(28, "D"))
            & np.isfinite(drawdowns)
        ]
        selected = np.asarray([], dtype=int)
        mode = "MATCHED_BAND"
        for band in (10.0, 15.0, 20.0):
            raw = eligible[(phases[eligible] == phase) & np.isfinite(percentiles[eligible]) & (np.abs(percentiles[eligible] - percentile) <= band)]
            selected = spaced(raw)
            if len(selected) >= 10 or band == 20.0:
                break
        if len(selected) < 10:
            phase_selected = spaced(eligible[phases[eligible] == phase])
            if len(phase_selected) > len(selected):
                selected = phase_selected
                mode = "PHASE_ONLY_FALLBACK"
        if len(selected) < 10:
            broad_selected = spaced(eligible)
            if len(broad_selected) > len(selected):
                selected = broad_selected
                mode = "BROAD_HISTORY_FALLBACK"
        sample = drawdowns[selected]
        sample = sample[np.isfinite(sample)]
        probabilities.append(float((sample <= -0.20).mean() * 100.0) if len(sample) else np.nan)
        sample_sizes.append(int(len(sample)))
        sample_modes.append(mode)
        confidences.append(drawdown_probability_confidence(len(sample), mode))
    weekly["QQQ_1M_DD20_Probability"] = probabilities
    weekly["QQQ_1M_DD20_AnalogN"] = sample_sizes
    weekly["QQQ_1M_DD20_SampleMode"] = sample_modes
    weekly["QQQ_1M_DD20_Confidence"] = confidences


def drawdown_probability_confidence(n: int, sample_mode: str = "MATCHED_BAND") -> str:
    suffix = "" if sample_mode == "MATCHED_BAND" else f" ({sample_mode})"
    if n >= 30:
        return f"HIGH{suffix}"
    if n >= 20:
        return f"MEDIUM{suffix}"
    if n >= 10:
        return f"LOW{suffix}"
    return f"VERY LOW SAMPLE{suffix}" if n > 0 else "INSUFFICIENT SAMPLE"


def forward_window_drawdown(close: pd.Series, low: pd.Series, weeks: int) -> pd.Series:
    closes = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(closes), np.nan, dtype=float)
    for index in range(max(0, len(closes) - weeks)):
        if not np.isfinite(closes[index]) or closes[index] == 0:
            continue
        future = lows[index + 1 : index + weeks + 1]
        future = future[np.isfinite(future)]
        if len(future) == weeks:
            result[index] = float(future.min() / closes[index] - 1.0)
    return pd.Series(result, index=close.index)


def calculate_qqq_current_risk_event_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Track QQQ risk-event starts and escalation milestones without future data."""
    dates = pd.to_datetime(frame.get("Date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
    activation = frame.get("CurrentRiskActivation", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    activation_available = (
        frame.get("CurrentRiskActivationAvailable", pd.Series(True, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    daily_status = (
        frame.get("CurrentMarketRiskState", pd.Series("DATA INCOMPLETE", index=frame.index))
        .fillna("DATA INCOMPLETE")
        .astype(str)
    )
    daily_signal = (
        frame.get("CurrentRiskDisplayClass", pd.Series("INACTIVE", index=frame.index))
        .fillna("INACTIVE")
        .astype(str)
    )
    credit = (
        frame.get("CurrentRiskCreditConfirmation", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    severity_by_status = {
        "LOW CONFIRMATION": 1,
        "MODERATE": 2,
        "HIGH RISK": 3,
        "RED FLAG": 4,
    }
    daily_severity = [
        severity_by_status.get(status, 0) if is_active else 0
        for status, is_active in zip(daily_status, activation)
    ]
    columns = [
        "CurrentRiskDailyStatus",
        "CurrentRiskDailySeverity",
        "CurrentRiskNewEvent",
        "CurrentRiskEventMarker",
        "CurrentRiskEventMarkerClass",
        "CurrentRiskEventMarkerReason",
        "CurrentRiskEventId",
        "CurrentRiskEventState",
        "CurrentRiskEventStartDate",
        "CurrentRiskEventLastActivationDate",
        "CurrentRiskEventMaxSeverity",
        "CurrentRiskEventPeakStatus",
        "CurrentRiskEventCreditConfirmed",
        "CurrentRiskEventNoActivationSessions",
    ]
    output: dict[str, list[Any]] = {column: [] for column in columns}
    event_id: str | None = None
    event_start: Any = pd.NaT
    last_activation: Any = pd.NaT
    event_max_severity = 0
    event_credit_confirmed = False
    no_activation_sessions = 0

    def peak_status(severity: int) -> str:
        return {
            0: "NORMAL",
            1: "LOW CONFIRMATION",
            2: "MODERATE",
            3: "HIGH RISK",
            4: "RED FLAG",
        }.get(severity, "NORMAL")

    for position, values in enumerate(
        zip(dates, activation, activation_available, daily_status, daily_signal, daily_severity, credit)
    ):
        date, is_active, inputs_available, status, signal, severity, credit_confirmed = values
        marker = False
        marker_class = "INACTIVE"
        marker_reason = ""
        is_new_event = False
        if event_id is None and is_active:
            event_start = pd.Timestamp(date) if pd.notna(date) else pd.NaT
            event_id = f"QQQ-RISK-{event_start:%Y%m%d}" if pd.notna(event_start) else f"QQQ-RISK-{position}"
            last_activation = event_start
            event_max_severity = int(severity)
            event_credit_confirmed = bool(severity == 4 and credit_confirmed)
            no_activation_sessions = 0
            marker = True
            marker_class = signal
            marker_reason = "EVENT_START"
            is_new_event = True
        elif event_id is not None and is_active:
            no_activation_sessions = 0
            last_activation = pd.Timestamp(date) if pd.notna(date) else last_activation
            if severity > event_max_severity:
                event_max_severity = int(severity)
                marker = True
                marker_class = signal
                marker_reason = "SEVERITY_ESCALATION"
                if severity == 4 and credit_confirmed:
                    event_credit_confirmed = True
            elif severity == 4 and credit_confirmed and not event_credit_confirmed:
                marker = True
                marker_class = "RED FLAG + CREDIT CONFIRMATION"
                marker_reason = "CREDIT_ESCALATION"
                event_credit_confirmed = True
        elif event_id is not None and not inputs_available:
            no_activation_sessions = 0
        elif event_id is not None:
            no_activation_sessions += 1
            if no_activation_sessions >= 5:
                event_id = None
                event_start = pd.NaT
                last_activation = pd.NaT
                event_max_severity = 0
                event_credit_confirmed = False
                no_activation_sessions = 0

        output["CurrentRiskDailyStatus"].append(status)
        output["CurrentRiskDailySeverity"].append(int(severity))
        output["CurrentRiskNewEvent"].append(bool(is_new_event))
        output["CurrentRiskEventMarker"].append(bool(marker))
        output["CurrentRiskEventMarkerClass"].append(marker_class)
        output["CurrentRiskEventMarkerReason"].append(marker_reason)
        output["CurrentRiskEventId"].append(event_id)
        output["CurrentRiskEventState"].append("ACTIVE" if event_id is not None else "CLOSED")
        output["CurrentRiskEventStartDate"].append(event_start)
        output["CurrentRiskEventLastActivationDate"].append(last_activation)
        output["CurrentRiskEventMaxSeverity"].append(int(event_max_severity))
        output["CurrentRiskEventPeakStatus"].append(peak_status(event_max_severity))
        output["CurrentRiskEventCreditConfirmed"].append(bool(event_credit_confirmed))
        output["CurrentRiskEventNoActivationSessions"].append(int(no_activation_sessions))
    return pd.DataFrame(output, index=frame.index)


def qqq_current_risk_new_event(activation: pd.Series, lookback_sessions: int = 5) -> pd.Series:
    active = activation.fillna(False).astype(bool)
    recently_active = (
        active.shift(1, fill_value=False)
        .rolling(lookback_sessions, min_periods=1)
        .max()
        .fillna(False)
        .astype(bool)
    )
    return active & ~recently_active


def qqq_component_columns() -> list[str]:
    return [
        "CurrentRiskDrawdownRisk",
        "CurrentRiskPriceCycleVulnerabilityRisk",
        "CurrentRiskBreadthRisk",
        "CurrentRiskRSIDivergenceRisk",
        "CurrentRiskVolatilityRisk",
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
    if state == "DATA INCOMPLETE":
        return state
    if state != "RED FLAG":
        return state if bool(active) else "INACTIVE"
    if bool(credit):
        return "RED FLAG + CREDIT CONFIRMATION"
    if bool(top):
        return "RED FLAG + TOP CONFIRMATION"
    return "RED FLAG"


def confirmation_label(signal: str) -> str:
    if signal == "RED FLAG + CREDIT CONFIRMATION":
        return "CREDIT CONFIRMATION"
    if signal == "RED FLAG + TOP CONFIRMATION":
        return "TOP CONFIRMATION"
    return "NONE" if signal != "DATA INCOMPLETE" else "DATA INCOMPLETE"


def primary_drivers(row: dict[str, Any], volatility_source: str) -> str:
    pairs = [
        ("Drawdown", row.get("CurrentRiskDrawdownRisk")),
        ("PVC", row.get("CurrentRiskPriceCycleVulnerabilityRisk")),
        ("Breadth", row.get("CurrentRiskBreadthRisk")),
        ("RSI", row.get("CurrentRiskRSIDivergenceRisk")),
        (volatility_source, row.get("CurrentRiskVolatilityRisk")),
        ("IWM/QQQ", row.get("CurrentRiskHighBetaRisk")),
        ("HY", row.get("CurrentRiskHYRisk")),
    ]
    available = [(label, safe_float(value)) for label, value in pairs if np.isfinite(safe_float(value))]
    if not available:
        return "DATA INCOMPLETE"
    return ", ".join(label for label, _ in sorted(available, key=lambda item: item[1], reverse=True)[:3])


def wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).where(avg_loss.ne(0.0), 100.0)


def expanding_percentile(series: pd.Series, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result: list[float] = []
    for index, value in enumerate(values):
        history = values.iloc[: index + 1].dropna()
        result.append(float((history <= value).mean() * 100.0) if len(history) >= min_periods and np.isfinite(safe_float(value)) else np.nan)
    return pd.Series(result, index=series.index, dtype="float64")


def classify_momentum_phase(level: Any, momentum: Any) -> str:
    level_value = safe_float(level)
    momentum_value = safe_float(momentum)
    if not np.isfinite(level_value) or not np.isfinite(momentum_value):
        return "DATA INCOMPLETE"
    if level_value > 0 and momentum_value > 0:
        return "RISK EXPANSION"
    if level_value > 0 and momentum_value < 0:
        return "LATE RISK EXPANSION"
    if level_value < 0 and momentum_value < 0:
        return "RISK CONTRACTION"
    if level_value < 0 and momentum_value > 0:
        return "EARLY RISK RECOVERY"
    return "TRANSITION"


def confirm_state(candidate: pd.Series, required: int = 2) -> pd.Series:
    result: list[str] = []
    active: str | None = None
    run_value: str | None = None
    run_length = 0
    for raw in candidate.astype("object"):
        value = str(raw) if pd.notna(raw) else "DATA INCOMPLETE"
        if value == "DATA INCOMPLETE":
            result.append(active or value)
            continue
        if value == run_value:
            run_length += 1
        else:
            run_value = value
            run_length = 1
        if active is None:
            active = value
        elif value != active and run_length >= required:
            active = value
        result.append(active)
    return pd.Series(result, index=candidate.index)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from financial_fragility import build_financial_fragility_snapshot


@dataclass
class GlobalDashboardSnapshot:
    liquidity: dict[str, Any]
    market: dict[str, Any]
    economy: dict[str, Any]
    rates: dict[str, Any]
    funding: dict[str, Any]
    treasury: dict[str, Any]
    fragility: dict[str, Any]
    cross_cycle: dict[str, Any]
    divergences: pd.DataFrame
    forward_outlook: pd.DataFrame
    freshness: pd.DataFrame
    fields: dict[str, Any]


def _latest(frame: pd.DataFrame | None, date_column: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    data = frame.copy()
    if date_column in data:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column]).sort_values(date_column)
    return data.iloc[-1].to_dict() if not data.empty else {}


def _latest_with_value(frame: pd.DataFrame | None, date_column: str, value_column: str) -> dict[str, Any]:
    """Use the latest dated observation that has the primary metric populated."""
    if frame is None or frame.empty or value_column not in frame.columns:
        return {}
    data = frame.copy()
    if date_column in data:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column]).sort_values(date_column)
    data = data.dropna(subset=[value_column])
    return data.iloc[-1].to_dict() if not data.empty else {}


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _text(value: Any, fallback: str = "UNAVAILABLE") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    result = str(value).strip()
    return result if result and result.lower() not in {"nan", "none"} else fallback


def _confidence(value: Any) -> str:
    text = _text(value, "INSUFFICIENT").upper()
    return "INSUFFICIENT" if text.startswith("INSUFFICIENT") else text


def _liquidity_percentile_status(value: Any) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "n/a"
    if number >= 90:
        return "EXTREME ACCELERATION"
    if number >= 70:
        return "ACCELERATION"
    if number >= 40:
        return "NEUTRAL"
    if number >= 10:
        return "DECELERATION"
    return "EXTREME DECELERATION"


def _liquidity_growth_status(component: str, growth: Any, percentile: Any) -> str:
    """Match the Liquidity Cycle tab's stabilization override for 52W growth."""
    bands = {
        "Global_M2": 1.0,
        "Global_CB_Assets": 1.5,
        "US_Net_Liquidity": 1.0,
    }
    growth_value = _number(growth)
    band = bands.get(component)
    if band is not None and np.isfinite(growth_value) and abs(growth_value * 100.0) <= band:
        return "STABILIZATION"
    return _liquidity_percentile_status(percentile)


def _liquidity_cycle_maturity(date_value: Any) -> float:
    date = _date(date_value)
    if pd.isna(date):
        return np.nan
    anchor = pd.Timestamp("2022-10-01")
    months = (date.year - anchor.year) * 12 + (date.month - anchor.month) + (date.day - 1) / 30.4375
    return months / 65.0 * 100.0


def _date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)


def _outlook_row(outlook: pd.DataFrame | None, horizon: str = "12M") -> dict[str, Any]:
    if outlook is None or outlook.empty or "Horizon" not in outlook:
        return {}
    rows = outlook.loc[outlook["Horizon"].astype(str).eq(horizon)]
    return rows.iloc[-1].to_dict() if not rows.empty else {}


def _risk_drivers(current: dict[str, Any]) -> str:
    existing = _text(current.get("HistoricalCurrentRiskPrimaryDrivers"), "")
    if existing:
        return existing
    channels = {
        "Breadth": current.get("CurrentRiskBreadthRisk", current.get("BreadthRisk")),
        "High Beta": current.get("CurrentRiskHighBetaRisk", current.get("HighBetaRisk")),
        "VIX": current.get("CurrentRiskVIXRisk", current.get("HistoricalCombinedVIXRisk")),
        "RSI Divergence": current.get("CurrentRiskRSIDivergenceRisk", current.get("RSIDivergenceRiskScore")),
        "Drawdown": current.get("CurrentRiskDrawdownRisk", current.get("HistoricalDrawdownRisk")),
        "High Yield": current.get("CurrentRiskHYRisk"),
    }
    ranked = sorted(
        ((name, _number(value)) for name, value in channels.items() if np.isfinite(_number(value))),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(name for name, _ in ranked[:3]) or "UNAVAILABLE"


def _sign(value: Any, domain: str) -> int | None:
    state = _text(value, "").upper()
    positive: dict[str, set[str]] = {
        "liquidity": {"EXPANSION", "BOTTOMING", "ACCELERATING", "IMPROVING", "STRONG", "VERY_STRONG"},
        "market": {"RISK EXPANSION", "LATE RISK EXPANSION", "EARLY RISK RECOVERY", "EXPANSION", "ACCELERATING"},
        "economy": {"IMPROVING", "STRONG EXPANSION", "EARLY RECOVERY"},
        "risk": {"LOW", "NORMAL", "FALLING"},
        "fiscal": {"POSITIVE / ACCELERATING", "FISCAL SUPPORT"},
        "treasury": {"INJECTION", "MILD INJECTION"},
        "conditions": {"EASING", "BROAD EASING", "NORMAL"},
        "funding": {"NORMAL", "FALLING"},
        "inflation": {"FALLING"},
    }
    negative: dict[str, set[str]] = {
        "liquidity": {"CONTRACTION", "PEAKING", "DETERIORATING", "DETERIORATING_FAST", "WEAK", "VERY_WEAK"},
        "market": {"RISK CONTRACTION", "CONTRACTION", "DECELERATING"},
        "economy": {"DETERIORATING", "DETERIORATING CONTRACTION", "LATE / SLOWING EXPANSION"},
        "risk": {"ELEVATED", "HIGH", "ACUTE", "RISING"},
        "fiscal": {"NEGATIVE / DECELERATING", "FISCAL PRESSURE"},
        "treasury": {"DRAIN", "MILD DRAIN"},
        "conditions": {"TIGHTENING", "BROAD TIGHTENING", "HIGH", "ACUTE"},
        "funding": {"TECHNICAL FUNDING PRESSURE", "PERSISTENT FUNDING PRESSURE", "SYSTEMIC FUNDING STRESS", "RISING"},
        "inflation": {"RISING"},
    }
    if state in positive.get(domain, set()) or any(token in state for token in positive.get(domain, set()) if len(token) > 8):
        return 1
    if state in negative.get(domain, set()) or any(token in state for token in negative.get(domain, set()) if len(token) > 8):
        return -1
    return 0 if state else None


def classify_cross_cycle(liquidity_state: Any, market_state: Any, economy_direction: Any) -> str:
    liquidity = _sign(liquidity_state, "liquidity")
    market = _sign(market_state, "market")
    economy = _sign(economy_direction, "economy")
    liquidity_text = _text(liquidity_state, "").upper()
    if liquidity_text == "BOTTOMING" and market != 1:
        return "EARLY LIQUIDITY TURN"
    if liquidity == 1 and market == -1:
        return "LIQUIDITY LEADING"
    if liquidity == 1 and market == 1 and economy == -1:
        return "MARKETS CONFIRMING"
    if liquidity == 1 and market == 1 and economy == 1:
        return "BROAD EXPANSION"
    if liquidity_text == "PEAKING" and market == 1:
        return "LIQUIDITY PEAKING"
    if market == -1 and economy == 1:
        return "MARKET ROLLING OVER"
    if liquidity == -1 and market == -1 and economy == -1:
        return "BROAD CONTRACTION"
    return "DIVERGENT"


def _divergence_state(left: int | None, right: int | None) -> str:
    if left is None or right is None:
        return "UNAVAILABLE"
    if left == right:
        return "ALIGNED"
    if left == 0 or right == 0:
        return "MILD DIVERGENCE"
    return "STRONG DIVERGENCE"


def build_divergences(
    liquidity: dict[str, Any], market: dict[str, Any], economy: dict[str, Any],
    rates: dict[str, Any], funding: dict[str, Any], treasury: dict[str, Any],
) -> pd.DataFrame:
    relationships = [
        ("Liquidity vs Market", _sign(liquidity.get("LiquidityForecastState"), "liquidity"), _sign(market.get("SPXMomentumCyclePhase"), "market"),
         f"Liquidity {_text(liquidity.get('LiquidityForecastState'))}; market {_text(market.get('SPXMomentumCyclePhase'))}."),
        ("Market vs Economy", _sign(market.get("SPXMomentumCyclePhase"), "market"), _sign(economy.get("BusinessCycleDirection"), "economy"),
         f"Market {_text(market.get('SPXMomentumCyclePhase'))}; growth {_text(economy.get('BusinessCycleDirection'))}."),
        ("Structural vs Medium-Term", _sign(market.get("StructuralROCMomentumState"), "market"), _sign(market.get("SPXMomentumCyclePhase"), "market"),
         f"Structural momentum {_text(market.get('StructuralROCMomentumState'))}; medium-term {_text(market.get('SPXMomentumCyclePhase'))}."),
        ("Medium-Term vs Current Risk", _sign(market.get("SPXMomentumCyclePhase"), "market"), _sign(market.get("CurrentMarketRiskState"), "risk"),
         f"Momentum {_text(market.get('SPXMomentumCyclePhase'))}; risk {_text(market.get('CurrentMarketRiskState'))}."),
        ("Fiscal vs Treasury", _sign(treasury.get("FiscalGrowthImpulse"), "fiscal"), _sign(treasury.get("TreasuryLiquidityState"), "treasury"),
         f"Fiscal {_text(treasury.get('FiscalGrowthImpulse'))}; Treasury liquidity {_text(treasury.get('TreasuryLiquidityState'))}."),
        ("Funding vs Financial Conditions", _sign(funding.get("FundingState"), "funding"), _sign(rates.get("CoreFCState"), "conditions"),
         f"Funding {_text(funding.get('FundingState'))}; financial conditions {_text(rates.get('CoreFCState'))}."),
        ("Growth vs Inflation", _sign(economy.get("BusinessCycleDirection"), "economy"), _sign(economy.get("InflationDirection"), "inflation"),
         f"Growth {_text(economy.get('BusinessCycleDirection'))}; inflation {_text(economy.get('InflationDirection'))}."),
    ]
    return pd.DataFrame(
        [
            {
                "Relationship": relationship,
                "State": _divergence_state(left, right),
                "Direction": "SAME" if left == right and left is not None else "OPPOSING" if left and right and left != right else "MIXED",
                "Short Interpretation": interpretation,
            }
            for relationship, left, right, interpretation in relationships
        ]
    )


def format_forward_outlook(asset_outlook: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "Asset", "3M Median", "6M Median", "12M Median", "P(Positive)", "P25", "P75",
        "DD >15%", "DD >25%", "DD >45%", "Independent N", "Avg Similarity", "Coverage",
        "Confidence", "Concentration Warning",
    ]
    if asset_outlook is None or asset_outlook.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for asset in ("SPY", "QQQ", "GLD", "BTC"):
        sample = asset_outlook.loc[asset_outlook["Asset"].astype(str).eq(asset)]
        if sample.empty:
            continue
        by_horizon = {str(row["Horizon"]): row for _, row in sample.iterrows()}
        twelve = by_horizon.get("12M", next(iter(by_horizon.values())))
        rows.append(
            {
                "Asset": asset,
                "3M Median": by_horizon.get("3M", {}).get("Median Forward Return", np.nan),
                "6M Median": by_horizon.get("6M", {}).get("Median Forward Return", np.nan),
                "12M Median": by_horizon.get("12M", {}).get("Median Forward Return", np.nan),
                "P(Positive)": twelve.get("Positive Return Probability", np.nan),
                "P25": twelve.get("P25 Return", np.nan),
                "P75": twelve.get("P75 Return", np.nan),
                "DD >15%": twelve.get("Risk of >15% Drawdown", np.nan),
                "DD >25%": twelve.get("Risk of >25% Drawdown", np.nan),
                "DD >45%": twelve.get("Risk of >45% Drawdown", np.nan),
                "Independent N": twelve.get("Independent Analog N", 0),
                "Avg Similarity": twelve.get("Average Similarity", np.nan),
                "Coverage": twelve.get("Average Coverage", np.nan),
                "Confidence": _confidence(twelve.get("Confidence")),
                "Concentration Warning": twelve.get("Concentration Warning", "OK"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _freshness_row(module: str, value: Any, cadence_days: int, status: Any = "") -> dict[str, Any]:
    as_of = _date(value)
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - as_of.normalize()).days if pd.notna(as_of) else np.nan
    explicit = _text(status, "")
    if pd.isna(as_of):
        state = "UNAVAILABLE"
    elif explicit and any(token in explicit.upper() for token in ("ERROR", "UNAVAILABLE", "INCOMPLETE")):
        state = "PARTIAL DATA"
    elif age > cadence_days:
        state = "STALE"
    else:
        state = "CURRENT"
    return {"Module": module, "As Of": as_of, "Age Days": age, "Status": state}


def build_global_dashboard_snapshot(
    *,
    liquidity_regime: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    forecast_status: dict[str, Any],
    market_snapshot: Any,
    business_snapshot: Any,
    macro_snapshot: Any,
    rates_snapshot: Any,
    funding_snapshot: Any,
    treasury_snapshot: Any,
    transition_snapshot: dict[str, Any] | None = None,
) -> GlobalDashboardSnapshot:
    liquidity_latest = _latest_with_value(liquidity_regime, "date", "global_liquidity_score")
    forecast_latest = _latest(forecast_frame, "Date")
    market_current = dict(getattr(market_snapshot, "current", {}) or {})
    market_12m = _outlook_row(getattr(market_snapshot, "outlook", pd.DataFrame()))
    business_current = dict(getattr(business_snapshot, "current", {}) or {})
    macro_current = dict(getattr(macro_snapshot, "current", {}) or {})
    rates_latest = _latest(getattr(rates_snapshot, "history", pd.DataFrame()), "Date")
    funding_latest = _latest(getattr(funding_snapshot, "daily", pd.DataFrame()), "Date")
    treasury_latest = _latest(getattr(treasury_snapshot, "weekly", pd.DataFrame()), "Date")
    treasury_monthly_latest = _latest(getattr(treasury_snapshot, "monthly", pd.DataFrame()), "Date")
    transition = transition_snapshot or {}

    liquidity = {
        "GlobalLiquidityState": _text(liquidity_latest.get("final_regime_label", liquidity_latest.get("impulse_state"))),
        "GlobalLiquidityDirection": _text(liquidity_latest.get("direction_13w_state")),
        "GlobalLiquidityDirection26W": _text(liquidity_latest.get("direction_26w_state")),
        "GlobalLiquidityDirection52W": _text(liquidity_latest.get("direction_52w_state")),
        "GlobalLiquidityScore": _number(liquidity_latest.get("global_liquidity_score")),
        "LiquidityCyclePhase": _text(liquidity_latest.get("long_cycle_phase")),
        "GlobalM2Value": _number(liquidity_latest.get("global_m2_usd_bn")),
        "GlobalCBAssetsValue": _number(liquidity_latest.get("global_cb_assets_usd_bn")),
        "USNetLiquidityValue": _number(liquidity_latest.get("us_net_liquidity_usd_bn")),
        "M2GrowthPercentileStatus": _liquidity_growth_status(
            "Global_M2", liquidity_latest.get("m2_growth"), liquidity_latest.get("m2_growth_pctl")
        ),
        "M2FastImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("m2_fast_impulse_pctl")),
        "M2MediumImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("m2_medium_impulse_pctl")),
        "M2SlowImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("m2_slow_impulse_pctl")),
        "CBGrowthPercentileStatus": _liquidity_growth_status(
            "Global_CB_Assets", liquidity_latest.get("cb_growth"), liquidity_latest.get("cb_growth_pctl")
        ),
        "CBFastImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("cb_fast_impulse_pctl")),
        "CBMediumImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("cb_medium_impulse_pctl")),
        "CBSlowImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("cb_slow_impulse_pctl")),
        "USNLGrowthPercentileStatus": _liquidity_growth_status(
            "US_Net_Liquidity", liquidity_latest.get("usnl_growth"), liquidity_latest.get("usnl_growth_pctl")
        ),
        "USNLFastImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("usnl_fast_impulse_pctl")),
        "USNLMediumImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("usnl_medium_impulse_pctl")),
        "USNLSlowImpulsePercentileStatus": _liquidity_percentile_status(liquidity_latest.get("usnl_slow_impulse_pctl")),
        "LiquidityCycleMaturity": _liquidity_cycle_maturity(liquidity_latest.get("date")),
        "LiquidityForecastState": _text(forecast_latest.get("LiquidityForecastState")),
        "LiquidityForecastSignal": _text(forecast_latest.get("LiquidityForwardSignal")),
        "LiquidityPressure": _number(forecast_latest.get("LiquidityPressureScore")),
        "PolicyResponse": _number(forecast_latest.get("PolicyResponseScore")),
        "LiquidityBreadthState": _text(forecast_latest.get("BreadthParticipationState")),
        "RiskOnState": _text(forecast_latest.get("RiskOnState")),
        "RiskReductionState": _text(forecast_latest.get("RiskReductionWarning")),
        "FastTransitionRisk": _number(transition.get("Fast_Transition_Risk")),
        "MacroTransitionRisk": _number(transition.get("Macro_Transition_Risk")),
        "NearTermTreasuryRefinancing": _number(treasury_monthly_latest.get("near_term_refinancing_pressure")),
        "LiquidityDataStatus": _text(liquidity_latest.get("data_status")),
        "AsOf": _date(forecast_latest.get("Date", liquidity_latest.get("date"))),
    }
    market = {
        "StructuralMarketCyclePhase": _text(market_current.get("StructuralMarketCyclePhase")),
        "StructuralExtensionZone": _text(market_current.get("StructuralExtensionZone")),
        "StructuralSMA_ROC_Phase": _text(market_current.get("Structural_SMA_ROC_Phase")),
        "StructuralExtensionPercentile": _number(market_current.get("StructuralExtensionPercentile")),
        "StructuralROCMomentumState": _text(market_current.get("StructuralROCMomentumState")),
        "SPX_ROC36M_3MMA": _number(market_current.get("SPX_ROC36M_3MMA")),
        "StructuralROCMomentum_12M": _number(market_current.get("StructuralROCMomentum_12M")),
        "StructuralMaturityPct": _number(market_current.get("StructuralMaturityPct")),
        "SPXMomentumCyclePhase": _text(market_current.get("MomentumCyclePhase")),
        "SMA200WExtensionPhase": _text(market_current.get("SMA200WZone")),
        "SMA200WExtensionPercentile": _number(market_current.get("SMA200WExtensionPercentile")),
        "MediumTerm_SMA_ROC_Phase": _text(market_current.get("MediumTerm_SMA_ROC_Phase")),
        "SPX_ROC12M_3MMA": _number(market_current.get("SPX_ROC12M_3MMA")),
        "SPX_ROC_Momentum_3M": _number(market_current.get("SPX_ROC_Momentum_3M")),
        "PrimaryCycleMonthsSinceTrough": _number(market_current.get("PrimaryCycleMonthsSinceTrough")),
        "LongCycleMaturityPct": _number(market_current.get("LongCycleMaturityPct")),
        "PrimaryCycleMaturityPct": _number(market_current.get("PrimaryCycleMaturityPct")),
        "LongCycleDirection": _text(market_current.get("LongCycleDirection")),
        "CurrentMarketRiskState": _text(market_current.get("CurrentMarketRiskState")),
        "CurrentMarketRiskDirection": _text(market_current.get("CurrentMarketRiskDirection")),
        "CurrentRiskBreadthRisk": _number(market_current.get("CurrentRiskBreadthRisk")),
        "CurrentRiskRSIDivergenceRisk": _number(market_current.get("CurrentRiskRSIDivergenceRisk")),
        "CurrentRiskVIXRisk": _number(market_current.get("CurrentRiskVIXRisk")),
        "CurrentRiskHighBetaRisk": _number(market_current.get("CurrentRiskHighBetaRisk")),
        "CurrentRiskHYRisk": _number(market_current.get("CurrentRiskHYRisk")),
        "CurrentRiskCreditConfirmation": _number(market_current.get("CurrentRiskCreditConfirmation")),
        "CurrentRiskSignalClass": _text(market_current.get("CurrentRiskSignalClass")),
        "CurrentMarketRiskDrivers": _risk_drivers(market_current),
        "PerformanceROC1M": _number(market_current.get("PerformanceROC1M")),
        "PerformanceROC3M": _number(market_current.get("PerformanceROC3M")),
        "PerformanceROC6M": _number(market_current.get("PerformanceROC6M")),
        "PerformanceROC12M": _number(market_current.get("PerformanceROC12M")),
        "HistoricalOutlook_12M_Median": _number(market_12m.get("Median Forward Return")),
        "HistoricalOutlook_12M_PPositive": _number(market_12m.get("Positive Return Probability")),
        "HistoricalOutlook_DD15": _number(market_12m.get("Risk of >15% Drawdown")),
        "HistoricalOutlook_IndependentN": int(_number(market_12m.get("Independent Analog N"))) if np.isfinite(_number(market_12m.get("Independent Analog N"))) else 0,
        "HistoricalOutlook_AverageSimilarity": _number(market_12m.get("Average Similarity")),
        "HistoricalOutlook_Coverage": _number(market_12m.get("Average Coverage")),
        "HistoricalOutlook_Confidence": _confidence(market_12m.get("Confidence")),
        "HistoricalOutlook_Concentration": _text(market_12m.get("Concentration Warning"), "OK"),
        "AsOf": _date(market_current.get("CurrentRiskAsOfDate", market_current.get("Date"))),
    }
    economy = {
        "BusinessCyclePhase": _text(business_current.get("BusinessCycleState")),
        "BusinessCycleDirection": _text(business_current.get("BusinessCycleDirection")),
        "BusinessCycleTransitionZone": business_current.get("BusinessCycleTransitionZone", np.nan),
        "BusinessCycleConfidence": _text(business_current.get("BusinessCycleConfidence")),
        "LaborCycleState": _text(business_current.get("LaborCycleState")),
        "ProductivityExpansionFlag": business_current.get("ProductivityExpansionFlag", np.nan),
        "InflationDirection": _text(business_current.get("InflationState")),
        "InflationLeadingScore": _number(business_current.get("InflationDirectionScore")),
        "RealizedInflationConfirmation": _text(business_current.get("InflationConfirmationStatus")),
        "InflationTransitionZone": business_current.get("InflationTransitionZone", np.nan),
        "EconomyRegime": _text(business_current.get("EconomyRegime")),
        "GrowthSurpriseState": _text(macro_current.get("GrowthState")),
        "InflationSurpriseState": _text(macro_current.get("CPISurpriseState")),
        "GrowthSurpriseScore": _number(macro_current.get("GrowthSurpriseScore")),
        "CPIContribution": _number(macro_current.get("CPIContribution")),
        "TurningSignal": _text(macro_current.get("TurningSignal")),
        "AsOf": _date(business_current.get("date")),
    }
    rates = {
        "RatesPressure": _number(rates_latest.get("RatesPressureScore")),
        "RatesDirection": _text(rates_latest.get("RatesDirection")),
        "CoreFCLevel": _number(rates_latest.get("FinancialConditionsLevel")),
        "CoreFCState": _text(rates_latest.get("FinancialConditionsDirection")),
        "CoreFCStress": _text(rates_latest.get("FinancialConditionsStressLevel")),
        "CreditState": _number(rates_latest.get("CreditLevel")),
        "CreditDirection": _text(rates_latest.get("CreditDirection")),
        "YieldCurveState": _text(rates_latest.get("YieldCurveRegime_26W")),
        "Regime": _text(rates_latest.get("RatesFinancialConditionsRegime")),
        "DXY": _number(rates_latest.get("DXY_Level")),
        "MOVE": _number(rates_latest.get("MOVE_Level")),
        "NFCI": _number(rates_latest.get("NFCILevel")),
        "ANFCI": _number(rates_latest.get("ANFCILevel")),
        "Confirmation": _text(rates_latest.get("FCConfirmationStatus")),
        "AsOf": _date(rates_latest.get("Date")),
    }
    funding = {
        "FundingState": _text(funding_latest.get("FundingState")),
        "FundingDirection": _text(funding_latest.get("FundingDirection")),
        "MoneyMarketStress": _number(funding_latest.get("MoneyMarketStress")),
        "ReservePressure": _number(funding_latest.get("ReservePressure")),
        "FundingCore": _number(funding_latest.get("FundingCore")),
        "CollateralStress": _number(funding_latest.get("CollateralStress")),
        "TechnicalFundingFlag": funding_latest.get("TechnicalFundingFlag", np.nan),
        "PersistentFundingFlag": funding_latest.get("PersistentFundingFlag", np.nan),
        "ReserveVulnerabilityWatch": funding_latest.get("ReserveVulnerabilityWatch", np.nan),
        "PrimaryDriver": _text(funding_latest.get("PrimaryDriver")),
        "DataCoverage": _text(funding_latest.get("DataCoverage")),
        "AsOf": _date(funding_latest.get("Date")),
    }
    treasury = {
        "TreasuryLiquidityState": _text(treasury_latest.get("TreasuryLiquidityState")),
        "TreasuryLiquidityImpulse": _number(treasury_latest.get("TreasuryLiquidityImpulse")),
        "TreasuryLiquidity4W": _number(treasury_latest.get("FastLiquidityImpulse")),
        "TreasuryLiquidity13W": _number(treasury_latest.get("MediumLiquidityImpulse")),
        "FedImpulse4W": _number(treasury_latest.get("FedImpulse_4W")),
        "TGAImpulse4W": _number(treasury_latest.get("TGAImpulse_4W")),
        "RRPImpulse4W": _number(treasury_latest.get("RRPImpulse_4W")),
        "FiscalGrowthImpulse": _text(treasury_latest.get("FiscalImpulseState")),
        "FiscalImpulse": _number(treasury_latest.get("FiscalImpulse")),
        "DeficitGDPPercentile": _number(treasury_latest.get("FiscalStancePercentile")),
        "FinancingPressure": _text(treasury_latest.get("TreasuryFinancingPressure")),
        "BillFinancingShare": _number(treasury_latest.get("BillFinancingShare")),
        "DurationSupplyProxy": _number(treasury_latest.get("DurationSupplyPercentile")),
        "AbsorptionState": _text(treasury_latest.get("AbsorptionCapacity")),
        "TotalIssuance": _number(treasury_latest.get("TotalNetIssuance4Q")),
        "PolicyMix": _text(treasury_latest.get("PolicyMix")),
        "AsOf": _date(treasury_latest.get("Date")),
    }
    try:
        fragility_snapshot = build_financial_fragility_snapshot(
            liquidity_regime=liquidity_regime,
            forecast_frame=forecast_frame,
            market_snapshot=market_snapshot,
            business_snapshot=business_snapshot,
            rates_snapshot=rates_snapshot,
            funding_snapshot=funding_snapshot,
            treasury_snapshot=treasury_snapshot,
            transition_snapshot=transition,
        )
        fragility_current = fragility_snapshot.current
        fragility = {
            "FinancialFragilityState": _text(fragility_current.get("GeneralRegime")),
            "FinancialFragilityDirection": _text(fragility_current.get("MacroPressureDirection")),
            "MacroPressure": _number(fragility_current.get("MacroPressure")),
            "Vulnerability": _number(fragility_current.get("MarketVulnerability")),
            "MarketStress": _number(fragility_current.get("MarketVolatilityStress")),
            "FundingStress": _number(fragility_current.get("FundingStress")),
            "TransitionRisk": _number(fragility_current.get("TransitionRisk")),
            "FragilityDrivers": _text(fragility_current.get("PrimaryDrivers")),
            "FragilityStabilizers": _text(fragility_current.get("PrimaryStabilizers")),
            "AsOf": _date(fragility_current.get("DashboardAsOf")),
        }
    except Exception:
        fragility = {
            "FinancialFragilityState": "UNAVAILABLE",
            "FinancialFragilityDirection": "UNAVAILABLE",
            "MacroPressure": np.nan,
            "Vulnerability": np.nan,
            "MarketStress": np.nan,
            "FundingStress": np.nan,
            "TransitionRisk": np.nan,
            "FragilityDrivers": "Production Financial Fragility module is unavailable.",
            "FragilityStabilizers": "UNAVAILABLE",
            "AsOf": pd.NaT,
        }
    cross_cycle = {
        "CrossCycleState": classify_cross_cycle(
            liquidity["LiquidityForecastState"], market["SPXMomentumCyclePhase"], economy["BusinessCycleDirection"]
        ),
        "LiquidityState": liquidity["LiquidityForecastState"],
        "MarketState": market["SPXMomentumCyclePhase"],
        "BusinessState": economy["BusinessCyclePhase"],
        "LiquidityToMarketLag": "UNAVAILABLE",
        "HistoricalLiquidityToMarketLag": "UNAVAILABLE",
        "MarketToBusinessLag": "UNAVAILABLE",
        "HistoricalMarketToBusinessLag": "UNAVAILABLE",
    }
    forward_outlook = format_forward_outlook(getattr(market_snapshot, "asset_outlook", pd.DataFrame()))

    rates_status = getattr(rates_snapshot, "status", {}) or {}
    funding_status = getattr(funding_snapshot, "status", {}) or {}
    treasury_status = getattr(treasury_snapshot, "status", {}) or {}
    freshness = pd.DataFrame(
        [
            _freshness_row("Global Liquidity Forecast", forecast_status.get("DataAsOf", liquidity.get("AsOf")), 21, forecast_status.get("LatestRefreshError", "")),
            _freshness_row("Market Cycle", market.get("AsOf"), 10),
            _freshness_row("Business Cycle / Inflation", economy.get("AsOf"), 70),
            _freshness_row("Rates & Financial Conditions", rates_status.get("DataAsOf", rates.get("AsOf")), 21),
            _freshness_row("Funding Conditions", funding_status.get("DataAsOf", funding.get("AsOf")), 10),
            _freshness_row("Treasury & Fiscal", treasury.get("AsOf"), 45, treasury_status.get("SourceStatus", "")),
            _freshness_row("Financial Fragility", pd.NaT, 1, "UNAVAILABLE"),
        ]
    )
    available_dates = freshness.loc[freshness["As Of"].notna(), "As Of"]
    fields = {
        **liquidity,
        **market,
        **economy,
        **rates,
        **funding,
        **treasury,
        **fragility,
        **cross_cycle,
        "DashboardAsOf": available_dates.min() if not available_dates.empty else pd.NaT,
    }
    for _, row in forward_outlook.iterrows():
        asset = str(row.get("Asset", "")).upper()
        if not asset:
            continue
        for horizon in ("3M", "6M", "12M"):
            fields[f"{asset}_{horizon}_Median"] = row.get(f"{horizon} Median", np.nan)
        fields[f"{asset}_PPositive"] = row.get("P(Positive)", np.nan)
        fields[f"{asset}_P25"] = row.get("P25", np.nan)
        fields[f"{asset}_P75"] = row.get("P75", np.nan)
        fields[f"{asset}_DD15"] = row.get("DD >15%", np.nan)
        fields[f"{asset}_DD25"] = row.get("DD >25%", np.nan)
        fields[f"{asset}_DD45"] = row.get("DD >45%", np.nan)
        fields[f"{asset}_IndependentN"] = row.get("Independent N", 0)
        fields[f"{asset}_AverageSimilarity"] = row.get("Avg Similarity", np.nan)
        fields[f"{asset}_Coverage"] = row.get("Coverage", np.nan)
        fields[f"{asset}_Confidence"] = row.get("Confidence", "INSUFFICIENT")
    return GlobalDashboardSnapshot(
        liquidity=liquidity,
        market=market,
        economy=economy,
        rates=rates,
        funding=funding,
        treasury=treasury,
        fragility=fragility,
        cross_cycle=cross_cycle,
        divergences=build_divergences(liquidity, market, economy, rates, funding, treasury),
        forward_outlook=forward_outlook,
        freshness=freshness,
        fields=fields,
    )

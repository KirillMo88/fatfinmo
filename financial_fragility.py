from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "FINANCIAL_FRAGILITY_V1"
PIT_WINDOW_WEEKS = 156
PIT_MIN_PERIODS = 52

SLOW_STATES = ("NORMAL", "WATCH", "HIGH", "EXTREME")
STRESS_STATES = SLOW_STATES
TRANSITION_STATES = ("STABLE", "MILD TRANSITION", "ELEVATED", "HIGH", "EXTREME TRANSITION")


@dataclass
class FinancialFragilitySnapshot:
    history: pd.DataFrame
    current: dict[str, Any]
    data_quality: pd.DataFrame
    as_of: pd.Timestamp | pd.NaT


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


def _date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    parsed = pd.Timestamp(parsed)
    return parsed.tz_localize(None) if parsed.tzinfo is not None else parsed


def _source_frame(source: Any, names: tuple[str, ...]) -> pd.DataFrame:
    for name in names:
        frame = getattr(source, name, None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame.copy()
    return pd.DataFrame()


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ("Date", "date", "ObservationDate"):
        if column in frame.columns:
            return column
    return None


def _normalise(frame: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    date_column = date_column or _date_column(out)
    if date_column is None and isinstance(out.index, pd.DatetimeIndex):
        out["Date"] = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
        date_column = "Date"
    if date_column is None:
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out[date_column], errors="coerce").dt.tz_localize(None).dt.normalize()
    # The source may have a DatetimeIndex also named "Date". Keep the
    # normalized observation date as the sole Date label before sorting.
    out = out.reset_index(drop=True).dropna(subset=["Date"]).sort_values("Date")
    return out.drop_duplicates("Date", keep="last").reset_index(drop=True)


def _numeric(source: pd.DataFrame, candidates: tuple[str, ...], index: pd.Index) -> pd.Series:
    for column in candidates:
        if column in source.columns:
            return pd.to_numeric(source[column], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index, dtype=float)


def _value(source: pd.DataFrame, candidates: tuple[str, ...], index: pd.Index) -> pd.Series:
    for column in candidates:
        if column in source.columns:
            return source[column].reindex(index)
    return pd.Series(np.nan, index=index, dtype=object)


def _asof(source: pd.DataFrame, dates: pd.DatetimeIndex, candidates: tuple[str, ...], numeric: bool = True) -> pd.Series:
    if source.empty:
        return pd.Series(np.nan, index=dates)
    frame = source.copy().set_index("Date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    column = next((name for name in candidates if name in frame.columns), None)
    if column is None:
        return pd.Series(np.nan, index=dates)
    values = frame[column]
    if numeric:
        values = pd.to_numeric(values, errors="coerce")
    return values.reindex(values.index.union(dates)).sort_index().ffill().reindex(dates)


def _current_percentile(history: pd.Series, value: Any) -> float:
    number = _number(value)
    sample = pd.to_numeric(history, errors="coerce").dropna().tail(PIT_WINDOW_WEEKS - 1)
    if not np.isfinite(number) or len(sample) < PIT_MIN_PERIODS - 1:
        return np.nan
    return float((sample.le(number).sum() + 1) / (len(sample) + 1) * 100.0)


def _weighted_current(values: dict[str, Any], weights: dict[str, float]) -> float:
    valid = [(_number(values.get(name)), weight) for name, weight in weights.items()]
    valid = [(value, weight) for value, weight in valid if np.isfinite(value)]
    if not valid:
        return np.nan
    return float(np.clip(sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid), 0.0, 100.0))


def _latest_value(frame: pd.DataFrame, column: str) -> tuple[float, pd.Timestamp | pd.NaT]:
    if frame.empty or column not in frame.columns:
        return np.nan, pd.NaT
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return np.nan, pd.NaT
    index = values.index[valid][-1]
    row = frame.loc[index]
    return _number(row[column]), _date(row.get("Date", index))


def _overlay_current_stress(
    current: dict[str, Any],
    history: pd.DataFrame,
    market_daily: pd.DataFrame,
    funding_daily: pd.DataFrame,
    rates_history: pd.DataFrame,
    market_current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the latest source observations for current cards without changing the weekly history."""
    result = dict(current)
    # Current Risk is already calculated by Market Cycle and exposed as its
    # snapshot.current payload. Preserve those daily statuses for the separate
    # Financial Fragility card instead of trying to infer them from weekly data.
    if isinstance(market_current, dict):
        for key, value in market_current.items():
            if key.startswith(("CurrentRisk", "CurrentMarketRisk")) or key == "ActiveStressChannels":
                result[key] = value

    market_fields = {
        "VIX": "VIX",
        "VIX3M": "VIX3M",
        "RealizedVolatility20D": "RealizedVol20D",
        "HY_OAS": "HY_OAS",
    }
    market_dates = []
    for output, source_column in market_fields.items():
        value, date = _latest_value(market_daily, source_column)
        if np.isfinite(value):
            result[output] = value
        if pd.notna(date):
            market_dates.append(date)

    if not np.isfinite(_number(result.get("HY_OAS"))):
        value, date = _latest_value(rates_history, "BAMLH0A0HYM2")
        if np.isfinite(value):
            result["HY_OAS"] = value
            result["CreditDataAsOf"] = date
    credit_value, credit_date = _latest_value(market_daily, "HY_OAS")
    if np.isfinite(credit_value):
        result["CreditDataAsOf"] = credit_date
    elif pd.isna(result.get("CreditDataAsOf")):
        result["CreditDataAsOf"] = _latest_value(rates_history, "BAMLH0A0HYM2")[1]

    result["MarketDataAsOf"] = max(market_dates) if market_dates else pd.NaT
    result["HYOAS_3Y_Percentile"] = _current_percentile(history.get("HY_OAS", pd.Series(dtype=float)), result.get("HY_OAS"))
    result["CreditStress"] = result["HYOAS_3Y_Percentile"]
    result["CreditStressState"] = _risk_state(result["CreditStress"])

    vix_values = pd.to_numeric(market_daily.get("VIX", pd.Series(dtype=float)), errors="coerce").dropna()
    vix_change = float(vix_values.iloc[-1] - vix_values.iloc[-21]) if len(vix_values) >= 21 else np.nan
    result["VIX20DChange"] = vix_change
    vix_value = _number(result.get("VIX"))
    vix3m_value = _number(result.get("VIX3M"))
    ratio = vix_value / vix3m_value if np.isfinite(vix_value) and np.isfinite(vix3m_value) and vix3m_value != 0 else np.nan
    result["VIX_VIX3M_Ratio"] = ratio
    result["VIXLevelRisk"] = np.select([vix_value <= 15, vix_value <= 20, vix_value <= 25, vix_value <= 35], [10.0, 30.0, 55.0, 80.0], default=100.0) if np.isfinite(vix_value) else np.nan
    result["VIXMomentumRisk"] = _current_percentile(history.get("VIX20DChange", pd.Series(dtype=float)), vix_change)
    result["VIXTermStructureRisk"] = np.select([ratio < 0.85, ratio < 0.95, ratio < 1.0, ratio < 1.10], [10.0, 30.0, 55.0, 80.0], default=100.0) if np.isfinite(ratio) else np.nan
    result["RealizedVolRisk"] = _current_percentile(history.get("RealizedVolatility20D", pd.Series(dtype=float)), result.get("RealizedVolatility20D"))
    result["MarketVolatilityStress"] = _weighted_current(
        {"vix": result["VIXLevelRisk"], "momentum": result["VIXMomentumRisk"], "term": result["VIXTermStructureRisk"], "realized": result["RealizedVolRisk"]},
        {"vix": 0.30, "momentum": 0.30, "term": 0.25, "realized": 0.15},
    )
    result["MarketVolatilityStressState"] = _risk_state(result["MarketVolatilityStress"], volatility=True)

    funding_latest = funding_daily.iloc[-1] if not funding_daily.empty else pd.Series(dtype=object)
    for column in ("FundingCore", "MoneyMarketStress", "FundingState", "PersistentFundingFlag", "ReservePressure", "CollateralStress", "MOVE", "MOVE_Z"):
        if column in funding_latest and pd.notna(funding_latest.get(column)):
            result[column] = funding_latest.get(column)
    result["FundingDataAsOf"] = _date(funding_latest.get("Date")) if not funding_latest.empty else pd.NaT
    move_history = pd.to_numeric(funding_daily.get("MOVE", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(move_history) >= 2:
        result["MOVEMean"] = float(move_history.mean())
        result["MOVEStd"] = float(move_history.std(ddof=0))
        if result["MOVEStd"] > 0 and np.isfinite(_number(result.get("MOVE"))):
            result["MOVE_Z"] = float((float(result["MOVE"]) - result["MOVEMean"]) / result["MOVEStd"])
    funding_base = 25.0 * _number(result.get("FundingCore"))
    funding_floor = {
        "NORMAL": 0.0,
        "TECHNICAL FUNDING PRESSURE": 25.0,
        "TREASURY VOLATILITY": 30.0,
        "PERSISTENT FUNDING PRESSURE": 65.0,
        "SYSTEMIC FUNDING STRESS": 90.0,
    }.get(_text(result.get("FundingState"), "").upper(), np.nan)
    funding_components = [value for value in (funding_base, funding_floor) if np.isfinite(value)]
    result["FundingStress"] = float(np.clip(max(funding_components), 0.0, 100.0)) if funding_components else np.nan
    result["FundingStressState"] = _risk_state(result["FundingStress"])
    return result


def pit_percentile(values: pd.Series, window: int = PIT_WINDOW_WEEKS, minimum: int = PIT_MIN_PERIODS) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def rank_last(sample: np.ndarray) -> float:
        valid = sample[np.isfinite(sample)]
        if len(valid) < minimum or not np.isfinite(sample[-1]):
            return np.nan
        return float((valid <= sample[-1]).sum() / len(valid) * 100.0)

    return numeric.rolling(window, min_periods=minimum).apply(rank_last, raw=True)


def _risk_state(value: Any, *, volatility: bool = False) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "UNAVAILABLE"
    if volatility:
        if number < 35:
            return "NORMAL"
        if number < 70:
            return "WATCH"
        if number < 85:
            return "HIGH"
        return "EXTREME"
    if number < 35:
        return "NORMAL"
    if number < 60:
        return "WATCH"
    if number < 80:
        return "HIGH"
    return "EXTREME"


def _rates_pressure_status(value: Any) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "UNAVAILABLE"
    if number <= 25:
        return "LOW"
    if number <= 50:
        return "ELEVATED"
    if number <= 75:
        return "HIGH"
    return "EXTREME"


def _move_status(value: Any) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "UNAVAILABLE"
    if number < 50:
        return "LOW"
    if number <= 80:
        return "NORMAL"
    if number <= 110:
        return "ELEVATED"
    if number <= 141:
        return "HIGH"
    return "EXTREME"


def _transition_state(value: Any) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "UNAVAILABLE"
    if number < 30:
        return "STABLE"
    if number < 50:
        return "MILD TRANSITION"
    if number < 70:
        return "ELEVATED"
    if number < 85:
        return "HIGH"
    return "EXTREME TRANSITION"


def _weighted(values: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=next(iter(values.values())).index)
    denominator = pd.Series(0.0, index=numerator.index)
    for name, series in values.items():
        valid = pd.to_numeric(series, errors="coerce").notna()
        numerator = numerator.add(pd.to_numeric(series, errors="coerce").fillna(0.0) * weights[name], fill_value=0.0)
        denominator = denominator.add(valid.astype(float) * weights[name], fill_value=0.0)
    return numerator.div(denominator.replace(0, np.nan)).clip(0.0, 100.0)


def _direction_risk(series: pd.Series, score: pd.Series | None = None) -> pd.Series:
    if score is not None and pd.to_numeric(score, errors="coerce").notna().any():
        percentile = pit_percentile(score)
        return percentile.where(percentile.notna(), 50.0)
    text = series.astype(str).str.upper()
    return pd.Series(
        np.select([text.str.contains("TIGHTEN|RISING|DRAIN|CONTRACT", regex=True), text.str.contains("EASING|FALLING|EXPAND|INJECT", regex=True)], [80.0, 20.0], default=50.0),
        index=series.index,
    ).where(series.notna())


def _state_start_and_duration(states: pd.Series, dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    changed = states.ne(states.shift())
    starts = dates.where(changed).ffill()
    duration = ((dates - starts).dt.days // 7 + 1).where(starts.notna())
    return starts, duration


def _state_sign(value: Any, domain: str) -> int:
    text = _text(value, "").upper()
    if domain == "liquidity":
        if any(token in text for token in ("EXPAND", "BOTTOM", "IMPROV")):
            return 1
        if any(token in text for token in ("CONTRACT", "PEAK", "DETERIOR", "WEAK")):
            return -1
    if domain == "market":
        if any(token in text for token in ("RISK EXPANSION", "RECOVERY", "EXPANSION")):
            return 1
        if any(token in text for token in ("CONTRACTION", "LATE RISK")):
            return -1
    if domain == "business":
        if any(token in text for token in ("EXPANSION", "RECOVERY", "IMPROV")):
            return 1
        if any(token in text for token in ("CONTRACTION", "SLOW", "DETERIOR")):
            return -1
    return 0


def _safe_change(series: pd.Series, periods: int = 1) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.diff(periods).abs()


def _top_names(row: pd.Series, mapping: dict[str, str], limit: int = 3) -> str:
    values = []
    for column, label in mapping.items():
        value = _number(row.get(column))
        if np.isfinite(value):
            values.append((value, label))
    values.sort(reverse=True)
    return ", ".join(label for _, label in values[:limit]) or "UNAVAILABLE"


def _regime_row(row: pd.Series) -> str:
    credit = _number(row.get("CreditStress"))
    funding = _number(row.get("FundingStress"))
    volatility = _number(row.get("MarketVolatilityStress"))
    macro = _number(row.get("MacroPressure"))
    vulnerability = _number(row.get("MarketVulnerability"))
    transition = _number(row.get("TransitionRisk"))
    persistent = bool(row.get("PersistentFundingFlag"))
    funding_state = _text(row.get("FundingState"), "")
    valid_stress = [x for x in (credit, funding, volatility) if np.isfinite(x)]
    if len(valid_stress) < 2:
        return "UNAVAILABLE"
    if sum(value >= 85 for value in valid_stress) >= 2:
        return "EXTREME"
    if funding_state == "SYSTEMIC FUNDING STRESS" and ((np.isfinite(credit) and credit >= 70) or (np.isfinite(volatility) and volatility >= 70)):
        return "EXTREME"
    if np.isfinite(volatility) and volatility >= 90 and ((np.isfinite(credit) and credit >= 70) or (np.isfinite(funding) and funding >= 70)):
        return "EXTREME"
    if len(valid_stress) == 3 and all(value >= 70 for value in valid_stress):
        return "EXTREME"
    if np.isfinite(volatility) and volatility >= 70:
        return "HIGH"
    if np.isfinite(credit) and credit >= 85 or np.isfinite(funding) and funding >= 85:
        return "HIGH"
    if np.isfinite(credit) and np.isfinite(funding) and credit >= 60 and funding >= 60:
        return "HIGH"
    if np.isfinite(credit) and np.isfinite(volatility) and credit >= 60 and volatility >= 50:
        return "HIGH"
    if np.isfinite(funding) and np.isfinite(volatility) and funding >= 60 and volatility >= 50:
        return "HIGH"
    if any(value >= 70 for value in valid_stress) and ((np.isfinite(macro) and macro >= 65) or (np.isfinite(vulnerability) and vulnerability >= 70) or (np.isfinite(transition) and transition >= 70)):
        return "HIGH"
    if persistent and np.isfinite(funding) and funding >= 70:
        return "HIGH"
    if any(value >= 50 for value in valid_stress):
        return "WATCH"
    if np.isfinite(macro) and np.isfinite(vulnerability) and macro >= 60 and vulnerability >= 60 and transition >= 50:
        return "WATCH"
    if np.isfinite(macro) and np.isfinite(vulnerability) and macro >= 60 and vulnerability >= 60 and any(value >= 35 for value in valid_stress):
        return "WATCH"
    if np.isfinite(transition) and transition >= 70 and ((np.isfinite(macro) and macro >= 50) or (np.isfinite(vulnerability) and vulnerability >= 55)):
        return "WATCH"
    return "NORMAL"


def build_financial_fragility_snapshot(
    *,
    liquidity_regime: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    market_snapshot: Any,
    business_snapshot: Any,
    rates_snapshot: Any,
    funding_snapshot: Any,
    treasury_snapshot: Any,
    transition_snapshot: dict[str, Any] | None = None,
) -> FinancialFragilitySnapshot:
    sources = {
        "liquidity": _normalise(liquidity_regime, "date"),
        "forecast": _normalise(forecast_frame),
        "market": _normalise(_source_frame(market_snapshot, ("history",))),
        "market_daily": _normalise(_source_frame(market_snapshot, ("daily",))),
        "business": _normalise(_source_frame(business_snapshot, ("history",)), "date"),
        "rates": _normalise(_source_frame(rates_snapshot, ("history",))),
        "funding": _normalise(_source_frame(funding_snapshot, ("weekly", "history"))),
        "funding_daily": _normalise(_source_frame(funding_snapshot, ("daily",))),
        "treasury": _normalise(_source_frame(treasury_snapshot, ("weekly", "history"))),
    }
    date_values = [frame["Date"].min() for frame in sources.values() if not frame.empty]
    end_values = [frame["Date"].max() for frame in sources.values() if not frame.empty]
    if not date_values or not end_values:
        return FinancialFragilitySnapshot(pd.DataFrame(), {}, pd.DataFrame(), pd.NaT)
    start = min(date_values)
    end = max(end_values)
    dates = pd.date_range(start, end, freq="W-FRI")
    idx = pd.DatetimeIndex(dates)
    # Keep the weekly date index while retaining Date as a display/export column.
    # Production-aligned series returned by _asof are indexed by these dates.
    history = pd.DataFrame({"Date": dates}, index=idx)

    # Source outputs are already PIT-safe weekly production snapshots. Every alignment is backward-only.
    history["SPX"] = _asof(sources["market"], idx, ("SPX_Close", "SPX"))
    history["LiquidityForecastState"] = _asof(sources["forecast"], idx, ("LiquidityForecastState",), numeric=False)
    history["GlobalLiquidityDirection"] = _asof(sources["liquidity"], idx, ("direction_13w_state", "GlobalLiquidityDirection"), numeric=False)
    history["LiquidityPressure"] = _asof(sources["forecast"], idx, ("LiquidityPressureScore", "LiquidityPressure"))
    history["LiquidityForecastScore"] = _asof(sources["liquidity"], idx, ("global_liquidity_score", "GlobalLiquidityScore"))
    history["RatesPressure"] = _asof(sources["rates"], idx, ("RatesPressureScore", "RatesPressure"))
    history["Curve26W"] = _asof(sources["rates"], idx, ("YieldCurveRegime_26W",), numeric=False)
    history["RatesDirection"] = _asof(sources["rates"], idx, ("RatesDirection",), numeric=False)
    history["CoreFCLevel"] = _asof(sources["rates"], idx, ("FinancialConditionsLevel", "CoreFCLevel"))
    history["CoreFCDirection"] = _asof(sources["rates"], idx, ("FinancialConditionsDirection", "CoreFCDirection"), numeric=False)
    history["CoreFCDirectionScore"] = _asof(sources["rates"], idx, ("FinancialConditionsDirectionScore", "CoreFCDirectionScore"))

    history["SMA200WStretch"] = _asof(sources["market"], idx, ("SMA200WExtensionPercentile", "SMA200WExtensionPercentile_PIT"))
    roc_momentum = _asof(sources["market"], idx, ("SPX_ROC_Momentum_3M", "ROCMomentum3M"))
    history["ROCMomentum3M"] = roc_momentum
    history["PositioningVulnerability"] = _asof(sources["market"], idx, ("PositioningRisk", "PositioningVulnerability"))
    history["PositioningVulnerabilityState"] = _asof(sources["market"], idx, ("PositioningVulnerability", "PositioningState"), numeric=False)
    history["ReserveVulnerabilityRaw"] = _asof(sources["funding"], idx, ("ReserveVulnerability",))
    history["BreadthRisk"] = _asof(sources["market"], idx, ("HistoricalBreadthRisk", "CurrentRiskBreadthRisk", "BreadthRisk"))
    history["HighBetaRisk"] = _asof(sources["market"], idx, ("HistoricalHighBetaRisk", "CurrentRiskHighBetaRisk", "HighBetaRisk"))
    history["RSIDivergenceRisk"] = _asof(sources["market"], idx, ("HistoricalRSIDivergenceRisk", "CurrentRiskRSIDivergenceRisk", "RSIDivergenceRiskScore"))
    history["SPXMomentumCyclePhase"] = _asof(sources["market"], idx, ("MomentumCyclePhase", "SPXMomentumCyclePhase"), numeric=False)
    history["MediumTermSMA_ROCPhase"] = _asof(sources["market"], idx, ("MediumTerm_SMA_ROC_Phase",), numeric=False)
    history["MarketCycleTransitionInput"] = _asof(sources["market"], idx, ("ROCMomentum3M", "SPX_ROC_Momentum_3M"))

    history["HY_OAS"] = _asof(sources["rates"], idx, ("BAMLH0A0HYM2", "HY_OAS", "HYOAS"))
    history["FundingCore"] = _asof(sources["funding"], idx, ("FundingCore",))
    history["MoneyMarketStress"] = _asof(sources["funding"], idx, ("MoneyMarketStress",))
    history["FundingState"] = _asof(sources["funding"], idx, ("FundingState",), numeric=False)
    history["PersistentFundingFlag"] = _asof(sources["funding"], idx, ("PersistentFundingFlag",), numeric=False)
    history["ReservePressure"] = _asof(sources["funding"], idx, ("ReservePressure",))
    history["CollateralStress"] = _asof(sources["funding"], idx, ("CollateralStress",))
    history["MOVE"] = _asof(sources["funding"], idx, ("MOVE",))
    history["MOVE_Z"] = _asof(sources["funding"], idx, ("MOVE_Z",))

    history["VIX"] = _asof(sources["market"], idx, ("VIX",))
    history["VIX3M"] = _asof(sources["market"], idx, ("VIX3M",))
    history["VIX20DChange"] = _asof(sources["market"], idx, ("VIX20DChange", "VIX_20D_Change"))
    history["RealizedVolatility20D"] = _asof(sources["market"], idx, ("RealizedVol20D", "SPX_RealizedVol20D", "RealizedVolatility_20D"))
    history["BusinessCyclePhase"] = _asof(sources["business"], idx, ("BusinessCycleState", "BusinessCyclePhase"), numeric=False)
    history["BusinessCycleLevel"] = _asof(sources["business"], idx, ("BusinessCycleLevel", "Level", "CompositeLevel"))
    history["BusinessCycleMomentum"] = _asof(sources["business"], idx, ("BusinessCycleMomentum", "Momentum"))
    history["BusinessCycleTransitionZone"] = _asof(sources["business"], idx, ("BusinessCycleTransitionZone", "TransitionZone"), numeric=False)

    # Risk transformations use trailing PIT distributions, never the full future sample.
    liquidity_map = history["LiquidityForecastState"].astype(str).str.upper().map({"EXPANSION": 15.0, "BOTTOMING": 30.0, "PEAKING": 65.0, "CONTRACTION": 85.0})
    history["ForecastRisk"] = liquidity_map
    history["LiquidityPressureRisk"] = (0.65 * liquidity_map + 0.35 * pd.to_numeric(history["LiquidityPressure"], errors="coerce")).clip(0, 100)
    history["RatesPressureRisk"] = pit_percentile(history["RatesPressure"])
    history["RatesPressureStatus"] = history["RatesPressureRisk"].map(_rates_pressure_status)
    history["CollateralStressStatus"] = history["MOVE"].map(_move_status)
    level_risk = pit_percentile(history["CoreFCLevel"])
    direction_risk = _direction_risk(history["CoreFCDirection"], history["CoreFCDirectionScore"])
    history["CoreFCLevelRisk"] = level_risk
    history["CoreFCDirectionRisk"] = direction_risk
    history["CoreFCPressure"] = (0.70 * level_risk + 0.30 * direction_risk).clip(0, 100)
    history["MacroPressure"] = _weighted(
        {"liquidity": history["LiquidityPressureRisk"], "rates": history["RatesPressureRisk"], "fc": history["CoreFCPressure"]},
        {"liquidity": 0.60, "rates": 0.20, "fc": 0.20},
    )
    history["MacroPressureState"] = history["MacroPressure"].map(_risk_state)

    history["ROCMomentum3MRisk"] = (100.0 - pit_percentile(history["ROCMomentum3M"])).clip(0, 100)
    reserve = pd.to_numeric(history["ReserveVulnerabilityRaw"], errors="coerce")
    history["ReserveVulnerability"] = (pit_percentile(reserve)).where(reserve.notna())
    history["CurrentRiskVulnerability"] = history[["BreadthRisk", "HighBetaRisk", "RSIDivergenceRisk"]].mean(axis=1, skipna=False)
    history["MarketVulnerability"] = _weighted(
        {"sma": history["SMA200WStretch"], "roc": history["ROCMomentum3MRisk"], "positioning": history["PositioningVulnerability"], "reserve": history["ReserveVulnerability"], "current": history["CurrentRiskVulnerability"]},
        {"sma": 0.55, "roc": 0.20, "positioning": 0.10, "reserve": 0.05, "current": 0.10},
    )
    history["MarketVulnerabilityState"] = history["MarketVulnerability"].map(_risk_state)

    history["HYOAS_3Y_Percentile"] = pit_percentile(history["HY_OAS"])
    history["CreditStress"] = history["HYOAS_3Y_Percentile"]
    history["CreditStressState"] = history["CreditStress"].map(_risk_state)
    funding_base = 25.0 * pd.to_numeric(history["FundingCore"], errors="coerce")
    funding_floor = history["FundingState"].astype(str).str.upper().map({"NORMAL": 0.0, "TECHNICAL FUNDING PRESSURE": 25.0, "TREASURY VOLATILITY": 30.0, "PERSISTENT FUNDING PRESSURE": 65.0, "SYSTEMIC FUNDING STRESS": 90.0})
    history["FundingStress"] = pd.concat([funding_base, funding_floor], axis=1).max(axis=1, skipna=True).clip(0, 100)
    history["FundingStressState"] = history["FundingStress"].map(_risk_state)

    vix = pd.to_numeric(history["VIX"], errors="coerce")
    vix_ratio = vix / pd.to_numeric(history["VIX3M"], errors="coerce").replace(0, np.nan)
    history["VIX_VIX3M_Ratio"] = vix_ratio
    history["VIXLevelRisk"] = np.select([vix.le(15), vix.le(20), vix.le(25), vix.le(35)], [10.0, 30.0, 55.0, 80.0], default=100.0)
    history.loc[vix.isna(), "VIXLevelRisk"] = np.nan
    vix_change = pd.to_numeric(history["VIX20DChange"], errors="coerce")
    if vix_change.isna().all():
        vix_change = vix.diff(4)
    history["VIX20DChange"] = vix_change
    history["VIXMomentumRisk"] = (pit_percentile(vix_change).sub(50.0).mul(2.0)).clip(0, 100)
    history["VIXTermStructureRisk"] = np.select([vix_ratio.lt(0.85), vix_ratio.lt(0.95), vix_ratio.lt(1.0), vix_ratio.lt(1.10)], [10.0, 30.0, 55.0, 80.0], default=100.0)
    history.loc[vix_ratio.isna(), "VIXTermStructureRisk"] = np.nan
    history["RealizedVolRisk"] = pit_percentile(history["RealizedVolatility20D"])
    history["MarketVolatilityStress"] = _weighted(
        {"vix": history["VIXLevelRisk"], "momentum": history["VIXMomentumRisk"], "term": history["VIXTermStructureRisk"], "realized": history["RealizedVolRisk"]},
        {"vix": 0.30, "momentum": 0.30, "term": 0.25, "realized": 0.15},
    )
    history["MarketVolatilityStressState"] = history["MarketVolatilityStress"].map(lambda x: _risk_state(x, volatility=True))

    # Transition components are turning-point measures, not stress scores.
    liq_sign = history["GlobalLiquidityDirection"].map(lambda value: _state_sign(value, "liquidity"))
    liq_change = liq_sign.ne(liq_sign.shift()).astype(float) * 45.0
    pressure_change = pit_percentile(_safe_change(history["LiquidityPressure"], 1)).fillna(0.0)
    history["LiquidityTransition"] = (liq_change + pressure_change * 0.35).clip(0, 100)
    market_sign = history["SPXMomentumCyclePhase"].map(lambda value: _state_sign(value, "market"))
    history["MarketCycleTransition"] = (market_sign.ne(market_sign.shift()).astype(float) * 50.0 + pit_percentile(_safe_change(history["MarketCycleTransitionInput"], 1)).fillna(0.0) * 0.50).clip(0, 100)
    business_sign = history["BusinessCyclePhase"].map(lambda value: _state_sign(value, "business"))
    business_zone = history["BusinessCycleTransitionZone"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    history["BusinessCycleTransition"] = (business_sign.ne(business_sign.shift()).astype(float) * 45.0 + business_zone.astype(float) * 45.0 + pit_percentile(_safe_change(history["BusinessCycleMomentum"], 1)).fillna(0.0) * 0.10).clip(0, 100)
    rates_changes = pd.concat([history["RatesDirection"].astype(str).ne(history["RatesDirection"].shift()), history["CoreFCDirection"].astype(str).ne(history["CoreFCDirection"].shift())], axis=1).sum(axis=1)
    history["RatesFCTransition"] = (rates_changes * 45.0 + pit_percentile(_safe_change(history["CoreFCDirectionScore"], 1)).fillna(0.0) * 0.55).clip(0, 100)
    cross = (liq_sign.ne(market_sign).astype(float) * 50.0 + liq_sign.ne(business_sign).astype(float) * 50.0).clip(0, 100)
    history["CrossCycleDivergence"] = cross
    history["TransitionRisk"] = _weighted(
        {"liquidity": history["LiquidityTransition"], "market": history["MarketCycleTransition"], "business": history["BusinessCycleTransition"], "rates": history["RatesFCTransition"], "cross": history["CrossCycleDivergence"]},
        {"liquidity": 0.30, "market": 0.25, "business": 0.20, "rates": 0.15, "cross": 0.10},
    )
    history["TransitionRiskState"] = history["TransitionRisk"].map(_transition_state)

    history["GeneralRegime"] = history.apply(_regime_row, axis=1)
    starts, duration = _state_start_and_duration(history["GeneralRegime"], history["Date"])
    history["GeneralRegimeStartDate"] = starts
    history["WeeksInGeneralRegime"] = duration
    history["MacroPressureDirection"] = np.where(history["MacroPressure"].diff() > 0, "RISING", np.where(history["MacroPressure"].diff() < 0, "FALLING", "STABLE"))
    history["MarketVulnerabilityDirection"] = np.where(history["MarketVulnerability"].diff() > 0, "RISING", np.where(history["MarketVulnerability"].diff() < 0, "FALLING", "STABLE"))
    history["DataCoverage"] = history[["MacroPressure", "MarketVulnerability", "CreditStress", "FundingStress", "MarketVolatilityStress", "TransitionRisk"]].notna().mean(axis=1) * 100.0
    history["FinancialFragilityModelVersion"] = MODEL_VERSION

    if history.empty:
        return FinancialFragilitySnapshot(history, {}, pd.DataFrame(), pd.NaT)
    latest = history.iloc[-1]
    driver_map = {"MacroPressure": "Macro pressure", "MarketVulnerability": "Market vulnerability", "CreditStress": "HY credit stress", "FundingStress": "Funding stress", "MarketVolatilityStress": "Volatility shock", "TransitionRisk": "Transition risk"}
    stabilizers = []
    for column, label in (("CreditStress", "Credit calm"), ("FundingStress", "Funding normal"), ("MarketVolatilityStress", "Volatility subdued"), ("LiquidityPressureRisk", "Liquidity pressure contained")):
        value = _number(latest.get(column))
        if np.isfinite(value) and value < 35:
            stabilizers.append(label)
    current = latest.to_dict()
    current.update({
        "DashboardAsOf": _date(latest.get("Date")),
        "GeneralRegime": _text(latest.get("GeneralRegime")),
        "PrimaryDrivers": _top_names(latest, driver_map),
        "PrimaryStabilizers": ", ".join(stabilizers[:3]) or "UNAVAILABLE",
        "KeyDivergence": _key_divergence(latest),
        "Interpretation": _interpretation(latest),
        "DataCoverage": _number(latest.get("DataCoverage")),
    })
    current = _overlay_current_stress(
        current,
        history,
        sources["market_daily"],
        sources["funding_daily"],
        sources["rates"],
        getattr(market_snapshot, "current", None),
    )
    current["RatesDataAsOf"] = sources["rates"]["Date"].max() if not sources["rates"].empty else pd.NaT
    current["RatesPressureStatus"] = _rates_pressure_status(current.get("RatesPressureRisk"))
    current["CollateralStressStatus"] = _move_status(current.get("MOVE"))
    current["GeneralRegime"] = _regime_row(pd.Series(current))
    current["Interpretation"] = _interpretation(pd.Series(current))
    current["PrimaryDrivers"] = _top_names(pd.Series(current), driver_map)
    current["PrimaryStabilizers"] = ", ".join(
        label
        for column, label in (("CreditStress", "Credit calm"), ("FundingStress", "Funding normal"), ("MarketVolatilityStress", "Volatility subdued"), ("LiquidityPressureRisk", "Liquidity pressure contained"))
        if np.isfinite(_number(current.get(column))) and _number(current.get(column)) < 35
    ) or "UNAVAILABLE"
    quality = pd.DataFrame([{"Component": column, "Available": bool(pd.to_numeric(history[column], errors="coerce").notna().any()), "Latest": bool(pd.notna(latest.get(column)))} for column in ("MacroPressure", "MarketVulnerability", "CreditStress", "FundingStress", "MarketVolatilityStress", "TransitionRisk")])
    return FinancialFragilitySnapshot(history, current, quality, _date(latest.get("Date")))


def _key_divergence(row: pd.Series) -> str:
    liquidity = _text(row.get("LiquidityForecastState"), "").upper()
    market = _text(row.get("SPXMomentumCyclePhase"), "").upper()
    business = _text(row.get("BusinessCyclePhase"), "").upper()
    if "CONTRACT" in liquidity and "EXPANSION" in market:
        return "Liquidity is contracting while market momentum remains expansionary."
    if "PEAK" in liquidity and "EXPANSION" in market:
        return "Liquidity is peaking while market momentum remains expansionary."
    if business and market and _state_sign(business, "business") != _state_sign(market, "market"):
        return "Business-cycle and market-cycle phases are diverging."
    return "No major cross-cycle divergence is detected."


def _interpretation(row: pd.Series) -> str:
    regime = _text(row.get("GeneralRegime"), "UNAVAILABLE")
    if regime == "NORMAL":
        return "Latent fragility is not currently accompanied by broad stress transmission."
    if regime == "WATCH":
        return "Pressure is beginning to transmit, but broad systemic stress is not confirmed."
    if regime == "HIGH":
        return "At least one material stress channel is active and requires close monitoring."
    if regime == "EXTREME":
        return "Multiple stress channels indicate systemic transmission."
    return "Financial fragility is only partially observable."

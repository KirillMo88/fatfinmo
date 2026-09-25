from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv
from fred_client import FredApiError, download_fred_series


BUSINESS_CYCLE_MODEL_VERSION = "BUSINESS_CYCLE_V1"
INFLATION_LAYER_MODEL_VERSION = "INFLATION_LAYER_V1"
ECONOMY_REGIME_MODEL_VERSION = "ECONOMY_REGIME_V1"

BUSINESS_CYCLE_STORAGE_DIR = Path("persistent") / "business_cycle"
BUSINESS_CYCLE_FRED_CACHE = BUSINESS_CYCLE_STORAGE_DIR / "fred_series.parquet"
INVESTING_ISM_SERVICES_URL = "https://www.investing.com/economic-calendar/ism-non-manufacturing-pmi-176"

BUSINESS_CYCLE_FRED_SERIES = {
    "NAPM": "ISM Manufacturing PMI",
    "CFNAI": "Chicago Fed National Activity Index",
    "ICSA": "Initial Claims",
    "CCSA": "Continuing Claims",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "INDPRO": "Industrial Production",
    "RSAFS": "Retail Sales",
    "PCEC96": "Real PCE",
    "W875RX1": "Real Personal Income Ex Transfers",
}

INFLATION_FRED_SERIES = {
    "T5YIE": "5-Year Breakeven Inflation Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "EXPINF1YR": "1-Year Expected Inflation",
    "EXPINF5YR": "5-Year Expected Inflation",
    "MICH": "University of Michigan 1Y Inflation Expectations",
    "T5YIFR": "5-Year, 5-Year Forward Inflation Expectation Rate",
    "CPIAUCSL": "Headline CPI",
    "CPILFESL": "Core CPI",
    "PCEPILFE": "Core PCE",
    "PPIACO": "PPI All Commodities",
}

ALL_FRED_SERIES = {**BUSINESS_CYCLE_FRED_SERIES, **INFLATION_FRED_SERIES}
TRADINGVIEW_ECONOMIC_FALLBACKS: dict[str, str] = {}

MONTHLY_RELEASE_LAGS = {
    "NAPM": 5,
    "CFNAI": 25,
    "UNRATE": 7,
    "PAYEMS": 7,
    "INDPRO": 18,
    "RSAFS": 18,
    "PCEC96": 32,
    "W875RX1": 32,
    "MICH": 30,
    "CPIAUCSL": 16,
    "CPILFESL": 16,
    "PCEPILFE": 32,
    "PPIACO": 16,
}
WEEKLY_RELEASE_LAGS = {"ICSA": 6, "CCSA": 6}
DAILY_SERIES = {"T5YIE", "T10YIE", "EXPINF1YR", "EXPINF5YR", "T5YIFR"}

BUSINESS_PHASES = [
    "STRONG EXPANSION",
    "LATE / SLOWING EXPANSION",
    "DETERIORATING CONTRACTION",
    "EARLY RECOVERY",
]
ECONOMY_REGIMES = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DISINFLATIONARY SLOWDOWN"]
ASSET_TICKERS = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "BTC": "BTC-USD"}
FORWARD_HORIZONS = {"3M": 13, "6M": 26, "12M": 52}


@dataclass
class BusinessCycleSnapshot:
    history: pd.DataFrame
    current: dict[str, Any]
    phase_returns: pd.DataFrame
    regime_returns: pd.DataFrame
    eta_squared: pd.DataFrame
    diagnostics: pd.DataFrame
    data_quality: pd.DataFrame


def build_business_cycle_snapshot(
    api_key: str | None = None,
    start_date: str | pd.Timestamp = "2010-01-01",
    end_date: str | pd.Timestamp | None = None,
) -> BusinessCycleSnapshot:
    end = pd.Timestamp(end_date).tz_localize(None).normalize() if end_date else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    display_start = pd.Timestamp(start_date).tz_localize(None).normalize()
    model_start = min(display_start, pd.Timestamp("1990-01-01"))
    calendar = pd.date_range(model_start, end, freq="W-FRI")
    fred = load_business_cycle_fred(api_key, observation_start=str(model_start.date()))
    full_history = build_business_cycle_history(calendar, fred)
    asset_prices = load_asset_prices(calendar)
    phase_returns = build_forward_return_stats(full_history, asset_prices, "BusinessCycleState", BUSINESS_PHASES)
    regime_returns = build_forward_return_stats(full_history, asset_prices, "EconomyRegime", ECONOMY_REGIMES)
    eta = build_eta_squared(full_history, asset_prices)
    history = full_history.loc[pd.to_datetime(full_history["date"], errors="coerce").ge(display_start)].reset_index(drop=True)
    diagnostics = build_diagnostics(history)
    quality = build_data_quality(fred, history)
    current = latest_current(history)
    return BusinessCycleSnapshot(history, current, phase_returns, regime_returns, eta, diagnostics, quality)


def load_business_cycle_fred(api_key: str | None = None, observation_start: str = "1990-01-01") -> pd.DataFrame:
    cache = read_fred_cache()
    frames: list[pd.DataFrame] = []
    for series_id in ALL_FRED_SERIES:
        frame = pd.DataFrame()
        try:
            frame = download_fred_series(series_id, api_key=api_key, observation_start=observation_start)
            if not frame.empty:
                frame["Description"] = ALL_FRED_SERIES[series_id]
                frame["IsReleaseDated"] = False
        except (FredApiError, Exception):
            frame = pd.DataFrame()

        if series_id == "NAPM":
            # Keep the full saved/release history and overlay only the latest
            # Investing observations; the live page is intentionally partial.
            cached_pmi = cache.loc[cache["Series_ID"].astype(str).str.upper().eq(series_id)] if not cache.empty else pd.DataFrame()
            release_pmi = load_pmi_release_fallback(observation_start)
            live_pmi = load_investing_pmi_releases(observation_start)
            pmi_frames = [candidate for candidate in [frame, cached_pmi, release_pmi, live_pmi] if not candidate.empty]
            if pmi_frames:
                frame = pd.concat(pmi_frames, ignore_index=True)
        if frame.empty and series_id == "NAPM":
            frame = load_pmi_release_fallback(observation_start)
        if frame.empty:
            cached = cache.loc[cache["Series_ID"].astype(str).str.upper().eq(series_id)] if not cache.empty else pd.DataFrame()
            if not cached.empty:
                frame = cached.copy()
        if frame.empty:
            frame = load_tradingview_economic_fallback(series_id, observation_start)
        if frame.empty:
            frame = pd.DataFrame(
                {
                    "Series_ID": [series_id],
                    "Date": [pd.NaT],
                    "Value": [np.nan],
                    "Description": [ALL_FRED_SERIES[series_id]],
                    "IsReleaseDated": [False],
                }
            )
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Series_ID", "Date", "Value", "Description", "IsReleaseDated"])
    if "IsReleaseDated" not in out.columns:
        out["IsReleaseDated"] = False
    out["Series_ID"] = out["Series_ID"].astype(str).str.upper()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out["IsReleaseDated"] = out["IsReleaseDated"].fillna(False).astype(bool)
    out = out.dropna(subset=["Date"]).sort_values(["Series_ID", "Date"]).drop_duplicates(["Series_ID", "Date"], keep="last")
    write_fred_cache(out)
    return out


def load_investing_pmi_releases(observation_start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch the latest visible ISM releases while preserving the existing history."""
    try:
        from macro_surprises import INVESTING_SOURCES, fetch_investing_releases

        spec = INVESTING_SOURCES["PMI"]
        releases = fetch_investing_releases("PMI", spec["url"], spec["definition"])
        dates = pd.to_datetime(releases["ReleaseDate"], errors="coerce")
        values = pd.to_numeric(releases["Actual"], errors="coerce")
        out = pd.DataFrame(
            {
                "Series_ID": "NAPM",
                "Date": dates,
                "Value": values,
                "Description": "ISM Manufacturing PMI (Investing.com release update)",
                "IsReleaseDated": True,
            }
        )
        return out.loc[dates.ge(pd.Timestamp(observation_start))].dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates("Date", keep="last")
    except Exception:
        return pd.DataFrame()


def load_investing_ism_services_releases(observation_start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch visible ISM Services releases from Investing.com.

    The historical FRED series remains available as a fallback in the macro
    tab, while Investing supplies the current release stream.
    """
    try:
        from macro_surprises import fetch_investing_releases

        releases = fetch_investing_releases(
            "ISM Services PMI",
            INVESTING_ISM_SERVICES_URL,
            "ISM Services PMI",
        )
        dates = pd.to_datetime(releases["ReleaseDate"], errors="coerce")
        values = pd.to_numeric(releases["Actual"], errors="coerce")
        out = pd.DataFrame(
            {
                "Series_ID": "NMFCI",
                "Date": dates,
                "Value": values,
                "Description": "ISM Services PMI (Investing.com release update)",
                "IsReleaseDated": True,
            }
        )
        return (
            out.loc[dates.ge(pd.Timestamp(observation_start))]
            .dropna(subset=["Date", "Value"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
        )
    except Exception:
        return pd.DataFrame()


def load_pmi_release_fallback(observation_start: str) -> pd.DataFrame:
    try:
        from macro_surprises import load_or_initialize_releases, select_point_in_time_releases

        releases, _, _ = load_or_initialize_releases()
        releases = select_point_in_time_releases(releases)
        rows = releases.loc[releases["Indicator"].astype(str).eq("PMI")].copy()
        rows["ReleaseDate"] = pd.to_datetime(rows["ReleaseDate"], errors="coerce")
        rows["Actual"] = pd.to_numeric(rows["Actual"], errors="coerce")
        rows = rows.loc[rows["ReleaseDate"].ge(pd.Timestamp(observation_start))]
        out = pd.DataFrame(
            {
                "Series_ID": "NAPM",
                "Date": rows["ReleaseDate"],
                "Value": rows["Actual"],
                "Description": "ISM Manufacturing PMI (release history)",
                "IsReleaseDated": True,
            }
        )
        return out.dropna(subset=["Date", "Value"]).sort_values("Date").drop_duplicates("Date", keep="last")
    except Exception:
        return pd.DataFrame()


def load_tradingview_economic_fallback(series_id: str, observation_start: str) -> pd.DataFrame:
    symbol = TRADINGVIEW_ECONOMIC_FALLBACKS.get(series_id.upper())
    if not symbol:
        return pd.DataFrame()
    try:
        from tradingview_mcp import get_economic_data, validate_economic_result

        result = get_economic_data(symbol, date_from=observation_start)
        valid, _ = validate_economic_result(result, min_observations=24, max_stale_days=240)
        if not valid or result.frame.empty:
            return pd.DataFrame()
        out = pd.DataFrame(
            {
                "Series_ID": series_id.upper(),
                "Date": pd.to_datetime(result.frame["date"], errors="coerce"),
                "Value": pd.to_numeric(result.frame["value"], errors="coerce"),
                "Description": f"{ALL_FRED_SERIES.get(series_id.upper(), series_id)} ({symbol})",
            }
        )
        return out.dropna(subset=["Date"]).sort_values("Date")
    except Exception:
        return pd.DataFrame()


def read_fred_cache() -> pd.DataFrame:
    try:
        if BUSINESS_CYCLE_FRED_CACHE.exists():
            return pd.read_parquet(BUSINESS_CYCLE_FRED_CACHE)
    except Exception:
        pass
    return pd.DataFrame(columns=["Series_ID", "Date", "Value", "Description", "IsReleaseDated"])


def write_fred_cache(frame: pd.DataFrame) -> None:
    try:
        BUSINESS_CYCLE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(BUSINESS_CYCLE_FRED_CACHE, index=False)
    except Exception:
        pass


def build_business_cycle_history(calendar: pd.DatetimeIndex, fred: pd.DataFrame) -> pd.DataFrame:
    history = pd.DataFrame({"date": calendar})
    for series_id in ALL_FRED_SERIES:
        history[series_id] = align_fred_series_to_weekly(
            fred_series(fred, series_id),
            calendar,
            series_id,
            release_dated=fred_series_is_release_dated(fred, series_id),
        ).to_numpy()

    history["ISM_3MMA"] = num(history["NAPM"]).rolling(13, min_periods=4).mean()
    history["CFNAI_3MMA"] = num(history["CFNAI"]).rolling(13, min_periods=4).mean()
    for col in ["INDPRO", "RSAFS", "PCEC96", "W875RX1", "PAYEMS"]:
        history[f"{col}_YoY"] = num(history[col]).pct_change(52, fill_method=None) * 100.0
        history[f"{col}_YoY_3MMA"] = history[f"{col}_YoY"].rolling(13, min_periods=4).mean()

    history["ISM_Z"] = expanding_z(history["ISM_3MMA"])
    history["CFNAI_Z"] = expanding_z(history["CFNAI_3MMA"])
    history["InitialClaims_Z_INV"] = -expanding_z(history["ICSA"])
    history["ContinuingClaims_Z_INV"] = -expanding_z(history["CCSA"])
    history["Unemployment_Z_INV"] = -expanding_z(history["UNRATE"])
    history["Payrolls_Z"] = expanding_z(history["PAYEMS_YoY_3MMA"])
    history["IndustrialProduction_Z"] = expanding_z(history["INDPRO_YoY_3MMA"])
    history["RetailSales_Z"] = expanding_z(history["RSAFS_YoY_3MMA"])
    history["RealPCE_Z"] = expanding_z(history["PCEC96_YoY_3MMA"])
    history["RealPersonalIncome_Z"] = expanding_z(history["W875RX1_YoY_3MMA"])

    history["SurveyScore"] = weighted_mean([history["ISM_Z"], history["CFNAI_Z"]], [0.5, 0.5])
    history["LaborScore"] = weighted_mean(
        [history["InitialClaims_Z_INV"], history["ContinuingClaims_Z_INV"], history["Unemployment_Z_INV"], history["Payrolls_Z"]],
        [0.35, 0.15, 0.25, 0.25],
    )
    history["ProductionScore"] = history["IndustrialProduction_Z"]
    history["DemandIncomeScore"] = weighted_mean([history["RetailSales_Z"], history["RealPCE_Z"], history["RealPersonalIncome_Z"]], [1, 1, 1])
    history["BusinessCycleLevel"] = weighted_mean(
        [history["SurveyScore"], history["ProductionScore"], history["DemandIncomeScore"], history["LaborScore"]],
        [0.30, 0.30, 0.25, 0.15],
    )
    for pillar in ["SurveyScore", "ProductionScore", "DemandIncomeScore", "LaborScore"]:
        history[f"{pillar}_Momentum_13W"] = num(history[pillar]).diff(13)
    history["SurveyMomentum_13W"] = history["SurveyScore_Momentum_13W"]
    history["ProductionMomentum_13W"] = history["ProductionScore_Momentum_13W"]
    history["DemandIncomeMomentum_13W"] = history["DemandIncomeScore_Momentum_13W"]
    history["LaborMomentum_13W"] = history["LaborScore_Momentum_13W"]
    history["BusinessCycleMomentum"] = weighted_mean(
        [history["SurveyMomentum_13W"], history["ProductionMomentum_13W"], history["DemandIncomeMomentum_13W"], history["LaborMomentum_13W"]],
        [0.40, 0.30, 0.20, 0.10],
    )
    history["BusinessCycleCandidateState"] = [
        classify_business_cycle(level, momentum) for level, momentum in zip(history["BusinessCycleLevel"], history["BusinessCycleMomentum"])
    ]
    history["BusinessCycleState"] = confirm_state(history["BusinessCycleCandidateState"])
    level_values = num(history["BusinessCycleLevel"])
    momentum_values = num(history["BusinessCycleMomentum"])
    history["BusinessCycleTransitionZone"] = (
        level_values.notna()
        & momentum_values.notna()
        & ((level_values.abs() < 0.10) | (momentum_values.abs() < 0.05))
    )
    history["BusinessCycleConfidence"] = [
        business_cycle_confidence(level, momentum, transition)
        for level, momentum, transition in zip(history["BusinessCycleLevel"], history["BusinessCycleMomentum"], history["BusinessCycleTransitionZone"])
    ]
    history["BusinessCyclePosition"] = np.select(
        [level_values > 0, level_values < 0],
        ["ABOVE TREND", "BELOW TREND"],
        default="DATA INCOMPLETE",
    )
    history["BusinessCycleDirection"] = np.select(
        [momentum_values > 0, momentum_values < 0],
        ["IMPROVING", "DETERIORATING"],
        default="DATA INCOMPLETE",
    )
    history["LaborCycleState"] = [classify_labor_cycle(score, momentum) for score, momentum in zip(history["LaborScore"], history["LaborMomentum_13W"])]
    history["ProductivityExpansionFlag"] = history["BusinessCycleState"].isin(["STRONG EXPANSION", "EARLY RECOVERY"]) & history["LaborCycleState"].isin(["LABOR SLOWING", "LABOR DETERIORATION"])

    add_inflation_layer(history)
    history["EconomyRegime"] = [
        classify_economy_regime(direction, inflation)
        for direction, inflation in zip(history["BusinessCycleDirection"], history["InflationState"])
    ]
    history["RegimeConfidence"] = [regime_confidence(bc, inf) for bc, inf in zip(history["BusinessCycleConfidence"], history["InflationConfidence"])]
    history["BusinessCycleModelVersion"] = BUSINESS_CYCLE_MODEL_VERSION
    history["InflationLayerModelVersion"] = INFLATION_LAYER_MODEL_VERSION
    history["EconomyRegimeModelVersion"] = ECONOMY_REGIME_MODEL_VERSION
    return history


def add_inflation_layer(history: pd.DataFrame) -> None:
    for col in ["T5YIE", "T10YIE", "EXPINF1YR", "EXPINF5YR", "MICH"]:
        history[f"{col}_Delta13W"] = num(history[col]).diff(13)
        history[f"{col}_Delta26W"] = num(history[col]).diff(26)
        history[f"{col}_Momentum"] = 0.70 * expanding_z(history[f"{col}_Delta26W"]) + 0.30 * expanding_z(history[f"{col}_Delta13W"])
        history[f"{col}_Change_4W"] = num(history[col]).diff(4)

    history["MarketPricingScore"] = weighted_mean([history["T5YIE_Momentum"], history["T10YIE_Momentum"]], [0.65, 0.35])
    history["ModelImpliedInflationScore"] = weighted_mean([history["EXPINF1YR_Momentum"], history["EXPINF5YR_Momentum"]], [0.60, 0.40])
    history["SurveyInflationScore"] = history["MICH_Momentum"]
    history["InflationDirectionScore"] = weighted_mean(
        [history["MarketPricingScore"], history["ModelImpliedInflationScore"], history["SurveyInflationScore"]],
        [0.35, 0.40, 0.25],
    )
    inflation_score = num(history["InflationDirectionScore"])
    history["InflationCandidateState"] = np.select(
        [inflation_score > 0, inflation_score < 0],
        ["RISING", "FALLING"],
        default="DATA INCOMPLETE",
    )
    history["InflationState"] = confirm_state(history["InflationCandidateState"])
    history["InflationTransitionZone"] = inflation_score.notna() & (inflation_score.abs() < 0.20)
    history["MarketPricingDirection"] = score_direction(history["MarketPricingScore"], "RISING", "FALLING")
    history["ModelImpliedDirection"] = score_direction(history["ModelImpliedInflationScore"], "RISING", "FALLING")
    history["SurveyInflationDirection"] = score_direction(history["SurveyInflationScore"], "RISING", "FALLING")
    history["InflationChannelAgreement"] = [
        channel_agreement(mp, mi, survey) for mp, mi, survey in zip(history["MarketPricingScore"], history["ModelImpliedInflationScore"], history["SurveyInflationScore"])
    ]
    history["InflationConfidence"] = [
        inflation_confidence(score, transition, agreement, state, mp, mi, survey)
        for score, transition, agreement, state, mp, mi, survey in zip(
            history["InflationDirectionScore"],
            history["InflationTransitionZone"],
            history["InflationChannelAgreement"],
            history["InflationState"],
            history["MarketPricingScore"],
            history["ModelImpliedInflationScore"],
            history["SurveyInflationScore"],
        )
    ]

    history["InflationCurve"] = num(history["EXPINF1YR"]) - num(history["EXPINF5YR"])
    history["InflationCurveChange_13W"] = num(history["InflationCurve"]).diff(13)
    history["InflationCurveChange_26W"] = num(history["InflationCurve"]).diff(26)
    history["T5YIFR_Change_13W"] = num(history["T5YIFR"]).diff(13)
    history["T5YIFR_Change_26W"] = num(history["T5YIFR"]).diff(26)
    structural_change = num(history["T5YIFR_Change_26W"])
    history["StructuralInflationDirection"] = np.select(
        [structural_change > 0, structural_change < 0],
        ["RISING", "FALLING"],
        default="DATA INCOMPLETE",
    )

    components = []
    for col in ["PCEPILFE", "CPILFESL", "CPIAUCSL", "PPIACO"]:
        yoy = num(history[col]).pct_change(52, fill_method=None) * 100.0
        annualized_3m = ((num(history[col]) / num(history[col]).shift(13)) ** 4 - 1.0) * 100.0
        component = 0.50 * expanding_z(yoy.diff(13)) + 0.50 * expanding_z(annualized_3m - yoy)
        history[f"{col}_YoY"] = yoy
        history[f"{col}_3MAnnualized"] = annualized_3m
        history[f"{col}_RealizedMomentum"] = component
        components.append(component)
    history["RealizedInflationMomentum"] = weighted_mean(components, [0.35, 0.30, 0.20, 0.15])
    realized_momentum = num(history["RealizedInflationMomentum"])
    history["RealizedInflationDirection"] = np.select(
        [realized_momentum > 0, realized_momentum < 0],
        ["ACCELERATING", "DISINFLATING"],
        default="DATA INCOMPLETE",
    )
    history["InflationConfirmationStatus"] = [
        inflation_confirmation(state, realized)
        for state, realized in zip(history["InflationState"], history["RealizedInflationDirection"])
    ]


def fred_series(fred: pd.DataFrame, series_id: str) -> pd.Series:
    if fred is None or fred.empty:
        return pd.Series(dtype="float64")
    rows = fred.loc[fred["Series_ID"].astype(str).str.upper().eq(series_id.upper())].copy()
    if rows.empty:
        return pd.Series(dtype="float64")
    return pd.Series(pd.to_numeric(rows["Value"], errors="coerce").values, index=pd.to_datetime(rows["Date"], errors="coerce")).dropna().sort_index()


def fred_series_is_release_dated(fred: pd.DataFrame, series_id: str) -> bool:
    if fred is None or fred.empty or "IsReleaseDated" not in fred.columns:
        return False
    rows = fred.loc[fred["Series_ID"].astype(str).str.upper().eq(series_id.upper()), "IsReleaseDated"]
    return bool(rows.fillna(False).astype(bool).any())


def align_fred_series_to_weekly(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    series_id: str,
    *,
    release_dated: bool = False,
) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(np.nan, index=calendar)
    source = pd.Series(pd.to_numeric(series, errors="coerce").values, index=pd.to_datetime(series.index, errors="coerce")).dropna().sort_index()
    if release_dated or series_id in DAILY_SERIES:
        weekly = source.resample("W-FRI").last()
    elif series_id in WEEKLY_RELEASE_LAGS:
        effective_index = source.index + pd.to_timedelta(WEEKLY_RELEASE_LAGS[series_id], unit="D")
        weekly = pd.Series(source.values, index=effective_index).sort_index().resample("W-FRI").last()
    else:
        lag = MONTHLY_RELEASE_LAGS.get(series_id, 20)
        effective_index = source.index + pd.offsets.MonthBegin(1) + pd.to_timedelta(lag, unit="D")
        weekly = pd.Series(source.values, index=effective_index).sort_index().resample("W-FRI").last()
    full_index = weekly.index.union(calendar)
    return weekly.reindex(full_index).sort_index().ffill().reindex(calendar)


def expanding_z(series: pd.Series, min_periods: int = 156) -> pd.Series:
    values = num(series)
    mean = values.expanding(min_periods=min_periods).mean()
    std = values.expanding(min_periods=min_periods).std(ddof=0)
    return (values - mean) / std.replace(0, np.nan)


def weighted_mean(series_list: list[pd.Series], weights: list[float]) -> pd.Series:
    frame = pd.concat([num(series) for series in series_list], axis=1)
    weight_arr = np.asarray(weights, dtype=float)
    values = frame.to_numpy(dtype=float)
    mask = np.isfinite(values)
    weighted = np.where(mask, values * weight_arr, 0.0).sum(axis=1)
    denom = np.where(mask, weight_arr, 0.0).sum(axis=1)
    return pd.Series(np.divide(weighted, denom, out=np.full(len(frame), np.nan), where=denom > 0), index=frame.index)


def confirm_state(candidate: pd.Series | list[str]) -> pd.Series:
    candidates = pd.Series(candidate).astype("object")
    out: list[Any] = []
    active = None
    prev = None
    for value in candidates:
        if pd.isna(value):
            out.append(active)
            prev = value
            continue
        if active is None:
            active = value
        elif value == prev and value != active:
            active = value
        out.append(active)
        prev = value
    return pd.Series(out, index=candidates.index)


def classify_business_cycle(level: float, momentum: float) -> str:
    if not np.isfinite(safe_float(level)) or not np.isfinite(safe_float(momentum)):
        return "DATA INCOMPLETE"
    if level > 0 and momentum > 0:
        return "STRONG EXPANSION"
    if level > 0 and momentum < 0:
        return "LATE / SLOWING EXPANSION"
    if level < 0 and momentum < 0:
        return "DETERIORATING CONTRACTION"
    if level < 0 and momentum > 0:
        return "EARLY RECOVERY"
    return "TRANSITION"


def business_cycle_confidence(level: float, momentum: float, transition: bool) -> str:
    if not np.isfinite(safe_float(level)) or not np.isfinite(safe_float(momentum)):
        return "LOW"
    if bool(transition):
        return "LOW"
    if abs(safe_float(level)) >= 0.25 and abs(safe_float(momentum)) >= 0.10:
        return "HIGH"
    return "MEDIUM"


def classify_labor_cycle(score: float, momentum: float) -> str:
    if not np.isfinite(safe_float(score)) or not np.isfinite(safe_float(momentum)):
        return "DATA INCOMPLETE"
    if score > 0 and momentum > 0:
        return "LABOR EXPANSION"
    if score > 0 and momentum < 0:
        return "LABOR SLOWING"
    if score < 0 and momentum < 0:
        return "LABOR DETERIORATION"
    if score < 0 and momentum > 0:
        return "LABOR RECOVERY"
    return "LABOR TRANSITION"


def score_direction(series: pd.Series, positive: str, negative: str) -> pd.Series:
    values = num(series)
    return pd.Series(
        np.select([values > 0, values < 0], [positive, negative], default="DATA INCOMPLETE"),
        index=series.index,
    )


def channel_agreement(*scores: float) -> str:
    signs = [np.sign(safe_float(score)) for score in scores if np.isfinite(safe_float(score)) and safe_float(score) != 0]
    if len(signs) < 2:
        return "CONFLICTING"
    dominant = max(sum(sign > 0 for sign in signs), sum(sign < 0 for sign in signs))
    if dominant == 3:
        return "3_OF_3"
    if dominant == 2:
        return "2_OF_3"
    if dominant == 1:
        return "1_OF_3"
    return "CONFLICTING"


def inflation_confidence(score: float, transition: bool, agreement: str, state: str, *channels: float) -> str:
    if not np.isfinite(safe_float(score)) or state == "DATA INCOMPLETE":
        return "LOW"
    if bool(transition) or agreement in {"1_OF_3", "CONFLICTING"}:
        return "LOW"
    direction = 1 if state == "RISING" else -1
    agreeing = sum(np.sign(safe_float(channel)) == direction for channel in channels if np.isfinite(safe_float(channel)) and safe_float(channel) != 0)
    if abs(safe_float(score)) >= 0.70 and agreeing == 3:
        return "HIGH"
    if abs(safe_float(score)) >= 0.20 and agreeing >= 2:
        return "MEDIUM"
    return "LOW"


def inflation_confirmation(inflation_state: str, realized_direction: str) -> str:
    if inflation_state == "RISING" and realized_direction == "ACCELERATING":
        return "CONFIRMED RISING"
    if inflation_state == "FALLING" and realized_direction == "DISINFLATING":
        return "CONFIRMED FALLING"
    if inflation_state == "RISING" and realized_direction == "DISINFLATING":
        return "EARLY RISING"
    if inflation_state == "FALLING" and realized_direction == "ACCELERATING":
        return "EARLY FALLING"
    return "DATA INCOMPLETE"


def classify_economy_regime(direction: str, inflation_state: str) -> str:
    if direction == "IMPROVING" and inflation_state == "FALLING":
        return "GOLDILOCKS"
    if direction == "IMPROVING" and inflation_state == "RISING":
        return "REFLATION"
    if direction == "DETERIORATING" and inflation_state == "RISING":
        return "STAGFLATION"
    if direction == "DETERIORATING" and inflation_state == "FALLING":
        return "DISINFLATIONARY SLOWDOWN"
    return "DATA INCOMPLETE"


def regime_confidence(business_confidence: str, inflation_confidence_value: str) -> str:
    if "DATA" in {str(business_confidence), str(inflation_confidence_value)}:
        return "LOW"
    if "LOW" in {business_confidence, inflation_confidence_value}:
        return "LOW"
    if "MEDIUM" in {business_confidence, inflation_confidence_value}:
        return "MEDIUM"
    return "HIGH"


def load_asset_prices(calendar: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for asset, ticker in ASSET_TICKERS.items():
        try:
            daily = download_completed_ohlcv(ticker, period="max")
            close = pd.to_numeric(daily.get("Close", pd.Series(dtype="float64")), errors="coerce").dropna()
            close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None)
            weekly = close.sort_index().resample("W-FRI").last().reindex(calendar).ffill()
            out[asset] = weekly
        except Exception:
            out[asset] = pd.Series(np.nan, index=calendar)
    return out


def build_forward_return_stats(history: pd.DataFrame, prices: dict[str, pd.Series], state_col: str, states: list[str]) -> pd.DataFrame:
    rows = []
    state_values = history[state_col].astype(str)
    dates = pd.to_datetime(history["date"], errors="coerce")
    for state in states:
        state_mask = state_values.eq(state)
        for asset, price in prices.items():
            asset_start = pd.Timestamp("2020-04-01") if asset == "BTC" else pd.Timestamp("2010-01-01")
            for horizon, weeks in FORWARD_HORIZONS.items():
                px = num(price).reset_index(drop=True)
                returns = px.shift(-weeks) / px - 1.0
                mask = state_mask & dates.ge(asset_start) & returns.notna()
                sample = returns.loc[mask]
                rows.append(
                    {
                        "Classifier": "Business Cycle" if state_col == "BusinessCycleState" else "Economy Regime",
                        "State": state,
                        "Asset": asset,
                        "Horizon": horizon,
                        "AverageReturn": float(sample.mean()) if not sample.empty else np.nan,
                        "MedianReturn": float(sample.median()) if not sample.empty else np.nan,
                        "HitRate": float((sample > 0).mean()) if not sample.empty else np.nan,
                        "Observations": int(sample.count()),
                        "SampleFlag": "LOW SAMPLE SIZE" if sample.count() < 20 else "OK",
                    }
                )
    return pd.DataFrame(rows)


def build_eta_squared(history: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for classifier, states in [("BusinessCycleState", BUSINESS_PHASES), ("EconomyRegime", ECONOMY_REGIMES)]:
        for asset, price in prices.items():
            start = pd.Timestamp("2020-04-01") if asset == "BTC" else pd.Timestamp("2010-01-01")
            dates = pd.to_datetime(history["date"], errors="coerce")
            px = num(price).reset_index(drop=True)
            for horizon, weeks in FORWARD_HORIZONS.items():
                returns = px.shift(-weeks) / px - 1.0
                valid = pd.DataFrame({"return": returns, "state": history[classifier].astype(str), "date": dates})
                valid = valid.loc[valid["date"].ge(start)].dropna(subset=["return"])
                eta = eta_squared(valid["return"], valid["state"]) if not valid.empty else np.nan
                rows.append({"Classifier": classifier, "Asset": asset, "Horizon": horizon, "EtaSquared": eta, "Observations": int(len(valid))})
    return pd.DataFrame(rows)


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    data = pd.DataFrame({"value": num(values), "group": groups}).dropna()
    if data.empty:
        return np.nan
    grand_mean = data["value"].mean()
    total = ((data["value"] - grand_mean) ** 2).sum()
    if total == 0:
        return np.nan
    between = data.groupby("group")["value"].agg(["count", "mean"])
    ss_between = (between["count"] * (between["mean"] - grand_mean) ** 2).sum()
    return float(ss_between / total)


def build_diagnostics(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["BusinessCycleCandidateState", "BusinessCycleState", "InflationCandidateState", "InflationState", "EconomyRegime"]:
        series = history[column].astype(str)
        transitions = int(series.ne(series.shift(1)).sum()) - 1
        durations = state_durations(series)
        years = max((pd.to_datetime(history["date"]).max() - pd.to_datetime(history["date"]).min()).days / 365.25, 1)
        rows.append(
            {
                "Series": column,
                "Transitions": max(transitions, 0),
                "TransitionsPerYear": max(transitions, 0) / years,
                "MedianDurationWeeks": float(np.median(durations)) if durations else np.nan,
                "StatesLE4Weeks": int(sum(duration <= 4 for duration in durations)),
                "StatesLE8Weeks": int(sum(duration <= 8 for duration in durations)),
            }
        )
    rows.append({"Series": "InflationTransitionZoneShare", "Transitions": np.nan, "TransitionsPerYear": np.nan, "MedianDurationWeeks": float(history["InflationTransitionZone"].mean()), "StatesLE4Weeks": np.nan, "StatesLE8Weeks": np.nan})
    return pd.DataFrame(rows)


def state_durations(series: pd.Series) -> list[int]:
    durations = []
    current = None
    count = 0
    for value in series:
        if value != current:
            if count:
                durations.append(count)
            current = value
            count = 1
        else:
            count += 1
    if count:
        durations.append(count)
    return durations


def build_data_quality(fred: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, description in ALL_FRED_SERIES.items():
        source = fred_series(fred, series_id)
        rows.append(
            {
                "Series": series_id,
                "Description": description,
                "FirstDate": source.index.min().date() if not source.empty else "",
                "LastDate": source.index.max().date() if not source.empty else "",
                "Observations": int(source.count()),
                "Status": "OK" if not source.empty else "MISSING",
            }
        )
    for column in ["BusinessCycleLevel", "BusinessCycleMomentum", "InflationDirectionScore", "RealizedInflationMomentum", "EconomyRegime"]:
        rows.append(
            {
                "Series": column,
                "Description": "Production output",
                "FirstDate": pd.to_datetime(history.loc[history[column].notna(), "date"]).min().date() if history[column].notna().any() else "",
                "LastDate": pd.to_datetime(history.loc[history[column].notna(), "date"]).max().date() if history[column].notna().any() else "",
                "Observations": int(history[column].notna().sum()),
                "Status": "OK" if history[column].notna().any() else "MISSING",
            }
        )
    return pd.DataFrame(rows)


def latest_current(history: pd.DataFrame) -> dict[str, Any]:
    valid = history.dropna(subset=["date"]).copy()
    if valid.empty:
        return {}
    row = valid.iloc[-1]
    return row.to_dict()


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan

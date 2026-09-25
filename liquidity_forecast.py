from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv_fresh, market_business_days_old
from fred_client import download_fred_series
from global_liquidity import GLOBAL_LIQUIDITY_STORAGE_DIR, update_global_liquidity


FORECAST_VERSION = "LIQ_FORECAST_V1"
BREADTH_VERSION = "BREADTH_V1"
SNAPSHOT_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "liquidity_forecast_weekly.csv"
STATUS_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "liquidity_forecast_status.json"
VALIDATION_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "liquidity_forecast_validation.csv"
ERROR_PATH = GLOBAL_LIQUIDITY_STORAGE_DIR / "liquidity_forecast_refresh_error.json"
PRICE_TICKERS = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "BTC": "BTC-USD", "RSP": "RSP", "IWM": "IWM", "MOVE": "^MOVE"}
PRESSURE_FACTORS = {
    "DXY_13W_Change_Pctl": "DXY_13W_Change",
    "MOVE_Pctl": "MOVE",
    "US2Y_Pctl": "US2Y",
    "Inverse_ISM_Pctl": "Inverse_ISM",
    "Inverse_CFNAI_Pctl": "Inverse_CFNAI",
    "ContinuingClaims_13W_Pctl": "ContinuingClaims_13W_Change",
    "TermPremium_26W_Change_Pctl": "US10Y_TermPremium_26W_Change",
}


def trailing_percentile(values: pd.Series, window: int = 156, minimum: int = 52) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")

    def rank_last(sample: np.ndarray) -> float:
        valid = sample[np.isfinite(sample)]
        if len(valid) < minimum or not np.isfinite(sample[-1]):
            return np.nan
        return float(100 * np.count_nonzero(valid <= sample[-1]) / len(valid))

    return numeric.rolling(window, min_periods=minimum).apply(rank_last, raw=True)


def _aligned(series: pd.Series, index: pd.DatetimeIndex, numeric: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=index)
    source = pd.to_numeric(series, errors="coerce") if numeric else series.copy()
    source.index = pd.to_datetime(source.index).tz_localize(None)
    source = source.dropna().sort_index()
    source = source[~source.index.duplicated(keep="last")]
    return source.reindex(source.index.union(index)).sort_index().ffill().reindex(index)


def _band(value: object, low: str, high: str) -> str:
    if pd.isna(value):
        return "DATA_INCOMPLETE"
    if float(value) < 40:
        return low
    if float(value) > 60:
        return high
    return "MID"


def build_forecast_frame(
    regime: pd.DataFrame,
    weekly: pd.DataFrame,
    macro: dict[str, pd.Series],
    prices: dict[str, pd.Series],
    market_history: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if regime.empty:
        return pd.DataFrame()
    today = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz="UTC")).tz_localize(None).normalize()
    last_friday = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
    if as_of is None and today.weekday() == 4:
        last_friday -= pd.Timedelta(days=7)
    base = regime.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    base = base.loc[base["date"] <= last_friday]
    if base.empty:
        return pd.DataFrame()
    frame = base.set_index("date")[["global_liquidity_score", "direction_13w", "cb_impulse", "usnl_impulse", "long_cycle_phase"]].copy()
    frame = frame.rename(columns={
        "global_liquidity_score": "GlobalLiquidityScore",
        "direction_13w": "GlobalLiquidity_Direction_13W",
        "cb_impulse": "CBImpulse",
        "usnl_impulse": "USNLImpulse",
        "long_cycle_phase": "LongLiquidityCyclePhase",
    })
    index = frame.index
    w = weekly.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    source_cols = [
        "US10Y_TermPremium", "US10Y_TermPremium_4W_Change", "US10Y_TermPremium_13W_Change", "US10Y_TermPremium_26W_Change",
        "SOFR", "EFFR", "SOFR_EFFR_Spread", "SOFR_EFFR_4W_Change", "SOFR_EFFR_13W_Change",
        "US_BankReserves", "US_BankReserves_4W_Change", "US_BankReserves_13W_Change", "US_BankReserves_26W_Change",
        "US_BankReserves_4W_PctChange", "US_BankReserves_13W_PctChange", "US_BankReserves_26W_PctChange", "dxy",
    ]
    for col in source_cols:
        frame[col] = _aligned(w[col], index) if col in w else np.nan
    frame["DXY_13W_Change"] = frame["dxy"].pct_change(13, fill_method=None) * 100

    for name in PRICE_TICKERS:
        values = prices.get(name, pd.Series(dtype="float64"))
        weekly_price = values.resample("W-FRI").last() if not values.empty else values
        frame[f"{name}_Close"] = _aligned(weekly_price, index)
    for name in ("RSP", "IWM"):
        ratio = frame[f"{name}_Close"] / frame["SPY_Close"].replace(0, np.nan)
        frame[f"{name}_SPY"] = ratio
        frame[f"{name}_SPY_13W_ChangePct"] = ratio.pct_change(13, fill_method=None) * 100
        frame[f"{name}_SPY_13W_Percentile"] = trailing_percentile(frame[f"{name}_SPY_13W_ChangePct"])

    frame["MOVE"] = frame["MOVE_Close"]
    for name in ("DGS2", "NAPM", "CFNAI", "CCSA"):
        series = macro.get(name, pd.Series(dtype="float64"))
        if not series.empty:
            series = series.copy()
            series.index = pd.to_datetime(series.index)
            if name in {"NAPM", "CFNAI"}:
                series.index = series.index + pd.offsets.MonthEnd(2)
            elif name == "CCSA":
                series.index = series.index + pd.Timedelta(days=7)
            else:
                series.index = series.index + pd.offsets.BDay(1)
        frame[name] = _aligned(series, index)
    frame["US2Y"] = frame["DGS2"]
    frame["ISM_Manufacturing"] = frame["NAPM"]
    frame["ContinuingClaims_13W_Change"] = frame["CCSA"].diff(13)
    frame["Inverse_ISM"] = -frame["ISM_Manufacturing"]
    frame["Inverse_CFNAI"] = -frame["CFNAI"]
    for out_col, in_col in PRESSURE_FACTORS.items():
        frame[out_col] = trailing_percentile(frame[in_col])
    frame["LiquidityPressureScore"] = frame[list(PRESSURE_FACTORS)].mean(axis=1, skipna=False)
    frame["LiquidityPressureBand"] = frame["LiquidityPressureScore"].map(lambda x: _band(x, "LOW", "HIGH"))
    frame["BankReservesImpulse"] = trailing_percentile(frame["US_BankReserves_13W_PctChange"])
    frame["PolicyResponseScore"] = frame[["CBImpulse", "USNLImpulse", "BankReservesImpulse"]].mean(axis=1, skipna=False)
    frame["PolicyResponseBand"] = frame["PolicyResponseScore"].map(lambda x: _band(x, "WEAK", "STRONG"))

    positive = frame["LiquidityPressureBand"].eq("HIGH") & frame["PolicyResponseBand"].eq("WEAK")
    negative = frame["LiquidityPressureBand"].eq("LOW") & frame["PolicyResponseBand"].eq("STRONG")
    complete = ~frame[["LiquidityPressureScore", "PolicyResponseScore", "GlobalLiquidity_Direction_13W"]].isna().any(axis=1)
    frame["LiquidityForwardSignal"] = np.select([positive, negative], ["POSITIVE", "NEGATIVE"], default="NEUTRAL")
    frame.loc[~complete, "LiquidityForwardSignal"] = "DATA_INCOMPLETE"
    direction = frame["GlobalLiquidity_Direction_13W"]
    signal = frame["LiquidityForwardSignal"]
    frame["LiquidityForecastState"] = np.select(
        [
            (direction <= 0) & signal.eq("POSITIVE"),
            (direction >= 0) & signal.eq("NEGATIVE"),
            (direction > 0) & ~signal.eq("NEGATIVE"),
            (direction < 0) & ~signal.eq("POSITIVE"),
        ],
        ["BOTTOMING", "PEAKING", "EXPANSION", "CONTRACTION"],
        default="NEUTRAL",
    )
    frame.loc[~complete, "LiquidityForecastState"] = "DATA_INCOMPLETE"

    rsp = frame["RSP_SPY_13W_Percentile"]
    iwm = frame["IWM_SPY_13W_Percentile"]
    frame["BreadthParticipationState"] = np.select(
        [(rsp >= 60) & (iwm >= 60), (rsp <= 40) & (iwm <= 40)],
        ["BROADENING", "NARROWING"], default="MIXED",
    )
    frame.loc[rsp.isna() | iwm.isna(), "BreadthParticipationState"] = "DATA_INCOMPLETE"
    frame["RiskOnConfirmation"] = frame["LiquidityForecastState"].eq("BOTTOMING") & frame["BreadthParticipationState"].eq("BROADENING")

    history = market_history.copy()
    if not history.empty:
        date_col = "Date" if "Date" in history else "date"
        history[date_col] = pd.to_datetime(history[date_col], errors="coerce")
        history = history.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)
    market_cols = {
        "Market_Regime": "StructuralMarketRegime",
        "Fast_Transition_Risk": "FastTransitionRisk",
        "Macro_Transition_Risk": "MacroTransitionRisk",
        "Credit_Risk": "CreditRisk",
        "TailRiskFlag": "TailRiskFlag",
    }
    for source, target in market_cols.items():
        frame[target] = _aligned(history[source], index, numeric=source not in {"Market_Regime", "TailRiskFlag"}) if source in history else np.nan
    regime_bull = frame["StructuralMarketRegime"].isin(["BULL", "BULL_HIGH_VOL"])
    confirmed = frame["RiskOnConfirmation"] & regime_bull & (pd.to_numeric(frame["MacroTransitionRisk"], errors="coerce") < 40) & (pd.to_numeric(frame["FastTransitionRisk"], errors="coerce") < 40)
    frame["RiskOnState"] = np.select([confirmed, frame["LiquidityForecastState"].eq("BOTTOMING")], ["CONFIRMED", "EARLY"], default="NONE")
    peaking = frame["LiquidityForecastState"].eq("PEAKING")
    narrowing = frame["BreadthParticipationState"].eq("NARROWING")
    high = peaking & narrowing & ((pd.to_numeric(frame["FastTransitionRisk"], errors="coerce") >= 40) | (pd.to_numeric(frame["CreditRisk"], errors="coerce") >= 60) | frame["StructuralMarketRegime"].isin(["CORRECTION", "STRESS"]))
    frame["RiskReductionWarning"] = np.select([high, peaking & narrowing, peaking], ["HIGH", "ELEVATED", "WATCH"], default="NONE")
    frame["FundingStressContext"] = "DATA_INCOMPLETE"
    funding = frame["SOFR_EFFR_Spread"]
    frame.loc[funding.notna(), "FundingStressContext"] = "NORMAL"
    elevated_funding = funding.notna() & ((funding.abs() > 0.10) | (frame["MOVE_Pctl"] > 80) | (frame["TermPremium_26W_Change_Pctl"] > 80))
    frame.loc[elevated_funding, "FundingStressContext"] = "ELEVATED"
    frame["LiquidityForecastModelVersion"] = FORECAST_VERSION
    frame["BreadthModelVersion"] = BREADTH_VERSION
    frame["Future_GLS_13W_Change"] = frame["GlobalLiquidityScore"].shift(-13) - frame["GlobalLiquidityScore"]
    frame["Future_GLS_26W_Change"] = frame["GlobalLiquidityScore"].shift(-26) - frame["GlobalLiquidityScore"]
    for asset in ("SPY", "QQQ", "GLD", "BTC"):
        price = frame[f"{asset}_Close"]
        for horizon in (13, 26):
            frame[f"{asset}_FutureReturn_{horizon}W"] = (price.shift(-horizon) / price - 1) * 100
            future_path = pd.concat([price.shift(-week) for week in range(1, horizon + 1)], axis=1)
            frame[f"{asset}_FutureWorstPath_{horizon}W"] = (future_path.min(axis=1) / price - 1) * 100
            frame.loc[price.shift(-horizon).isna(), f"{asset}_FutureWorstPath_{horizon}W"] = np.nan
    frame = frame.loc[frame.index >= pd.Timestamp("2016-01-01")]
    return frame.reset_index().rename(columns={"date": "Date"})


def read_forecast_snapshot() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(SNAPSHOT_PATH, parse_dates=["Date"]) if SNAPSHOT_PATH.exists() else pd.DataFrame()
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    if ERROR_PATH.exists():
        try:
            status["LatestRefreshError"] = json.loads(ERROR_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status["LatestRefreshError"] = {"Reason": "Failed refresh; previous snapshot retained"}
    return frame, status


def build_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("EXPANSION", "PEAKING", "CONTRACTION", "BOTTOMING"):
        sample = frame.loc[frame["LiquidityForecastState"].eq(state)]
        direction = 1 if state in {"EXPANSION", "BOTTOMING"} else -1
        row = {"Analysis": "Liquidity State", "Group": state, "Asset": "GLS", "Threshold": 60}
        for horizon in (13, 26):
            valid = pd.to_numeric(sample[f"Future_GLS_{horizon}W_Change"], errors="coerce").dropna()
            row[f"N_{horizon}W"] = len(valid)
            row[f"Avg_{horizon}W"] = valid.mean()
            row[f"HitRate_{horizon}W"] = (valid * direction > 0).mean() * 100 if len(valid) else np.nan
        rows.append(row)
    for threshold in (50, 60, 70):
        bottoming = frame["LiquidityForecastState"].eq("BOTTOMING")
        broadening = (frame["RSP_SPY_13W_Percentile"] >= threshold) & (frame["IWM_SPY_13W_Percentile"] >= threshold)
        for group, mask in (("BOTTOMING", bottoming), ("BOTTOMING+BROADENING", bottoming & broadening)):
            for asset in ("SPY", "QQQ", "GLD", "BTC"):
                row = {"Analysis": "Breadth Confirmation", "Group": group, "Asset": asset, "Threshold": threshold}
                subset = frame.loc[mask]
                for horizon in (13, 26):
                    returns = pd.to_numeric(subset[f"{asset}_FutureReturn_{horizon}W"], errors="coerce").dropna()
                    worst = pd.to_numeric(subset[f"{asset}_FutureWorstPath_{horizon}W"], errors="coerce").reindex(returns.index)
                    row[f"N_{horizon}W"] = len(returns)
                    row[f"Avg_{horizon}W"] = returns.mean()
                    row[f"Median_{horizon}W"] = returns.median()
                    row[f"HitRate_{horizon}W"] = (returns > 0).mean() * 100 if len(returns) else np.nan
                    row[f"AvgWorstPath_{horizon}W"] = worst.mean()
                    row[f"Drawdown10Prob_{horizon}W"] = (worst <= -10).mean() * 100 if len(worst) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def refresh_forecast_snapshot(api_key: str | None = None) -> tuple[pd.DataFrame, dict]:
    from macro_research_export import build_global_liquidity_regime_frame_for_export, build_market_transition_history_for_export

    raw, monthly, weekly = update_global_liquidity(api_key=api_key, force=True)
    regime = build_global_liquidity_regime_frame_for_export(monthly, weekly)
    macro = {}
    source_status = {}
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    def freshness(series: pd.Series, max_age_days: int, market: bool = False) -> str:
        if series.empty:
            return "EMPTY"
        if market:
            age = market_business_days_old(series)
            return "CURRENT" if age is not None and age <= max_age_days else "STALE"
        latest = pd.to_datetime(series.index, errors="coerce").max()
        return "CURRENT" if pd.notna(latest) and (today - latest).days <= max_age_days else "STALE"

    from business_cycle import load_tradingview_economic_fallback

    market_sources: dict = {}
    try:
        market = build_market_transition_history_for_export(api_key, source_sink=market_sources)
        source_status["MarketRegime"] = freshness(pd.Series(index=pd.to_datetime(market["Date"]), dtype="float64"), 21) if not market.empty else "EMPTY"
    except Exception as exc:
        market = pd.DataFrame()
        source_status["MarketRegime"] = f"ERROR: {type(exc).__name__}"

    for name in ("DGS2", "NAPM", "CFNAI", "CCSA"):
        try:
            if name == "NAPM":
                fetched = load_tradingview_economic_fallback("NAPM", "2010-01-01")
            elif name == "DGS2" and not market_sources.get("fred_data", pd.DataFrame()).empty:
                source = market_sources["fred_data"]
                fetched = source.loc[source["Series_ID"].eq("DGS2"), ["Date", "Value"]]
                if fetched.empty:
                    fetched = download_fred_series(name, api_key=api_key, observation_start="2010-01-01")
            else:
                fetched = download_fred_series(name, api_key=api_key, observation_start="2010-01-01")
            macro[name] = pd.Series(pd.to_numeric(fetched["Value"], errors="coerce").to_numpy(), index=pd.to_datetime(fetched["Date"])).dropna()
            source_status[name] = freshness(macro[name], 90 if name in {"NAPM", "CFNAI"} else 21)
        except Exception as exc:
            macro[name] = pd.Series(dtype="float64")
            source_status[name] = f"ERROR: {type(exc).__name__}"
    prices = {}
    for name, ticker in PRICE_TICKERS.items():
        try:
            bars = market_sources.get("yahoo_weekly", {}).get(ticker, pd.DataFrame())
            if bars.empty:
                bars, _ = download_completed_ohlcv_fresh(
                    ticker, period="max", max_business_days_old=7,
                )
            prices[name] = pd.to_numeric(bars["Close"], errors="coerce").dropna()
            source_status[name] = freshness(prices[name], 7, market=True)
        except Exception as exc:
            prices[name] = pd.Series(dtype="float64")
            source_status[name] = f"ERROR: {type(exc).__name__}"
    for name in ("THREEFYTP10", "SOFR", "EFFR", "WRESBAL"):
        observations = raw.loc[raw["series_id"].eq(name) & raw["raw_value"].notna(), "observation_date"]
        source_status[name] = freshness(pd.Series(index=pd.to_datetime(observations, errors="coerce"), dtype="float64"), 21)
    failed = [name for name, state in source_status.items() if state != "CURRENT"]
    if failed and SNAPSHOT_PATH.exists():
        raise RuntimeError(f"Liquidity forecast source refresh failed ({', '.join(failed)}); retained the previous successful snapshot")
    frame = build_forecast_frame(regime, weekly, macro, prices, market)
    required = ["LiquidityPressureScore", "PolicyResponseScore", "RSP_SPY_13W_Percentile", "IWM_SPY_13W_Percentile"]
    if frame.empty or not frame.tail(8)[required].notna().all(axis=1).any():
        raise RuntimeError("Liquidity forecast sources are incomplete; retained the previous successful snapshot")
    status = {
        "ModelVersion": FORECAST_VERSION,
        "DataAsOf": str(pd.Timestamp(frame["Date"].max()).date()),
        "CalculatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "TimingConvention": "Trailing weekly windows with conservative release lags; existing GLS monthly timing and revised FRED histories are not vintage point-in-time",
        "SourceStatus": source_status,
        "PopulatedForecastWeeks": int(frame["LiquidityPressureScore"].notna().sum()),
    }
    validation = build_validation_frame(frame)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = SNAPSHOT_PATH.with_suffix(".csv.tmp")
    tmp_json = STATUS_PATH.with_suffix(".json.tmp")
    tmp_validation = VALIDATION_PATH.with_suffix(".csv.tmp")
    frame.to_csv(tmp_csv, index=False)
    validation.to_csv(tmp_validation, index=False)
    tmp_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp_csv, SNAPSHOT_PATH)
    os.replace(tmp_validation, VALIDATION_PATH)
    os.replace(tmp_json, STATUS_PATH)
    return frame, status

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

from correction_bottom import calculate_correction_bottom_indicator
from current_risk import CURRENT_RISK_MODEL_VERSION, calculate_current_risk_v1
from fred_client import FredApiError, download_fred_series
from hy_oas import combine_hy_oas_sources, weekly_archive_available_frame
from market_model import calculate_positioning_risk_history, market_model_config
from positioning import read_processed


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

MARKET_CYCLE_MODEL_VERSION = "MARKET_CYCLE_V1"
MARKET_CYCLE_TICKERS = ("^GSPC", "SPY", "^VIX", "^VIX3M", "QQQ", "GLD", "BTC-USD", "RSP", "IWM", "XLI", "XLP")
MARKET_CYCLE_SPX_REFERENCE_PATH = Path(__file__).with_name("data") / "SPCFD_SPX_1W_reference.csv"
CURRENT_RISK_RAW_TICKERS = {"SPY": "SPY_RAW", "QQQ": "QQQ_RAW"}
MARKET_CYCLE_TRADINGVIEW_BREADTH = {"S5FI": "SPXAboveSMA50D", "S5TH": "SPXAboveSMA200D"}
FORWARD_HORIZONS = {"3M": 13, "6M": 26, "12M": 52}
ANALOG_FEATURE_WEIGHTS = {
    "StructuralSimilarity": 0.15,
    "MomentumSimilarity": 0.30,
    "SMA200WSimilarity": 0.25,
    "CurrentRiskSimilarity": 0.25,
    "PositioningSimilarity": 0.05,
}
ANALOG_MIN_EPISODE_SPACING_WEEKS = 13


@dataclass
class MarketCycleSnapshot:
    history: pd.DataFrame
    monthly: pd.DataFrame
    daily: pd.DataFrame
    correction_daily: pd.DataFrame
    current: dict[str, Any]
    outlook: pd.DataFrame
    asset_outlook: pd.DataFrame
    analogs: pd.DataFrame
    category_validation: pd.DataFrame
    data_quality: pd.DataFrame
    interpretation: str


def build_market_cycle_snapshot(end_date: str | pd.Timestamp | None = None) -> MarketCycleSnapshot:
    end = pd.Timestamp(end_date).tz_localize(None).normalize() if end_date else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    raw = download_market_cycle_prices(end)
    daily = build_daily_frame(raw, end)
    weekly = build_weekly_frame(raw, end)
    monthly = build_monthly_frame(raw, end)
    structural = build_structural_monthly(monthly)
    momentum = build_momentum_monthly(monthly)
    weekly_history = build_weekly_history(weekly, daily, structural, momentum)
    add_positioning(weekly_history)
    correction_daily = build_daily_correction_history(daily, weekly_history)
    add_vulnerability(weekly_history)
    add_analog_engine(weekly_history)
    current = overlay_latest_daily_current_risk(latest_current(weekly_history), daily)
    current.update(calculate_current_spx_performance(daily))
    correction_current = latest_current(correction_daily)
    for key, value in correction_current.items():
        if str(key).startswith(("Correction", "Extreme", "Tactical", "Durable", "BearRisk", "LatestDurable", "PostTactical", "Wave", "SPYVolume", "HYOAS")) or key in {
            "SPY_Close",
            "SPY_Low",
            "SPY_Volume",
            "SPY_RSI14",
            "SPY_5D_Return",
            "SPY_SMA200D",
            "SMA200D_Slope20D",
            "SPY_ROC3M",
            "SPY_ROC6M",
            "HY_OAS",
            "VIX",
            "VIX_Stress",
            "VIXRetreat20D",
            "RSI_Stress",
            "SPY5D_DownsideStress",
            "Breadth50_Stress",
            "Breadth200_Stress",
            "Breadth50_5D_Change",
            "Breadth200_5D_Change",
            "SPXAboveSMA50D",
            "SPXAboveSMA200D",
            "BreadthSourceFrequency",
            "VolumeConfirmation",
            "PriceMomentumExhaustion",
            "VolatilityExhaustion",
            "BreadthExhaustion",
            "CreditPositioningExhaustion",
            "RiskAppetiteExhaustion",
            "PriceRecoveryScore",
            "VIXRecoveryScore",
            "BreadthRecoveryScore",
            "RiskAppetiteRecoveryScore",
            "BreadthWashout",
            "BreadthRecovery",
            "VIXPeakSignal",
            "VIXTermStructureNormalization",
            "HYOASDivergence",
            "CFTCDeleveragingState",
        }:
            current[key] = value
    outlook, analogs = build_historical_outlook(weekly_history, current)
    asset_outlook = build_asset_historical_outlook(analogs)
    category_validation = build_category_validation(weekly_history, current)
    quality = build_data_quality(raw, weekly_history, monthly)
    interpretation = build_interpretation(current, outlook)
    return MarketCycleSnapshot(
        history=weekly_history,
        monthly=monthly,
        daily=daily,
        correction_daily=correction_daily,
        current=current,
        outlook=outlook,
        asset_outlook=asset_outlook,
        analogs=analogs,
        category_validation=category_validation,
        data_quality=quality,
        interpretation=interpretation,
    )


def download_market_cycle_prices(end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker in MARKET_CYCLE_TICKERS:
        try:
            raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
            frame = extract_ohlcv(raw, ticker)
            if not frame.empty:
                frame = frame.loc[pd.to_datetime(frame.index, errors="coerce").tz_localize(None) <= end].copy()
            out[ticker] = frame
        except Exception:
            out[ticker] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    for ticker, output_key in CURRENT_RISK_RAW_TICKERS.items():
        try:
            raw = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
            frame = extract_ohlcv(raw, ticker)
            if not frame.empty:
                frame = frame.loc[pd.to_datetime(frame.index, errors="coerce").tz_localize(None) <= end].copy()
            out[output_key] = frame
        except Exception:
            out[output_key] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    for symbol in MARKET_CYCLE_TRADINGVIEW_BREADTH:
        out[symbol] = load_tradingview_breadth_series(symbol, end)
    return out


def load_tradingview_breadth_series(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    frame = load_tradingview_ohlcv_frame(f"INDEX:{symbol}", end)
    if not frame.empty:
        return frame

    try:
        from tradingview_mcp import get_economic_data, validate_economic_result

        result = get_economic_data(symbol, date_from="1990-01-01")
        valid, _ = validate_economic_result(result, min_observations=52, max_stale_days=30)
        if not valid or result.frame.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        dates = pd.to_datetime(result.frame["date"], errors="coerce")
        values = pd.to_numeric(result.frame["value"], errors="coerce")
        frame = pd.DataFrame({"Close": values.to_numpy()}, index=dates)
        frame = frame.dropna(subset=["Close"]).sort_index()
        if frame.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None)
        frame["Open"] = frame["Close"]
        frame["High"] = frame["Close"]
        frame["Low"] = frame["Close"]
        frame["Volume"] = np.nan
        return frame.loc[frame.index <= end, ["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def load_tradingview_ohlcv_frame(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    try:
        from tradingview_mcp import call_tool

        payload = call_tool(
            "get_ohlcv",
            {"symbol": symbol, "interval": "1D", "count": 5000, "summary": False},
        )
        bars = payload.get("bars") or payload.get("data") or payload.get("candles") or []
        rows = []
        for bar in bars:
            if not isinstance(bar, dict):
                continue
            timestamp = pd.to_datetime(bar.get("t"), unit="s", errors="coerce", utc=True)
            close = pd.to_numeric(bar.get("c"), errors="coerce")
            if pd.isna(timestamp) or not np.isfinite(safe_float(close)):
                continue
            rows.append(
                {
                    "Date": timestamp.tz_localize(None).normalize(),
                    "Open": pd.to_numeric(bar.get("o"), errors="coerce"),
                    "High": pd.to_numeric(bar.get("h"), errors="coerce"),
                    "Low": pd.to_numeric(bar.get("l"), errors="coerce"),
                    "Close": close,
                    "Volume": pd.to_numeric(bar.get("v"), errors="coerce"),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date", keep="last")
        frame = frame.set_index("Date")
        return frame.loc[frame.index <= end, ["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def load_spx_reference_weekly(end: pd.Timestamp) -> pd.DataFrame:
    """Load the supplied TradingView weekly SPX baseline used by cycle validation."""
    if not MARKET_CYCLE_SPX_REFERENCE_PATH.exists():
        return pd.DataFrame(columns=["Date", "SPX_Close", "SPX_Low"])
    try:
        source = pd.read_csv(MARKET_CYCLE_SPX_REFERENCE_PATH, usecols=["time", "low", "close"])
        dates = pd.to_datetime(source["time"], errors="coerce")
        out = pd.DataFrame(
            {
                "Date": dates,
                "SPX_Close": pd.to_numeric(source["close"], errors="coerce"),
                "SPX_Low": pd.to_numeric(source["low"], errors="coerce"),
            }
        )
        return (
            out.dropna(subset=["Date", "SPX_Close"])
            .loc[lambda frame: frame["Date"].ge(pd.Timestamp("1920-01-01")) & frame["Date"].le(end)]
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(columns=["Date", "SPX_Close", "SPX_Low"])


def extract_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        try:
            if ticker in frame.columns.get_level_values(0):
                frame = frame.xs(ticker, axis=1, level=0)
            elif ticker in frame.columns.get_level_values(1):
                frame = frame.xs(ticker, axis=1, level=1)
        except Exception:
            pass
    keep = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame.columns]
    if not keep:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = frame[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    return out.dropna(subset=["Close"]).sort_index()


def build_daily_frame(raw: dict[str, pd.DataFrame], end: pd.Timestamp) -> pd.DataFrame:
    spx = raw.get("^GSPC", pd.DataFrame()).copy()
    if spx.empty:
        return pd.DataFrame()
    d = pd.DataFrame(index=spx.index)
    d["SPX_Close"] = pd.to_numeric(spx["Close"], errors="coerce")
    d["SPX_Low"] = pd.to_numeric(spx.get("Low", spx["Close"]), errors="coerce")
    spy = raw.get("SPY", pd.DataFrame()).copy()
    if not spy.empty:
        d["SPY_Close"] = pd.to_numeric(spy.get("Close"), errors="coerce").reindex(d.index)
        d["SPY_Low"] = pd.to_numeric(spy.get("Low", spy.get("Close")), errors="coerce").reindex(d.index)
        d["SPY_Volume"] = pd.to_numeric(spy.get("Volume"), errors="coerce").reindex(d.index)
    else:
        d["SPY_Close"] = np.nan
        d["SPY_Low"] = np.nan
        d["SPY_Volume"] = np.nan
    spy_raw = raw.get("SPY_RAW", pd.DataFrame()).copy()
    if not spy_raw.empty:
        d["SPY_RawClose"] = pd.to_numeric(spy_raw.get("Close"), errors="coerce").reindex(d.index)
    else:
        d["SPY_RawClose"] = np.nan
    qqq_raw = raw.get("QQQ_RAW", pd.DataFrame()).copy()
    if not qqq_raw.empty:
        d["QQQ_RawClose"] = pd.to_numeric(qqq_raw.get("Close"), errors="coerce").reindex(d.index)
    else:
        d["QQQ_RawClose"] = np.nan
    for ticker, col in [("^VIX", "VIX"), ("^VIX3M", "VIX3M"), ("QQQ", "QQQ"), ("RSP", "RSP"), ("IWM", "IWM")]:
        series = pd.to_numeric(raw.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype="float64")), errors="coerce")
        d[col] = series.reindex(d.index).ffill()
    for symbol, col in MARKET_CYCLE_TRADINGVIEW_BREADTH.items():
        series = pd.to_numeric(raw.get(symbol, pd.DataFrame()).get("Close", pd.Series(dtype="float64")), errors="coerce")
        aligned = series.reindex(d.index)
        d[f"{col}_Observed"] = aligned.notna()
        d[col] = aligned.ffill().clip(lower=0.0, upper=100.0)
    d = d.loc[d.index <= end].copy()
    returns = d["SPX_Close"].pct_change(fill_method=None)
    d["SPX_RealizedVol20D"] = returns.rolling(20, min_periods=15).std() * np.sqrt(252.0)
    d["RealizedVol20D"] = d["SPX_RealizedVol20D"]
    d["SPX_DrawdownFromATH"] = d["SPX_Close"] / d["SPX_Close"].cummax() - 1.0
    d["VIXTermStructure"] = d["VIX"] / d["VIX3M"]
    d["RSP_SPX_4W"] = (d["RSP"] / d["SPX_Close"]).pct_change(20, fill_method=None)
    d["IWM_SPX_4W"] = (d["IWM"] / d["SPX_Close"]).pct_change(20, fill_method=None)
    return d


def build_weekly_frame(raw: dict[str, pd.DataFrame], end: pd.Timestamp) -> pd.DataFrame:
    close = {}
    low = {}
    for ticker, frame in raw.items():
        if frame.empty:
            close[ticker] = pd.Series(dtype="float64")
            low[ticker] = pd.Series(dtype="float64")
        else:
            close[ticker] = pd.to_numeric(frame["Close"], errors="coerce").resample("W-FRI").last()
            low[ticker] = pd.to_numeric(frame["Low"], errors="coerce").resample("W-FRI").min() if "Low" in frame else pd.Series(dtype="float64")
    reference = load_spx_reference_weekly(end)
    if reference.empty:
        spx = close.get("^GSPC", pd.Series(dtype="float64")).dropna()
        w = pd.DataFrame({"Date": spx.index, "SPX_Close": spx.values})
        w["SPX_Low"] = low.get("^GSPC", pd.Series(dtype="float64")).reindex(spx.index).ffill().to_numpy()
    else:
        w = reference.copy()
        live_spx = close.get("^GSPC", pd.Series(dtype="float64")).dropna()
        live_dates = live_spx.index[live_spx.index > pd.Timestamp(w["Date"].max())]
        if len(live_dates):
            live = pd.DataFrame(
                {
                    "Date": live_dates,
                    "SPX_Close": live_spx.reindex(live_dates).to_numpy(),
                    "SPX_Low": low.get("^GSPC", pd.Series(dtype="float64")).reindex(live_dates).to_numpy(),
                }
            )
            w = pd.concat([w, live], ignore_index=True)
    target_dates = pd.DatetimeIndex(w["Date"])

    def align_to_target(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return np.full(len(target_dates), np.nan)
        source = values.rename("value").rename_axis("Date").reset_index().sort_values("Date")
        target = pd.DataFrame({"Date": target_dates}).sort_values("Date")
        aligned = pd.merge_asof(target, source, on="Date", direction="backward")
        return aligned["value"].to_numpy()

    for ticker, col in [("SPY", "SPY"), ("^VIX", "VIX"), ("^VIX3M", "VIX3M"), ("QQQ", "QQQ"), ("GLD", "GLD"), ("BTC-USD", "BTC"), ("RSP", "RSP"), ("IWM", "IWM"), ("XLI", "XLI"), ("XLP", "XLP")]:
        w[col] = align_to_target(close.get(ticker, pd.Series(dtype="float64")))
    for symbol, col in MARKET_CYCLE_TRADINGVIEW_BREADTH.items():
        w[col] = np.clip(align_to_target(close.get(symbol, pd.Series(dtype="float64"))), 0.0, 100.0)
    return w.loc[w["Date"].le(end)].reset_index(drop=True)


def build_monthly_frame(raw: dict[str, pd.DataFrame], end: pd.Timestamp) -> pd.DataFrame:
    spx = raw.get("^GSPC", pd.DataFrame()).copy()
    if spx.empty:
        return pd.DataFrame(columns=["Date", "SPX_Close"])
    close = pd.to_numeric(spx["Close"], errors="coerce").resample("ME").last().dropna()
    return pd.DataFrame({"Date": close.index, "SPX_Close": close.values}).loc[lambda x: x["Date"].le(end)].reset_index(drop=True)


def calculate_current_spx_performance(daily: pd.DataFrame) -> dict[str, float]:
    """Match TradingView monthly ROC using the latest available monthly bar."""
    if daily.empty or "SPX_Close" not in daily:
        return {}
    close = pd.to_numeric(daily["SPX_Close"], errors="coerce").dropna()
    if close.empty:
        return {}
    monthly_close = close.resample("ME").last().dropna()
    current: dict[str, float] = {}
    for months in [1, 3, 6, 12]:
        if len(monthly_close) > months:
            current[f"PerformanceROC{months}M"] = float(monthly_close.iloc[-1] / monthly_close.iloc[-1 - months] - 1.0)
    return current


def overlay_latest_daily_current_risk(current: dict[str, Any], daily: pd.DataFrame) -> dict[str, Any]:
    """Overlay the latest daily Current Risk state without replacing weekly cycle context."""
    result = dict(current)
    if daily.empty:
        return result
    d = daily.copy()
    d.index.name = None
    if "Date" not in d:
        d["Date"] = pd.to_datetime(d.index, errors="coerce")
    else:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    if d.empty:
        return result
    latest = d.iloc[-1].to_dict()
    passthrough = {
        "CurrentMarketRiskState",
        "CurrentMarketRiskDirection",
        "ActiveStressChannels",
        "VIX",
        "SPXAboveSMA50D",
        "SPXAboveSMA200D",
        "HY_OAS",
        "HYOASSourceFrequency",
    }
    for key, value in latest.items():
        name = str(key)
        if name.startswith(("CurrentRisk", "PVC_", "HistoricalCurrentRisk", "HistoricalDrawdown", "HistoricalHighBeta", "HistoricalBreadth", "HistoricalRSI", "HistoricalVIX")) or name in passthrough:
            result[name] = value
    result["CurrentRiskAsOfDate"] = pd.Timestamp(d.iloc[-1]["Date"])
    result["CurrentRiskFrequency"] = "DAILY"
    return result


def build_structural_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    if m.empty:
        return m
    close = pd.to_numeric(m["SPX_Close"], errors="coerce")
    m["SPX_SMA200M"] = close.rolling(200, min_periods=120).mean()
    m["StructuralExtensionPct"] = close / m["SPX_SMA200M"] - 1.0
    m["StructuralExtensionPercentile"] = expanding_percentile(m["StructuralExtensionPct"], min_periods=120)
    m["StructuralExtensionZone"] = m["StructuralExtensionPercentile"].map(classify_sma200w_zone)
    m["SPX_ROC36M"] = close.pct_change(36, fill_method=None)
    m["SPX_ROC36M_3MMA"] = m["SPX_ROC36M"].rolling(3, min_periods=2).mean()
    for months in [1, 3, 6, 12]:
        m[f"StructuralROCMomentum_{months}M"] = m["SPX_ROC36M_3MMA"] - m["SPX_ROC36M_3MMA"].shift(months)
    m["StructuralROCMomentum"] = m["StructuralROCMomentum_12M"]
    m["StructuralROCMomentumState"] = m["StructuralROCMomentum"].map(classify_structural_roc_momentum)
    m["StructuralMomentumCycleCandidate"] = [
        classify_momentum_phase(level, momentum)
        for level, momentum in zip(m["SPX_ROC36M_3MMA"], m["StructuralROCMomentum"])
    ]
    m["StructuralMomentumCyclePhase"] = confirm_state(m["StructuralMomentumCycleCandidate"], required=2)
    bottom_dates = detect_structural_bottoms(m)
    median_duration = 291.0
    m["LastStructuralBottomDate"] = pd.NaT
    for idx, date in enumerate(pd.to_datetime(m["Date"], errors="coerce")):
        prior = [bottom for bottom in bottom_dates if bottom <= date]
        if prior:
            m.loc[idx, "LastStructuralBottomDate"] = prior[-1]
    last_bottom = pd.to_datetime(m["LastStructuralBottomDate"], errors="coerce")
    m["StructuralCycleAgeMonths"] = ((pd.to_datetime(m["Date"]) - last_bottom).dt.days / 30.4375).where(last_bottom.notna())
    m["HistoricalStructuralExpansionMedianMonths"] = median_duration
    m["HistoricalStructuralExpansionMin"] = 285.0
    m["HistoricalStructuralExpansionMax"] = 297.0
    m["StructuralAgePctOfHistoricalMedian"] = m["StructuralCycleAgeMonths"] / median_duration
    m["StructuralExtensionDirection_12M"] = m["StructuralExtensionPct"].diff(12)
    m["StructuralMarketCyclePhase"] = [
        classify_structural_phase(row) for row in m.to_dict("records")
    ]
    m["Structural_SMA_ROC_Candidate"] = [
        classify_structural_sma_roc_phase(row) for row in m.to_dict("records")
    ]
    m["Structural_SMA_ROC_Phase"] = confirm_state(m["Structural_SMA_ROC_Candidate"], required=2)
    m["StructuralDataStatus"] = np.where(m["StructuralExtensionPercentile"].notna() & m["SPX_ROC36M_3MMA"].notna(), "FULL", "PARTIAL")
    return m


def detect_structural_bottoms(m: pd.DataFrame) -> list[pd.Timestamp]:
    candidates: list[pd.Timestamp] = []
    ext = pd.to_numeric(m["StructuralExtensionPct"], errors="coerce")
    pctl = pd.to_numeric(m["StructuralExtensionPercentile"], errors="coerce")
    close = pd.to_numeric(m["SPX_Close"], errors="coerce")
    fwd_12m = close.shift(-12) / close - 1.0
    for idx in range(24, len(m) - 12):
        if not np.isfinite(ext.iloc[idx]) or not np.isfinite(pctl.iloc[idx]):
            continue
        local_low = close.iloc[idx] <= close.iloc[idx - 12 : idx + 13].min()
        depressed = ext.iloc[idx] < -0.20 or pctl.iloc[idx] <= 10
        recovery = fwd_12m.iloc[idx] > 0.15
        if local_low and depressed and recovery:
            date = pd.Timestamp(m["Date"].iloc[idx])
            if not candidates or (date - candidates[-1]).days >= 3650:
                candidates.append(date)
    anchors = [pd.Timestamp("1932-06-30"), pd.Timestamp("1974-12-31"), pd.Timestamp("2009-03-31")]
    available_start = pd.Timestamp(m["Date"].min()) if not m.empty else pd.Timestamp("1900-01-01")
    for anchor in anchors:
        if anchor >= available_start and all(abs((anchor - c).days) > 365 * 5 for c in candidates):
            candidates.append(anchor)
    return sorted(candidates)


def build_momentum_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly[["Date", "SPX_Close"]].copy()
    if m.empty:
        return m
    close = pd.to_numeric(m["SPX_Close"], errors="coerce")
    for months in [1, 3, 6, 12]:
        m[f"SPX_ROC{months}M"] = close.pct_change(months, fill_method=None)
    m["SPX_ROC12M_3MMA"] = m["SPX_ROC12M"].rolling(3, min_periods=2).mean()
    for months in [1, 3, 6, 12]:
        m[f"SPX_ROC_Momentum_{months}M"] = m["SPX_ROC12M_3MMA"] - m["SPX_ROC12M_3MMA"].shift(months)
    m["MomentumCycleCandidate"] = [
        classify_momentum_phase(level, mom) for level, mom in zip(m["SPX_ROC12M_3MMA"], m["SPX_ROC_Momentum_3M"])
    ]
    m["MomentumCyclePhase"] = confirm_state(m["MomentumCycleCandidate"], required=2)
    m["LastMomentumCyclePeak"] = cycle_turn_date(m, "peak")
    m["LastMomentumCycleTrough"] = cycle_turn_date(m, "trough")
    m["MomentumDataStatus"] = np.where(m["MomentumCyclePhase"].ne("DATA INCOMPLETE"), "FULL", "PARTIAL")
    return m


def cycle_turn_date(m: pd.DataFrame, mode: str) -> pd.Series:
    roc = pd.to_numeric(m["SPX_ROC12M_3MMA"], errors="coerce")
    dates = pd.to_datetime(m["Date"], errors="coerce")
    turns: list[pd.Timestamp] = []
    out = []
    for idx in range(len(m)):
        if idx >= 2 and idx < len(m) - 2 and np.isfinite(roc.iloc[idx]):
            window = roc.iloc[idx - 2 : idx + 3]
            if mode == "peak" and roc.iloc[idx] == window.max():
                turns.append(dates.iloc[idx])
            if mode == "trough" and roc.iloc[idx] == window.min():
                turns.append(dates.iloc[idx])
        out.append(turns[-1] if turns else pd.NaT)
    return pd.Series(out)


def build_weekly_history(weekly: pd.DataFrame, daily: pd.DataFrame, structural: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    if w.empty:
        return w
    w["Date"] = pd.to_datetime(w["Date"], errors="coerce")
    w = pd.merge_asof(w.sort_values("Date"), structural.drop(columns=["SPX_Close"], errors="ignore").sort_values("Date"), on="Date", direction="backward")
    w = pd.merge_asof(w.sort_values("Date"), momentum.drop(columns=["SPX_Close"], errors="ignore").sort_values("Date"), on="Date", direction="backward")
    close = pd.to_numeric(w["SPX_Close"], errors="coerce")
    w["SPX_SMA200W"] = close.rolling(200, min_periods=200).mean()
    w["SMA200WExtensionPct"] = close / w["SPX_SMA200W"] - 1.0
    w["SMA200WExtensionPercentile"] = expanding_percentile(w["SMA200WExtensionPct"], min_periods=200)
    w["SMA200WZone"] = w["SMA200WExtensionPercentile"].map(classify_sma200w_zone)
    w["ForwardMaxDrawdown_1M"] = forward_max_drawdown_from_low(close, pd.to_numeric(w.get("SPX_Low", close), errors="coerce"), 4)
    w["MediumTerm_SMA_ROC_Phase"] = [
        classify_medium_sma_roc_phase(row) for row in w.to_dict("records")
    ]
    add_market_maturity(w)
    add_long_extension_amplitude(w)
    add_confirmations(w)
    add_current_risk(w, daily)
    w["MarketVulnerability"] = "DATA INCOMPLETE"
    w["HistoricalOutlookDataStatus"] = np.where(w["SPX_Close"].notna(), "PARTIAL", "INSUFFICIENT")
    w["MarketCycleModelVersion"] = MARKET_CYCLE_MODEL_VERSION
    w["LastUpdated"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    return w


def build_daily_correction_history(daily: pd.DataFrame, weekly_context: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or "SPY_Close" not in daily:
        return pd.DataFrame()
    d = daily.copy()
    d.index.name = None
    d["Date"] = pd.to_datetime(d.index, errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    context_cols = [
        "Date",
        "AAII_Bearish_Percentile",
        "VIX_AssetManager_Percentile",
        "PositioningVulnerability",
        "CurrentMarketRiskState",
        "HistoricalCurrentRiskState",
        "HistoricalCurrentRiskScore",
    ]
    if not weekly_context.empty and "Date" in weekly_context:
        selected_context_cols = [
            col for col in context_cols if col == "Date" or (col in weekly_context.columns and col not in d.columns)
        ]
        ctx = weekly_context[selected_context_cols].copy()
        ctx["Date"] = pd.to_datetime(ctx["Date"], errors="coerce")
        ctx = ctx.dropna(subset=["Date"]).sort_values("Date")
        if not ctx.empty and len(ctx.columns) > 1:
            d = pd.merge_asof(d.sort_values("Date"), ctx, on="Date", direction="backward")
    if "HY_OAS" not in d or pd.to_numeric(d["HY_OAS"], errors="coerce").notna().sum() == 0:
        hy_oas = load_hy_oas_daily(d["Date"].max())
        if not hy_oas.empty:
            d = pd.merge_asof(d.sort_values("Date"), hy_oas.sort_values("Date"), on="Date", direction="backward")
        else:
            d["HY_OAS"] = np.nan
    add_correction_bottom_indicator(d, frequency="daily")
    d["CorrectionFrequency"] = "DAILY"
    return d.loc[d["Date"].ge(pd.Timestamp("2005-01-01"))].reset_index(drop=True)


def load_hy_oas_daily(end_date: Any) -> pd.DataFrame:
    baseline = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])
    baseline_path = Path(__file__).with_name("data") / "BAMLH0A0HYM2_weekly.csv"
    if baseline_path.exists():
        try:
            raw = pd.read_csv(baseline_path, usecols=["time", "close"])
            baseline = weekly_archive_available_frame(raw)
        except Exception:
            baseline = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])

    tv_frame = load_tradingview_ohlcv_frame(
        "FRED:BAMLH0A0HYM2",
        pd.Timestamp(end_date) if pd.notna(end_date) else pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    if tv_frame.empty:
        tradingview = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])
    else:
        tradingview = pd.DataFrame(
            {
                "Date": pd.to_datetime(tv_frame.index, errors="coerce"),
                "HY_OAS": pd.to_numeric(tv_frame["Close"], errors="coerce"),
                "HYOASSourceFrequency": "DAILY_TRADINGVIEW",
            }
        ).dropna(subset=["Date", "HY_OAS"])

    try:
        frame = download_fred_series(
            "BAMLH0A0HYM2",
            observation_start="1993-01-01",
        )
    except Exception:
        frame = pd.DataFrame(columns=["Date", "Value"])
    result = frame[["Date", "Value"]].copy() if not frame.empty else pd.DataFrame(columns=["Date", "Value"])
    if not result.empty:
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
        result["HY_OAS"] = pd.to_numeric(result["Value"], errors="coerce")
        result["HYOASSourceFrequency"] = "DAILY_FRED"
        result = result.dropna(subset=["Date", "HY_OAS"])[["Date", "HY_OAS", "HYOASSourceFrequency"]]
    return combine_hy_oas_sources(baseline, tradingview, result, end_date=end_date)


def add_confirmations(w: pd.DataFrame) -> None:
    spx = pd.to_numeric(w["SPX_Close"], errors="coerce")
    for asset in ["QQQ", "BTC", "RSP", "IWM"]:
        ratio = pd.to_numeric(w[asset], errors="coerce") / spx
        w[f"{asset}_SPX_RS_13W"] = ratio.pct_change(13, fill_method=None)
        w[f"{asset}_SPX_RS_26W"] = ratio.pct_change(26, fill_method=None)
        w[f"{asset}_SPX_RS_4W"] = ratio.pct_change(4, fill_method=None)
        w[f"{asset}_SPX_RS_13W_Pctl"] = rolling_percentile(w[f"{asset}_SPX_RS_13W"], window=156, min_periods=52)
        w[f"{asset}_SPX_RS_26W_Pctl"] = rolling_percentile(w[f"{asset}_SPX_RS_26W"], window=156, min_periods=52)
        w[f"{asset}_SPX_RS_4W_Pctl"] = rolling_percentile(w[f"{asset}_SPX_RS_4W"], window=156, min_periods=52)
        w[f"{asset}_SPX_State"] = w[f"{asset}_SPX_RS_13W_Pctl"].map(classify_relative_state)
    xli_xlp = pd.to_numeric(w["XLI"], errors="coerce") / pd.to_numeric(w["XLP"], errors="coerce")
    w["XLI_XLP_RS_13W"] = xli_xlp.pct_change(13, fill_method=None)
    w["XLI_XLP_RS_26W"] = xli_xlp.pct_change(26, fill_method=None)
    w["XLI_XLP_Pctl"] = rolling_percentile(w["XLI_XLP_RS_13W"], window=156, min_periods=52)
    w["XLI_XLP_State"] = w["XLI_XLP_Pctl"].map(classify_relative_state)
    w["QQQ_SPX_State"] = w["QQQ_SPX_State"]
    w["BTC_SPX_State"] = w["BTC_SPX_State"]
    w["BreadthParticipationState"] = [
        classify_breadth_participation(rsp, iwm)
        for rsp, iwm in zip(w["RSP_SPX_RS_13W_Pctl"], w["IWM_SPX_RS_13W_Pctl"])
    ]
    add_high_beta_risk(w)
    add_breadth_risk(w)
    w["MarketTurningSignal"] = [
        classify_turning_signal(row) for row in w.to_dict("records")
    ]


def add_current_risk(w: pd.DataFrame, daily: pd.DataFrame) -> None:
    if w.empty or daily.empty:
        for column in [
            "CurrentMarketRiskState",
            "CurrentRiskModelVersion",
            "CurrentRiskActivation",
            "CurrentRiskEscalationScoreV2",
            "CurrentRiskDrawdownRisk",
            "CurrentRiskPriceCycleVulnerabilityRisk",
            "CurrentRiskBreadthRisk",
            "CurrentRiskRSIDivergenceRisk",
            "CurrentRiskVIXRisk",
            "CurrentRiskHighBetaRisk",
            "CurrentRiskHYRisk",
        ]:
            w[column] = "DATA INCOMPLETE" if column.endswith(("State", "Version")) else np.nan
        return

    add_forward_crash_risk(w)
    d = daily.copy()
    d.index.name = None
    d["Date"] = pd.to_datetime(d.index, errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "HY_OAS" not in d or pd.to_numeric(d["HY_OAS"], errors="coerce").notna().sum() == 0:
        hy_oas = load_hy_oas_daily(d["Date"].max())
        if not hy_oas.empty:
            d = pd.merge_asof(d, hy_oas.sort_values("Date"), on="Date", direction="backward")
        else:
            d["HY_OAS"] = np.nan
            d["HYOASSourceFrequency"] = "UNAVAILABLE"

    spy_weekly = (
        d.set_index("Date")["SPY_RawClose"]
        .pipe(pd.to_numeric, errors="coerce")
        .resample("W-FRI")
        .last()
        .dropna()
    )
    weekly_context = pd.DataFrame({"Date": spy_weekly.index, "SPY_WeeklyClose": spy_weekly.to_numpy()})
    weekly_context["SPY_SMA200W"] = spy_weekly.rolling(200, min_periods=200).mean().to_numpy()
    weekly_context["SPY_WeeklyRSI14"] = RSIIndicator(close=spy_weekly, window=14).rsi().to_numpy()
    weekly_context["SPY_26W_High"] = spy_weekly.rolling(26, min_periods=26).max().to_numpy()
    weekly_context["SPY_WeeklyRSI26WMax"] = (
        pd.Series(weekly_context["SPY_WeeklyRSI14"], index=weekly_context.index)
        .rolling(26, min_periods=13)
        .max()
        .to_numpy()
    )
    d = pd.merge_asof(d, weekly_context.sort_values("Date"), on="Date", direction="backward")

    drawdown_context_cols = [
        "Date",
        "SPX_1M_DD20_Probability",
        "SPX_1M_DD20_AnalogN",
        "SPX_1M_MedianMaxDrawdown",
        "SPX_1M_MaxDrawdownP10",
        "SPX_1M_DD20_Confidence",
        "SPX_1M_DD20_SampleMode",
    ]
    drawdown_context = w[[column for column in drawdown_context_cols if column in w]].copy()
    drawdown_context["Date"] = pd.to_datetime(drawdown_context["Date"], errors="coerce")
    d = pd.merge_asof(d, drawdown_context.dropna(subset=["Date"]).sort_values("Date"), on="Date", direction="backward")
    calculated = calculate_current_risk_v1(d)

    calculated_by_date = calculated.set_index("Date")
    for column in calculated.columns:
        if column == "Date":
            continue
        daily[column] = calculated_by_date[column].reindex(pd.to_datetime(daily.index, errors="coerce")).to_numpy()

    risk_columns = [
        column
        for column in calculated.columns
        if column.startswith(("CurrentRisk", "PVC_", "HistoricalCurrentRisk", "HistoricalDrawdown", "HistoricalHighBeta", "HistoricalBreadth", "HistoricalRSI", "HistoricalVIX"))
        or column in {"CurrentMarketRiskState", "CurrentMarketRiskDirection", "ActiveStressChannels"}
    ]
    weekly_risk = calculated.set_index("Date")[risk_columns].resample("W-FRI").last().reset_index()
    merged = pd.merge_asof(
        w[["Date"]].sort_values("Date"),
        weekly_risk.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    for column in risk_columns:
        w[column] = merged[column].to_numpy()
    w["CurrentRiskDataStatus"] = np.where(
        pd.to_numeric(w.get("CurrentRiskDataCoverage"), errors="coerce").ge(100.0),
        "FULL",
        "PARTIAL",
    )
    w["CurrentRiskModelVersion"] = CURRENT_RISK_MODEL_VERSION


def _legacy_add_current_risk(w: pd.DataFrame, daily: pd.DataFrame) -> None:
    if daily.empty:
        for col in [
            "VIXLevelPercentile",
            "VIXMomentumPercentile",
            "VIXTermStructure",
            "VIXTermStructurePercentile",
            "RealizedVol20D",
            "RealizedVolPercentile",
            "RealizedVolRisk",
            "VIXLevelRisk",
            "VIXMomentumRisk",
            "VIXTermStructureRisk",
            "SPXDrawdownFromATH",
            "ShortTermBreadthRisk",
            "RSIDivergence",
            "RSIDivergenceRiskScore",
            "SPX_1M_DD20_Probability",
            "SPX_1M_DD20_AnalogN",
            "SPX_1M_MedianMaxDrawdown",
            "SPX_1M_MaxDrawdownP10",
            "SPX_1M_DD20_SampleMode",
            "CurrentMarketRiskState",
            "CurrentMarketRiskDirection",
            "HistoricalCurrentRiskState",
            "HistoricalCurrentRiskStateStartDate",
            "HistoricalCurrentRiskWeeksInState",
            "HistoricalCurrentRiskScore",
            "HistoricalCurrentRiskPrimaryDrivers",
            "HistoricalDrawdownRisk",
            "HistoricalHighBetaRisk",
            "HistoricalBreadthRisk",
            "HistoricalRSIDivergenceRisk",
            "HistoricalVIXLevelRisk",
            "HistoricalVIXMomentumRisk",
            "HistoricalVIXTermStructureRisk",
            "HistoricalCombinedVIXRisk",
        ]:
            w[col] = np.nan if "Percentile" in col or col in {"VIXTermStructure", "RealizedVol20D", "SPXDrawdownFromATH"} else "DATA INCOMPLETE"
        return
    d = daily.copy()
    d["Date"] = pd.to_datetime(d.index, errors="coerce")
    weekly_risk = pd.DataFrame({"Date": d.resample("W-FRI", on="Date").last().index})
    weekly_last = d.resample("W-FRI", on="Date").last()
    for col in ["VIX", "VIXTermStructure", "SPX_RealizedVol20D", "SPX_DrawdownFromATH", "RSP_SPX_4W", "IWM_SPX_4W"]:
        weekly_risk[col] = weekly_last[col].to_numpy() if col in weekly_last.columns else np.nan
    weekly_risk["VIX_4W_Change"] = pd.to_numeric(weekly_risk["VIX"], errors="coerce").diff(4)
    weekly_risk["VIXLevelPercentile"] = rolling_percentile(weekly_risk["VIX"], 156, 52)
    weekly_risk["VIXMomentumPercentile"] = rolling_percentile(weekly_risk["VIX_4W_Change"], 156, 52)
    weekly_risk["VIXTermStructurePercentile"] = rolling_percentile(weekly_risk["VIXTermStructure"], 156, 52)
    weekly_risk["RealizedVol20D"] = weekly_risk["SPX_RealizedVol20D"]
    weekly_risk["RealizedVolPercentile"] = rolling_percentile(weekly_risk["RealizedVol20D"], 156, 52)
    weekly_risk["RealizedVolRisk"] = weekly_risk["RealizedVolPercentile"]
    weekly_risk["VIXLevelRisk"] = weekly_risk["VIX"].map(vix_level_risk_score)
    weekly_risk["VIXMomentumRisk"] = (2.0 * (weekly_risk["VIXMomentumPercentile"] - 50.0)).clip(lower=0.0, upper=100.0)
    weekly_risk["VIXTermStructureRisk"] = weekly_risk["VIXTermStructure"].map(vix_term_structure_risk_score)
    weekly_risk["SPXDrawdownFromATH"] = weekly_risk["SPX_DrawdownFromATH"]
    weekly_risk["DrawdownRiskPercentile"] = rolling_percentile(-weekly_risk["SPXDrawdownFromATH"], 156, 52)
    weekly_risk["ShortTermBreadthRisk"] = [
        classify_short_breadth(rsp, iwm) for rsp, iwm in zip(weekly_risk["RSP_SPX_4W"], weekly_risk["IWM_SPX_4W"])
    ]
    weekly_risk["VIXTermStructureState"] = weekly_risk["VIXTermStructure"].map(classify_vix_term_structure)
    add_forward_crash_risk(w)
    add_rsi_divergence_risk(w)
    enrich = [
        "Date",
        "QQQHighBetaRisk",
        "BTCHighBetaRisk",
        "HighBetaRisk",
        "HighBetaRiskState",
        "HighBetaRiskDirection",
        "BreadthLevelRisk",
        "BreadthMomentumRisk",
        "BreadthRisk",
        "BreadthRiskState",
        "Breadth50LevelRisk",
        "Breadth200LevelRisk",
        "Breadth50MomentumRisk",
        "Breadth200MomentumRisk",
        "Delta4W_Above50D",
        "Delta4W_Above200D",
        "BreadthRiskSource",
        "SPXAboveSMA50D",
        "SPXAboveSMA200D",
        "SPX_1M_DD20_Probability",
        "SPX_1M_DD20_AnalogN",
        "SPX_1M_MedianMaxDrawdown",
        "SPX_1M_MaxDrawdownP10",
        "SPX_1M_DD20_Confidence",
        "SPX_1M_DD20_SampleMode",
        "RSIDivergenceActive",
        "RSIDivergenceState",
        "RSIDivergenceRiskScore",
        "RSIDivergenceStartDate",
        "RSIDivergenceDurationWeeks",
        "RSIDropPoints",
        "RSIDivergencePriceGain",
    ]
    weekly_risk = pd.merge_asof(weekly_risk.sort_values("Date"), w[enrich].sort_values("Date"), on="Date", direction="backward")
    weekly_risk["CurrentMarketRiskState"] = [
        classify_current_risk(row) for row in weekly_risk.to_dict("records")
    ]
    weekly_risk["HistoricalCurrentRiskState"] = confirm_state(weekly_risk["CurrentMarketRiskState"], required=2)
    risk_level = weekly_risk["CurrentMarketRiskState"].map({"LOW": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "ACUTE": 4})
    weekly_risk["CurrentMarketRiskDirection"] = np.select(
        [risk_level.diff(4) > 0.25, risk_level.diff(4) < -0.25],
        ["RISING", "FALLING"],
        default="STABLE",
    )
    weekly_risk["ActiveStressChannels"] = [active_stress_channels(row) for row in weekly_risk.to_dict("records")]
    weekly_risk["HistoricalDrawdownRisk"] = pd.to_numeric(weekly_risk["SPX_1M_DD20_Probability"], errors="coerce")
    weekly_risk["HistoricalHighBetaRisk"] = pd.to_numeric(weekly_risk["HighBetaRisk"], errors="coerce")
    weekly_risk["HistoricalBreadthRisk"] = pd.to_numeric(weekly_risk["BreadthRisk"], errors="coerce")
    weekly_risk["HistoricalRSIDivergenceRisk"] = pd.to_numeric(weekly_risk["RSIDivergenceRiskScore"], errors="coerce")
    weekly_risk["HistoricalVIXLevelRisk"] = pd.to_numeric(weekly_risk["VIXLevelRisk"], errors="coerce")
    weekly_risk["HistoricalVIXMomentumRisk"] = pd.to_numeric(weekly_risk["VIXMomentumRisk"], errors="coerce")
    weekly_risk["HistoricalVIXTermStructureRisk"] = pd.to_numeric(weekly_risk["VIXTermStructureRisk"], errors="coerce")
    weekly_risk["HistoricalCombinedVIXRisk"] = (
        0.30 * weekly_risk["HistoricalVIXLevelRisk"]
        + 0.30 * weekly_risk["HistoricalVIXMomentumRisk"]
        + 0.40 * weekly_risk["HistoricalVIXTermStructureRisk"]
    ).clip(lower=0.0, upper=100.0)
    weekly_risk["HistoricalCurrentRiskScore"] = [
        current_risk_supporting_score(row) for row in weekly_risk.to_dict("records")
    ]
    weekly_risk["HistoricalCurrentRiskPrimaryDrivers"] = [
        current_risk_primary_drivers(row) for row in weekly_risk.to_dict("records")
    ]
    add_historical_state_duration(
        weekly_risk,
        state_col="HistoricalCurrentRiskState",
        start_col="HistoricalCurrentRiskStateStartDate",
        weeks_col="HistoricalCurrentRiskWeeksInState",
    )
    weekly_risk["RSIDivergence"] = weekly_risk["RSIDivergenceState"].fillna("NONE").to_numpy()
    keep = [
        "Date",
        "VIXLevelPercentile",
        "VIXMomentumPercentile",
        "VIXTermStructure",
        "VIXTermStructurePercentile",
        "VIXTermStructureState",
        "RealizedVol20D",
        "RealizedVolPercentile",
        "RealizedVolRisk",
        "VIXLevelRisk",
        "VIXMomentumRisk",
        "VIXTermStructureRisk",
        "SPXDrawdownFromATH",
        "DrawdownRiskPercentile",
        "SPX_1M_DD20_Probability",
        "SPX_1M_DD20_AnalogN",
        "SPX_1M_MedianMaxDrawdown",
        "SPX_1M_MaxDrawdownP10",
        "SPX_1M_DD20_Confidence",
        "SPX_1M_DD20_SampleMode",
        "ShortTermBreadthRisk",
        "RSIDivergence",
        "RSIDivergenceActive",
        "RSIDivergenceState",
        "RSIDivergenceRiskScore",
        "RSIDivergenceStartDate",
        "RSIDivergenceDurationWeeks",
        "RSIDropPoints",
        "RSIDivergencePriceGain",
        "CurrentMarketRiskState",
        "CurrentMarketRiskDirection",
        "ActiveStressChannels",
        "HistoricalCurrentRiskState",
        "HistoricalCurrentRiskStateStartDate",
        "HistoricalCurrentRiskWeeksInState",
        "HistoricalCurrentRiskScore",
        "HistoricalCurrentRiskPrimaryDrivers",
        "HistoricalDrawdownRisk",
        "HistoricalHighBetaRisk",
        "HistoricalBreadthRisk",
        "HistoricalRSIDivergenceRisk",
        "HistoricalVIXLevelRisk",
        "HistoricalVIXMomentumRisk",
        "HistoricalVIXTermStructureRisk",
        "HistoricalCombinedVIXRisk",
    ]
    merged = pd.merge_asof(w[["Date"]].sort_values("Date"), weekly_risk[keep].sort_values("Date"), on="Date", direction="backward")
    for col in keep:
        if col != "Date":
            w[col] = merged[col].to_numpy()
    w["CurrentRiskDataStatus"] = np.where(w["VIXLevelPercentile"].notna() & w["RealizedVolPercentile"].notna(), "FULL", "PARTIAL")


def add_correction_bottom_indicator(w: pd.DataFrame, frequency: str = "weekly") -> None:
    if w.empty:
        return
    if str(frequency).lower() != "daily":
        return
    calculated = calculate_correction_bottom_indicator(w)
    for column in calculated.columns:
        w[column] = calculated[column].to_numpy()


def _legacy_correction_bottom_indicator(w: pd.DataFrame, frequency: str = "weekly") -> None:
    if w.empty or "SPX_Close" not in w:
        return
    is_daily = str(frequency).lower() == "daily"
    divergence_lookback = 20 if is_daily else 13
    medium_roc_period = 5 if is_daily else 4
    slow_roc_period = 20 if is_daily else 13
    exhaustion_lookback = 20 if is_daily else 13
    recovery_lookback = 20
    bottom_low_lookback = 20 if is_daily else 13
    recent_exhaustion_lookback = 20 if is_daily else 4
    ratio_recovery_lookback = 20 if is_daily else 13
    recovery_breadth_period = 5 if is_daily else 4
    monitor_delta = pd.offsets.BDay(20) if is_daily else pd.Timedelta(weeks=4)
    close = pd.to_numeric(w["SPX_Close"], errors="coerce")
    low = pd.to_numeric(w.get("SPX_Low", close), errors="coerce").fillna(close)
    dates = pd.to_datetime(w["Date"], errors="coerce")
    peak = close.cummax()
    drawdown = close / peak - 1.0
    rsi = RSIIndicator(close=close, window=14).rsi()
    roc_1p = close.pct_change(1, fill_method=None)
    roc_medium = close.pct_change(medium_roc_period, fill_method=None)
    roc_slow = close.pct_change(slow_roc_period, fill_method=None)
    vix = pd.to_numeric(w.get("VIX", pd.Series(np.nan, index=w.index)), errors="coerce")
    vix3m = pd.to_numeric(w.get("VIX3M", pd.Series(np.nan, index=w.index)), errors="coerce")
    vix_ratio = vix / vix3m
    vix_1p_change = vix.pct_change(1, fill_method=None)
    realized = pd.to_numeric(w.get("RealizedVol20D", pd.Series(np.nan, index=w.index)), errors="coerce")
    breadth50 = pd.to_numeric(w.get("SPXAboveSMA50D", pd.Series(np.nan, index=w.index)), errors="coerce")
    breadth200 = pd.to_numeric(w.get("SPXAboveSMA200D", pd.Series(np.nan, index=w.index)), errors="coerce")
    qqq_rs = pd.to_numeric(w.get("QQQ", pd.Series(np.nan, index=w.index)), errors="coerce") / close
    iwm_rs = pd.to_numeric(w.get("IWM", pd.Series(np.nan, index=w.index)), errors="coerce") / close
    aa_bear = pd.to_numeric(w.get("AAII_Bearish_Percentile", pd.Series(np.nan, index=w.index)), errors="coerce")
    vix_asset_mgr = pd.to_numeric(w.get("VIX_AssetManager_Percentile", pd.Series(np.nan, index=w.index)), errors="coerce")

    price_exhaustion = []
    vol_exhaustion = []
    breadth_exhaustion = []
    credit_positioning_exhaustion = []
    risk_appetite_exhaustion = []
    exhaustion = []
    price_recovery = []
    vix_recovery = []
    breadth_recovery = []
    risk_appetite_recovery = []
    recovery = []
    coverage = []

    rsi_div_flags = []
    roc_div_flags = []
    vix_div_flags = []
    breadth_div_flags = []
    breadth_washout_flags = []
    breadth_recovery_flags = []
    vix_peak_flags = []
    term_norm_flags = []
    hy_div_flags = []
    cftc_state = []

    for idx in range(len(w)):
        prior_start = max(0, idx - divergence_lookback)
        prior_slice = slice(prior_start, idx)
        lower_low = idx > 0 and np.isfinite(close.iloc[idx]) and np.isfinite(close.iloc[prior_slice].min()) and close.iloc[idx] <= close.iloc[prior_slice].min()
        rsi_prior_low = rsi.iloc[prior_slice].min()
        roc_prior_low = roc_slow.iloc[prior_slice].min()
        rsi_div = bool(lower_low and np.isfinite(rsi.iloc[idx]) and np.isfinite(rsi_prior_low) and rsi.iloc[idx] > rsi_prior_low)
        roc_div = bool(lower_low and np.isfinite(roc_slow.iloc[idx]) and np.isfinite(roc_prior_low) and roc_slow.iloc[idx] > roc_prior_low)
        momentum_decel = score_momentum_deceleration(roc_medium.iloc[idx], roc_medium.iloc[prior_start : idx + 1])
        px_block, px_cov = weighted_block(
            [
                (100.0 if rsi_div else 0.0, 0.25),
                (100.0 if roc_div else 0.0, 0.20),
                (momentum_decel, 0.35),
                (np.nan, 0.20),
            ]
        )

        recent_vix_peak = vix.iloc[max(0, idx - exhaustion_lookback) : idx + 1].max()
        vix_decel = score_decline_from_peak(vix.iloc[idx], recent_vix_peak, [0.05, 0.10, 0.20])
        vix_prior_high = vix.iloc[prior_slice].max()
        vix_div = bool(lower_low and np.isfinite(vix.iloc[idx]) and np.isfinite(vix_prior_high) and vix.iloc[idx] < vix_prior_high)
        ratio_recent_peak = vix_ratio.iloc[max(0, idx - exhaustion_lookback) : idx + 1].max()
        term_norm = score_decline_from_peak(vix_ratio.iloc[idx], ratio_recent_peak, [0.03, 0.06, 0.10])
        rv_peak = realized.iloc[max(0, idx - exhaustion_lookback) : idx + 1].max()
        rv_exh = score_decline_from_peak(realized.iloc[idx], rv_peak, [0.05, 0.10, 0.20])
        vol_block, vol_cov = weighted_block(
            [
                (vix_decel, 0.30),
                (100.0 if vix_div else 0.0, 0.20),
                (term_norm, 0.20),
                (rv_exh, 0.20),
            ]
        )

        b50_recent_low = breadth50.iloc[max(0, idx - exhaustion_lookback) : idx + 1].min()
        b200_recent_low = breadth200.iloc[max(0, idx - exhaustion_lookback) : idx + 1].min()
        b50_prior_low = breadth50.iloc[prior_slice].min()
        breadth_div = bool(lower_low and np.isfinite(breadth50.iloc[idx]) and np.isfinite(b50_prior_low) and breadth50.iloc[idx] > b50_prior_low)
        b50_recovery = score_rise_from_low(breadth50.iloc[idx], b50_recent_low, [5.0, 10.0, 20.0])
        b200_recovery = score_rise_from_low(breadth200.iloc[idx], b200_recent_low, [3.0, 7.0, 12.0])
        b50_washout = np.isfinite(breadth50.iloc[idx]) and breadth50.iloc[idx] <= 25
        b_block, b_cov = weighted_block(
            [
                (100.0 if breadth_div else 0.0, 0.35),
                (b50_recovery, 0.35),
                (b200_recovery, 0.15),
                (70.0 if b50_washout else 0.0, 0.15),
            ]
        )

        positioning_peak = max(safe_float(aa_bear.iloc[idx]), safe_float(vix_asset_mgr.iloc[idx]))
        positioning_roll_peak = max(
            safe_float(aa_bear.iloc[max(0, idx - exhaustion_lookback) : idx + 1].max()),
            safe_float(vix_asset_mgr.iloc[max(0, idx - exhaustion_lookback) : idx + 1].max()),
        )
        pos_exh = score_decline_from_peak(positioning_peak, positioning_roll_peak, [5.0, 10.0, 20.0], absolute=True)
        cp_block, cp_cov = weighted_block([(pos_exh, 1.0)])

        qqq_rec = score_ratio_recovery(qqq_rs, idx, lookback=ratio_recovery_lookback)
        iwm_rec = score_ratio_recovery(iwm_rs, idx, lookback=ratio_recovery_lookback)
        ra_block, ra_cov = weighted_block([(qqq_rec, 0.50), (iwm_rec, 0.50)])

        ex_score, ex_cov = weighted_block(
            [
                (px_block, 0.25),
                (vol_block, 0.25),
                (b_block, 0.20),
                (cp_block, 0.20),
                (ra_block, 0.10),
            ]
        )

        prev_5_high = close.iloc[max(0, idx - 5) : idx].max()
        prev_10_high = close.iloc[max(0, idx - 10) : idx].max()
        price_rec_score = 0.0
        price_rec_score += 30.0 if np.isfinite(roc_1p.iloc[idx]) and roc_1p.iloc[idx] > 0 else 0.0
        price_rec_score += 30.0 if np.isfinite(prev_5_high) and close.iloc[idx] > prev_5_high else 0.0
        price_rec_score += 40.0 if np.isfinite(prev_10_high) and close.iloc[idx] > prev_10_high else 0.0
        vix_rec_score = score_decline_from_peak(vix.iloc[idx], vix.iloc[max(0, idx - recovery_lookback) : idx + 1].max(), [0.10, 0.15, 0.20, 0.25])
        breadth_rec_score, breadth_rec_cov = weighted_block(
            [
                (score_rise_from_low(breadth50.iloc[idx], breadth50.iloc[max(0, idx - recovery_lookback) : idx + 1].min(), [5.0, 10.0, 20.0]), 0.70),
                (score_rise_from_low(breadth200.iloc[idx], breadth200.iloc[max(0, idx - recovery_lookback) : idx + 1].min(), [3.0, 7.0, 12.0]), 0.30),
            ]
        )
        risk_app_rec_score = ra_block
        rec_score, rec_cov = weighted_block(
            [
                (price_rec_score, 0.30),
                (vix_rec_score, 0.25),
                (breadth_rec_score, 0.25),
                (risk_app_rec_score, 0.20),
            ]
        )

        price_exhaustion.append(px_block)
        vol_exhaustion.append(vol_block)
        breadth_exhaustion.append(b_block)
        credit_positioning_exhaustion.append(cp_block)
        risk_appetite_exhaustion.append(ra_block)
        exhaustion.append(ex_score)
        price_recovery.append(price_rec_score)
        vix_recovery.append(vix_rec_score)
        breadth_recovery.append(breadth_rec_score)
        risk_appetite_recovery.append(risk_app_rec_score)
        recovery.append(rec_score)
        coverage.append((ex_cov + rec_cov) / 2.0 * 100.0)
        rsi_div_flags.append(rsi_div)
        roc_div_flags.append(roc_div)
        vix_div_flags.append(vix_div)
        breadth_div_flags.append(breadth_div)
        breadth_washout_flags.append(bool(b50_washout))
        breadth_recovery_flags.append(bool(np.isfinite(b50_recovery) and b50_recovery >= 50))
        vix_peak_flags.append(bool(np.isfinite(vix_decel) and vix_decel >= 50))
        term_norm_flags.append(bool(np.isfinite(term_norm) and term_norm >= 50))
        hy_div_flags.append(False)
        cftc_state.append("UNAVAILABLE")

    model_state = []
    bottoming_price = np.nan
    anchor_low = np.nan
    anchor_date = pd.NaT
    monitor_until = pd.NaT
    recovery_high = np.nan
    active_state = "NORMAL"
    undercut_active_idx: int | None = None
    bottoming_events = []
    recovery_events = []
    reaccel_events = []
    reclaim_events = []
    breakdown_events = []
    new_wave_events = []
    undercut_flags = []
    reclaim_flags = []
    breakdown_flags = []
    reaccel_watch_flags = []
    reaccel_status = []
    new_wave_watch_flags = []
    policy_backstop = []
    correction_peak = []
    anchor_lows = []
    recovery_highs = []
    event_type = []
    active_stress_domains = []

    for idx in range(len(w)):
        date = dates.iloc[idx]
        dd = safe_float(drawdown.iloc[idx])
        ex_score = safe_float(exhaustion[idx])
        rec_score = safe_float(recovery[idx])
        correction_peak.append(peak.iloc[idx])
        current_event = ""
        current_type = correction_type(dd)
        crisis_mode = dd <= -0.15 and (safe_float(vix.iloc[idx]) >= 35 or safe_float(vix_ratio.iloc[idx]) >= 1.0)
        policy_backstop.append("OFF" if not crisis_mode else "EMERGING")

        price_stress = np.isfinite(roc_1p.iloc[idx]) and roc_1p.iloc[idx] < 0
        vix_stress = is_new_high(vix, idx, 10) or pct_rise_from_low(vix, idx, 5) >= 0.20
        breadth_stress = is_new_low(breadth50, idx, 10) or value_change(breadth50, idx, 5) <= -8.0
        credit_stress = False
        stress_count = int(price_stress) + int(vix_stress) + int(breadth_stress) + int(credit_stress)
        active_stress_domains.append(stress_count)

        if active_state == "RECOVERY CONFIRMED":
            recovery_high = max(recovery_high, safe_float(close.iloc[idx])) if np.isfinite(recovery_high) else safe_float(close.iloc[idx])
            rec_dd = close.iloc[idx] / recovery_high - 1.0 if np.isfinite(recovery_high) and recovery_high > 0 else np.nan
            new_watch = np.isfinite(rec_dd) and rec_dd <= -0.04 and safe_float(vix_1p_change.iloc[idx]) > 0 and value_change(breadth50, idx, 1) < 0
            new_wave = np.isfinite(rec_dd) and rec_dd <= -0.05
            if new_wave:
                active_state = "NEW CORRECTION WAVE"
                current_event = "NEW CORRECTION WAVE"
                bottoming_price = np.nan
                anchor_low = np.nan
                recovery_high = np.nan
            elif new_watch:
                active_state = "NEW CORRECTION WAVE WATCH"
                current_event = "NEW CORRECTION WAVE WATCH"

        if dd > -0.03 and active_state not in {"RECOVERY CONFIRMED", "NEW CORRECTION WAVE WATCH"}:
            active_state = "NORMAL"
        elif dd <= -0.03 and dd > -0.05 and active_state == "NORMAL":
            active_state = "CORRECTION WATCH"
        elif dd <= -0.05 and active_state in {"NORMAL", "CORRECTION WATCH", "NEW CORRECTION WAVE"}:
            active_state = "CORRECTION"

        if dd <= -0.05 and active_state in {"CORRECTION", "STRESS BUILDING", "CAPITULATION"}:
            if ex_score >= 65:
                active_state = "EARLY EXHAUSTION"
            elif ex_score >= 45:
                active_state = "CAPITULATION"
            elif ex_score >= 30:
                active_state = "STRESS BUILDING"

        if dd <= -0.05 and ex_score >= 50 and rec_score >= 40 and active_state not in {"BOTTOMING", "RECOVERY CONFIRMED"}:
            active_state = "BOTTOMING"
            bottoming_price = safe_float(close.iloc[idx])
            anchor_low = safe_float(low.iloc[max(0, idx - bottom_low_lookback) : idx + 1].min())
            anchor_date = date
            monitor_until = date + monitor_delta
            current_event = "BOTTOMING"

        re_watch = False
        undercut = False
        reclaim = False
        breakdown = False
        new_watch = active_state == "NEW CORRECTION WAVE WATCH"
        if active_state in {"BOTTOMING", "RE-ACCELERATION WATCH", "RE-ACCELERATION", "LOW UNDERCUT"} and pd.notna(monitor_until) and date <= monitor_until:
            re_watch = bool(vix_stress and (breadth_stress or credit_stress) and np.isfinite(bottoming_price) and close.iloc[idx] <= bottoming_price * 1.02)
            if re_watch and active_state == "BOTTOMING":
                active_state = "RE-ACCELERATION WATCH"
                current_event = "RE-ACCELERATION WATCH"
            if re_watch and price_stress:
                active_state = "RE-ACCELERATION"
                current_event = "RE-ACCELERATION"
            if np.isfinite(anchor_low):
                undercut = bool(low.iloc[idx] < anchor_low * 0.995)
                if undercut:
                    active_state = "LOW UNDERCUT"
                    undercut_active_idx = idx
                    current_event = "LOW UNDERCUT"
                if undercut_active_idx is not None and idx - undercut_active_idx <= 1 and close.iloc[idx] > anchor_low:
                    depth = low.iloc[undercut_active_idx] / anchor_low - 1.0
                    reclaim = bool(-0.015 <= depth <= -0.005)
                    if reclaim:
                        active_state = "CAPITULATION RECLAIM"
                        current_event = "CAPITULATION RECLAIM"
                        monitor_until = date + monitor_delta
                below_2 = close.iloc[idx] < anchor_low * 0.98
                below_1_twice = idx > 0 and close.iloc[idx] < anchor_low * 0.99 and close.iloc[idx - 1] < anchor_low * 0.99
                breakdown = bool(below_2 or below_1_twice)
                if breakdown:
                    active_state = "CONFIRMED BREAKDOWN"
                    current_event = "CONFIRMED BREAKDOWN"
                    bottoming_price = np.nan
                    anchor_low = np.nan
                    monitor_until = pd.NaT

        recent_exhaustion = max(exhaustion[max(0, idx - recent_exhaustion_lookback) : idx + 1]) if idx >= 0 else np.nan
        vix_decline = decline_from_peak(vix.iloc[idx], vix.iloc[max(0, idx - recovery_lookback) : idx + 1].max())
        breadth_or_risk = (
            (value_change(breadth50, idx, recovery_breadth_period) >= 5.0)
            or score_ratio_recovery(qqq_rs, idx, lookback=ratio_recovery_lookback) >= 50
            or score_ratio_recovery(iwm_rs, idx, lookback=ratio_recovery_lookback) >= 50
        )
        prior_10_high = close.iloc[max(0, idx - 10) : idx].max()
        recovery_confirmed = (
            np.isfinite(recent_exhaustion)
            and recent_exhaustion >= 50
            and np.isfinite(prior_10_high)
            and close.iloc[idx] > prior_10_high
            and vix_decline >= 0.20
            and breadth_or_risk
        )
        if recovery_confirmed:
            active_state = "RECOVERY CONFIRMED"
            current_event = "RECOVERY CONFIRMED"
            recovery_high = safe_float(close.iloc[idx])
            monitor_until = pd.NaT

        if active_state == "CONFIRMED BREAKDOWN" and ex_score < 50 and dd <= -0.05:
            active_state = "CORRECTION"

        model_state.append(active_state)
        undercut_flags.append(undercut)
        reclaim_flags.append(reclaim)
        breakdown_flags.append(breakdown)
        reaccel_watch_flags.append(re_watch)
        reaccel_status.append("WATCH" if re_watch and active_state != "RE-ACCELERATION" else "ACTIVE" if active_state == "RE-ACCELERATION" else "OFF")
        new_wave_watch_flags.append(new_watch)
        anchor_lows.append(anchor_low)
        recovery_highs.append(recovery_high)
        event_type.append(current_event)
        bottoming_events.append(current_event == "BOTTOMING")
        recovery_events.append(current_event == "RECOVERY CONFIRMED")
        reaccel_events.append(current_event == "RE-ACCELERATION")
        reclaim_events.append(current_event == "CAPITULATION RECLAIM")
        breakdown_events.append(current_event == "CONFIRMED BREAKDOWN")
        new_wave_events.append(current_event == "NEW CORRECTION WAVE")

    w["CorrectionModelVersion"] = "CORRECTION_END_V1"
    w["CorrectionFrequency"] = "DAILY" if is_daily else "WEEKLY"
    w["CorrectionDrawdown"] = drawdown
    w["CorrectionType"] = [correction_type(v) for v in drawdown]
    w["CorrectionState"] = model_state
    w["CorrectionExhaustionScore"] = exhaustion
    w["CorrectionRecoveryScore"] = recovery
    w["CorrectionDataCoverage"] = coverage
    w["PriceMomentumExhaustion"] = price_exhaustion
    w["VolatilityExhaustion"] = vol_exhaustion
    w["BreadthExhaustion"] = breadth_exhaustion
    w["CreditPositioningExhaustion"] = credit_positioning_exhaustion
    w["RiskAppetiteExhaustion"] = risk_appetite_exhaustion
    w["PriceRecoveryScore"] = price_recovery
    w["VIXRecoveryScore"] = vix_recovery
    w["BreadthRecoveryScore"] = breadth_recovery
    w["RiskAppetiteRecoveryScore"] = risk_appetite_recovery
    w["CorrectionRSIBullishDivergence"] = rsi_div_flags
    w["CorrectionROCBullishDivergence"] = roc_div_flags
    w["CorrectionVIXPositiveDivergence"] = vix_div_flags
    w["CorrectionBreadthPositiveDivergence"] = breadth_div_flags
    w["BreadthWashout"] = breadth_washout_flags
    w["BreadthRecovery"] = breadth_recovery_flags
    w["VIXPeakSignal"] = vix_peak_flags
    w["VIXTermStructureNormalization"] = term_norm_flags
    w["HYOASDivergence"] = hy_div_flags
    w["CFTCDeleveragingState"] = cftc_state
    w["CorrectionLowUndercut"] = undercut_flags
    w["CorrectionCapitulationReclaim"] = reclaim_flags
    w["CorrectionConfirmedBreakdown"] = breakdown_flags
    w["CorrectionReaccelerationWatch"] = reaccel_watch_flags
    w["CorrectionReaccelerationStatus"] = reaccel_status
    w["CorrectionNewWaveWatch"] = new_wave_watch_flags
    w["CorrectionPolicyBackstop"] = policy_backstop
    w["CorrectionPeak"] = correction_peak
    w["CorrectionAnchorLow"] = anchor_lows
    w["CorrectionRecoveryHigh"] = recovery_highs
    w["CorrectionEventType"] = event_type
    w["CorrectionBottomingEvent"] = bottoming_events
    w["CorrectionRecoveryConfirmedEvent"] = recovery_events
    w["CorrectionReaccelerationEvent"] = reaccel_events
    w["CorrectionCapitulationReclaimEvent"] = reclaim_events
    w["CorrectionBreakdownEvent"] = breakdown_events
    w["CorrectionNewWaveEvent"] = new_wave_events
    w["CorrectionActiveStressDomains"] = active_stress_domains


def weighted_block(items: list[tuple[Any, float]]) -> tuple[float, float]:
    total_weight = 0.0
    weighted_sum = 0.0
    available_weight = 0.0
    for value, weight in items:
        wgt = safe_float(weight)
        if not np.isfinite(wgt) or wgt <= 0:
            continue
        total_weight += wgt
        score = safe_float(value)
        if not np.isfinite(score):
            continue
        available_weight += wgt
        weighted_sum += np.clip(score, 0.0, 100.0) * wgt
    if available_weight <= 0:
        return np.nan, 0.0
    coverage = available_weight / total_weight if total_weight > 0 else 0.0
    return float(weighted_sum / available_weight), float(coverage)


def score_momentum_deceleration(current: Any, window: pd.Series) -> float:
    current_v = safe_float(current)
    values = pd.to_numeric(window, errors="coerce").dropna()
    if not np.isfinite(current_v) or len(values) < 3:
        return np.nan
    trough = values.min()
    if not np.isfinite(trough) or trough >= 0:
        return 0.0
    improvement = current_v - trough
    return float(np.clip(100.0 * improvement / max(abs(trough), 0.01), 0.0, 100.0))


def score_decline_from_peak(current: Any, peak: Any, thresholds: list[float], absolute: bool = False) -> float:
    decline = decline_from_peak(current, peak, absolute=absolute)
    if not np.isfinite(decline) or not thresholds:
        return np.nan
    thresholds = sorted([abs(t) for t in thresholds if np.isfinite(safe_float(t))])
    if not thresholds:
        return np.nan
    if decline <= 0:
        return 0.0
    return float(np.clip(100.0 * decline / thresholds[-1], 0.0, 100.0))


def score_rise_from_low(current: Any, low: Any, thresholds: list[float]) -> float:
    current_v = safe_float(current)
    low_v = safe_float(low)
    thresholds = sorted([abs(t) for t in thresholds if np.isfinite(safe_float(t))])
    if not np.isfinite(current_v) or not np.isfinite(low_v) or not thresholds:
        return np.nan
    rise = current_v - low_v
    return float(np.clip(100.0 * rise / thresholds[-1], 0.0, 100.0))


def score_ratio_recovery(series: pd.Series, idx: int, lookback: int = 13) -> float:
    if idx < 0 or idx >= len(series):
        return np.nan
    value = safe_float(series.iloc[idx])
    low = safe_float(series.iloc[max(0, idx - lookback) : idx + 1].min())
    if not np.isfinite(value) or not np.isfinite(low) or low <= 0:
        return np.nan
    return float(np.clip(100.0 * ((value / low) - 1.0) / 0.06, 0.0, 100.0))


def correction_type(drawdown: Any) -> str:
    dd = safe_float(drawdown)
    if not np.isfinite(dd) or dd > -0.03:
        return "NONE"
    if dd > -0.05:
        return "PULLBACK"
    if dd > -0.10:
        return "SHALLOW CORRECTION"
    if dd > -0.15:
        return "STANDARD CORRECTION"
    if dd > -0.25:
        return "DEEP CORRECTION"
    return "CRASH / BEAR MARKET"


def is_new_high(series: pd.Series, idx: int, lookback: int) -> bool:
    if idx < 0 or idx >= len(series):
        return False
    value = safe_float(series.iloc[idx])
    window = pd.to_numeric(series.iloc[max(0, idx - lookback + 1) : idx + 1], errors="coerce")
    return bool(np.isfinite(value) and not window.dropna().empty and value >= window.max())


def is_new_low(series: pd.Series, idx: int, lookback: int) -> bool:
    if idx < 0 or idx >= len(series):
        return False
    value = safe_float(series.iloc[idx])
    window = pd.to_numeric(series.iloc[max(0, idx - lookback + 1) : idx + 1], errors="coerce")
    return bool(np.isfinite(value) and not window.dropna().empty and value <= window.min())


def pct_rise_from_low(series: pd.Series, idx: int, lookback: int) -> float:
    if idx < 0 or idx >= len(series):
        return np.nan
    value = safe_float(series.iloc[idx])
    low = safe_float(pd.to_numeric(series.iloc[max(0, idx - lookback + 1) : idx + 1], errors="coerce").min())
    if not np.isfinite(value) or not np.isfinite(low) or low <= 0:
        return np.nan
    return float(value / low - 1.0)


def value_change(series: pd.Series, idx: int, periods: int) -> float:
    if idx < periods or idx >= len(series):
        return np.nan
    current = safe_float(series.iloc[idx])
    prior = safe_float(series.iloc[idx - periods])
    if not np.isfinite(current) or not np.isfinite(prior):
        return np.nan
    return float(current - prior)


def decline_from_peak(current: Any, peak: Any, absolute: bool = False) -> float:
    current_v = safe_float(current)
    peak_v = safe_float(peak)
    if not np.isfinite(current_v) or not np.isfinite(peak_v):
        return np.nan
    if absolute:
        return float(max(peak_v - current_v, 0.0))
    if peak_v <= 0:
        return np.nan
    return float(max(peak_v - current_v, 0.0) / peak_v)


def add_positioning(w: pd.DataFrame) -> None:
    try:
        aaii = read_processed("aaii")
        master = read_processed("cftc_master")
        positioning = calculate_positioning_risk_history(aaii, master)
    except Exception:
        positioning = pd.DataFrame()
    if positioning.empty:
        w["AAII_Bearish_Percentile"] = np.nan
        w["VIX_AssetManager_Percentile"] = np.nan
        w["PositioningRisk"] = np.nan
        w["PositioningVulnerability"] = "DATA INCOMPLETE"
        return
    p = positioning.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    cols = ["AAII_Bearish_3Y_Percentile", "VIX_AssetManager_NetPctOI_3Y_Percentile", "PositioningRisk"]
    p = p[["Date"] + [col for col in cols if col in p.columns]].dropna(subset=["Date"]).sort_values("Date")
    merged = pd.merge_asof(w[["Date"]].sort_values("Date"), p, on="Date", direction="backward")
    w["AAII_Bearish_Percentile"] = merged.get("AAII_Bearish_3Y_Percentile", pd.Series(np.nan, index=w.index)).to_numpy()
    w["VIX_AssetManager_Percentile"] = merged.get("VIX_AssetManager_NetPctOI_3Y_Percentile", pd.Series(np.nan, index=w.index)).to_numpy()
    positioning_risk = pd.to_numeric(
        merged.get("PositioningRisk", pd.Series(np.nan, index=w.index)), errors="coerce"
    )
    w["PositioningRisk"] = positioning_risk.to_numpy()
    w["PositioningVulnerability"] = positioning_risk.map(classify_positioning_vulnerability).to_numpy()


def add_vulnerability(w: pd.DataFrame) -> None:
    w["MarketVulnerability"] = [classify_vulnerability(row) for row in w.to_dict("records")]


def add_analog_engine(w: pd.DataFrame) -> None:
    asset_columns = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "BTC": "BTC"}
    for horizon, weeks in FORWARD_HORIZONS.items():
        close = pd.to_numeric(w["SPX_Close"], errors="coerce")
        w[f"ForwardReturn_{horizon}"] = close.shift(-weeks) / close - 1.0
        w[f"ForwardMaxDrawdown_{horizon}"] = forward_max_drawdown(close, weeks)
        for asset, column in asset_columns.items():
            asset_close = pd.to_numeric(w.get(column), errors="coerce")
            w[f"ForwardReturn_{asset}_{horizon}"] = asset_close.shift(-weeks) / asset_close - 1.0
            w[f"ForwardMaxDrawdown_{asset}_{horizon}"] = forward_max_drawdown(asset_close, weeks)


def build_historical_outlook(w: pd.DataFrame, current: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not current:
        return pd.DataFrame(), pd.DataFrame()
    analogs = select_analogs(w, current)
    rows = []
    for horizon in FORWARD_HORIZONS:
        sample = analogs.dropna(subset=[f"ForwardReturn_{horizon}", f"ForwardMaxDrawdown_{horizon}"])
        returns = pd.to_numeric(sample[f"ForwardReturn_{horizon}"], errors="coerce")
        drawdowns = pd.to_numeric(sample[f"ForwardMaxDrawdown_{horizon}"], errors="coerce")
        n = int(len(sample))
        avg_similarity = float(sample["Similarity"].mean()) if n else np.nan
        avg_coverage = float(sample["Coverage"].mean()) if n else np.nan
        rows.append(
            {
                "Horizon": horizon,
                "Median Forward Return": returns.median(),
                "Mean Forward Return": returns.mean(),
                "Positive Return Probability": (returns > 0).mean() if n else np.nan,
                "P25 Return": returns.quantile(0.25) if n else np.nan,
                "P75 Return": returns.quantile(0.75) if n else np.nan,
                "Risk of >15% Drawdown": (drawdowns <= -0.15).mean() if n else np.nan,
                "Risk of >25% Drawdown": (drawdowns <= -0.25).mean() if n else np.nan,
                "Risk of >45% Drawdown": (drawdowns <= -0.45).mean() if n else np.nan,
                "Independent Analog N": n,
                "Raw Candidate N": int(analogs.attrs.get("raw_matches", len(analogs))),
                "Average Similarity": avg_similarity,
                "Average Coverage": avg_coverage,
                "Concentration Warning": analogs.attrs.get("concentration_warning", "OK"),
                "Confidence": outlook_confidence(n, avg_coverage, avg_similarity, returns),
            }
        )
    return pd.DataFrame(rows), analogs


def build_asset_historical_outlook(analogs: pd.DataFrame) -> pd.DataFrame:
    """Summarize multiple assets over the one production analog episode set."""
    if analogs is None or analogs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for asset in ("SPY", "QQQ", "GLD", "BTC"):
        for horizon in FORWARD_HORIZONS:
            return_col = f"ForwardReturn_{asset}_{horizon}"
            drawdown_col = f"ForwardMaxDrawdown_{asset}_{horizon}"
            if return_col not in analogs or drawdown_col not in analogs:
                continue
            sample = analogs.dropna(subset=[return_col, drawdown_col])
            returns = pd.to_numeric(sample[return_col], errors="coerce")
            drawdowns = pd.to_numeric(sample[drawdown_col], errors="coerce")
            n = int(len(sample))
            avg_similarity = float(sample["Similarity"].mean()) if n else np.nan
            avg_coverage = float(sample["Coverage"].mean()) if n else np.nan
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "Median Forward Return": returns.median(),
                    "Mean Forward Return": returns.mean(),
                    "Positive Return Probability": (returns > 0).mean() if n else np.nan,
                    "P25 Return": returns.quantile(0.25) if n else np.nan,
                    "P75 Return": returns.quantile(0.75) if n else np.nan,
                    "Risk of >15% Drawdown": (drawdowns <= -0.15).mean() if n else np.nan,
                    "Risk of >25% Drawdown": (drawdowns <= -0.25).mean() if n else np.nan,
                    "Risk of >45% Drawdown": (drawdowns <= -0.45).mean() if n else np.nan,
                    "Independent Analog N": n,
                    "Average Similarity": avg_similarity,
                    "Average Coverage": avg_coverage,
                    "Concentration Warning": analogs.attrs.get("concentration_warning", "OK"),
                    "Confidence": outlook_confidence(n, avg_coverage, avg_similarity, returns),
                }
            )
    return pd.DataFrame(rows)


def select_analogs(w: pd.DataFrame, current: dict[str, Any], min_spacing_weeks: int = ANALOG_MIN_EPISODE_SPACING_WEEKS) -> pd.DataFrame:
    data = w.copy()
    latest_date = pd.Timestamp(current.get("Date"))
    data = data.loc[pd.to_datetime(data["Date"], errors="coerce") < latest_date - pd.Timedelta(weeks=52)].copy()
    if data.empty:
        return pd.DataFrame()
    current_series = pd.Series(current)
    detail_rows = []
    for _, row in data.iterrows():
        detail_rows.append(analog_similarity(row, current_series))
    details = pd.DataFrame(detail_rows, index=data.index)
    for col in details.columns:
        data[col] = details[col]
    data["AnalogTier"] = np.select([data["Coverage"] >= 0.85, data["Coverage"] >= 0.65], ["FULL ANALOG", "CORE ANALOG"], default="REJECT")
    threshold = 70.0
    matched = pd.DataFrame()
    while threshold >= 50.0:
        matched = data.loc[(data["Similarity"] >= threshold) & (data["Coverage"] >= 0.65)].copy()
        selected_preview = decluster_analogs(matched, min_spacing_weeks)
        if len(selected_preview) >= 30 or threshold <= 50:
            break
        threshold -= 5.0
    raw_matches = len(matched)
    selected = decluster_analogs(matched, min_spacing_weeks)
    selected.attrs["raw_matches"] = raw_matches
    selected.attrs["threshold"] = threshold
    selected.attrs["concentration_warning"] = analog_concentration_warning(selected)
    return selected.sort_values("Date")


def analog_similarity(row: pd.Series, current: pd.Series) -> dict[str, float]:
    structural, structural_cov = weighted_similarity(
        row,
        current,
        [
            ("SPX_ROC36M_3MMA", 0.50, "exp", 0.30),
            ("StructuralROCMomentum", 0.30, "exp", 0.18),
            ("StructuralExtensionDirection_12M", 0.20, "exp", 0.20),
        ],
    )
    momentum, momentum_cov = weighted_similarity(
        row,
        current,
        [
            ("SPX_ROC12M", 0.30, "exp", 0.25),
            ("SPX_ROC12M_3MMA", 0.45, "exp", 0.22),
            ("SPX_ROC_Momentum_3M", 0.25, "exp", 0.12),
        ],
    )
    sma200w, sma200w_cov = weighted_similarity(
        row,
        current,
        [
            ("SMA200WExtensionPercentile", 0.65, "percentile", 1.0),
            ("SMA200WExtensionPct", 0.35, "exp", 0.25),
        ],
    )
    current_risk, current_risk_cov = weighted_similarity(
        row,
        current,
        [
            ("HistoricalDrawdownRisk", 0.16, "points", 15.0),
            ("HistoricalHighBetaRisk", 0.16, "points", 25.0),
            ("HistoricalBreadthRisk", 0.16, "points", 25.0),
            ("HistoricalRSIDivergenceRisk", 0.12, "points", 30.0),
            ("HistoricalVIXLevelRisk", 0.10, "points", 25.0),
            ("HistoricalVIXMomentumRisk", 0.10, "points", 25.0),
            ("HistoricalVIXTermStructureRisk", 0.12, "points", 25.0),
            ("RealizedVolRisk", 0.08, "points", 25.0),
        ],
    )
    positioning, positioning_cov = weighted_similarity(
        row,
        current,
        [
            ("AAII_Bearish_Percentile", 0.50, "percentile", 1.0),
            ("VIX_AssetManager_Percentile", 0.50, "percentile", 1.0),
        ],
    )
    blocks = [
        ("StructuralSimilarity", structural, structural_cov),
        ("MomentumSimilarity", momentum, momentum_cov),
        ("SMA200WSimilarity", sma200w, sma200w_cov),
        ("CurrentRiskSimilarity", current_risk, current_risk_cov),
        ("PositioningSimilarity", positioning, positioning_cov),
    ]
    weighted_total = 0.0
    available_weight = 0.0
    coverage_weight = 0.0
    for name, value, cov in blocks:
        block_weight = ANALOG_FEATURE_WEIGHTS[name]
        coverage_weight += block_weight * cov
        if np.isfinite(value) and cov > 0:
            weighted_total += block_weight * value
            available_weight += block_weight
    similarity = weighted_total / available_weight if available_weight > 0 else np.nan
    coverage = coverage_weight / sum(ANALOG_FEATURE_WEIGHTS.values())
    return {
        "Similarity": similarity,
        "Coverage": coverage,
        "FeatureCoveragePct": coverage * 100.0,
        "StructuralSimilarity": structural,
        "MomentumSimilarity": momentum,
        "SMA200WSimilarity": sma200w,
        "CurrentRiskSimilarity": current_risk,
        "PositioningSimilarity": positioning,
    }


def weighted_similarity(row: pd.Series, current: pd.Series, features: list[tuple[str, float, str, float]]) -> tuple[float, float]:
    weighted = 0.0
    available = 0.0
    total = sum(weight for _, weight, _, _ in features)
    for col, weight, method, scale in features:
        a = safe_float(row.get(col))
        b = safe_float(current.get(col))
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if method == "percentile":
            score = np.clip(100.0 - abs(a - b), 0.0, 100.0)
        elif method == "points":
            score = np.clip(100.0 * np.exp(-abs(a - b) / scale), 0.0, 100.0)
        else:
            score = np.clip(100.0 * np.exp(-abs(a - b) / scale), 0.0, 100.0)
        weighted += weight * score
        available += weight
    if available <= 0:
        return np.nan, 0.0
    return float(weighted / available), float(available / total) if total > 0 else 0.0


def decluster_analogs(matched: pd.DataFrame, min_spacing_weeks: int) -> pd.DataFrame:
    if matched.empty:
        return matched.copy()
    ordered = matched.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.dropna(subset=["Date"]).sort_values("Date")
    episodes = []
    current_rows = []
    last_date: pd.Timestamp | None = None
    max_gap = pd.Timedelta(weeks=min_spacing_weeks)
    for _, row in ordered.iterrows():
        date = pd.Timestamp(row["Date"])
        if last_date is None or date - last_date <= max_gap:
            current_rows.append(row)
        else:
            episodes.append(pd.DataFrame(current_rows))
            current_rows = [row]
        last_date = date
    if current_rows:
        episodes.append(pd.DataFrame(current_rows))
    selected = []
    for episode in episodes:
        if "Similarity" in episode:
            best_idx = pd.to_numeric(episode["Similarity"], errors="coerce").idxmax()
        else:
            best_idx = episode.index[0]
        best = episode.loc[best_idx].copy()
        best["AnalogDate"] = best["Date"]
        best["AnalogEpisodeStart"] = episode["Date"].min()
        best["AnalogEpisodeEnd"] = episode["Date"].max()
        best["AnalogPeakSimilarity"] = pd.to_numeric(episode.get("Similarity", pd.Series(np.nan, index=episode.index)), errors="coerce").max()
        best["AnalogAverageSimilarity"] = pd.to_numeric(episode.get("Similarity", pd.Series(np.nan, index=episode.index)), errors="coerce").mean()
        selected.append(best)
    out = pd.DataFrame(selected)
    sort_cols = [col for col in ["Similarity", "Coverage"] if col in out]
    return out.sort_values(sort_cols, ascending=False).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def analog_concentration_warning(analogs: pd.DataFrame) -> str:
    if analogs.empty or "Date" not in analogs:
        return "INSUFFICIENT SAMPLE"
    dates = pd.to_datetime(analogs["Date"], errors="coerce").dropna()
    if dates.empty:
        return "INSUFFICIENT SAMPLE"
    total = len(dates)
    years = dates.dt.year
    five_year_windows = []
    for start in range(int(years.min()), int(years.max()) + 1):
        count = int(((years >= start) & (years <= start + 4)).sum())
        five_year_windows.append(count / total)
    if five_year_windows and max(five_year_windows) > 0.50:
        return "CONCENTRATION WARNING"
    decade_share = years.map(lambda y: (int(y) // 10) * 10).value_counts(normalize=True).max()
    if decade_share > 0.40:
        return "CONCENTRATION WARNING"
    return "OK"


def build_category_validation(w: pd.DataFrame, current: dict[str, Any]) -> pd.DataFrame:
    if not current:
        return pd.DataFrame()
    mask = (
        w["StructuralMarketCyclePhase"].astype(str).eq(str(current.get("StructuralMarketCyclePhase")))
        & w["MomentumCyclePhase"].astype(str).eq(str(current.get("MomentumCyclePhase")))
        & w["SMA200WZone"].astype(str).eq(str(current.get("SMA200WZone")))
    )
    sample = decluster_analogs(w.loc[mask].dropna(subset=["SPX_Close"]).copy(), 10)
    rows = []
    for horizon in FORWARD_HORIZONS:
        returns = pd.to_numeric(sample.get(f"ForwardReturn_{horizon}", pd.Series(dtype="float64")), errors="coerce").dropna()
        drawdowns = pd.to_numeric(sample.get(f"ForwardMaxDrawdown_{horizon}", pd.Series(dtype="float64")), errors="coerce").dropna()
        rows.append(
            {
                "Horizon": horizon,
                "Median Return": returns.median() if not returns.empty else np.nan,
                "Risk of >15% Drawdown": (drawdowns <= -0.15).mean() if not drawdowns.empty else np.nan,
                "N Independent Episodes": int(min(len(returns), len(drawdowns))),
                "Sample Warning": "LOW SAMPLE" if min(len(returns), len(drawdowns)) < 10 else "OK",
            }
        )
    return pd.DataFrame(rows)


def build_data_quality(raw: dict[str, pd.DataFrame], w: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in MARKET_CYCLE_TICKERS:
        frame = raw.get(ticker, pd.DataFrame())
        rows.append(
            {
                "Series": ticker,
                "FirstDate": frame.index.min().date() if not frame.empty else "",
                "LastDate": frame.index.max().date() if not frame.empty else "",
                "Observations": int(len(frame)),
                "Status": "OK" if not frame.empty else "MISSING",
            }
        )
    for col in ["StructuralMarketCyclePhase", "MomentumCyclePhase", "SMA200WZone", "CurrentMarketRiskState", "SPXAboveSMA50D", "SPXAboveSMA200D"]:
        rows.append(
            {
                "Series": col,
                "FirstDate": pd.to_datetime(w.loc[w[col].notna(), "Date"]).min().date() if col in w and w[col].notna().any() else "",
                "LastDate": pd.to_datetime(w.loc[w[col].notna(), "Date"]).max().date() if col in w and w[col].notna().any() else "",
                "Observations": int(w[col].notna().sum()) if col in w else 0,
                "Status": "OK" if col in w and w[col].notna().any() else "MISSING",
            }
        )
    return pd.DataFrame(rows)


def build_interpretation(current: dict[str, Any], outlook: pd.DataFrame) -> str:
    if not current:
        return "Market Cycle data is unavailable."
    lines = [
        (
            f"STRUCTURAL MARKET: SPX is in {current.get('StructuralMarketCyclePhase', 'n/a')}. "
            f"Price is {fmt_pct(current.get('StructuralExtensionPct'))} versus its 200M SMA, "
            f"at the {fmt_num(current.get('StructuralExtensionPercentile'))} percentile. "
            f"The secular cycle age is {fmt_num(current.get('StructuralCycleAgeMonths'), 0)} months and "
            f"36M structural ROC momentum is {current.get('StructuralROCMomentumState', 'n/a')}."
        ),
        (
            f"MOMENTUM: SPX momentum cycle is {current.get('MomentumCyclePhase', 'n/a')}. "
            f"12M ROC is {fmt_pct(current.get('SPX_ROC12M_3MMA'))}; "
            f"3M ROC momentum is {fmt_pct(current.get('SPX_ROC_Momentum_3M'))}."
        ),
        (
            f"EXTENSION: SPX is {current.get('SMA200WZone', 'n/a')} relative to its 200W SMA, "
            f"with raw extension of {fmt_pct(current.get('SMA200WExtensionPct'))}."
        ),
        (
            f"CURRENT RISK: Current Market Risk is {current.get('CurrentMarketRiskState', 'n/a')} "
            f"and direction is {current.get('CurrentMarketRiskDirection', 'n/a')}. "
            f"Active stress channels: {fmt_num(current.get('ActiveStressChannels'), 0)}."
        ),
    ]
    if outlook is not None and not outlook.empty:
        row = outlook.loc[outlook["Horizon"].eq("12M")]
        if not row.empty:
            r = row.iloc[0]
            lines.append(
                f"HISTORICAL OUTLOOK: Based on {int(r.get('Independent Analog N', 0))} independent analog episodes, "
                f"the median 12M forward SPX return was {fmt_pct(r.get('Median Forward Return'))}, "
                f"while {fmt_pct(r.get('Risk of >15% Drawdown'))} of analogous periods had a drawdown greater than 15%. "
                f"Confidence is {r.get('Confidence', 'n/a')}."
            )
    return "\n\n".join(lines)


def latest_current(w: pd.DataFrame) -> dict[str, Any]:
    if w.empty:
        return {}
    valid = w.dropna(subset=["Date", "SPX_Close"]).copy()
    return valid.iloc[-1].to_dict() if not valid.empty else {}


def expanding_percentile(series: pd.Series, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = []
    for idx, value in enumerate(values):
        hist = values.iloc[: idx + 1].dropna()
        if len(hist) < min_periods or not np.isfinite(safe_float(value)):
            out.append(np.nan)
        else:
            out.append(float((hist <= value).mean() * 100.0))
    return pd.Series(out, index=series.index)


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = []
    for idx, value in enumerate(values):
        hist = values.iloc[max(0, idx - window + 1) : idx + 1].dropna()
        if len(hist) < min_periods or not np.isfinite(safe_float(value)):
            out.append(np.nan)
        else:
            out.append(float((hist <= value).mean() * 100.0))
    return pd.Series(out, index=series.index)


def confirm_state(candidate: pd.Series, required: int = 2) -> pd.Series:
    out = []
    active = None
    run_value = None
    run_length = 0
    for value in candidate.astype("object"):
        if pd.isna(value) or value == "DATA INCOMPLETE":
            out.append(active or "DATA INCOMPLETE")
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
        out.append(active)
    return pd.Series(out, index=candidate.index)


def classify_structural_roc_momentum(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v > 0.03:
        return "ACCELERATING"
    if v < -0.03:
        return "DECELERATING"
    return "STABLE"


def classify_structural_phase(row: dict[str, Any]) -> str:
    pctl = safe_float(row.get("StructuralExtensionPercentile"))
    age = safe_float(row.get("StructuralAgePctOfHistoricalMedian"))
    roc_state = str(row.get("StructuralROCMomentumState", "DATA INCOMPLETE"))
    ext = safe_float(row.get("StructuralExtensionPct"))
    ext_dir = safe_float(row.get("StructuralExtensionDirection_12M"))
    if not np.isfinite(pctl) or not np.isfinite(ext):
        return "DATA INCOMPLETE"
    if ext < -0.12 and ext_dir < 0:
        return "STRUCTURAL CONTRACTION"
    if pctl >= 97.5 and age >= 0.75 and roc_state == "DECELERATING":
        return "STRUCTURAL TOP ZONE"
    if pctl >= 90 and age >= 0.65:
        return "LATE STRUCTURAL EXPANSION"
    if pctl >= 75 or age >= 0.55:
        return "MATURE STRUCTURAL EXPANSION"
    if pctl <= 45 and roc_state == "ACCELERATING" and age < 0.45:
        return "EARLY STRUCTURAL EXPANSION"
    return "MID STRUCTURAL EXPANSION"


def classify_structural_sma_roc_phase(row: dict[str, Any]) -> str:
    pctl = safe_float(row.get("StructuralExtensionPercentile"))
    ext = safe_float(row.get("StructuralExtensionPct"))
    roc = safe_float(row.get("SPX_ROC36M_3MMA"))
    roc_momentum = safe_float(row.get("StructuralROCMomentum"))
    roc_state = str(row.get("StructuralROCMomentumState", "DATA INCOMPLETE"))
    if not np.isfinite(pctl) or not np.isfinite(ext) or roc_state == "DATA INCOMPLETE":
        return "DATA INCOMPLETE"
    accelerating = roc_state == "ACCELERATING"
    decelerating = roc_state == "DECELERATING"
    roc_positive = np.isfinite(roc) and roc >= 0
    roc_negative = np.isfinite(roc) and roc < 0
    if (pctl <= 25 or ext <= 0) and accelerating and (np.isfinite(roc_momentum) and roc_momentum > 0):
        return "EARLY RECOVERY"
    if (ext <= 0 or pctl <= 35) and decelerating and (roc_negative or ext <= -0.05):
        return "CONTRACTION"
    if pctl >= 90 and decelerating:
        return "LATE EXPANSION"
    if pctl >= 75 and (roc_positive or not decelerating):
        return "MATURE EXPANSION"
    if pctl <= 55 and accelerating:
        return "EARLY EXPANSION"
    if roc_negative and decelerating and ext <= 0.05:
        return "CONTRACTION"
    return "EXPANSION"


def classify_momentum_phase(level: Any, momentum: Any) -> str:
    l = safe_float(level)
    m = safe_float(momentum)
    if not np.isfinite(l) or not np.isfinite(m):
        return "DATA INCOMPLETE"
    if l > 0 and m > 0:
        return "RISK EXPANSION"
    if l > 0 and m < 0:
        return "LATE RISK EXPANSION"
    if l < 0 and m < 0:
        return "RISK CONTRACTION"
    if l < 0 and m > 0:
        return "EARLY RISK RECOVERY"
    return "TRANSITION"


def classify_medium_sma_roc_phase(row: dict[str, Any]) -> str:
    momentum = str(row.get("MomentumCyclePhase", "DATA INCOMPLETE"))
    zone = str(row.get("SMA200WZone", "DATA INCOMPLETE"))
    pctl = safe_float(row.get("SMA200WExtensionPercentile"))
    if momentum == "DATA INCOMPLETE" or not np.isfinite(pctl):
        return "DATA INCOMPLETE"
    if momentum == "EARLY RISK RECOVERY" and zone in {"EXTREME OVERSOLD", "OVERSOLD", "NORMAL"}:
        return "EARLY RECOVERY"
    if momentum == "RISK EXPANSION" and pctl >= 75:
        return "STRETCHED EXPANSION"
    if momentum == "RISK EXPANSION":
        return "EXPANSION"
    if momentum == "LATE RISK EXPANSION":
        return "LATE EXPANSION"
    if momentum == "RISK CONTRACTION" and pctl <= 25:
        return "CAPITULATION / OVERSOLD"
    if momentum == "RISK CONTRACTION":
        return "CONTRACTION"
    return "EXPANSION"


def classify_sma200w_zone(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v < 10:
        return "EXTREME OVERSOLD"
    if v < 25:
        return "OVERSOLD"
    if v < 75:
        return "NORMAL"
    if v < 90:
        return "EXTENDED"
    if v < 97.5:
        return "OVEREXTENDED"
    return "EXTREME OVEREXTENSION"


def classify_relative_state(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v >= 80:
        return "STRONG POSITIVE"
    if v >= 60:
        return "POSITIVE"
    if v <= 20:
        return "STRONG NEGATIVE"
    if v <= 40:
        return "NEGATIVE"
    return "NEUTRAL"


def classify_breadth_participation(rsp_pctl: Any, iwm_pctl: Any) -> str:
    rsp = safe_float(rsp_pctl)
    iwm = safe_float(iwm_pctl)
    if not np.isfinite(rsp) or not np.isfinite(iwm):
        return "DATA INCOMPLETE"
    if rsp >= 60 and iwm >= 60:
        return "BROADENING"
    if rsp <= 40 and iwm <= 40:
        return "NARROWING"
    return "MIXED"


def classify_turning_signal(row: dict[str, Any]) -> str:
    phase = str(row.get("MomentumCyclePhase", ""))
    candidate = str(row.get("MomentumCycleCandidate", ""))
    qqq = str(row.get("QQQ_SPX_State", ""))
    btc = str(row.get("BTC_SPX_State", ""))
    breadth = str(row.get("BreadthParticipationState", ""))
    roc_mom = safe_float(row.get("SPX_ROC_Momentum_3M"))
    if phase == "RISK CONTRACTION" and (qqq in {"POSITIVE", "STRONG POSITIVE"} or btc in {"POSITIVE", "STRONG POSITIVE"} or breadth == "BROADENING"):
        return "EARLY IMPROVEMENT"
    if roc_mom > 0 and (qqq in {"POSITIVE", "STRONG POSITIVE"} or btc in {"POSITIVE", "STRONG POSITIVE"}):
        return "CONFIRMED IMPROVEMENT"
    if phase in {"RISK EXPANSION", "LATE RISK EXPANSION"} and candidate == "LATE RISK EXPANSION" and (qqq in {"NEGATIVE", "STRONG NEGATIVE"} or breadth == "NARROWING"):
        return "EARLY DETERIORATION"
    if candidate == "RISK CONTRACTION" and (qqq in {"NEGATIVE", "STRONG NEGATIVE"} or breadth == "NARROWING"):
        return "CONFIRMED DETERIORATION"
    return "NEUTRAL"


def add_high_beta_risk(w: pd.DataFrame) -> None:
    for asset in ["QQQ", "BTC"]:
        risk_13 = 100.0 - pd.to_numeric(w.get(f"{asset}_SPX_RS_13W_Pctl"), errors="coerce")
        risk_26 = 100.0 - pd.to_numeric(w.get(f"{asset}_SPX_RS_26W_Pctl"), errors="coerce")
        w[f"{asset}HighBetaRisk"] = (0.65 * risk_13 + 0.35 * risk_26).clip(0.0, 100.0)
    qqq = pd.to_numeric(w["QQQHighBetaRisk"], errors="coerce")
    btc = pd.to_numeric(w["BTCHighBetaRisk"], errors="coerce")
    combined = (0.70 * qqq + 0.30 * btc).clip(0.0, 100.0)
    btc_only = (btc >= 70.0) & (qqq < 50.0)
    w["HighBetaRisk"] = combined.where(~btc_only, np.minimum(combined, 69.0))
    w["HighBetaRiskState"] = w["HighBetaRisk"].map(classify_risk_state)
    change_4w = pd.to_numeric(w["HighBetaRisk"], errors="coerce").diff(4)
    w["HighBetaRiskDirection"] = np.select([change_4w >= 8.0, change_4w <= -8.0], ["RISING", "FALLING"], default="STABLE")


def add_breadth_risk(w: pd.DataFrame) -> None:
    above50 = pd.to_numeric(w.get("SPXAboveSMA50D"), errors="coerce")
    above200 = pd.to_numeric(w.get("SPXAboveSMA200D"), errors="coerce")
    if above50.notna().sum() >= 52 and above200.notna().sum() >= 52:
        w["Breadth50LevelRisk"] = 100.0 - expanding_percentile(above50, min_periods=52)
        w["Breadth200LevelRisk"] = 100.0 - expanding_percentile(above200, min_periods=52)
        w["BreadthLevelRisk"] = (0.60 * w["Breadth50LevelRisk"] + 0.40 * w["Breadth200LevelRisk"]).clip(0.0, 100.0)
        delta50 = above50.diff(4)
        delta200 = above200.diff(4)
        w["Delta4W_Above50D"] = delta50
        w["Delta4W_Above200D"] = delta200
        w["Breadth50MomentumRisk"] = 100.0 - expanding_percentile(delta50, min_periods=52)
        w["Breadth200MomentumRisk"] = 100.0 - expanding_percentile(delta200, min_periods=52)
        w["BreadthMomentumRisk"] = (0.60 * w["Breadth50MomentumRisk"] + 0.40 * w["Breadth200MomentumRisk"]).clip(0.0, 100.0)
        w["BreadthRiskSource"] = "TradingView MCP / S5FI,S5TH"
    else:
        rsp_13 = pd.to_numeric(w.get("RSP_SPX_RS_13W_Pctl"), errors="coerce")
        iwm_13 = pd.to_numeric(w.get("IWM_SPX_RS_13W_Pctl"), errors="coerce")
        rsp_4 = pd.to_numeric(w.get("RSP_SPX_RS_4W_Pctl"), errors="coerce")
        iwm_4 = pd.to_numeric(w.get("IWM_SPX_RS_4W_Pctl"), errors="coerce")
        w["Breadth50LevelRisk"] = np.nan
        w["Breadth200LevelRisk"] = np.nan
        w["Delta4W_Above50D"] = np.nan
        w["Delta4W_Above200D"] = np.nan
        w["Breadth50MomentumRisk"] = np.nan
        w["Breadth200MomentumRisk"] = np.nan
        w["BreadthLevelRisk"] = (0.60 * (100.0 - rsp_13) + 0.40 * (100.0 - iwm_13)).clip(0.0, 100.0)
        w["BreadthMomentumRisk"] = (0.60 * (100.0 - rsp_4) + 0.40 * (100.0 - iwm_4)).clip(0.0, 100.0)
        w["BreadthRiskSource"] = "Proxy fallback / RSP,IWM relative strength"
    w["BreadthRisk"] = (0.65 * w["BreadthLevelRisk"] + 0.35 * w["BreadthMomentumRisk"]).clip(0.0, 100.0)
    w["BreadthRiskState"] = w["BreadthRisk"].map(classify_risk_state)


def add_market_maturity(w: pd.DataFrame) -> None:
    dates = pd.to_datetime(w["Date"], errors="coerce")
    structural = cycle_maturity_fields(
        dates,
        pd.Timestamp("2009-02-28"),
        291.0,
        pd.to_numeric(w.get("StructuralExtensionPercentile"), errors="coerce"),
    )
    medium = cycle_maturity_fields(
        dates,
        pd.Timestamp("2022-10-31"),
        45.0,
        pd.to_numeric(w.get("SMA200WExtensionPercentile"), errors="coerce"),
    )
    for key, values in structural.items():
        w[f"Structural{key}"] = values
    for key, values in medium.items():
        w[f"Medium{key}"] = values


def cycle_maturity_fields(dates: pd.Series, start: pd.Timestamp, months: float, actual: pd.Series) -> dict[str, Any]:
    elapsed = ((dates - start).dt.days / 30.4375).astype("float64")
    progress = elapsed / months * 100.0
    in_cycle = elapsed >= 0
    reference = pd.Series(np.nan, index=dates.index, dtype="float64")
    reference.loc[in_cycle & (elapsed <= months)] = 100.0 * np.sin(np.pi * elapsed.loc[in_cycle & (elapsed <= months)] / months)
    reference.loc[in_cycle & (elapsed > months)] = 0.0
    gap = pd.to_numeric(actual, errors="coerce") - reference
    status = [
        classify_maturity_status(progress.iloc[idx], gap.iloc[idx])
        for idx in range(len(dates))
    ]
    return {
        "ReferenceCycleValue": reference,
        "CycleTimeProgressPct": progress.where(in_cycle),
        "ActualMaturity": actual,
        "CycleTimingGap": gap,
        "MaturityStatus": status,
    }


def classify_maturity_status(progress: Any, gap: Any) -> str:
    p = safe_float(progress)
    g = safe_float(gap)
    if not np.isfinite(p) or not np.isfinite(g):
        return "DATA INCOMPLETE"
    if p > 100:
        return "REFERENCE CYCLE EXPIRED"
    if g >= 15:
        return "LEADING REFERENCE"
    if g <= -15:
        return "LAGGING REFERENCE"
    return "ON TRACK"


def add_long_extension_amplitude(w: pd.DataFrame) -> None:
    required = {"Date", "SPX_Close", "SPX_SMA200W", "SMA200WExtensionPct", "StructuralExtensionPct", "StructuralExtensionPercentile"}
    if w.empty or not required.issubset(set(w.columns)):
        for col in [
            "PrimaryCycleComposite",
            "PrimaryMarketCycle",
            "PrimaryCycleTrough",
            "PrimaryCycleLastTroughDate",
            "PrimaryCyclePreviousTroughDate",
            "PrimaryCycleTroughToTroughMonths",
            "PrimaryCycleMonthsSinceTrough",
            "PrimaryCycleMaturityPct",
            "PrimaryCycleDirection",
            "StructuralExtensionSmooth",
            "StructuralMomentum12M",
            "StructuralDiagnosticRegime",
            "LongMarketExtensionCycle",
            "LongCycleTrough",
            "LongCycleLastTroughDate",
            "LongCyclePreviousTroughDate",
            "LongCycleTroughToTroughMonths",
            "LongCycleMonthsSinceTrough",
            "LongCycleMaturityPct",
            "LongCycleDirection",
            "LongCycleAmplitudeHilbert",
            "LongCycleAmplitudeCausal",
            "LongCycleAmplitudePercentile",
            "LongCycleAmplitudeState",
            "LongCycleAmplitudeDirection",
            "Corr_LongAmplitude_vs_StructuralExtension",
            "Corr_LongAmplitude_vs_StructuralMomentum",
            "StructuralTrough",
            "StructuralPeak",
            "StructuralLastTroughDate",
            "StructuralMonthsSinceTrough",
            "StructuralReferenceCycle",
            "StructuralReferenceFullLengthMonths",
            "StructuralReferenceTroughToPeakMonths",
            "StructuralReferenceProgressPct",
        ]:
            w[col] = "DATA INCOMPLETE" if "Direction" in col or "State" in col or col == "StructuralDiagnosticRegime" else np.nan
        return

    base = w[list(required)].copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    base["SPX_RSI14W"] = RSIIndicator(close=pd.to_numeric(w["SPX_Close"], errors="coerce"), window=14).rsi()
    monthly = base.dropna(subset=["Date"]).sort_values("Date").copy()
    monthly["_Month"] = monthly["Date"].dt.to_period("M")
    monthly = monthly.groupby("_Month", sort=True).tail(1).copy()
    monthly["Date"] = monthly["_Month"].dt.to_timestamp("M")
    monthly = monthly.drop(columns=["_Month"]).reset_index(drop=True)
    if monthly.empty:
        return
    close_m = pd.to_numeric(monthly["SPX_Close"], errors="coerce")
    monthly["SPX_ROC12M_Monthly"] = close_m.pct_change(12, fill_method=None)
    early_sma200m = close_m.rolling(200, min_periods=60).mean()
    early_structural_ext = (close_m / early_sma200m - 1.0) * 100.0
    structural_ext = (pd.to_numeric(monthly["StructuralExtensionPct"], errors="coerce") * 100.0).combine_first(early_structural_ext)
    distance_200w = pd.to_numeric(monthly["SMA200WExtensionPct"], errors="coerce") * 100.0
    primary_components = pd.DataFrame(
        {
            "roc": pd.to_numeric(monthly["SPX_ROC12M_Monthly"], errors="coerce"),
            "extension": distance_200w,
            "rsi": pd.to_numeric(monthly["SPX_RSI14W"], errors="coerce"),
        },
        index=monthly.index,
    )
    primary_common = primary_components.dropna()
    primary_zscores = primary_components.copy()
    for column in primary_components.columns:
        primary_zscores[column] = (primary_components[column] - primary_common[column].mean()) / primary_common[column].std(ddof=0)
    monthly["PrimaryCycleComposite"] = primary_zscores.mean(axis=1, skipna=False)
    monthly["PrimaryMarketCycle"] = standard_zscore(
        fft_bandpass_cycle(monthly["PrimaryCycleComposite"], min_period=30.0, max_period=54.0)
    )
    add_cycle_trough_metadata(
        monthly,
        cycle_col="PrimaryMarketCycle",
        prefix="PrimaryCycle",
        min_spacing_months=24.0,
        window_months=4,
        full_cycle_months=40.0,
    )
    monthly["PrimaryCycleDirection"] = cycle_direction(monthly["PrimaryMarketCycle"])

    monthly["StructuralExtensionSmooth"] = structural_ext.rolling(3, min_periods=2).mean()
    monthly["StructuralMomentum12M"] = monthly["StructuralExtensionSmooth"] - monthly["StructuralExtensionSmooth"].shift(12)
    monthly["StructuralDiagnosticRegime"] = [
        classify_structural_diagnostic_regime(ext, mom)
        for ext, mom in zip(monthly["StructuralExtensionSmooth"], monthly["StructuralMomentum12M"])
    ]
    monthly["LongMarketExtensionCycle"] = standard_zscore(
        fft_bandpass_cycle(distance_200w, min_period=75.0, max_period=105.0)
    )
    add_cycle_trough_metadata(
        monthly,
        cycle_col="LongMarketExtensionCycle",
        prefix="LongCycle",
        min_spacing_months=60.0,
        window_months=8,
        full_cycle_months=83.0,
    )
    monthly["LongCycleDirection"] = cycle_direction(monthly["LongMarketExtensionCycle"])
    monthly["LongCycleAmplitudeHilbert"] = hilbert_amplitude(monthly["LongMarketExtensionCycle"])
    monthly["LongCycleAmplitudeCausal"] = distance_200w.rolling(84, min_periods=48).std()
    monthly["LongCycleAmplitudePercentile"] = expanding_percentile(monthly["LongCycleAmplitudeCausal"], min_periods=48)
    monthly["LongCycleAmplitudeState"] = monthly["LongCycleAmplitudePercentile"].map(classify_long_amplitude_state)
    amp_change_12m = pd.to_numeric(monthly["LongCycleAmplitudeCausal"], errors="coerce").diff(12)
    monthly["LongCycleAmplitudeDirection"] = amp_change_12m.map(classify_amplitude_direction)
    corr_ext = rolling_corr_full(monthly["LongCycleAmplitudeCausal"], monthly["StructuralExtensionSmooth"])
    corr_mom = rolling_corr_full(monthly["LongCycleAmplitudeCausal"], monthly["StructuralMomentum12M"])
    monthly["Corr_LongAmplitude_vs_StructuralExtension"] = corr_ext
    monthly["Corr_LongAmplitude_vs_StructuralMomentum"] = corr_mom
    add_structural_turning_points(monthly)
    add_structural_reference_cycle(monthly)

    keep = [
        "Date",
        "PrimaryCycleComposite",
        "PrimaryMarketCycle",
        "PrimaryCycleTrough",
        "PrimaryCycleLastTroughDate",
        "PrimaryCyclePreviousTroughDate",
        "PrimaryCycleTroughToTroughMonths",
        "PrimaryCycleMonthsSinceTrough",
        "PrimaryCycleMaturityPct",
        "PrimaryCycleDirection",
        "StructuralExtensionSmooth",
        "StructuralMomentum12M",
        "StructuralDiagnosticRegime",
        "LongMarketExtensionCycle",
        "LongCycleTrough",
        "LongCycleLastTroughDate",
        "LongCyclePreviousTroughDate",
        "LongCycleTroughToTroughMonths",
        "LongCycleMonthsSinceTrough",
        "LongCycleMaturityPct",
        "LongCycleDirection",
        "LongCycleAmplitudeHilbert",
        "LongCycleAmplitudeCausal",
        "LongCycleAmplitudePercentile",
        "LongCycleAmplitudeState",
        "LongCycleAmplitudeDirection",
        "Corr_LongAmplitude_vs_StructuralExtension",
        "Corr_LongAmplitude_vs_StructuralMomentum",
        "StructuralTrough",
        "StructuralPeak",
        "StructuralLastTroughDate",
        "StructuralMonthsSinceTrough",
        "StructuralMaturityPct",
        "StructuralReferenceCycle",
        "StructuralReferenceFullLengthMonths",
        "StructuralReferenceTroughToPeakMonths",
        "StructuralReferenceProgressPct",
    ]
    merged = pd.merge_asof(
        w[["Date"]].sort_values("Date"),
        monthly[keep].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    for col in keep:
        if col != "Date":
            w[col] = merged[col].to_numpy()


def classify_structural_diagnostic_regime(extension: Any, momentum: Any) -> str:
    ext = safe_float(extension)
    mom = safe_float(momentum)
    if not np.isfinite(ext) or not np.isfinite(mom):
        return "DATA INCOMPLETE"
    if ext <= 0 and mom > 0:
        return "STRUCTURAL RECOVERY"
    if ext > 0 and mom > 0:
        return "STRUCTURAL EXPANSION"
    if ext > 0 and mom <= 0:
        return "STRUCTURAL DECLINE"
    return "STRUCTURAL CONTRACTION / TROUGH"


def classify_long_amplitude_state(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v < 20:
        return "COMPRESSED"
    if v < 40:
        return "LOW"
    if v < 60:
        return "NORMAL"
    if v < 80:
        return "HIGH"
    return "EXPANDED"


def classify_amplitude_direction(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v > 1.0:
        return "RISING"
    if v < -1.0:
        return "FALLING"
    return "STABLE"


def fft_bandpass_cycle(series: pd.Series, min_period: float, max_period: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < int(max_period):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    # Run the full-history FFT only on the available cycle input. Leading
    # pre-indicator rows are not observations and must not be padded into the
    # spectrum, otherwise the boundary padding shifts the extracted phase.
    segment = values.loc[valid.index[0] : valid.index[-1]].interpolate(limit_direction="both")
    filled = segment
    demeaned = filled - filled.mean()
    n = len(demeaned)
    spectrum = np.fft.rfft(demeaned.to_numpy())
    freqs = np.fft.rfftfreq(n, d=1.0)
    low_freq = 1.0 / max_period
    high_freq = 1.0 / min_period
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    filtered = np.fft.irfft(spectrum * mask, n=n)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[segment.index] = filtered
    return out


def hilbert_amplitude(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < 24:
        return pd.Series(np.nan, index=series.index)
    filled = values.interpolate(limit_direction="both")
    x = filled.to_numpy(dtype="float64")
    n = len(x)
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    out = pd.Series(np.abs(analytic), index=series.index)
    out.loc[values.isna()] = np.nan
    return out


def rolling_corr_full(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(frame) < 24:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def standard_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < 24:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - valid.mean()) / std


def cycle_direction(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    slope = values.diff(3)
    valid = values.dropna()
    threshold = max(float(valid.std()) * 0.05, 0.03) if len(valid) >= 24 and np.isfinite(valid.std()) else 0.03
    states = []
    for value in slope:
        v = safe_float(value)
        if not np.isfinite(v):
            states.append("DATA INCOMPLETE")
        elif v > threshold:
            states.append("UPSWING")
        elif v < -threshold:
            states.append("DOWNSWING")
        else:
            states.append("TURNING / FLAT")
    return confirm_state(pd.Series(states, index=series.index), required=2)


def add_cycle_trough_metadata(
    monthly: pd.DataFrame,
    cycle_col: str,
    prefix: str,
    min_spacing_months: float,
    window_months: int,
    full_cycle_months: float,
) -> None:
    values = pd.to_numeric(monthly.get(cycle_col), errors="coerce")
    dates = pd.to_datetime(monthly.get("Date"), errors="coerce")
    trough_idx = detect_spaced_extrema(values, dates, mode="min", min_spacing_months=min_spacing_months, window_months=window_months)
    trough_dates = [pd.Timestamp(dates.iloc[idx]) for idx in trough_idx if pd.notna(dates.iloc[idx])]
    trough_set = set(trough_idx)
    monthly[f"{prefix}Trough"] = [idx in trough_set for idx in range(len(monthly))]
    last_dates = []
    prev_dates = []
    durations = []
    months_since = []
    for date in dates:
        prior = [trough for trough in trough_dates if pd.notna(date) and trough <= date]
        if not prior:
            last_dates.append(pd.NaT)
            prev_dates.append(pd.NaT)
            durations.append(np.nan)
            months_since.append(np.nan)
            continue
        last = prior[-1]
        prev = prior[-2] if len(prior) >= 2 else pd.NaT
        last_dates.append(last)
        prev_dates.append(prev)
        durations.append(months_between(prev, last) if pd.notna(prev) else np.nan)
        months_since.append(months_between(last, date))
    monthly[f"{prefix}LastTroughDate"] = last_dates
    monthly[f"{prefix}PreviousTroughDate"] = prev_dates
    monthly[f"{prefix}TroughToTroughMonths"] = durations
    monthly[f"{prefix}MonthsSinceTrough"] = months_since
    monthly[f"{prefix}MaturityPct"] = pd.to_numeric(monthly[f"{prefix}MonthsSinceTrough"], errors="coerce") / full_cycle_months * 100.0


def add_structural_turning_points(monthly: pd.DataFrame) -> None:
    smooth = pd.to_numeric(monthly.get("StructuralExtensionSmooth"), errors="coerce")
    dates = pd.to_datetime(monthly.get("Date"), errors="coerce")
    trough_idx = detect_spaced_extrema(
        smooth,
        dates,
        mode="min",
        min_spacing_months=180.0,
        window_months=18,
        threshold=-10.0,
    )
    peak_idx = detect_spaced_extrema(
        smooth,
        dates,
        mode="max",
        min_spacing_months=120.0,
        window_months=18,
        threshold=50.0,
    )
    trough_set = set(trough_idx)
    peak_set = set(peak_idx)
    trough_dates = [pd.Timestamp(dates.iloc[idx]) for idx in trough_idx if pd.notna(dates.iloc[idx])]
    monthly["StructuralTrough"] = [idx in trough_set for idx in range(len(monthly))]
    monthly["StructuralPeak"] = [idx in peak_set for idx in range(len(monthly))]
    last_dates = []
    months_since = []
    for date in dates:
        prior = [trough for trough in trough_dates if pd.notna(date) and trough <= date]
        if prior:
            last_dates.append(prior[-1])
            months_since.append(months_between(prior[-1], date))
        else:
            last_dates.append(pd.NaT)
            months_since.append(np.nan)
    monthly["StructuralLastTroughDate"] = last_dates
    monthly["StructuralMonthsSinceTrough"] = months_since
    monthly["StructuralMaturityPct"] = pd.to_numeric(monthly["StructuralMonthsSinceTrough"], errors="coerce") / 461.0 * 100.0


def add_structural_reference_cycle(monthly: pd.DataFrame) -> None:
    trough_to_peak = 291.0
    full_length = 461.0
    elapsed = pd.to_numeric(monthly.get("StructuralMonthsSinceTrough"), errors="coerce")
    reference = []
    progress = []
    for value in elapsed:
        months = safe_float(value)
        if not np.isfinite(months):
            reference.append(np.nan)
            progress.append(np.nan)
            continue
        if months <= trough_to_peak:
            ref = 100.0 * np.sin((np.pi / 2.0) * months / trough_to_peak)
        elif months <= full_length:
            ref = 100.0 * np.cos((np.pi / 2.0) * (months - trough_to_peak) / (full_length - trough_to_peak))
        else:
            ref = 0.0
        reference.append(float(np.clip(ref, 0.0, 100.0)))
        progress.append(float(min(months / full_length * 100.0, 100.0)))
    monthly["StructuralReferenceCycle"] = reference
    monthly["StructuralReferenceFullLengthMonths"] = full_length
    monthly["StructuralReferenceTroughToPeakMonths"] = trough_to_peak
    monthly["StructuralReferenceProgressPct"] = progress


def detect_spaced_extrema(
    values: pd.Series,
    dates: pd.Series,
    mode: str,
    min_spacing_months: float,
    window_months: int,
    threshold: float | None = None,
) -> list[int]:
    candidates: list[int] = []
    for idx in range(window_months, len(values) - window_months):
        value = safe_float(values.iloc[idx])
        date = dates.iloc[idx] if idx < len(dates) else pd.NaT
        if not np.isfinite(value) or pd.isna(date):
            continue
        if threshold is not None:
            if mode == "min" and value >= threshold:
                continue
            if mode == "max" and value <= threshold:
                continue
        window = pd.to_numeric(values.iloc[idx - window_months : idx + window_months + 1], errors="coerce").dropna()
        if window.empty:
            continue
        is_extreme = value <= window.min() if mode == "min" else value >= window.max()
        if is_extreme:
            candidates.append(idx)
    selected: list[int] = []
    for idx in candidates:
        if not selected:
            selected.append(idx)
            continue
        distance = months_between(dates.iloc[selected[-1]], dates.iloc[idx])
        if np.isfinite(distance) and distance < min_spacing_months:
            replace = values.iloc[idx] < values.iloc[selected[-1]] if mode == "min" else values.iloc[idx] > values.iloc[selected[-1]]
            if bool(replace):
                selected[-1] = idx
        else:
            selected.append(idx)
    return selected


def months_between(start: Any, end: Any) -> float:
    start_ts = pd.Timestamp(start) if pd.notna(start) else pd.NaT
    end_ts = pd.Timestamp(end) if pd.notna(end) else pd.NaT
    if pd.isna(start_ts) or pd.isna(end_ts):
        return np.nan
    return float((end_ts - start_ts).days / 30.4375)


def add_forward_crash_risk(w: pd.DataFrame, min_spacing_weeks: int = 8) -> None:
    dates = pd.to_datetime(w["Date"], errors="coerce")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    min_spacing = np.timedelta64(int(min_spacing_weeks * 7), "D")
    drawdowns = pd.to_numeric(w.get("ForwardMaxDrawdown_1M"), errors="coerce").to_numpy(dtype="float64")
    phases = w.get("MomentumCyclePhase", pd.Series("DATA INCOMPLETE", index=w.index)).astype(str).to_numpy()
    pctls = pd.to_numeric(w.get("SMA200WExtensionPercentile"), errors="coerce").to_numpy(dtype="float64")

    def spaced_indices(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= 1:
            return indices
        kept: list[int] = []
        last_date: np.datetime64 | None = None
        for raw_idx in indices:
            current_date = date_values[int(raw_idx)]
            if last_date is None or current_date - last_date >= min_spacing:
                kept.append(int(raw_idx))
                last_date = current_date
        return np.asarray(kept, dtype=int)

    probs: list[float] = []
    ns: list[int] = []
    medians: list[float] = []
    p10s: list[float] = []
    confidences: list[str] = []
    sample_modes: list[str] = []
    all_indices = np.arange(len(w), dtype=int)
    for idx in range(len(w)):
        phase = phases[idx]
        pctl = safe_float(pctls[idx])
        if phase == "DATA INCOMPLETE" or not np.isfinite(pctl):
            probs.append(np.nan)
            ns.append(0)
            medians.append(np.nan)
            p10s.append(np.nan)
            confidences.append("INSUFFICIENT SAMPLE")
            sample_modes.append("DATA INCOMPLETE")
            continue
        eligible = all_indices[(date_values < date_values[idx] - np.timedelta64(28, "D")) & np.isfinite(drawdowns)]
        selected_idx = np.asarray([], dtype=int)
        sample_mode = "MATCHED_BAND"
        for band in [10.0, 15.0, 20.0]:
            raw = eligible[(phases[eligible] == phase) & np.isfinite(pctls[eligible]) & (np.abs(pctls[eligible] - pctl) <= band)]
            selected_idx = spaced_indices(raw)
            if len(selected_idx) >= 10 or band == 20.0:
                break
        if len(selected_idx) < 10:
            phase_selected = spaced_indices(eligible[phases[eligible] == phase])
            if len(phase_selected) > len(selected_idx):
                selected_idx = phase_selected
                sample_mode = "PHASE_ONLY_FALLBACK"
        if len(selected_idx) < 10:
            broad_selected = spaced_indices(eligible)
            if len(broad_selected) > len(selected_idx):
                selected_idx = broad_selected
                sample_mode = "BROAD_HISTORY_FALLBACK"
        dd = drawdowns[selected_idx]
        dd = dd[np.isfinite(dd)]
        n = int(len(dd))
        probs.append(float((dd <= -0.20).mean() * 100.0) if n else np.nan)
        ns.append(n)
        medians.append(float(np.median(dd)) if n else np.nan)
        p10s.append(float(np.quantile(dd, 0.10)) if n else np.nan)
        confidences.append(drawdown_probability_confidence(n, sample_mode))
        sample_modes.append(sample_mode)
    w["SPX_1M_DD20_Probability"] = probs
    w["SPX_1M_DD20_AnalogN"] = ns
    w["SPX_1M_MedianMaxDrawdown"] = medians
    w["SPX_1M_MaxDrawdownP10"] = p10s
    w["SPX_1M_DD20_Confidence"] = confidences
    w["SPX_1M_DD20_SampleMode"] = sample_modes


def drawdown_probability_confidence(n: int, sample_mode: str = "MATCHED_BAND") -> str:
    suffix = "" if sample_mode == "MATCHED_BAND" else f" ({sample_mode})"
    if n >= 30:
        return f"HIGH{suffix}"
    if n >= 20:
        return f"MEDIUM{suffix}"
    if n >= 10:
        return f"LOW{suffix}"
    return f"VERY LOW SAMPLE{suffix}" if n > 0 else "INSUFFICIENT SAMPLE"


def add_rsi_divergence_risk(w: pd.DataFrame) -> None:
    close = pd.to_numeric(w["SPX_Close"], errors="coerce")
    rsi = RSIIndicator(close=close, window=14).rsi()
    peak = (close == close.rolling(5, center=True, min_periods=3).max()) & close.notna() & rsi.notna()
    prior_peak_idx: int | None = None
    active: dict[str, Any] | None = None
    rows = []
    for idx in range(len(w)):
        if bool(peak.iloc[idx]):
            if prior_peak_idx is not None and idx - prior_peak_idx >= 8:
                price_gain = close.iloc[idx] / close.iloc[prior_peak_idx] - 1.0
                drop = rsi.iloc[prior_peak_idx] - rsi.iloc[idx]
                if price_gain > 0 and drop >= 3.0:
                    active = {
                        "start_idx": prior_peak_idx,
                        "start_date": w["Date"].iloc[prior_peak_idx],
                        "prior_rsi": rsi.iloc[prior_peak_idx],
                        "latest_peak_idx": idx,
                        "latest_peak_price": close.iloc[idx],
                    }
            if prior_peak_idx is None or close.iloc[idx] >= close.iloc[prior_peak_idx]:
                prior_peak_idx = idx
        if active is not None:
            if np.isfinite(rsi.iloc[idx]) and rsi.iloc[idx] >= safe_float(active.get("prior_rsi")):
                active = None
        if active is None:
            rows.append((False, "NONE", 0.0, pd.NaT, np.nan, np.nan, np.nan))
            continue
        duration = idx - int(active["start_idx"])
        drop = safe_float(active.get("prior_rsi")) - safe_float(rsi.iloc[idx])
        price_gain = safe_float(close.iloc[idx] / close.iloc[int(active["start_idx"])] - 1.0)
        score, state = rsi_divergence_score(duration, drop, price_gain)
        rows.append((True, state, score, active["start_date"], duration, drop, price_gain))
    out = pd.DataFrame(
        rows,
        columns=[
            "RSIDivergenceActive",
            "RSIDivergenceState",
            "RSIDivergenceRiskScore",
            "RSIDivergenceStartDate",
            "RSIDivergenceDurationWeeks",
            "RSIDropPoints",
            "RSIDivergencePriceGain",
        ],
    )
    for col in out.columns:
        w[col] = out[col].to_numpy()


def rsi_divergence_score(duration: Any, drop: Any, price_gain: Any) -> tuple[float, str]:
    weeks = safe_float(duration)
    rsi_drop = safe_float(drop)
    gain = safe_float(price_gain)
    if not np.isfinite(weeks) or not np.isfinite(rsi_drop) or rsi_drop < 3.0 or weeks < 8:
        return 0.0, "NONE"
    score = 15.0
    if rsi_drop >= 5:
        score += 15.0
    if rsi_drop >= 8:
        score += 20.0
    if rsi_drop >= 12:
        score += 15.0
    if 13 <= weeks <= 20:
        score += 20.0
    elif 8 <= weeks <= 12:
        score += 8.0
    elif 21 <= weeks <= 30:
        score += 12.0
    elif weeks > 30:
        score += 8.0
    if np.isfinite(gain) and gain >= 0.08:
        score += 10.0
    if weeks > 30 and not (rsi_drop >= 12 and np.isfinite(gain) and gain >= 0.08):
        score = min(score, 70.0)
    score = float(np.clip(score, 0.0, 100.0))
    return score, classify_risk_state(score, none_label="NONE")


def vix_level_risk_score(value: Any) -> float:
    v = safe_float(value)
    if not np.isfinite(v):
        return np.nan
    return float(np.interp(v, [10.0, 15.0, 20.0, 25.0, 35.0, 50.0], [5.0, 20.0, 45.0, 65.0, 85.0, 100.0]))


def vix_term_structure_risk_score(value: Any) -> float:
    v = safe_float(value)
    if not np.isfinite(v):
        return np.nan
    return float(np.interp(v, [0.75, 0.85, 0.95, 1.00, 1.10, 1.20], [5.0, 20.0, 45.0, 65.0, 90.0, 100.0]))


def classify_risk_state(value: Any, none_label: str = "LOW") -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v <= 0 and none_label == "NONE":
        return "NONE"
    if v < 30:
        return none_label if none_label == "NONE" else "LOW"
    if v < 50:
        return "NORMAL"
    if v < 70:
        return "ELEVATED"
    if v < 90:
        return "HIGH"
    return "EXTREME"


def classify_vix_term_structure(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v >= 1.12:
        return "SEVERELY_INVERTED"
    if v >= 1.0:
        return "INVERTED"
    if v >= 0.95:
        return "FLAT"
    return "NORMAL"


def classify_short_breadth(rsp_4w: Any, iwm_4w: Any) -> str:
    rsp = safe_float(rsp_4w)
    iwm = safe_float(iwm_4w)
    if not np.isfinite(rsp) or not np.isfinite(iwm):
        return "DATA INCOMPLETE"
    if rsp < -0.04 and iwm < -0.04:
        return "STRESSED"
    if rsp < -0.02 or iwm < -0.02:
        return "WEAKENING"
    if rsp > 0.02 and iwm > 0.02:
        return "LOW"
    return "NORMAL"


def active_stress_channels(row: dict[str, Any]) -> int:
    count = 0
    for col in [
        "HighBetaRisk",
        "BreadthRisk",
        "RSIDivergenceRiskScore",
        "RealizedVolRisk",
        "VIXTermStructureRisk",
        "VIXMomentumRisk",
        "VIXLevelRisk",
    ]:
        count += safe_float(row.get(col)) >= 70
    count += safe_float(row.get("SPX_1M_DD20_Probability")) >= 20
    return int(count)


def current_risk_supporting_score(row: dict[str, Any]) -> float:
    early_values = [safe_float(row.get(col)) for col in ["HighBetaRisk", "BreadthRisk", "RSIDivergenceRiskScore"]]
    vol_values = [safe_float(row.get(col)) for col in ["VIXLevelRisk", "VIXMomentumRisk", "VIXTermStructureRisk", "RealizedVolRisk"]]
    early_clean = [value for value in early_values if np.isfinite(value)]
    vol_clean = [value for value in vol_values if np.isfinite(value)]
    early = float(np.mean(early_clean)) if early_clean else np.nan
    vol = float(np.mean(vol_clean)) if vol_clean else np.nan
    crash = safe_float(row.get("SPX_1M_DD20_Probability"))
    pieces = [value for value in [early, vol, crash] if np.isfinite(value)]
    return float(np.clip(max(pieces), 0.0, 100.0)) if pieces else np.nan


def current_risk_primary_drivers(row: dict[str, Any]) -> str:
    drivers: list[str] = []
    thresholds = [
        ("Drawdown Risk", "SPX_1M_DD20_Probability", 10.0),
        ("High Beta Risk", "HighBetaRisk", 70.0),
        ("Breadth Risk", "BreadthRisk", 70.0),
        ("RSI Divergence Risk", "RSIDivergenceRiskScore", 70.0),
        ("VIX Level", "VIXLevelRisk", 70.0),
        ("VIX Momentum", "VIXMomentumRisk", 70.0),
        ("VIX Term Structure", "VIXTermStructureRisk", 70.0),
        ("Realized Volatility", "RealizedVolRisk", 70.0),
    ]
    for label, col, threshold in thresholds:
        value = safe_float(row.get(col))
        if np.isfinite(value) and value >= threshold:
            drivers.append(label)
    if not drivers:
        return "None"
    return ", ".join(drivers[:4])


def add_historical_state_duration(w: pd.DataFrame, state_col: str, start_col: str, weeks_col: str) -> None:
    dates = pd.to_datetime(w["Date"], errors="coerce")
    states = w[state_col].astype(str)
    starts = []
    weeks = []
    active_state = None
    active_start = pd.NaT
    for date, state in zip(dates, states):
        if pd.isna(date) or state in {"", "nan", "NaN", "DATA INCOMPLETE"}:
            starts.append(pd.NaT)
            weeks.append(np.nan)
            continue
        if state != active_state:
            active_state = state
            active_start = date
        starts.append(active_start)
        weeks.append(float((date - active_start).days / 7.0) if pd.notna(active_start) else np.nan)
    w[start_col] = starts
    w[weeks_col] = weeks


def classify_current_risk(row: dict[str, Any]) -> str:
    high_beta = safe_float(row.get("HighBetaRisk"))
    breadth = safe_float(row.get("BreadthRisk"))
    rsi = safe_float(row.get("RSIDivergenceRiskScore"))
    crash = safe_float(row.get("SPX_1M_DD20_Probability"))
    vix_level = safe_float(row.get("VIXLevelRisk"))
    vix_momentum = safe_float(row.get("VIXMomentumRisk"))
    term = safe_float(row.get("VIXTermStructureRisk"))
    realized = safe_float(row.get("RealizedVolRisk"))
    early_high = sum(safe_float(v) >= 70 for v in [high_beta, breadth, rsi])
    early_elevated = sum(safe_float(v) >= 50 for v in [high_beta, breadth, rsi])
    vol_high = sum(safe_float(v) >= 70 for v in [vix_level, vix_momentum, term, realized])
    vol_elevated = sum(safe_float(v) >= 50 for v in [vix_level, vix_momentum, term, realized])
    crash_elevated = np.isfinite(crash) and crash >= 10
    crash_high = np.isfinite(crash) and crash >= 20
    if term >= 90 and (vix_level >= 85 or realized >= 85) and (early_high >= 1 or crash_high):
        return "ACUTE"
    if (
        (high_beta >= 70 and breadth >= 70 and vol_elevated >= 1)
        or (crash_high and term >= 70)
        or (realized >= 70 and vix_level >= 70 and early_elevated >= 1)
        or vol_high >= 3
    ):
        return "HIGH"
    if early_high >= 2 or (early_high >= 1 and crash_elevated) or vol_elevated >= 2 or vix_momentum >= 70:
        return "ELEVATED"
    if vol_elevated == 0 and early_elevated == 0 and (not np.isfinite(crash) or crash < 5):
        return "LOW"
    return "NORMAL"


def classify_positioning_vulnerability(value: Any) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "DATA INCOMPLETE"
    if v >= 90:
        return "EXTREME"
    if v >= 75:
        return "CROWDED"
    if v >= 60:
        return "ELEVATED"
    if v >= 35:
        return "NORMAL"
    return "LOW"


def classify_vulnerability(row: dict[str, Any]) -> str:
    points = 0
    points += safe_float(row.get("StructuralExtensionPercentile")) >= 90
    points += safe_float(row.get("StructuralAgePctOfHistoricalMedian")) >= 0.75
    points += str(row.get("StructuralROCMomentumState")) == "DECELERATING"
    points += safe_float(row.get("SMA200WExtensionPercentile")) >= 90
    points += str(row.get("PositioningVulnerability")) in {"CROWDED", "EXTREME"}
    if points >= 4:
        return "HIGH"
    if points >= 2:
        return "ELEVATED"
    if points == 1:
        return "NORMAL"
    return "LOW"


def forward_max_drawdown(close: pd.Series, weeks: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce")
    out = []
    for idx, value in enumerate(values):
        if not np.isfinite(safe_float(value)) or idx + weeks >= len(values):
            out.append(np.nan)
            continue
        path = values.iloc[idx + 1 : idx + weeks + 1].dropna()
        out.append(float(path.min() / value - 1.0) if not path.empty else np.nan)
    return pd.Series(out, index=close.index)


def forward_max_drawdown_from_low(close: pd.Series, low: pd.Series, weeks: int) -> pd.Series:
    closes = pd.to_numeric(close, errors="coerce")
    lows = pd.to_numeric(low, errors="coerce")
    out = []
    for idx, value in enumerate(closes):
        if not np.isfinite(safe_float(value)) or idx + weeks >= len(closes):
            out.append(np.nan)
            continue
        path = lows.iloc[idx + 1 : idx + weeks + 1].dropna()
        out.append(float(path.min() / value - 1.0) if not path.empty else np.nan)
    return pd.Series(out, index=close.index)


def outlook_confidence(n: int, coverage: float, similarity: float, returns: pd.Series) -> str:
    dispersion = returns.quantile(0.75) - returns.quantile(0.25) if len(returns.dropna()) >= 5 else np.nan
    if n < 10:
        return "INSUFFICIENT HISTORICAL SAMPLE"
    if n >= 50 and coverage >= 0.80 and similarity >= 70 and (not np.isfinite(dispersion) or dispersion < 0.30):
        return "HIGH"
    if n >= 25 and coverage >= 0.65:
        return "MEDIUM"
    return "LOW"


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def fmt_pct(value: Any) -> str:
    v = safe_float(value)
    return "n/a" if not np.isfinite(v) else f"{v:.1%}"


def fmt_num(value: Any, decimals: int = 1) -> str:
    v = safe_float(value)
    return "n/a" if not np.isfinite(v) else f"{v:.{decimals}f}"

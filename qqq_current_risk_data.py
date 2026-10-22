from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv
from fred_client import download_fred_series
from historical_archive import read_raw_table
from hy_oas import combine_hy_oas_sources, weekly_archive_available_frame
from qqq_current_risk import build_qqq_weekly_context, calculate_qqq_current_risk_variants


QQQ_CURRENT_RISK_TICKERS = ("QQQ", "IWM", "^VIX", "^VXN")
NDFI_SYMBOL = "INDEX:NDFI"
HY_OAS_SERIES = "BAMLH0A0HYM2"


@dataclass(frozen=True)
class QQQCurrentRiskSnapshot:
    shared: pd.DataFrame
    variants: dict[str, pd.DataFrame]
    source_status: dict[str, str]
    as_of: pd.Timestamp | None


def build_qqq_current_risk_snapshot(end_date: Any = None) -> QQQCurrentRiskSnapshot:
    """Load one immutable market snapshot and calculate both VIX/VXN variants."""
    end = _normalize_end_date(end_date)
    qqq = _load_market("QQQ", archive_key="QQQ_RAW", auto_adjust=False, end=end)
    iwm = _load_market("IWM", archive_key="IWM_RAW", auto_adjust=False, end=end)
    vix = _load_market("^VIX", archive_key="^VIX", auto_adjust=True, end=end)
    vxn = _load_market("^VXN", archive_key="^VXN", auto_adjust=True, end=end)
    ndfi, ndfi_status = load_ndfi_history(end=end)
    hy_oas, hy_status = load_hy_oas_history(end=end)

    status = {
        "QQQ": _frame_status(qqq),
        "IWM": _frame_status(iwm),
        "VIX": _frame_status(vix),
        "VXN": _frame_status(vxn),
        "NDFI": ndfi_status,
        "HY_OAS": hy_status,
    }
    if qqq.empty:
        return QQQCurrentRiskSnapshot(pd.DataFrame(), {}, status, None)

    daily = pd.DataFrame({"Date": pd.to_datetime(qqq.index, errors="coerce")})
    daily = daily.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    daily["QQQ_RawClose"] = _values_on_dates(qqq, daily["Date"])
    daily["IWM_RawClose"] = _values_on_dates(iwm, daily["Date"])
    daily["VIX"] = _values_on_dates(vix, daily["Date"])
    daily["VXN"] = _values_on_dates(vxn, daily["Date"])
    daily["NDFI"] = _values_on_dates(ndfi, daily["Date"])

    if not hy_oas.empty:
        daily = pd.merge_asof(
            daily.sort_values("Date"),
            hy_oas.sort_values("Date"),
            on="Date",
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        daily["HY_OAS"] = np.nan
        daily["HYOASSourceFrequency"] = "UNAVAILABLE"

    weekly = build_qqq_weekly_context(qqq, now=end)
    if not weekly.empty:
        daily = pd.merge_asof(
            daily.sort_values("Date"),
            weekly.sort_values("Date"),
            on="Date",
            direction="backward",
            allow_exact_matches=True,
        )
    variants = calculate_qqq_current_risk_variants(daily)
    as_of = pd.Timestamp(daily["Date"].max()).normalize() if not daily.empty else None
    return QQQCurrentRiskSnapshot(daily.reset_index(drop=True), variants, status, as_of)


def load_ndfi_history(end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    """Load Nasdaq breadth from TradingView and fall back to the two-version archive."""
    try:
        from tradingview_mcp import get_ohlcv_data

        live = get_ohlcv_data(NDFI_SYMBOL, interval="1D", count=5000)
        frame = _tradingview_close_frame(live)
        frame = _validate_ndfi(frame, end=end)
        if not frame.empty:
            return frame, "LIVE"
    except Exception:
        pass
    archived = read_raw_table("TRADINGVIEW", "NDFI")
    if not archived.empty:
        date_column = "date" if "date" in archived else "Date"
        close_column = "close" if "close" in archived else "Value"
        frame = pd.DataFrame(
            {"Close": pd.to_numeric(archived[close_column], errors="coerce").to_numpy()},
            index=pd.to_datetime(archived[date_column], errors="coerce"),
        )
        frame.attrs["source_status"] = archived.attrs.get("archive_status", "ARCHIVED")
        frame = _validate_ndfi(frame, end=end)
        if not frame.empty:
            return frame, str(frame.attrs.get("source_status", "ARCHIVED"))
    return _empty_market_frame(), "UNAVAILABLE"


def load_hy_oas_history(end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    baseline = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])
    baseline_path = Path(__file__).with_name("data") / "BAMLH0A0HYM2_weekly.csv"
    if baseline_path.exists():
        try:
            baseline = weekly_archive_available_frame(pd.read_csv(baseline_path, usecols=["time", "close"]))
        except Exception:
            baseline = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])

    tradingview = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])
    try:
        from tradingview_mcp import get_ohlcv_data

        tv = get_ohlcv_data(f"FRED:{HY_OAS_SERIES}", interval="1D", count=5000)
        if not tv.empty:
            tradingview = pd.DataFrame(
                {
                    "Date": pd.to_datetime(tv["date"], errors="coerce"),
                    "HY_OAS": pd.to_numeric(tv["close"], errors="coerce"),
                    "HYOASSourceFrequency": "DAILY_TRADINGVIEW",
                }
            ).dropna(subset=["Date", "HY_OAS"])
    except Exception:
        pass

    fred = pd.DataFrame(columns=["Date", "HY_OAS", "HYOASSourceFrequency"])
    fred_status = "UNAVAILABLE"
    try:
        raw = download_fred_series(HY_OAS_SERIES, observation_start="1993-01-01")
        fred_status = str(raw.attrs.get("source_status", "LIVE"))
        if not raw.empty:
            fred = pd.DataFrame(
                {
                    "Date": pd.to_datetime(raw["Date"], errors="coerce"),
                    "HY_OAS": pd.to_numeric(raw["Value"], errors="coerce"),
                    "HYOASSourceFrequency": "DAILY_FRED",
                }
            ).dropna(subset=["Date", "HY_OAS"])
    except Exception:
        pass
    combined = combine_hy_oas_sources(baseline, tradingview, fred, end_date=end)
    if combined.empty:
        return combined, "UNAVAILABLE"
    status = "LIVE" if not tradingview.empty or fred_status == "LIVE" else fred_status
    return combined, status


def _load_market(
    ticker: str,
    *,
    archive_key: str,
    auto_adjust: bool,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    frame = download_completed_ohlcv(
        ticker,
        period="max",
        auto_adjust=auto_adjust,
        archive_key=archive_key,
        allow_archive_fallback=True,
    )
    if frame.empty:
        return frame
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if end is not None:
        frame = frame.loc[frame.index <= end]
    return frame


def _tradingview_close_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame or "close" not in frame:
        return _empty_market_frame()
    out = pd.DataFrame(
        {"Close": pd.to_numeric(frame["close"], errors="coerce").to_numpy()},
        index=pd.to_datetime(frame["date"], errors="coerce"),
    )
    out = out.loc[~out.index.isna()].dropna(subset=["Close"]).sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def _validate_ndfi(frame: pd.DataFrame, end: pd.Timestamp | None) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return _empty_market_frame()
    attrs = dict(frame.attrs)
    out = frame.copy()
    # TradingView daily bars carry the exchange-session timestamp (for example
    # 13:30) while Yahoo daily bars are indexed at midnight.  NDFI is a daily
    # observation, so retain the calendar date rather than the intraday time;
    # otherwise every exact alignment against QQQ produces NaN.
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.loc[~out.index.isna() & out["Close"].between(0.0, 100.0)].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    if end is not None:
        out = out.loc[out.index <= end]
    out.attrs.update(attrs)
    return out


def _values_on_dates(frame: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    if frame is None or frame.empty or "Close" not in frame:
        return np.full(len(dates), np.nan)
    values = pd.to_numeric(frame["Close"], errors="coerce")
    values.index = pd.to_datetime(values.index, errors="coerce").tz_localize(None).normalize()
    values = values.loc[~values.index.isna()]
    values = values.loc[~values.index.duplicated(keep="last")]
    target_dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).tz_localize(None).normalize()
    return values.reindex(target_dates).to_numpy()


def _frame_status(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "UNAVAILABLE"
    return str(frame.attrs.get("source_status", "LIVE"))


def _normalize_end_date(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return (timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp).normalize()


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

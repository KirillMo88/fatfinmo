from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from threading import Thread
import time
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd

from finance_core import download_completed_ohlcv


BYBIT_ASSET_MAP = {
    "BTC-USD": {"symbol": "BTCUSDT", "category": "linear"},
    "ETH-USD": {"symbol": "ETHUSDT", "category": "linear"},
    "SOL-USD": {"symbol": "SOLUSDT", "category": "linear"},
    "SUI-USD": {"symbol": "SUIUSDT", "category": "linear"},
    "TRX-USD": {"symbol": "TRXUSDT", "category": "linear"},
    "HYPE-USD": {"symbol": "HYPEUSDT", "category": "linear"},
}

BYBIT_CONFIG = {
    "base_url": "https://api.bybit.com",
    "timeout": 15,
    "max_retries": 3,
    "target_start_date": "2018-03-01",
    "history_ttl_seconds": 21600,
    "ticker_ttl_seconds": 600,
    "instrument_ttl_seconds": 86400,
}

BYBIT_EXCHANGE = "BYBIT"
_UPDATE_THREAD: Thread | None = None
BYBIT_STORAGE_PATH = Path(
    os.environ.get(
        "BYBIT_DERIVATIVES_STORAGE",
        str(Path(__file__).with_name("persistent") / "bybit_derivatives" / "weekly_bybit_derivatives.csv"),
    )
)
WEEKLY_COLUMNS = [
    "timestamp",
    "date",
    "asset",
    "exchange",
    "exchange_symbol",
    "category",
    "price",
    "open_interest_raw",
    "open_interest_usd",
    "oi_change_1d_pct",
    "oi_change_1w_pct",
    "oi_change_4w_pct",
    "oi_change_13w_pct",
    "oi_change_4w_percentile",
    "funding_rate",
    "funding_1d",
    "funding_7d",
    "funding_28d",
    "funding_28d_percentile",
    "mark_price",
    "index_price",
    "perp_premium_pct",
    "perp_premium_7d_avg",
    "perp_premium_28d_avg",
    "perp_premium_percentile",
    "volume_24h",
    "turnover_24h",
    "oi_price_regime",
    "history_start_date",
    "data_status",
    "last_updated",
    "outlier_flags",
]


class BybitAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class DerivativesObservation:
    exchange: str
    asset: str
    exchange_symbol: str
    timestamp: pd.Timestamp
    open_interest_raw: float = math.nan
    open_interest_usd: float = math.nan
    funding_rate: float = math.nan
    mark_price: float = math.nan
    index_price: float = math.nan
    premium_pct: float = math.nan
    volume: float = math.nan
    turnover: float = math.nan


@dataclass(frozen=True)
class CryptoDerivativesState:
    asset: str
    exchange: str
    exchange_symbol: str
    price: float = math.nan
    open_interest_usd: float = math.nan
    oi_change_1w_pct: float = math.nan
    oi_change_4w_pct: float = math.nan
    oi_change_13w_pct: float = math.nan
    oi_change_4w_percentile: float = math.nan
    funding_current: float = math.nan
    funding_7d: float = math.nan
    funding_28d: float = math.nan
    funding_percentile: float = math.nan
    perp_premium_pct: float = math.nan
    premium_28d_avg: float = math.nan
    premium_percentile: float = math.nan
    oi_price_regime: str = "n/a"
    history_start_date: str = "n/a"
    last_updated: str = "n/a"
    data_status: str = "ERROR"


class PublicBybitClient:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**BYBIT_CONFIG, **(config or {})}
        self.base_url = str(self.config["base_url"]).rstrip("/")

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(int(self.config["max_retries"])):
            try:
                with httpx.Client(timeout=float(self.config["timeout"])) as client:
                    response = client.get(url, params=params)
                if response.status_code == 429:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))
                    continue
                response.raise_for_status()
                payload = response.json()
                if int(payload.get("retCode", -1)) != 0:
                    raise BybitAPIError(str(payload.get("retMsg") or payload))
                return payload
            except Exception as exc:
                last_error = exc
                time.sleep(min(0.75 * (2**attempt), 6.0))
        raise BybitAPIError(str(last_error) if last_error else "Bybit request failed")


def start_background_update_if_stale(path: Path = BYBIT_STORAGE_PATH, force: bool = False) -> bool:
    global _UPDATE_THREAD
    if _UPDATE_THREAD is not None and _UPDATE_THREAD.is_alive():
        return True
    if not force and is_storage_fresh(path):
        return False
    _UPDATE_THREAD = Thread(target=_safe_background_update, args=(path, force), daemon=True)
    _UPDATE_THREAD.start()
    return True


def bybit_update_in_progress() -> bool:
    return _UPDATE_THREAD is not None and _UPDATE_THREAD.is_alive()


def _safe_background_update(path: Path, force: bool) -> None:
    try:
        update_all_bybit_assets(path=path, force=force)
    except Exception:
        pass


def asset_config(asset: str) -> dict[str, str]:
    if asset not in BYBIT_ASSET_MAP:
        raise KeyError(f"Unsupported Bybit asset: {asset}")
    return BYBIT_ASSET_MAP[asset]


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)


def to_ms(value: pd.Timestamp | str | datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def from_ms(value: Any) -> pd.Timestamp:
    return pd.to_datetime(pd.to_numeric(value, errors="coerce"), unit="ms", utc=True).tz_localize(None)


def safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return math.nan
        out = float(value)
        return out if np.isfinite(out) else math.nan
    except Exception:
        return math.nan


def validate_bybit_instrument(asset: str, client: PublicBybitClient | None = None) -> dict[str, Any]:
    cfg = asset_config(asset)
    client = client or PublicBybitClient()
    payload = client.get(
        "/v5/market/instruments-info",
        {"category": cfg["category"], "symbol": cfg["symbol"], "limit": 1000},
    )
    instruments = payload.get("result", {}).get("list", []) or []
    instrument = next((item for item in instruments if item.get("symbol") == cfg["symbol"]), None)
    if not instrument:
        return {"asset": asset, **cfg, "available": False, "data_status": "INSTRUMENT_UNAVAILABLE"}
    is_active = str(instrument.get("status")) == "Trading"
    is_perpetual = str(instrument.get("contractType")) == "LinearPerpetual"
    is_linear = cfg["category"] == "linear" and str(payload.get("result", {}).get("category", cfg["category"])) == "linear"
    available = bool(is_active and is_perpetual and is_linear)
    return {
        "asset": asset,
        **cfg,
        "available": available,
        "data_status": "CURRENT" if available else "INSTRUMENT_UNAVAILABLE",
        "instrument": instrument,
    }


def load_bybit_ticker(asset: str, client: PublicBybitClient | None = None) -> dict[str, Any]:
    cfg = asset_config(asset)
    client = client or PublicBybitClient()
    payload = client.get("/v5/market/tickers", {"category": cfg["category"], "symbol": cfg["symbol"]})
    tickers = payload.get("result", {}).get("list", []) or []
    ticker = next((item for item in tickers if item.get("symbol") == cfg["symbol"]), tickers[0] if tickers else {})
    ticker = dict(ticker)
    ticker["timestamp"] = payload.get("time") or to_ms(now_utc())
    return ticker


def paginated_history(
    client: PublicBybitClient,
    path: str,
    params: dict[str, Any],
    timestamp_field: str,
    start_ms: int,
    end_ms: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    end_ms = end_ms or to_ms(now_utc())
    cursor = ""
    seen_cursors: set[str] = set()
    current_end = end_ms
    while current_end >= start_ms:
        request_params = {**params, "startTime": start_ms, "endTime": current_end, "limit": 200}
        if cursor:
            request_params["cursor"] = cursor
        payload = client.get(path, request_params)
        result = payload.get("result", {}) or {}
        page = result.get("list", []) or []
        if not page:
            break
        rows.extend(page)
        timestamps = pd.to_numeric([item.get(timestamp_field) for item in page], errors="coerce")
        timestamps = [int(value) for value in timestamps if np.isfinite(value)]
        if not timestamps:
            break
        min_ts = min(timestamps)
        if min_ts <= start_ms:
            break
        next_cursor = str(result.get("nextPageCursor") or "")
        if next_cursor and next_cursor not in seen_cursors:
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            cursor = ""
            current_end = min_ts - 1
    return rows


def load_bybit_open_interest(
    asset: str,
    client: PublicBybitClient | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    cfg = asset_config(asset)
    client = client or PublicBybitClient()
    start_ms = to_ms(start or BYBIT_CONFIG["target_start_date"])
    end_ms = to_ms(end or now_utc())
    rows = paginated_history(
        client,
        "/v5/market/open-interest",
        {"category": cfg["category"], "symbol": cfg["symbol"], "intervalTime": "1d"},
        "timestamp",
        start_ms,
        end_ms,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open_interest_raw"])
    frame["timestamp"] = frame["timestamp"].map(from_ms)
    frame["open_interest_raw"] = frame["openInterest"].map(safe_float)
    if "openInterestValue" in frame.columns:
        frame["open_interest_usd"] = frame["openInterestValue"].map(safe_float)
    return frame[["timestamp", "open_interest_raw"] + (["open_interest_usd"] if "open_interest_usd" in frame.columns else [])].dropna(subset=["timestamp"])


def load_bybit_funding_history(
    asset: str,
    client: PublicBybitClient | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    cfg = asset_config(asset)
    client = client or PublicBybitClient()
    start_ms = to_ms(start or BYBIT_CONFIG["target_start_date"])
    end_ms = to_ms(end or now_utc())
    rows = paginated_history(
        client,
        "/v5/market/funding/history",
        {"category": cfg["category"], "symbol": cfg["symbol"]},
        "fundingRateTimestamp",
        start_ms,
        end_ms,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    frame["timestamp"] = frame["fundingRateTimestamp"].map(from_ms)
    frame["funding_rate"] = frame["fundingRate"].map(safe_float)
    return frame[["timestamp", "funding_rate"]].dropna(subset=["timestamp"])


def load_existing_price_series(asset: str, start: str | pd.Timestamp | None = None) -> pd.Series:
    ohlcv = download_completed_ohlcv(asset, period="max")
    if ohlcv.empty or "Close" not in ohlcv.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    if start is not None:
        close = close.loc[pd.Timestamp(start).normalize() :]
    return close.sort_index()


def trailing_percentile(series: pd.Series, window: int = 156, min_periods: int = 104) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def rank_last(x: np.ndarray) -> float:
        clean = x[np.isfinite(x)]
        if len(clean) < min_periods or not np.isfinite(clean[-1]):
            return math.nan
        return float((clean <= clean[-1]).sum() / len(clean) * 100.0)

    return values.rolling(window=window, min_periods=min_periods).apply(rank_last, raw=True)


def normalize_daily_oi(oi: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    if oi.empty:
        return pd.DataFrame(columns=["date", "open_interest_raw", "open_interest_usd"])
    d = oi.copy()
    d["date"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"]).sort_values("timestamp").groupby("date", as_index=False).last()
    d["open_interest_raw"] = pd.to_numeric(d["open_interest_raw"], errors="coerce")
    if "open_interest_usd" not in d.columns:
        d["open_interest_usd"] = np.nan
    d["price"] = align_price_on_or_before(price, d["date"])
    missing_usd = pd.to_numeric(d["open_interest_usd"], errors="coerce").isna()
    d.loc[missing_usd, "open_interest_usd"] = d.loc[missing_usd, "open_interest_raw"] * d.loc[missing_usd, "price"]
    d["oi_change_1d_pct"] = d["open_interest_usd"].pct_change(1, fill_method=None)
    d["oi_change_1w_pct"] = d["open_interest_usd"].pct_change(7, fill_method=None)
    d["oi_change_4w_pct"] = d["open_interest_usd"].pct_change(28, fill_method=None)
    d["oi_change_13w_pct"] = d["open_interest_usd"].pct_change(91, fill_method=None)
    return d


def normalize_daily_funding(funding: pd.DataFrame) -> pd.DataFrame:
    if funding.empty:
        return pd.DataFrame(columns=["date", "funding_rate", "funding_1d", "funding_7d", "funding_28d"])
    d = funding.copy()
    d["date"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"]).sort_values("timestamp")
    daily_sum = d.groupby("date")["funding_rate"].sum().rename("funding_1d")
    daily_last = d.groupby("date")["funding_rate"].last().rename("funding_rate")
    out = pd.concat([daily_last, daily_sum], axis=1).sort_index().reset_index()
    out["funding_7d"] = out["funding_1d"].rolling(7, min_periods=1).sum()
    out["funding_28d"] = out["funding_1d"].rolling(28, min_periods=1).sum()
    return out


def align_price_on_or_before(price: pd.Series, dates: pd.Series) -> pd.Series:
    if price.empty:
        return pd.Series(np.nan, index=dates.index)
    price_frame = pd.DataFrame({"date": pd.to_datetime(price.index), "price": pd.to_numeric(price.to_numpy(), errors="coerce")})
    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.normalize()
    target = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize(), "_idx": dates.index})
    merged = pd.merge_asof(target.sort_values("date"), price_frame.sort_values("date"), on="date", direction="backward")
    return merged.set_index("_idx").reindex(dates.index)["price"]


def weekly_last(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if frame.empty:
        return frame
    d = frame.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    d["date"] = d[date_col].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    return d.groupby("date", as_index=False).last()


def build_ticker_frame(asset: str, ticker: dict[str, Any]) -> pd.DataFrame:
    ts = from_ms(ticker.get("timestamp") or to_ms(now_utc()))
    mark_price = safe_float(ticker.get("markPrice"))
    index_price = safe_float(ticker.get("indexPrice"))
    last_price = safe_float(ticker.get("lastPrice"))
    basis_rate = safe_float(ticker.get("basisRate"))
    basis = safe_float(ticker.get("basis"))
    if np.isfinite(basis_rate):
        premium_pct = basis_rate * 100.0
    elif np.isfinite(mark_price) and np.isfinite(index_price) and index_price > 0:
        premium_pct = (mark_price / index_price - 1.0) * 100.0
    else:
        premium_pct = math.nan
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "date": ts.normalize(),
                "asset": asset,
                "price": mark_price if np.isfinite(mark_price) else last_price,
                "open_interest_raw": safe_float(ticker.get("openInterest")),
                "open_interest_usd": safe_float(ticker.get("openInterestValue")),
                "funding_rate": safe_float(ticker.get("fundingRate")),
                "mark_price": mark_price,
                "index_price": index_price,
                "basis": basis,
                "basis_rate": basis_rate,
                "perp_premium_pct": premium_pct,
                "volume_24h": safe_float(ticker.get("volume24h")),
                "turnover_24h": safe_float(ticker.get("turnover24h")),
            }
        ]
    )


def build_weekly_layer(
    asset: str,
    instrument: dict[str, Any],
    oi: pd.DataFrame,
    funding: pd.DataFrame,
    ticker: dict[str, Any],
    price: pd.Series,
) -> pd.DataFrame:
    cfg = asset_config(asset)
    daily_oi = normalize_daily_oi(oi, price)
    daily_funding = normalize_daily_funding(funding)
    weekly_oi = weekly_last(daily_oi)
    weekly_funding = weekly_last(daily_funding)
    weekly_price = weekly_last(pd.DataFrame({"date": pd.to_datetime(price.index), "price": pd.to_numeric(price.to_numpy(), errors="coerce")}))
    weekly_ticker = weekly_last(build_ticker_frame(asset, ticker))
    weekly = weekly_price[["date", "price"]] if not weekly_price.empty else pd.DataFrame(columns=["date", "price"])
    for part in [weekly_oi, weekly_funding, weekly_ticker]:
        if part.empty:
            continue
        weekly = weekly.merge(part.drop(columns=["price"], errors="ignore"), on="date", how="outer")
    weekly = weekly.sort_values("date").reset_index(drop=True)
    if "price_x" in weekly.columns:
        weekly["price"] = weekly["price_x"]
    if "price_y" in weekly.columns:
        weekly["price"] = weekly.get("price", pd.Series(np.nan, index=weekly.index)).combine_first(weekly["price_y"])
    weekly = weekly.drop(columns=["price_x", "price_y"], errors="ignore")
    if "price" not in weekly.columns:
        weekly["price"] = np.nan
    ticker_week = weekly_ticker[["date", "price", "open_interest_raw", "open_interest_usd", "funding_rate", "mark_price", "index_price", "perp_premium_pct", "volume_24h", "turnover_24h"]]
    if not ticker_week.empty:
        weekly = merge_ticker_week(weekly, ticker_week)
    weekly["oi_change_4w_percentile"] = trailing_percentile(weekly.get("oi_change_4w_pct", pd.Series(np.nan, index=weekly.index)))
    weekly["funding_28d_percentile"] = trailing_percentile(weekly.get("funding_28d", pd.Series(np.nan, index=weekly.index)))
    weekly["perp_premium_7d_avg"] = pd.to_numeric(weekly.get("perp_premium_pct", np.nan), errors="coerce").rolling(1, min_periods=1).mean()
    weekly["perp_premium_28d_avg"] = pd.to_numeric(weekly.get("perp_premium_pct", np.nan), errors="coerce").rolling(4, min_periods=1).mean()
    weekly["perp_premium_percentile"] = trailing_percentile(weekly.get("perp_premium_pct", pd.Series(np.nan, index=weekly.index)))
    weekly["oi_price_regime"] = classify_oi_price_regime(weekly["price"].pct_change(1, fill_method=None), weekly.get("oi_change_1w_pct", pd.Series(np.nan, index=weekly.index)))
    history_start = fmt_date(weekly["date"].min()) if not weekly.empty else "n/a"
    updated = fmt_datetime(now_utc())
    weekly["timestamp"] = weekly["date"].dt.strftime("%Y-%m-%d")
    weekly["asset"] = asset
    weekly["exchange"] = BYBIT_EXCHANGE
    weekly["exchange_symbol"] = cfg["symbol"]
    weekly["category"] = cfg["category"]
    weekly["history_start_date"] = history_start
    weekly["data_status"] = freshness_status(weekly["date"].max() if not weekly.empty else None)
    weekly["last_updated"] = updated
    weekly["outlier_flags"] = build_outlier_flags(weekly)
    for column in WEEKLY_COLUMNS:
        if column not in weekly.columns:
            weekly[column] = np.nan
    return weekly[WEEKLY_COLUMNS].drop_duplicates(subset=["exchange", "asset", "exchange_symbol", "timestamp"]).sort_values("date")


def merge_ticker_week(weekly: pd.DataFrame, ticker_week: pd.DataFrame) -> pd.DataFrame:
    out = weekly.merge(ticker_week, on="date", how="outer", suffixes=("", "_ticker"))
    for column in ["price", "open_interest_raw", "open_interest_usd", "funding_rate", "mark_price", "index_price", "perp_premium_pct", "volume_24h", "turnover_24h"]:
        ticker_col = f"{column}_ticker"
        if ticker_col in out.columns:
            base = pd.to_numeric(out.get(column), errors="coerce")
            ticker_values = pd.to_numeric(out[ticker_col], errors="coerce")
            out[column] = base.mask(ticker_values.notna(), ticker_values)
    return out.drop(columns=[col for col in out.columns if col.endswith("_ticker")], errors="ignore")


def classify_oi_price_regime(price_return_1w: pd.Series, oi_change_1w: pd.Series) -> pd.Series:
    labels = []
    for px, oi in zip(pd.to_numeric(price_return_1w, errors="coerce"), pd.to_numeric(oi_change_1w, errors="coerce")):
        if not np.isfinite(px) or not np.isfinite(oi) or px == 0 or oi == 0:
            labels.append("n/a")
        elif px > 0 and oi > 0:
            labels.append("PRICE_UP_OI_UP")
        elif px > 0 and oi < 0:
            labels.append("PRICE_UP_OI_DOWN")
        elif px < 0 and oi > 0:
            labels.append("PRICE_DOWN_OI_UP")
        else:
            labels.append("PRICE_DOWN_OI_DOWN")
    return pd.Series(labels, index=price_return_1w.index)


def build_outlier_flags(weekly: pd.DataFrame) -> pd.Series:
    flags = []
    oi_change = pd.to_numeric(weekly.get("oi_change_1d_pct", np.nan), errors="coerce")
    premium = pd.to_numeric(weekly.get("perp_premium_pct", np.nan), errors="coerce")
    for oi, prem in zip(oi_change, premium):
        row_flags = []
        if np.isfinite(oi) and abs(oi) > 0.50:
            row_flags.append("OI_DAILY_CHANGE_OUTLIER_CHECK")
        if np.isfinite(prem) and abs(prem) > 2.0:
            row_flags.append("PREMIUM_OUTLIER_CHECK")
        flags.append(";".join(row_flags))
    return pd.Series(flags, index=weekly.index)


def freshness_status(latest_date: Any) -> str:
    if latest_date is None or pd.isna(latest_date):
        return "ERROR"
    age_days = (now_utc().normalize() - pd.Timestamp(latest_date).normalize()).days
    if age_days <= 3:
        return "CURRENT"
    if age_days <= 7:
        return "DELAYED"
    if age_days <= 21:
        return "STALE"
    return "ERROR"


def status_frame(asset: str, status: str) -> pd.DataFrame:
    cfg = asset_config(asset)
    ts = now_utc()
    row = {column: np.nan for column in WEEKLY_COLUMNS}
    row.update(
        {
            "timestamp": ts.strftime("%Y-%m-%d"),
            "date": ts.normalize(),
            "asset": asset,
            "exchange": BYBIT_EXCHANGE,
            "exchange_symbol": cfg["symbol"],
            "category": cfg["category"],
            "history_start_date": "n/a",
            "data_status": status,
            "last_updated": fmt_datetime(ts),
            "outlier_flags": "",
        }
    )
    return pd.DataFrame([row], columns=WEEKLY_COLUMNS)


def read_bybit_storage(path: Path = BYBIT_STORAGE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    frame = pd.read_csv(path)
    for column in WEEKLY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame[WEEKLY_COLUMNS]


def write_bybit_storage(frame: pd.DataFrame, path: Path = BYBIT_STORAGE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(subset=["exchange", "asset", "exchange_symbol", "timestamp"], keep="last")
    out.to_csv(path, index=False)


def update_bybit_asset(
    asset: str,
    existing: pd.DataFrame | None = None,
    client: PublicBybitClient | None = None,
    price_loader: Callable[[str, str | pd.Timestamp | None], pd.Series] = load_existing_price_series,
) -> pd.DataFrame:
    client = client or PublicBybitClient()
    existing = existing.copy() if existing is not None else pd.DataFrame(columns=WEEKLY_COLUMNS)
    try:
        validation = validate_bybit_instrument(asset, client)
        if not validation.get("available"):
            return merge_asset_history(existing, status_frame(asset, "INSTRUMENT_UNAVAILABLE"))
        asset_existing = existing[existing["asset"].eq(asset)].copy() if not existing.empty else pd.DataFrame(columns=WEEKLY_COLUMNS)
        start = BYBIT_CONFIG["target_start_date"]
        if not asset_existing.empty:
            latest = pd.to_datetime(asset_existing["date"], errors="coerce").max()
            if pd.notna(latest):
                start = max(pd.Timestamp(BYBIT_CONFIG["target_start_date"]), latest - pd.Timedelta(days=220))
        price = price_loader(asset, start)
        oi = load_bybit_open_interest(asset, client, start=start)
        funding = load_bybit_funding_history(asset, client, start=start)
        ticker = load_bybit_ticker(asset, client)
        weekly_new = build_weekly_layer(asset, validation.get("instrument", {}), oi, funding, ticker, price)
        return merge_asset_history(existing, weekly_new)
    except Exception:
        return merge_asset_history(existing, status_frame(asset, "ERROR"))


def update_all_bybit_assets(
    path: Path = BYBIT_STORAGE_PATH,
    client: PublicBybitClient | None = None,
    force: bool = False,
) -> pd.DataFrame:
    existing = pd.DataFrame(columns=WEEKLY_COLUMNS) if force else read_bybit_storage(path)
    if not force and is_storage_fresh(path):
        return existing
    merged = existing
    for asset in BYBIT_ASSET_MAP:
        merged = update_bybit_asset(asset, merged, client=client)
    write_bybit_storage(merged, path)
    return read_bybit_storage(path)


def is_storage_fresh(path: Path = BYBIT_STORAGE_PATH) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < int(BYBIT_CONFIG["history_ttl_seconds"])


def merge_asset_history(existing: pd.DataFrame, asset_frame: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = asset_frame.copy()
    else:
        asset = str(asset_frame["asset"].dropna().iloc[0]) if not asset_frame.empty else ""
        without_asset = existing[~existing["asset"].eq(asset)].copy()
        old_asset = existing[existing["asset"].eq(asset)].copy()
        merged = pd.concat([without_asset, old_asset, asset_frame], ignore_index=True)
    for column in WEEKLY_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    merged = merged.drop_duplicates(subset=["exchange", "asset", "exchange_symbol", "timestamp"], keep="last")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged[WEEKLY_COLUMNS].sort_values(["asset", "date"])


def latest_states(frame: pd.DataFrame) -> dict[str, CryptoDerivativesState]:
    states: dict[str, CryptoDerivativesState] = {}
    for asset, cfg in BYBIT_ASSET_MAP.items():
        asset_frame = frame[frame["asset"].eq(asset)].copy() if not frame.empty else pd.DataFrame()
        if asset_frame.empty:
            states[asset] = CryptoDerivativesState(asset=asset, exchange=BYBIT_EXCHANGE, exchange_symbol=cfg["symbol"], data_status="ERROR")
            continue
        asset_frame["date"] = pd.to_datetime(asset_frame["date"], errors="coerce")
        row = asset_frame.sort_values("date").iloc[-1]
        states[asset] = CryptoDerivativesState(
            asset=asset,
            exchange=str(row.get("exchange") or BYBIT_EXCHANGE),
            exchange_symbol=str(row.get("exchange_symbol") or cfg["symbol"]),
            price=safe_float(row.get("price")),
            open_interest_usd=safe_float(row.get("open_interest_usd")),
            oi_change_1w_pct=safe_float(row.get("oi_change_1w_pct")),
            oi_change_4w_pct=safe_float(row.get("oi_change_4w_pct")),
            oi_change_13w_pct=safe_float(row.get("oi_change_13w_pct")),
            oi_change_4w_percentile=safe_float(row.get("oi_change_4w_percentile")),
            funding_current=safe_float(row.get("funding_rate")),
            funding_7d=safe_float(row.get("funding_7d")),
            funding_28d=safe_float(row.get("funding_28d")),
            funding_percentile=safe_float(row.get("funding_28d_percentile")),
            perp_premium_pct=safe_float(row.get("perp_premium_pct")),
            premium_28d_avg=safe_float(row.get("perp_premium_28d_avg")),
            premium_percentile=safe_float(row.get("perp_premium_percentile")),
            oi_price_regime=str(row.get("oi_price_regime") or "n/a"),
            history_start_date=str(row.get("history_start_date") or "n/a"),
            last_updated=str(row.get("last_updated") or "n/a"),
            data_status=str(row.get("data_status") or "ERROR"),
        )
    return states


def fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fmt_datetime(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetSpec:
    """Provider and normalization contract for a market data series."""

    canonical_asset_id: str
    display_name: str
    source_id: str
    provider_symbol: str
    instrument_type: str
    currency: str
    price_unit: str
    session_calendar: str
    source_timezone: str
    adjustment_mode: str
    base_timeframe: str
    provider_label: str
    provenance_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BAR_COLUMNS = [
    "bar_id",
    "timestamp",
    "period_start",
    "period_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timeframe",
    "is_closed",
    "source_id",
    "provider_symbol",
    "source_timeframe",
    "last_source_bar_time",
    "lineage",
]


class MarketDataError(RuntimeError):
    pass


def load_base_bars(
    spec: AssetSpec,
    *,
    force: bool = False,
    yahoo_loader: Callable[..., pd.DataFrame] | None = None,
    tradingview_loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if spec.source_id == "yahoo_finance":
        bars = load_yahoo_daily(spec, loader=yahoo_loader)
    elif spec.source_id == "tradingview_mcp":
        bars = load_tradingview_bars(
            spec,
            spec.base_timeframe,
            force=force,
            loader=tradingview_loader,
        )
    else:
        raise MarketDataError(f"Unsupported source_id={spec.source_id!r}")
    bars = validate_bars(bars, spec)
    bars = bars.loc[bars["is_closed"]].copy()
    if bars.empty:
        raise MarketDataError(f"{spec.canonical_asset_id} returned no closed {spec.base_timeframe} bars")
    return bars.reset_index(drop=True)


def load_display_bars(
    spec: AssetSpec,
    timeframe: str,
    base_bars: pd.DataFrame,
    *,
    force: bool = False,
    tradingview_loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if timeframe not in {"1D", "1W", "1M"}:
        raise MarketDataError(f"Unsupported chart timeframe={timeframe!r}")
    if spec.base_timeframe == "1D":
        if timeframe == "1D":
            return base_bars.copy()
        return aggregate_daily_bars(base_bars, timeframe, spec)

    if timeframe == spec.base_timeframe:
        return base_bars.copy()
    # TradingView is the only current weekly-base source. Direct provider bars
    # preserve real daily/monthly OHLC and never synthesize daily observations.
    return validate_bars(
        load_tradingview_bars(spec, timeframe, force=force, loader=tradingview_loader),
        spec,
    ).reset_index(drop=True)


def load_yahoo_daily(
    spec: AssetSpec,
    *,
    loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if loader is None:
        try:
            import yfinance as yf
        except Exception as exc:
            raise MarketDataError(f"Yahoo Finance adapter unavailable: {exc}") from exc
        loader = yf.download
    fetch = loader
    try:
        raw = fetch(
            spec.provider_symbol,
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise MarketDataError(f"Yahoo download failed for {spec.provider_symbol}: {exc}") from exc
    frame = _extract_yahoo_ohlcv(raw, spec.provider_symbol)
    if frame.empty:
        raise MarketDataError(f"Yahoo returned no OHLC for {spec.provider_symbol}")
    dates = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_localize(None)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = _normalized_frame(
        dates=dates,
        opens=frame["Open"],
        highs=frame["High"],
        lows=frame["Low"],
        closes=frame["Close"],
        volumes=frame.get("Volume"),
        timeframe="1D",
        spec=spec,
        now=now,
    )
    # Yahoo index volumes are frequently zero placeholders, not observations.
    if spec.instrument_type == "price_index":
        out["volume"] = np.nan
    return out


def load_tradingview_bars(
    spec: AssetSpec,
    timeframe: str,
    *,
    force: bool = False,
    loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if loader is None:
        try:
            from tradingview_mcp import get_ohlcv_data
        except Exception as exc:
            raise MarketDataError(f"TradingView adapter unavailable: {exc}") from exc
        loader = get_ohlcv_data
    interval = timeframe
    symbols = [spec.provider_symbol]
    # Keep a provider fallback for transient symbol outages. The canonical
    # GOLD source is OANDA:XAUUSD; TVC:GOLD is only a fallback and normalized
    # provenance remains the configured OANDA symbol.
    if spec.provider_symbol == "OANDA:XAUUSD":
        symbols.append("TVC:GOLD")
    elif spec.provider_symbol == "TVC:GOLD":
        symbols.append("GOLD")
    best: pd.DataFrame | None = None
    errors: list[str] = []
    minimum = 300 if timeframe == "1D" else 200
    for symbol in symbols:
        try:
            try:
                frame = loader(symbol, interval=interval, count=5000, force=force)
            except TypeError:
                frame = loader(symbol, interval=interval, count=5000)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            continue
        if frame is None or frame.empty:
            errors.append(f"{symbol}: empty response")
            continue
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            errors.append(f"{symbol}: missing {sorted(required.difference(frame.columns))}")
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
        normalized = _normalized_frame(
            dates=dates,
            opens=frame["open"],
            highs=frame["high"],
            lows=frame["low"],
            closes=frame["close"],
            volumes=frame.get("volume"),
            timeframe=timeframe,
            spec=spec,
            now=pd.Timestamp.now(tz="UTC").tz_localize(None),
        )
        if best is None or len(normalized) > len(best):
            best = normalized
        if len(normalized) >= minimum:
            return normalized
    if best is not None:
        return best
    detail = "; ".join(errors) or "no valid response"
    raise MarketDataError(f"TradingView download failed for {spec.provider_symbol}: {detail}")


def _normalized_frame(
    *,
    dates: pd.Series | pd.DatetimeIndex,
    opens: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: pd.Series | None,
    timeframe: str,
    spec: AssetSpec,
    now: pd.Timestamp,
) -> pd.DataFrame:
    date_values = pd.Series(pd.to_datetime(dates, errors="coerce")).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "timestamp": date_values,
            "open": pd.to_numeric(pd.Series(opens).reset_index(drop=True), errors="coerce"),
            "high": pd.to_numeric(pd.Series(highs).reset_index(drop=True), errors="coerce"),
            "low": pd.to_numeric(pd.Series(lows).reset_index(drop=True), errors="coerce"),
            "close": pd.to_numeric(pd.Series(closes).reset_index(drop=True), errors="coerce"),
            "volume": (
                pd.to_numeric(pd.Series(volumes).reset_index(drop=True), errors="coerce")
                if volumes is not None
                else np.nan
            ),
        }
    )
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    out["timeframe"] = timeframe
    out["period_start"] = out["timestamp"]
    out["period_end"] = out["timestamp"].map(lambda value: _period_end(pd.Timestamp(value), timeframe, spec))
    out["is_closed"] = out["period_end"].le(now)
    out["source_id"] = spec.source_id
    out["provider_symbol"] = spec.provider_symbol
    out["source_timeframe"] = timeframe
    out["last_source_bar_time"] = out["timestamp"]
    out["lineage"] = out["timestamp"].map(lambda value: [pd.Timestamp(value).isoformat()])
    out["bar_id"] = [
        _bar_id(spec.canonical_asset_id, timeframe, pd.Timestamp(value))
        for value in out["timestamp"]
    ]
    return out[BAR_COLUMNS]


def _period_end(timestamp: pd.Timestamp, timeframe: str, spec: AssetSpec) -> pd.Timestamp:
    value = pd.Timestamp(timestamp).tz_localize(None)
    if timeframe == "1D":
        return value.normalize() + pd.Timedelta(days=1)
    if timeframe == "1W":
        # Provider timestamps are treated as period starts. This conservative
        # rule prevents the active week from confirming a structural pivot.
        return value.normalize() + pd.Timedelta(days=7)
    if timeframe == "1M":
        return value.normalize() + pd.offsets.MonthBegin(1)
    raise MarketDataError(f"Unsupported timeframe={timeframe!r}")


def aggregate_daily_bars(daily: pd.DataFrame, timeframe: str, spec: AssetSpec) -> pd.DataFrame:
    if timeframe not in {"1W", "1M"}:
        raise MarketDataError("Daily bars can only be aggregated to 1W or 1M")
    if daily.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    values = daily.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values = values.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    rule = "W-SUN" if spec.session_calendar == "24X7_UTC" else "W-FRI"
    if timeframe == "1M":
        rule = "ME"

    records: list[dict] = []
    try:
        grouped = values.groupby(pd.Grouper(freq=rule))
    except ValueError:
        grouped = values.groupby(pd.Grouper(freq="M" if timeframe == "1M" else rule))
    for period_end, group in grouped:
        if group.empty:
            continue
        lineage = [str(item) for row in group["lineage"] for item in (row if isinstance(row, list) else [row])]
        volume = pd.to_numeric(group["volume"], errors="coerce").sum(min_count=1)
        period_start = pd.Timestamp(group.index.min())
        # Grouper labels W-FRI/W-SUN and month-end at the start of the final
        # calendar day.  The period is not closed until that day has ended.
        period_finish = pd.Timestamp(period_end) + pd.Timedelta(days=1)
        records.append(
            {
                "bar_id": _bar_id(spec.canonical_asset_id, timeframe, period_start),
                "timestamp": period_start,
                "period_start": period_start,
                "period_end": period_finish,
                "open": float(group["open"].iloc[0]),
                "high": float(pd.to_numeric(group["high"], errors="coerce").max()),
                "low": float(pd.to_numeric(group["low"], errors="coerce").min()),
                "close": float(group["close"].iloc[-1]),
                "volume": float(volume) if pd.notna(volume) else np.nan,
                "timeframe": timeframe,
                "is_closed": bool(group["is_closed"].all() and period_finish <= pd.Timestamp.now(tz="UTC").tz_localize(None)),
                "source_id": spec.source_id,
                "provider_symbol": spec.provider_symbol,
                "source_timeframe": "1D",
                "last_source_bar_time": pd.Timestamp(group.index.max()),
                "lineage": lineage,
            }
        )
    return pd.DataFrame(records, columns=BAR_COLUMNS)


def validate_bars(frame: pd.DataFrame, spec: AssetSpec) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise MarketDataError(f"{spec.canonical_asset_id}: empty OHLC history")
    missing = [column for column in BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise MarketDataError(f"{spec.canonical_asset_id}: missing normalized columns {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if out["timestamp"].duplicated().any():
        raise MarketDataError(f"{spec.canonical_asset_id}: duplicate bar timestamps")
    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp").reset_index(drop=True)
    numeric = out[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        ~np.isfinite(numeric).all(axis=1)
        | numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise MarketDataError(f"{spec.canonical_asset_id}: {int(invalid.sum())} invalid OHLC bars")
    return out.reset_index(drop=True)


def data_version(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    payload = frame[["timestamp", "open", "high", "low", "close", "is_closed"]].copy()
    payload["timestamp"] = pd.to_datetime(payload["timestamp"], errors="coerce").astype(str)
    digest = hashlib.sha256(payload.to_csv(index=False, float_format="%.10f").encode("utf-8")).hexdigest()
    return digest[:20]


def _extract_yahoo_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            frame = frame[ticker].copy()
        elif ticker in frame.columns.get_level_values(1):
            frame = frame.xs(ticker, axis=1, level=1).copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def _bar_id(asset_id: str, timeframe: str, timestamp: pd.Timestamp) -> str:
    raw = f"{asset_id}|{timeframe}|{pd.Timestamp(timestamp).isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

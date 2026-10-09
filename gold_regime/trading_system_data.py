from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from tradingview_mcp import get_ohlcv_data

from .trading_system import normalize_ohlcv, prepare_usd_ohlcv


TV_SYMBOLS = {
    "GOLD": "TVC:GOLD",
    "SILVER": "TVC:SILVER",
    "PLATINUM": "TVC:PLATINUM",
}
YAHOO_SYMBOLS = ("3GOL.L", "3SIL.L")


def load_trading_system_data() -> dict[str, Any]:
    """Load daily signals/benchmark and adjusted Yahoo execution prices.

    TradingView's own OHLCV cache is reused. yfinance `auto_adjust=True`
    provides consistently adjusted OHLC for total-return performance; price
    observations are aligned exactly and are never forward-filled.
    """
    output: dict[str, Any] = {
        "daily": {},
        "monthly": {},
        "gold_weekly": pd.Series(dtype=float, name="gold_price"),
        "leveraged": {},
        "currencies": {},
        "status": {},
        "currency_warning": None,
        "source_note": "",
    }
    cache_dir = Path("persistent") / "finance_cache" / "gold_regime" / "trading_system"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for asset, symbol in TV_SYMBOLS.items():
        output["daily"][asset] = _tradingview_bars(symbol, "1D", output["status"])
        output["monthly"][asset] = _tradingview_bars(symbol, "1M", output["status"])
    gold_weekly = _tradingview_bars("TVC:GOLD", "1W", output["status"])
    output["gold_weekly"] = gold_weekly["close"].dropna().rename("gold_price") if "close" in gold_weekly else pd.Series(dtype=float, name="gold_price")

    currencies: dict[str, str] = {}
    for ticker in YAHOO_SYMBOLS:
        try:
            frame, refresh_note = _incremental_yahoo_ohlc(ticker, cache_dir)
            if frame.empty:
                raise ValueError("Adjusted OHLC history is empty.")
            currency = _yahoo_currency(ticker)
            currencies[ticker] = currency
            output["leveraged"][ticker] = frame
            output["status"][ticker] = f"yfinance auto-adjusted OHLC · {currency} · {frame.index[-1]:%Y-%m-%d}{refresh_note}"
        except Exception as exc:
            output["leveraged"][ticker] = pd.DataFrame()
            output["status"][ticker] = f"Unavailable: {exc}"

    output["currencies"] = currencies
    distinct = sorted(set(currencies.values()))
    if len(distinct) > 1:
        output["currency_warning"] = f"3GOL.L and 3SIL.L report different currencies: {', '.join(distinct)}. Each series is converted to USD before alignment."

    fx_by_currency: dict[str, pd.DataFrame] = {}
    for currency in distinct:
        if currency.upper() in {"USD", "US DOLLAR", "US DOLLARS"}:
            continue
        fx_symbol = "GBPUSD=X" if currency in {"GBp", "GBX", "GBpence"} else f"{currency.upper()}USD=X"
        try:
            fx_by_currency[currency], _ = _incremental_yahoo_ohlc(fx_symbol, cache_dir)
        except Exception as exc:
            fx_by_currency[currency] = pd.DataFrame()
            output["status"][f"FX {currency}/USD"] = f"Unavailable: {exc}"

    for ticker, frame in list(output["leveraged"].items()):
        currency = currencies.get(ticker)
        if frame.empty or not currency:
            continue
        if currency.upper() in {"USD", "US DOLLAR", "US DOLLARS"}:
            output["leveraged"][ticker] = frame
            continue
        fx = fx_by_currency.get(currency, pd.DataFrame())
        if fx.empty:
            output["leveraged"][ticker] = pd.DataFrame()
            output["status"][ticker] = f"Unavailable: no {currency}/USD FX history for USD conversion."
            continue
        output["leveraged"][ticker] = prepare_usd_ohlcv(frame, currency, fx["close"], fx["open"])
        if output["leveraged"][ticker].empty:
            output["status"][ticker] = f"Unavailable: no exact price/FX date alignment for USD conversion ({currency})."
        else:
            output["status"][ticker] += " · converted to USD"

    output["source_note"] = (
        "Ratio and spot metals: synchronized TradingView daily/monthly observations. "
        "Leveraged instruments: yfinance auto-adjusted daily OHLC; currency checked and converted to USD."
    )
    return output


def _tradingview_bars(symbol: str, interval: str, status: dict[str, str]) -> pd.DataFrame:
    try:
        frame = get_ohlcv_data(symbol, interval=interval, count=5000)
        values = normalize_ohlcv(frame)
        if values.empty or values["close"].dropna().empty:
            raise ValueError("TradingView returned no valid bars.")
        if interval == "1D":
            today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
            values = values.loc[values.index < today_utc]
            if values.empty:
                raise ValueError("TradingView has not published a completed daily bar yet.")
        status[f"{symbol} {interval}"] = f"TradingView MCP · {values.index[-1]:%Y-%m-%d}"
        return values
    except Exception as exc:
        status[f"{symbol} {interval}"] = f"Unavailable: {exc}"
        return pd.DataFrame()


def _extract_yahoo_ohlc(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("empty yfinance response")
    values = frame
    if isinstance(values.columns, pd.MultiIndex):
        if symbol in values.columns.get_level_values(0):
            values = values[symbol]
        elif symbol in values.columns.get_level_values(1):
            values = values.xs(symbol, axis=1, level=1)
        elif len(set(values.columns.get_level_values(1))) == 1:
            values = values.xs(values.columns.get_level_values(1)[0], axis=1, level=1)
        else:
            raise KeyError(f"OHLC columns for {symbol} are missing.")
    renamed = {str(column).lower(): column for column in values.columns}
    output = pd.DataFrame(index=values.index)
    for field in ("open", "high", "low", "close"):
        if field not in renamed:
            raise KeyError(f"Adjusted {field.title()} is missing for {symbol}.")
        output[field] = pd.to_numeric(values[renamed[field]], errors="coerce")
    return output.dropna(subset=["open", "close"])


def _yahoo_currency(ticker: str) -> str:
    asset = yf.Ticker(ticker)
    try:
        currency = asset.fast_info.get("currency")
    except Exception:
        currency = None
    if not currency:
        try:
            currency = asset.get_info().get("currency")
        except Exception:
            currency = None
    if not currency:
        raise ValueError("Yahoo Finance did not report the instrument currency.")
    return str(currency).strip()


def _incremental_yahoo_ohlc(symbol: str, cache_dir: Path) -> tuple[pd.DataFrame, str]:
    """Read persisted adjusted OHLC and request only the recent overlap on refresh."""
    safe_symbol = "".join(character if character.isalnum() else "_" for character in symbol)
    cache_path = cache_dir / f"{safe_symbol}_adjusted_ohlc.csv"
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = normalize_ohlcv(pd.read_csv(cache_path))
        except Exception:
            cached = pd.DataFrame()
    try:
        request: dict[str, Any] = {
            "tickers": symbol,
            "interval": "1d",
            "auto_adjust": True,
            "actions": False,
            "progress": False,
            "threads": False,
        }
        if cached.empty:
            request["period"] = "max"
            refresh_note = " · full history initialized"
        else:
            # Overlap a few sessions to reconcile recent corrections while
            # retaining older bars locally instead of downloading max history.
            request["start"] = (cached.index[-1] - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            refresh_note = " · incremental cache"
        downloaded = yf.download(**request)
        updated = normalize_ohlcv(_extract_yahoo_ohlc(downloaded, symbol))
        combined = pd.concat([cached, updated]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if combined.empty:
            raise ValueError("yfinance returned no usable adjusted bars.")
        persisted = combined.copy()
        persisted.insert(0, "date", persisted.index)
        temporary = cache_path.with_suffix(".tmp.csv")
        persisted.to_csv(temporary, index=False)
        temporary.replace(cache_path)
        return combined, refresh_note
    except Exception as exc:
        if not cached.empty:
            return cached, f" · using local cache; refresh failed: {exc}"
        raise ValueError(f"yfinance history unavailable: {exc}") from exc

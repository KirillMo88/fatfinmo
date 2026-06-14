from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD


REPORT_COLUMNS = [
    "Perf_1D_%",
    "Perf_1W_%",
    "Perf_1M_%",
    "Perf_3M_%",
    "Price_vs_52W_High_%",
    "RSI_14",
]
ANALYSIS_TICKERS = ("QQQ", "GLD", "BTC-USD")


def extract_ohlcv_frame(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if px is None or px.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(px.columns, pd.MultiIndex):
        if ticker in px.columns.get_level_values(-1):
            px = px.xs(ticker, axis=1, level=-1)
        elif ticker in px.columns.get_level_values(0):
            px = px.xs(ticker, axis=1, level=0)
        else:
            px = px.droplevel(-1, axis=1)
    out = px.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(out.columns):
        return pd.DataFrame(columns=required)
    out = out[required].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna(subset=["Close"]).sort_index()


def drop_incomplete_daily_bar(
    ohlcv: pd.DataFrame, now_utc: datetime | None = None
) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv
    now_utc = now_utc or datetime.now(timezone.utc)
    utc_date = pd.Timestamp(now_utc).tz_convert("UTC").tz_localize(None).normalize()
    out = ohlcv.copy()
    if pd.Timestamp(out.index[-1]).normalize() >= utc_date:
        out = out.iloc[:-1]
    return out


def download_completed_ohlcv(ticker: str, period: str = "10y") -> pd.DataFrame:
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            frame = drop_incomplete_daily_bar(extract_ohlcv_frame(raw, ticker))
            if not frame.empty:
                return frame
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def value_on_or_before(series: pd.Series, at: pd.Timestamp) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index().loc[:at]
    return float(values.iloc[-1]) if not values.empty else math.nan


def calendar_performance(close: pd.Series, days: int) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return math.nan
    end = pd.Timestamp(close.index[-1])
    start_value = value_on_or_before(close, end - pd.Timedelta(days=days))
    end_value = float(close.iloc[-1])
    if not np.isfinite(start_value) or start_value == 0:
        return math.nan
    return (end_value / start_value - 1.0) * 100.0


def weekly_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv
    weekly = (
        ohlcv.resample("W-FRI")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Close"])
    )
    if not weekly.empty and weekly.index[-1] > ohlcv.index[-1]:
        weekly = weekly.iloc[:-1]
    return weekly


def crossover_dates(
    close: pd.Series, lookback_bars: int = 14
) -> tuple[list[str], list[str]]:
    close = pd.to_numeric(close, errors="coerce").dropna()
    fast = close.rolling(50, min_periods=50).mean()
    slow = close.rolling(200, min_periods=200).mean()
    pair = pd.concat([fast.rename("fast"), slow.rename("slow")], axis=1).dropna()
    if len(pair) < 2:
        return [], []
    above = pair["fast"] > pair["slow"]
    previous = above.shift(1)
    golden = above & previous.eq(False)
    death = above.eq(False) & previous.eq(True)
    window = max(1, lookback_bars)
    recent_index = pair.index[-window:]
    return (
        [
            pd.Timestamp(v).date().isoformat()
            for v in recent_index[golden.reindex(recent_index, fill_value=False)]
        ],
        [
            pd.Timestamp(v).date().isoformat()
            for v in recent_index[death.reindex(recent_index, fill_value=False)]
        ],
    )


def ticker_metrics(ticker: str, ohlcv: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    if len(close) < 200:
        raise ValueError(f"Not enough data for {ticker}")
    high_52w = float(close.tail(277).max())
    current = float(close.iloc[-1])
    daily_golden, daily_death = crossover_dates(close, 14)
    weekly = weekly_ohlcv(ohlcv)
    weekly_golden, weekly_death = crossover_dates(weekly["Close"], 14)
    rsi = RSIIndicator(close, window=14).rsi()
    values = {
        "ticker": ticker,
        "data_date": pd.Timestamp(close.index[-1]).date().isoformat(),
        "Perf_1D_%": calendar_performance(close, 1),
        "Perf_1W_%": calendar_performance(close, 7),
        "Perf_1M_%": calendar_performance(close, 30),
        "Perf_3M_%": calendar_performance(close, 90),
        "Price_vs_52W_High_%": (current / high_52w - 1.0) * 100.0,
        "RSI_14": float(rsi.dropna().iloc[-1]),
        "crosses": {
            "daily_golden": daily_golden,
            "daily_death": daily_death,
            "weekly_golden": weekly_golden,
            "weekly_death": weekly_death,
        },
    }
    for key in REPORT_COLUMNS:
        if not np.isfinite(values[key]):
            values[key] = None
    return values


def load_full_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = payload["Full list"]
    return list(
        dict.fromkeys(
            ticker
            for group in full.values()
            for tickers in group.values()
            for ticker in tickers
        )
    )


def build_daily_report(universe_path: Path, workers: int = 6) -> dict[str, Any]:
    tickers = load_full_list(universe_path)
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_completed_ohlcv, ticker): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows.append(ticker_metrics(ticker, future.result()))
            except Exception as exc:
                errors[ticker] = str(exc)

    rankings = {}
    for key, label in (
        ("Perf_1D_%", "day"),
        ("Perf_1W_%", "week"),
        ("Perf_1M_%", "month"),
    ):
        rankings[label] = sorted(
            (row for row in rows if row[key] is not None),
            key=lambda row: row[key],
            reverse=True,
        )[:5]

    cross_map = {
        "daily_golden": [],
        "daily_death": [],
        "weekly_golden": [],
        "weekly_death": [],
    }
    for row in sorted(rows, key=lambda item: item["ticker"]):
        for kind in cross_map:
            cross_map[kind].extend(
                {"ticker": row["ticker"], "date": date}
                for date in row["crosses"][kind]
            )
    for events in cross_map.values():
        events.sort(key=lambda event: (event["date"], event["ticker"]), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": "Full list",
        "asset_count": len(rows),
        "rankings": rankings,
        "crosses": cross_map,
        "errors": errors,
    }


def _atr(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = ohlcv["Close"].shift(1)
    true_range = pd.concat(
        [
            ohlcv["High"] - ohlcv["Low"],
            (ohlcv["High"] - previous_close).abs(),
            (ohlcv["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _zigzag_state(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv["Close"].astype(float)
    threshold = ((_atr(ohlcv) / close) * 3.0).clip(lower=0.025, upper=0.15)
    state = pd.Series(np.nan, index=close.index, dtype=float)
    direction = 0
    pivot_price = float(close.iloc[0])
    leg = 0
    for i in range(1, len(close)):
        price = float(close.iloc[i])
        limit = float(threshold.iloc[i]) if np.isfinite(threshold.iloc[i]) else 0.05
        change = price / pivot_price - 1.0
        if direction >= 0 and change <= -limit:
            direction = -1
            pivot_price = price
            leg += 1
        elif direction <= 0 and change >= limit:
            direction = 1
            pivot_price = price
            leg += 1
        elif direction > 0 and price > pivot_price:
            pivot_price = price
        elif direction < 0 and price < pivot_price:
            pivot_price = price
        state.iloc[i] = direction * (1.0 + (leg % 5) / 10.0)
    return state.ffill().fillna(0.0)


def _feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv["Close"].astype(float)
    returns = close.pct_change()
    macd_hist = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
    features = pd.DataFrame(index=close.index)
    features["elliott"] = _zigzag_state(ohlcv)
    features["rsi"] = RSIIndicator(close, window=14).rsi()
    features["rsi_delta"] = features["rsi"].diff(5)
    features["macd_hist"] = macd_hist / close
    features["macd_delta"] = features["macd_hist"].diff(5)
    features["roc12"] = ROCIndicator(close, window=12).roc()
    for window in (21, 63, 126):
        features[f"mom{window}"] = close.pct_change(window) * 100.0
    features["sma50"] = (close / close.rolling(50).mean() - 1.0) * 100.0
    features["sma200"] = (close / close.rolling(200).mean() - 1.0) * 100.0
    features["vol63"] = returns.rolling(63).std() * np.sqrt(63) * 100.0
    features["future63"] = close.shift(-63) / close - 1.0
    return features.replace([np.inf, -np.inf], np.nan)


def _scenario_label(future_return: float, expected_volatility: float) -> str:
    threshold = 0.5 * expected_volatility / 100.0
    if future_return > threshold:
        return "bullish"
    if future_return < -threshold:
        return "bearish"
    return "neutral"


def analyze_ticker(ticker: str, ohlcv: pd.DataFrame) -> dict[str, Any]:
    features = _feature_frame(ohlcv)
    columns = [
        "elliott",
        "rsi",
        "rsi_delta",
        "macd_hist",
        "macd_delta",
        "roc12",
        "mom21",
        "mom63",
        "mom126",
        "sma50",
        "sma200",
        "vol63",
    ]
    current = features[columns].dropna().iloc[-1]
    history = features[columns + ["future63"]].dropna()
    history = history.loc[history.index <= features.index[-64]]
    if len(history) < 75:
        raise ValueError(f"Not enough calibrated history for {ticker}")
    mean = history[columns].mean()
    std = history[columns].std().replace(0, 1.0)
    distances = (((history[columns] - current) / std) ** 2).mean(axis=1) ** 0.5
    nearest = distances.nsmallest(75)
    weights = 1.0 / (nearest + 0.05)
    totals = {"bullish": 1.0, "bearish": 1.0, "neutral": 1.0}
    for index, distance_weight in weights.items():
        row = history.loc[index]
        totals[_scenario_label(row["future63"], row["vol63"])] += float(
            distance_weight
        )
    total_weight = sum(totals.values())
    probabilities = {
        key: round(value / total_weight * 100.0, 1)
        for key, value in totals.items()
    }
    rounding_delta = round(100.0 - sum(probabilities.values()), 1)
    leading = max(probabilities, key=probabilities.get)
    probabilities[leading] = round(probabilities[leading] + rounding_delta, 1)

    elliott = "impulse_up" if current["elliott"] > 0 else "corrective_down"
    reasons = [
        f"Elliott: {elliott}",
        f"RSI(14): {current['rsi']:.1f} ({'rising' if current['rsi_delta'] > 0 else 'falling'})",
        f"MACD histogram: {'positive' if current['macd_hist'] > 0 else 'negative'}, {'improving' if current['macd_delta'] > 0 else 'weakening'}",
        f"ROC(12): {current['roc12']:.1f}%",
        f"Momentum 21/63/126: {current['mom21']:.1f}% / {current['mom63']:.1f}% / {current['mom126']:.1f}%",
        f"Price vs SMA50/SMA200: {current['sma50']:.1f}% / {current['sma200']:.1f}%",
    ]
    return {
        "ticker": ticker,
        "data_date": pd.Timestamp(ohlcv.index[-1]).date().isoformat(),
        "horizon_bars": 63,
        "analog_count": 75,
        "elliott_state": elliott,
        "indicators": {
            key: round(float(current[key]), 4)
            for key in columns
            if key != "elliott"
        },
        "probabilities": probabilities,
        "reasons": reasons,
    }


def build_weekly_report() -> dict[str, Any]:
    analyses = []
    errors = {}
    for ticker in ANALYSIS_TICKERS:
        try:
            analyses.append(analyze_ticker(ticker, download_completed_ohlcv(ticker)))
        except Exception as exc:
            errors[ticker] = str(exc)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_bars": 63,
        "analyses": analyses,
        "errors": errors,
        "disclaimer": "Вероятности основаны на исторических аналогах и не являются инвестиционной рекомендацией.",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

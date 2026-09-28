from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import ASSET_SPECS, ENGINE_VERSION, AssetSpec
from .data import ElliottDataError, data_version, load_base_bars, load_display_bars
from .engine import ElliottWaveEngine
from .storage import (
    read_manifest,
    read_quotes,
    read_snapshot,
    write_manifest,
    write_quotes,
    write_snapshot,
)


def refresh_asset(spec: AssetSpec, *, force: bool = False) -> dict[str, Any]:
    previous = read_snapshot(spec.canonical_asset_id)
    base_bars = load_base_bars(spec, force=force)
    snapshot = ElliottWaveEngine().analyze(base_bars, spec)
    chart_bars: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for timeframe in ("1D", "1W", "1M"):
        try:
            chart_bars[timeframe] = load_display_bars(spec, timeframe, base_bars, force=force)
        except Exception as exc:
            warnings.append(f"{timeframe}: {type(exc).__name__}: {exc}")
    snapshot["display_timeframes"] = sorted(chart_bars)
    snapshot["display_history"] = {
        timeframe: {
            "start": pd.Timestamp(frame.iloc[0]["timestamp"]).isoformat(),
            "end": pd.Timestamp(frame.iloc[-1]["timestamp"]).isoformat(),
            "bar_count": int(len(frame)),
            "source_timeframe": timeframe,
        }
        for timeframe, frame in sorted(chart_bars.items())
        if frame is not None and not frame.empty
    }
    daily = chart_bars.get("1D")
    if spec.base_timeframe == "1W" and daily is not None and not daily.empty:
        base_start = pd.Timestamp(base_bars.iloc[0]["timestamp"])
        daily_start = pd.Timestamp(daily.iloc[0]["timestamp"])
        if daily_start > base_start:
            warnings.append(
                f"1D display starts {daily_start.date()} while the 1W analytical history starts {base_start.date()}; "
                "TradingView MCP exposes at most 5000 OHLCV bars per request."
            )
            snapshot.setdefault("quality_flags", []).append("DISPLAY_HISTORY_TRUNCATED_1D")
    snapshot["display_warnings"] = warnings
    snapshot["analysis_snapshot_id"] = snapshot["snapshot_id"]
    snapshot["display_data_versions"] = {
        timeframe: data_version(frame)
        for timeframe, frame in sorted(chart_bars.items())
        if frame is not None and not frame.empty
    }
    snapshot["snapshot_id"] = _published_snapshot_id(snapshot)
    if previous and previous.get("snapshot_id") != snapshot.get("snapshot_id"):
        snapshot["previous_snapshot_id"] = previous.get("snapshot_id")
        if previous.get("data_version") != snapshot.get("data_version") and previous.get("as_of") == snapshot.get("as_of"):
            snapshot.setdefault("quality_flags", []).append("DATA_REVISION")
            snapshot.setdefault("events", []).append(
                _transition_event("DATA_REVISION", spec.canonical_asset_id, previous, snapshot)
            )
        if previous.get("main_scenario_id") != snapshot.get("main_scenario_id"):
            snapshot.setdefault("events", []).append(
                _transition_event("SCENARIO_SUPERSEDED", spec.canonical_asset_id, previous, snapshot)
            )
        snapshot["events"] = sorted(snapshot.get("events", []), key=lambda row: (str(row.get("known_at")), str(row.get("event_id"))))
    write_snapshot(snapshot, base_bars, chart_bars)
    return snapshot


def refresh_all_assets(*, force: bool = False) -> dict[str, Any]:
    previous = read_manifest() or {"assets": []}
    previous_by_asset = {row.get("canonical_asset_id"): row for row in previous.get("assets", [])}
    entries: list[dict[str, Any]] = []
    run_at = pd.Timestamp.now(tz="UTC").isoformat()
    for asset_id, spec in ASSET_SPECS.items():
        try:
            snapshot = refresh_asset(spec, force=force)
            entries.append(
                {
                    "canonical_asset_id": asset_id,
                    "snapshot_id": snapshot["snapshot_id"],
                    "status": "CURRENT",
                    "stale_reason": None,
                    "as_of": snapshot["as_of"],
                    "created_at": snapshot["created_at"],
                }
            )
        except Exception as exc:
            old = previous_by_asset.get(asset_id)
            if old and old.get("snapshot_id") and read_snapshot(asset_id, old["snapshot_id"]):
                entries.append(
                    {
                        **old,
                        "status": "UPDATE_FAILED_USING_PREVIOUS",
                        "stale_reason": f"{type(exc).__name__}: {exc}",
                        "failed_at": run_at,
                    }
                )
            else:
                entries.append(
                    {
                        "canonical_asset_id": asset_id,
                        "snapshot_id": None,
                        "status": "ERROR",
                        "stale_reason": f"{type(exc).__name__}: {exc}",
                        "failed_at": run_at,
                    }
                )
    manifest = {
        "manifest_id": _manifest_id(entries, run_at),
        "published_at": run_at,
        "engine_version": ENGINE_VERSION,
        "assets": entries,
    }
    write_manifest(manifest)
    return manifest


def refresh_quote_overlay() -> dict[str, Any]:
    assets: dict[str, Any] = {}
    previous_assets = read_quotes().get("assets", {})
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for asset_id, spec in ASSET_SPECS.items():
        try:
            if spec.source_id == "yahoo_finance":
                import yfinance as yf

                frame = yf.download(
                    spec.provider_symbol,
                    period="5d",
                    interval="5m",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                close = _last_yahoo_close(frame, spec.provider_symbol)
            else:
                from tradingview_mcp import get_ohlcv_data

                frame = get_ohlcv_data(spec.provider_symbol, interval="1D", count=2, force=True)
                close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[-1])
            assets[asset_id] = {"price": close, "observed_at": now, "status": "CURRENT"}
        except Exception as exc:
            previous = previous_assets.get(asset_id, {})
            assets[asset_id] = {
                "price": previous.get("price"),
                "observed_at": previous.get("observed_at"),
                "failed_at": now,
                "status": "UPDATE_FAILED_USING_PREVIOUS" if previous.get("price") is not None else "ERROR",
                "reason": str(exc),
            }
    payload = {"updated_at": now, "assets": assets}
    write_quotes(payload)
    return payload


def _last_yahoo_close(frame: pd.DataFrame, ticker: str) -> float:
    if frame is None or frame.empty:
        raise ElliottDataError(f"No quote for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            close = frame[ticker]["Close"]
        else:
            close = frame.xs("Close", axis=1, level=0)[ticker]
    else:
        close = frame["Close"]
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        raise ElliottDataError(f"No finite quote for {ticker}")
    return float(values.iloc[-1])


def _manifest_id(entries: list[dict[str, Any]], run_at: str) -> str:
    raw = json.dumps({"entries": entries, "run_at": run_at}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _published_snapshot_id(snapshot: dict[str, Any]) -> str:
    identity = {
        "analysis_snapshot_id": snapshot.get("analysis_snapshot_id"),
        "display_data_versions": snapshot.get("display_data_versions", {}),
        "display_timeframes": snapshot.get("display_timeframes", []),
    }
    raw = json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _transition_event(
    event_type: str,
    asset_id: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "event_type": event_type,
        "canonical_asset_id": asset_id,
        "previous_snapshot_id": previous.get("snapshot_id"),
        "snapshot_id": current.get("snapshot_id"),
        "previous_scenario_id": previous.get("main_scenario_id"),
        "scenario_id": current.get("main_scenario_id"),
        "previous_data_version": previous.get("data_version"),
        "data_version": current.get("data_version"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "event_id": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18],
        **payload,
        "observed_at": current.get("as_of"),
        "known_at": current.get("created_at"),
    }

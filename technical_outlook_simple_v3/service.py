from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from market_data import AssetSpec, load_base_bars, load_yahoo_daily, validate_bars

from .config import CONFIG_VERSION, CORE_ASSETS, CORE_ASSET_KEYS, MODEL_VERSION, SR_ENGINE_VERSION, yahoo_asset_spec
from .engine import TechnicalOutlookSimpleV3Engine
from .llm import call_llm_interpretation, validate_llm_output
from .storage import read_chart_bars, read_latest_snapshot, read_manifest, read_settings, write_manifest, write_snapshot


TickerLoader = Callable[[AssetSpec], pd.DataFrame]


def analyze_requested_asset(
    ticker: str,
    *,
    run_at: datetime | None = None,
    loader: TickerLoader | None = None,
    allow_scheduled_llm: bool = False,
) -> dict[str, Any]:
    symbol = validate_ticker(ticker)
    # GLD was the pre-v3 ETF identifier. Keep accepting it in forms/URLs, but
    # route SIMPLE v3 analysis to the canonical TradingView GOLD asset.
    if symbol == "GLD":
        symbol = "GOLD"
    spec = CORE_ASSETS.get(symbol) or yahoo_asset_spec(symbol)
    previous = read_latest_snapshot(symbol)
    bars = (loader or _load)(spec)
    snapshot, charts = TechnicalOutlookSimpleV3Engine().analyze(bars, spec, previous=previous, created_at=run_at)
    snapshot = apply_llm_schedule(
        snapshot,
        previous,
        read_settings(),
        run_at or datetime.now(timezone.utc),
        allow_scheduled_llm=allow_scheduled_llm,
    )
    write_snapshot(snapshot, charts)
    return snapshot


def refresh_all_core_assets(
    *, run_at: datetime | None = None, loader: TickerLoader | None = None,
) -> dict[str, Any]:
    when = run_at or datetime.now(timezone.utc)
    previous_entries = {item.get("ticker"): item for item in read_manifest().get("assets", [])}
    entries: list[dict[str, Any]] = []
    for ticker in CORE_ASSET_KEYS:
        try:
            snapshot = analyze_requested_asset(ticker, run_at=when, loader=loader, allow_scheduled_llm=True)
            entries.append({
                "ticker": ticker, "snapshot_id": snapshot["snapshot_id"], "status": "CURRENT",
                "as_of": snapshot["as_of_timestamp"], "quant_updated_at": snapshot["quant_updated_at"],
                "llm_updated_at": snapshot.get("llm_updated_at"),
            })
        except Exception as exc:
            previous = previous_entries.get(ticker) or {}
            latest = read_latest_snapshot(ticker)
            entries.append({
                **previous,
                "ticker": ticker,
                "snapshot_id": latest.get("snapshot_id") if latest else previous.get("snapshot_id"),
                "status": "UPDATE_FAILED_USING_PREVIOUS" if latest else "ERROR",
                "stale_reason": f"{type(exc).__name__}: {exc}",
                "failed_at": pd.Timestamp(when).isoformat(),
            })
    manifest = {
        "model_version": MODEL_VERSION,
        "config_version": CONFIG_VERSION,
        "sr_engine_version": SR_ENGINE_VERSION,
        "updated_at": pd.Timestamp(when).isoformat(),
        "assets": entries,
    }
    write_manifest(manifest)
    return manifest


def apply_llm_schedule(
    snapshot: dict[str, Any],
    previous: dict[str, Any] | None,
    settings: dict[str, Any],
    run_at: datetime,
    *,
    allow_scheduled_llm: bool,
    llm_caller: Callable[[dict[str, Any]], tuple[dict[str, Any], str]] = call_llm_interpretation,
) -> dict[str, Any]:
    enabled = bool(settings.get("use_llm_interpretation", False))
    snapshot["llm_enabled"] = enabled
    if not enabled:
        snapshot["llm_status"] = "DISABLED"
        return snapshot
    _copy_previous_llm(snapshot, previous)
    if not allow_scheduled_llm or pd.Timestamp(run_at).weekday() != 4:
        snapshot["llm_status"] = "USING_LATEST_WEEKLY" if snapshot.get("llm_interpretation") else "AWAITING_FRIDAY_REFRESH"
        return snapshot
    previous_update = pd.to_datetime((previous or {}).get("llm_updated_at"), errors="coerce", utc=True)
    if pd.notna(previous_update) and previous_update.date() == pd.Timestamp(run_at).date():
        snapshot["llm_status"] = "USING_LATEST_WEEKLY"
        return snapshot
    try:
        interpretation, model = llm_caller(snapshot)
        _set_llm_output(snapshot, interpretation, model, run_at, "CURRENT")
    except Exception as exc:
        snapshot["llm_status"] = "UPDATE_FAILED_USING_PREVIOUS" if snapshot.get("llm_interpretation") else "FAILED"
        snapshot["llm_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def run_llm_now(
    ticker: str,
    *,
    run_at: datetime | None = None,
    llm_caller: Callable[[dict[str, Any]], tuple[dict[str, Any], str]] = call_llm_interpretation,
) -> dict[str, Any]:
    symbol = validate_ticker(ticker)
    if symbol == "GLD":
        symbol = "GOLD"
    previous = read_latest_snapshot(symbol)
    if not previous:
        raise ValueError(f"{symbol}: no SIMPLE v3 Quant snapshot is available")
    if previous.get("model_version") != MODEL_VERSION:
        raise ValueError(f"{symbol}: SIMPLE v3 Quant snapshot is stale")
    when = run_at or datetime.now(timezone.utc)
    interpretation, model = llm_caller(previous)
    snapshot = deepcopy(previous)
    snapshot["snapshot_id"] = _llm_snapshot_id(previous, when)
    _set_llm_output(snapshot, interpretation, model, when, "MANUAL_CURRENT")
    charts = {timeframe: read_chart_bars(symbol, str(previous["snapshot_id"]), timeframe) for timeframe in ("1W", "1D")}
    write_snapshot(snapshot, charts)
    _update_manifest(snapshot, when)
    return snapshot


def validate_ticker(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not symbol or len(symbol) > 24 or not re.fullmatch(r"[A-Z0-9^.=\-]+", symbol):
        raise ValueError("Unsupported ticker format")
    return symbol


def _load(spec: AssetSpec) -> pd.DataFrame:
    # The canonical Technical Outlook refresh runs immediately before SIMPLE
    # v3 in the shared nightly pipeline, so reuse its freshly persisted market
    # data instead of making a second provider request for every core asset.
    if spec.source_id == "yahoo_finance":
        # Yahoo's live daily candle can temporarily violate final OHLC
        # invariants while it is still forming. SIMPLE v3 is completed-bar
        # only, so exclude it before validation; malformed closed history must
        # still fail loudly.
        return _validate_completed_bars(load_yahoo_daily(spec), spec)
    return load_base_bars(spec, force=False)


def _validate_completed_bars(frame: pd.DataFrame, spec: AssetSpec) -> pd.DataFrame:
    if frame is None or frame.empty or "is_closed" not in frame.columns:
        return validate_bars(frame, spec)
    closed = frame.loc[frame["is_closed"].fillna(False)].copy()
    return validate_bars(closed, spec).reset_index(drop=True)


def _copy_previous_llm(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> None:
    if not previous or not previous.get("llm_interpretation"):
        return
    try:
        interpretation = validate_llm_output(previous["llm_interpretation"], snapshot)
    except (TypeError, ValueError):
        return
    snapshot["llm_interpretation"] = interpretation
    for field in ("llm_model", "llm_updated_at"):
        if previous.get(field) is not None:
            snapshot[field] = previous[field]


def _set_llm_output(snapshot: dict[str, Any], interpretation: dict[str, Any], model: str, run_at: datetime, status: str) -> None:
    snapshot["llm_enabled"] = True
    snapshot["llm_interpretation"] = validate_llm_output(interpretation, snapshot)
    snapshot["llm_model"] = model
    snapshot["llm_updated_at"] = pd.Timestamp(run_at).isoformat()
    snapshot["llm_status"] = status


def _update_manifest(snapshot: dict[str, Any], when: datetime) -> None:
    entries = list(read_manifest().get("assets") or [])
    item = {
        "ticker": snapshot.get("ticker"), "snapshot_id": snapshot.get("snapshot_id"), "status": "CURRENT",
        "as_of": snapshot.get("as_of_timestamp"), "quant_updated_at": snapshot.get("quant_updated_at"),
        "llm_updated_at": snapshot.get("llm_updated_at"),
    }
    for index, existing in enumerate(entries):
        if existing.get("ticker") == snapshot.get("ticker"):
            entries[index] = item
            break
    else:
        entries.append(item)
    write_manifest({
        "model_version": MODEL_VERSION,
        "config_version": CONFIG_VERSION,
        "sr_engine_version": SR_ENGINE_VERSION,
        "updated_at": pd.Timestamp(when).isoformat(),
        "assets": entries,
    })


def _llm_snapshot_id(snapshot: dict[str, Any], when: datetime) -> str:
    import hashlib
    value = f"{snapshot.get('snapshot_id')}|{pd.Timestamp(when).isoformat()}|LLM"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

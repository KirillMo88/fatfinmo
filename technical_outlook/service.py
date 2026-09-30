from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from market_data import AssetSpec, load_base_bars

from .analytics import snapshot_id as build_snapshot_id
from .config import CORE_ASSETS, MODEL_VERSION, yahoo_asset_spec
from .engine import TechnicalOutlookEngine
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
    spec = CORE_ASSETS.get(symbol) or yahoo_asset_spec(symbol)
    previous = read_latest_snapshot(symbol)
    bars = (loader or _load)(spec)
    snapshot, charts = TechnicalOutlookEngine().analyze(bars, spec, previous=previous, created_at=run_at)
    settings = read_settings()
    snapshot = apply_llm_schedule(
        snapshot,
        previous,
        settings,
        run_at or datetime.now(timezone.utc),
        allow_scheduled_llm=allow_scheduled_llm,
    )
    write_snapshot(snapshot, charts)
    return snapshot


def refresh_all_core_assets(
    *,
    run_at: datetime | None = None,
    loader: TickerLoader | None = None,
) -> dict[str, Any]:
    when = run_at or datetime.now(timezone.utc)
    previous_manifest = read_manifest()
    previous_entries = {item.get("ticker"): item for item in previous_manifest.get("assets", [])}
    entries: list[dict[str, Any]] = []
    for ticker in CORE_ASSETS:
        try:
            snapshot = analyze_requested_asset(ticker, run_at=when, loader=loader, allow_scheduled_llm=True)
            entries.append({
                "ticker": ticker,
                "snapshot_id": snapshot["snapshot_id"],
                "status": "CURRENT",
                "as_of": snapshot["as_of_timestamp"],
                "quant_updated_at": snapshot["quant_updated_at"],
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
    manifest = {"model_version": MODEL_VERSION, "updated_at": pd.Timestamp(when).isoformat(), "assets": entries}
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
    if not allow_scheduled_llm or not is_friday_night_run(run_at):
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
    """Run the LLM on the latest persisted Quant snapshot, bypassing the Friday schedule."""
    symbol = validate_ticker(ticker)
    previous = read_latest_snapshot(symbol)
    if not previous:
        raise ValueError(f"{symbol}: no Quant snapshot is available")
    if previous.get("model_version") != MODEL_VERSION:
        raise ValueError(f"{symbol}: Quant snapshot is stale; wait for the current weekly/daily refresh")
    when = run_at or datetime.now(timezone.utc)
    interpretation, model = llm_caller(previous)
    snapshot = deepcopy(previous)
    created_iso = pd.Timestamp(when).isoformat()
    snapshot["snapshot_id"] = build_snapshot_id(
        symbol,
        str(snapshot.get("as_of_timestamp") or ""),
        created_iso,
        str(snapshot.get("data_version") or ""),
    )
    _set_llm_output(snapshot, interpretation, model, when, "MANUAL_CURRENT")
    charts = {
        timeframe: read_chart_bars(symbol, str(previous["snapshot_id"]), timeframe)
        for timeframe in ("1W", "1D")
    }
    write_snapshot(snapshot, charts)
    _update_manifest_for_snapshot(snapshot, when)
    return snapshot


def is_friday_night_run(run_at: datetime) -> bool:
    return pd.Timestamp(run_at).weekday() == 4


def validate_ticker(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not symbol or len(symbol) > 24 or not re.fullmatch(r"[A-Z0-9^.=\-]+", symbol):
        raise ValueError("Unsupported ticker format")
    return symbol


def _load(spec: AssetSpec) -> pd.DataFrame:
    return load_base_bars(spec, force=True)


def _copy_previous_llm(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> None:
    if not previous or not previous.get("llm_interpretation"):
        return
    try:
        interpretation = validate_llm_output(previous["llm_interpretation"], snapshot)
    except (TypeError, ValueError):
        return
    snapshot["llm_interpretation"] = interpretation
    for field in ("llm_model", "llm_updated_at", "llm_difference"):
        if previous.get(field) is not None:
            snapshot[field] = previous[field]


def _set_llm_output(
    snapshot: dict[str, Any],
    interpretation: dict[str, Any],
    model: str,
    run_at: datetime,
    status: str,
) -> None:
    validated = validate_llm_output(interpretation, snapshot)
    snapshot["llm_enabled"] = True
    snapshot["llm_interpretation"] = validated
    snapshot["llm_difference"] = validated.get("interpretation_difference") or None
    snapshot["llm_model"] = model
    snapshot["llm_updated_at"] = pd.Timestamp(run_at).isoformat()
    snapshot["llm_status"] = status


def _update_manifest_for_snapshot(snapshot: dict[str, Any], run_at: datetime) -> None:
    manifest = read_manifest()
    entries = list(manifest.get("assets") or [])
    updated = False
    for index, item in enumerate(entries):
        if item.get("ticker") == snapshot.get("ticker"):
            entries[index] = {
                **item,
                "snapshot_id": snapshot.get("snapshot_id"),
                "status": "CURRENT",
                "as_of": snapshot.get("as_of_timestamp"),
                "quant_updated_at": snapshot.get("quant_updated_at"),
                "llm_updated_at": snapshot.get("llm_updated_at"),
            }
            updated = True
            break
    if not updated:
        entries.append({
            "ticker": snapshot.get("ticker"),
            "snapshot_id": snapshot.get("snapshot_id"),
            "status": "CURRENT",
            "as_of": snapshot.get("as_of_timestamp"),
            "quant_updated_at": snapshot.get("quant_updated_at"),
            "llm_updated_at": snapshot.get("llm_updated_at"),
        })
    write_manifest({"model_version": MODEL_VERSION, "updated_at": pd.Timestamp(run_at).isoformat(), "assets": entries})

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import ANALYSIS_WINDOWS, ASSET_SPECS, DEFAULT_ANALYSIS_WINDOW, ENGINE_VERSION, AssetSpec
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


def refresh_asset(
    spec: AssetSpec,
    *,
    force: bool = False,
    analysis_window: str = DEFAULT_ANALYSIS_WINDOW,
) -> dict[str, Any]:
    return refresh_asset_windows(spec, force=force, analysis_windows=(analysis_window,))[analysis_window]


def refresh_asset_windows(
    spec: AssetSpec,
    *,
    force: bool = False,
    analysis_windows: tuple[str, ...] | list[str] = tuple(ANALYSIS_WINDOWS),
) -> dict[str, dict[str, Any]]:
    base_bars = load_base_bars(spec, force=force)
    full_chart_bars: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for timeframe in ("1D", "1W", "1M"):
        try:
            full_chart_bars[timeframe] = load_display_bars(spec, timeframe, base_bars, force=force)
        except Exception as exc:
            warnings.append(f"{timeframe}: {type(exc).__name__}: {exc}")
    results: dict[str, dict[str, Any]] = {}
    engine = ElliottWaveEngine()
    for analysis_window in analysis_windows:
        if analysis_window not in ANALYSIS_WINDOWS:
            raise ValueError(f"Unsupported analysis window: {analysis_window}")
        previous = read_snapshot(spec.canonical_asset_id, analysis_window=analysis_window)
        snapshot = engine.analyze(base_bars, spec, analysis_window=analysis_window)
        analysis_start = pd.Timestamp(snapshot["analysis_start"])
        window_base_bars = _from_analysis_start(base_bars, analysis_start)
        chart_bars = {
            timeframe: _from_analysis_start(frame, analysis_start)
            for timeframe, frame in full_chart_bars.items()
            if frame is not None and not frame.empty
        }
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
        snapshot["display_warnings"] = list(warnings)
        if spec.source_id == "tradingview_mcp" and len(base_bars) >= 5000:
            snapshot.setdefault("quality_flags", []).append("PROVIDER_5000_BAR_LIMIT")
            snapshot["provider_bar_limit"] = 5000
            snapshot["display_warnings"].append(
                "Daily analytical history is limited to the latest 5000 bars returned by TradingView MCP."
            )
        snapshot["analysis_snapshot_id"] = snapshot["snapshot_id"]
        snapshot["display_data_versions"] = {
            timeframe: data_version(frame)
            for timeframe, frame in sorted(chart_bars.items())
            if frame is not None and not frame.empty
        }
        snapshot["snapshot_id"] = _published_snapshot_id(snapshot)
        if previous and previous.get("snapshot_id") != snapshot.get("snapshot_id"):
            snapshot["previous_snapshot_id"] = previous.get("snapshot_id")
            _merge_wave_map_history(previous, snapshot)
            if previous.get("data_version") != snapshot.get("data_version") and previous.get("as_of") == snapshot.get("as_of"):
                snapshot.setdefault("quality_flags", []).append("DATA_REVISION")
                snapshot.setdefault("events", []).append(
                    _transition_event("DATA_REVISION", spec.canonical_asset_id, previous, snapshot)
                )
            if previous.get("main_root_scenario_id") != snapshot.get("main_root_scenario_id"):
                snapshot.setdefault("events", []).append(
                    _transition_event("SCENARIO_SUPERSEDED", spec.canonical_asset_id, previous, snapshot)
                )
            snapshot["events"] = sorted(
                snapshot.get("events", []),
                key=lambda row: (str(row.get("known_at")), str(row.get("event_id"))),
            )
        write_snapshot(snapshot, window_base_bars, chart_bars)
        results[analysis_window] = snapshot
    return results


def refresh_all_assets(*, force: bool = False) -> dict[str, Any]:
    previous = read_manifest() or {"assets": []}
    previous_by_asset = {row.get("canonical_asset_id"): row for row in previous.get("assets", [])}
    entries: list[dict[str, Any]] = []
    run_at = pd.Timestamp.now(tz="UTC").isoformat()
    for asset_id, spec in ASSET_SPECS.items():
        try:
            snapshots = refresh_asset_windows(spec, force=force)
            snapshot = snapshots[DEFAULT_ANALYSIS_WINDOW]
            entries.append(
                {
                    "canonical_asset_id": asset_id,
                    "snapshot_id": snapshot["snapshot_id"],
                    "default_analysis_window": DEFAULT_ANALYSIS_WINDOW,
                    "windows": {
                        window: {
                            "snapshot_id": item["snapshot_id"],
                            "status": "CURRENT",
                            "stale_reason": None,
                            "as_of": item["as_of"],
                            "analysis_start": item["analysis_start"],
                            "created_at": item["created_at"],
                        }
                        for window, item in snapshots.items()
                    },
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
        "analysis_window": snapshot.get("analysis_window"),
        "analysis_start": snapshot.get("analysis_start"),
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
        "analysis_window": current.get("analysis_window"),
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


def _from_analysis_start(frame: pd.DataFrame, analysis_start: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.loc[timestamps >= analysis_start].reset_index(drop=True)


def _merge_wave_map_history(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Keep in-window nodes that disappeared from the latest winning map auditable."""
    current_nodes = {str(node.get("node_id")): node for node in current.get("wave_nodes", [])}
    current_selected = [node for node in current_nodes.values() if node.get("map_status") != "ALTERNATIVE"]
    analysis_start = pd.Timestamp(current["analysis_start"])
    carried: list[dict[str, Any]] = []
    for old in previous.get("wave_nodes", []):
        old_id = str(old.get("node_id") or "")
        start = pd.to_datetime((old.get("start_point") or {}).get("pivot_time"), utc=True, errors="coerce")
        if not old_id or old_id in current_nodes or pd.isna(start) or start < analysis_start:
            continue
        superseded = dict(old)
        superseded["map_status"] = "SUPERSEDED"
        replacement = _replacement_node(old, current_selected)
        superseded["superseded_by_node_id"] = replacement and replacement.get("node_id")
        superseded["change_reason"] = "Reclassified by the latest closed-bar Wave Map conflict resolution."
        carried.append(superseded)
        if replacement:
            replacement.setdefault("supersedes", []).append(old_id)
            current.setdefault("structural_edges", []).append(
                _structural_edge("SUPERSEDES", str(replacement["node_id"]), old_id)
            )
            current.setdefault("events", []).append(
                _node_superseded_event(old_id, str(replacement["node_id"]), current)
            )
    if carried:
        current.setdefault("wave_nodes", []).extend(carried)
        current.setdefault("historical_completed_node_ids", []).extend(
            node["node_id"] for node in carried
        )


def _replacement_node(old: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    same_degree = [node for node in candidates if node.get("degree") == old.get("degree")]
    if not same_degree:
        return None
    old_start = int(old.get("start_pivot_index") or 0)
    old_end = int(old.get("end_pivot_index") or 0)
    overlapping = [
        node
        for node in same_degree
        if max(old_start, int(node.get("start_pivot_index") or 0))
        < min(old_end, int(node.get("end_pivot_index") or 0))
    ]
    pool = overlapping or same_degree
    return max(pool, key=lambda node: (int(node.get("end_pivot_index") or 0), str(node.get("node_id"))))


def _structural_edge(edge_type: str, source: str, target: str) -> dict[str, Any]:
    payload = {"edge_type": edge_type, "source_node_id": source, "target_node_id": target}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"edge_id": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18], **payload}


def _node_superseded_event(old_id: str, replacement_id: str, current: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "event_type": "NODE_SUPERSEDED",
        "old_node_id": old_id,
        "new_node_id": replacement_id,
        "change_reason": "Latest closed-bar evidence changed the winning same-degree structure.",
        "analysis_window": current.get("analysis_window"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "event_id": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18],
        **payload,
        "observed_at": current.get("as_of"),
        "known_at": current.get("created_at"),
    }

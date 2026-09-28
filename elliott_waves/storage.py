from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STORAGE_DIR = Path("persistent") / "elliott_waves"
SNAPSHOT_DIR = STORAGE_DIR / "snapshots"
MANIFEST_PATH = STORAGE_DIR / "manifest_latest.json"
JOURNAL_PATH = STORAGE_DIR / "events.jsonl"
QUOTE_PATH = STORAGE_DIR / "quotes_latest.json"


def write_snapshot(
    snapshot: dict[str, Any],
    base_bars: pd.DataFrame,
    chart_bars: dict[str, pd.DataFrame],
) -> None:
    asset_id = str(snapshot["canonical_asset_id"])
    snapshot_id = str(snapshot["snapshot_id"])
    directory = SNAPSHOT_DIR / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = directory / f"{snapshot_id}.json"
    existing = _read_json(snapshot_path)
    if existing:
        # Snapshot identifiers are content-addressed.  Preserve the original
        # created_at and payload instead of rewriting history on an idempotent run.
        snapshot.clear()
        snapshot.update(existing)
    else:
        _atomic_json(snapshot_path, snapshot)
    base_path = directory / f"{snapshot_id}_base.parquet"
    if not base_path.exists():
        _atomic_parquet(base_path, base_bars)
    for timeframe, frame in chart_bars.items():
        if frame is None or frame.empty:
            continue
        chart_path = directory / f"{snapshot_id}_chart_{timeframe}.parquet"
        if not chart_path.exists():
            _atomic_parquet(chart_path, frame)
    _atomic_json(
        directory / "latest.json",
        {
            "canonical_asset_id": asset_id,
            "snapshot_id": snapshot_id,
            "created_at": snapshot.get("created_at"),
            "as_of": snapshot.get("as_of"),
        },
    )
    _append_journal(snapshot)


def read_snapshot(asset_id: str, snapshot_id: str | None = None) -> dict[str, Any] | None:
    directory = SNAPSHOT_DIR / asset_id
    if snapshot_id is None:
        pointer = _read_json(directory / "latest.json")
        if not pointer:
            return None
        snapshot_id = str(pointer.get("snapshot_id") or "")
    if not snapshot_id:
        return None
    return _read_json(directory / f"{snapshot_id}.json")


def read_base_bars(asset_id: str, snapshot_id: str) -> pd.DataFrame:
    return _read_parquet(SNAPSHOT_DIR / asset_id / f"{snapshot_id}_base.parquet")


def read_chart_bars(asset_id: str, snapshot_id: str, timeframe: str) -> pd.DataFrame:
    path = SNAPSHOT_DIR / asset_id / f"{snapshot_id}_chart_{timeframe}.parquet"
    return _read_parquet(path)


def write_manifest(payload: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(MANIFEST_PATH, payload)


def read_manifest() -> dict[str, Any] | None:
    return _read_json(MANIFEST_PATH)


def read_quotes() -> dict[str, Any]:
    return _read_json(QUOTE_PATH) or {"assets": {}}


def write_quotes(payload: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(QUOTE_PATH, payload)


def _append_journal(snapshot: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_entry = {
        "journal_id": f"snapshot:{snapshot.get('snapshot_id')}",
        "event_type": "SNAPSHOT_PUBLISHED",
        "canonical_asset_id": snapshot.get("canonical_asset_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "created_at": snapshot.get("created_at"),
        "as_of": snapshot.get("as_of"),
        "data_version": snapshot.get("data_version"),
        "main_scenario_id": snapshot.get("main_scenario_id"),
        "quality_flags": snapshot.get("quality_flags", []),
    }
    entries = [snapshot_entry]
    for event in snapshot.get("events", []):
        entries.append(
            {
                "journal_id": f"event:{snapshot.get('canonical_asset_id')}:{event.get('event_id')}",
                "canonical_asset_id": snapshot.get("canonical_asset_id"),
                "snapshot_id": snapshot.get("snapshot_id"),
                "data_version": snapshot.get("data_version"),
                **event,
            }
        )
    existing_ids: set[str] = set()
    if JOURNAL_PATH.exists():
        try:
            with JOURNAL_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        journal_id = row.get("journal_id")
                        if journal_id:
                            existing_ids.add(str(journal_id))
                    except Exception:
                        continue
        except Exception:
            pass
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        for entry in entries:
            if str(entry["journal_id"]) in existing_ids:
                continue
            handle.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
            existing_ids.add(str(entry["journal_id"]))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].map(lambda value: isinstance(value, (list, dict, tuple))).any():
            safe[column] = safe[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, default=_json_default)
                if isinstance(value, (list, dict, tuple))
                else value
            )
    safe.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
        if "lineage" in frame.columns:
            frame["lineage"] = frame["lineage"].map(_maybe_json)
        return frame
    except Exception:
        return pd.DataFrame()


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

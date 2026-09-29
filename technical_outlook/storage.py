from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STORAGE_DIR = Path(__file__).resolve().parents[1] / "persistent" / "technical_outlook"
SNAPSHOT_DIR = STORAGE_DIR / "snapshots"
CHART_DIR = STORAGE_DIR / "charts"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
SETTINGS_PATH = STORAGE_DIR / "settings.json"


def write_snapshot(snapshot: dict[str, Any], charts: dict[str, pd.DataFrame]) -> Path:
    ticker = safe_ticker(str(snapshot["ticker"]))
    snapshot_id = str(snapshot["snapshot_id"])
    directory = SNAPSHOT_DIR / ticker
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot_id}.json"
    _atomic_json(path, snapshot)
    _atomic_json(directory / "latest.json", {"snapshot_id": snapshot_id, "path": path.name})

    chart_directory = CHART_DIR / ticker
    chart_directory.mkdir(parents=True, exist_ok=True)
    for timeframe, frame in charts.items():
        if frame is None or frame.empty:
            continue
        _atomic_parquet(chart_directory / f"{snapshot_id}_{timeframe}.parquet", frame)
    return path


def read_latest_snapshot(ticker: str) -> dict[str, Any] | None:
    directory = SNAPSHOT_DIR / safe_ticker(ticker)
    pointer = _read_json(directory / "latest.json")
    if not pointer:
        return None
    return _read_json(directory / str(pointer.get("path") or ""))


def read_chart_bars(ticker: str, snapshot_id: str, timeframe: str) -> pd.DataFrame:
    path = CHART_DIR / safe_ticker(ticker) / f"{snapshot_id}_{timeframe}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def write_manifest(payload: dict[str, Any]) -> None:
    _atomic_json(MANIFEST_PATH, payload)


def read_manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH) or {"assets": []}


def read_settings() -> dict[str, Any]:
    return {"use_llm_interpretation": False, **(_read_json(SETTINGS_PATH) or {})}


def write_settings(settings: dict[str, Any]) -> None:
    current = read_settings()
    current.update(settings)
    _atomic_json(SETTINGS_PATH, current)


def snapshot_history(ticker: str) -> list[dict[str, Any]]:
    directory = SNAPSHOT_DIR / safe_ticker(ticker)
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "latest.json":
            continue
        payload = _read_json(path)
        if payload:
            items.append(payload)
    return items


def safe_ticker(ticker: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", ".", "_"} else "_" for character in ticker.upper())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _jsonable(payload)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(serializable, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value

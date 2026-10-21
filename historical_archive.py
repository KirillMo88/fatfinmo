from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import pandas as pd


DEFAULT_ARCHIVE_ROOT = Path("persistent") / "historical_archive"
FRAME_COLUMNS = ("Series_ID", "Date", "Value")


class HistoricalArchiveError(RuntimeError):
    """Raised when an archive candidate or stored version is invalid."""


@dataclass(frozen=True)
class ArchiveResult:
    provider: str
    dataset: str
    status: str
    rows: int
    last_observation: str | None
    path: Path


def archive_root() -> Path:
    configured = os.environ.get("HISTORICAL_ARCHIVE_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_ARCHIVE_ROOT


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("._") or "unknown"


def dataset_dir(provider: str, dataset: str, *, root: Path | None = None) -> Path:
    return Path(root or archive_root()) / _slug(provider.upper()) / _slug(dataset.upper())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _coerce_frame(
    frame: pd.DataFrame,
    *,
    series_id: str | None,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise HistoricalArchiveError("archive candidate is empty")
    missing = {date_column, value_column} - set(frame.columns)
    if missing:
        raise HistoricalArchiveError(f"archive candidate is missing columns: {', '.join(sorted(missing))}")

    out = frame.copy()
    if series_id is not None:
        out["Series_ID"] = str(series_id).strip().upper()
    elif "Series_ID" not in out.columns:
        raise HistoricalArchiveError("archive candidate has no Series_ID column")
    out = out.rename(columns={date_column: "Date", value_column: "Value"})
    out["Series_ID"] = out["Series_ID"].astype(str).str.strip().str.upper()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Series_ID", "Date", "Value"])
    out = out.sort_values(["Series_ID", "Date"]).drop_duplicates(["Series_ID", "Date"], keep="last")
    if out.empty:
        raise HistoricalArchiveError("archive candidate has no valid observations")
    return out[list(FRAME_COLUMNS)].reset_index(drop=True)


def _validate_candidate(candidate: pd.DataFrame, current: pd.DataFrame, min_rows: int) -> None:
    if len(candidate) < max(1, int(min_rows)):
        raise HistoricalArchiveError(f"archive candidate has only {len(candidate)} rows; minimum is {min_rows}")
    if current.empty:
        return
    for series_id, previous in current.groupby("Series_ID"):
        incoming = candidate.loc[candidate["Series_ID"].eq(series_id)]
        if incoming.empty:
            raise HistoricalArchiveError(f"archive candidate lost series {series_id}")
        previous_dates = pd.to_datetime(previous["Date"], errors="coerce").dropna()
        incoming_dates = pd.to_datetime(incoming["Date"], errors="coerce").dropna()
        if previous_dates.empty or incoming_dates.empty:
            continue
        if incoming_dates.max() < previous_dates.max():
            raise HistoricalArchiveError(f"archive candidate for {series_id} ends before last_good")
        allowed_loss = max(2, int(len(previous) * 0.02))
        if len(incoming) + allowed_loss < len(previous):
            raise HistoricalArchiveError(f"archive candidate for {series_id} lost too much history")


def read_version(
    provider: str,
    dataset: str,
    *,
    version: str = "last_good",
    root: Path | None = None,
) -> pd.DataFrame:
    if version not in {"last_good", "previous_good"}:
        raise ValueError("version must be last_good or previous_good")
    path = dataset_dir(provider, dataset, root=root) / f"{version}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=list(FRAME_COLUMNS))
    try:
        return _coerce_frame(
            pd.read_parquet(path),
            series_id=None,
            date_column="Date",
            value_column="Value",
        )
    except Exception as exc:
        raise HistoricalArchiveError(f"stored {version} is invalid: {exc}") from exc


def read_best_available(provider: str, dataset: str, *, root: Path | None = None) -> pd.DataFrame:
    errors: list[str] = []
    for version in ("last_good", "previous_good"):
        try:
            frame = read_version(provider, dataset, version=version, root=root)
            if not frame.empty:
                frame.attrs["archive_status"] = "ARCHIVED" if version == "last_good" else "ARCHIVED_PREVIOUS"
                frame.attrs["archive_version"] = version
                return frame
        except HistoricalArchiveError as exc:
            errors.append(str(exc))
    empty = pd.DataFrame(columns=list(FRAME_COLUMNS))
    if errors:
        empty.attrs["archive_errors"] = errors
    return empty


def save_frame(
    provider: str,
    dataset: str,
    frame: pd.DataFrame,
    *,
    series_id: str | None = None,
    date_column: str = "Date",
    value_column: str = "Value",
    min_rows: int = 2,
    root: Path | None = None,
    source: str | None = None,
) -> ArchiveResult:
    target_dir = dataset_dir(provider, dataset, root=root)
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = _coerce_frame(
        frame,
        series_id=series_id,
        date_column=date_column,
        value_column=value_column,
    )
    try:
        current = read_version(provider, dataset, root=root)
    except HistoricalArchiveError:
        current = pd.DataFrame(columns=list(FRAME_COLUMNS))
    _validate_candidate(candidate, current, min_rows)

    last_path = target_dir / "last_good.parquet"
    previous_path = target_dir / "previous_good.parquet"
    metadata_path = target_dir / "metadata.json"
    temporary = target_dir / f".candidate.{uuid4().hex}.parquet.tmp"
    candidate.to_parquet(temporary, index=False)
    candidate_hash = _sha256(temporary)
    current_hash = _sha256(last_path) if last_path.exists() else None
    now = datetime.now(timezone.utc).isoformat()
    metadata: dict = {}
    if metadata_path.exists():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = parsed if isinstance(parsed, dict) else {}
        except (OSError, ValueError):
            metadata = {}

    if candidate_hash == current_hash:
        temporary.unlink(missing_ok=True)
        metadata["last_checked_at_utc"] = now
        _atomic_json(metadata_path, metadata)
        status = "UNCHANGED"
    else:
        if last_path.exists():
            previous_tmp = target_dir / f".previous.{uuid4().hex}.tmp"
            shutil.copy2(last_path, previous_tmp)
            os.replace(previous_tmp, previous_path)
            if metadata.get("last_good"):
                metadata["previous_good"] = metadata["last_good"]
        os.replace(temporary, last_path)
        metadata["last_good"] = {
            "stored_at_utc": now,
            "rows": len(candidate),
            "first_observation": candidate["Date"].min().date().isoformat(),
            "last_observation": candidate["Date"].max().date().isoformat(),
            "sha256": candidate_hash,
            "source": source or provider,
        }
        metadata["last_checked_at_utc"] = now
        _atomic_json(metadata_path, metadata)
        status = "CREATED" if current.empty else "ROTATED"

    return ArchiveResult(
        provider=str(provider).upper(),
        dataset=str(dataset).upper(),
        status=status,
        rows=len(candidate),
        last_observation=candidate["Date"].max().date().isoformat(),
        path=last_path,
    )


def save_fred_series(series_id: str, frame: pd.DataFrame, *, root: Path | None = None) -> ArchiveResult:
    return save_frame(
        "FRED",
        series_id,
        frame,
        series_id=series_id,
        min_rows=2,
        root=root,
        source=f"FRED / {str(series_id).upper()}",
    )


def save_raw_table(
    provider: str,
    dataset: str,
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    group_column: str,
    min_rows: int = 2,
    root: Path | None = None,
    source: str | None = None,
) -> ArchiveResult:
    """Version a validated provider table while preserving all source metadata."""
    if frame is None:
        raise HistoricalArchiveError("raw archive candidate is empty or invalid")
    missing = {date_column, value_column, group_column} - set(frame.columns)
    if frame.empty or missing:
        detail = f"; missing {', '.join(sorted(missing))}" if missing else ""
        raise HistoricalArchiveError(f"raw archive candidate is empty or invalid{detail}")
    candidate = frame.copy()
    candidate[date_column] = pd.to_datetime(candidate[date_column], errors="coerce", utc=True).dt.tz_localize(None)
    candidate[value_column] = pd.to_numeric(candidate[value_column], errors="coerce")
    candidate[group_column] = candidate[group_column].astype(str).str.strip()
    candidate = candidate.dropna(subset=[date_column, value_column])
    candidate = candidate.loc[candidate[group_column].ne("")]
    candidate = candidate.sort_values([group_column, date_column]).drop_duplicates(
        [group_column, date_column], keep="last"
    ).reset_index(drop=True)
    if len(candidate) < max(1, int(min_rows)):
        raise HistoricalArchiveError(f"raw archive candidate has only {len(candidate)} rows")

    target_dir = dataset_dir(provider, dataset, root=root)
    target_dir.mkdir(parents=True, exist_ok=True)
    last_path = target_dir / "last_good.parquet"
    previous_path = target_dir / "previous_good.parquet"
    metadata_path = target_dir / "metadata.json"
    current = pd.DataFrame()
    if last_path.exists():
        try:
            current = pd.read_parquet(last_path)
        except Exception:
            current = pd.DataFrame()
    if not current.empty and group_column in current:
        current_dates = pd.to_datetime(current[date_column], errors="coerce")
        for group, old_group in current.assign(_archive_date=current_dates).groupby(group_column):
            old_valid = old_group.dropna(subset=["_archive_date"])
            new_group = candidate.loc[candidate[group_column].eq(group)]
            if old_valid.empty:
                continue
            if new_group.empty:
                raise HistoricalArchiveError(f"raw archive candidate lost group {group}")
            if pd.to_datetime(new_group[date_column]).max() < old_valid["_archive_date"].max():
                raise HistoricalArchiveError(f"raw archive candidate for {group} ends before last_good")
            allowed_loss = max(2, int(len(old_valid) * 0.02))
            if len(new_group) + allowed_loss < len(old_valid):
                raise HistoricalArchiveError(f"raw archive candidate for {group} lost too much history")

    temporary = target_dir / f".candidate.{uuid4().hex}.parquet.tmp"
    candidate.to_parquet(temporary, index=False)
    candidate_hash = _sha256(temporary)
    current_hash = _sha256(last_path) if last_path.exists() else None
    now = datetime.now(timezone.utc).isoformat()
    metadata: dict = {}
    if metadata_path.exists():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = parsed if isinstance(parsed, dict) else {}
        except (OSError, ValueError):
            pass
    if candidate_hash == current_hash:
        temporary.unlink(missing_ok=True)
        status = "UNCHANGED"
    else:
        if last_path.exists():
            previous_tmp = target_dir / f".previous.{uuid4().hex}.tmp"
            shutil.copy2(last_path, previous_tmp)
            os.replace(previous_tmp, previous_path)
            if metadata.get("last_good"):
                metadata["previous_good"] = metadata["last_good"]
        os.replace(temporary, last_path)
        metadata["last_good"] = {
            "stored_at_utc": now,
            "rows": len(candidate),
            "first_observation": candidate[date_column].min().date().isoformat(),
            "last_observation": candidate[date_column].max().date().isoformat(),
            "sha256": candidate_hash,
            "source": source or provider,
            "date_column": date_column,
            "value_column": value_column,
            "group_column": group_column,
        }
        status = "CREATED" if current.empty else "ROTATED"
    metadata["last_checked_at_utc"] = now
    _atomic_json(metadata_path, metadata)
    return ArchiveResult(
        provider=str(provider).upper(),
        dataset=str(dataset).upper(),
        status=status,
        rows=len(candidate),
        last_observation=candidate[date_column].max().date().isoformat(),
        path=last_path,
    )


def read_raw_table(
    provider: str,
    dataset: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    target_dir = dataset_dir(provider, dataset, root=root)
    for version in ("last_good", "previous_good"):
        path = target_dir / f"{version}.parquet"
        try:
            if path.exists():
                frame = pd.read_parquet(path)
                if not frame.empty:
                    frame.attrs["archive_status"] = "ARCHIVED" if version == "last_good" else "ARCHIVED_PREVIOUS"
                    return frame
        except Exception:
            continue
    return pd.DataFrame()


def save_model_market_history(
    ticker: str,
    frame: pd.DataFrame,
    *,
    root: Path | None = None,
) -> ArchiveResult:
    if frame is None or frame.empty or "Close" not in frame:
        raise HistoricalArchiveError(f"market archive candidate is empty for {ticker}")
    candidate = frame.copy()
    candidate.index = pd.to_datetime(candidate.index, errors="coerce", utc=True).tz_localize(None)
    candidate.index.name = "Date"
    candidate = candidate.reset_index()
    candidate["Series_ID"] = str(ticker).strip().upper()
    return save_raw_table(
        "MODEL_MARKET",
        ticker,
        candidate,
        date_column="Date",
        value_column="Close",
        group_column="Series_ID",
        min_rows=52,
        root=root,
        source=f"Yahoo Finance adjusted OHLCV / {ticker}",
    )


def read_model_market_history(
    ticker: str,
    *,
    period: str = "max",
    root: Path | None = None,
) -> pd.DataFrame:
    frame = read_raw_table("MODEL_MARKET", ticker, root=root)
    if frame.empty or "Date" not in frame or "Close" not in frame:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
    lookback = _market_period_lookback(period)
    if lookback is not None and not frame.empty:
        frame = frame.loc[frame.index >= frame.index.max() - lookback]
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    out = frame[["Open", "High", "Low", "Close", "Volume"]]
    out.attrs.update(frame.attrs)
    out.attrs["source_status"] = frame.attrs.get("archive_status", "ARCHIVED")
    return out


def _market_period_lookback(period: str) -> pd.Timedelta | None:
    value = str(period or "max").strip().lower()
    if value == "max":
        return None
    match = re.fullmatch(r"(\d+)(d|mo|y)", value)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    days = amount if unit == "d" else amount * 31 if unit == "mo" else amount * 366
    return pd.Timedelta(days=days)


def save_validated_file(
    provider: str,
    dataset: str,
    source_path: Path,
    *,
    root: Path | None = None,
) -> ArchiveResult:
    source_path = Path(source_path)
    if not source_path.exists() or source_path.stat().st_size <= 0:
        raise HistoricalArchiveError(f"validated source file is missing or empty: {source_path}")
    target_dir = dataset_dir(provider, dataset, root=root)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix or ".bin"
    last_path = target_dir / f"last_good{suffix}"
    previous_path = target_dir / f"previous_good{suffix}"
    metadata_path = target_dir / "metadata.json"
    source_hash = _sha256(source_path)
    current_hash = _sha256(last_path) if last_path.exists() else None
    now = datetime.now(timezone.utc).isoformat()
    metadata: dict = {}
    if metadata_path.exists():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = parsed if isinstance(parsed, dict) else {}
        except (OSError, ValueError):
            pass
    if source_hash == current_hash:
        status = "UNCHANGED"
    else:
        if last_path.exists():
            previous_tmp = target_dir / f".previous.{uuid4().hex}.tmp"
            shutil.copy2(last_path, previous_tmp)
            os.replace(previous_tmp, previous_path)
            if metadata.get("last_good"):
                metadata["previous_good"] = metadata["last_good"]
        last_tmp = target_dir / f".last.{uuid4().hex}.tmp"
        shutil.copy2(source_path, last_tmp)
        os.replace(last_tmp, last_path)
        metadata["last_good"] = {
            "stored_at_utc": now,
            "bytes": source_path.stat().st_size,
            "sha256": source_hash,
            "source_file": source_path.name,
        }
        status = "CREATED" if current_hash is None else "ROTATED"
    metadata["last_checked_at_utc"] = now
    _atomic_json(metadata_path, metadata)
    return ArchiveResult(
        provider=str(provider).upper(),
        dataset=str(dataset).upper(),
        status=status,
        rows=0,
        last_observation=None,
        path=last_path,
    )


def restore_validated_file(
    provider: str,
    dataset: str,
    destination: Path,
    *,
    root: Path | None = None,
) -> bool:
    target_dir = dataset_dir(provider, dataset, root=root)
    destination = Path(destination)
    for version in ("last_good", "previous_good"):
        candidates = list(target_dir.glob(f"{version}.*"))
        if not candidates:
            continue
        source = candidates[0]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            return True
        except OSError:
            continue
    return False


def read_fred_series(
    series_id: str,
    *,
    observation_start: str | None = None,
    observation_end: str | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    frame = read_best_available("FRED", series_id, root=root)
    if frame.empty:
        return frame
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if observation_start:
        frame = frame.loc[dates.ge(pd.Timestamp(observation_start))]
        dates = pd.to_datetime(frame["Date"], errors="coerce")
    if observation_end:
        frame = frame.loc[dates.le(pd.Timestamp(observation_end))]
    return frame.reset_index(drop=True)


def archive_manifest(*, root: Path | None = None) -> pd.DataFrame:
    base = Path(root or archive_root())
    rows: list[dict] = []
    if not base.exists():
        return pd.DataFrame()
    for metadata_path in base.glob("*/*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            last = metadata.get("last_good") or {}
            rows.append(
                {
                    "Provider": metadata_path.parent.parent.name,
                    "Dataset": metadata_path.parent.name,
                    "Rows": last.get("rows"),
                    "FirstObservation": last.get("first_observation"),
                    "LastObservation": last.get("last_observation"),
                    "StoredAtUTC": last.get("stored_at_utc"),
                    "HasPrevious": (metadata_path.parent / "previous_good.parquet").exists(),
                }
            )
        except (OSError, ValueError):
            continue
    return pd.DataFrame(rows).sort_values(["Provider", "Dataset"]).reset_index(drop=True) if rows else pd.DataFrame()


def registered_series_status(series_ids: Iterable[str], *, root: Path | None = None) -> dict[str, str]:
    return {
        str(series_id).upper(): (
            "AVAILABLE" if not read_best_available("FRED", str(series_id), root=root).empty else "MISSING"
        )
        for series_id in series_ids
    }

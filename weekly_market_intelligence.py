from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from screener_metrics import historical_momentum_52w_metrics


ANALYSIS_VERSION = "1.0.0"
PROMPT_VERSION = "1.0"
RULES_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
DEFAULT_DB_PATH = Path("persistent") / "weekly_market_intelligence" / "weekly_market_intelligence.sqlite3"

THESIS_TYPES = {
    "RISK_APPETITE_ROTATION",
    "GEOGRAPHIC_ROTATION",
    "COMMODITY_ROTATION",
    "DURATION_CREDIT_DIVERGENCE",
    "MOMENTUM_TRANSITION",
    "MARKET_BREADTH",
    "GROUP_SPECIFIC",
}
LIFECYCLES = {"NEW", "DEVELOPING", "ESTABLISHED", "WEAKENING", "RESOLVED", "INVALIDATED"}
EVIDENCE_ASSESSMENTS = {"STRONG", "MODERATE", "PRELIMINARY"}

DEFAULT_RELATIONSHIPS: tuple[tuple[str, str, str], ...] = (
    ("Crypto BTC", "Crypto Altcoins", "CRYPTO_LEADERSHIP"),
    ("Gold", "Silver", "PRECIOUS_METALS_LEADERSHIP"),
    ("US Sectors Aggressive", "US Sectors Defensive", "US_RISK_LEADERSHIP"),
    ("US Sectors Defensive", "US Sectors Commodity", "DEFENSIVE_COMMODITY"),
    ("Tech", "AI", "TECH_AI_LEADERSHIP"),
    ("SPY", "LATAM", "US_LATAM_ROTATION"),
    ("SPY", "Asia", "US_ASIA_ROTATION"),
    ("SPY", "EU", "US_EU_ROTATION"),
    ("Energy", "Metals", "ENERGY_METALS_ROTATION"),
    ("Energy", "Agriculture", "ENERGY_AGRICULTURE_ROTATION"),
    ("Bonds UST", "Bonds HY", "DURATION_CREDIT"),
    ("Bonds UST MT", "Bonds UST LT", "DURATION_CURVE"),
)


@dataclass(frozen=True)
class IntelligenceConfig:
    timezone_name: str = "Europe/Lisbon"
    schedule_hour: int = 23
    schedule_minute: int = 30
    min_group_coverage: float = 0.60
    min_complete_group_ratio: float = 0.80
    max_events: int = 15
    relative_threshold_pp: float = 5.0
    relative_shift_threshold_pp: float = 3.0
    trend_divergence_threshold: float = 15.0
    momentum_spread_threshold: float = 10.0
    momentum_delta_threshold: float = 8.0
    breadth_delta_threshold_pp: float = 15.0
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 5000

    @classmethod
    def from_env(cls) -> "IntelligenceConfig":
        schedule = os.getenv("WMI_SCHEDULE_LOCAL", "23:30").strip().split(":", 1)
        try:
            hour, minute = int(schedule[0]), int(schedule[1])
        except (ValueError, IndexError):
            hour, minute = 23, 30
        return cls(
            timezone_name=os.getenv("WMI_TIMEZONE", "Europe/Lisbon"),
            schedule_hour=max(0, min(23, hour)),
            schedule_minute=max(0, min(59, minute)),
            min_group_coverage=float(os.getenv("WMI_MIN_GROUP_COVERAGE", "0.60")),
            min_complete_group_ratio=float(os.getenv("WMI_MIN_COMPLETE_GROUP_RATIO", "0.80")),
            max_events=max(12, min(18, int(os.getenv("WMI_MAX_EVENTS", "15")))),
            relative_threshold_pp=float(os.getenv("WMI_RELATIVE_THRESHOLD_PP", "5")),
            relative_shift_threshold_pp=float(os.getenv("WMI_RELATIVE_SHIFT_THRESHOLD_PP", "3")),
            trend_divergence_threshold=float(os.getenv("WMI_TREND_DIVERGENCE_THRESHOLD", "15")),
            momentum_spread_threshold=float(os.getenv("WMI_MOMENTUM_SPREAD_THRESHOLD", "10")),
            momentum_delta_threshold=float(os.getenv("WMI_MOMENTUM_DELTA_THRESHOLD", "8")),
            breadth_delta_threshold_pp=float(os.getenv("WMI_BREADTH_DELTA_THRESHOLD_PP", "15")),
            model=os.getenv("WMI_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-sol")),
            reasoning_effort=os.getenv("WMI_REASONING_EFFORT", "medium"),
            max_output_tokens=int(os.getenv("WMI_MAX_OUTPUT_TOKENS", "5000")),
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def latest_eligible_week_end(now_utc: datetime, config: IntelligenceConfig) -> date:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local_now = now_utc.astimezone(ZoneInfo(config.timezone_name))
    days_since_friday = (local_now.weekday() - 4) % 7
    candidate = local_now.date() - timedelta(days=days_since_friday)
    gate = datetime.combine(
        candidate + timedelta(days=1),
        time(config.schedule_hour, config.schedule_minute),
        ZoneInfo(config.timezone_name),
    )
    if local_now < gate:
        candidate -= timedelta(days=7)
    return candidate


def snapshot_cutoff(week_end: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(week_end, time(23, 59, 59), timezone.utc))


def latest_completed_week_end(now_utc: datetime) -> date:
    """Latest Friday whose UTC cutoff has fully elapsed; used by explicit manual runs."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    now_utc = now_utc.astimezone(timezone.utc)
    days_since_friday = (now_utc.weekday() - 4) % 7
    candidate = now_utc.date() - timedelta(days=days_since_friday)
    if now_utc < snapshot_cutoff(candidate).to_pydatetime():
        candidate -= timedelta(days=7)
    return candidate


class IntelligenceRepository:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    run_key TEXT NOT NULL UNIQUE,
                    week_end TEXT NOT NULL,
                    data_snapshot_id TEXT,
                    snapshot_as_of TEXT,
                    generation_timestamp TEXT,
                    publication_timestamp TEXT,
                    model_identifier TEXT,
                    prompt_version TEXT NOT NULL,
                    rules_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    analysis_version TEXT NOT NULL,
                    selected_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    output_json TEXT,
                    thesis_ids_json TEXT NOT NULL DEFAULT '[]',
                    lifecycle_changes_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    validation_json TEXT NOT NULL DEFAULT '{}',
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost REAL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reports_week_status ON reports(week_end DESC, status);
                CREATE TABLE IF NOT EXISTS group_snapshots (
                    snapshot_id TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    analysis_version TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(week_end, analysis_version, group_id)
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_week ON group_snapshots(week_end DESC);
                CREATE TABLE IF NOT EXISTS market_events (
                    event_id TEXT PRIMARY KEY,
                    week_end TEXT NOT NULL,
                    theme_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    significance_score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_theme_week ON market_events(theme_key, week_end DESC);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_key TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_run ON attempts(run_key, attempted_at DESC);
                """
            )

    def is_published(self, week_end: date, analysis_version: str = ANALYSIS_VERSION) -> bool:
        run_key = f"{week_end.isoformat()}:{analysis_version}"
        with self.connect() as db:
            return db.execute(
                "SELECT 1 FROM reports WHERE run_key=? AND status='PUBLISHED'", (run_key,)
            ).fetchone() is not None

    def claim_run(self, week_end: date, analysis_version: str = ANALYSIS_VERSION) -> str:
        """Atomically prevent duplicate model calls across scheduler and UI processes."""
        run_key = f"{week_end.isoformat()}:{analysis_version}"
        report_id = _stable_id("report", run_key)
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(hours=2)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, updated_at FROM reports WHERE run_key=?", (run_key,)
            ).fetchone()
            if row and row["status"] == "PUBLISHED":
                return "ALREADY_PUBLISHED"
            if row and row["status"] == "GENERATING":
                try:
                    updated_at = datetime.fromisoformat(row["updated_at"])
                except (TypeError, ValueError):
                    updated_at = now
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                if updated_at >= stale_before:
                    return "RUN_IN_PROGRESS"
            timestamp = now.isoformat()
            db.execute(
                """INSERT OR REPLACE INTO reports(
                    report_id, run_key, week_end, prompt_version, rules_version, schema_version,
                    analysis_version, status, validation_json, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report_id,
                    run_key,
                    week_end.isoformat(),
                    PROMPT_VERSION,
                    RULES_VERSION,
                    SCHEMA_VERSION,
                    analysis_version,
                    "GENERATING",
                    "{}",
                    timestamp,
                    timestamp,
                ),
            )
        return "CLAIMED"

    def record_attempt(self, run_key: str, status: str, details: Mapping[str, Any] | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO attempts(run_key, attempted_at, status, details_json) VALUES(?,?,?,?)",
                (run_key, datetime.now(timezone.utc).isoformat(), status, _json(details or {})),
            )

    def save_snapshots(self, snapshot_id: str, week_end: date, snapshots: Sequence[Mapping[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            for item in snapshots:
                db.execute(
                    """INSERT OR REPLACE INTO group_snapshots
                    (snapshot_id, week_end, analysis_version, group_id, group_name, payload_json, created_at)
                    VALUES(?,?,?,?,?,?,?)""",
                    (
                        snapshot_id,
                        week_end.isoformat(),
                        ANALYSIS_VERSION,
                        item["groupId"],
                        item["groupName"],
                        _json(item),
                        now,
                    ),
                )

    def load_snapshots(self, week_end: date) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload_json FROM group_snapshots WHERE week_end=? ORDER BY group_name",
                (week_end.isoformat(),),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def load_snapshots_at_or_before(self, week_end: date) -> list[dict[str, Any]]:
        with self.connect() as db:
            row = db.execute(
                "SELECT week_end FROM group_snapshots WHERE week_end<=? ORDER BY week_end DESC LIMIT 1",
                (week_end.isoformat(),),
            ).fetchone()
        return self.load_snapshots(date.fromisoformat(row[0])) if row else []

    def save_events(self, week_end: date, events: Sequence[Mapping[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            for event in events:
                db.execute(
                    """INSERT OR REPLACE INTO market_events
                    (event_id, week_end, theme_key, event_type, significance_score, payload_json, created_at)
                    VALUES(?,?,?,?,?,?,?)""",
                    (
                        event["eventId"],
                        week_end.isoformat(),
                        event["themeKey"],
                        event["type"],
                        event["significanceScore"],
                        _json(event),
                        now,
                    ),
                )

    def load_events(self, week_end: date) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload_json FROM market_events WHERE week_end=? ORDER BY significance_score DESC",
                (week_end.isoformat(),),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def previous_event(self, theme_key: str, before_week: date) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT payload_json FROM market_events
                WHERE theme_key=? AND week_end<? ORDER BY week_end DESC LIMIT 1""",
                (theme_key, before_week.isoformat()),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def previous_published(self, before_week: date | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM reports WHERE status='PUBLISHED'"
        params: list[Any] = []
        if before_week is not None:
            query += " AND week_end<?"
            params.append(before_week.isoformat())
        query += " ORDER BY week_end DESC, publication_timestamp DESC LIMIT 1"
        with self.connect() as db:
            row = db.execute(query, params).fetchone()
        return self._decode_report(row) if row else None

    def list_published(self, limit: int = 104) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM reports WHERE status='PUBLISHED' ORDER BY week_end DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_report(row) for row in rows]

    def latest_run_status(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT week_end, status, error, updated_at, validation_json FROM reports ORDER BY week_end DESC, updated_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["validation"] = json.loads(result.pop("validation_json") or "{}")
        return result

    @staticmethod
    def _decode_report(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in (
            "selected_event_ids_json",
            "output_json",
            "thesis_ids_json",
            "lifecycle_changes_json",
            "validation_json",
        ):
            raw = result.pop(key, None)
            result[key.removesuffix("_json")] = json.loads(raw) if raw else None
        return result

    def publish(
        self,
        *,
        week_end: date,
        snapshot_id: str,
        output: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any],
        usage: Mapping[str, Any],
        model: str,
        lifecycle_changes: Sequence[Mapping[str, Any]],
        generation_timestamp: str,
    ) -> str:
        run_key = f"{week_end.isoformat()}:{ANALYSIS_VERSION}"
        report_id = _stable_id("report", run_key)
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            existing = db.execute("SELECT status FROM reports WHERE run_key=?", (run_key,)).fetchone()
            if existing and existing[0] == "PUBLISHED":
                return report_id
            db.execute(
                """INSERT OR REPLACE INTO reports(
                    report_id, run_key, week_end, data_snapshot_id, snapshot_as_of,
                    generation_timestamp, publication_timestamp, model_identifier,
                    prompt_version, rules_version, schema_version, analysis_version,
                    selected_event_ids_json, output_json, thesis_ids_json,
                    lifecycle_changes_json, status, validation_json, input_tokens,
                    output_tokens, total_tokens, estimated_cost, error, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report_id,
                    run_key,
                    week_end.isoformat(),
                    snapshot_id,
                    snapshot_cutoff(week_end).isoformat(),
                    generation_timestamp,
                    now,
                    model,
                    PROMPT_VERSION,
                    RULES_VERSION,
                    SCHEMA_VERSION,
                    ANALYSIS_VERSION,
                    _json([event["eventId"] for event in events]),
                    _json(output),
                    _json([thesis["themeKey"] for thesis in output.get("theses", [])]),
                    _json(list(lifecycle_changes)),
                    "PUBLISHED",
                    _json(validation),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("total_tokens"),
                    usage.get("estimated_cost"),
                    None,
                    now,
                    now,
                ),
            )
        return report_id

    def record_failure(self, week_end: date, status: str, error: str, validation: Any = None) -> None:
        run_key = f"{week_end.isoformat()}:{ANALYSIS_VERSION}"
        report_id = _stable_id("report", run_key)
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            existing = db.execute("SELECT status FROM reports WHERE run_key=?", (run_key,)).fetchone()
            if existing and existing[0] == "PUBLISHED":
                return
            db.execute(
                """INSERT OR REPLACE INTO reports(
                    report_id, run_key, week_end, prompt_version, rules_version, schema_version,
                    analysis_version, status, validation_json, error, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report_id,
                    run_key,
                    week_end.isoformat(),
                    PROMPT_VERSION,
                    RULES_VERSION,
                    SCHEMA_VERSION,
                    ANALYSIS_VERSION,
                    status,
                    _json(validation or {}),
                    error,
                    now,
                    now,
                ),
            )


def signed_adx_di(adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    adx_values = pd.to_numeric(adx, errors="coerce")
    plus_values = pd.to_numeric(plus_di, errors="coerce")
    minus_values = pd.to_numeric(minus_di, errors="coerce")
    balance = (plus_values - minus_values) / (plus_values + minus_values).replace(0.0, np.nan)
    strength = pd.Series(
        np.interp(adx_values.fillna(0.0), [15, 20, 25, 30, 35], [0, 40, 70, 90, 100], left=0, right=100),
        index=adx_values.index,
    )
    return (strength * (balance / 0.25).clip(-1, 1)).where(adx_values.notna() & balance.notna())


def _calendar_return(close: pd.Series, days: int) -> float | None:
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if len(prices) < 2:
        return None
    target = prices.index[-1] - pd.Timedelta(days=days)
    position = prices.index.searchsorted(target, side="right") - 1
    if position < 0 or float(prices.iloc[position]) == 0:
        return None
    return _safe_float((float(prices.iloc[-1]) / float(prices.iloc[position]) - 1.0) * 100.0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce")
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    return result.where(average_loss.ne(0.0), 100.0)


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.to_datetime(result.index, utc=True, errors="coerce")
    result.index = index
    return result.loc[~result.index.isna()].sort_index()


def build_instrument_metrics(
    tickers: Iterable[str],
    week_end: date,
    history_loader: Callable[[str], pd.DataFrame],
    weekly_builder: Callable[[pd.DataFrame, bool], pd.DataFrame],
    adx_calculator: Callable[[pd.Series, pd.Series, pd.Series, int], tuple[pd.Series, pd.Series, pd.Series]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = snapshot_cutoff(week_end)
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for ticker in sorted(set(str(item) for item in tickers if item)):
        try:
            history = history_loader(ticker)
            if history is None or history.empty:
                errors[ticker] = "empty history"
                continue
            history = _normalize_index(history)
            history = history.loc[history.index <= cutoff]
            required = ["High", "Low", "Close"]
            if history.empty or any(column not in history for column in required):
                errors[ticker] = "missing OHLC data"
                continue
            for column in ("Open", "High", "Low", "Close", "Volume"):
                if column not in history:
                    history[column] = np.nan
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if close.empty:
                errors[ticker] = "missing close"
                continue
            weekly = weekly_builder(history[["Open", "High", "Low", "Close", "Volume"]], False)
            weekly = _normalize_index(weekly)
            daily_adx, daily_plus, daily_minus = adx_calculator(
                history["High"], history["Low"], history["Close"], 14
            )
            weekly_adx, weekly_plus, weekly_minus = adx_calculator(
                weekly["High"], weekly["Low"], weekly["Close"], 14
            )
            daily_signed = signed_adx_di(daily_adx, daily_plus, daily_minus)
            weekly_signed = signed_adx_di(weekly_adx, weekly_plus, weekly_minus)
            perf12m_pct, _ = historical_momentum_52w_metrics(close)
            rows.append(
                {
                    "ticker": ticker,
                    "lastDate": close.index[-1].date().isoformat(),
                    "w1": _calendar_return(close, 7),
                    "m1": _calendar_return(close, 30),
                    "m3": _calendar_return(close, 91),
                    "m6": _calendar_return(close, 182),
                    "rsiDaily": _safe_float(_rsi(close, period=14).iloc[-1]),
                    "rsiWeekly": _safe_float(_rsi(weekly["Close"], period=14).iloc[-1]),
                    "signedADXDI14D": _safe_float(daily_signed.iloc[-1]),
                    "signedADXDI14W": _safe_float(weekly_signed.iloc[-1]),
                    "perf12MPercentile": _safe_float(perf12m_pct),
                }
            )
        except Exception as exc:
            errors[ticker] = str(exc)[:300]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for source, target in (("m1", "p1m"), ("m3", "p3m"), ("m6", "p6m")):
            frame[target] = pd.to_numeric(frame[source], errors="coerce").rank(pct=True, method="average") * 100.0
        frame["midTerm"] = 0.20 * frame["p1m"] + 0.45 * frame["p3m"] + 0.35 * frame["p6m"]
        frame["shortTerm"] = 0.45 * frame["p1m"] + 0.35 * frame["p3m"] + 0.20 * frame["p6m"]
        frame["momentumSpread"] = frame["shortTerm"] - frame["midTerm"]
    return frame, {"errors": errors, "requested": len(set(tickers)), "valid": len(frame)}


def aggregate_group_snapshots(
    instrument_metrics: pd.DataFrame,
    group_members: Mapping[str, Sequence[str]],
    week_end: date,
    config: IntelligenceConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_ticker = instrument_metrics.set_index("ticker") if not instrument_metrics.empty else pd.DataFrame()
    snapshots: list[dict[str, Any]] = []
    for group_name, members_raw in sorted(group_members.items()):
        members = sorted(set(str(item) for item in members_raw if item))
        available = by_ticker.loc[by_ticker.index.intersection(members)].copy() if not by_ticker.empty else pd.DataFrame()
        valid_count = len(available)
        coverage = valid_count / len(members) if members else 0.0

        def median(column: str) -> float | None:
            if available.empty or column not in available:
                return None
            return _safe_float(pd.to_numeric(available[column], errors="coerce").median())

        def breadth(column: str, positive: bool) -> float | None:
            if available.empty or column not in available:
                return None
            values = pd.to_numeric(available[column], errors="coerce").dropna()
            if values.empty:
                return None
            matched = values.gt(0) if positive else values.lt(0)
            return _safe_float(matched.mean() * 100.0)

        extension_valid = available[["rsiWeekly", "perf12MPercentile"]].dropna() if not available.empty else pd.DataFrame()
        extension = None
        if not extension_valid.empty:
            extension = _safe_float(
                ((extension_valid["rsiWeekly"] >= 70) & (extension_valid["perf12MPercentile"] >= 70)).mean()
                * 100.0
            )
        dispersion = {
            column: _safe_float(pd.to_numeric(available[column], errors="coerce").quantile(0.75)
                                - pd.to_numeric(available[column], errors="coerce").quantile(0.25))
            for column in ("m1", "m3", "m6")
        } if not available.empty else {"m1": None, "m3": None, "m6": None}
        snapshots.append(
            {
                "groupId": group_name,
                "groupName": group_name,
                "weekEnd": week_end.isoformat(),
                "performance": {key: median(key) for key in ("w1", "m1", "m3", "m6")},
                "momentum": {
                    "midTerm": median("midTerm"),
                    "shortTerm": median("shortTerm"),
                    "spread": median("momentumSpread"),
                },
                "trend": {
                    "signedADXDI14D": median("signedADXDI14D"),
                    "signedADXDI14W": median("signedADXDI14W"),
                },
                "rsi": {"daily": median("rsiDaily"), "weekly": median("rsiWeekly")},
                "historical": {"perf12MPercentile": median("perf12MPercentile")},
                "breadth": {
                    "bullishDaily": breadth("signedADXDI14D", True),
                    "bullishWeekly": breadth("signedADXDI14W", True),
                    "bearishDaily": breadth("signedADXDI14D", False),
                    "bearishWeekly": breadth("signedADXDI14W", False),
                    "extensionRisk": extension,
                },
                "dispersion": dispersion,
                "dataQuality": {
                    "constituentCount": len(members),
                    "validCount": valid_count,
                    "coverage": round(coverage, 6),
                    "isComplete": coverage >= config.min_group_coverage,
                },
            }
        )
    complete = sum(bool(item["dataQuality"]["isComplete"]) for item in snapshots)
    complete_ratio = complete / len(snapshots) if snapshots else 0.0
    latest_dates = instrument_metrics["lastDate"].tolist() if not instrument_metrics.empty else []
    stale = []
    for _, row in instrument_metrics.iterrows():
        last_date = date.fromisoformat(row["lastDate"])
        tolerance = 0 if str(row["ticker"]).endswith("-USD") else 4
        if (week_end - last_date).days > tolerance:
            stale.append({"ticker": row["ticker"], "lastDate": row["lastDate"]})
    readiness = {
        "ready": bool(snapshots) and complete_ratio >= config.min_complete_group_ratio and not stale,
        "completeGroups": complete,
        "groupCount": len(snapshots),
        "completeGroupRatio": round(complete_ratio, 6),
        "staleInstruments": stale,
        "latestDate": max(latest_dates) if latest_dates else None,
    }
    return snapshots, readiness


def _event(
    week_end: date,
    event_type: str,
    theme_key: str,
    group_ids: Sequence[str],
    current: Mapping[str, Any],
    historical: Mapping[str, Any] | None,
    materiality: float,
    novelty: float,
    persistence_weeks: int,
) -> dict[str, Any]:
    groups = sorted(set(group_ids))
    event_id = _stable_id("evt", week_end.isoformat(), theme_key, event_type)
    supporting = [_stable_id("cond", theme_key, "support", event_type)]
    invalidating = [_stable_id("cond", theme_key, "invalidate", event_type)]
    cross_group = min(100.0, 35.0 + max(0, len(groups) - 1) * 25.0)
    persistence = min(100.0, persistence_weeks * 25.0)
    significance = 0.35 * materiality + 0.25 * novelty + 0.25 * cross_group + 0.15 * persistence
    return {
        "eventId": event_id,
        "themeKey": theme_key,
        "type": event_type,
        "groupIds": groups,
        "currentEvidence": dict(current),
        "historicalEvidence": dict(historical or {}),
        "firstObservedWeek": week_end.isoformat(),
        "persistenceWeeks": persistence_weeks,
        "significanceScore": round(significance, 2),
        "noveltyScore": round(novelty, 2),
        "supportingConditionIds": supporting,
        "invalidationConditionIds": invalidating,
        "conditionDefinitions": [
            {
                "conditionId": supporting[0],
                "kind": "CONFIRMATION",
                "description": f"Связь {event_type} сохраняется или усиливается в следующих завершённых недельных данных.",
            },
            {
                "conditionId": invalidating[0],
                "kind": "INVALIDATION",
                "description": f"Направление связи {event_type} разворачивается либо её материальность опускается ниже заданного правила.",
            },
        ],
    }


def detect_events(
    snapshots: Sequence[Mapping[str, Any]],
    previous_snapshots: Sequence[Mapping[str, Any]],
    week_end: date,
    config: IntelligenceConfig,
    repository: IntelligenceRepository | None = None,
    relationships: Sequence[tuple[str, str, str]] = DEFAULT_RELATIONSHIPS,
    four_week_snapshots: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    current = {item["groupId"]: item for item in snapshots}
    previous = {item["groupId"]: item for item in previous_snapshots}
    candidates: list[dict[str, Any]] = []

    def val(item: Mapping[str, Any] | None, *path: str) -> float | None:
        node: Any = item
        for key in path:
            if not isinstance(node, Mapping):
                return None
            node = node.get(key)
        return _safe_float(node)

    def add(event_type: str, theme: str, groups: Sequence[str], evidence: Mapping[str, Any], old: Mapping[str, Any] | None, materiality: float) -> None:
        prior_event = repository.previous_event(theme, week_end) if repository else None
        weeks = int(prior_event.get("persistenceWeeks", 0)) + 1 if prior_event and prior_event.get("type") == event_type else 1
        novelty = 100.0 if weeks == 1 else max(20.0, 70.0 - weeks * 10.0)
        item = _event(week_end, event_type, theme, groups, evidence, old, min(100.0, materiality), novelty, weeks)
        if prior_event and weeks > 1:
            item["firstObservedWeek"] = prior_event.get("firstObservedWeek", week_end.isoformat())
        candidates.append(item)

    for left_id, right_id, family in relationships:
        if left_id not in current or right_id not in current:
            continue
        left, right = current[left_id], current[right_id]
        left_perf, right_perf = val(left, "performance", "m1"), val(right, "performance", "m1")
        if left_perf is None or right_perf is None:
            continue
        spread = left_perf - right_perf
        old_spread = None
        if left_id in previous and right_id in previous:
            old_left, old_right = val(previous[left_id], "performance", "m1"), val(previous[right_id], "performance", "m1")
            if old_left is not None and old_right is not None:
                old_spread = old_left - old_right
        event_type = None
        if old_spread is not None and np.sign(spread) != np.sign(old_spread) and abs(spread) >= config.relative_shift_threshold_pp:
            event_type = "RELATIVE_LEADERSHIP_SHIFT"
        elif abs(spread) >= config.relative_threshold_pp and (old_spread is None or abs(old_spread) < config.relative_threshold_pp):
            event_type = "RELATIVE_LEADERSHIP_ESTABLISHED"
        elif old_spread is not None and abs(old_spread) >= config.relative_threshold_pp and abs(spread) < abs(old_spread) - config.relative_shift_threshold_pp:
            event_type = "RELATIVE_LEADERSHIP_WEAKENING"
        elif spread <= -config.relative_threshold_pp and (old_spread is None or old_spread > -config.relative_threshold_pp):
            event_type = "RELATIVE_UNDERPERFORMANCE"
        if event_type:
            left_m3, right_m3 = val(left, "performance", "m3"), val(right, "performance", "m3")
            add(
                event_type,
                f"relationship:{family}",
                [left_id, right_id],
                {
                    "leftGroup": left_id,
                    "rightGroup": right_id,
                    "m1SpreadPp": round(spread, 2),
                    "m3SpreadPp": _safe_float(left_m3 - right_m3) if left_m3 is not None and right_m3 is not None else None,
                },
                {"previousM1SpreadPp": _safe_float(old_spread)},
                min(100.0, abs(spread) * 8.0),
            )

    for group_id, item in current.items():
        if not item.get("dataQuality", {}).get("isComplete"):
            continue
        daily, weekly = val(item, "trend", "signedADXDI14D"), val(item, "trend", "signedADXDI14W")
        if daily is not None and weekly is not None:
            gap = daily - weekly
            old_daily = val(previous.get(group_id), "trend", "signedADXDI14D")
            old_weekly = val(previous.get(group_id), "trend", "signedADXDI14W")
            trend_changed = (
                old_daily is None
                or old_weekly is None
                or np.sign(daily) != np.sign(old_daily)
                or np.sign(weekly) != np.sign(old_weekly)
                or abs(daily - old_daily) >= config.momentum_delta_threshold
                or abs(weekly - old_weekly) >= config.momentum_delta_threshold
            )
            event_type = None
            if weekly > 0 and daily < 0 and abs(gap) >= config.trend_divergence_threshold:
                event_type = "TACTICAL_CORRECTION"
            elif weekly < 0 and daily > 0 and abs(gap) >= config.trend_divergence_threshold:
                event_type = "EARLY_TREND_RECOVERY"
            elif weekly < 0 and daily < 0:
                event_type = "STRUCTURAL_TREND_DETERIORATION"
            elif abs(gap) >= config.trend_divergence_threshold:
                event_type = "TREND_TIMEFRAME_DIVERGENCE"
            if event_type and trend_changed:
                add(event_type, f"trend:{group_id}", [group_id], {"daily": daily, "weekly": weekly, "gap": _safe_float(gap)}, None, min(100, abs(gap)))
        short, mid = val(item, "momentum", "shortTerm"), val(item, "momentum", "midTerm")
        old = previous.get(group_id)
        old_short = val(old, "momentum", "shortTerm")
        if short is not None and mid is not None:
            spread = short - mid
            delta = short - old_short if old_short is not None else None
            event_type = None
            if delta is not None and abs(delta) >= config.momentum_delta_threshold and np.sign(short - 50) != np.sign(old_short - 50):
                event_type = "MOMENTUM_REVERSAL"
            elif delta is not None and delta >= config.momentum_delta_threshold:
                event_type = "MOMENTUM_ACCELERATION"
            elif delta is not None and delta <= -config.momentum_delta_threshold:
                event_type = "MOMENTUM_DECELERATION"
            elif abs(spread) >= config.momentum_spread_threshold and (
                old is None
                or val(old, "momentum", "spread") is None
                or abs(spread - float(val(old, "momentum", "spread"))) >= config.momentum_delta_threshold
            ):
                event_type = "MOMENTUM_DIVERGENCE"
            if event_type:
                add(event_type, f"momentum:{group_id}", [group_id], {"shortTerm": short, "midTerm": mid, "spread": _safe_float(spread), "weeklyDelta": _safe_float(delta)}, None, min(100, max(abs(spread), abs(delta or 0)) * 4))
        bullish = val(item, "breadth", "bullishWeekly")
        old_bullish = val(old, "breadth", "bullishWeekly")
        if bullish is not None and old_bullish is not None:
            delta = bullish - old_bullish
            if abs(delta) >= config.breadth_delta_threshold_pp:
                event_type = "BULLISH_BREADTH_EXPANSION" if delta > 0 else "BULLISH_BREADTH_CONTRACTION"
                add(event_type, f"breadth:{group_id}", [group_id], {"bullishWeekly": bullish, "deltaPp": _safe_float(delta)}, {"previousBullishWeekly": old_bullish}, min(100, abs(delta) * 3))
        extension = val(item, "breadth", "extensionRisk")
        old_extension = val(old, "breadth", "extensionRisk")
        if extension is not None and old_extension is not None:
            delta = extension - old_extension
            event_type = None
            if extension > 0 and old_extension == 0:
                event_type = "EXTENSION_RISK_EMERGING"
            elif delta >= config.breadth_delta_threshold_pp:
                event_type = "EXTENSION_RISK_BROADENING"
            elif delta <= -config.breadth_delta_threshold_pp:
                event_type = "EXTENSION_RISK_EASING"
            if event_type:
                add(event_type, f"extension:{group_id}", [group_id], {"extensionRisk": extension, "deltaPp": _safe_float(delta)}, {"previousExtensionRisk": old_extension}, min(100, abs(delta) * 3))

    type_groups: dict[str, list[str]] = {}
    for candidate in candidates:
        type_groups.setdefault(candidate["type"], []).extend(candidate["groupIds"])
    for event_type, groups in type_groups.items():
        unique_groups = sorted(set(groups))
        if len(unique_groups) >= 3 and event_type in {
            "BULLISH_BREADTH_EXPANSION", "BULLISH_BREADTH_CONTRACTION", "MOMENTUM_ACCELERATION", "MOMENTUM_DECELERATION"
        }:
            cluster_type = "BROADENING_MARKET_PARTICIPATION" if event_type in {"BULLISH_BREADTH_EXPANSION", "MOMENTUM_ACCELERATION"} else "NARROWING_MARKET_LEADERSHIP"
            add(cluster_type, f"cluster:{cluster_type}", unique_groups, {"sourceEventType": event_type, "groupCount": len(unique_groups)}, None, min(100, 40 + len(unique_groups) * 10))

    relationship_events = {item["themeKey"]: item for item in candidates if item["themeKey"].startswith("relationship:")}
    risk_event = relationship_events.get("relationship:US_RISK_LEADERSHIP")
    defensive = current.get("US Sectors Defensive")
    if risk_event and defensive and risk_event["currentEvidence"].get("m1SpreadPp", 0) < 0 and (val(defensive, "trend", "signedADXDI14W") or 0) > 0:
        add(
            "DEFENSIVE_RELATIVE_STRENGTH",
            "cluster:DEFENSIVE_RELATIVE_STRENGTH",
            ["US Sectors Aggressive", "US Sectors Defensive"],
            {"relationshipEventId": risk_event["eventId"], "defensiveWeeklyTrend": val(defensive, "trend", "signedADXDI14W")},
            None,
            70,
        )
    crypto_event = relationship_events.get("relationship:CRYPTO_LEADERSHIP")
    btc, altcoins = current.get("Crypto BTC"), current.get("Crypto Altcoins")
    if crypto_event and btc and altcoins and (val(btc, "trend", "signedADXDI14D") or 0) < 0 and (val(altcoins, "trend", "signedADXDI14D") or 0) < 0:
        add(
            "SPECULATIVE_LEADERSHIP_CONTRACTION",
            "cluster:SPECULATIVE_LEADERSHIP_CONTRACTION",
            ["Crypto BTC", "Crypto Altcoins"],
            {"relationshipEventId": crypto_event["eventId"], "bothTacticalTrendsNegative": True},
            None,
            70,
        )
    geographic = [
        item for key, item in relationship_events.items()
        if key in {"relationship:US_LATAM_ROTATION", "relationship:US_ASIA_ROTATION", "relationship:US_EU_ROTATION"}
    ]
    if len(geographic) >= 2:
        add(
            "GEOGRAPHIC_ROTATION",
            "cluster:GEOGRAPHIC_ROTATION",
            sorted({group for item in geographic for group in item["groupIds"]}),
            {"relationshipEventIds": [item["eventId"] for item in geographic]},
            None,
            min(100, 50 + len(geographic) * 12),
        )
    duration_credit = relationship_events.get("relationship:DURATION_CREDIT")
    if duration_credit:
        add(
            "DURATION_CREDIT_DIVERGENCE",
            "cluster:DURATION_CREDIT_DIVERGENCE",
            duration_credit["groupIds"],
            {"relationshipEventId": duration_credit["eventId"], **duration_credit["currentEvidence"]},
            None,
            75,
        )
    commodity_relations = [
        item for key, item in relationship_events.items()
        if key in {"relationship:ENERGY_METALS_ROTATION", "relationship:ENERGY_AGRICULTURE_ROTATION"}
    ]
    if len(commodity_relations) >= 2:
        add(
            "COMMODITY_LEADERSHIP_CONCENTRATION",
            "cluster:COMMODITY_LEADERSHIP_CONCENTRATION",
            sorted({group for item in commodity_relations for group in item["groupIds"]}),
            {"relationshipEventIds": [item["eventId"] for item in commodity_relations]},
            None,
            75,
        )

    four_week = {item["groupId"]: item for item in four_week_snapshots}
    for candidate in candidates:
        four_week_deltas: dict[str, Any] = {}
        for group_id in candidate["groupIds"]:
            current_group, old_group = current.get(group_id), four_week.get(group_id)
            if not current_group or not old_group:
                continue
            for label, path in (
                ("m1", ("performance", "m1")),
                ("m3", ("performance", "m3")),
                ("shortMomentum", ("momentum", "shortTerm")),
                ("weeklyTrend", ("trend", "signedADXDI14W")),
                ("bullishWeeklyBreadth", ("breadth", "bullishWeekly")),
            ):
                current_value, old_value = val(current_group, *path), val(old_group, *path)
                if current_value is not None and old_value is not None:
                    four_week_deltas[f"{group_id}:{label}"] = _safe_float(current_value - old_value)
        if four_week_deltas:
            candidate["historicalEvidence"]["fourWeekDeltas"] = four_week_deltas
    deduped = {event["eventId"]: event for event in candidates}
    return sorted(deduped.values(), key=lambda item: (-item["significanceScore"], item["eventId"]))[: config.max_events]


def output_schema() -> dict[str, Any]:
    thesis = {
        "type": "object",
        "additionalProperties": False,
        "required": ["themeKey", "type", "groupIds", "headline", "interpretation", "whyItMatters", "alternativeInterpretation", "supportingEventIds", "contradictingEventIds", "confirmationConditionIds", "invalidationConditionIds", "proposedLifecycle", "evidenceAssessment"],
        "properties": {
            "themeKey": {"type": "string", "maxLength": 140},
            "type": {"type": "string", "enum": sorted(THESIS_TYPES)},
            "groupIds": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
            "headline": {"type": "string", "maxLength": 180},
            "interpretation": {"type": "string", "maxLength": 1200},
            "whyItMatters": {"type": "string", "maxLength": 700},
            "alternativeInterpretation": {"type": ["string", "null"], "maxLength": 700},
            "supportingEventIds": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "contradictingEventIds": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "confirmationConditionIds": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "invalidationConditionIds": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "proposedLifecycle": {"type": "string", "enum": sorted(LIFECYCLES)},
            "evidenceAssessment": {"type": "string", "enum": sorted(EVIDENCE_ASSESSMENTS)},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schemaVersion", "executive", "theses", "weeklyAssessment", "limitations"],
        "properties": {
            "schemaVersion": {"type": "string", "const": SCHEMA_VERSION},
            "executive": {
                "type": "object", "additionalProperties": False,
                "required": ["headline", "interpretation", "thesisKeys"],
                "properties": {
                    "headline": {"type": "string", "maxLength": 180},
                    "interpretation": {"type": "string", "maxLength": 1200},
                    "thesisKeys": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
            },
            "theses": {"type": "array", "items": thesis, "maxItems": 3},
            "weeklyAssessment": {
                "type": "object", "additionalProperties": False,
                "required": ["materialChangeDetected", "interpretation"],
                "properties": {
                    "materialChangeDetected": {"type": "boolean"},
                    "interpretation": {"type": "string", "maxLength": 900},
                },
            },
            "limitations": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 5},
        },
    }


SYSTEM_PROMPT_RU = """Ты — аналитик межрыночной структуры. Все числовые показатели и события уже рассчитаны кодом.
Пиши только по-русски и только о группах Group 2. Не упоминай отдельные ETF или тикеры. Не пересказывай механические
состояния; выявляй взаимосвязи, изменения лидерства, противоречия и ограничения. Не придумывай новости, потоки,
причины, пороги, прогнозы цен или торговые рекомендации. Корреляция не является причинностью. Каждая гипотеза
должна опираться на допустимые eventId и conditionId, учитывать альтернативное объяснение и добавлять смысл сверх
предоставленных статусов. Если значимого изменения нет, верни пустой список theses. Возвращай только JSON по схеме."""


def build_llm_context(
    week_end: date,
    snapshots: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    previous_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    previous_theses = []
    if previous_report:
        previous_theses = [
            {
                "themeKey": thesis.get("themeKey"),
                "type": thesis.get("type"),
                "groupIds": thesis.get("groupIds", []),
                "headline": thesis.get("headline"),
                "lifecycle": thesis.get("lifecycle", thesis.get("proposedLifecycle")),
                "supportingEventIds": thesis.get("supportingEventIds", []),
            }
            for thesis in (previous_report.get("output") or {}).get("theses", [])
        ]
    return {
        "weekEnd": week_end.isoformat(),
        "groups": list(snapshots),
        "significantEvents": list(events),
        "previousTheses": previous_theses,
        "permitted": {
            "groupIds": sorted(item["groupId"] for item in snapshots),
            "eventIds": [item["eventId"] for item in events],
            "conditionIds": sorted({condition for item in events for condition in item["supportingConditionIds"] + item["invalidationConditionIds"]}),
            "themeKeys": sorted({item["themeKey"] for item in events} | {item.get("themeKey") for item in previous_theses if item.get("themeKey")}),
        },
        "conditionRegistry": [
            definition
            for event in events
            for definition in event.get("conditionDefinitions", [])
        ],
    }


def validate_output(
    output: Mapping[str, Any],
    context: Mapping[str, Any],
    ticker_names: Iterable[str] = (),
    previous_report: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    required = {"schemaVersion", "executive", "theses", "weeklyAssessment", "limitations"}
    if not isinstance(output, Mapping) or not required.issubset(output):
        return ["Некорректная структура верхнего уровня."]
    if output.get("schemaVersion") != SCHEMA_VERSION:
        errors.append("Некорректная версия схемы.")
    theses = output.get("theses")
    if not isinstance(theses, list) or len(theses) > 3:
        errors.append("theses должен быть списком максимум из 3 элементов.")
        theses = []
    permitted = context.get("permitted", {})
    groups, events, conditions = map(set, (permitted.get("groupIds", []), permitted.get("eventIds", []), permitted.get("conditionIds", [])))
    themes: set[str] = set()
    normalized_texts: list[str] = []
    forbidden = {
        str(item).upper()
        for item in ticker_names
        if len(str(item)) >= 2 and str(item) not in groups
    }
    causal_patterns = (r"\bиз-за\b", r"\bвызван[аоы]?\b", r"\bпривел[ао]?\s+к\b", r"\bпоток[аиов]*\b")
    for index, thesis in enumerate(theses):
        if not isinstance(thesis, Mapping):
            errors.append(f"theses[{index}] не является объектом.")
            continue
        theme = thesis.get("themeKey")
        if not isinstance(theme, str) or not theme or theme in themes:
            errors.append(f"theses[{index}]: themeKey отсутствует или дублируется.")
        elif theme not in set(permitted.get("themeKeys", [])):
            errors.append(f"theses[{index}]: недопустимый themeKey.")
        themes.add(str(theme))
        if thesis.get("type") not in THESIS_TYPES:
            errors.append(f"theses[{index}]: неизвестный type.")
        if thesis.get("proposedLifecycle") not in LIFECYCLES:
            errors.append(f"theses[{index}]: неизвестный lifecycle.")
        if thesis.get("evidenceAssessment") not in EVIDENCE_ASSESSMENTS:
            errors.append(f"theses[{index}]: неизвестная оценка доказательств.")
        thesis_groups = thesis.get("groupIds", [])
        supporting = thesis.get("supportingEventIds", [])
        contradicting = thesis.get("contradictingEventIds", [])
        confirms = thesis.get("confirmationConditionIds", [])
        invalidates = thesis.get("invalidationConditionIds", [])
        for field_name, references in (
            ("groupIds", thesis_groups),
            ("supportingEventIds", supporting),
            ("contradictingEventIds", contradicting),
            ("confirmationConditionIds", confirms),
            ("invalidationConditionIds", invalidates),
        ):
            if len(references) != len(set(references)):
                errors.append(f"theses[{index}].{field_name}: повторяющиеся ссылки запрещены.")
        if not thesis_groups or not set(thesis_groups).issubset(groups):
            errors.append(f"theses[{index}]: недопустимые Group 2 ссылки.")
        if not supporting or not set(supporting).issubset(events):
            errors.append(f"theses[{index}]: отсутствует допустимое supporting evidence.")
        if not set(contradicting).issubset(events):
            errors.append(f"theses[{index}]: недопустимое contradicting evidence.")
        if not set(confirms).issubset(conditions) or not set(invalidates).issubset(conditions):
            errors.append(f"theses[{index}]: недопустимые conditionId.")
        combined = " ".join(str(thesis.get(key, "")) for key in ("headline", "interpretation", "whyItMatters", "alternativeInterpretation"))
        upper_text = combined.upper()
        mentioned = sorted(symbol for symbol in forbidden if re.search(rf"(?<![A-Z0-9-]){re.escape(symbol)}(?![A-Z0-9-])", upper_text))
        if mentioned:
            errors.append(f"theses[{index}]: упомянуты отдельные инструменты: {', '.join(mentioned[:5])}.")
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in causal_patterns):
            errors.append(f"theses[{index}]: неподтверждённое причинное утверждение или потоки.")
        normalized = re.sub(r"\W+", " ", combined.lower()).strip()
        tokens = set(normalized.split())
        for prior in normalized_texts:
            prior_tokens = set(prior.split())
            union = tokens | prior_tokens
            if union and len(tokens & prior_tokens) / len(union) >= 0.72:
                errors.append(f"theses[{index}]: аналитически дублирует другую гипотезу.")
        normalized_texts.append(normalized)
    executive = output.get("executive", {})
    executive_thesis_keys = executive.get("thesisKeys", [])
    if len(executive_thesis_keys) != len(set(executive_thesis_keys)):
        errors.append("Executive thesisKeys содержит повторяющиеся ссылки.")
    if not set(executive_thesis_keys).issubset(themes):
        errors.append("Executive содержит неизвестный thesisKey.")
    if not context.get("significantEvents") and theses:
        errors.append("Гипотезы запрещены при отсутствии значимых событий.")
    if previous_report:
        previous = {(item.get("themeKey"), re.sub(r"\W+", " ", str(item.get("interpretation", "")).lower()).strip()) for item in (previous_report.get("output") or {}).get("theses", [])}
        for thesis, normalized in zip(theses, normalized_texts):
            if (thesis.get("themeKey"), normalized) in previous:
                errors.append(f"Гипотеза {thesis.get('themeKey')} дословно повторяет прошлую неделю.")
    return errors


def deterministic_continuity_output(week_end: date, readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "executive": {
            "headline": "Существенных новых изменений рыночной структуры не выявлено",
            "interpretation": "Детерминированный анализ завершён без событий достаточной материальности для нового LLM-синтеза. Предыдущие опубликованные гипотезы следует рассматривать как контекст, а не как автоматически подтверждённые выводы.",
            "thesisKeys": [],
        },
        "theses": [],
        "weeklyAssessment": {
            "materialChangeDetected": False,
            "interpretation": f"За неделю, завершившуюся {week_end.isoformat()}, квалифицирующих изменений не обнаружено; LLM-вызов не выполнялся.",
        },
        "limitations": [f"Полнота Group 2: {readiness.get('completeGroups', 0)} из {readiness.get('groupCount', 0)} групп."],
    }


def _usage(response: Any, config: IntelligenceConfig) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    input_rate = float(os.getenv("WMI_INPUT_COST_PER_MILLION", "0"))
    output_rate = float(os.getenv("WMI_OUTPUT_COST_PER_MILLION", "0"))
    cost = None
    if input_tokens is not None and output_tokens is not None:
        cost = round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0, 6)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "estimated_cost": cost}


def call_llm(context: Mapping[str, Any], config: IntelligenceConfig, correction_errors: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, Any], str]:
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    prompt = "Подготовь Weekly Market Intelligence по разрешённым данным:\n" + _json(context)
    if correction_errors:
        prompt += "\nПредыдущий JSON отклонён. Исправь только перечисленные ошибки, не меняя разрешённый набор доказательств:\n- " + "\n- ".join(correction_errors)
    response = OpenAI(api_key=key).responses.create(
        model=config.model,
        reasoning={"effort": config.reasoning_effort},
        instructions=SYSTEM_PROMPT_RU,
        input=prompt,
        max_output_tokens=config.max_output_tokens,
        text={"format": {"type": "json_schema", "name": "weekly_market_intelligence", "strict": True, "schema": output_schema()}},
    )
    status = getattr(response, "status", None)
    if status and status != "completed":
        incomplete = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None) if incomplete else None
        raise RuntimeError(f"OpenAI response did not complete (status={status}, reason={reason or 'unknown'})")
    raw = getattr(response, "output_text", "") or ""
    if not raw.strip():
        raise RuntimeError("OpenAI response contained no structured output")
    return json.loads(raw), _usage(response, config), raw


def finalize_lifecycle(output: dict[str, Any], previous_report: Mapping[str, Any] | None, events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior = {item.get("themeKey"): item for item in (previous_report.get("output") or {}).get("theses", [])} if previous_report else {}
    event_by_id = {event["eventId"]: event for event in events}
    changes: list[dict[str, Any]] = []
    active_keys: set[str] = set()
    for thesis in output.get("theses", []):
        key = thesis["themeKey"]
        active_keys.add(key)
        old = prior.get(key)
        support = [event_by_id[item] for item in thesis.get("supportingEventIds", []) if item in event_by_id]
        weakening = support and all("WEAKENING" in event["type"] or "CONTRACTION" in event["type"] or "EASING" in event["type"] for event in support)
        reversal = any("SHIFT" in event["type"] or "REVERSAL" in event["type"] for event in support)
        if old is None:
            lifecycle = "NEW"
        elif reversal and old.get("proposedLifecycle", old.get("lifecycle")) in {"DEVELOPING", "ESTABLISHED"}:
            lifecycle = "INVALIDATED"
        elif weakening:
            lifecycle = "WEAKENING"
        elif max((event.get("persistenceWeeks", 1) for event in support), default=1) >= 3:
            lifecycle = "ESTABLISHED"
        else:
            lifecycle = "DEVELOPING"
        thesis["lifecycle"] = lifecycle
        changes.append({"themeKey": key, "from": old.get("lifecycle", old.get("proposedLifecycle")) if old else None, "to": lifecycle})
    # Omission is not evidence of resolution. Unchanged prior theses remain stored in
    # their previous published report until a qualifying event explicitly updates them.
    return output, changes


@dataclass
class RunResult:
    status: str
    week_end: str
    report_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def generate_weekly_intelligence(
    *,
    week_end: date,
    group_members: Mapping[str, Sequence[str]],
    history_loader: Callable[[str], pd.DataFrame],
    weekly_builder: Callable[[pd.DataFrame, bool], pd.DataFrame],
    adx_calculator: Callable[[pd.Series, pd.Series, pd.Series, int], tuple[pd.Series, pd.Series, pd.Series]],
    repository: IntelligenceRepository | None = None,
    config: IntelligenceConfig | None = None,
    llm_callable: Callable[[Mapping[str, Any], IntelligenceConfig, Sequence[str] | None], tuple[dict[str, Any], dict[str, Any], str]] = call_llm,
) -> RunResult:
    config = config or IntelligenceConfig.from_env()
    repository = repository or IntelligenceRepository(os.getenv("WMI_DB_PATH", str(DEFAULT_DB_PATH)))
    run_key = f"{week_end.isoformat()}:{ANALYSIS_VERSION}"
    claim_status = repository.claim_run(week_end)
    if claim_status == "ALREADY_PUBLISHED":
        return RunResult("SKIPPED_ALREADY_PUBLISHED", week_end.isoformat())
    if claim_status == "RUN_IN_PROGRESS":
        return RunResult("SKIPPED_RUN_IN_PROGRESS", week_end.isoformat())
    repository.record_attempt(run_key, "STARTED")
    try:
        tickers = sorted({ticker for members in group_members.values() for ticker in members})
        metrics, metric_diagnostics = build_instrument_metrics(
            tickers, week_end, history_loader, weekly_builder, adx_calculator
        )
        snapshots, readiness = aggregate_group_snapshots(metrics, group_members, week_end, config)
        readiness["metricDiagnostics"] = metric_diagnostics
        if not readiness["ready"]:
            repository.record_attempt(run_key, "DATA_NOT_READY", readiness)
            repository.record_failure(week_end, "DATA_NOT_READY", "Group 2 data snapshot is incomplete", readiness)
            return RunResult("DATA_NOT_READY", week_end.isoformat(), details=readiness)
        snapshot_id = _stable_id("snapshot", week_end.isoformat(), ANALYSIS_VERSION, _json(snapshots))
        repository.save_snapshots(snapshot_id, week_end, snapshots)
        previous_week = repository.previous_published(week_end)
        previous_snapshots = repository.load_snapshots(date.fromisoformat(previous_week["week_end"])) if previous_week else []
        four_week_snapshots = repository.load_snapshots_at_or_before(week_end - timedelta(days=28))
        events = detect_events(
            snapshots,
            previous_snapshots,
            week_end,
            config,
            repository,
            four_week_snapshots=four_week_snapshots,
        )
        repository.save_events(week_end, events)
        context = build_llm_context(week_end, snapshots, events, previous_week)
        generated_at = datetime.now(timezone.utc).isoformat()
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
        if not events:
            output = deterministic_continuity_output(week_end, readiness)
            validation_errors: list[str] = []
            model = "DETERMINISTIC_NO_MATERIAL_CHANGE"
        else:
            model = config.model
            output, usage, _ = llm_callable(context, config, None)
            validation_errors = validate_output(output, context, tickers, previous_week)
            if validation_errors:
                repository.record_attempt(run_key, "VALIDATION_RETRY", {"errors": validation_errors})
                output, retry_usage, _ = llm_callable(context, config, validation_errors)
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[key] = (usage.get(key) or 0) + (retry_usage.get(key) or 0)
                if usage.get("estimated_cost") is not None and retry_usage.get("estimated_cost") is not None:
                    usage["estimated_cost"] += retry_usage["estimated_cost"]
                validation_errors = validate_output(output, context, tickers, previous_week)
        if validation_errors:
            repository.record_attempt(run_key, "VALIDATION_FAILED", {"errors": validation_errors})
            repository.record_failure(week_end, "VALIDATION_FAILED", "LLM output validation failed", {"errors": validation_errors})
            return RunResult("VALIDATION_FAILED", week_end.isoformat(), details={"errors": validation_errors})
        output, lifecycle_changes = finalize_lifecycle(output, previous_week, events)
        report_id = repository.publish(
            week_end=week_end,
            snapshot_id=snapshot_id,
            output=output,
            events=events,
            validation={"valid": True, "errors": []},
            usage=usage,
            model=model,
            lifecycle_changes=lifecycle_changes,
            generation_timestamp=generated_at,
        )
        repository.record_attempt(run_key, "PUBLISHED", {"reportId": report_id, "eventCount": len(events)})
        return RunResult("PUBLISHED", week_end.isoformat(), report_id, {"events": len(events), "theses": len(output.get("theses", []))})
    except Exception as exc:
        repository.record_attempt(run_key, "FAILED", {"error": str(exc)[:1000]})
        repository.record_failure(week_end, "FAILED", str(exc)[:2000])
        raise

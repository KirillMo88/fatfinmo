from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import app
from job_locking import advisory_file_lock
import positioning
from liquidity_forecast import ERROR_PATH, SNAPSHOT_PATH, refresh_forecast_snapshot
from rates_financial_conditions import SNAPSHOT_PATH as RATES_FC_SNAPSHOT_PATH, refresh_snapshot as refresh_rates_fc_snapshot
from funding_conditions import WEEKLY_PATH as FUNDING_SNAPSHOT_PATH, refresh_snapshot as refresh_funding_snapshot
from treasury_fiscal_regime import SNAPSHOT_PATH as TREASURY_FISCAL_SNAPSHOT_PATH, refresh_snapshot as refresh_treasury_fiscal_snapshot
from treasury_funding_policy import refresh_snapshot as refresh_treasury_funding_policy_snapshot
from technical_outlook_simple_v3.config import (
    CONFIG_VERSION as TECHNICAL_OUTLOOK_SIMPLE_V3_CONFIG_VERSION,
    CORE_ASSET_KEYS as TECHNICAL_OUTLOOK_SIMPLE_V3_CORE_ASSET_KEYS,
    MODEL_VERSION as TECHNICAL_OUTLOOK_SIMPLE_V3_MODEL_VERSION,
    SR_ENGINE_VERSION as TECHNICAL_OUTLOOK_SIMPLE_V3_SR_ENGINE_VERSION,
)
from technical_outlook_simple_v3.service import refresh_all_core_assets as refresh_technical_outlook_simple_v3_assets
from technical_outlook_simple_v3.storage import (
    MANIFEST_PATH as TECHNICAL_OUTLOOK_SIMPLE_V3_MANIFEST_PATH,
    read_manifest as read_technical_outlook_simple_v3_manifest,
)


JOB_DIR = Path("persistent") / "job_status"
JOB_LOG_PATH = JOB_DIR / "refresh_jobs.jsonl"
DEFAULT_NIGHTLY_UTC = "02:30"
DEFAULT_FUNDING_LATE_RETRY_UTC = "05:00"
DEFAULT_WEEKLY_POSITIONING_UTC = "12:30"
DEFAULT_MARKET_PERFORMANCE_INTERVAL_SECONDS = 600
DEFAULT_SCHEDULER_POLL_SECONDS = 30
DEFAULT_NIGHTLY_TIMEOUT_SECONDS = 7200
DEFAULT_WEEKLY_POSITIONING_TIMEOUT_SECONDS = 3600
DEFAULT_MARKET_PERFORMANCE_TIMEOUT_SECONDS = 480
DEFAULT_REFRESH_TIMEOUT_SECONDS = 1800
DEFAULT_REFRESH_RETRY_SECONDS = 300
MAX_REFRESH_RETRY_SECONDS = 3600
MAX_CONCURRENT_SCHEDULED_JOBS = 2


def job_lock(job_name: str):
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    return advisory_file_lock(JOB_DIR / f"{job_name}.lock")


@dataclass
class ScheduledJob:
    name: str
    command: str
    due_at: datetime
    timeout_seconds: int
    daily_at: dt_time | None = None
    weekly_weekday: int | None = None
    weekly_at: dt_time | None = None
    interval_seconds: int | None = None
    process: subprocess.Popen | None = None
    process_started_at: float | None = None
    next_regular_at: datetime | None = None
    retry_attempts: int = 0


def log_job(job: str, started: float, status: str, rows_updated: int = 0, source_status: Any = None, error: str | None = None) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "Job": job,
        "StartedAt": pd.to_datetime(started, unit="s", utc=True).isoformat(),
        "FinishedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "DurationSeconds": round(time.time() - started, 3),
        "Status": status,
        "RowsUpdated": int(rows_updated),
        "SourceStatus": source_status,
        "Errors": error,
    }
    with JOB_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def log_scheduler(message: str) -> None:
    print(f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}", flush=True)


def default_divergence_config() -> tuple[dict[str, Any], str]:
    cfg = dict(app.DIVERGENCE_DEFAULTS)
    profile_defaults = app.DIVERGENCE_PROFILE_DEFAULTS["Weekly"]
    cfg["profile"] = "Weekly"
    cfg["pivot_window"] = int(profile_defaults["pivot_window"])
    cfg["lookback_bars"] = int(profile_defaults["lookback_bars"])
    signature = f"profile={cfg['profile']}:pv={cfg['pivot_window']}:lb={cfg['lookback_bars']}"
    return cfg, signature


def run_nightly_analytics() -> None:
    started = time.time()
    rows_updated = 0
    try:
        with job_lock("nightly_analytics"):
            universe_map = app.load_universe_map()
            divergence_cfg, divergence_signature = default_divergence_config()
            refresh_nonce = int(time.time())
            for universe_name, universe in universe_map.items():
                universe_signature = f"{universe_name}:{str(universe)}"
                key = app.snapshot_hash(universe_signature, divergence_signature)
                frame, calculated_at = app.compute_slow_metrics_table(
                    universe,
                    universe_signature,
                    divergence_cfg,
                    divergence_signature,
                    refresh_nonce,
                    app.performance_benchmark_for_universe(universe_name),
                )
                meta = app.snapshot_metadata("CURRENT", len(frame), data_as_of=f"{frame['Ticker'].nunique()} tickers")
                meta["RefreshNonce"] = int(refresh_nonce)
                meta["SourceCalculatedAt"] = calculated_at
                app.atomic_write_snapshot("screener_snapshot_latest", key, frame, meta)
                rows_updated += len(frame)
            simple_v3_manifest = refresh_technical_outlook_simple_v3_assets(run_at=datetime.now(timezone.utc))
            rows_updated += sum(1 for item in simple_v3_manifest.get("assets", []) if item.get("status") == "CURRENT")
        log_job("nightly_analytics", started, "CURRENT", rows_updated)
    except FileExistsError:
        log_job("nightly_analytics", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("nightly_analytics", started, "FAILED", rows_updated, error=str(exc))
        raise


def update_market_performance_overlay() -> None:
    started = time.time()
    rows_updated = 0
    try:
        with job_lock("market_performance_10m"):
            universe_map = app.load_universe_map()
            _, divergence_signature = default_divergence_config()
            refresh_bucket = int(time.time() // app.AUTO_REFRESH_SECONDS)
            for universe_name, universe in universe_map.items():
                universe_signature = f"{universe_name}:{str(universe)}"
                key = app.snapshot_hash(universe_signature, divergence_signature)
                slow_df, _ = app.read_snapshot_frame("screener_snapshot_latest", key)
                if slow_df.empty:
                    continue
                overlay, _, _ = app.compute_performance_table(slow_df, key, refresh_bucket)
                rows_updated += len(overlay)
        log_job("market_performance_10m", started, "CURRENT", rows_updated)
    except FileExistsError:
        log_job("market_performance_10m", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("market_performance_10m", started, "FAILED", rows_updated, error=str(exc))
        raise


def run_weekly_positioning() -> None:
    started = time.time()
    rows_updated = 0
    source_status: Any = None
    try:
        data = positioning.update_positioning_data(force=True)
        master = data.get("cftc_master", pd.DataFrame())
        rows_updated = len(master) if hasattr(master, "__len__") else 0
        source_status = data.get("status")
        failed_sources = {
            name: detail
            for name in ("CFTC Commodities", "CFTC Financials")
            if (detail := (source_status or {}).get(name, {})).get("status") != "CURRENT"
        }
        if failed_sources:
            raise RuntimeError(f"CFTC source refresh failed; cached data retained: {failed_sources}")
        log_job("weekly_positioning", started, "CURRENT", rows_updated, source_status=source_status)
    except FileExistsError:
        log_job("weekly_positioning", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("weekly_positioning", started, "FAILED", rows_updated, source_status=source_status, error=str(exc))
        raise


def run_liquidity_forecast() -> None:
    started = time.time()
    try:
        with job_lock("liquidity_forecast"):
            frame, status = refresh_forecast_snapshot(api_key=os.getenv("FRED_API_KEY"))
            ERROR_PATH.unlink(missing_ok=True)
        log_job("liquidity_forecast", started, "CURRENT", len(frame), source_status=status.get("SourceStatus"))
    except FileExistsError:
        log_job("liquidity_forecast", started, "SKIPPED_LOCKED")
    except Exception as exc:
        ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        ERROR_PATH.write_text(json.dumps({"FailedAt": pd.Timestamp.now(tz="UTC").isoformat(), "Reason": type(exc).__name__, "SnapshotRetained": SNAPSHOT_PATH.exists()}), encoding="utf-8")
        log_job("liquidity_forecast", started, "FAILED", error=str(exc))
        raise


def run_rates_financial_conditions(refresh: bool = False) -> None:
    started = time.time()
    try:
        with job_lock("rates_financial_conditions"):
            snapshot = refresh_rates_fc_snapshot(api_key=os.getenv("FRED_API_KEY"), refresh=refresh)
        log_job("rates_financial_conditions", started, "CURRENT", len(snapshot.history), source_status=snapshot.status.get("SourceStatus"))
    except FileExistsError:
        log_job("rates_financial_conditions", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("rates_financial_conditions", started, "FAILED", error=str(exc))
        raise


def run_funding_conditions() -> None:
    started = time.time()
    try:
        with job_lock("funding_conditions"):
            snapshot = refresh_funding_snapshot(api_key=os.getenv("FRED_API_KEY"))
        log_job("funding_conditions", started, "CURRENT", len(snapshot.weekly), source_status=snapshot.status.get("SourceStatus"))
    except FileExistsError:
        log_job("funding_conditions", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("funding_conditions", started, "FAILED", error=str(exc))
        raise


def run_funding_conditions_late_retry() -> None:
    started = time.time()
    try:
        with job_lock("funding_conditions"):
            snapshot = refresh_funding_snapshot(api_key=os.getenv("FRED_API_KEY"), refresh=True)
        log_job("funding_conditions_late_retry", started, "CURRENT", len(snapshot.weekly),
                source_status=snapshot.status.get("SourceStatus"))
    except FileExistsError:
        log_job("funding_conditions_late_retry", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("funding_conditions_late_retry", started, "FAILED", error=str(exc))
        raise


def run_rates_financial_conditions_late_retry() -> None:
    started = time.time()
    try:
        with job_lock("rates_financial_conditions"):
            snapshot = refresh_rates_fc_snapshot(api_key=os.getenv("FRED_API_KEY"), refresh=True)
        log_job("rates_financial_conditions_late_retry", started, "CURRENT", len(snapshot.history),
                source_status=snapshot.status.get("SourceStatus"))
    except FileExistsError:
        log_job("rates_financial_conditions_late_retry", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("rates_financial_conditions_late_retry", started, "FAILED", error=str(exc))
        raise


def run_treasury_fiscal_regime() -> None:
    started = time.time()
    try:
        with job_lock("treasury_fiscal_regime"):
            snapshot = refresh_treasury_fiscal_snapshot(api_key=os.getenv("FRED_API_KEY"))
            funding_policy = refresh_treasury_funding_policy_snapshot(
                api_key=os.getenv("FRED_API_KEY"), treasury_snapshot=snapshot,
            )
        log_job("treasury_fiscal_regime", started, "CURRENT", len(snapshot.weekly),
                source_status={
                    "TreasuryFiscal": snapshot.status.get("SourceStatus"),
                    "TreasuryFundingPolicy": funding_policy.status,
                })
    except FileExistsError:
        log_job("treasury_fiscal_regime", started, "SKIPPED_LOCKED")
    except Exception as exc:
        log_job("treasury_fiscal_regime", started, "FAILED", error=str(exc))
        raise


def parse_utc_hhmm(value: str, fallback: str) -> dt_time:
    raw = (value or fallback).strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        return dt_time(int(hour_text), int(minute_text), tzinfo=timezone.utc)
    except Exception:
        log_scheduler(f"Invalid UTC time {raw!r}; using {fallback}")
        hour_text, minute_text = fallback.split(":", 1)
        return dt_time(int(hour_text), int(minute_text), tzinfo=timezone.utc)


def next_daily_run(now: datetime, run_at: dt_time) -> datetime:
    candidate = datetime.combine(now.date(), run_at, tzinfo=timezone.utc)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def next_weekly_run(now: datetime, weekday: int, run_at: dt_time) -> datetime:
    candidate = datetime.combine(now.date(), run_at, tzinfo=timezone.utc)
    days_ahead = (weekday - now.weekday()) % 7
    candidate += timedelta(days=days_ahead)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def last_successful_job_time(job_name: str) -> datetime | None:
    if not JOB_LOG_PATH.exists():
        return None
    latest: datetime | None = None
    try:
        with JOB_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("Job") != job_name or entry.get("Status") != "CURRENT":
                    continue
                finished = pd.to_datetime(entry.get("FinishedAt"), utc=True, errors="coerce")
                if pd.isna(finished):
                    continue
                value = finished.to_pydatetime()
                if latest is None or value > latest:
                    latest = value
    except OSError:
        return None
    return latest


def initial_daily_due(now: datetime, run_at: dt_time, job_name: str) -> datetime:
    upcoming = next_daily_run(now, run_at)
    latest_due = upcoming - timedelta(days=1)
    successful_at = last_successful_job_time(job_name)
    if latest_due <= now and (successful_at is None or successful_at < latest_due):
        return latest_due
    return upcoming


def initial_weekly_due(now: datetime, weekday: int, run_at: dt_time, job_name: str) -> datetime:
    upcoming = next_weekly_run(now, weekday, run_at)
    latest_due = upcoming - timedelta(days=7)
    successful_at = last_successful_job_time(job_name)
    if latest_due <= now and (successful_at is None or successful_at < latest_due):
        return latest_due
    return upcoming


def _last_job_status_since(job_name: str, started_at: float) -> str | None:
    if not JOB_LOG_PATH.exists():
        return None
    latest_started = float("-inf")
    latest_status: str | None = None
    try:
        with JOB_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("Job") != job_name:
                    continue
                started = pd.to_datetime(entry.get("StartedAt"), utc=True, errors="coerce")
                if pd.isna(started):
                    continue
                started_epoch = started.timestamp()
                if started_epoch >= started_at and started_epoch >= latest_started:
                    latest_started = started_epoch
                    latest_status = str(entry.get("Status", ""))
    except OSError:
        return None
    return latest_status


def _next_regular_run(job: ScheduledJob, now: datetime) -> datetime:
    if job.daily_at is not None:
        return next_daily_run(now, job.daily_at)
    if job.weekly_weekday is not None:
        if job.weekly_at is None:
            raise ValueError(f"Scheduled weekly job has no run time: {job.name}")
        return next_weekly_run(now, job.weekly_weekday, job.weekly_at)
    if job.interval_seconds is not None:
        return now + timedelta(seconds=job.interval_seconds)
    return datetime.max.replace(tzinfo=timezone.utc)


def configured_timeout(env_name: str, default: int) -> int:
    try:
        value = int(os.getenv(env_name, str(default)))
    except (TypeError, ValueError):
        log_scheduler(f"Invalid {env_name}; using {default}s")
        return default
    return value if value > 0 else default


def schedule_retry(job: ScheduledJob, now: datetime, base_seconds: int) -> int:
    job.retry_attempts += 1
    delay_seconds = min(base_seconds * (2 ** min(job.retry_attempts - 1, 10)), MAX_REFRESH_RETRY_SECONDS)
    retry_at = now + timedelta(seconds=delay_seconds)
    if job.next_regular_at is not None and retry_at >= job.next_regular_at:
        job.due_at = job.next_regular_at
    else:
        job.due_at = retry_at
    return delay_seconds


def launch_scheduled_job(job: ScheduledJob, retry_seconds: int = DEFAULT_REFRESH_RETRY_SECONDS) -> None:
    started_at = time.time()
    if job.next_regular_at is not None and job.due_at >= job.next_regular_at:
        job.retry_attempts = 0
    command = [sys.executable, str(Path(__file__).resolve()), job.command]
    try:
        job.process = subprocess.Popen(command, cwd=Path.cwd())
    except Exception as exc:
        log_job(job.name, started_at, "FAILED", error=f"Could not start scheduled process: {exc}")
        delay_seconds = schedule_retry(job, datetime.now(timezone.utc), retry_seconds)
        log_scheduler(f"{job.name} could not start; retrying in {delay_seconds}s: {exc}")
        return
    job.process_started_at = started_at
    now = datetime.now(timezone.utc)
    job.next_regular_at = _next_regular_run(job, now)
    job.due_at = job.next_regular_at
    log_scheduler(
        f"Started {job.name} pid={job.process.pid}, timeout={job.timeout_seconds}s, "
        f"next_due={job.next_regular_at.isoformat()}"
    )


def reap_scheduled_jobs(jobs: list[ScheduledJob], retry_seconds: int = DEFAULT_REFRESH_RETRY_SECONDS) -> None:
    now = datetime.now(timezone.utc)
    for job in jobs:
        process = job.process
        started_at = job.process_started_at
        if process is None or started_at is None:
            continue
        elapsed = time.time() - started_at
        return_code = process.poll()
        timed_out = return_code is None and elapsed >= job.timeout_seconds
        if timed_out:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return_code = process.returncode
            log_job(job.name, started_at, "FAILED", error=f"Scheduled job timed out after {job.timeout_seconds}s")
            log_scheduler(f"{job.name} timed out after {job.timeout_seconds}s; process stopped")
        elif return_code is not None:
            status = _last_job_status_since(job.name, started_at)
            if return_code == 0 and status == "CURRENT":
                job.retry_attempts = 0
                log_scheduler(f"Finished {job.name} successfully")
            else:
                error = f"Scheduled process exited with code {return_code}; job status={status or 'MISSING'}"
                if return_code == 0 and status == "SKIPPED_LOCKED":
                    error = "Scheduled run could not acquire its active job lock"
                log_job(job.name, started_at, "FAILED", error=error)
                log_scheduler(f"{job.name} did not complete successfully: {error}")

        if timed_out or (return_code is not None and (return_code != 0 or _last_job_status_since(job.name, started_at) != "CURRENT")):
            delay_seconds = schedule_retry(job, now, retry_seconds)
            log_scheduler(f"{job.name} retry scheduled in {delay_seconds}s")
        if return_code is not None:
            job.process = None
            job.process_started_at = None


def latest_screener_snapshot_time() -> datetime | None:
    latest: datetime | None = None
    for meta_path in app.SNAPSHOT_DIR.glob("screener_snapshot_latest_*.json"):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            calculated_at = pd.to_datetime(payload.get("CalculatedAt"), utc=True)
            if pd.isna(calculated_at):
                continue
            ts = calculated_at.to_pydatetime()
            if latest is None or ts > latest:
                latest = ts
        except Exception:
            continue
    return latest


def run_scheduler() -> None:
    nightly_at = parse_utc_hhmm(os.getenv("SCREENER_NIGHTLY_UTC", DEFAULT_NIGHTLY_UTC), DEFAULT_NIGHTLY_UTC)
    funding_retry_at = parse_utc_hhmm(
        os.getenv("SCREENER_FUNDING_LATE_RETRY_UTC", DEFAULT_FUNDING_LATE_RETRY_UTC),
        DEFAULT_FUNDING_LATE_RETRY_UTC,
    )
    weekly_at = parse_utc_hhmm(
        os.getenv("SCREENER_WEEKLY_POSITIONING_UTC", DEFAULT_WEEKLY_POSITIONING_UTC),
        DEFAULT_WEEKLY_POSITIONING_UTC,
    )
    overlay_interval = int(os.getenv("SCREENER_MARKET_PERFORMANCE_INTERVAL_SECONDS", str(DEFAULT_MARKET_PERFORMANCE_INTERVAL_SECONDS)))
    poll_seconds = int(os.getenv("SCREENER_SCHEDULER_POLL_SECONDS", str(DEFAULT_SCHEDULER_POLL_SECONDS)))
    weekly_weekday = int(os.getenv("SCREENER_WEEKLY_POSITIONING_WEEKDAY", "5"))

    now = datetime.now(timezone.utc)
    retry_seconds = configured_timeout("SCREENER_REFRESH_RETRY_SECONDS", DEFAULT_REFRESH_RETRY_SECONDS)
    jobs = [
        ScheduledJob("nightly_analytics", "nightly-analytics", initial_daily_due(now, nightly_at, "nightly_analytics"), configured_timeout("SCREENER_NIGHTLY_TIMEOUT_SECONDS", DEFAULT_NIGHTLY_TIMEOUT_SECONDS), daily_at=nightly_at),
        ScheduledJob("liquidity_forecast", "liquidity-forecast", initial_daily_due(now, nightly_at, "liquidity_forecast"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=nightly_at),
        ScheduledJob("rates_financial_conditions", "rates-financial-conditions", initial_daily_due(now, nightly_at, "rates_financial_conditions"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=nightly_at),
        ScheduledJob("funding_conditions", "funding-conditions", initial_daily_due(now, nightly_at, "funding_conditions"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=nightly_at),
        ScheduledJob("treasury_fiscal_regime", "treasury-fiscal-regime", initial_daily_due(now, nightly_at, "treasury_fiscal_regime"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=nightly_at),
        ScheduledJob("rates_financial_conditions_late_retry", "rates-financial-conditions-late-retry", initial_daily_due(now, funding_retry_at, "rates_financial_conditions_late_retry"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=funding_retry_at),
        ScheduledJob("funding_conditions_late_retry", "funding-conditions-late-retry", initial_daily_due(now, funding_retry_at, "funding_conditions_late_retry"), configured_timeout("SCREENER_REFRESH_TIMEOUT_SECONDS", DEFAULT_REFRESH_TIMEOUT_SECONDS), daily_at=funding_retry_at),
        ScheduledJob("weekly_positioning", "weekly-positioning", initial_weekly_due(now, weekly_weekday, weekly_at, "weekly_positioning"), configured_timeout("SCREENER_WEEKLY_POSITIONING_TIMEOUT_SECONDS", DEFAULT_WEEKLY_POSITIONING_TIMEOUT_SECONDS), weekly_weekday=weekly_weekday, weekly_at=weekly_at),
        ScheduledJob("market_performance_10m", "market-performance", now, configured_timeout("SCREENER_MARKET_PERFORMANCE_TIMEOUT_SECONDS", DEFAULT_MARKET_PERFORMANCE_TIMEOUT_SECONDS), interval_seconds=max(60, overlay_interval)),
    ]
    log_scheduler(
        f"Scheduler started; max_parallel={MAX_CONCURRENT_SCHEDULED_JOBS}; "
        + ", ".join(f"{job.name}={job.due_at.isoformat()}" for job in jobs)
    )

    if os.getenv("SCREENER_RUN_NIGHTLY_ON_START_IF_MISSING", "1").strip().lower() in {"1", "true", "yes"}:
        by_name = {job.name: job for job in jobs}
        if latest_screener_snapshot_time() is None:
            by_name["nightly_analytics"].due_at = now
        manifest = read_technical_outlook_simple_v3_manifest()
        if (
            not TECHNICAL_OUTLOOK_SIMPLE_V3_MANIFEST_PATH.exists()
            or manifest.get("model_version") != TECHNICAL_OUTLOOK_SIMPLE_V3_MODEL_VERSION
            or manifest.get("config_version") != TECHNICAL_OUTLOOK_SIMPLE_V3_CONFIG_VERSION
            or manifest.get("sr_engine_version") != TECHNICAL_OUTLOOK_SIMPLE_V3_SR_ENGINE_VERSION
            or {item.get("ticker") for item in manifest.get("assets", [])} != set(TECHNICAL_OUTLOOK_SIMPLE_V3_CORE_ASSET_KEYS)
        ):
            # This refresh is also part of Nightly Analytics; schedule it at the
            # same time instead of blocking the scheduler during startup.
            by_name["nightly_analytics"].due_at = min(by_name["nightly_analytics"].due_at, now)
        if not SNAPSHOT_PATH.exists():
            by_name["liquidity_forecast"].due_at = now
        if not RATES_FC_SNAPSHOT_PATH.exists():
            by_name["rates_financial_conditions"].due_at = now
        if not FUNDING_SNAPSHOT_PATH.exists():
            by_name["funding_conditions"].due_at = now
        if not TREASURY_FISCAL_SNAPSHOT_PATH.exists():
            by_name["treasury_fiscal_regime"].due_at = now

    while True:
        reap_scheduled_jobs(jobs, retry_seconds)
        now_dt = datetime.now(timezone.utc)
        active_count = sum(job.process is not None for job in jobs)
        for job in jobs:
            if job.due_at <= now_dt and job.process is None:
                if active_count >= MAX_CONCURRENT_SCHEDULED_JOBS:
                    break
                launch_scheduled_job(job, retry_seconds)
                if job.process is None and job.due_at <= now_dt:
                    # Failed process creation is already logged and scheduled
                    # for retry by launch_scheduled_job.
                    continue
                if job.process is not None:
                    active_count += 1
        time.sleep(max(5, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Screener refresh jobs")
    parser.add_argument("job", choices=["market-performance", "nightly-analytics", "liquidity-forecast", "rates-financial-conditions", "rates-financial-conditions-late-retry", "funding-conditions", "funding-conditions-late-retry", "treasury-fiscal-regime", "weekly-positioning", "scheduler"])
    args = parser.parse_args()
    if args.job == "market-performance":
        update_market_performance_overlay()
    elif args.job == "nightly-analytics":
        run_nightly_analytics()
    elif args.job == "liquidity-forecast":
        run_liquidity_forecast()
    elif args.job == "rates-financial-conditions":
        run_rates_financial_conditions()
    elif args.job == "rates-financial-conditions-late-retry":
        run_rates_financial_conditions_late_retry()
    elif args.job == "funding-conditions":
        run_funding_conditions()
    elif args.job == "funding-conditions-late-retry":
        run_funding_conditions_late_retry()
    elif args.job == "treasury-fiscal-regime":
        run_treasury_fiscal_regime()
    elif args.job == "weekly-positioning":
        run_weekly_positioning()
    elif args.job == "scheduler":
        run_scheduler()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import app
import positioning
from liquidity_forecast import ERROR_PATH, SNAPSHOT_PATH, refresh_forecast_snapshot
from rates_financial_conditions import SNAPSHOT_PATH as RATES_FC_SNAPSHOT_PATH, refresh_snapshot as refresh_rates_fc_snapshot
from funding_conditions import WEEKLY_PATH as FUNDING_SNAPSHOT_PATH, refresh_snapshot as refresh_funding_snapshot
from treasury_fiscal_regime import SNAPSHOT_PATH as TREASURY_FISCAL_SNAPSHOT_PATH, refresh_snapshot as refresh_treasury_fiscal_snapshot
from treasury_funding_policy import refresh_snapshot as refresh_treasury_funding_policy_snapshot


JOB_DIR = Path("persistent") / "job_status"
JOB_LOG_PATH = JOB_DIR / "refresh_jobs.jsonl"
DEFAULT_NIGHTLY_UTC = "02:30"
DEFAULT_FUNDING_LATE_RETRY_UTC = "05:00"
DEFAULT_WEEKLY_POSITIONING_UTC = "12:30"
DEFAULT_MARKET_PERFORMANCE_INTERVAL_SECONDS = 600
DEFAULT_SCHEDULER_POLL_SECONDS = 30


@contextmanager
def job_lock(job_name: str):
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = JOB_DIR / f"{job_name}.lock"
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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
                )
                meta = app.snapshot_metadata("CURRENT", len(frame), data_as_of=f"{frame['Ticker'].nunique()} tickers")
                meta["RefreshNonce"] = int(refresh_nonce)
                meta["SourceCalculatedAt"] = calculated_at
                app.atomic_write_snapshot("screener_snapshot_latest", key, frame, meta)
                rows_updated += len(frame)
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
        with job_lock("weekly_positioning"):
            data = positioning.update_positioning_data(force=True)
            master = data.get("cftc_master", pd.DataFrame())
            rows_updated = len(master) if hasattr(master, "__len__") else 0
            source_status = data.get("status")
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


def run_job_safely(name: str, func) -> None:
    try:
        log_scheduler(f"Starting {name}")
        func()
        log_scheduler(f"Finished {name}")
    except Exception as exc:
        log_scheduler(f"{name} failed: {exc}")


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
    next_nightly = next_daily_run(now, nightly_at)
    next_funding_retry = next_daily_run(now, funding_retry_at)
    next_weekly = next_weekly_run(now, weekly_weekday, weekly_at)
    next_overlay = time.time()

    log_scheduler(
        f"Scheduler started; nightly={next_nightly.isoformat()}, "
        f"funding_late_retry={next_funding_retry.isoformat()}, "
        f"weekly_positioning={next_weekly.isoformat()}, overlay_interval={overlay_interval}s"
    )

    if os.getenv("SCREENER_RUN_NIGHTLY_ON_START_IF_MISSING", "1").strip().lower() in {"1", "true", "yes"}:
        if latest_screener_snapshot_time() is None:
            run_job_safely("nightly_analytics_startup", run_nightly_analytics)
        if not SNAPSHOT_PATH.exists():
            run_job_safely("liquidity_forecast_startup", run_liquidity_forecast)
        if not RATES_FC_SNAPSHOT_PATH.exists():
            run_job_safely("rates_financial_conditions_startup", run_rates_financial_conditions)
        if not FUNDING_SNAPSHOT_PATH.exists():
            run_job_safely("funding_conditions_startup", run_funding_conditions)
        if not TREASURY_FISCAL_SNAPSHOT_PATH.exists():
            run_job_safely("treasury_fiscal_regime_startup", run_treasury_fiscal_regime)

    while True:
        now_dt = datetime.now(timezone.utc)
        now_seconds = time.time()
        if now_dt >= next_nightly:
            run_job_safely("nightly_analytics", run_nightly_analytics)
            run_job_safely("liquidity_forecast", run_liquidity_forecast)
            run_job_safely("rates_financial_conditions", run_rates_financial_conditions)
            run_job_safely("funding_conditions", run_funding_conditions)
            run_job_safely("treasury_fiscal_regime", run_treasury_fiscal_regime)
            next_nightly = next_daily_run(datetime.now(timezone.utc), nightly_at)
            log_scheduler(f"Next nightly_analytics={next_nightly.isoformat()}")
        if now_dt >= next_weekly:
            run_job_safely("weekly_positioning", run_weekly_positioning)
            next_weekly = next_weekly_run(datetime.now(timezone.utc), weekly_weekday, weekly_at)
            log_scheduler(f"Next weekly_positioning={next_weekly.isoformat()}")
        if now_dt >= next_funding_retry:
            run_job_safely("rates_financial_conditions_late_retry", run_rates_financial_conditions_late_retry)
            run_job_safely("funding_conditions_late_retry", run_funding_conditions_late_retry)
            next_funding_retry = next_daily_run(datetime.now(timezone.utc), funding_retry_at)
            log_scheduler(f"Next funding_conditions_late_retry={next_funding_retry.isoformat()}")
        if now_seconds >= next_overlay:
            run_job_safely("market_performance_10m", update_market_performance_overlay)
            next_overlay = time.time() + max(60, overlay_interval)
        time.sleep(max(5, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Screener refresh jobs")
    parser.add_argument("job", choices=["market-performance", "nightly-analytics", "liquidity-forecast", "rates-financial-conditions", "funding-conditions", "treasury-fiscal-regime", "weekly-positioning", "scheduler"])
    args = parser.parse_args()
    if args.job == "market-performance":
        update_market_performance_overlay()
    elif args.job == "nightly-analytics":
        run_nightly_analytics()
    elif args.job == "liquidity-forecast":
        run_liquidity_forecast()
    elif args.job == "rates-financial-conditions":
        run_rates_financial_conditions()
    elif args.job == "funding-conditions":
        run_funding_conditions()
    elif args.job == "treasury-fiscal-regime":
        run_treasury_fiscal_regime()
    elif args.job == "weekly-positioning":
        run_weekly_positioning()
    elif args.job == "scheduler":
        run_scheduler()


if __name__ == "__main__":
    main()

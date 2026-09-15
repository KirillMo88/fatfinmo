from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

import app
import positioning


JOB_DIR = Path("persistent") / "job_status"
JOB_LOG_PATH = JOB_DIR / "refresh_jobs.jsonl"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Screener refresh jobs")
    parser.add_argument("job", choices=["market-performance", "nightly-analytics", "weekly-positioning"])
    args = parser.parse_args()
    if args.job == "market-performance":
        update_market_performance_overlay()
    elif args.job == "nightly-analytics":
        run_nightly_analytics()
    elif args.job == "weekly-positioning":
        run_weekly_positioning()


if __name__ == "__main__":
    main()

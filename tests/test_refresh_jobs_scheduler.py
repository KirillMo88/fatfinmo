import json
from datetime import datetime, time, timezone

import refresh_jobs


def test_initial_daily_due_catches_up_once_and_respects_success(tmp_path, monkeypatch):
    log_path = tmp_path / "refresh_jobs.jsonl"
    monkeypatch.setattr(refresh_jobs, "JOB_LOG_PATH", log_path)
    now = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
    run_at = time(2, 30, tzinfo=timezone.utc)

    assert refresh_jobs.initial_daily_due(now, run_at, "nightly_analytics") == datetime(
        2026, 10, 6, 2, 30, tzinfo=timezone.utc
    )

    log_path.write_text(json.dumps({
        "Job": "nightly_analytics",
        "Status": "CURRENT",
        "FinishedAt": "2026-10-06T02:45:00+00:00",
    }) + "\n", encoding="utf-8")
    assert refresh_jobs.initial_daily_due(now, run_at, "nightly_analytics") == datetime(
        2026, 10, 7, 2, 30, tzinfo=timezone.utc
    )


def test_initial_weekly_due_catches_up_after_missed_saturday(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_jobs, "JOB_LOG_PATH", tmp_path / "refresh_jobs.jsonl")
    now = datetime(2026, 10, 6, 9, 0, tzinfo=timezone.utc)
    run_at = time(12, 30, tzinfo=timezone.utc)

    assert refresh_jobs.initial_weekly_due(now, 5, run_at, "weekly_positioning") == datetime(
        2026, 10, 3, 12, 30, tzinfo=timezone.utc
    )

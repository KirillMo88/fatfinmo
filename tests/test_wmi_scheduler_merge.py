from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import pytest

import refresh_jobs


def test_stale_lock_file_does_not_block_new_job(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_jobs, "JOB_DIR", tmp_path)
    (tmp_path / "nightly_analytics.lock").write_text("1", encoding="utf-8")

    with refresh_jobs.job_lock("nightly_analytics"):
        pass
    with refresh_jobs.job_lock("nightly_analytics"):
        pass


@pytest.mark.parametrize(
    ("result_status", "expected_log_status"),
    [
        ("PUBLISHED", "CURRENT"),
        ("SKIPPED_ALREADY_PUBLISHED", "CURRENT"),
        ("DATA_NOT_READY", "DATA_NOT_READY"),
    ],
)
def test_wmi_result_status_controls_scheduler_retry(
    tmp_path, monkeypatch, result_status, expected_log_status
):
    monkeypatch.setattr(refresh_jobs, "job_lock", lambda _: nullcontext())
    monkeypatch.setattr(
        refresh_jobs,
        "IntelligenceConfig",
        SimpleNamespace(from_env=lambda: object()),
    )
    monkeypatch.setattr(
        refresh_jobs,
        "latest_eligible_week_end",
        lambda *_: date(2026, 10, 16),
    )
    monkeypatch.setattr(refresh_jobs, "IntelligenceRepository", lambda _: object())
    monkeypatch.setattr(
        refresh_jobs,
        "generate_weekly_intelligence",
        lambda **_: SimpleNamespace(
            status=result_status, week_end="2026-10-16", details={"theses": 2}
        ),
    )
    monkeypatch.setenv("WMI_DB_PATH", str(tmp_path / "wmi.sqlite3"))
    logged = []
    monkeypatch.setattr(
        refresh_jobs,
        "log_job",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    refresh_jobs.run_weekly_market_intelligence()

    assert logged[0][0][0] == "weekly_market_intelligence"
    assert logged[0][0][2] == expected_log_status
    assert logged[0][1]["source_status"]["Result"] == result_status


def test_scheduler_registers_existing_and_wmi_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENER_RUN_NIGHTLY_ON_START_IF_MISSING", "0")
    monkeypatch.setattr(refresh_jobs, "JOB_LOG_PATH", tmp_path / "refresh_jobs.jsonl")
    messages = []
    monkeypatch.setattr(refresh_jobs, "log_scheduler", messages.append)
    monkeypatch.setattr(refresh_jobs, "launch_scheduled_job", lambda *_: None)
    monkeypatch.setattr(refresh_jobs, "reap_scheduled_jobs", lambda *_: None)
    monkeypatch.setattr(
        refresh_jobs.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(StopIteration),
    )

    with pytest.raises(StopIteration):
        refresh_jobs.run_scheduler()

    startup = messages[0]
    assert "nightly_analytics=" in startup
    assert "ramp_ai_index=" in startup
    assert "weekly_market_intelligence=" in startup

from __future__ import annotations

import pytest
from fastapi import HTTPException

import interviews_worker
from interviews_core import InterviewSource, connect_db, init_db, upsert_source


def test_api_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVIEWS_API_TOKEN", "secret")
    interviews_worker.authorize("Bearer secret")
    with pytest.raises(HTTPException) as exc:
        interviews_worker.authorize("Bearer wrong")
    assert exc.value.status_code == 401


def test_check_is_skipped_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert interviews_worker.run_check() == 0


def source(slug: str) -> InterviewSource:
    return InterviewSource(
        slug=slug,
        url=f"https://www.macrovoices.com/podcast-transcripts/{slug}",
        title="Larry McDonald: Market Test",
        published_at="2026-06-12T10:00:00-04:00",
        speakers=("Larry McDonald",),
    )


def test_existing_database_only_processes_unknown_sources(tmp_path) -> None:
    connection = connect_db(tmp_path / "interviews.db")
    init_db(connection)
    upsert_source(connection, source("old-pending"))

    result = interviews_worker.new_sources_only(
        connection,
        [source("old-pending"), source("new-interview")],
    )

    assert [item.slug for item in result] == ["new-interview"]
    connection.close()


def test_empty_database_allows_initial_backfill(tmp_path) -> None:
    connection = connect_db(tmp_path / "interviews.db")
    init_db(connection)

    result = interviews_worker.new_sources_only(
        connection,
        [source("first"), source("second")],
    )

    assert [item.slug for item in result] == ["first", "second"]
    connection.close()

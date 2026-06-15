from __future__ import annotations

import pytest
from fastapi import HTTPException

import interviews_worker


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

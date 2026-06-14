from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest
from fastapi import HTTPException

import finance_api


def test_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_API_TOKEN", "secret")
    finance_api.authorize("Bearer secret")
    with pytest.raises(HTTPException) as exc:
        finance_api.authorize("Bearer wrong")
    assert exc.value.status_code == 401


def test_missing_cache_returns_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(finance_api, "CACHE_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        finance_api.daily_report()
    assert exc.value.status_code == 503


def test_fresh_cache_is_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(finance_api, "CACHE_DIR", tmp_path)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "value": 1}
    (tmp_path / "daily.json").write_text(json.dumps(payload), encoding="utf-8")
    assert finance_api.daily_report()["value"] == 1


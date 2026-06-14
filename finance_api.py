from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException

from finance_core import build_daily_report, build_weekly_report, write_json_atomic


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("FINANCE_CACHE_DIR", ROOT / "persistent/finance_cache"))
UNIVERSE_PATH = Path(
    os.getenv("FINANCE_UNIVERSE_PATH", ROOT / "custom_universe_lists.json")
)
MOSCOW = ZoneInfo("Europe/Moscow")


def _cache_path(kind: str) -> Path:
    return CACHE_DIR / f"{kind}.json"


def _load(kind: str, max_age: timedelta) -> dict:
    path = _cache_path(kind)
    if not path.exists():
        raise HTTPException(503, f"{kind} report is not ready")
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated = datetime.fromisoformat(payload["generated_at"])
    if datetime.now(timezone.utc) - generated > max_age:
        raise HTTPException(503, f"{kind} report is stale")
    return payload


async def _refresh(kind: str) -> None:
    builder = (
        (lambda: build_daily_report(UNIVERSE_PATH))
        if kind == "daily"
        else build_weekly_report
    )
    try:
        payload = await asyncio.to_thread(builder)
        write_json_atomic(_cache_path(kind), payload)
        logger.info("Refreshed %s finance report", kind)
    except Exception:
        logger.exception("Failed to refresh %s finance report", kind)


async def _refresh_loop() -> None:
    daily_marker = None
    weekly_marker = None
    while True:
        now = datetime.now(MOSCOW)
        if now.weekday() < 5 and (now.hour, now.minute) >= (8, 15):
            if daily_marker != now.date():
                await _refresh("daily")
                daily_marker = now.date()
        if now.weekday() == 5 and (now.hour, now.minute) >= (10, 0):
            if weekly_marker != now.date():
                await _refresh("weekly")
                weekly_marker = now.date()
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    initial = [
        kind
        for kind in ("daily", "weekly")
        if not _cache_path(kind).exists()
    ]
    for kind in initial:
        asyncio.create_task(_refresh(kind))
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


app = FastAPI(title="FatFinMo Finance Reports", lifespan=lifespan)


def authorize(authorization: str = Header(default="")) -> None:
    expected = os.getenv("FINANCE_API_TOKEN", "")
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(401, "Unauthorized")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/reports/daily", dependencies=[Depends(authorize)])
def daily_report() -> dict:
    return _load("daily", timedelta(hours=96))


@app.get("/v1/reports/weekly", dependencies=[Depends(authorize)])
def weekly_report() -> dict:
    return _load("weekly", timedelta(days=8))


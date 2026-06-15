from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import secrets
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
import httpx

from interviews_core import (
    connect_db,
    discover_sources,
    init_db,
    list_ready,
    process_source,
)


logger = logging.getLogger(__name__)
DB_PATH = Path(
    os.getenv("INTERVIEWS_DB_PATH", "persistent/interviews/interviews.db")
)
TIMEZONE = ZoneInfo(os.getenv("INTERVIEWS_TIMEZONE", "Europe/Moscow"))
WEEKDAY = int(os.getenv("INTERVIEWS_WEEKDAY", "4"))
HOUR = int(os.getenv("INTERVIEWS_HOUR", "8"))
INITIAL_LIMIT = int(os.getenv("INTERVIEWS_INITIAL_LIMIT", "10"))


def run_check() -> int:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        logger.warning("OPENAI_API_KEY is not configured; interview check skipped")
        return 0
    connection = connect_db(DB_PATH)
    init_db(connection)
    try:
        with httpx.Client() as client:
            existing_count = connection.execute(
                "SELECT COUNT(*) FROM interviews"
            ).fetchone()[0]
            sources = discover_sources(
                client, limit=INITIAL_LIMIT if existing_count == 0 else 50
            )
            processed = 0
            for source in reversed(sources):
                try:
                    processed += int(process_source(connection, client, source))
                except Exception:
                    logger.exception("Failed to process interview %s", source.slug)
            return processed
    finally:
        connection.close()


async def scheduler_loop() -> None:
    marker = None
    while True:
        now = datetime.now(TIMEZONE)
        due = now.weekday() == WEEKDAY and now.hour >= HOUR
        if due and marker != now.date():
            await asyncio.to_thread(run_check)
            marker = now.date()
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    connection = connect_db(DB_PATH)
    init_db(connection)
    count = connection.execute("SELECT COUNT(*) FROM interviews").fetchone()[0]
    connection.close()
    if count == 0:
        asyncio.create_task(asyncio.to_thread(run_check))
    task = asyncio.create_task(scheduler_loop())
    yield
    task.cancel()


app = FastAPI(title="FatFinMo Interviews", lifespan=lifespan)


def authorize(authorization: str = Header(default="")) -> None:
    expected = os.getenv("INTERVIEWS_API_TOKEN", "")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not expected or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Unauthorized")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


@app.get("/v1/interviews", dependencies=[Depends(authorize)])
def interviews(limit: int = 50) -> dict:
    connection = connect_db(DB_PATH)
    try:
        return {"interviews": list_ready(connection, min(max(limit, 1), 200))}
    finally:
        connection.close()

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import httpx
from openai import OpenAI


LIST_URL = "https://www.macrovoices.com/podcast-transcripts?start=0"
BASE_URL = "https://www.macrovoices.com"
DISCLAIMER = (
    "Материал носит информационный характер и не является индивидуальной "
    "инвестиционной рекомендацией."
)


@dataclass(frozen=True, slots=True)
class InterviewSource:
    slug: str
    url: str
    title: str
    published_at: str
    speakers: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slug_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


def extract_speakers(title: str) -> tuple[str, ...]:
    prefix = normalize_space(title.split(":", 1)[0])
    prefix = re.sub(r"^(dr|prof)\.?\s+", "", prefix, flags=re.IGNORECASE)
    names = [
        normalize_space(name)
        for name in re.split(r"\s+(?:and|&)\s+|,\s*(?=[A-Z])", prefix)
        if normalize_space(name)
    ]
    return tuple(dict.fromkeys(names or [prefix]))


def parse_listing(html: str, limit: int | None = None) -> list[InterviewSource]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[InterviewSource] = []
    seen: set[str] = set()
    for article in soup.select('[itemtype*="BlogPosting"]'):
        link = article.select_one('h2 a[itemprop="url"], h1 a[itemprop="url"]')
        time_node = article.select_one('time[itemprop="dateCreated"]')
        if link is None or time_node is None:
            continue
        url = urljoin(BASE_URL, str(link.get("href", "")))
        slug = slug_from_url(url)
        if not re.match(r"^\d+-", slug) or slug in seen:
            continue
        title = normalize_space(link.get_text(" ", strip=True))
        published = str(time_node.get("datetime", "")).strip()
        if not published:
            continue
        results.append(
            InterviewSource(
                slug=slug,
                url=url,
                title=title,
                published_at=published,
                speakers=extract_speakers(title),
            )
        )
        seen.add(slug)
        if limit is not None and len(results) >= limit:
            break
    return results


def parse_transcript(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one('[itemprop="articleBody"]')
    if body is None:
        raise ValueError("MacroVoices articleBody was not found")
    text = "\n".join(
        normalize_space(node.get_text(" ", strip=True))
        for node in body.select("p, li")
        if normalize_space(node.get_text(" ", strip=True))
    )
    if len(text) < 1000:
        raise ValueError("MacroVoices transcript is unexpectedly short")
    return text


def transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text(text: str, max_chars: int = 45000) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        if len(paragraph) > max_chars:
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars])
            continue
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _response_text(client: OpenAI, model: str, instructions: str, text: str) -> str:
    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions=instructions,
        input=text,
    )
    output = response.output_text.strip()
    if not output:
        raise RuntimeError("OpenAI returned an empty response")
    return output


def summarize_interview(
    source: InterviewSource,
    transcript: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    client = client or OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
    chunks = split_text(transcript)
    part_instruction = (
        "Ты финансовый аналитик. Проанализируй часть транскрипта MacroVoices "
        "на русском языке. Зафиксируй только утверждения участников: тезисы, "
        "аргументы, факты и цифры, прогнозы, риски, противоречия, упомянутые "
        "рынки, активы и сектора. Не придумывай отсутствующие сведения."
    )
    notes = [
        _response_text(
            client,
            model,
            part_instruction,
            f"Интервью: {source.title}\nЧасть {index}/{len(chunks)}:\n{chunk}",
        )
        for index, chunk in enumerate(chunks, 1)
    ]
    final_instruction = (
        "Создай детальное структурированное саммари интервью на русском языке "
        "для частного инвестора. Используй Markdown и разделы: Краткий вывод; "
        "Основные тезисы; Аргументы, факты и цифры; Рыночные последствия; "
        "Активы и сектора; Прогнозы и сценарии; Риски и противоречия; "
        "Практические выводы. Четко отделяй слова гостя от аналитического "
        "вывода, не добавляй факты вне заметок. Заверши заданным дисклеймером."
    )
    summary = _response_text(
        client,
        model,
        final_instruction,
        (
            f"Название: {source.title}\n"
            f"Спикеры: {', '.join(source.speakers)}\n"
            f"Дата: {source.published_at}\n\n"
            + "\n\n".join(
                f"Заметки части {index}:\n{note}"
                for index, note in enumerate(notes, 1)
            )
            + f"\n\nОбязательный дисклеймер: {DISCLAIMER}"
        ),
    )
    if DISCLAIMER not in summary:
        summary = f"{summary}\n\n{DISCLAIMER}"
    return summary


def _interview_reference(item: dict) -> str:
    speakers = ", ".join(item.get("speakers", []))
    date_text = str(item.get("published_at", ""))[:10]
    return (
        f"Title: {item.get('title', '')}\n"
        f"Date: {date_text}\n"
        f"Experts: {speakers}\n"
        f"URL: {item.get('url', '')}\n"
        f"Summary:\n{item.get('summary', '')}"
    )


def _batch_interviews(items: list[dict], max_chars: int = 50000) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for item in items:
        reference = _interview_reference(item)
        if current and size + len(reference) > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += len(reference)
    if current:
        batches.append(current)
    return batches


def analyze_topic_from_summaries(
    items: list[dict],
    topic: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    topic = normalize_space(topic)
    if not topic:
        raise ValueError("Topic is required")
    if not items:
        return "В архиве пока нет готовых саммари для анализа."
    client = client or OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-5.5-2026-04-23")
    batches = _batch_interviews(items)
    extraction_instruction = (
        "Ты анализируешь архив русских саммари интервью MacroVoices. "
        "Найди все мнения, аргументы и прогнозы экспертов по заданной теме. "
        "Обязательно сохраняй имя эксперта, дату интервью и название интервью. "
        "Если в конкретном интервью нет содержательного мнения по теме, так и напиши. "
        "Не добавляй факты, которых нет в саммари."
    )
    extracts = []
    for index, batch in enumerate(batches, 1):
        payload = "\n\n---\n\n".join(_interview_reference(item) for item in batch)
        extracts.append(
            _response_text(
                client,
                model,
                extraction_instruction,
                (
                    f"Тема: {topic}\n"
                    f"Пакет интервью {index} из {len(batches)}\n\n"
                    f"{payload}"
                ),
            )
        )

    synthesis_instruction = (
        "Сделай итоговый анализ на русском языке по теме пользователя на основе "
        "извлечений из архива интервью. Используй Markdown. Структура: "
        "Краткий вывод; Мнения экспертов; Согласие и расхождения; Как менялось "
        "мнение во времени; Рыночные последствия; Что отслеживать дальше. "
        "В разделе 'Мнения экспертов' указывай эксперта, дату интервью и название. "
        "Если данных мало, явно скажи, что вывод ограничен архивом саммари. "
        "Не придумывай мнения и не цитируй исходные транскрипты."
    )
    return _response_text(
        client,
        model,
        synthesis_instruction,
        f"Тема: {topic}\n\nИзвлечения:\n\n" + "\n\n---\n\n".join(extracts),
    )


def connect_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS interviews (
            slug TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            published_at TEXT NOT NULL,
            speakers_json TEXT NOT NULL,
            transcript_hash TEXT,
            summary TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            discovered_at TEXT NOT NULL,
            processed_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_interviews_published
            ON interviews(published_at DESC);
        """
    )
    connection.commit()


def upsert_source(
    connection: sqlite3.Connection, source: InterviewSource
) -> bool:
    existing = connection.execute(
        "SELECT slug FROM interviews WHERE slug = ?", (source.slug,)
    ).fetchone()
    now = utc_now()
    connection.execute(
        """
        INSERT INTO interviews (
            slug, url, title, published_at, speakers_json,
            status, discovered_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            url=excluded.url,
            title=excluded.title,
            published_at=excluded.published_at,
            speakers_json=excluded.speakers_json,
            updated_at=excluded.updated_at
        """,
        (
            source.slug,
            source.url,
            source.title,
            source.published_at,
            json.dumps(source.speakers, ensure_ascii=False),
            now,
            now,
        ),
    )
    connection.commit()
    return existing is None


def get_record(connection: sqlite3.Connection, slug: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM interviews WHERE slug = ?", (slug,)
    ).fetchone()


def save_summary(
    connection: sqlite3.Connection,
    source: InterviewSource,
    content_hash: str,
    summary: str,
) -> None:
    now = utc_now()
    connection.execute(
        """
        UPDATE interviews SET transcript_hash=?, summary=?, status='ready',
            error=NULL, processed_at=?, updated_at=?
        WHERE slug=?
        """,
        (content_hash, summary, now, now, source.slug),
    )
    connection.commit()


def save_error(
    connection: sqlite3.Connection, slug: str, message: str
) -> None:
    connection.execute(
        "UPDATE interviews SET status='error', error=?, updated_at=? WHERE slug=?",
        (message[:2000], utc_now(), slug),
    )
    connection.commit()


def row_to_dict(row: sqlite3.Row) -> dict:
    payload = dict(row)
    payload["speakers"] = json.loads(payload.pop("speakers_json"))
    return payload


def list_ready(
    connection: sqlite3.Connection, limit: int = 200
) -> list[dict]:
    rows = connection.execute(
        """
        SELECT * FROM interviews WHERE status='ready'
        ORDER BY published_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_text(client: httpx.Client, url: str) -> str:
    response = client.get(
        url,
        headers={"User-Agent": "FatFinMo Interviews/1.0"},
        follow_redirects=True,
        timeout=45,
    )
    response.raise_for_status()
    return response.text


def discover_sources(client: httpx.Client, limit: int = 50) -> list[InterviewSource]:
    sources: list[InterviewSource] = []
    seen: set[str] = set()
    offset = 0
    while len(sources) < limit:
        page_url = f"{LIST_URL.rsplit('=', 1)[0]}={offset}"
        page_sources = parse_listing(fetch_text(client, page_url))
        added = 0
        for source in page_sources:
            if source.slug in seen:
                continue
            seen.add(source.slug)
            sources.append(source)
            added += 1
            if len(sources) >= limit:
                break
        if added == 0 or len(page_sources) == 0:
            break
        offset += len(page_sources)
    return sources


def process_source(
    connection: sqlite3.Connection,
    http_client: httpx.Client,
    source: InterviewSource,
    openai_client: OpenAI | None = None,
    model: str | None = None,
) -> bool:
    upsert_source(connection, source)
    try:
        transcript = parse_transcript(fetch_text(http_client, source.url))
        content_hash = transcript_hash(transcript)
        record = get_record(connection, source.slug)
        if (
            record is not None
            and record["status"] == "ready"
            and record["transcript_hash"] == content_hash
        ):
            return False
        summary = summarize_interview(
            source, transcript, client=openai_client, model=model
        )
        save_summary(connection, source, content_hash, summary)
        return True
    except Exception as exc:
        save_error(connection, source.slug, str(exc))
        raise


def source_payload(source: InterviewSource) -> dict:
    return asdict(source)

from __future__ import annotations

from pathlib import Path

from interviews_core import (
    DISCLAIMER,
    InterviewSource,
    connect_db,
    extract_speakers,
    init_db,
    list_ready,
    parse_listing,
    parse_transcript,
    save_summary,
    split_text,
    transcript_hash,
    upsert_source,
)


LISTING = """
<div itemprop="blogPost" itemtype="http://schema.org/BlogPosting">
  <h2><a itemprop="url" href="/podcast-transcripts/123-larry-mcdonald-test">
    Larry McDonald: Market Test
  </a></h2>
  <time itemprop="dateCreated" datetime="2026-06-12T10:00:00-04:00"></time>
</div>
"""


def source() -> InterviewSource:
    return InterviewSource(
        slug="123-larry-mcdonald-test",
        url="https://www.macrovoices.com/podcast-transcripts/123-larry-mcdonald-test",
        title="Larry McDonald: Market Test",
        published_at="2026-06-12T10:00:00-04:00",
        speakers=("Larry McDonald",),
    )


def test_listing_and_speaker_parsing() -> None:
    items = parse_listing(LISTING)
    assert items == [source()]
    assert extract_speakers("Dr. Pippa Malmgren: Test") == ("Pippa Malmgren",)
    assert extract_speakers("Alice Smith & Bob Jones: Test") == (
        "Alice Smith",
        "Bob Jones",
    )


def test_transcript_is_extracted_only_from_article_body() -> None:
    body = " ".join(["Important market statement."] * 60)
    html = (
        "<p>navigation noise</p>"
        f'<div itemprop="articleBody"><p>{body}</p></div>'
    )
    assert parse_transcript(html).startswith("Important")
    assert "navigation noise" not in parse_transcript(html)


def test_split_text_preserves_content() -> None:
    text = "\n".join(f"paragraph {index}" for index in range(100))
    chunks = split_text(text, max_chars=100)
    assert len(chunks) > 1
    assert "\n".join(chunks) == text


def test_database_archive_and_deduplication(tmp_path: Path) -> None:
    connection = connect_db(tmp_path / "interviews.db")
    init_db(connection)
    assert upsert_source(connection, source()) is True
    assert upsert_source(connection, source()) is False
    save_summary(
        connection,
        source(),
        transcript_hash("transcript"),
        f"Summary\n\n{DISCLAIMER}",
    )
    rows = list_ready(connection)
    assert len(rows) == 1
    assert rows[0]["speakers"] == ["Larry McDonald"]
    assert rows[0]["status"] == "ready"
    connection.close()

from __future__ import annotations

from pathlib import Path

from interviews_core import (
    DISCLAIMER,
    InterviewSource,
    _batch_interviews,
    analyze_topic_from_summaries,
    connect_db,
    extract_speakers,
    init_db,
    list_ready,
    parse_listing,
    parse_transcript,
    discover_sources,
    save_summary,
    split_text,
    relevant_topic_items,
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


def test_discovery_follows_pagination() -> None:
    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def get(self, url: str, **_: object) -> FakeResponse:
            if url.endswith("start=0"):
                return FakeResponse(LISTING)
            if url.endswith("start=1"):
                return FakeResponse(
                    LISTING.replace("123-larry-mcdonald-test", "124-next")
                    .replace("Larry McDonald: Market Test", "Jane Doe: Next")
                )
            return FakeResponse("")

    items = discover_sources(FakeClient(), limit=2)  # type: ignore[arg-type]
    assert [item.slug for item in items] == [
        "123-larry-mcdonald-test",
        "124-next",
    ]


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


class FakeOpenAIResponse:
    output_text = "Expert view"


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOpenAIResponse()


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_topic_analysis_uses_summaries_with_metadata() -> None:
    client = FakeOpenAI()
    result = analyze_topic_from_summaries(
        [
            {
                "title": "Larry McDonald: Market Test",
                "published_at": "2026-06-12T10:00:00-04:00",
                "speakers": ["Larry McDonald"],
                "url": "https://example.com/interview",
                "summary": "Gold is discussed as a hard asset.",
            }
        ],
        "золото",
        client=client,  # type: ignore[arg-type]
        model="test-model",
    )
    assert result == "Expert view"
    joined_inputs = "\n".join(call["input"] for call in client.responses.calls)
    assert "Larry McDonald" in joined_inputs
    assert "2026-06-12" in joined_inputs
    assert "Gold is discussed" in joined_inputs
    assert "золото" in joined_inputs
    assert len(client.responses.calls) == 1


def test_topic_analysis_rejects_empty_topic() -> None:
    try:
        analyze_topic_from_summaries([], "   ", client=FakeOpenAI())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Topic is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_batch_interviews_splits_large_archive() -> None:
    items = [
        {"title": str(index), "summary": "x" * 80, "speakers": [], "url": ""}
        for index in range(3)
    ]
    assert len(_batch_interviews(items, max_chars=120)) == 3


def test_topic_analysis_returns_fast_message_when_no_matches() -> None:
    client = FakeOpenAI()
    result = analyze_topic_from_summaries(
        [
            {
                "title": "Rates",
                "published_at": "2026-06-12",
                "speakers": ["Expert"],
                "url": "",
                "summary": "Discussion about interest rates.",
            }
        ],
        "золото",
        client=client,  # type: ignore[arg-type]
    )
    assert "не найдено" in result
    assert client.responses.calls == []


def test_relevant_topic_items_uses_aliases_and_limits() -> None:
    items = [
        {
            "title": f"Interview {index}",
            "published_at": f"2026-06-{index:02d}",
            "speakers": ["Expert"],
            "summary": "Gold miners and precious metals.",
        }
        for index in range(12)
    ]
    result = relevant_topic_items(items, "золото", limit=5)
    assert len(result) == 5
    assert result[0]["published_at"] == "2026-06-11"

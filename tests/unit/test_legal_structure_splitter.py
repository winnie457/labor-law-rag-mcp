from src.ingestion.chunking.legal_structure_splitter import (
    LegalStructureSplitter,
    LegalTextChunk,
    parse_front_matter,
)
from src.libs.splitter.base_splitter import BaseSplitter


class FallbackSplitter(BaseSplitter):
    def split_text(self, text: str) -> list[str]:
        return [text]


def test_parse_front_matter_returns_body_and_supported_metadata():
    body, metadata = parse_front_matter(
        "---\ntitle: Labor Law\ndoc_no: Order 42\nunknown: ignored\n---\nArticle text"
    )

    assert body == "Article text"
    assert metadata == {"title": "Labor Law", "doc_no": "Order 42"}


def test_split_recognizes_markdown_article_headings_and_chapters():
    splitter = LegalStructureSplitter()

    chunks = splitter.split(
        "## 第一章 总则\n### 第一条 本法适用。\n### 第二条 本法实施。",
        FallbackSplitter(),
    )

    assert chunks is not None
    article_chunk = next(chunk for chunk in chunks if chunk.article_no)
    assert article_chunk.chapter == "## 第一章 总则"
    assert article_chunk.article_no == "第一条、第二条"
    assert "### 第一条" in article_chunk.text


def test_split_recognizes_chapter_title_without_whitespace():
    splitter = LegalStructureSplitter()

    chunks = splitter.split(
        "第一章总则\n第一条本法适用。",
        FallbackSplitter(),
    )

    assert chunks is not None
    article_chunk = next(chunk for chunk in chunks if chunk.article_no)
    assert article_chunk.chapter == "第一章总则"


def test_long_article_chunks_obey_limit_with_overlap():
    splitter = LegalStructureSplitter(chunk_size=30, overlap=5)

    chunks = splitter._split_long_article(
        LegalTextChunk(text="甲" * 10 + "。" + "乙" * 28 + "。", article_no="第一条")
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert chunks[1].text[:5] == chunks[0].text[-5:]

"""Structure-aware splitting for legal documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import yaml

from src.libs.splitter.base_splitter import BaseSplitter

_NUMBER = r"(?:[零〇一二两三四五六七八九十百千万亿\d]+|[（(][零〇一二两三四五六七八九十百千万亿\d]+[）)])"
_ARTICLE_PATTERN = re.compile(rf"^\s*(?:#{{1,6}}\s*)?(第{_NUMBER}条(?:之{_NUMBER})?)")
_CHAPTER_PATTERN = re.compile(rf"^\s*(?:#{{1,6}}\s*)?第{_NUMBER}[章节](?:\s*.*)?$")
_FRONT_MATTER_FIELDS = {
    "title": ("title", "document_title"),
    "doc_no": ("doc_no", "document_no", "document_number", "文号"),
    "effective_date": ("effective_date", "effective", "施行日期"),
    "status": ("status", "validity", "状态"),
    "source_url": ("source_url", "url", "来源链接"),
    "note": ("note", "备注"),
    "region": ("region", "地区"),
    "authority_level": ("authority_level", "效力层级"),
}


@dataclass(frozen=True)
class LegalTextChunk:
    text: str
    chapter: str = ""
    article_no: str = ""


def parse_front_matter(text: str) -> tuple[str, dict[str, str]]:
    """Remove a YAML front matter block and return its supported metadata.

    Malformed or non-mapping front matter is treated as ordinary document text
    (rather than failing ingestion) so a stray ``---`` horizontal rule cannot
    block the whole pipeline.
    """
    match = re.match(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        return text, {}

    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return text, {}

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return text, {}

    metadata: dict[str, str] = {}
    for target, aliases in _FRONT_MATTER_FIELDS.items():
        value = next((parsed[key] for key in aliases if key in parsed), None)
        if value is not None:
            metadata[target] = str(value).strip()

    return text[match.end():], metadata


class LegalStructureSplitter:
    """Split article-based laws while leaving generic splitting unchanged."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        short_article_group_limit: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if short_article_group_limit <= 0:
            raise ValueError("short_article_group_limit must be positive")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.short_article_group_limit = short_article_group_limit

    def split(
        self,
        text: str,
        fallback_splitter: BaseSplitter,
    ) -> Optional[list[LegalTextChunk]]:
        """Return legal chunks, or None when no article markers are present."""
        lines = text.splitlines(keepends=True)
        prefix_lines: list[str] = []
        articles: list[LegalTextChunk] = []
        current_lines: list[str] = []
        current_article_no = ""
        current_chapter = ""
        pending_chapter_lines: list[str] = []
        seen_article = False

        def finish_article() -> None:
            if current_lines:
                article_text = "".join(current_lines).strip()
                if article_text:
                    articles.append(
                        LegalTextChunk(
                            text=article_text,
                            chapter=current_chapter,
                            article_no=current_article_no,
                        )
                    )

        for line in lines:
            stripped = line.strip()
            chapter_match = _CHAPTER_PATTERN.match(stripped)
            if chapter_match:
                finish_article()
                current_lines = []
                current_article_no = ""
                current_chapter = stripped
                # The chapter stays on the following article so it remains
                # visible in the chunk text for the second chapter onwards.
                # The first chapter is already captured in ``chapter`` and in
                # the context header, so it is not duplicated into article text.
                if seen_article:
                    pending_chapter_lines.append(line)
                continue

            article_match = _ARTICLE_PATTERN.match(stripped)
            if article_match:
                finish_article()
                current_article_no = article_match.group(1).strip()
                current_lines = pending_chapter_lines + [line]
                pending_chapter_lines = []
                seen_article = True
                continue

            if current_lines:
                current_lines.append(line)
            elif seen_article:
                pending_chapter_lines.append(line)
            else:
                prefix_lines.append(line)

        finish_article()
        if pending_chapter_lines:
            prefix_lines.extend(pending_chapter_lines)

        if not articles:
            return None

        chunks: list[LegalTextChunk] = []
        prefix = "".join(prefix_lines).strip()
        if prefix:
            chunks.extend(
                LegalTextChunk(text=fragment.strip())
                for fragment in fallback_splitter.split_text(prefix)
                if fragment.strip()
            )

        for article in self._merge_short_articles(articles):
            chunks.extend(self._split_long_article(article))
        return chunks

    def _merge_short_articles(
        self,
        articles: list[LegalTextChunk],
    ) -> list[LegalTextChunk]:
        merged: list[LegalTextChunk] = []
        for article in articles:
            previous = merged[-1] if merged else None
            combined_length = (
                len(previous.text) + 1 + len(article.text) if previous else 0
            )
            if (
                previous is not None
                and previous.chapter == article.chapter
                and len(previous.text) <= self.short_article_group_limit
                and len(article.text) <= self.short_article_group_limit
                and combined_length <= self.short_article_group_limit
            ):
                merged[-1] = LegalTextChunk(
                    text=f"{previous.text}\n{article.text}",
                    chapter=article.chapter,
                    article_no=f"{previous.article_no}、{article.article_no}",
                )
            else:
                merged.append(article)
        return merged

    def _split_long_article(self, article: LegalTextChunk) -> list[LegalTextChunk]:
        if len(article.text) <= self.chunk_size:
            return [article]

        clauses = re.split(r"(?<=[。；;])", article.text)
        chunks: list[str] = []
        current = ""
        for clause in clauses:
            if not clause:
                continue
            if len(clause) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_oversized_clause(clause))
                continue
            if current and len(current) + len(clause) > self.chunk_size:
                chunks.append(current)
                current = current[-self.overlap:] if self.overlap else ""
                while len(current) + len(clause) > self.chunk_size:
                    available = self.chunk_size - len(current)
                    current += clause[:available]
                    chunks.append(current)
                    clause = clause[available:]
                    current = chunks[-1][-self.overlap:] if self.overlap else ""
                current += clause
            else:
                current += clause

        if current:
            chunks.append(current)

        return [
            LegalTextChunk(
                text=chunk.strip(),
                chapter=article.chapter,
                article_no=article.article_no,
            )
            for chunk in chunks
            if chunk.strip()
        ]

    def _split_oversized_clause(self, text: str) -> list[str]:
        step = self.chunk_size - self.overlap
        return [
            text[start:start + self.chunk_size]
            for start in range(0, len(text), step)
        ]

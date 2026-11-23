"""Shared tokenization for BM25 index and query side.

jieba splits hyphenated/underscored English terms (``machine-learning`` →
``machine`` + ``learning``) and drops the separator.  Protecting those runs
before segmentation keeps model names, code identifiers and legal document
numbers as single searchable terms, and keeps index-side and query-side
tokenization identical.
"""

from __future__ import annotations

import re

import jieba

_PROTECTED_TERM = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


def tokenize_mixed(text: str) -> list[str]:
    """Segment text with jieba while preserving hyphenated/underscored runs."""
    tokens: list[str] = []
    cursor = 0
    for match in _PROTECTED_TERM.finditer(text):
        tokens.extend(jieba.lcut(text[cursor:match.start()]))
        tokens.append(match.group(0))
        cursor = match.end()
    tokens.extend(jieba.lcut(text[cursor:]))
    return [token.strip() for token in tokens if token and token.strip()]

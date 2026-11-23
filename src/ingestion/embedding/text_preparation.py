"""Prepare separate retrieval text without changing user-visible chunk text."""

from src.core.types import Chunk


def build_embedding_text(chunk: Chunk) -> str:
    context_header = str(chunk.metadata.get("context_header", "")).strip()
    if not context_header:
        return chunk.text
    return f"{context_header}\n{chunk.text}"

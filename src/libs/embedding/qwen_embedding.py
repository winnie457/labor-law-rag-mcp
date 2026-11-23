"""Qwen Embedding implementation.

This module implements the Qwen embedding provider using the DashScope OpenAI-compatible
API. Qwen exposes an OpenAI-like embeddings endpoint at the DashScope compatible-mode URL,
so we build on the existing OpenAIEmbedding implementation without duplicating the request
logic.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class QwenEmbeddingError(RuntimeError):
    """Raised when Qwen embeddings API call fails."""


class QwenEmbedding(OpenAIEmbedding):
    """Qwen embedding provider using the DashScope OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # DashScope rejects requests with more than 10 texts ("batch size is
    # invalid, it should not be larger than 10"), so the provider splits
    # large inputs internally instead of leaking this limit to callers.
    MAX_BATCH_SIZE = 10

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = getattr(settings.embedding, "model", "text-embedding-v3")
        if not self.model:
            self.model = "text-embedding-v3"

        resolved_api_key = api_key or getattr(settings.embedding, "api_key", None) or os.environ.get("QWEN_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Qwen API key not provided. Set settings.embedding.api_key, "
                "QWEN_API_KEY environment variable, or pass api_key parameter."
            )

        if base_url is None:
            base_url = getattr(settings.embedding, "base_url", None) or self.DEFAULT_BASE_URL

        super().__init__(
            settings=settings,
            api_key=resolved_api_key,
            base_url=base_url,
            **kwargs,
        )

    def get_dimension(self) -> int:
        """Return Qwen embedding dimensionality for the configured model."""
        if getattr(self, "dimensions", None) is not None:
            return int(self.dimensions)
        return {"text-embedding-v3": 1024}.get(self.model, 1024)

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings, splitting requests into DashScope-sized batches.

        DashScope accepts at most ``MAX_BATCH_SIZE`` texts per request, so a
        large input list is transparently split into multiple calls while
        preserving the original order.

        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.

        Returns:
            One embedding vector per input text, in the same order.

        Raises:
            ValueError: If texts list is empty.
            QwenEmbeddingError: If any underlying API call fails.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings: List[List[float]] = []

        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start:start + self.MAX_BATCH_SIZE]
            try:
                embeddings.extend(
                    super().embed(texts=batch, trace=trace, **kwargs)
                )
            except Exception as e:
                raise QwenEmbeddingError(
                    f"Qwen embedding failed for texts {start}-{start + len(batch)}: {e}"
                ) from e

        return embeddings

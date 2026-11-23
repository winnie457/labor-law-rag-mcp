"""Answer Generator for RAG responses.

Generates a natural-language answer from the user query and retrieved
document chunks using the configured LLM. Designed to fail gracefully:
if the LLM is unavailable or returns an error, ``generate()`` returns
``None`` so the caller can fall back to showing raw retrieval results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from src.core.settings import Settings, resolve_path
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.libs.llm.base_llm import BaseLLM, Message

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/answer_generation.txt"


class AnswerGenerator:
    """Generate an LLM answer grounded in retrieved chunks.

    Args:
        settings: Application settings (used to build the LLM).
        llm: Optional pre-built LLM instance (mainly for testing).
        prompt_path: Path to the answer-generation prompt template.
        max_context_chunks: Maximum number of chunks fed to the LLM.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        max_context_chunks: int = 5,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self.max_context_chunks = max_context_chunks

    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load the LLM via the factory.

        Returns None if the LLM cannot be created (e.g. no provider
        configured or API key missing) so the caller can degrade
        gracefully.
        """
        if self._llm is None:
            try:
                from src.libs.llm.llm_factory import LLMFactory

                self._llm = LLMFactory.create(self._settings)
                logger.info("LLM initialized for answer generation")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LLM for answer generation: %s. "
                    "Falling back to retrieval-only responses.",
                    exc,
                )
                self._llm = None
        return self._llm

    def _load_prompt_template(self) -> str:
        if self._prompt_template is None:
            prompt_file = resolve_path(self._prompt_path)
            self._prompt_template = Path(prompt_file).read_text(
                encoding="utf-8"
            )
        return self._prompt_template

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved chunks as a numbered context block."""
        chunks = results[: self.max_context_chunks]
        blocks = []
        for idx, result in enumerate(chunks, start=1):
            source = result.metadata.get("source_path", "") or result.metadata.get(
                "source", ""
            )
            title = result.metadata.get("title", "")
            header = f"[{idx}] {title}" if title else f"[{idx}] {source}"
            text = (result.text or "").strip()
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        trace: Optional[object] = None,
    ) -> Optional[str]:
        """Generate an answer grounded in the retrieved results.

        Args:
            query: The user's question.
            results: Retrieved document chunks (already ranked).
            trace: Optional trace context for observability.

        Returns:
            The generated answer string, or ``None`` if the LLM is
            unavailable or the call fails.
        """
        if not results:
            return None

        llm = self.llm
        if llm is None:
            return None

        try:
            template = self._load_prompt_template()
            context = self._build_context(results)
            prompt = template.format(query=query, context=context)

            from src.libs.llm.base_llm import Message

            messages = [Message(role="user", content=prompt)]
            response = llm.chat(messages, trace=trace)
            answer = (response.content or "").strip()
            return answer or None
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            return None

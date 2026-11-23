"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?
- Context Recall: Does the context support the reference answer?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"

SUPPORTED_METRICS = {
    FAITHFULNESS,
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.
        - context_recall: Measures whether the context supports the reference answer.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Optional reference answer or evaluation metadata.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)
        reference_answer = kwargs.get("reference_answer")
        if reference_answer is None and isinstance(ground_truth, dict):
            reference_answer = ground_truth.get("reference_answer")

        try:
            result = self._run_ragas(
                query, contexts, generated_answer, reference_answer=reference_answer
            )
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    def evaluate_batch(
        self,
        samples: Sequence[Dict[str, Any]],
        concurrency: int = 3,
    ) -> List[Any]:
        """Evaluate samples concurrently while sharing one async client lifecycle.

        Failed samples are returned as exceptions at their corresponding
        positions so other test cases can still complete.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not samples:
            return []

        validated_samples: List[Dict[str, Any]] = []
        for sample in samples:
            query = sample["query"]
            chunks = sample["retrieved_chunks"]
            answer = sample["generated_answer"]
            reference = sample.get("reference_answer")
            self.validate_query(query)
            self.validate_retrieved_chunks(chunks)
            if not answer or not answer.strip():
                raise ValueError("Each Ragas sample requires a non-empty generated_answer")
            if (
                {CONTEXT_PRECISION, CONTEXT_RECALL}.intersection(self._metric_names)
                and not reference
            ):
                raise ValueError(
                    "Each Ragas sample requires a reference_answer for context metrics"
                )
            validated_samples.append({
                "query": query,
                "contexts": self._extract_texts(chunks),
                "answer": answer,
                "reference_answer": reference,
            })

        return asyncio.run(
            self._run_ragas_batch_async(validated_samples, concurrency)
        )

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run all selected Ragas metrics within one event loop."""
        return asyncio.run(
            self._run_ragas_async(
                query, contexts, answer, reference_answer=reference_answer
            )
        )

    async def _run_ragas_async(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Execute selected Ragas metrics for one sample."""
        llm, embeddings = self._build_wrappers()
        try:
            return await self._score_ragas_metrics(
                llm, embeddings, query, contexts, answer, reference_answer
            )
        finally:
            await self._close_wrappers(llm, embeddings)

    async def _run_ragas_batch_async(
        self,
        samples: List[Dict[str, Any]],
        concurrency: int,
    ) -> List[Any]:
        """Score samples concurrently, sharing clients within this event loop."""
        llm, embeddings = self._build_wrappers()
        semaphore = asyncio.Semaphore(concurrency)

        async def score_sample(sample: Dict[str, Any]) -> Dict[str, float]:
            async with semaphore:
                return await self._score_ragas_metrics(
                    llm,
                    embeddings,
                    sample["query"],
                    sample["contexts"],
                    sample["answer"],
                    sample["reference_answer"],
                )

        try:
            return list(
                await asyncio.gather(
                    *(score_sample(sample) for sample in samples),
                    return_exceptions=True,
                )
            )
        finally:
            await self._close_wrappers(llm, embeddings)

    async def _score_ragas_metrics(
        self,
        llm: Any,
        embeddings: Any,
        query: str,
        contexts: List[str],
        answer: str,
        reference_answer: Optional[str],
    ) -> Dict[str, float]:
        """Score all configured metrics for one sample."""
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        async def score_metric(metric_name: str) -> float:
            if metric_name == FAITHFULNESS:
                result = await Faithfulness(llm=llm).ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                result = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
                    user_input=query, response=answer
                )
            elif metric_name == CONTEXT_PRECISION:
                if not reference_answer:
                    raise ValueError(
                        "A non-empty reference_answer is required for context_precision"
                    )
                result = await ContextPrecision(llm=llm).ascore(
                    user_input=query,
                    reference=reference_answer,
                    retrieved_contexts=contexts,
                )
            elif metric_name == CONTEXT_RECALL:
                if not reference_answer:
                    raise ValueError(
                        "A non-empty reference_answer is required for context_recall"
                    )
                result = await ContextRecall(llm=llm).ascore(
                    user_input=query,
                    reference=reference_answer,
                    retrieved_contexts=contexts,
                )
            else:
                raise ValueError(f"Unsupported Ragas metric: {metric_name}")

            return float(result.value) if result.value is not None else 0.0

        values = await asyncio.gather(
            *(score_metric(metric_name) for metric_name in self._metric_names),
            return_exceptions=True,
        )
        for value in values:
            if isinstance(value, BaseException):
                raise value
        return {
            metric_name: float(value)
            for metric_name, value in zip(self._metric_names, values)
        }

    @staticmethod
    async def _close_wrappers(llm: Any, embeddings: Any) -> None:
        clients = [
            getattr(getattr(llm, "client", None), "client", None),
            getattr(embeddings, "client", None),
        ]
        closed_ids = set()
        for client in clients:
            if client is not None and id(client) not in closed_ids:
                await client.close()
                closed_ids.add(id(client))

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Uses Ragas 0.4+ native API (InstructorLLM + OpenAIEmbeddings)
        instead of deprecated LangchainLLMWrapper.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure-compatible mode: if azure_endpoint is configured, use Azure
        # client even when provider is "openai" (matches project convention).
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        llm_base_url = getattr(llm_cfg, "base_url", None)
        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider in {"openai", "deepseek", "qwen"}:
            llm_client = AsyncOpenAI(
                api_key=llm_cfg.api_key,
                base_url=llm_base_url or None,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai-compatible (openai, deepseek, qwen)"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        # ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Same Azure-compatible mode detection for embeddings
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        emb_base_url = getattr(emb_cfg, "base_url", None)
        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider in {"openai", "deepseek", "qwen"}:
            emb_client = AsyncOpenAI(
                api_key=emb_cfg.api_key,
                base_url=emb_base_url or None,
            )
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai-compatible (openai, deepseek, qwen)"
            )

        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]

"""Custom evaluator implementation for lightweight metrics.

This evaluator computes simple, deterministic metrics such as hit rate and MRR.
It is designed for fast regression checks and sanity validation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr).

    Two matching modes are supported:

    1. Chunk-level (legacy): ground_truth is a list of ids, or a dict with
       an ``ids`` key. Retrieved chunks must expose an identifier field.
       Supported id fields: id, chunk_id, document_id, doc_id.
    2. Source-level: ground_truth is a dict with a ``sources`` (or
       ``expected_sources``) key holding file names. A retrieved chunk is
       considered relevant when its ``metadata.source_path`` ends with one
       of the expected file names. This keeps ground truth stable when
       chunking parameters change.
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")
    _SOURCE_FIELDS = ("source_path", "source", "file_path")
    _GROUND_TRUTH_SOURCE_KEYS = ("sources", "expected_sources")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth ids or structure.
            trace: Optional TraceContext (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        results: Dict[str, float] = {}

        # Clause-level mode: ground truth specifies source file + clause number.
        # This is stricter than source-level: the retrieved chunk must come from
        # the expected file AND contain the expected clause number.
        has_chunk_ids = isinstance(ground_truth, dict) and ground_truth.get("ids")
        ground_truth_clauses = (
            [] if has_chunk_ids else self._extract_ground_truth_clauses(ground_truth)
        )
        if ground_truth_clauses:
            retrieved_items = self._extract_retrieved_items(retrieved_chunks)
            if "hit_rate" in self.metrics:
                results["hit_rate"] = self._compute_clause_hit_rate(
                    retrieved_items, ground_truth_clauses
                )
            if "mrr" in self.metrics:
                results["mrr"] = self._compute_clause_mrr(
                    retrieved_items, ground_truth_clauses
                )
            return results

        # Source-level mode: ground truth specifies file names instead of ids.
        ground_truth_sources = (
            [] if has_chunk_ids else self._extract_ground_truth_sources(ground_truth)
        )
        if ground_truth_sources:
            retrieved_sources = self._extract_retrieved_sources(retrieved_chunks)
            if "hit_rate" in self.metrics:
                results["hit_rate"] = self._compute_source_hit_rate(
                    retrieved_sources, ground_truth_sources
                )
            if "mrr" in self.metrics:
                results["mrr"] = self._compute_source_mrr(
                    retrieved_sources, ground_truth_sources
                )
            return results

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)

        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_ids, ground_truth_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_ids, ground_truth_ids)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground truth ids from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items."""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary)."""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    # ------------------------------------------------------------------
    # Source-level matching (file name based, independent of chunk ids)
    # ------------------------------------------------------------------

    def _extract_ground_truth_sources(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract expected file names from ground_truth."""
        if not isinstance(ground_truth, dict):
            return []

        raw: Any = None
        for key in self._GROUND_TRUTH_SOURCE_KEYS:
            if ground_truth.get(key):
                raw = ground_truth[key]
                break
        if not raw:
            return []

        if isinstance(raw, str):
            raw = [raw]

        sources: List[str] = []
        for item in raw:
            if isinstance(item, str):
                sources.append(item)
            elif isinstance(item, dict):
                for field in (*self._SOURCE_FIELDS, "file_name"):
                    if item.get(field):
                        sources.append(str(item[field]))
                        break
        return [source for source in sources if source.strip()]

    def _extract_retrieved_sources(self, items: Iterable[Any]) -> List[str]:
        """Extract source paths from retrieved chunks, preserving rank order."""
        sources: List[str] = []
        for item in items:
            metadata = None
            if isinstance(item, dict):
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                direct_fields = (item, metadata)
            else:
                metadata = getattr(item, "metadata", None)
                direct_fields = (item, metadata if isinstance(metadata, dict) else {})

            source = ""
            for container in direct_fields:
                if not isinstance(container, dict):
                    continue
                for field in self._SOURCE_FIELDS:
                    if container.get(field):
                        source = str(container[field])
                        break
                if source:
                    break

            if not source:
                attr = getattr(item, "source_path", None)
                source = str(attr) if attr else ""

            sources.append(source)
        return sources

    @staticmethod
    def _source_matches(retrieved_source: str, expected_source: str) -> bool:
        """Check whether a retrieved source path ends with the expected file name."""
        retrieved = str(retrieved_source).replace("\\", "/").rstrip("/")
        expected = str(expected_source).replace("\\", "/").strip().strip("/")
        if not retrieved or not expected:
            return False
        return retrieved == expected or retrieved.endswith("/" + expected)

    def _compute_source_hit_rate(
        self, retrieved_sources: Sequence[str], expected_sources: Sequence[str]
    ) -> float:
        """Compute source-level hit rate (binary)."""
        if not expected_sources:
            return 0.0
        for retrieved in retrieved_sources:
            if any(self._source_matches(retrieved, expected) for expected in expected_sources):
                return 1.0
        return 0.0

    def _compute_source_mrr(
        self, retrieved_sources: Sequence[str], expected_sources: Sequence[str]
    ) -> float:
        """Compute source-level MRR (first relevant source in rank order)."""
        if not expected_sources:
            return 0.0
        for rank, retrieved in enumerate(retrieved_sources, start=1):
            if any(self._source_matches(retrieved, expected) for expected in expected_sources):
                return 1.0 / rank
        return 0.0

    # ------------------------------------------------------------------
    # Clause-level matching (source file + clause number)
    # ------------------------------------------------------------------

    _GROUND_TRUTH_CLAUSE_KEYS = ("expected_clauses", "clauses")

    # Chinese numeral map for clause number normalization (e.g. 第四十七条 <-> 第47条)
    _CN_NUMERALS = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        "十": 10, "百": 100, "千": 1000, "万": 10000,
    }

    def _extract_ground_truth_clauses(self, ground_truth: Optional[Any]) -> List[Dict[str, str]]:
        """Extract expected clauses from ground_truth.

        Each clause is a dict with ``source`` (file name) and ``clause``
        (e.g. "第四十七条" or "第47条").
        """
        if not isinstance(ground_truth, dict):
            return []

        raw: Any = None
        for key in self._GROUND_TRUTH_CLAUSE_KEYS:
            if ground_truth.get(key):
                raw = ground_truth[key]
                break
        if not raw:
            return []

        if isinstance(raw, dict):
            raw = [raw]

        clauses: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            clause = str(item.get("clause", "")).strip()
            if source and clause:
                clauses.append({"source": source, "clause": clause})
        return clauses

    def _extract_retrieved_items(self, items: Iterable[Any]) -> List[Dict[str, str]]:
        """Extract (source_path, text) pairs from retrieved chunks, preserving rank."""
        extracted: List[Dict[str, str]] = []
        for item in items:
            source = ""
            text = ""

            if isinstance(item, dict):
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                for container in (item, metadata):
                    for field in self._SOURCE_FIELDS:
                        if container.get(field):
                            source = str(container[field])
                            break
                    if source:
                        break
                text = str(item.get("text", item.get("content", "")))
            else:
                metadata = getattr(item, "metadata", None)
                if isinstance(metadata, dict):
                    for field in self._SOURCE_FIELDS:
                        if metadata.get(field):
                            source = str(metadata[field])
                            break
                if not source:
                    attr = getattr(item, "source_path", None)
                    source = str(attr) if attr else ""
                text = str(getattr(item, "text", getattr(item, "content", "")))

            extracted.append({"source": source, "text": text})
        return extracted

    @classmethod
    def _normalize_clause_number(cls, clause: str) -> str:
        """Convert a clause label to a canonical "第N条" form where N is Arabic.

        "第四十七条" -> "第47条", "第47条" -> "第47条", "第八十七条" -> "第87条".
        Also handles "款" forms: "第四十七条第二款" -> "第47条第2款".
        """
        import re

        text = str(clause).strip()
        if not text:
            return text

        def _cn_to_arabic(cn: str) -> int:
            """Convert a Chinese numeral string to int (handles up to 万)."""
            total = 0
            current = 0
            for ch in cn:
                val = cls._CN_NUMERALS.get(ch)
                if val is None:
                    continue
                if val >= 10:
                    if current == 0:
                        current = 1
                    total += current * val
                    current = 0
                else:
                    current = val
            total += current
            return total

        # Pattern: 第<number>条[第<number>款]
        def _replace(match: re.Match) -> str:
            prefix = match.group(1)  # "条" or ""
            num_str = match.group(2)
            suffix = match.group(3)  # "条" or "款"
            if num_str.isdigit():
                arabic = int(num_str)
            else:
                arabic = _cn_to_arabic(num_str)
            return f"{prefix}第{arabic}{suffix}"

        # Match 第<cn-or-arabic>条 or 第<cn-or-arabic>款
        result = re.sub(r"(条?)第([零一二两三四五六七八九十百千万\d]+)([条款])", _replace, text)
        return result

    @staticmethod
    def _arabic_to_chinese(num: int) -> str:
        """Convert a positive integer to Chinese numerals (up to 9999)."""
        digits = "零一二三四五六七八九"
        if num == 0:
            return "零"
        if num < 10:
            return digits[num]
        if num < 20:
            return "十" + (digits[num - 10] if num > 10 else "")
        if num < 100:
            tens, ones = divmod(num, 10)
            return digits[tens] + "十" + (digits[ones] if ones else "")
        if num < 1000:
            hundreds, rest = divmod(num, 100)
            result = digits[hundreds] + "百"
            if rest == 0:
                return result
            if rest < 10:
                return result + "零" + digits[rest]
            return result + CustomEvaluator._arabic_to_chinese(rest)
        # 1000-9999
        thousands, rest = divmod(num, 1000)
        result = digits[thousands] + "千"
        if rest == 0:
            return result
        if rest < 100:
            return result + "零" + CustomEvaluator._arabic_to_chinese(rest)
        return result + CustomEvaluator._arabic_to_chinese(rest)

    @classmethod
    def _arabic_marker_to_chinese(cls, marker: str) -> str:
        """Convert an Arabic clause marker to Chinese numerals.

        "第4条" -> "第四条", "第47条第2款" -> "第四十七条第二款".
        """
        import re

        def _replace(match: re.Match) -> str:
            prefix = match.group(1)
            num = int(match.group(2))
            suffix = match.group(3)
            return f"{prefix}第{cls._arabic_to_chinese(num)}{suffix}"

        return re.sub(r"(条?)第(\d+)([条款])", _replace, marker)

    @classmethod
    def _clause_matches(cls, chunk: Dict[str, str], expected: Dict[str, str]) -> bool:
        """Check whether a retrieved chunk matches an expected clause.

        A chunk matches when:
        1. Its source_path ends with the expected file name, AND
        2. Its text contains the expected clause number (in either Chinese
           or Arabic numeral form).
        """
        if not cls._source_matches(chunk.get("source", ""), expected["source"]):
            return False

        text = chunk.get("text", "")
        if not text:
            return False

        import re
        # The retrieved text may use Chinese numerals (第四条) while the
        # ground truth may use Arabic (第4条) or vice versa. Check both forms.
        normalized_expected = cls._normalize_clause_number(expected["clause"])
        m = re.search(r"第\d+条(?:第\d+款)?", normalized_expected)
        if not m:
            return False
        arabic_marker = m.group(0)   # e.g. "第4条" or "第47条第2款"
        # Convert back to Chinese numerals to also match "第四条" in text.
        chinese_marker = cls._arabic_marker_to_chinese(arabic_marker)
        return arabic_marker in text or chinese_marker in text

    def _compute_clause_hit_rate(
        self, retrieved_items: Sequence[Dict[str, str]], expected_clauses: Sequence[Dict[str, str]]
    ) -> float:
        """Compute clause-level hit rate (binary).

        Hit if at least one retrieved chunk matches any expected clause.
        """
        if not expected_clauses:
            return 0.0
        for chunk in retrieved_items:
            if any(self._clause_matches(chunk, clause) for clause in expected_clauses):
                return 1.0
        return 0.0

    def _compute_clause_mrr(
        self, retrieved_items: Sequence[Dict[str, str]], expected_clauses: Sequence[Dict[str, str]]
    ) -> float:
        """Compute clause-level MRR (first matching chunk in rank order)."""
        if not expected_clauses:
            return 0.0
        for rank, chunk in enumerate(retrieved_items, start=1):
            if any(self._clause_matches(chunk, clause) for clause in expected_clauses):
                return 1.0 / rank
        return 0.0
#!/usr/bin/env python
"""Run Custom retrieval and Ragas generation evaluations on the golden set.

Usage:
    python scripts/evaluate_golden.py
    python scripts/evaluate_golden.py --top-k 5 --no-rerank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Golden Test Set with Custom metrics and Ragas."
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to the golden test set JSON.",
    )
    parser.add_argument("--collection", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for a quicker retrieval-only comparison.",
    )
    parser.add_argument(
        "--rerank-retries",
        type=int,
        default=2,
        help="Retries after reranker timeout/fallback (default: 2).",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip Ragas LLM-as-judge scoring; report local retrieval metrics only.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation; compute retrieval metrics (HitRate/MRR) only.",
    )
    return parser.parse_args()


def build_components(settings, collection: str, enable_reranker: bool):
    from scripts.eval_compare import build_retrievers
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor

    dense_retriever, sparse_retriever = build_retrievers(settings, collection)
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    reranker = None
    if enable_reranker:
        from src.core.query_engine.reranker import create_core_reranker

        reranker = create_core_reranker(settings)
    return hybrid_search, reranker


def _is_finite_score(score: float) -> bool:
    return score == score and abs(score) != float("inf")


def run_ragas(cases, query_results, settings) -> tuple[dict[str, float], int, list[str]]:
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(settings=settings, metrics=RAGAS_METRICS)
    score_rows: list[dict[str, float]] = []
    errors: list[str] = []
    sample_ids: list[str] = []
    samples: list[dict[str, object]] = []

    for case, result in zip(cases, query_results):
        if not case.answerable:
            continue
        if result.error:
            errors.append(f"{case.test_id}: {result.error}")
            continue
        if not result.generated_answer:
            errors.append(f"{case.test_id}: RAG returned no answer")
            continue

        sample_ids.append(case.test_id)
        samples.append({
            "query": case.query,
            "retrieved_chunks": result.retrieved_contexts or [""],
            "generated_answer": result.generated_answer,
            "reference_answer": case.reference_answer,
        })

    print(f"[*] Ragas scoring {len(samples)} answerable cases (concurrency=3)...")
    outputs = evaluator.evaluate_batch(samples, concurrency=3)
    for test_id, output in zip(sample_ids, outputs):
        if isinstance(output, BaseException):
            errors.append(f"{test_id}: {output}")
            print(f"[ERROR] Ragas evaluation failed for {test_id}: {output}")
            continue

        missing = set(RAGAS_METRICS) - set(output)
        if missing:
            error = f"Ragas omitted metrics: {', '.join(sorted(missing))}"
            errors.append(f"{test_id}: {error}")
            print(f"[ERROR] {test_id}: {error}")
            continue
        invalid = {
            name: value
            for name, value in output.items()
            if not _is_finite_score(value)
        }
        if invalid:
            error = f"Ragas returned non-finite scores: {invalid}"
            errors.append(f"{test_id}: {error}")
            print(f"[ERROR] {test_id}: {error}")
            continue

        score_rows.append(output)
        print(f"Ragas [{test_id}]: {output}")

    aggregates = {
        metric: sum(row[metric] for row in score_rows) / len(score_rows)
        for metric in RAGAS_METRICS
    } if score_rows else {}
    return aggregates, len(score_rows), errors


def print_markdown_report(
    custom_metrics: dict[str, float],
    ragas_metrics: dict[str, float],
    case_count: int,
    answerable_count: int,
    custom_count: int,
    ragas_count: int,
) -> None:
    print("\n## Golden Test Set 评估结果\n")
    print(
        f"- 用例：{case_count}（可回答 {answerable_count}，"
        f"不可回答 {case_count - answerable_count}）"
    )
    print(f"- Custom：{custom_count}/{answerable_count} 条可回答用例")
    if ragas_metrics:
        print(f"- Ragas：{ragas_count}/{answerable_count} 条可回答用例")
    else:
        print("- Ragas：已跳过（--no-ragas）")
    print("\n| 评估体系 | 指标 | 分数 |")
    print("|---|---|---:|")
    custom_labels = (
        ("hit_rate", "Hit Rate@5"),
        ("mrr", "MRR"),
        ("empty_recall", "不可回答用例空召回率"),
        ("abstention_rate", "不可回答用例拒答率"),
        ("hallucination_rate", "不可回答用例幻觉率"),
    )
    for key, label in custom_labels:
        if key in custom_metrics:
            print(f"| Custom | {label} | {custom_metrics[key]:.4f} |")
    if ragas_metrics:
        ragas_labels = (
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall"),
        )
        for key, label in ragas_labels:
            value = ragas_metrics.get(key)
            score = f"{value:.4f}" if value is not None else "未计算"
            print(f"| Ragas | {label} | {score} |")


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        print("[ERROR] --top-k must be positive.", file=sys.stderr)
        return 2
    if args.rerank_retries < 0:
        print("[ERROR] --rerank-retries cannot be negative.", file=sys.stderr)
        return 2

    from src.core.response.answer_generator import AnswerGenerator
    from src.core.settings import load_settings
    from src.libs.evaluator.custom_evaluator import CustomEvaluator
    from src.observability.evaluation.eval_runner import EvalRunner, load_test_set

    test_set_path = Path(args.test_set)
    if not test_set_path.is_absolute():
        test_set_path = PROJECT_ROOT / test_set_path
    raw_test_set = json.loads(test_set_path.read_text(encoding="utf-8"))
    cases = load_test_set(test_set_path)
    if not cases:
        print("[ERROR] Golden test set is empty.", file=sys.stderr)
        return 2

    settings = load_settings()
    collection = args.collection or raw_test_set.get("collection", "default")
    print(
        f"Golden set: {test_set_path} | collection: {collection} | "
        f"top-k: {args.top_k} | cases: {len(cases)}"
    )

    try:
        hybrid_search, reranker = build_components(
            settings, collection, enable_reranker=not args.no_rerank
        )
        if args.retrieval_only:
            answer_generator = None
            generate_answers = False
            print("[*] Retrieval-only mode: skipping LLM answer generation.")
        else:
            answer_generator = AnswerGenerator(settings=settings)
            generate_answers = True
        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=CustomEvaluator(metrics=["hit_rate", "mrr"]),
            answer_generator=(
                None
                if answer_generator is None
                else lambda query, chunks: answer_generator.generate(query, chunks)
            ),
            reranker=reranker,
            rerank_retries=args.rerank_retries,
            generate_answers=generate_answers,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to initialize evaluation: {exc}", file=sys.stderr)
        return 2

    try:
        custom_report = runner.run(
            test_set_path=test_set_path,
            top_k=args.top_k,
            collection=collection,
        )
    except Exception as exc:
        print(f"[ERROR] Custom evaluation failed: {exc}", file=sys.stderr)
        return 1

    answerable_count = sum(case.answerable for case in cases)
    custom_count = sum(
        case.answerable and not result.error
        for case, result in zip(cases, custom_report.query_results)
    )
    print("\nCustom evaluation completed.")

    if args.no_ragas:
        print("\n[*] Skipping Ragas scoring (--no-ragas).")
        ragas_metrics: dict[str, float] = {}
        ragas_count = 0
        ragas_errors: list[str] = []
    else:
        try:
            ragas_metrics, ragas_count, ragas_errors = run_ragas(
                cases, custom_report.query_results, settings
            )
        except Exception as exc:
            print(f"[ERROR] Failed to initialize Ragas evaluation: {exc}", file=sys.stderr)
            return 1

    print_markdown_report(
        custom_metrics=custom_report.aggregate_metrics,
        ragas_metrics=ragas_metrics,
        case_count=len(cases),
        answerable_count=answerable_count,
        custom_count=custom_count,
        ragas_count=ragas_count,
    )

    errors = [
        f"{result.query}: {result.error}"
        for result in custom_report.query_results
        if result.error
    ] + ragas_errors
    if errors:
        print(f"\n[ERROR] {len(errors)} case(s) were not fully evaluated:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

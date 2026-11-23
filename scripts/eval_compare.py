#!/usr/bin/env python
"""Compare retrieval configurations on the labour-law golden set.

Configurations:
    1. dense-only      : dense retrieval only
    2. sparse-only     : BM25 retrieval only
    3. hybrid          : dense + sparse with RRF fusion
    4. hybrid+rerank   : hybrid candidates reranked by cross-encoder

Ground truth is file-name based (source_path.endswith(file_name)), so chunking
parameter changes do not invalidate the golden set.

Usage:
    python scripts/eval_compare.py
    python scripts/eval_compare.py --top-k 5 --collection default
    python scripts/eval_compare.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval configurations.")
    parser.add_argument("--test-set", default="tests/fixtures/golden_test_set.json")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


def build_retrievers(settings: Any, collection: str):
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)

    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    return dense_retriever, sparse_retriever


def run_config(
    name: str,
    settings: Any,
    collection: str,
    dense_retriever: Any,
    sparse_retriever: Any,
    test_set: str,
    top_k: int,
    enable_dense: bool,
    enable_sparse: bool,
    use_reranker: bool,
) -> Dict[str, Any]:
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
    from src.core.query_engine.query_processor import QueryProcessor
    from src.libs.evaluator.custom_evaluator import CustomEvaluator
    from src.observability.evaluation.eval_runner import EvalRunner

    config = HybridSearchConfig(
        enable_dense=enable_dense,
        enable_sparse=enable_sparse,
    )

    # Build RRF fusion from settings (rrf_k defaults to 60).
    # Without this, HybridSearch falls back to simple interleaving, which
    # does not actually combine dense+sparse ranking signals.
    rrf_k = 60
    retrieval_config = getattr(settings, "retrieval", None)
    if retrieval_config is not None:
        rrf_k = getattr(retrieval_config, "rrf_k", 60)
    fusion = RRFFusion(k=rrf_k)

    hybrid_search = HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever if enable_dense else None,
        sparse_retriever=sparse_retriever if enable_sparse else None,
        fusion=fusion,
        config=config,
    )

    reranker = None
    if use_reranker:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings)

    evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        reranker=reranker,
    )

    report = runner.run(test_set_path=test_set, top_k=top_k, collection=collection)

    evaluated = [qr for qr in report.query_results if qr.metrics]
    hits = sum(1 for qr in evaluated if qr.metrics.get("hit_rate", 0.0) >= 1.0)
    return {
        "name": name,
        "hit_rate": report.aggregate_metrics.get("hit_rate", 0.0),
        "mrr": report.aggregate_metrics.get("mrr", 0.0),
        "evaluated": len(evaluated),
        "hits": hits,
        "query_results": report.query_results,
        "total_elapsed_ms": report.total_elapsed_ms,
    }


def main() -> int:
    args = parse_args()

    from src.core.settings import load_settings
    from src.observability.evaluation.eval_runner import load_test_set

    settings = load_settings()
    collection = args.collection or "default"

    raw = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    category_by_query = {
        tc["query"]: tc.get("category", "unknown") for tc in raw["test_cases"]
    }
    test_cases = load_test_set(args.test_set)
    answerable = [tc for tc in test_cases if tc.expected_sources or tc.expected_chunk_ids]
    unanswerable = [tc for tc in test_cases if not (tc.expected_sources or tc.expected_chunk_ids)]

    print(f"Golden set : {args.test_set}")
    print(f"Collection : {collection}")
    print(f"Queries    : {len(test_cases)} total / {len(answerable)} answerable / {len(unanswerable)} unanswerable")
    print(f"Top-K      : {args.top_k}\n")

    dense_retriever, sparse_retriever = build_retrievers(settings, collection)

    configs = [
        ("dense-only", True, False, False),
        ("sparse-only", False, True, False),
        ("hybrid", True, True, False),
        ("hybrid+rerank", True, True, True),
    ]

    results: List[Dict[str, Any]] = []
    for name, enable_dense, enable_sparse, use_reranker in configs:
        print(f"[*] Running {name} ...")
        result = run_config(
            name=name,
            settings=settings,
            collection=collection,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            test_set=args.test_set,
            top_k=args.top_k,
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            use_reranker=use_reranker,
        )
        results.append(result)
        print(
            f"    hit@{args.top_k}={result['hit_rate']:.4f}  "
            f"MRR={result['mrr']:.4f}  "
            f"({result['hits']}/{result['evaluated']})  "
            f"{result['total_elapsed_ms']/1000:.1f}s"
        )

    print("\n" + "=" * 66)
    print("  RETRIEVAL COMPARISON")
    print("=" * 66)
    print(f"  {'config':<16}{'hit@' + str(args.top_k):<10}{'MRR':<10}{'hits':<10}")
    print("-" * 66)
    for r in results:
        print(
            f"  {r['name']:<16}{r['hit_rate']:<10.4f}{r['mrr']:<10.4f}"
            f"{str(r['hits']) + '/' + str(r['evaluated']):<10}"
        )
    print("=" * 66)

    # Per-category breakdown (hit rate only)
    categories = ["single_point", "cross_document"]
    print("\n  PER-CATEGORY hit@" + str(args.top_k))
    print("-" * 66)
    header = "  " + "category".ljust(18) + "".join(name.ljust(18) for name, *_ in configs)
    print(header)
    for category in categories:
        row = "  " + category.ljust(18)
        for r in results:
            values = [
                qr.metrics.get("hit_rate", 0.0)
                for qr in r["query_results"]
                if qr.metrics and category_by_query.get(qr.query) == category
            ]
            value = sum(values) / len(values) if values else 0.0
            row += f"{value:.4f}".ljust(18)
        print(row)
    print("-" * 66)

    # Unanswerable probes: retrieval never returns empty, so report what came back.
    if unanswerable:
        print("\n  UNANSWERABLE PROBES (excluded from metrics; inspect top-1 source)")
        print("-" * 66)
        for tc in unanswerable:
            print(f"  Q: {tc.query}")
            for r in results:
                match = next((qr for qr in r["query_results"] if qr.query == tc.query), None)
                top = match.retrieved_sources[0] if match and match.retrieved_sources else "(empty)"
                top = Path(top).name if top and top != "(empty)" else top
                print(f"     {r['name']:<15} top1={top}")
        print("-" * 66)

    if args.verbose:
        print("\n  PER-QUERY DETAIL")
        print("-" * 66)
        for tc in test_cases:
            print(f"\n  [{category_by_query.get(tc.query, '?')}] {tc.query}")
            for r in results:
                match = next((qr for qr in r["query_results"] if qr.query == tc.query), None)
                if not match:
                    continue
                sources = ", ".join(Path(s).name for s in match.retrieved_sources[:3]) or "(empty)"
                metric = (
                    f"hit={match.metrics.get('hit_rate', 0.0):.0f} rr={match.metrics.get('mrr', 0.0):.2f}"
                    if match.metrics else "unscored"
                )
                print(f"     {r['name']:<15} {metric:<22} {sources}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

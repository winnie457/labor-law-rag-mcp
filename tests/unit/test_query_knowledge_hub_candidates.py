from unittest.mock import Mock

from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)


def test_search_requests_at_least_thirty_candidates_for_reranking():
    hybrid_search = Mock()
    hybrid_search.search.return_value = []
    reranker = Mock()
    reranker.is_enabled = True
    tool = QueryKnowledgeHubTool(
        config=QueryKnowledgeHubConfig(enable_rerank=True),
        hybrid_search=hybrid_search,
        reranker=reranker,
    )

    tool._perform_search("query", top_k=5)

    hybrid_search.search.assert_called_once()
    assert hybrid_search.search.call_args.kwargs["top_k"] == 30


def test_search_keeps_requested_candidate_count_without_reranking():
    hybrid_search = Mock()
    hybrid_search.search.return_value = []
    tool = QueryKnowledgeHubTool(
        config=QueryKnowledgeHubConfig(enable_rerank=False),
        hybrid_search=hybrid_search,
    )

    tool._perform_search("query", top_k=5)

    assert hybrid_search.search.call_args.kwargs["top_k"] == 5

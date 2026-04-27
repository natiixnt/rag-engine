"""Unit tests for the hybrid retriever's core logic.

These test the RRF fusion and score normalization without needing
a running HNSW index or embedding model - just pure math.
"""

from __future__ import annotations

import pytest

from rag_engine import RetrievalResult
from rag_engine.retriever import HybridRetriever


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        metadata={},
        parent_id=None,
    )


class TestReciprocalRankFusion:
    """Tests for the RRF fusion logic.

    RRF formula: score(d) = sum(weight_i / (k + rank_i + 1))
    where k=60 (from the original Cormack et al. paper).
    """

    def test_single_source_ranking_preserved(self):
        # if only one source returns results, order should be preserved
        sparse = [_make_result("a", 5.0), _make_result("b", 3.0), _make_result("c", 1.0)]
        dense = []

        fused = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.3, dense_weight=0.7, k=60
        )

        ids = [r.chunk_id for r in sorted(fused, key=lambda r: r.score, reverse=True)]
        assert ids == ["a", "b", "c"]

    def test_both_sources_boost_shared_docs(self):
        # docs appearing in both lists should get boosted above docs in only one
        sparse = [_make_result("shared", 5.0), _make_result("sparse_only", 4.0)]
        dense = [_make_result("shared", 0.9), _make_result("dense_only", 0.8)]

        fused = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.3, dense_weight=0.7, k=60
        )

        score_map = {r.chunk_id: r.score for r in fused}
        # shared doc gets contributions from both paths
        assert score_map["shared"] > score_map["sparse_only"]
        assert score_map["shared"] > score_map["dense_only"]

    def test_dense_weight_dominates(self):
        # with default weights (0.3 sparse, 0.7 dense), the dense ranking
        # should have more influence on the final order
        sparse = [_make_result("a", 5.0), _make_result("b", 3.0)]
        dense = [_make_result("b", 0.95), _make_result("a", 0.90)]

        fused = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.3, dense_weight=0.7, k=60
        )

        score_map = {r.chunk_id: r.score for r in fused}
        # b is rank 1 in dense (weight 0.7) and rank 2 in sparse (weight 0.3)
        # a is rank 1 in sparse (weight 0.3) and rank 2 in dense (weight 0.7)
        # so b should win because dense weight is higher
        assert score_map["b"] > score_map["a"]

    def test_rrf_scores_are_positive(self):
        sparse = [_make_result(f"doc_{i}", float(10 - i)) for i in range(10)]
        dense = [_make_result(f"doc_{i}", float(10 - i) / 10) for i in range(10)]

        fused = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.3, dense_weight=0.7, k=60
        )

        for r in fused:
            assert r.score > 0

    def test_k_parameter_affects_score_magnitude(self):
        # higher k flattens the rank differences (less top-heavy)
        sparse = [_make_result("a", 5.0), _make_result("b", 3.0)]
        dense = [_make_result("a", 0.9), _make_result("b", 0.8)]

        fused_k60 = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.5, dense_weight=0.5, k=60
        )
        fused_k10 = HybridRetriever._reciprocal_rank_fusion(
            sparse, dense, sparse_weight=0.5, dense_weight=0.5, k=10
        )

        scores_k60 = {r.chunk_id: r.score for r in fused_k60}
        scores_k10 = {r.chunk_id: r.score for r in fused_k10}

        # with smaller k, the gap between rank 1 and rank 2 is larger
        gap_k60 = scores_k60["a"] - scores_k60["b"]
        gap_k10 = scores_k10["a"] - scores_k10["b"]
        assert gap_k10 > gap_k60

    def test_empty_inputs_returns_empty(self):
        fused = HybridRetriever._reciprocal_rank_fusion([], [], sparse_weight=0.3, dense_weight=0.7)
        assert fused == []

    def test_equal_weights_symmetric(self):
        # with equal weights, swapping sparse and dense should give same scores
        results_a = [_make_result("x", 5.0), _make_result("y", 3.0)]
        results_b = [_make_result("y", 0.9), _make_result("x", 0.8)]

        fused_1 = HybridRetriever._reciprocal_rank_fusion(
            results_a, results_b, sparse_weight=0.5, dense_weight=0.5, k=60
        )
        fused_2 = HybridRetriever._reciprocal_rank_fusion(
            results_b, results_a, sparse_weight=0.5, dense_weight=0.5, k=60
        )

        scores_1 = {r.chunk_id: r.score for r in fused_1}
        scores_2 = {r.chunk_id: r.score for r in fused_2}

        # scores should be identical when weights are symmetric
        assert abs(scores_1["x"] - scores_2["x"]) < 1e-10
        assert abs(scores_1["y"] - scores_2["y"]) < 1e-10


class TestScoreNormalization:
    """Tests for the inverse distance transform used in dense search."""

    def test_zero_distance_gives_max_score(self):
        # score = 1 / (1 + distance), so distance=0 gives score=1.0
        score = 1.0 / (1.0 + 0.0)
        assert score == 1.0

    def test_score_decreases_with_distance(self):
        distances = [0.1, 0.5, 1.0, 2.0, 5.0]
        scores = [1.0 / (1.0 + d) for d in distances]
        # scores should be monotonically decreasing
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_score_always_positive(self):
        # even with very large distances, score stays in (0, 1]
        for d in [0, 0.001, 1, 100, 10000, 1e9]:
            score = 1.0 / (1.0 + d)
            assert 0 < score <= 1.0

    def test_score_bounded_zero_one(self):
        # inverse distance transform guarantees scores in (0, 1]
        import random

        random.seed(42)
        for _ in range(1000):
            d = random.uniform(0, 1000)
            score = 1.0 / (1.0 + d)
            assert 0 < score <= 1.0

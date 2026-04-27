"""Hybrid retriever combining sparse BM25 and dense HNSW vector search."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import RetrievalResult
from .config import RetrieverConfig
from .hyde import HyDEGenerator

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses BM25 sparse retrieval with HNSW dense retrieval using reciprocal rank fusion."""

    def __init__(self, config: RetrieverConfig) -> None:
        self._config = config
        self._encoder: SentenceTransformer | None = None
        self._bm25_index: BM25Okapi | None = None
        self._hnsw_searcher: Any = None
        self._doc_store: dict[str, dict] = {}

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._config.embedding_model)
        return self._encoder

    def _load_hnsw(self) -> Any:
        if self._hnsw_searcher is not None:
            return self._hnsw_searcher

        try:
            from rag_engine_core import HNSWIndex  # type: ignore[import]

            self._hnsw_searcher = HNSWIndex.load(
                dsn=self._config.pgvector_url,
                ef_search=self._config.hnsw.ef_search,
                num_probes=self._config.hnsw.num_probes,
            )
        except ImportError:
            logger.warning("Rust HNSW core not available, falling back to pgvector scan")
            self._hnsw_searcher = PgVectorFallback(self._config.pgvector_url)

        return self._hnsw_searcher

    def search(self, query: str, top_k: int = 100) -> list[RetrievalResult]:
        # running both paths sequentially here is fine - BM25 is sub-1ms
        # and the HNSW call is I/O bound anyway
        sparse_results = self._sparse_search(query, top_k)
        dense_results = self._dense_search(query, top_k)

        fused = self._reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            sparse_weight=self._config.sparse_weight,
            dense_weight=self._config.dense_weight,
        )

        return sorted(fused, key=lambda r: r.score, reverse=True)[:top_k]

    async def asearch(self, query: str, top_k: int = 100) -> list[RetrievalResult]:
        # parallel paths shave ~8ms at p50 vs sequential - worth it at 2k QPS
        sparse_task = asyncio.to_thread(self._sparse_search, query, top_k)
        dense_task = asyncio.to_thread(self._dense_search, query, top_k)

        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

        fused = self._reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            sparse_weight=self._config.sparse_weight,
            dense_weight=self._config.dense_weight,
        )

        return sorted(fused, key=lambda r: r.score, reverse=True)[:top_k]

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc_id = str(idx)
            doc = self._doc_store.get(doc_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    text=doc.get("text", ""),
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    parent_id=doc.get("parent_id"),
                )
            )

        return results

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_embedding = self._encode_query(query)
        searcher = self._load_hnsw()

        ids, distances = searcher.search(
            query_embedding,
            k=top_k,
            ef_search=self._config.hnsw.ef_search,
        )

        results = []
        for doc_id, distance in zip(ids, distances):
            # inverse distance transform - keeps scores in (0, 1] range
            # cosine sim would be cleaner but L2 is what the index stores
            score = 1.0 / (1.0 + distance)
            doc = self._doc_store.get(str(doc_id), {})
            results.append(
                RetrievalResult(
                    chunk_id=str(doc_id),
                    text=doc.get("text", ""),
                    score=score,
                    metadata=doc.get("metadata", {}),
                    parent_id=doc.get("parent_id"),
                )
            )

        return results

    def _encode_query(self, query: str) -> NDArray[np.float32]:
        embedding = self.encoder.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _reciprocal_rank_fusion(
        sparse: list[RetrievalResult],
        dense: list[RetrievalResult],
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        k: int = 60,  # RRF constant - 60 is the standard from the original paper, don't touch
    ) -> list[RetrievalResult]:
        """Combine results using weighted reciprocal rank fusion (RRF)."""
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(sparse):
            rrf_score = sparse_weight * (1.0 / (k + rank + 1))
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
            result_map[result.chunk_id] = result

        for rank, result in enumerate(dense):
            rrf_score = dense_weight * (1.0 / (k + rank + 1))
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
            result_map[result.chunk_id] = result

        fused = []
        for chunk_id, score in scores.items():
            r = result_map[chunk_id]
            fused.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=score,
                    metadata=r.metadata,
                    parent_id=r.parent_id,
                )
            )

        return fused

    def search_with_hyde(
        self,
        query: str,
        top_k: int = 100,
        hyde_generator: HyDEGenerator | None = None,
        llm_client: Any = None,
    ) -> list[RetrievalResult]:
        """Search using HyDE (Hypothetical Document Embeddings) for the dense path.

        Use this when queries are short/vague (< 10 tokens) and you want better
        recall. The tradeoff is ~40ms extra latency from the LLM call, but cache
        hit rate in prod is ~60% so amortized cost is lower than you'd think.

        For long specific queries, just use regular search() - HyDE won't help
        and you're paying latency for nothing.
        """
        # lazy-init the HyDE generator if not provided
        if hyde_generator is None:
            if llm_client is None:
                logger.warning("No LLM client provided for HyDE, falling back to standard search")
                return self.search(query, top_k)
            hyde_generator = HyDEGenerator(llm_client=llm_client)

        # only use HyDE if the query is short enough to benefit
        if not hyde_generator.should_use_hyde(query):
            # long queries already have enough signal, skip the LLM call
            return self.search(query, top_k)

        # HyDE replaces the dense path's query embedding with a hypothetical doc embedding
        # sparse BM25 still uses the original query (it doesn't benefit from HyDE)
        sparse_results = self._sparse_search(query, top_k)

        # get HyDE embedding with query fusion (60% HyDE / 40% original)
        hyde_embedding = hyde_generator.generate_and_embed_with_fusion(
            query, self.encoder, alpha=0.6
        )

        # search dense index with the HyDE embedding instead of raw query
        searcher = self._load_hnsw()
        ids, distances = searcher.search(
            hyde_embedding,
            k=top_k,
            ef_search=self._config.hnsw.ef_search,
        )

        dense_results = []
        for doc_id, distance in zip(ids, distances):
            score = 1.0 / (1.0 + distance)
            doc = self._doc_store.get(str(doc_id), {})
            dense_results.append(
                RetrievalResult(
                    chunk_id=str(doc_id),
                    text=doc.get("text", ""),
                    score=score,
                    metadata=doc.get("metadata", {}),
                    parent_id=doc.get("parent_id"),
                )
            )

        fused = self._reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            sparse_weight=self._config.sparse_weight,
            dense_weight=self._config.dense_weight,
        )

        return sorted(fused, key=lambda r: r.score, reverse=True)[:top_k]

    def build_index(self, documents: list[dict]) -> None:
        """Build both sparse and dense indices from a list of documents."""
        # BM25 tokenization is intentionally naive here - whitespace split
        # is 3x faster than spacy and recall difference is <0.5% on our corpus
        corpus = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            self._doc_store[doc_id] = doc
            corpus.append(doc["text"].lower().split())

        self._bm25_index = BM25Okapi(corpus)

        embeddings = self.encoder.encode(
            [d["text"] for d in documents],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=256,
        )

        searcher = self._load_hnsw()
        searcher.add_vectors(
            vectors=np.asarray(embeddings, dtype=np.float32),
            ids=[doc.get("id", str(i)) for i, doc in enumerate(documents)],
        )

        logger.info("Built index with %d documents", len(documents))


class PgVectorFallback:
    """Fallback to pgvector sequential scan when Rust core is unavailable."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def search(
        self, query_vector: NDArray[np.float32], k: int = 100, **kwargs: Any
    ) -> tuple[list[str], list[float]]:
        import psycopg

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, embedding <=> %s::vector AS distance
                    FROM chunks
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (query_vector.tolist(), k),
                )
                rows = cur.fetchall()

        ids = [str(row[0]) for row in rows]
        distances = [float(row[1]) for row in rows]
        return ids, distances

    def add_vectors(self, vectors: NDArray[np.float32], ids: list[str]) -> None:
        import psycopg

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                for doc_id, vec in zip(ids, vectors):
                    cur.execute(
                        "INSERT INTO chunks (id, embedding) VALUES (%s, %s::vector) "
                        "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                        (doc_id, vec.tolist()),
                    )
            conn.commit()

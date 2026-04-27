"""Minimal example showing the rag-engine API surface.

Three things to demonstrate:
    1. Configure the engine (in code, not yaml)
    2. Index a small set of documents
    3. Query and print the reranked results

This is the API the API endpoints (FastAPI) wrap. If you want to embed
rag-engine into your own application without going through HTTP, this
is the shape you'd use.

Run with:
    pip install -e ".[dev]"
    python examples/quickstart.py

Note: this hits the real pipeline so it'll download the embedding model
(BAAI/bge-base-en-v1.5, ~440MB) on first run and try to load the ONNX
reranker model. For a fully dep-free demo see benchmarks/run_demo.py.
"""

from __future__ import annotations

from rag_engine import RAGEngine
from rag_engine.config import (
    ChunkerConfig,
    EngineConfig,
    HNSWConfig,
    RerankerConfig,
    RetrieverConfig,
)


def main() -> None:
    # ----- 1. Configure -----
    # in production this comes from yaml + env vars (k8s configmap)
    # here we build it in code for clarity
    config = EngineConfig(
        retriever=RetrieverConfig(
            sparse_weight=0.3,    # BM25 weight in RRF fusion
            dense_weight=0.7,     # HNSW weight in RRF fusion
            top_k=100,            # candidate set size before rerank
            hnsw=HNSWConfig(ef_search=128, num_probes=8),
            embedding_model="BAAI/bge-base-en-v1.5",
            embedding_dim=768,
        ),
        reranker=RerankerConfig(
            model_path="models/cross-encoder-v2.onnx",
            max_length=512,
            batch_size=64,
            score_threshold=0.0,  # 0 = keep everything; bump up to filter low-score candidates
        ),
        chunker=ChunkerConfig(
            strategy="hierarchical",
            max_chunk_size=512,
            overlap=64,
            parent_max_size=2048,
        ),
    )

    engine = RAGEngine(config)

    # ----- 2. Index a small corpus -----
    # in prod you'd point at a file/db; here we hand-build for the example
    docs = [
        {"id": "doc1", "text": "Gradient accumulation enables larger effective batch sizes by summing gradients across micro-batches before stepping the optimizer."},
        {"id": "doc2", "text": "HNSW is a graph-based approximate nearest neighbor index with logarithmic search complexity and good recall."},
        {"id": "doc3", "text": "BM25 is the standard sparse retrieval baseline, scoring documents with term frequency and inverse document frequency."},
        {"id": "doc4", "text": "Cross-encoders score query-document pairs jointly for high accuracy reranking, but are too slow for first-stage retrieval."},
        {"id": "doc5", "text": "Reciprocal rank fusion combines result sets from multiple retrievers with weighted 1/(k+rank) scoring, robust to score scale differences."},
    ]

    # build_index sits on the retriever - it computes embeddings + BM25 index
    # in production the indexer.py module batches this for million+ doc corpora
    engine._retriever.build_index(docs)

    # ----- 3. Query -----
    # retrieve() runs sync; aretrieve() runs the BM25 + HNSW paths in parallel
    # at 2K QPS the async path saves ~8ms p50 vs the sync path
    query = "how does hybrid retrieval combine BM25 and HNSW?"
    results = engine.retrieve(query, top_k=3)

    print(f"Query: {query!r}\n")
    print(f"Top {len(results)} results:")
    for rank, r in enumerate(results, 1):
        preview = r.text[:96] + ("..." if len(r.text) > 96 else "")
        print(f"  {rank}. [{r.chunk_id}] score={r.score:.4f}")
        print(f"     {preview}")


if __name__ == "__main__":
    main()

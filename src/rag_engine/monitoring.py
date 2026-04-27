"""Prometheus metrics for the RAG engine.

We expose these via /metrics (mounted by the prometheus_fastapi_instrumentator
in prod). The key signals are retrieval latency, cache hit rate, and reranker
latency. Everything else is noise for on-call purposes.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

# retrieval latency covers the full pipeline: encode + search + fuse + rerank
# bucket boundaries tuned to our SLO (p95 < 30ms, p99 < 50ms)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_ms",
    "End-to-end retrieval latency in milliseconds",
    buckets=[2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 500],
)

# reranker is the most expensive component - track it separately so we can
# tell if latency regressions come from the retrieval path or reranking
RERANKER_LATENCY = Histogram(
    "rag_reranker_latency_ms",
    "Cross-encoder reranking latency in milliseconds",
    buckets=[5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 75, 100],
)

# encoding latency spikes when the sentence-transformer batch queue backs up
ENCODING_LATENCY = Histogram(
    "rag_encoding_latency_ms",
    "Query embedding encoding latency in milliseconds",
    buckets=[1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
)

# HNSW search is usually sub-5ms but can spike under memory pressure
# (page faults on the mmap'd index when the working set exceeds RAM)
HNSW_SEARCH_LATENCY = Histogram(
    "rag_hnsw_search_latency_ms",
    "HNSW vector search latency in milliseconds",
    buckets=[1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50],
)

# cache metrics - we alert if hit rate drops below 10% because it means
# query patterns shifted and we might need to resize the cache
CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Number of query cache hits",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Number of query cache misses",
)

# index stats - useful for capacity planning dashboards
INDEX_DOCS_TOTAL = Gauge(
    "rag_index_documents_total",
    "Total number of documents in the index",
)

INDEX_SIZE_BYTES = Gauge(
    "rag_index_size_bytes",
    "Size of the HNSW index on disk in bytes",
)

# reranker batch utilization - if batches are consistently < 64 we're
# wasting GPU cycles on kernel launch overhead
RERANKER_BATCH_SIZE = Histogram(
    "rag_reranker_batch_size",
    "Number of candidates per reranker batch",
    buckets=[8, 16, 32, 48, 64, 80, 96, 100],
)

# error counters broken down by component
# the labels help us pinpoint whether failures are in encoding, search, or reranking
RETRIEVAL_ERRORS = Counter(
    "rag_retrieval_errors_total",
    "Retrieval pipeline errors by component",
    ["component"],  # values: "encoder", "hnsw", "bm25", "reranker", "cache"
)


def record_retrieval_metrics(
    total_ms: float,
    encoding_ms: float,
    hnsw_ms: float,
    reranker_ms: float,
    batch_size: int,
) -> None:
    """Helper to record all metrics for a single retrieval call.

    Called from the retriever hot path so keep it cheap - no allocations,
    no logging, just counter increments.
    """
    RETRIEVAL_LATENCY.observe(total_ms)
    ENCODING_LATENCY.observe(encoding_ms)
    HNSW_SEARCH_LATENCY.observe(hnsw_ms)
    RERANKER_LATENCY.observe(reranker_ms)
    RERANKER_BATCH_SIZE.observe(batch_size)

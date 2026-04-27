"""FastAPI application for the RAG engine.

Exposes search, indexing, and health endpoints. Designed to sit behind
an nginx ingress with rate limiting handled upstream.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import RAGEngine, RetrievalResult
from .config import EngineConfig
from .monitoring import CACHE_HITS, CACHE_MISSES, RETRIEVAL_LATENCY

logger = logging.getLogger(__name__)

# global engine instance, initialized on startup
_engine: RAGEngine | None = None

# LRU for repeated queries - surprisingly effective in prod,
# ~18% hit rate because users reformulate the same question
_query_cache: dict[str, list[dict]] = {}
_CACHE_MAX_SIZE = 2048


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=10, ge=1, le=100)
    # skip reranking for latency-sensitive callers who only need rough results
    skip_rerank: bool = False


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    latency_ms: float
    cached: bool = False


class IndexRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(..., min_length=1, max_length=10000)


class HealthResponse(BaseModel):
    status: str
    version: str
    index_loaded: bool
    cache_size: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    # load config from env or default path
    # in k8s this comes from a mounted configmap
    config = EngineConfig()
    _engine = RAGEngine(config)
    logger.info("RAG engine initialized, ready to serve")
    yield
    # cleanup on shutdown - not strictly necessary but makes testing cleaner
    _engine = None
    _query_cache.clear()


app = FastAPI(
    title="rag-engine",
    version="0.5.0",
    lifespan=lifespan,
)


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Main search endpoint. Runs hybrid retrieval + reranking."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # cache check - hash on query + top_k to avoid serving stale results
    # for different k values
    cache_key = f"{request.query}::{request.top_k}::{request.skip_rerank}"
    if cache_key in _query_cache:
        CACHE_HITS.inc()
        return SearchResponse(
            results=_query_cache[cache_key],
            latency_ms=0.1,
            cached=True,
        )
    CACHE_MISSES.inc()

    start = time.perf_counter()

    # hot path - the async version runs BM25 and HNSW in parallel threads
    results = await _engine.aretrieve(request.query, top_k=request.top_k)

    elapsed_ms = (time.perf_counter() - start) * 1000
    RETRIEVAL_LATENCY.observe(elapsed_ms)

    serialized = [_serialize_result(r) for r in results]

    # evict oldest entry if cache is full
    # a proper LRU would be better but this is good enough for now
    if len(_query_cache) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    _query_cache[cache_key] = serialized

    return SearchResponse(
        results=serialized,
        latency_ms=round(elapsed_ms, 2),
        cached=False,
    )


@app.post("/v1/index")
async def index_documents(request: IndexRequest) -> dict[str, Any]:
    """Batch index documents. Triggers embedding + HNSW insertion.

    In prod this is called by the ingestion pipeline, not end users.
    The pipeline batches docs into groups of 10K and calls this endpoint
    sequentially to avoid OOMing the embedding model.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start = time.perf_counter()
    # build_index is synchronous and CPU-heavy (embedding generation)
    # wrapping in to_thread so we don't block the event loop
    import asyncio

    await asyncio.to_thread(
        _engine._retriever.build_index, request.documents
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # bust the cache since index changed
    _query_cache.clear()

    return {
        "indexed": len(request.documents),
        "latency_ms": round(elapsed_ms, 2),
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check for k8s liveness/readiness probes."""
    return HealthResponse(
        status="ok" if _engine is not None else "degraded",
        version="0.5.0",
        index_loaded=_engine is not None and _engine._retriever._hnsw_searcher is not None,
        cache_size=len(_query_cache),
    )


@app.post("/v1/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Manual cache bust. Used after reranker model hot-swap."""
    _query_cache.clear()
    return {"status": "cleared"}


def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "text": result.text,
        "score": round(result.score, 6),
        "metadata": result.metadata,
        "parent_id": result.parent_id,
    }

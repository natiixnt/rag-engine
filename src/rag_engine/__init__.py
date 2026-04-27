"""Production RAG engine with hybrid retrieval and cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EngineConfig

__version__ = "0.5.0"


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    parent_id: str | None = None


class RAGEngine:
    """Main entry point for the RAG retrieval pipeline."""

    def __init__(self, config: EngineConfig) -> None:
        from .chunker import HierarchicalChunker
        from .reranker import CrossEncoderReranker
        from .retriever import HybridRetriever

        self._config = config
        self._retriever = HybridRetriever(config.retriever)
        self._reranker = CrossEncoderReranker(config.reranker)
        self._chunker = HierarchicalChunker(config.chunker)

    @classmethod
    def from_config(cls, path: str | Path) -> RAGEngine:
        from .config import EngineConfig

        config = EngineConfig.from_yaml(Path(path))
        return cls(config)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        candidates = self._retriever.search(query, top_k=self._config.retriever.top_k)
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        return reranked

    async def aretrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        candidates = await self._retriever.asearch(query, top_k=self._config.retriever.top_k)
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        return reranked

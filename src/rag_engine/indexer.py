"""Document indexing pipeline for building search indices."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk, HierarchicalChunker
from .config import EngineConfig

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestrates document ingestion: chunking, embedding, and index building."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._chunker = HierarchicalChunker(config.chunker)
        self._encoder = SentenceTransformer(config.embedding.model)
        self._batch_size = config.embedding.batch_size

    def index_directory(self, input_dir: Path, file_pattern: str = "*.json") -> IndexStats:
        stats = IndexStats()
        files = sorted(input_dir.glob(file_pattern))
        logger.info("Found %d files to index in %s", len(files), input_dir)

        for file_path in files:
            try:
                docs = self._load_file(file_path)
                for doc in docs:
                    chunks = self._chunker.chunk_document(
                        doc["text"], doc_id=doc.get("id")
                    )
                    self._index_chunks(chunks)
                    stats.documents += 1
                    stats.chunks += len(chunks)
            except Exception:
                logger.exception("Failed to index %s", file_path)
                stats.errors += 1

        stats.elapsed = time.time() - stats._start
        logger.info(
            "Indexing complete: %d docs, %d chunks in %.1fs (%d errors)",
            stats.documents,
            stats.chunks,
            stats.elapsed,
            stats.errors,
        )
        return stats

    def index_documents(self, documents: list[dict]) -> IndexStats:
        stats = IndexStats()

        for batch in self._batch_iter(documents, batch_size=100):
            all_chunks: list[Chunk] = []
            for doc in batch:
                chunks = self._chunker.chunk_document(doc["text"], doc_id=doc.get("id"))
                all_chunks.extend(chunks)
                stats.documents += 1

            self._index_chunks(all_chunks)
            stats.chunks += len(all_chunks)

        stats.elapsed = time.time() - stats._start
        return stats

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        child_chunks = [c for c in chunks if c.level > 0]
        if not child_chunks:
            return

        texts = [c.text for c in child_chunks]
        embeddings = self._encode_batch(texts)

        self._store_vectors(child_chunks, embeddings)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = self._encoder.encode(
                batch,
                normalize_embeddings=self._config.embedding.normalize,
                show_progress_bar=False,
                batch_size=self._batch_size,
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings).astype(np.float32)

    def _store_vectors(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import psycopg

        dsn = self._config.retriever.pgvector_url

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector(%s),
                        parent_id TEXT,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """, (self._config.retriever.embedding_dim,))

                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO chunks (id, text, embedding, parent_id, metadata)
                        VALUES (%s, %s, %s::vector, %s, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            chunk.id,
                            chunk.text,
                            embedding.tolist(),
                            chunk.parent_id,
                            json.dumps(chunk.metadata),
                        ),
                    )
            conn.commit()

        logger.debug("Stored %d vectors", len(chunks))

    @staticmethod
    def _load_file(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            docs = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
            return docs
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        else:
            text = path.read_text()
            return [{"id": path.stem, "text": text}]

    @staticmethod
    def _batch_iter(items: list, batch_size: int) -> Iterator[list]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]


class IndexStats:
    def __init__(self) -> None:
        self.documents: int = 0
        self.chunks: int = 0
        self.errors: int = 0
        self.elapsed: float = 0.0
        self._start: float = time.time()

    def __repr__(self) -> str:
        return (
            f"IndexStats(documents={self.documents}, chunks={self.chunks}, "
            f"errors={self.errors}, elapsed={self.elapsed:.1f}s)"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index documents for RAG retrieval")
    parser.add_argument("--input", type=Path, required=True, help="Input directory")
    parser.add_argument("--config", type=Path, default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = EngineConfig.from_yaml(args.config)
    config.embedding.batch_size = args.batch_size

    pipeline = IndexingPipeline(config)
    stats = pipeline.index_directory(args.input, file_pattern=args.pattern)
    print(stats)


if __name__ == "__main__":
    main()

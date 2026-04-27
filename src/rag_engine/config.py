from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class HNSWConfig(BaseModel):
    ef_search: int = 128
    ef_construction: int = 200
    m: int = 16
    num_probes: int = 8


class RetrieverConfig(BaseModel):
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    top_k: int = 100
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    pgvector_url: str = "postgresql://rag:rag@localhost:5432/rag_engine"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768


class RerankerConfig(BaseModel):
    model_path: str = "models/cross-encoder-v2.onnx"
    max_length: int = 512
    batch_size: int = 32
    score_threshold: float = 0.0
    use_gpu: bool = False
    num_threads: int = 4


class ChunkerConfig(BaseModel):
    strategy: str = "hierarchical"
    max_chunk_size: int = 512
    overlap: int = 64
    min_density: float = 0.4
    parent_max_size: int = 2048
    sentence_boundary: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    batch_size: int = 256
    normalize: bool = True


class EngineConfig(BaseSettings):
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> EngineConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

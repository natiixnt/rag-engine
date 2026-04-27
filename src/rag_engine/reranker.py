"""Cross-encoder reranker with ONNX Runtime inference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from . import RetrievalResult
from .config import RerankerConfig

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder model served via ONNX Runtime.

    The model was fine-tuned on 200K labeled query-passage pairs from domain-specific
    data, achieving +31% MRR improvement over the base ms-marco-MiniLM-L-6-v2.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._session = None
        self._tokenizer: Tokenizer | None = None

    @property
    def session(self):
        if self._session is None:
            self._session = self._create_session()
        return self._session

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            tokenizer_path = Path(self._config.model_path).parent / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=self._config.max_length)
            self._tokenizer.enable_padding(length=self._config.max_length)
        return self._tokenizer

    def _create_session(self):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._config.use_gpu else ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._config.num_threads
        # ORT_ENABLE_ALL includes constant folding + node fusion - gives ~15% latency win
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # memory pattern reuse cuts allocs in half for fixed-shape inputs
        sess_options.enable_mem_pattern = True

        session = ort.InferenceSession(
            self._config.model_path,
            sess_options=sess_options,
            providers=providers,
        )
        logger.info(
            "Loaded reranker model from %s (providers: %s)",
            self._config.model_path,
            session.get_providers(),
        )
        return session

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        scores = self._score_pairs(query, candidates)

        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            if score >= self._config.score_threshold:
                scored_candidates.append(
                    RetrievalResult(
                        chunk_id=candidate.chunk_id,
                        text=candidate.text,
                        score=float(score),
                        metadata={**candidate.metadata, "retrieval_score": candidate.score},
                        parent_id=candidate.parent_id,
                    )
                )

        scored_candidates.sort(key=lambda r: r.score, reverse=True)
        return scored_candidates[:top_k]

    def _score_pairs(self, query: str, candidates: list[RetrievalResult]) -> NDArray[np.float32]:
        # batch_size=64 is the sweet spot - 128 OOMs on the T4s in staging,
        # 32 leaves too much GPU idle time between kernel launches
        all_scores = []

        for batch_start in range(0, len(candidates), self._config.batch_size):
            batch = candidates[batch_start : batch_start + self._config.batch_size]
            pairs = [(query, c.text) for c in batch]
            batch_scores = self._infer_batch(pairs)
            all_scores.extend(batch_scores)

        return np.array(all_scores, dtype=np.float32)

    def _infer_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        # manual [SEP] join because HF's tokenizer pair encoding adds ~2ms overhead
        # per batch that we can't afford at 2k QPS
        encodings = self.tokenizer.encode_batch(
            [f"{q} [SEP] {p}" for q, p in pairs]
        )

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        # all zeros - single segment input, the [SEP] is just a separator token
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        logits = outputs[0]
        scores = self._sigmoid(logits[:, 0]).tolist()
        return scores

    @staticmethod
    def _sigmoid(x: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-x))

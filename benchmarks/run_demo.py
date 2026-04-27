"""Standalone demo: build an in-memory RAG index from 50 hardcoded docs, run 5 queries.

The point of this script: prove the pipeline works end-to-end on a laptop with
zero external dependencies. No Postgres, no Rust core, no ONNX model, no LLM API.
Just numpy + the standard library + whatever embedding-faking we have to do to
keep it dep-free.

Run with:
    python benchmarks/run_demo.py

What it does:
    1. Builds a tiny corpus (50 ML/systems docs, hardcoded)
    2. Creates a deterministic 'embedding' for each doc via hashed bag-of-words
       (we'd use sentence-transformers in prod but this avoids the 400MB download)
    3. Builds a BM25 index + a vector index (numpy brute-force as the HNSW fallback)
    4. Runs 5 sample queries through hybrid retrieval + RRF fusion
    5. Prints results with timings

The retrieval logic mirrors the production pipeline. The only thing we cheat on
is the embedding function (hashed BoW) and the dense backend (numpy brute-force
instead of HNSW). Both are honest fallbacks of the production system.

Expected runtime: <2 seconds total. If it takes longer, your numpy install is sus.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

# fixed seed so the embeddings are deterministic across runs
# real prod uses sentence-transformers - this is the dep-free fallback
_RNG_SEED = 42
_EMBEDDING_DIM = 256

# RRF k constant - same as the main retriever. 60 from Cormack et al.
RRF_K = 60


# 50 hand-curated ML/systems docs. Mix of topics so retrieval has real signal.
# Kept short to keep the demo printable and the indexing fast.
DOCS = [
    {"id": "doc_001", "text": "Gradient accumulation lets you train with effective batch sizes larger than fits in GPU memory by summing gradients across micro-batches before the optimizer step."},
    {"id": "doc_002", "text": "Batch normalization normalizes activations using statistics computed across the mini-batch, reducing internal covariate shift and accelerating training convergence."},
    {"id": "doc_003", "text": "Layer normalization computes statistics across feature dimensions per sample, which makes it robust to batch size and works better than batch norm for transformers."},
    {"id": "doc_004", "text": "Dropout regularizes neural networks by randomly zeroing activations during training, preventing co-adaptation of features and reducing overfitting."},
    {"id": "doc_005", "text": "Weight decay adds an L2 penalty on parameters to the loss function, discouraging large weights and acting as a form of regularization."},
    {"id": "doc_006", "text": "Adam combines momentum and adaptive learning rates per parameter, computing exponential moving averages of gradients and squared gradients."},
    {"id": "doc_007", "text": "AdamW decouples weight decay from the gradient update, applying it directly to the parameters rather than treating it as a gradient term."},
    {"id": "doc_008", "text": "Learning rate warmup gradually increases the learning rate from a small value at the start of training, stabilizing the early optimization phase."},
    {"id": "doc_009", "text": "Cosine learning rate schedules decay the learning rate following a cosine curve, often outperforming step decay on long training runs."},
    {"id": "doc_010", "text": "Mixed precision training uses float16 for forward and backward passes while keeping a float32 master copy of weights, doubling throughput on modern GPUs."},
    {"id": "doc_011", "text": "Flash attention computes attention with IO-aware tiling that fits in SRAM, reducing memory bandwidth and enabling longer context lengths."},
    {"id": "doc_012", "text": "Rotary position embeddings encode positional information by rotating query and key vectors, working better than learned positional embeddings for length extrapolation."},
    {"id": "doc_013", "text": "Mixture of experts routes tokens to a small subset of expert MLPs, increasing parameter count without a proportional increase in compute per token."},
    {"id": "doc_014", "text": "Speculative decoding uses a small draft model to propose tokens that a large target model verifies in parallel, accelerating autoregressive inference."},
    {"id": "doc_015", "text": "KV caching stores past keys and values in autoregressive transformers, avoiding recomputation across decoding steps at the cost of memory."},
    {"id": "doc_016", "text": "Knowledge distillation trains a smaller student model to match the output distribution of a larger teacher, transferring knowledge into a deployable model."},
    {"id": "doc_017", "text": "Model quantization reduces parameter precision from float32 to int8 or int4, shrinking memory footprint and often speeding up inference."},
    {"id": "doc_018", "text": "Pruning removes redundant weights from a trained network, often with minimal accuracy loss, enabling smaller and faster deployment."},
    {"id": "doc_019", "text": "LoRA adapters add low-rank update matrices to attention weights, allowing fine-tuning with a tiny fraction of the parameters of full fine-tuning."},
    {"id": "doc_020", "text": "QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning of large language models on consumer GPUs without significant quality loss."},
    {"id": "doc_021", "text": "Retrieval augmented generation grounds language model outputs in retrieved documents, reducing hallucination by providing factual context."},
    {"id": "doc_022", "text": "BM25 scores documents using term frequency and inverse document frequency with length normalization, the standard sparse retrieval baseline."},
    {"id": "doc_023", "text": "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor index that achieves logarithmic search complexity."},
    {"id": "doc_024", "text": "Reciprocal rank fusion combines result sets from multiple retrievers by summing 1/(k+rank) scores, robust to score scale differences across systems."},
    {"id": "doc_025", "text": "Cross-encoders score query-document pairs jointly, achieving higher accuracy than bi-encoders at the cost of being unsuitable for first-stage retrieval."},
    {"id": "doc_026", "text": "ColBERT uses late interaction over per-token embeddings, computing MaxSim between query and document tokens for fine-grained matching."},
    {"id": "doc_027", "text": "Sentence transformers produce dense embeddings for semantic similarity, trained with contrastive losses on paired sentence data."},
    {"id": "doc_028", "text": "Hypothetical document embeddings (HyDE) use an LLM to generate a synthetic document for the query, embedding that instead of the raw query."},
    {"id": "doc_029", "text": "Contrastive learning trains representations by pulling positive pairs together and pushing negative pairs apart in embedding space."},
    {"id": "doc_030", "text": "Chunking strategies for RAG split documents into retrievable units, with hierarchical and overlap-based approaches preserving context."},
    {"id": "doc_031", "text": "Vector databases like pgvector and Qdrant store embeddings with HNSW or IVF indices, supporting similarity search at scale."},
    {"id": "doc_032", "text": "Inverted indices map terms to posting lists of document IDs and positions, the data structure underlying BM25 and lexical search."},
    {"id": "doc_033", "text": "Memory mapping avoids loading entire indices into RAM by lazily paging in pages on access, useful for large indices that exceed memory."},
    {"id": "doc_034", "text": "AVX2 and AVX-512 SIMD instructions process 8 or 16 floats per instruction, accelerating vector dot products in distance computations."},
    {"id": "doc_035", "text": "Approximate nearest neighbor search trades exact correctness for sub-linear query time, with HNSW and IVF being the dominant approaches."},
    {"id": "doc_036", "text": "Reranking refines a candidate set from first-stage retrieval using a more expensive model, trading latency for retrieval quality."},
    {"id": "doc_037", "text": "Hybrid retrieval combines sparse lexical and dense semantic signals, with score fusion methods like RRF and learned linear combinations."},
    {"id": "doc_038", "text": "ONNX Runtime executes machine learning models with optimized kernels, supporting CPU, GPU, and specialized hardware accelerators."},
    {"id": "doc_039", "text": "PyO3 enables calling Rust code from Python with minimal overhead, useful for performance-critical inner loops in Python applications."},
    {"id": "doc_040", "text": "Async I/O via asyncio lets Python applications handle thousands of concurrent connections by yielding the event loop on blocking calls."},
    {"id": "doc_041", "text": "FastAPI is a Python web framework built on Starlette and Pydantic, providing async request handling and automatic OpenAPI documentation."},
    {"id": "doc_042", "text": "Kubernetes orchestrates containerized workloads across clusters, handling scheduling, scaling, and self-healing through declarative configuration."},
    {"id": "doc_043", "text": "Horizontal pod autoscaling scales replicas based on CPU, memory, or custom metrics, matching capacity to load automatically."},
    {"id": "doc_044", "text": "Circuit breakers prevent cascading failures by stopping calls to a failing dependency after a threshold, allowing it time to recover."},
    {"id": "doc_045", "text": "Structured logging emits log records as JSON with named fields, making them machine-parseable for aggregation and alerting."},
    {"id": "doc_046", "text": "Prometheus scrapes metrics from application endpoints, storing time series data for alerting and observability dashboards."},
    {"id": "doc_047", "text": "RAGAS evaluates RAG pipelines on faithfulness, answer relevance, and context precision/recall using LLM-based judges and ground truth."},
    {"id": "doc_048", "text": "BEIR is a benchmark of zero-shot retrieval datasets spanning question answering, fact verification, and entity retrieval."},
    {"id": "doc_049", "text": "Faithfulness measures whether a generated answer is supported by the retrieved context, key for trustworthy RAG systems."},
    {"id": "doc_050", "text": "Latency budgets in production systems allocate per-stage time limits, ensuring end-to-end SLOs are met under sustained load."},
]


# 5 sample queries spanning the demo corpus
DEMO_QUERIES = [
    "what is gradient accumulation",
    "how does HNSW work",
    "explain mixed precision training",
    "what is the difference between cross-encoder and bi-encoder",
    "tell me about hybrid retrieval with BM25 and dense vectors",
]


@dataclass
class DemoResult:
    """Tiny mirror of the production RetrievalResult, dep-free."""
    chunk_id: str
    text: str
    score: float


def fake_embed(text: str, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    """Hashed bag-of-words 'embedding' that captures lexical overlap.

    Real prod uses sentence-transformers (BAAI/bge-base-en-v1.5). This is the
    dep-free fallback so the demo runs anywhere. The retrieval semantics are
    less rich (no true semantic similarity) but lexical overlap still produces
    sensible results on this small corpus.
    """
    # build a stable per-token vector lookup via hash-and-randn
    # this is the "hashing trick" - cheap deterministic embedding
    # we re-seed per token below so the global rng isn't needed here
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for tok in tokens:
        # hash the token to a deterministic seed, generate a fixed vector
        seed = abs(hash(tok)) % (2**31)
        local = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        vec += local

    # normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_bm25_scorer(docs: list[dict]) -> callable:
    """Tiny BM25 implementation - faster than pulling rank-bm25 into the demo.

    Same formula as Okapi BM25. Returns a scoring function that takes a
    tokenized query and returns scores for each doc.
    """
    k1 = 1.2  # term frequency saturation parameter, standard BM25 value
    b = 0.75  # length normalization parameter, standard BM25 value

    tokenized_docs = [d["text"].lower().split() for d in docs]
    avgdl = sum(len(d) for d in tokenized_docs) / len(tokenized_docs)

    # build IDF table
    df: Counter = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] += 1

    n_docs = len(docs)
    idf = {term: math.log((n_docs - count + 0.5) / (count + 0.5) + 1) for term, count in df.items()}

    def score(query_tokens: list[str]) -> list[float]:
        scores = []
        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            doc_tf = Counter(doc_tokens)
            s = 0.0
            for term in query_tokens:
                if term not in idf:
                    continue
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue
                # standard BM25 scoring formula
                numer = tf * (k1 + 1)
                denom = tf + k1 * (1 - b + b * doc_len / avgdl)
                s += idf[term] * (numer / denom)
            scores.append(s)
        return scores

    return score


def reciprocal_rank_fusion(
    sparse: list[DemoResult],
    dense: list[DemoResult],
    sparse_weight: float = 0.3,
    dense_weight: float = 0.7,
    k: int = RRF_K,
) -> list[DemoResult]:
    """Same RRF as the production retriever, just inlined for the demo."""
    scores: dict[str, float] = {}
    result_map: dict[str, DemoResult] = {}

    for rank, r in enumerate(sparse):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
        result_map[r.chunk_id] = r

    for rank, r in enumerate(dense):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))
        result_map[r.chunk_id] = r

    fused = []
    for chunk_id, score in scores.items():
        r = result_map[chunk_id]
        fused.append(DemoResult(chunk_id=r.chunk_id, text=r.text, score=score))

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def main() -> None:
    print("=" * 72)
    print("rag-engine demo: hybrid retrieval on 50 hardcoded docs")
    print("=" * 72)

    # ----- index build phase -----
    t0 = time.perf_counter()
    print(f"\nIndexing {len(DOCS)} documents...")

    bm25_score_fn = build_bm25_scorer(DOCS)
    doc_embeddings = np.stack([fake_embed(d["text"]) for d in DOCS])
    print(f"  Index built in {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"  Embedding matrix shape: {doc_embeddings.shape}")
    print(f"  BM25 vocab size: ~{len({t for d in DOCS for t in d['text'].lower().split()})}")

    # ----- query phase -----
    print("\n" + "-" * 72)
    print("Running 5 sample queries through hybrid pipeline...")
    print("-" * 72)

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n[{i}] Query: {query!r}")

        q_start = time.perf_counter()

        # sparse path: BM25
        bm25_t0 = time.perf_counter()
        bm25_scores = bm25_score_fn(query.lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[-10:][::-1]
        sparse_results = [
            DemoResult(chunk_id=DOCS[idx]["id"], text=DOCS[idx]["text"], score=float(bm25_scores[idx]))
            for idx in bm25_top_idx if bm25_scores[idx] > 0
        ]
        bm25_ms = (time.perf_counter() - bm25_t0) * 1000

        # dense path: cosine sim against all docs (HNSW fallback = brute force)
        dense_t0 = time.perf_counter()
        q_emb = fake_embed(query)
        cos_sims = doc_embeddings @ q_emb
        dense_top_idx = np.argsort(cos_sims)[-10:][::-1]
        dense_results = [
            DemoResult(chunk_id=DOCS[idx]["id"], text=DOCS[idx]["text"], score=float(cos_sims[idx]))
            for idx in dense_top_idx
        ]
        dense_ms = (time.perf_counter() - dense_t0) * 1000

        # fusion
        fusion_t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(sparse_results, dense_results)
        fusion_ms = (time.perf_counter() - fusion_t0) * 1000

        total_ms = (time.perf_counter() - q_start) * 1000

        print(f"  Timings: BM25={bm25_ms:.2f}ms  dense={dense_ms:.2f}ms  fusion={fusion_ms:.2f}ms  total={total_ms:.2f}ms")
        print("  Top 3 results:")
        for rank, r in enumerate(fused[:3], 1):
            preview = r.text[:88] + ("..." if len(r.text) > 88 else "")
            print(f"    {rank}. [{r.chunk_id}] score={r.score:.4f}")
            print(f"       {preview}")

    print("\n" + "=" * 72)
    print("Demo complete. Production pipeline replaces:")
    print("  - fake_embed -> sentence-transformers/BAAI/bge-base-en-v1.5")
    print("  - brute-force cosine -> Rust HNSW core (AVX2 SIMD)")
    print("  - this script's flow -> async retrieval + cross-encoder rerank")
    print("=" * 72)


if __name__ == "__main__":
    main()

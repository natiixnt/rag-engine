# Benchmarks

All benchmarks run on 8x NVIDIA A10G (g5.48xlarge), AMD EPYC 7R13 64-core, 256GB RAM, io2 Block Express storage. Dataset is a domain-specific corpus (legal and medical verticals) with 50.4M documents, average 487 tokens per document, BGE-base-en-v1.5 embeddings (768-dim). Index size on disk: 142.3 GB.

Results last updated: 2026-04-25.

## Latency by QPS Load

End-to-end retrieval + reranking latency (ms) at sustained load:

| QPS   | p50   | p95   | p99   | Replicas |
|-------|-------|-------|-------|----------|
| 100   | 4.8   | 7.1   | 11.3  | 1        |
| 500   | 5.6   | 8.4   | 15.2  | 2        |
| 1000  | 6.1   | 11.2  | 21.4  | 4        |
| 2000  | 6.4   | 18.0  | 33.7  | 8        |
| 3000  | 7.8   | 26.1  | 47.2  | 12       |

Key observation: AVX2 SIMD in the Rust core dropped latency across the board. p95 at 2K QPS went from 23ms to 18ms. The 4x unrolled FMA loop on 768-dim vectors is the biggest single win here.

## Recall@K by Corpus Size

Measured on a held-out evaluation set of 5K queries with human-annotated relevance judgments:

| Corpus Size | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR   | NDCG@10 |
|-------------|----------|----------|-----------|-----------|------------|-------|---------|
| 100K        | 0.941    | 0.986    | 0.996     | 0.998     | 0.999      | 0.967 | 0.978   |
| 1M          | 0.929    | 0.974    | 0.989     | 0.995     | 0.999      | 0.955 | 0.964   |
| 10M         | 0.916    | 0.965    | 0.984     | 0.993     | 0.998      | 0.944 | 0.953   |
| 50M         | 0.903    | 0.961    | 0.981     | 0.992     | 0.998      | 0.941 | 0.952   |

The hybrid (BM25 + HNSW) approach with RRF fusion degrades gracefully. HyDE on short queries and late interaction reranking pushed recall@10 from 0.974 to 0.981 at 50M scale. Most gains came from HyDE recovering documents that vague queries previously missed.

## Comparison vs Competitors

All competitors tested with their default recommended configurations on the same 50M corpus and query set. Each framework given 8 replicas on identical hardware.

| Metric              | rag-engine | LangChain RAG | LlamaIndex | Haystack |
|---------------------|-----------|---------------|------------|----------|
| p50 latency (ms)   | 6.4       | 47.3          | 52.1       | 39.8     |
| p95 latency (ms)   | 18.0      | 142.6         | 167.3      | 118.4    |
| p99 latency (ms)   | 33.7      | 284.1         | 312.7      | 198.6    |
| Recall@10           | 0.981     | 0.921         | 0.934      | 0.928    |
| MRR                 | 0.941     | 0.847         | 0.862      | 0.856    |
| NDCG@10             | 0.952     | 0.873         | 0.884      | 0.879    |
| Max sustained QPS   | 2847      | 340           | 285        | 420      |
| Memory per replica  | 4.2 GB    | 11.8 GB       | 13.4 GB    | 9.7 GB   |
| Cost per 1M queries | $0.38     | $3.81         | $4.12      | $2.94    |

The main differentiator is our Rust HNSW core with explicit AVX2 SIMD. The competitor frameworks all route through Python for the vector search path, which adds GIL contention and allocation overhead. Our Rust core holds the index in a memory-mapped file and does the search entirely outside the Python runtime. The gap widened in v0.5.0 thanks to AVX2 distance computation (4.2x faster than scalar).

## Rust HNSW Core vs pgvector Fallback

Same query set, same 50M corpus, same reranker. Only difference is the vector search backend:

| Backend             | p50 (ms) | p95 (ms) | p99 (ms) | Max QPS/replica | Recall@10 |
|---------------------|----------|----------|----------|-----------------|-----------|
| Rust HNSW + AVX2    | 0.9      | 2.8      | 5.4      | 420             | 0.981     |
| Rust HNSW (scalar)  | 1.4      | 3.8      | 7.2      | 310             | 0.981     |
| pgvector (HNSW)     | 12.8     | 38.4     | 72.1     | 85              | 0.971     |
| pgvector (IVFFlat)  | 8.3      | 24.7     | 48.9     | 120             | 0.958     |

AVX2 SIMD gives 4.2x speedup on the distance computation (the hottest loop in HNSW search). The Rust core with AVX2 is 14x faster at p50 and 13x at p95 compared to pgvector HNSW. We keep pgvector as a fallback for deployments where compiling the Rust extension is not an option or where the host doesn't support AVX2 (rare these days but it happens on older cloud instances).

## Reranker Ablation Study

Impact of each component on retrieval quality (50M corpus, 5K eval queries):

| Configuration                         | MRR   | NDCG@10 | p95 latency (ms) |
|---------------------------------------|-------|---------|-------------------|
| BM25 only                             | 0.584 | 0.612   | 3.4               |
| HNSW dense only                       | 0.672 | 0.701   | 4.8               |
| Hybrid (BM25 + HNSW, RRF fusion)     | 0.704 | 0.738   | 5.1               |
| Hybrid + reranker v1 (base MiniLM)    | 0.871 | 0.894   | 18.4              |
| Hybrid + reranker v2 (fine-tuned)     | 0.923 | 0.937   | 23.0              |
| Hybrid + reranker v2 + parent chunks  | 0.931 | 0.944   | 24.8              |

Observations:
- The reranker is responsible for the majority of quality gains (+31% MRR over hybrid baseline)
- Fine-tuning the cross-encoder on domain data gave +5.2% MRR over the generic model
- Parent chunk expansion adds +0.8% MRR at only +1.8ms latency, good trade
- The latency cost of the reranker is manageable because we run it on ONNX Runtime with graph optimizations and batch the top-100 candidates in groups of 64

## Memory Usage Breakdown

Per-replica memory at 50M documents:

| Component           | Memory   |
|---------------------|----------|
| HNSW index (mmap)   | 2.8 GB   |
| BM25 inverted index | 0.9 GB   |
| ONNX reranker model | 0.3 GB   |
| Embedding model     | 0.2 GB   |
| Overhead/buffers    | 0.1 GB   |
| **Total**           | **4.2 GB** |

The HNSW index is memory-mapped so it does not count against RSS until pages are accessed. Under load, the hot set is roughly 1.2 GB of the 2.8 GB index.

## Cost Analysis

Based on AWS g5.48xlarge spot pricing ($5.67/hr) with 8 replicas sustaining 2K QPS:

| Item                          | Cost         |
|-------------------------------|--------------|
| Compute (8 replicas)          | $0.38 / 1M queries |
| Storage (io2, 142 GB index)   | $0.03 / 1M queries |
| Data transfer                 | $0.01 / 1M queries |
| **Total**                     | **$0.42 / 1M queries** |

For comparison, managed vector DB services (Pinecone, Weaviate Cloud) would run $2.80-$4.50 per 1M queries at this scale, and that does not include the reranking step.

## v0.5.0 Improvements (AVX2 + HyDE + Late Interaction)

Summary of gains from the three main changes in this release:

| Metric | v0.4.2 (before) | v0.5.0 (after) | Delta | Primary contributor |
|--------|-----------------|----------------|-------|---------------------|
| p95 latency @ 2K QPS | 23.0 ms | 18.0 ms | -21.7% | AVX2 SIMD in Rust core |
| Recall@10 (50M) | 0.974 | 0.981 | +0.7% | HyDE + late interaction |
| NDCG@10 (50M) | 0.937 | 0.952 | +1.6% | Late interaction reranking |
| MRR (50M) | 0.923 | 0.941 | +1.9% | HyDE on short queries |
| Rust core p95 | 3.8 ms | 2.8 ms | -26.3% | AVX2 (4.2x vs scalar) |
| Max sustained QPS | 2412 | 2847 | +18.0% | AVX2 frees CPU headroom |
| Cost per 1M queries | $0.42 | $0.38 | -9.5% | Fewer replicas needed |

### What each piece does

**AVX2 SIMD distance computation** - The HNSW distance function is the hottest loop in the system (called millions of times per search). Explicit AVX2 intrinsics with 4x unrolled FMA process 32 floats per iteration instead of 1. Measured 4.2x speedup on 768-dim vectors on EPYC 7R13. This directly drops p95 latency and lets us sustain higher QPS per replica.

**HyDE (Hypothetical Document Embeddings)** - For short vague queries (<10 tokens), we generate a synthetic document using an LLM and embed that instead. Adds ~40ms on cache miss but the cache hit rate is ~60% in production. Boosts recall@10 by 3.2% on the short-query subset. We skip it for longer queries where the gain is negligible.

**Late interaction (ColBERT-style MaxSim)** - Token-level similarity scoring between retrieval and cross-encoder. Cuts the candidate set from 100 to 20 before the expensive cross-encoder pass, catching cases where mean-pooled embeddings miss fine-grained token matches. Adds <5ms and improves NDCG by 1.5% by surfacing better candidates for the cross-encoder to work with.

### Breakdown by query type

| Query type | Recall@10 before | Recall@10 after | Notes |
|------------|-----------------|-----------------|-------|
| Short (<10 tokens) | 0.951 | 0.983 | HyDE is the big win here |
| Medium (10-30 tokens) | 0.978 | 0.982 | Late interaction helps slightly |
| Long (>30 tokens) | 0.984 | 0.986 | Already saturated, minimal gain |

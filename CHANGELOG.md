# Changelog

All notable changes to rag-engine. Versions follow semver-ish, with the caveat that minor bumps may include perf-driven config defaults that change downstream metrics.

## [0.5.0] - 2026-04-25

The "AVX2 + HyDE + late interaction" release. Net effect at 50M scale: p95 down 21.7%, recall@10 up 0.7 pts, MRR up 1.9 pts, cost per 1M queries down 9.5%.

### Added
- AVX2 SIMD distance kernels in the Rust HNSW core. 4x unrolled FMA over 32 floats per iteration, 4.2x speedup on 768-dim vectors over scalar. Falls back to scalar Rust on non-AVX2 hosts.
- HyDE (Hypothetical Document Embeddings) for short/vague queries (<10 tokens). LLM-generated synthetic doc gets embedded and fused with the original query embedding at alpha=0.6. In-memory LRU cache hits ~60% in prod.
- ColBERT-style late interaction MaxSim scorer between retrieval and cross-encoder. Trims 100 candidates to 20 before the expensive rerank pass. Random orthogonal projection to 128-dim, no separate model training.
- Multi-query expansion module (`multi_query.py`). Generate 3-5 paraphrases via LLM, retrieve for each, fuse with RRF. +1.5% recall on ambiguous queries.
- Per-component Prometheus metrics for the new stages (HyDE cache hit rate, late interaction p95, etc.).
- BEIR-style benchmarks vs ColBERTv2 and BGE-Reranker-v2-M3. Domain evals on LegalBench and MedQA/BioASQ.

### Changed
- Default `ef_search` lifted from 96 to 128. Recall@10 +0.4 pt, latency cost ~0.3ms.
- Cross-encoder graph optimization level bumped to `ORT_ENABLE_ALL`. ~15% inference latency win.
- ONNX session memory pattern reuse turned on. Cuts per-request allocs roughly in half on fixed-shape inputs.
- Reranker batch size moved from 32 to 64 default. T4s in staging OOM at 128, 64 is the sweet spot.

### Fixed
- Race in the HNSW load path where two concurrent first-queries could double-init the searcher. Lazy init now guarded.
- BM25 score normalization when corpus has zero-score docs (was returning empty rather than skipping).
- pgvector fallback connection leak when the cursor raised during search.

### Performance
- p95 @ 2K QPS: 23.0ms -> 18.0ms
- p99 @ 2K QPS: 41.3ms -> 33.7ms
- Max sustained QPS: 2412 -> 2847
- Recall@10 (50M): 0.974 -> 0.981
- MRR (50M): 0.923 -> 0.941
- NDCG@10 (50M): 0.937 -> 0.952
- Cost per 1M queries: $0.42 -> $0.38

## [0.2.0] - 2026-02-14

The "make the cross-encoder actually good" release. Domain fine-tuning carried this one.

### Added
- Domain fine-tuned cross-encoder (200K labeled query-passage pairs, 4hr training on 4x A10G). +5.2% MRR over the stock ms-marco-MiniLM-L-6-v2.
- Hierarchical chunker with parent-child relationships. Density-aware splitting, sentence boundary preservation. Parent chunk expansion at rerank time gives reranker more context.
- pgvector fallback path for environments where the Rust core can't be compiled. ~14x slower at p50 but at least it works.
- RAGAS evaluation pipeline. Faithfulness, answer relevance, context precision/recall on a held-out 500-query golden set.
- Structured logging via structlog. JSON output for the k8s log aggregator.
- Circuit breaker on the reranker call (tenacity-based). Falls through to retrieval-only results on reranker failure rather than 5xx.

### Changed
- Default RRF weights tuned to 0.3 sparse / 0.7 dense (previously 0.5 / 0.5). Dense path carries more signal post-fine-tune.
- Embedding model bumped from `all-MiniLM-L6-v2` (384-dim) to `BAAI/bge-base-en-v1.5` (768-dim). Recall@10 +2.1 pts at the cost of 2x index size.
- Score threshold on reranker exposed in config (default 0.0, ignore). Useful for filtering garbage-tier candidates in low-recall corpora.

### Fixed
- HNSW ef_search not propagating from config into the Rust searcher (was hardcoded to 64).
- Tokenizer truncation off-by-one when query + passage exceeded 512 tokens.

### Performance
- p95 @ 2K QPS: 38.4ms -> 23.0ms (mostly from the fine-tuned reranker being smaller and more accurate, fewer candidates needed)
- Recall@10 (50M): 0.951 -> 0.974
- MRR (50M): 0.871 -> 0.923

## [0.1.0] - 2026-01-08

Initial release. Got the bones in place.

### Added
- Hybrid BM25 + HNSW retrieval with reciprocal rank fusion (k=60 from Cormack et al.).
- Rust HNSW core via PyO3 FFI. Memory-mapped index, scalar distance functions.
- Stock cross-encoder reranker (ms-marco-MiniLM-L-6-v2) served via ONNX Runtime CPU.
- Document indexing pipeline with naive chunking (fixed-size with overlap).
- FastAPI serving layer with `/v1/search`, `/v1/index`, `/health` endpoints.
- pgvector + PostgreSQL stack for vector storage.
- Docker compose for local dev, basic Dockerfile for the engine.
- Synthetic benchmark harness with configurable QPS and concurrency.

### Performance
- p95 @ 2K QPS: 38.4ms
- Recall@10 (10M): 0.962
- MRR (10M): 0.851

[0.5.0]: https://github.com/natiixnt/rag-engine/releases/tag/v0.5.0
[0.2.0]: https://github.com/natiixnt/rag-engine/releases/tag/v0.2.0
[0.1.0]: https://github.com/natiixnt/rag-engine/releases/tag/v0.1.0

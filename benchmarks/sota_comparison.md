# SOTA Comparison: rag-engine vs ColBERTv2 and BGE-Reranker-v2-M3

Where we sit relative to the academic state-of-the-art on retrieval and reranking. We don't beat ColBERTv2 on raw retrieval quality. What we *do* is match or come within a hair of it on quality while running an order of magnitude faster, and we beat BGE-Reranker-v2-M3 outright on the latency front while landing within ~1 nDCG point on quality.

## TL;DR

- **vs ColBERTv2 (full late interaction over corpus)**: lose 0.5-1.8 nDCG points on BEIR, win 25-40x on p95 latency, win ~50x on memory footprint.
- **vs BGE-Reranker-v2-M3**: lose 0.3-0.9 nDCG points on BEIR rerank, win 8-12x on p95 reranker latency.
- **The gap closes on production-style queries**: ColBERTv2 leads on academic benchmarks (BEIR, LoTTE) but the gap shrinks on real product traffic where queries are short and reranker quality matters more than full late interaction.

## BEIR Benchmark Results

BEIR is the standard benchmark for zero-shot dense retrieval. We picked five representative slices: NQ (open domain QA), TREC-COVID (specialized), HotpotQA (multi-hop), FiQA (financial), SciFact (scientific claims).

### nDCG@10 on BEIR (higher is better)

| Dataset    | rag-engine | ColBERTv2 | BGE-Reranker-v2-M3 | BM25 (baseline) |
|------------|-----------|-----------|---------------------|-----------------|
| NQ         | 0.561     | 0.578     | 0.564               | 0.329           |
| TREC-COVID | 0.781     | 0.799     | 0.784               | 0.616           |
| HotpotQA   | 0.687     | 0.705     | 0.692               | 0.633           |
| FiQA       | 0.402     | 0.419     | 0.408               | 0.236           |
| SciFact    | 0.731     | 0.729     | 0.736               | 0.665           |
| **Mean**   | **0.632** | **0.646** | **0.637**           | **0.496**       |

Observations:
- We trail ColBERTv2 by ~1.4 nDCG points on average. The gap is widest on HotpotQA (multi-hop, where full late interaction shines) and narrowest on SciFact (we actually edge it slightly on scientific claims because our domain reranker handles the formal phrasing better).
- BGE-Reranker-v2-M3 sits between us and ColBERTv2. We're ~0.5 nDCG points behind on average. The win on SciFact is interesting: their reranker is trained on a more academic-leaning corpus.
- BM25 baseline is what you get without any of the tricks. The gap from BM25 to rag-engine (+13.6 nDCG points avg) is what you're paying for.

### Recall@10 on BEIR

| Dataset    | rag-engine | ColBERTv2 | BGE-Reranker-v2-M3 |
|------------|-----------|-----------|---------------------|
| NQ         | 0.948     | 0.964     | 0.951               |
| TREC-COVID | 0.732     | 0.748     | 0.741               |
| HotpotQA   | 0.834     | 0.871     | 0.842               |
| FiQA       | 0.682     | 0.701     | 0.689               |
| SciFact    | 0.969     | 0.971     | 0.974               |
| **Mean**   | **0.833** | **0.851** | **0.839**           |

We trail by 1.8 recall points on average. HotpotQA is the worst at 3.7 points behind because multi-hop reasoning genuinely benefits from full late interaction over the corpus.

## Latency: Where We Win

This is the headline. We hold our quality steady while running dramatically faster.

### Retrieval Stage Latency (p95, ms)

Same hardware (g5.48xlarge), same 1M doc subset for fair comparison since ColBERTv2 doesn't scale to 50M without massive infrastructure.

| Stage              | rag-engine | ColBERTv2 | BGE-Reranker-v2-M3 |
|--------------------|-----------|-----------|---------------------|
| Initial retrieval  | 5.1 ms    | 87.3 ms   | 6.2 ms              |
| Late interaction   | 3.4 ms    | 142.6 ms  | -                   |
| Reranking          | 9.5 ms    | -         | 81.4 ms             |
| **Total p95**      | **18.0 ms** | **229.9 ms** | **87.6 ms**     |

### Why the gap exists

**ColBERTv2 latency**: Full late interaction over the entire corpus means storing per-token embeddings for every document. At 50M docs that's a 50x storage explosion, and the search itself has to do MaxSim over millions of token embeddings. Even with PLAID centroid pruning the latency is structurally bounded by the per-token ops.

**BGE-Reranker-v2-M3 latency**: It's a 568M param model. Quality is great but inference cost per query-doc pair is brutal. Our domain fine-tuned MiniLM is 110M params and ONNX-optimized.

**Our shortcut**: We use late interaction *only* as a top-100 -> top-20 filter, not as the primary retrieval mechanism. Random orthogonal projection to 128-dim, MaxSim over 20 candidates max. This captures most of ColBERT's quality benefit (~80% of it on our internal evals) without the infrastructure cost.

## Memory Footprint

For the same 1M doc corpus:

| System              | Index size | RAM footprint | Notes |
|---------------------|-----------|----------------|-------|
| rag-engine          | 2.84 GB   | 1.2 GB hot     | Memory-mapped HNSW |
| ColBERTv2 (PLAID)   | 142.7 GB  | 38.4 GB        | Per-token embeddings + centroids |
| BGE-Reranker-v2-M3  | 2.84 GB   | 4.1 GB         | Same retrieval, model in RAM |

ColBERTv2's index is 50x larger. That's the price of full late interaction.

## Throughput

Max sustained QPS at p95 < 50ms target on identical 8-replica deployments:

| System              | Max QPS | At p95 |
|---------------------|---------|--------|
| rag-engine          | 2847    | 18.0 ms |
| BGE-Reranker-v2-M3  | 312     | 87.6 ms |
| ColBERTv2 (PLAID)   | 84      | 229.9 ms (above target) |

ColBERTv2 cannot meet the latency target on this hardware. To hit 2K QPS you'd need ~24x the replicas, blowing past any reasonable cost envelope.

## Honest Tradeoffs

### Where ColBERTv2 still wins

- **Compositional/multi-hop queries**: HotpotQA, MuSiQue. Full late interaction over the corpus catches token-level matches that get lost in our top-100 retrieval pre-filter. If you genuinely need SOTA on compositional retrieval and latency isn't a constraint, use ColBERTv2.
- **Long-tail vocabulary**: Niche domains where query terms barely appear in training data. ColBERTv2's per-token granularity helps more there.
- **Zero-shot to new domains**: Our cross-encoder is fine-tuned on domain pairs. ColBERTv2 generalizes better to genuinely new domains without retraining.

### Where BGE-Reranker-v2-M3 still wins

- **Multilingual**: Their model is trained on 100+ languages. Ours is English-tuned. If you need multilingual rerank out of the box, use theirs.
- **No fine-tuning data**: If you don't have labeled query-passage pairs, BGE-Reranker-v2-M3 zero-shot is better than our generic baseline.

### Where we win

- **Production latency budgets**: <50ms p95 at 2K+ QPS is non-negotiable for most product use cases. We deliver it. ColBERTv2 fundamentally can't.
- **Cost per query**: 8-10x cheaper than competitors at the same throughput.
- **Memory efficiency**: 50x smaller index than ColBERTv2 means we can shard fewer ways and use cheaper instance types.
- **Domain fit**: When you have labeled domain pairs (and most production teams do), our fine-tuned cross-encoder beats general-purpose rerankers on that domain.

## Conclusion

The honest answer: rag-engine is not a SOTA-on-quality system. It's a SOTA-on-quality-per-millisecond and SOTA-on-quality-per-dollar system. For the BEIR-leaderboard chasers, ColBERTv2 still rules. For anyone shipping retrieval into a product with real users and a finite cloud bill, this trade is the one you want to make.

If your traffic is dominated by short user queries, single-hop questions, or domain-specific corpora where you can fine-tune the reranker, the gap to SOTA closes to <1 nDCG point. If your traffic is heavily compositional or multi-hop, the gap widens to 2-3 nDCG points and ColBERTv2 becomes worth the latency cost.

## Reproducibility

All numbers above are reproducible with:

```bash
# BEIR runs
python benchmarks/run_beir.py --dataset nq trec-covid hotpotqa fiqa scifact \
    --systems rag_engine colbertv2 bge_reranker_v2_m3 bm25

# Latency comparison
python benchmarks/run_latency_comparison.py --systems rag_engine colbertv2 bge \
    --qps 2000 --replicas 8 --duration-min 30
```

ColBERTv2 numbers used the official `colbert-ai==0.2.19` package with PLAID indexing. BGE-Reranker-v2-M3 numbers used `BAAI/bge-reranker-v2-m3` from HuggingFace, ONNX exported with `optimum.onnxruntime`. Hardware was identical across runs (g5.48xlarge, 8 replicas, EPYC 7R13).

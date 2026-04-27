"""Benchmark suite for RAG engine retrieval performance."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    total_queries: int = 0
    successful: int = 0
    failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def qps(self) -> float:
        return self.successful / self.elapsed if self.elapsed > 0 else 0

    @property
    def p50(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0

    @property
    def p95(self) -> float:
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0

    @property
    def p99(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "successful": self.successful,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed, 2),
            "qps": round(self.qps, 1),
            "latency_ms": {
                "p50": round(self.p50, 2),
                "p95": round(self.p95, 2),
                "p99": round(self.p99, 2),
                "mean": round(statistics.mean(self.latencies_ms), 2) if self.latencies_ms else 0,
                "stdev": round(statistics.stdev(self.latencies_ms), 2) if len(self.latencies_ms) > 1 else 0,
            },
            "error_rate": round(self.failed / self.total_queries, 6) if self.total_queries else 0,
        }


async def run_single_query(engine, query: str, result: BenchmarkResult) -> None:
    start = time.perf_counter()
    try:
        await engine.aretrieve(query, top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.latencies_ms.append(elapsed_ms)
        result.successful += 1
    except Exception:
        result.failed += 1
    result.total_queries += 1


async def run_benchmark(
    config_path: Path,
    queries: list[str],
    concurrency: int,
    warmup: int = 100,
) -> BenchmarkResult:
    from rag_engine import RAGEngine

    engine = RAGEngine.from_config(config_path)

    # Warmup phase
    for q in queries[:warmup]:
        try:
            await engine.aretrieve(q, top_k=10)
        except Exception:
            pass

    result = BenchmarkResult()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_query(query: str) -> None:
        async with semaphore:
            await run_single_query(engine, query, result)

    result.start_time = time.perf_counter()
    tasks = [bounded_query(q) for q in queries]
    await asyncio.gather(*tasks)
    result.end_time = time.perf_counter()

    return result


def load_queries(path: Path, n: int) -> list[str]:
    if path.exists():
        with open(path) as f:
            queries = [json.loads(line)["query"] for line in f if line.strip()]
        if len(queries) >= n:
            return queries[:n]
        # Repeat to reach desired count
        multiplier = (n // len(queries)) + 1
        return (queries * multiplier)[:n]

    # Generate synthetic queries for testing
    rng = np.random.default_rng(42)
    templates = [
        "What is {}?",
        "How does {} work?",
        "Explain the concept of {}",
        "What are the benefits of {}?",
        "Compare {} and {}",
    ]
    topics = [
        "gradient descent", "attention mechanism", "batch normalization",
        "dropout regularization", "learning rate scheduling", "weight decay",
        "knowledge distillation", "model quantization", "flash attention",
        "rotary position embeddings", "mixture of experts", "speculative decoding",
    ]
    queries = []
    for _ in range(n):
        template = templates[rng.integers(len(templates))]
        chosen = rng.choice(topics, size=template.count("{}"), replace=False)
        queries.append(template.format(*chosen))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG engine benchmarks")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--query-file", type=Path, default=Path("benchmarks/queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    queries = load_queries(args.query_file, args.queries)
    print(f"Running benchmark: {args.queries} queries, concurrency={args.concurrency}")

    result = asyncio.run(
        run_benchmark(args.config, queries, args.concurrency, args.warmup)
    )

    output = result.to_dict()
    print(json.dumps(output, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()

"""RAGAS evaluation suite for measuring RAG pipeline quality."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return Dataset.from_dict({
        "question": [r["question"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
        "contexts": [r.get("contexts", []) for r in records],
    })


def generate_answers(dataset: Dataset, config_path: Path) -> Dataset:
    from rag_engine import RAGEngine

    engine = RAGEngine.from_config(config_path)

    answers = []
    retrieved_contexts = []

    for question in dataset["question"]:
        results = engine.retrieve(question, top_k=5)
        contexts = [r.text for r in results]
        retrieved_contexts.append(contexts)

        # For eval purposes, use top context as answer proxy
        answers.append(contexts[0] if contexts else "")

    return dataset.add_column("answer", answers).add_column(
        "retrieved_contexts", retrieved_contexts
    )


def run_evaluation(dataset: Dataset) -> dict:
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    eval_dataset = Dataset.from_dict({
        "question": dataset["question"],
        "answer": dataset["answer"],
        "contexts": dataset["retrieved_contexts"],
        "ground_truth": dataset["ground_truth"],
    })

    result = evaluate(eval_dataset, metrics=metrics)
    return dict(result)


def compute_retrieval_metrics(dataset: Dataset) -> dict:
    """Compute retrieval-specific metrics independent of RAGAS."""
    recalls = []
    precisions = []
    mrrs = []

    for i in range(len(dataset)):
        ground_truth = dataset["ground_truth"][i]
        contexts = dataset["retrieved_contexts"][i]

        # Simple overlap-based relevance
        relevant_retrieved = 0
        first_relevant_rank = None

        for rank, ctx in enumerate(contexts):
            overlap = len(set(ground_truth.lower().split()) & set(ctx.lower().split()))
            relevance = overlap / max(len(ground_truth.split()), 1)

            if relevance > 0.3:
                relevant_retrieved += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1

        recall = min(relevant_retrieved / max(1, 1), 1.0)
        precision = relevant_retrieved / len(contexts) if contexts else 0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    return {
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "mrr": float(np.mean(mrrs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on RAG engine")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to golden set JSONL")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/ragas_eval.json"))
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("Loading golden set from %s", args.dataset)
    dataset = load_golden_set(args.dataset)
    logger.info("Loaded %d evaluation examples", len(dataset))

    if not args.skip_generation:
        logger.info("Generating answers with RAG engine")
        dataset = generate_answers(dataset, args.config)

    logger.info("Running RAGAS evaluation")
    ragas_scores = run_evaluation(dataset)

    logger.info("Computing retrieval metrics")
    retrieval_metrics = compute_retrieval_metrics(dataset)

    results = {
        "ragas_scores": ragas_scores,
        "retrieval_metrics": retrieval_metrics,
        "num_examples": len(dataset),
        "config": str(args.config),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Results:")
    print(f"  Faithfulness:       {ragas_scores.get('faithfulness', 'N/A'):.4f}")
    print(f"  Answer Relevancy:   {ragas_scores.get('answer_relevancy', 'N/A'):.4f}")
    print(f"  Context Precision:  {ragas_scores.get('context_precision', 'N/A'):.4f}")
    print(f"  Context Recall:     {ragas_scores.get('context_recall', 'N/A'):.4f}")
    print(f"  Answer Correctness: {ragas_scores.get('answer_correctness', 'N/A'):.4f}")
    print(f"\n  Retrieval MRR:      {retrieval_metrics['mrr']:.4f}")
    print(f"  Retrieval Recall:   {retrieval_metrics['recall_mean']:.4f}")
    print(f"  Retrieval Precision:{retrieval_metrics['precision_mean']:.4f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

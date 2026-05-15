"""
SENT-10: FinBERT sentiment scorer.

Model: ProsusAI/finbert — BERT fine-tuned on 10K financial sentences.
Labels: positive, negative, neutral (softmax probabilities).

The pipeline is loaded once and cached as a module-level singleton to avoid
reloading the 440 MB model on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Pipeline

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,          # return all 3 label scores
            truncation=True,
            max_length=512,
        )
    return _pipeline


def score_chunk(text: str) -> dict[str, float]:
    """Run FinBERT on a single text chunk.

    Returns dict with keys: positive, negative, neutral (summing to ~1.0).
    """
    pipe = _get_pipeline()
    results = pipe(text[:2000])  # hard safety cap

    # results is a list of lists when top_k=None: [[{label, score}, ...]]
    label_scores = results[0] if isinstance(results[0], list) else results
    return {item["label"].lower(): item["score"] for item in label_scores}


def score_chunks(chunks: list[str]) -> list[dict[str, float]]:
    """Score a list of chunks in a single batched forward pass."""
    if not chunks:
        return []
    pipe = _get_pipeline()
    results = pipe(chunks, truncation=True, max_length=512, batch_size=8)
    out = []
    for result in results:
        label_scores = result if isinstance(result, list) else [result]
        out.append({item["label"].lower(): item["score"] for item in label_scores})
    return out

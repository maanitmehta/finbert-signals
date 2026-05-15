"""
SENT-11: Document-level score aggregator.

Combines per-chunk FinBERT scores into a single document-level score using
length-weighted averaging (longer chunks carry more signal).

composite_score = positive - negative  ∈ [-1, +1]
  > 0  → net positive sentiment
  < 0  → net negative sentiment
  ≈ 0  → neutral
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.chunker import chunk
from nlp.scorer import score_chunks


def score_document(text: str) -> dict[str, float]:
    """Chunk text, run FinBERT on each chunk, return aggregated scores.

    Returns:
        {
            "positive": float,
            "negative": float,
            "neutral":  float,
            "composite_score": float,   # positive - negative
            "chunk_count": int,
        }
    """
    chunks = chunk(text)
    if not chunks:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "composite_score": 0.0, "chunk_count": 0}

    chunk_scores = score_chunks(chunks)
    weights = [len(c.split()) for c in chunks]
    total_weight = sum(weights)

    agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for scores, w in zip(chunk_scores, weights):
        frac = w / total_weight
        for label in agg:
            agg[label] += scores.get(label, 0.0) * frac

    agg["composite_score"] = round(agg["positive"] - agg["negative"], 6)
    agg["chunk_count"] = len(chunks)
    return agg


if __name__ == "__main__":
    sample = """
    Apple reported record quarterly revenue of $124 billion, beating analyst expectations.
    iPhone sales grew 8% year-over-year driven by strong demand in emerging markets.
    However, the company warned of supply chain disruptions that may pressure margins in Q2.
    Services revenue hit an all-time high of $26 billion, up 14% from the prior year.
    """
    result = score_document(sample)
    for k, v in result.items():
        print(f"  {k:<20} {v}")

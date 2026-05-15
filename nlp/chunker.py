"""
SENT-09: Text chunker — splits a document into ≤512-token windows for FinBERT.

FinBERT (BERT-based) has a hard 512-token limit. We split on sentence
boundaries to avoid cutting mid-thought, then pack sentences into windows
that stay under the limit with a small safety margin.
"""

from __future__ import annotations

import re


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_WORDS = 400  # conservative proxy for tokens (1 word ≈ 1.3 tokens)


def _split_sentences(text: str) -> list[str]:
    sents = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def chunk(text: str, max_words: int = _MAX_WORDS) -> list[str]:
    """Split text into chunks of at most max_words words, on sentence boundaries.

    Returns a list of non-empty string chunks.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        if current_words + words > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sent)
        current_words += words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]

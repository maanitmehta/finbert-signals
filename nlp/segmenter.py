"""
Filing segmenter — splits 8-K press release text into labelled chunks
for the Transcript Scrubber.

Because BeautifulSoup strips most whitespace, we split on sentence
boundaries then pack sentences into ~60-word windows, which maps cleanly
onto FinBERT's 512-token limit.

Section types
─────────────
  overview    — opening / introductory paragraphs
  financial   — metrics, tables, $ amounts
  management  — CEO / CFO direct quotes and commentary
  guidance    — forward-looking statements
  disclaimer  — legal boilerplate / risk language
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TypedDict

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Section classifier ─────────────────────────────────────────────────────────

_DISCLAIMER_KWS = frozenset([
    "forward-looking", "safe harbor", "risk factor", "cannot guarantee",
    "material uncertainty", "subject to change", "actual results may",
])
_GUIDANCE_KWS = frozenset([
    "we expect", "we anticipate", "we project", "going forward",
    "next quarter", "next fiscal", "fiscal 20", "our outlook", "we believe will",
])
_FINANCIAL_KWS = frozenset([
    "revenue", "gross margin", "operating income", "earnings per share",
    "diluted", "net income", "cash flow", "basis points", "year-over-year",
    "quarter-over-quarter", "sequentially",
])
_MGMT_KWS = frozenset([
    '" said', "said.", "stated.", "we believe", "our strategy",
    "chief executive", "chief financial", "tim cook", "ceo", "cfo",
])

_SECTION_ORDER = ["disclaimer", "guidance", "financial", "management"]


def _classify(text: str) -> str:
    t = text.lower()
    for section in _SECTION_ORDER:
        kws = {
            "disclaimer": _DISCLAIMER_KWS,
            "guidance":   _GUIDANCE_KWS,
            "financial":  _FINANCIAL_KWS,
            "management": _MGMT_KWS,
        }[section]
        if any(kw in t for kw in kws):
            return section
    return "overview"


# ── Segmenter ──────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


class Chunk(TypedDict):
    index:           int
    text:            str
    section:         str
    word_count:      int
    positive:        float
    negative:        float
    neutral:         float
    composite_score: float


def segment(text: str, target_words: int = 60, min_words: int = 15) -> list[Chunk]:
    """Split text into sentence-grouped chunks with section labels.

    Groups sentences until the chunk reaches ~target_words, then starts a
    new chunk. Chunks shorter than min_words are merged into the next one.
    """
    sentences = _SENT_SPLIT.split(text.strip())

    chunks: list[Chunk] = []
    buffer:  list[str]  = []
    buf_wc = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        wc = len(sent.split())
        buffer.append(sent)
        buf_wc += wc

        if buf_wc >= target_words:
            combined = " ".join(buffer)
            chunks.append(_make_chunk(len(chunks), combined))
            buffer  = []
            buf_wc  = 0

    if buffer and buf_wc >= min_words:
        combined = " ".join(buffer)
        chunks.append(_make_chunk(len(chunks), combined))

    return chunks


def _make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        index=index,
        text=text,
        section=_classify(text),
        word_count=len(text.split()),
        positive=0.0,
        negative=0.0,
        neutral=1.0,
        composite_score=0.0,
    )


# ── FinBERT scorer ─────────────────────────────────────────────────────────────

def score_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Run FinBERT on all chunks in one batched pass. Mutates chunks in-place."""
    from nlp.scorer import score_chunks as _score

    texts   = [c["text"] for c in chunks]
    results = _score(texts)

    for chunk, sc in zip(chunks, results):
        chunk["positive"]        = float(sc.get("positive", 0.0))
        chunk["negative"]        = float(sc.get("negative", 0.0))
        chunk["neutral"]         = float(sc.get("neutral",  1.0))
        chunk["composite_score"] = chunk["positive"] - chunk["negative"]

    return chunks


if __name__ == "__main__":
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT raw_text, filing_date, form_type FROM filings WHERE ticker='AAPL' ORDER BY filing_date DESC LIMIT 1"
    ).fetchone()
    if not row:
        print("No AAPL filings in DB. Run the EDGAR fetcher first.")
    else:
        print(f"Segmenting {row['form_type']} filed {row['filing_date']}...")
        chunks = segment(row["raw_text"])
        scored = score_chunks(chunks)
        for c in scored:
            bar = "█" * int(abs(c["composite_score"]) * 20)
            sign = "+" if c["composite_score"] >= 0 else "-"
            print(f"  [{c['index']:02d}] {c['section']:<12} {sign}{abs(c['composite_score']):.3f}  {bar}  {c['text'][:60]}...")

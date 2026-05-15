"""
Entity-level sentiment scorer for the tag cloud.

Pipeline:
  1. spaCy NER extracts named entities (PERSON, ORG, PRODUCT, GPE, …)
  2. For each entity, collect all sentences that mention it
  3. Run FinBERT on those sentences (batched)
  4. Return per-entity aggregated sentiment + mention count + example quote

Falls back to a financial-keyword extraction if the spaCy model is absent.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ──────────────────────────────────────────────────────────────────

# spaCy entity types worth tracking in a financial context
_KEEP_TYPES = {"PERSON", "ORG", "PRODUCT", "GPE", "NORP", "EVENT", "LAW"}

_TYPE_LABEL = {
    "PERSON":  "Executive",
    "ORG":     "Company",
    "PRODUCT": "Product",
    "GPE":     "Country/Region",
    "NORP":    "Group",
    "EVENT":   "Event",
    "LAW":     "Regulation",
}

# Keyword fallback — always run alongside spaCy to capture financial terms
# that NER models often miss (they're common nouns, not proper nouns)
_FINANCIAL_KW: dict[str, list[str]] = {
    "Macro":   ["inflation", "interest rate", "fed", "gdp", "recession",
                "tariff", "trade war", "rate hike", "monetary"],
    "Metric":  ["revenue", "earnings", "eps", "gross margin", "guidance",
                "outlook", "forecast", "dividend", "buyback", "cash flow",
                "operating income", "net income"],
    "Risk":    ["supply chain", "competition", "regulation", "lawsuit",
                "antitrust", "investigation", "sanction"],
    "Market":  ["wall street", "nasdaq", "nyse", "s&p 500", "dow jones",
                "market share", "valuation", "price target"],
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ── spaCy loader ───────────────────────────────────────────────────────────────

_nlp = None
_nlp_tried = False


def _get_nlp():
    global _nlp, _nlp_tried
    if _nlp_tried:
        return _nlp
    _nlp_tried = True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
    return _nlp


# ── Core scorer ────────────────────────────────────────────────────────────────

def score_entities(
    text: str,
    min_mentions: int = 1,
    max_entities: int = 40,
    max_sents_per_entity: int = 6,
) -> dict[str, dict]:
    """Extract and score named entities + financial keywords from text.

    Returns:
        {entity_label: {type, mentions, composite_score, positive, negative,
                        neutral, example}}
    """
    sentences = _split_sentences(text)
    entity_sents: dict[str, list[str]] = defaultdict(list)
    entity_types: dict[str, str] = {}

    # ── spaCy NER pass ────────────────────────────────────────────────────────
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text[:120_000])   # cap for speed
        for sent in doc.sents:
            sent_txt = sent.text.strip()
            if len(sent_txt.split()) < 5:
                continue
            for ent in sent.ents:
                if ent.label_ not in _KEEP_TYPES:
                    continue
                key = ent.text.strip()
                if len(key) < 3 or key.isdigit():
                    continue
                entity_sents[key].append(sent_txt)
                entity_types[key] = _TYPE_LABEL.get(ent.label_, "Other")

    # ── Keyword pass (always runs, complements NER) ───────────────────────────
    text_lower = text.lower()
    for category, keywords in _FINANCIAL_KW.items():
        for kw in keywords:
            if kw not in text_lower:
                continue
            for sent in sentences:
                if kw in sent.lower():
                    key = kw.title()
                    entity_sents[key].append(sent)
                    entity_types[key] = category

    # ── Filter by min_mentions ────────────────────────────────────────────────
    filtered = {
        k: v for k, v in entity_sents.items()
        if len(v) >= min_mentions and len(k) <= 60
    }
    if not filtered:
        return {}

    # Sort by mention count and cap
    ranked = sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)
    ranked = ranked[:max_entities]

    # ── FinBERT scoring ───────────────────────────────────────────────────────
    from nlp.scorer import score_chunks as _score

    results: dict[str, dict] = {}
    for entity, sents in ranked:
        sample = sents[:max_sents_per_entity]
        scores = _score(sample)
        if not scores:
            continue

        avg_pos = sum(s.get("positive", 0.0) for s in scores) / len(scores)
        avg_neg = sum(s.get("negative", 0.0) for s in scores) / len(scores)
        avg_neu = sum(s.get("neutral",  0.0) for s in scores) / len(scores)

        results[entity] = {
            "type":            entity_types.get(entity, "Other"),
            "mentions":        len(sents),
            "composite_score": round(avg_pos - avg_neg, 4),
            "positive":        round(avg_pos, 4),
            "negative":        round(avg_neg, 4),
            "neutral":         round(avg_neu, 4),
            "example":         sents[0][:300],
        }

    return results


if __name__ == "__main__":
    from db import get_conn
    conn = get_conn()
    row = conn.execute(
        "SELECT raw_text, filing_date FROM filings WHERE ticker='AAPL' "
        "ORDER BY LENGTH(raw_text) DESC LIMIT 1"
    ).fetchone()
    if not row:
        print("No AAPL filings found.")
    else:
        print(f"Scoring entities in {row['filing_date']} filing…")
        results = score_entities(row["raw_text"])
        sorted_r = sorted(results.items(), key=lambda x: abs(x[1]["composite_score"]), reverse=True)
        for entity, data in sorted_r[:20]:
            bar = "█" * int(max(0, data["composite_score"]) * 15)
            bar += "░" * int(max(0, -data["composite_score"]) * 15)
            print(f"  {entity:<25} {data['composite_score']:+.3f}  {bar:<20}  "
                  f"({data['type']}, n={data['mentions']})")

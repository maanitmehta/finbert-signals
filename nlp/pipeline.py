"""
Scoring pipeline — reads unscored filings and articles from SQLite,
runs FinBERT on each, and writes to sentiment_scores.
Called before alignment so sentiment_scores is populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_conn, init_db
from nlp.aggregator import score_document


def _already_scored(conn, source: str, source_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM sentiment_scores WHERE source = ? AND source_id = ?",
        (source, source_id),
    ).fetchone() is not None


def score_filings(ticker: str | None = None) -> int:
    """Score all unscored 8-K filings. Returns count of new scores written."""
    init_db()
    ticker_upper = ticker.upper() if ticker else None
    stored = 0

    with get_conn() as conn:
        sql = "SELECT id, ticker, filing_date, raw_text FROM filings WHERE raw_text IS NOT NULL"
        params = []
        if ticker_upper:
            sql += " AND ticker = ?"
            params.append(ticker_upper)

        rows = conn.execute(sql, params).fetchall()
        print(f"Scoring {len(rows)} filing(s)...")

        for row in rows:
            if _already_scored(conn, "edgar", row["id"]):
                continue

            scores = score_document(row["raw_text"])
            conn.execute(
                """INSERT INTO sentiment_scores
                   (ticker, event_date, source, source_id,
                    positive, negative, neutral, composite_score)
                   VALUES (?, ?, 'edgar', ?, ?, ?, ?, ?)""",
                (
                    row["ticker"], row["filing_date"], row["id"],
                    scores["positive"], scores["negative"],
                    scores["neutral"], scores["composite_score"],
                ),
            )
            stored += 1
            print(
                f"  [edgar] {row['ticker']} {row['filing_date']} "
                f"composite={scores['composite_score']:+.3f} "
                f"({scores['chunk_count']} chunks)"
            )

    return stored


def score_articles(ticker: str | None = None) -> int:
    """Score all unscored news articles. Returns count of new scores written."""
    init_db()
    ticker_upper = ticker.upper() if ticker else None
    stored = 0

    with get_conn() as conn:
        sql = """
            SELECT id, ticker, published_at,
                   (COALESCE(title, '') || '. ' ||
                    COALESCE(description, '') || ' ' ||
                    COALESCE(content, '')) AS text
            FROM articles
            WHERE relevant = 1
        """
        params = []
        if ticker_upper:
            sql += " AND ticker = ?"
            params.append(ticker_upper)

        rows = conn.execute(sql, params).fetchall()
        print(f"Scoring {len(rows)} article(s)...")

        for row in rows:
            if _already_scored(conn, "news", row["id"]):
                continue

            text = (row["text"] or "").strip()
            if len(text) < 30:
                continue

            scores = score_document(text)
            conn.execute(
                """INSERT INTO sentiment_scores
                   (ticker, event_date, source, source_id,
                    positive, negative, neutral, composite_score)
                   VALUES (?, ?, 'news', ?, ?, ?, ?, ?)""",
                (
                    row["ticker"], row["published_at"], row["id"],
                    scores["positive"], scores["negative"],
                    scores["neutral"], scores["composite_score"],
                ),
            )
            stored += 1

        print(f"Scored {stored} new articles")

    return stored


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    n1 = score_filings(ticker)
    n2 = score_articles(ticker)
    print(f"\nDone — {n1} filing(s), {n2} article(s) scored.")

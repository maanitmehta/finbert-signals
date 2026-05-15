"""
SENT-12: Alignment engine — joins sentiment_scores with prices.

Sentiment event on date T is matched to the next available trading day
within MAX_FORWARD_DAYS days (default 7). This correctly handles:
  - Filings/articles published on weekends or market holidays
  - After-hours earnings releases where the reaction is the next open

Signals with no matching price within the window are dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from db import get_conn

MAX_FORWARD_DAYS = 7  # max calendar days to look ahead for a trading day


def get_aligned(ticker: str, source: str | None = None) -> pd.DataFrame:
    """Return DataFrame of sentiment scores joined to next-day returns.

    Uses forward-fill: if the event falls on a weekend or holiday, match it
    to the next available trading day within MAX_FORWARD_DAYS. Signals with
    no price within that window are dropped.

    Columns:
        ticker, event_date, source, composite_score,
        positive, negative, neutral, next_day_return, price_date
    """
    # Subquery finds the nearest trading day on or after the event date,
    # capped at MAX_FORWARD_DAYS to prevent stale cross-year matches.
    # News articles additionally require relevant = 1 to exclude off-topic noise.
    sql = f"""
        SELECT
            s.ticker,
            s.event_date,
            s.source,
            s.composite_score,
            s.positive,
            s.negative,
            s.neutral,
            p.next_day_return,
            p.date AS price_date
        FROM sentiment_scores s
        INNER JOIN prices p
            ON  p.ticker = s.ticker
            AND p.date = (
                SELECT MIN(p2.date)
                FROM prices p2
                WHERE p2.ticker   = s.ticker
                  AND p2.date    >= s.event_date
                  AND p2.date    <= date(s.event_date, '+{MAX_FORWARD_DAYS} days')
                  AND p2.next_day_return IS NOT NULL
            )
        LEFT JOIN articles a
            ON  s.source_id = a.id
            AND s.source    = 'news'
        WHERE s.ticker = ?
          AND (s.source = 'edgar' OR a.relevant = 1)
    """
    params: list = [ticker.upper()]

    if source:
        sql += " AND s.source = ?"
        params.append(source)

    sql += " ORDER BY s.event_date"

    with get_conn() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    return df


def get_daily_sentiment(ticker: str, source: str | None = None) -> pd.DataFrame:
    """Aggregate multiple intra-day scores into one row per (ticker, date).

    When several articles land on the same day, averages their composite
    scores before alignment with price returns.
    """
    df = get_aligned(ticker, source)
    if df.empty:
        return df

    agg = (
        df.groupby(["ticker", "event_date", "next_day_return"])
        .agg(
            composite_score=("composite_score", "mean"),
            positive=("positive", "mean"),
            negative=("negative", "mean"),
            neutral=("neutral", "mean"),
            n_signals=("composite_score", "count"),
        )
        .reset_index()
        .sort_values("event_date")
    )
    return agg


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = get_aligned(t)
    print(df.to_string(index=False))
    print(f"\n{len(df)} aligned rows for {t}")

"""
NewsAPI fetcher — pulls *financial* headlines for a ticker.
Free tier: 100 requests/day, articles up to 1 month old.

Three-layer noise filter:
  1. Domain whitelist  — only reputable financial news sources
  2. Query improvement — ticker OR (company AND financial-terms)
  3. Relevance check   — title/description must reference the ticker or
                         company name with at least one financial keyword
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from config import NEWSAPI_KEY
from db import get_conn, init_db

_BASE = "https://newsapi.org/v2/everything"

# ── Layer 1: domain whitelist ──────────────────────────────────────────────────
# Restricts NewsAPI search to financial / business news publishers only.
_FINANCE_DOMAINS = ",".join([
    "reuters.com",
    "cnbc.com",
    "marketwatch.com",
    "seekingalpha.com",
    "fool.com",
    "thestreet.com",
    "benzinga.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "businessinsider.com",
    "barrons.com",
    "investopedia.com",
    "zacks.com",
    "apnews.com",
])

# ── Layer 3: relevance keywords ───────────────────────────────────────────────
_FINANCIAL_KWS = frozenset({
    # Unambiguous financial nouns only — verbs like "shares" excluded
    "stock", "stocks", "earnings", "revenue", "guidance",
    "dividend", "analyst", "forecast", "quarterly", "eps", "profit",
    "loss", "margin", "outlook", "beat", "miss", "upgrade", "downgrade",
    "price target", "market cap", "fiscal", "results", "sales", "growth",
    "ceo", "cfo", "acquisition", "merger", "ipo", "investor", "investors",
    "wall street", "nasdaq", "nyse", "q1", "q2", "q3", "q4",
    "annual report", "10-k", "10-q", "8-k", "sec filing",
    "short interest", "valuation", "pe ratio", "buy rating", "sell rating",
    "shipment", "shipments", "market share", "supply chain",
    "tariff", "tariffs", "trade", "gdp", "inflation", "rate hike",
})


def _build_query(ticker: str, company_name: str | None) -> str:
    """Layer 2: smart query that keeps ticker hits and filters company-name hits.

    Ticker symbols (e.g. 'AAPL') almost always appear in financial context.
    Company names (e.g. 'Apple') are ambiguous — require a financial co-term.
    """
    fin = (
        "(stock OR shares OR earnings OR revenue OR guidance OR analyst "
        "OR quarterly OR investor OR profit OR results OR forecast)"
    )

    if company_name:
        safe = company_name.replace('"', "")
        return f'"{ticker}" OR ("{safe}" AND {fin})'

    return f'"{ticker}"'


def _is_relevant(ticker: str, company_name: str, title: str, description: str) -> bool:
    """Layer 3: post-fetch relevance check on title + description.

    Returns True when the article demonstrably discusses the company in a
    financial context rather than just mentioning a shared word (e.g. 'Apple').
    """
    text      = f"{title} {description}".lower()
    title_l   = title.lower()
    ticker_l  = ticker.lower()
    company_l = (company_name or "").lower()

    # Ticker symbol in title is a very strong financial signal
    if ticker_l in title_l:
        return True

    # Company name present + at least one financial keyword
    company_root = company_l[:10]  # e.g. "apple inc." → "apple inc."
    if company_root and company_root in text:
        return any(kw in text for kw in _FINANCIAL_KWS)

    return False


def fetch_and_store(
    ticker: str,
    company_name: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page_size: int = 100,
) -> int:
    """Fetch financial news for ticker, filter for relevance, store in DB.

    Returns number of newly stored articles.
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set in .env")

    init_db()
    ticker = ticker.upper()

    params: dict = {
        "q":         _build_query(ticker, company_name),
        "language":  "en",
        "sortBy":    "publishedAt",
        "pageSize":  min(page_size, 100),
        "apiKey":    NEWSAPI_KEY,
        "domains":   _FINANCE_DOMAINS,
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    resp = requests.get(_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', data)}")

    articles = data.get("articles", [])
    stored = 0

    with get_conn() as conn:
        for art in articles:
            url = art.get("url", "")
            if not url:
                continue

            published_at = (art.get("publishedAt") or "")[:10]
            title        = art.get("title")       or ""
            description  = art.get("description") or ""
            content      = art.get("content")     or ""
            source_name  = (art.get("source") or {}).get("name", "")

            relevant = 1 if _is_relevant(ticker, company_name or "", title, description) else 0

            result = conn.execute(
                """INSERT OR IGNORE INTO articles
                   (ticker, published_at, title, description, content, url, source_name, relevant)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, published_at, title, description, content, url, source_name, relevant),
            )
            if result.rowcount:
                stored += 1

    return stored


def backfill_relevance(ticker: str, company_name: str) -> tuple[int, int]:
    """Re-score relevance for all existing articles for this ticker.

    Returns (marked_relevant, marked_noise).
    Deletes sentiment_scores rows for articles now marked as noise.
    """
    ticker = ticker.upper()
    marked_rel = marked_noise = 0

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, description FROM articles WHERE ticker = ?",
            (ticker,),
        ).fetchall()

        for row in rows:
            relevant = 1 if _is_relevant(ticker, company_name, row["title"] or "", row["description"] or "") else 0
            conn.execute("UPDATE articles SET relevant = ? WHERE id = ?", (relevant, row["id"]))

            if relevant:
                marked_rel += 1
            else:
                marked_noise += 1
                # Remove the stale sentiment score so it won't pollute analysis
                conn.execute(
                    "DELETE FROM sentiment_scores WHERE source = 'news' AND source_id = ?",
                    (row["id"],),
                )

    return marked_rel, marked_noise


if __name__ == "__main__":
    import sys
    t       = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    company = sys.argv[2] if len(sys.argv) > 2 else "Apple"
    n = fetch_and_store(t, company_name=company)
    print(f"\nStored {n} new article(s) for {t}")

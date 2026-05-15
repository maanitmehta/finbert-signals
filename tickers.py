"""
S&P 500 ticker list fetched from Wikipedia and cached locally for 7 days.
Falls back to a hardcoded list of major names if the fetch fails.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

_CACHE = Path(__file__).parent / "data" / "sp500_tickers.json"
_WIKI  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_TTL   = timedelta(days=7)

_FALLBACK = [
    {"label": "AAPL — Apple",        "value": "AAPL", "name": "Apple"},
    {"label": "MSFT — Microsoft",    "value": "MSFT", "name": "Microsoft"},
    {"label": "AMZN — Amazon",       "value": "AMZN", "name": "Amazon"},
    {"label": "GOOGL — Alphabet",    "value": "GOOGL", "name": "Alphabet"},
    {"label": "META — Meta",         "value": "META",  "name": "Meta"},
    {"label": "NVDA — Nvidia",       "value": "NVDA",  "name": "Nvidia"},
    {"label": "TSLA — Tesla",        "value": "TSLA",  "name": "Tesla"},
    {"label": "JPM — JPMorgan",      "value": "JPM",   "name": "JPMorgan"},
    {"label": "V — Visa",            "value": "V",     "name": "Visa"},
    {"label": "JNJ — Johnson & Johnson", "value": "JNJ", "name": "Johnson & Johnson"},
    {"label": "SPY — S&P 500 ETF",   "value": "SPY",   "name": "S&P 500"},
    {"label": "QQQ — Nasdaq 100 ETF","value": "QQQ",   "name": "Nasdaq 100"},
]


def _scrape_wikipedia() -> list[dict]:
    resp = requests.get(
        _WIKI, timeout=15,
        headers={"User-Agent": "SentimentSignals/1.0 maanitkeyhem@gmail.com"},
    )
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})
    if not table:
        raise ValueError("S&P 500 table not found on Wikipedia page")

    all_rows = table.find_all("tr")
    # First row contains headers (no <thead>/<tbody> on this Wikipedia page)
    headers  = [th.get_text(strip=True) for th in all_rows[0].find_all(["th", "td"])]
    data_rows = all_rows[1:]

    sym_idx  = next((i for i, h in enumerate(headers) if "symbol" in h.lower()), 0)
    name_idx = next((i for i, h in enumerate(headers) if "security" in h.lower()), 1)

    tickers: list[dict] = []
    for row in data_rows:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(sym_idx, name_idx):
            continue
        symbol = cells[sym_idx].get_text(strip=True).replace(".", "-")
        name   = cells[name_idx].get_text(strip=True)
        if symbol and name:
            tickers.append({
                "label": f"{symbol} — {name}",
                "value": symbol,
                "name":  name,
            })

    return sorted(tickers, key=lambda x: x["value"])


def get_sp500_tickers(force_refresh: bool = False) -> list[dict]:
    """Return S&P 500 ticker list as Dash dropdown options (cached 7 days).

    Each entry: {"label": "AAPL — Apple Inc.", "value": "AAPL", "name": "Apple Inc."}
    """
    if not force_refresh and _CACHE.exists():
        age = datetime.now() - datetime.fromtimestamp(_CACHE.stat().st_mtime)
        if age < _TTL:
            try:
                return json.loads(_CACHE.read_text())
            except Exception:
                pass

    try:
        tickers = _scrape_wikipedia()
        _CACHE.write_text(json.dumps(tickers, indent=2))
        print(f"[tickers] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as exc:
        print(f"[tickers] Wikipedia fetch failed ({exc}), using fallback list")
        return _FALLBACK


def ticker_name_map() -> dict[str, str]:
    """Return {ticker: company_name} for auto-filling the company name field."""
    return {t["value"]: t["name"] for t in get_sp500_tickers()}


if __name__ == "__main__":
    tickers = get_sp500_tickers(force_refresh=True)
    print(f"Total tickers: {len(tickers)}")
    for t in tickers[:10]:
        print(f"  {t['label']}")

"""
SEC EDGAR fetcher — pulls 8-K filings (earnings press releases) for a ticker
and stores raw text in the filings table.

Rate limit: SEC allows 10 req/sec; we sleep 0.15s between archive requests.
User-Agent is required by SEC policy: https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import requests
from bs4 import BeautifulSoup

from config import EDGAR_USER_AGENT
from db import get_conn, init_db

_DATA_HOST = "https://data.sec.gov"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_DATA_HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
_WWW_HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

_cik_cache: dict[str, str] = {}


def _get_cik(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    resp = requests.get(_TICKERS_URL, headers=_WWW_HEADERS, timeout=15)
    resp.raise_for_status()

    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company list")


def _get_recent_8k_filings(cik: str, max_count: int) -> list[dict]:
    url = f"{_DATA_HOST}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=_DATA_HEADERS, timeout=15)
    resp.raise_for_status()

    recent = resp.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for form, acc, date, doc in zip(forms, accessions, dates, primary_docs):
        if form in ("8-K", "8-K/A"):
            results.append({
                "accession": acc,
                "filing_date": date,
                "form_type": form,
                "primary_doc": doc,
            })
            if len(results) >= max_count:
                break

    return results


def _fetch_exhibit_99_1(cik_int: int, acc_clean: str, accession: str) -> str | None:
    """Try to fetch Exhibit 99.1 (the actual earnings press release) from an 8-K filing."""
    index_url = f"{_ARCHIVE_BASE}/{cik_int}/{acc_clean}/{accession}-index.htm"
    time.sleep(0.15)
    try:
        resp = requests.get(index_url, headers=_WWW_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            row_text = " ".join(c.text.strip().lower() for c in cells)
            if "ex-99" in row_text or "99.1" in row_text:
                link = row.find("a", href=True)
                if link:
                    exhibit_url = "https://www.sec.gov" + link["href"]
                    time.sleep(0.15)
                    ex_resp = requests.get(exhibit_url, headers=_WWW_HEADERS, timeout=20)
                    ex_resp.raise_for_status()
                    ex_soup = BeautifulSoup(ex_resp.text, "html.parser")
                    text = ex_soup.get_text(separator=" ", strip=True)
                    if len(text.strip()) > 300:
                        return text
    except Exception:
        pass
    return None


def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str | None:
    cik_int   = int(cik)
    acc_clean = accession.replace("-", "")

    # Prefer Exhibit 99.1 — the actual earnings press release.
    # The primary 8-K document is usually just a short cover page.
    exhibit = _fetch_exhibit_99_1(cik_int, acc_clean, accession)
    if exhibit:
        return exhibit

    # Fall back to the primary document
    url = f"{_ARCHIVE_BASE}/{cik_int}/{acc_clean}/{primary_doc}"
    time.sleep(0.15)
    try:
        resp = requests.get(url, headers=_WWW_HEADERS, timeout=20)
        resp.raise_for_status()

        if primary_doc.lower().endswith((".htm", ".html")):
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return resp.text

    except Exception:
        return None


def fetch_and_store(ticker: str, max_count: int = 10) -> int:
    """Fetch up to max_count recent 8-K filings for ticker and store raw text.

    Returns the number of new filings stored.
    """
    init_db()
    ticker = ticker.upper()
    cik = _get_cik(ticker)
    filings = _get_recent_8k_filings(cik, max_count)

    stored = 0
    with get_conn() as conn:
        for f in filings:
            exists = conn.execute(
                "SELECT id FROM filings WHERE accession = ?", (f["accession"],)
            ).fetchone()
            if exists:
                continue

            text = _fetch_filing_text(cik, f["accession"], f["primary_doc"])
            if not text or len(text.strip()) < 100:
                continue

            conn.execute(
                """INSERT OR IGNORE INTO filings
                   (ticker, cik, accession, filing_date, form_type, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, cik, f["accession"], f["filing_date"], f["form_type"], text),
            )
            stored += 1
            print(f"  [{ticker}] Stored {f['form_type']} filed {f['filing_date']}")

    return stored


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    n = fetch_and_store(t, max_count=5)
    print(f"\nTotal stored: {n} new filing(s) for {t}")

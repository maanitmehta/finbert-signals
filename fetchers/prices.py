"""
yfinance price fetcher — downloads OHLCV and computes next-day returns.
next_day_return[t] = (close[t+1] - close[t]) / close[t]
The final row in any range will have next_day_return = NULL (no future day known).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from db import get_conn, init_db


def fetch_and_store(ticker: str, start: str, end: str) -> int:
    """Download prices for ticker between start/end (YYYY-MM-DD) and upsert into prices table.

    Returns number of rows inserted (skips duplicates).
    """
    init_db()
    ticker = ticker.upper()

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return 0

    # yfinance ≥0.2 returns MultiIndex columns for single ticker in some versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.strftime("%Y-%m-%d")
    df["next_day_return"] = df["close"].pct_change(1).shift(-1)

    stored = 0
    with get_conn() as conn:
        for _, row in df.iterrows():
            ndr = None if pd.isna(row["next_day_return"]) else float(row["next_day_return"])
            vol = None if pd.isna(row["volume"]) else int(row["volume"])

            result = conn.execute(
                """INSERT OR IGNORE INTO prices
                   (ticker, date, open, high, low, close, volume, next_day_return)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker,
                    row["date"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    vol,
                    ndr,
                ),
            )
            stored += result.rowcount

    return stored


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    n = fetch_and_store(t, start="2023-01-01", end="2025-06-01")
    print(f"Stored {n} price rows for {t}")

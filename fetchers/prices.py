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


def compute_and_store_horizons(ticker: str) -> int:
    """Compute and persist multi-day forward returns from prices already in DB.

    For every row in the prices table for `ticker`, computes:
        return_Nd = (close[t+n] / close[t]) - 1   for n ∈ {2, 3, 5, 10, 20}

    These are close-to-close returns over n trading days, calculated entirely
    from the OHLCV data already stored — no additional yfinance calls needed.
    Rows near the end of the series that lack n future closes are set to NULL.

    Returns the number of price rows updated.
    """
    ticker = ticker.upper()
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker=? ORDER BY date",
            conn,
            params=[ticker],
        )

    if df.empty:
        return 0

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    close = df["close"]

    horizon_series = {
        n: (close.shift(-n) / close - 1)
        for n in (2, 3, 5, 10, 20)
    }

    updates = []
    for ts, _ in df.iterrows():
        vals = [
            None if pd.isna(horizon_series[n].get(ts)) else float(horizon_series[n].get(ts))
            for n in (2, 3, 5, 10, 20)
        ]
        updates.append((*vals, ticker, ts.strftime("%Y-%m-%d")))

    with get_conn() as conn:
        conn.executemany(
            """UPDATE prices
               SET return_2d=?, return_3d=?, return_5d=?, return_10d=?, return_20d=?
               WHERE ticker=? AND date=?""",
            updates,
        )

    return len(updates)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    n = fetch_and_store(t, start="2023-01-01", end="2025-06-01")
    print(f"Stored {n} price rows for {t}")

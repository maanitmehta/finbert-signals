"""
SENT-15: Paper-trading backtest engine.

Takes a signals DataFrame (output of generate_signals on daily_df) and
simulates a simple long/short/hold strategy with full capital allocation:
  LONG  → hold 1 share-equivalent (position = +1)
  SHORT → short 1 share-equivalent (position = -1)
  HOLD  → stay in cash (return = 0)

Also tracks a buy-and-hold benchmark for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def run(signals_df: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    """Paper-trade a pre-built signals DataFrame.

    Args:
        signals_df: Must have columns: event_date, signal, next_day_return.
                    Usually the output of generate_signals(get_daily_sentiment(...)).
        initial_capital: Starting cash in dollars.

    Returns:
        Copy of signals_df with added columns:
          position        — +1 / -1 / 0
          portfolio_value — strategy equity after each day
          bnh_value       — buy-and-hold equity after each day
    """
    df = signals_df.dropna(subset=["next_day_return"]).copy()
    df = df.sort_values("event_date").reset_index(drop=True)

    portfolio = initial_capital
    bnh = initial_capital
    port_vals: list[float] = []
    bnh_vals: list[float] = []

    for _, row in df.iterrows():
        sig = str(row.get("signal", "HOLD"))
        ret = float(row.get("next_day_return") or 0.0)

        port_ret = ret if sig == "LONG" else (-ret if sig == "SHORT" else 0.0)
        portfolio *= 1 + port_ret
        bnh *= 1 + ret

        port_vals.append(portfolio)
        bnh_vals.append(bnh)

    df["position"] = df["signal"].map({"LONG": 1, "SHORT": -1, "HOLD": 0}).fillna(0).astype(int)
    df["portfolio_value"] = port_vals
    df["bnh_value"] = bnh_vals

    return df


if __name__ == "__main__":
    from signals.aligner import get_daily_sentiment
    from signals.generator import generate_signals

    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = generate_signals(get_daily_sentiment(t))
    bt = run(df)
    print(bt[["event_date", "signal", "next_day_return", "portfolio_value", "bnh_value"]].to_string(index=False))

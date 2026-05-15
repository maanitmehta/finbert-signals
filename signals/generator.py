"""
SENT-13: Signal generator — threshold-based long/short signals.

composite_score = positive - negative ∈ [-1, +1]
  ≥  long_threshold  → LONG  (bullish)
  ≤ short_threshold  → SHORT (bearish)
  otherwise          → HOLD
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
) -> pd.DataFrame:
    """Add 'signal' column to an aligned DataFrame.

    Args:
        df: Output of aligner.get_aligned() or get_daily_sentiment()
        long_threshold:  composite_score cutoff for LONG  (default 0.1)
        short_threshold: composite_score cutoff for SHORT (default -0.1)

    Returns:
        Same DataFrame with an added 'signal' column: LONG | SHORT | HOLD
    """
    df = df.copy()
    conditions = [
        df["composite_score"] >= long_threshold,
        df["composite_score"] <= short_threshold,
    ]
    df["signal"] = pd.Categorical(
        pd.Series("HOLD", index=df.index)
        .where(~conditions[0], "LONG")
        .where(~conditions[1], "SHORT"),
        categories=["LONG", "HOLD", "SHORT"],
    )
    return df


def signal_summary(df: pd.DataFrame) -> dict:
    """Return counts and mean next-day returns per signal."""
    if "signal" not in df.columns:
        df = generate_signals(df)

    summary = {}
    for sig in ["LONG", "HOLD", "SHORT"]:
        subset = df[df["signal"] == sig]["next_day_return"].dropna()
        summary[sig] = {
            "count": len(subset),
            "mean_return": round(subset.mean(), 4) if len(subset) else None,
            "hit_rate": round((subset > 0).mean(), 4) if len(subset) else None,
        }
    return summary


if __name__ == "__main__":
    import sys
    from signals.aligner import get_aligned

    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = get_aligned(t)
    if df.empty:
        print("No aligned data — run the scoring pipeline first.")
        sys.exit(0)

    df = generate_signals(df)
    print(df[["event_date", "source", "composite_score", "signal", "next_day_return"]].to_string(index=False))
    print("\nSignal summary:")
    for sig, stats in signal_summary(df).items():
        print(f"  {sig:<5}  n={stats['count']}  mean_return={stats['mean_return']}  hit_rate={stats['hit_rate']}")

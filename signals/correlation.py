"""
SENT-14: Correlation analysis — sentiment composite score vs. next-day return.

Uses both Pearson (linear) and Spearman (rank-based, more robust to outliers)
correlations with two-sided p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scipy import stats


def analyze(df: pd.DataFrame) -> dict:
    """Compute sentiment–return correlation statistics.

    Args:
        df: DataFrame with at least 'composite_score' and 'next_day_return' columns.

    Returns dict with keys:
        n, pearson_r, pearson_p, spearman_r, spearman_p,
        n_long, mean_long_return, hit_rate_long,
        n_short, mean_short_return, hit_rate_short,
        n_hold, mean_hold_return
    """
    clean = df.dropna(subset=["composite_score", "next_day_return"]).copy()

    if len(clean) < 3:
        return {"error": "insufficient data (need ≥3 observations)", "n": len(clean)}

    x = clean["composite_score"].astype(float)
    y = clean["next_day_return"].astype(float)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    def _bucket_stats(mask: pd.Series) -> dict:
        subset = clean.loc[mask, "next_day_return"]
        if subset.empty:
            return {"n": 0, "mean_return": None, "hit_rate": None}
        return {
            "n": len(subset),
            "mean_return": round(float(subset.mean()), 4),
            "hit_rate": round(float((subset > 0).mean()), 4),
        }

    long_mask  = clean["composite_score"] >= 0.1
    short_mask = clean["composite_score"] <= -0.1
    hold_mask  = ~long_mask & ~short_mask

    return {
        "n": len(clean),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "long":  _bucket_stats(long_mask),
        "short": _bucket_stats(short_mask),
        "hold":  _bucket_stats(hold_mask),
    }


def print_report(results: dict) -> None:
    if "error" in results:
        print(f"Cannot compute: {results['error']} (n={results['n']})")
        return

    print(f"Observations : {results['n']}")
    print(f"Pearson  r   : {results['pearson_r']:+.4f}  (p={results['pearson_p']:.4f})")
    print(f"Spearman r   : {results['spearman_r']:+.4f}  (p={results['spearman_p']:.4f})")
    print()
    for bucket in ("long", "short", "hold"):
        b = results[bucket]
        print(
            f"  {bucket.upper():<5}  n={b['n']}  "
            f"mean_return={b['mean_return']}  "
            f"hit_rate={b['hit_rate']}"
        )


if __name__ == "__main__":
    from signals.aligner import get_aligned

    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = get_aligned(t)
    if df.empty:
        print("No aligned data — run the scoring pipeline first.")
        sys.exit(0)

    results = analyze(df)
    print_report(results)

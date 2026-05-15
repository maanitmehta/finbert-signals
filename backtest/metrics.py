"""
SENT-16: Performance metrics from the backtest equity curve.

  total_return  — (final_value / initial_capital) - 1
  sharpe_ratio  — annualised Sharpe (no risk-free rate assumed)
  max_drawdown  — peak-to-trough decline in portfolio value
  hit_rate      — fraction of active trades where direction was correct
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def compute(
    bt_df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    ann: int = 252,
) -> dict:
    """Compute performance metrics from engine.run() output.

    Returns an empty dict if bt_df is empty or missing required columns.
    """
    if bt_df.empty or "portfolio_value" not in bt_df.columns:
        return {}

    pv = bt_df["portfolio_value"]
    total_return = float(pv.iloc[-1] / initial_capital) - 1.0

    # Daily P&L per unit capital — only on active (non-HOLD) days
    daily_rets = (bt_df["position"] * bt_df["next_day_return"].fillna(0)).astype(float)
    std = daily_rets.std()
    sharpe = float((daily_rets.mean() / std) * np.sqrt(ann)) if std > 0 else 0.0

    # Max drawdown
    rolling_peak = pv.cummax()
    max_dd = float(((pv - rolling_peak) / rolling_peak).min())

    # Hit rate — fraction of active trades in the right direction
    active = bt_df[bt_df["position"] != 0]
    if len(active):
        correct = (
            ((active["position"] == 1)  & (active["next_day_return"] > 0)) |
            ((active["position"] == -1) & (active["next_day_return"] < 0))
        )
        hit_rate: float | None = float(correct.mean())
    else:
        hit_rate = None

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate":     round(hit_rate, 4) if hit_rate is not None else None,
        "n_trades":     int((bt_df["position"] != 0).sum()),
        "n_long":       int((bt_df["position"] ==  1).sum()),
        "n_short":      int((bt_df["position"] == -1).sum()),
        "final_value":  round(float(pv.iloc[-1]), 2),
    }


if __name__ == "__main__":
    from backtest.engine import run
    from signals.aligner import get_daily_sentiment
    from signals.generator import generate_signals

    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = run(generate_signals(get_daily_sentiment(t)))
    for k, v in compute(df).items():
        print(f"  {k:<18} {v}")

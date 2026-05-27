"""
SENT-IC: Alpha Decay & PnL Attribution — IC computation module.

Information Coefficient (IC) = Spearman rank-correlation between a signal score
and a forward return.  Decay of IC across holding horizons reveals how quickly
the signal's predictive edge dissipates — the primary diagnostic for choosing
optimal rebalancing frequency.

Public API:
  load_signals_with_horizons()  join sentiment_scores with multi-horizon returns
  compute_ic_series()           IC / t-stat / p-value at each horizon
  compute_rolling_ic()          rolling IC for stability analysis
  compute_pnl_attribution()     gross alpha / timing slippage / execution drag
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Maps holding-horizon (days) → prices table column name
_RETURN_COL: dict[int, str] = {
    1:  "next_day_return",
    2:  "return_2d",
    3:  "return_3d",
    5:  "return_5d",
    10: "return_10d",
    20: "return_20d",
}

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "sentiment.db"


def load_signals_with_horizons(
    ticker: str,
    db_path: Path | str | None = None,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
) -> pd.DataFrame:
    """Load FinBERT signals joined with multi-horizon forward returns.

    Applies the same forward-fill alignment as signals/aligner.py: a sentiment
    event on date T is matched to the next available trading day within 7 days.
    Multi-horizon return columns (return_2d … return_20d) are attached from the
    prices table (populated by fetchers/prices.py::compute_and_store_horizons).

    Also computes timing_slippage per trade: the overnight gap between the
    signal-day close and the next-day open.  This captures the execution-lag
    cost of waiting for next-open fills rather than executing at signal-day close.

    Args:
        ticker:          Equity symbol (e.g. "AAPL").
        db_path:         Path to sentiment.db; defaults to data/sentiment.db.
        long_threshold:  composite_score cutoff for LONG (default 0.1).
        short_threshold: composite_score cutoff for SHORT (default -0.1).

    Returns DataFrame with columns:
        signal_date, composite_score, signal_type, direction,
        price_date, signal_close, entry_open, timing_slippage,
        next_day_return, return_2d, return_3d, return_5d, return_10d, return_20d

    Returns an empty DataFrame when fewer than 5 aligned rows exist.
    """
    db = Path(db_path or _DEFAULT_DB)

    sql = """
        SELECT
            s.event_date           AS signal_date,
            AVG(s.composite_score) AS composite_score,
            p.date                 AS price_date,
            p.close                AS signal_close,
            p.next_day_return,
            p.return_2d,
            p.return_3d,
            p.return_5d,
            p.return_10d,
            p.return_20d
        FROM sentiment_scores s
        INNER JOIN prices p
            ON  p.ticker = s.ticker
            AND p.date   = (
                SELECT MIN(p2.date)
                FROM   prices p2
                WHERE  p2.ticker = s.ticker
                  AND  p2.date  >= s.event_date
                  AND  p2.date  <= date(s.event_date, '+7 days')
                  AND  p2.next_day_return IS NOT NULL
            )
        LEFT JOIN articles a
            ON  s.source_id = a.id
            AND s.source    = 'news'
        WHERE s.ticker = ?
          AND (s.source = 'edgar' OR a.relevant = 1)
        GROUP BY p.date
        ORDER BY p.date
    """

    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(sql, conn, params=[ticker.upper()])
        prices_open = pd.read_sql_query(
            "SELECT date, open FROM prices WHERE ticker=? ORDER BY date",
            conn,
            params=[ticker.upper()],
        )

    if df.empty or len(df) < 5:
        return pd.DataFrame()

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["price_date"]  = pd.to_datetime(df["price_date"])

    # Signal classification
    df["signal_type"] = pd.cut(
        df["composite_score"],
        bins=[-np.inf, short_threshold, long_threshold, np.inf],
        labels=["SHORT", "HOLD", "LONG"],
    ).astype(str)
    df["direction"] = df["signal_type"].map({"LONG": 1, "SHORT": -1, "HOLD": 0}).astype(float)

    # Timing slippage: direction × (next_open − signal_close) / signal_close
    prices_open["date"] = pd.to_datetime(prices_open["date"])
    open_series = prices_open.set_index("date")["open"]

    def _next_open(price_date: pd.Timestamp) -> float:
        later = open_series[open_series.index > price_date]
        return float(later.iloc[0]) if not later.empty else np.nan

    df["entry_open"] = df["price_date"].apply(_next_open)
    sc = df["signal_close"]
    df["timing_slippage"] = np.where(
        sc.notna() & (sc != 0),
        df["direction"] * (df["entry_open"] - sc) / sc,
        np.nan,
    )

    return df.reset_index(drop=True)


def compute_ic_series(
    signals_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Information Coefficient at each holding horizon with significance tests.

    IC = Spearman rank-correlation(signal_score, forward_return_at_horizon).
    Spearman is used instead of Pearson because it is robust to the fat-tailed
    return distributions typical of equities, and measures rank-ordering ability —
    the question a portfolio manager cares about is 'does a higher score rank
    higher returns?', not 'is the linear relationship exactly proportional?'

    Significance test:
      t-stat = IC × √(N-2) / √(1-IC²)   ~ t(N-2) under H₀: ρ = 0
      Error bar: IC ± 1 SE, where SE ≈ √((1-IC²) / (N-2))

    The horizon where IC first becomes statistically insignificant (p > 0.05)
    defines the maximum useful holding period — the key output of this panel
    for choosing between daily, weekly, and monthly rebalancing frequencies.

    Returns DataFrame indexed by horizon with:
        ic, ic_se, t_stat, p_value, n_trades, significant
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20]

    rows = []
    for h in horizons:
        col = _RETURN_COL.get(h)
        if col is None or col not in signals_df.columns:
            rows.append({
                "horizon": h, "ic": np.nan, "ic_se": np.nan,
                "t_stat": np.nan, "p_value": 1.0, "n_trades": 0, "significant": False,
            })
            continue

        sub = signals_df.dropna(subset=["composite_score", col])
        n   = len(sub)

        if n < 5:
            rows.append({
                "horizon": h, "ic": np.nan, "ic_se": np.nan,
                "t_stat": np.nan, "p_value": 1.0, "n_trades": n, "significant": False,
            })
            continue

        ic, p_val = stats.spearmanr(sub["composite_score"], sub[col])
        ic, p_val = float(ic), float(p_val)
        denom  = max(1e-12, 1.0 - ic**2)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(denom)
        se     = np.sqrt(max(0.0, denom / max(1, n - 2)))

        rows.append({
            "horizon":    h,
            "ic":         round(ic, 4),
            "ic_se":      round(se, 4),
            "t_stat":     round(t_stat, 3),
            "p_value":    round(p_val, 4),
            "n_trades":   n,
            "significant": bool(p_val < 0.05),
        })

    return pd.DataFrame(rows).set_index("horizon")


def compute_rolling_ic(
    signals_df: pd.DataFrame,
    horizon: int = 1,
    window: int = 60,
) -> pd.Series:
    """Rolling IC at a fixed horizon over a sliding window of trades.

    Rolling IC is the primary tool for diagnosing signal stability over time.
    Periods where the rolling IC drops below zero indicate that the signal is
    actively losing money in that window — a warning of alpha decay, regime
    change, or information leakage.  The width of the window controls the
    sensitivity vs. smoothness trade-off: a narrow window (20 trades) reacts
    quickly but is noisy; a wide window (100 trades) is smoother but lags.

    The rolling axis is trade-count, not calendar time, because IC measures
    predictive accuracy per signal regardless of calendar spacing between signals.

    Args:
        signals_df: output of load_signals_with_horizons()
        horizon:    holding-period in trading days (default 1)
        window:     trades per rolling window (default 60)

    Returns a Series indexed by signal_date with rolling IC values.
    NaN is returned for the first min(window, 5) entries.
    """
    col = _RETURN_COL.get(horizon, "next_day_return")
    if col not in signals_df.columns:
        return pd.Series(dtype=float)

    sub = (
        signals_df
        .dropna(subset=["composite_score", col])
        .sort_values("signal_date")
        .reset_index(drop=True)
    )

    ics: list[float] = []
    for i in range(len(sub)):
        s     = max(0, i - window + 1)
        chunk = sub.iloc[s : i + 1]
        if len(chunk) < 5:
            ics.append(np.nan)
        else:
            ic, _ = stats.spearmanr(chunk["composite_score"], chunk[col])
            ics.append(round(float(ic), 4))

    return pd.Series(ics, index=sub["signal_date"].values, name="rolling_ic")


def compute_pnl_attribution(
    signals_df: pd.DataFrame,
    spread_bps: int = 5,
    impact_bps: int = 2,
) -> dict:
    """Decompose total strategy PnL into signal alpha, timing slippage, and execution drag.

    A PnL attribution answers: 'Where did the money come from, and how much was
    lost to execution costs?'  The three-way decomposition is standard in
    systematic trading risk-management:

      signal_alpha:    IC-weighted directional return — the gross close-to-close
                       PnL generated by following the signal.
                       = Σ direction × next_day_return  (active trades only)

      timing_slippage: Cost of executing at next-day *open* rather than at
                       signal-day *close*.  Positive values mean the stock gapped
                       in our favour; negative values (typical) mean we paid a
                       premium vs the theoretical signal-day execution.
                       = Σ direction × (entry_open − signal_close) / signal_close

      execution_drag:  Fixed transaction costs per round-trip:
                       bid-ask spread (default 5 bps) + market impact (default 2 bps).
                       = −n_active_trades × (spread_bps + impact_bps) / 10 000

      net_pnl:         signal_alpha + timing_slippage + execution_drag

    All values returned in basis points (1 bps = 0.01%).

    Args:
        signals_df:  output of load_signals_with_horizons()
        spread_bps:  assumed one-way bid-ask spread in bps (default 5)
        impact_bps:  assumed market-impact cost in bps (default 2)

    Returns dict: signal_alpha, timing_slippage, execution_drag, net_pnl (all bps), n_trades
    """
    active = signals_df[signals_df["signal_type"] != "HOLD"].copy()
    n = len(active)

    if n == 0:
        return {
            "signal_alpha": 0.0, "timing_slippage": 0.0,
            "execution_drag": 0.0, "net_pnl": 0.0, "n_trades": 0,
        }

    gross = float(
        (active["direction"].fillna(0) * active["next_day_return"].fillna(0)).sum()
    )
    slip  = float(active["timing_slippage"].fillna(0).sum())
    drag  = -n * (spread_bps + impact_bps) / 10_000
    net   = gross + slip + drag

    bps = 10_000
    return {
        "signal_alpha":    round(gross * bps, 1),
        "timing_slippage": round(slip  * bps, 1),
        "execution_drag":  round(drag  * bps, 1),
        "net_pnl":         round(net   * bps, 1),
        "n_trades":        n,
    }

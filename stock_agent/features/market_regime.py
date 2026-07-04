"""Market-regime filter for the momentum path ("chong du dinh" / avoid buying tops).

Two proven gates, combined (validated in scratch/edge_experiment.py on VN100):

1. **Index trend**: VNINDEX close > its EMA50. In the 2022 crash this alone flipped
   the momentum branch from -10.5% to +8.9% and cut drawdown roughly in half.
2. **Breadth**: fraction of the universe trading above its own SMA20. The Feb-Jun 2026
   grind exposed that VNINDEX can rise on a few mega-caps (+3%) while the average stock
   falls (-5%); an index-only gate is fooled there, a breadth gate is not.

A day is "risk-on" (regime = +1) only when BOTH the index is above its EMA50 AND breadth
is at/above the threshold; otherwise "risk-off" (-1). Everything is computed as of the
signal date d (data <= d), so attaching regime[d] to a frame is causal — no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, sma


def index_ema_regime(index_df: pd.DataFrame, ema_period: int = 50) -> dict:
    """Map each index date -> True if close > EMA(ema_period)."""
    d = index_df.sort_values("date").reset_index(drop=True).copy()
    e = ema(d["close"], ema_period)
    return {
        row_date: bool(np.isfinite(ev) and cv > ev)
        for row_date, cv, ev in zip(d["date"], d["close"], e)
    }


def breadth_series(universe_frames: dict[str, pd.DataFrame], ma: int = 20) -> dict:
    """Map each date -> fraction of symbols whose close is above their own SMA(ma)."""
    above: dict = {}
    total: dict = {}
    for _, df in universe_frames.items():
        if df is None or df.empty:
            continue
        d = df.sort_values("date").reset_index(drop=True)
        m = sma(d["close"], ma)
        for row_date, cv, mv in zip(d["date"], d["close"], m):
            if not np.isfinite(mv):
                continue
            total[row_date] = total.get(row_date, 0) + 1
            if cv > mv:
                above[row_date] = above.get(row_date, 0) + 1
    return {dt: above.get(dt, 0) / n for dt, n in total.items() if n > 0}


def market_regime_map(
    index_df: pd.DataFrame,
    universe_frames: dict[str, pd.DataFrame] | None = None,
    ema_period: int = 50,
    breadth_ma: int = 20,
    breadth_min: float = 0.4,
) -> dict:
    """Combine index trend + breadth into date -> +1 (risk-on) / -1 (risk-off).

    Set breadth_min <= 0 to disable the breadth gate (index-only).
    """
    idx_ok = index_ema_regime(index_df, ema_period)
    use_breadth = breadth_min > 0 and universe_frames
    breadth = breadth_series(universe_frames, breadth_ma) if use_breadth else {}
    regime: dict = {}
    for dt, ok in idx_ok.items():
        healthy = ok and (breadth.get(dt, 1.0) >= breadth_min if use_breadth else True)
        regime[dt] = 1 if healthy else -1
    return regime


def attach_market_regime(frame: pd.DataFrame, regime_map: dict) -> pd.DataFrame:
    """Add a ``market_regime`` column (float +1/-1) keyed by date. Missing dates -> +1
    (fail-open: absence of regime data must not silently block every trade)."""
    out = frame.copy()
    out["market_regime"] = out["date"].map(lambda d: float(regime_map.get(d, 1.0)))
    return out

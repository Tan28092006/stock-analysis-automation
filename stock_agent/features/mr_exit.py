"""Canonical mean-reversion exit replay — the ONE implementation of the "T+2 settlement
lock, then first of stop / target / T+max_hold time-stop" rule.

Before this module the same loop was copy-pasted in four places (win_probability._label_trade,
position_manager.check_positions, pipeline.forward_test._replay_mr, features.backtest) with
subtly different bar-counting, so live sell-alerts, training labels, and backtest results
could silently drift. All four now call `simulate_mr_exit` so the exit *logic* is identical;
each caller only chooses its own entry bar/price and maps the reason to its display string.

Two distinct horizons, do not confuse them:
  * settle_lock (T+2): cannot sell during the first `settle_lock` bars after entry (VN
    settlement). Stop/target are only checked from entry_idx + settle_lock onward.
  * max_hold (T+15 for MR): force-exit at entry_idx + max_hold if no stop/target hit first.
"""
from __future__ import annotations

import pandas as pd


def simulate_mr_exit(frame: pd.DataFrame, entry_idx: int, stop: float, target: float,
                     max_hold: int, settle_lock: int = 2):
    """Replay one MR trade's exit on `frame` (needs low/high/close columns, sorted by date).

    The position is entered at bar ``entry_idx``. It cannot be sold for the first
    ``settle_lock`` bars; then the FIRST of stop (low<=stop) / target (high>=target) wins;
    otherwise it time-exits at ``entry_idx + max_hold`` (close). Prices returned are the
    raw stop/target level or the horizon close — callers layer slippage/price-limits/costs.

    Returns ``(exit_idx, exit_price, reason, resolved)``:
      * reason ∈ {"stop", "target", "time"}; callers map to their own labels.
      * resolved=True when a stop/target fired or the full max_hold window was in the data.
      * resolved=False means the horizon runs past the available data with no stop/target yet
        → the position is still open. exit_idx/exit_price then point at the LAST available bar
        (a partial mark), so training labels can use it while live/forward callers treat
        resolved=False as "pending / keep holding".
    """
    n = len(frame)
    if entry_idx >= n:
        return None, None, "pending", False
    lo = frame["low"].to_numpy()
    hi = frame["high"].to_numpy()
    cl = frame["close"].to_numpy()
    horizon = entry_idx + max_hold
    last = min(horizon, n - 1)
    for j in range(entry_idx + settle_lock, last + 1):
        if stop and lo[j] <= stop:
            return j, float(stop), "stop", True
        if target and hi[j] >= target:
            return j, float(target), "target", True
    if horizon <= n - 1:
        return horizon, float(cl[horizon]), "time", True
    return n - 1, float(cl[n - 1]), "time", False

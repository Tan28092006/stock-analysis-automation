"""Canonical bracket exit replay: "T+2 settlement lock, then first of stop / target,
else a time-stop at entry_idx + max_hold (close)".

This module owns the exit *rule*. It does NOT own entry fills, gap fills, price-limit
clamps, liquidity, settlement of cash, or costs: each caller layers those itself, and they
still differ between callers (see "Paths that do NOT call this function" below).

Callers (each picks its own entry bar/price, horizon and maps `reason` to its labels):
  * features/backtest.py            run_backtest            (calendar horizon -> bars)
  * agents/parallel_engine.py       build_labeled_dataset_fast (calendar horizon -> bars;
                                    the legacy `net_t2_*` training labels)
  * features/win_probability.py     _trade_label_record     (max_hold in bars)
  * features/position_manager.py    check_positions         (anchored on recorded entry)
  * pipeline/forward_test.py        _replay_mr              (max_hold in bars)
  * pipeline/paper_scoring.py       _outcome                (max_hold in bars; adds
                                    min(stop, open) gap fill and a locked-bar check)
Calendar-horizon callers pass max_hold = bar distance to the first/exact bar on the
calendar exit date, so a missing bar (suspension, or a holiday absent from
HOSE_HOLIDAYS) changes the bar count; bar-horizon callers always count bars.

Paths that do NOT call this function (different exit semantics, by design or debt):
  * features/backtest.py run_portfolio_backtest / _close_due_positions: stop/target are
    checked ONLY on the planned exit date, then FORCED_CLOSE at the end of the sample.
  * scripts/period_backtest.py replay_mr (research Broker): T+3 share availability,
    opening-price fill gate, gap stop at min(stop, open), fixed_hold policy.
  * pipeline/label_resolver.py: close-to-close at the horizon, no stop/target.
  * research_ml/labels.py: fixed-horizon close label (e + horizon), no stop/target.

Two distinct horizons, do not confuse them:
  * settle_lock (T+2): cannot sell during the first `settle_lock` bars after entry (VN
    settlement). Stop/target are only checked from entry_idx + settle_lock onward. Daily
    bars cannot show that T+2 shares are only sellable from the afternoon session.
  * max_hold: force-exit at entry_idx + max_hold if no stop/target hit first. This
    function does not stop a caller from passing max_hold < settle_lock.
A falsy stop/target (0/None) disables that leg.
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
        (a partial mark). Training MUST reject it; live/forward callers treat
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

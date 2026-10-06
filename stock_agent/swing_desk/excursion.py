"""Excursion statistics under the T+2.5 lock (descriptive, no signal).

Entry e buys at the open of session e. Shares are unsellable until the afternoon of
e+2, so the *locked* adverse excursion is min(L_e, L_{e+1}) / O_e - 1. Percent units.
"p80" of an adverse excursion is the level that 80% of entries stayed above, i.e. the
20th percentile of the signed value (more negative = worse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _q(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level)) if len(values) else float("nan")


def excursion_stats(df: pd.DataFrame, lookback: int = 250) -> dict:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    locked_idx = np.arange(max(0, n - 1 - lookback), n - 1)
    locked = np.array([(min(lo[e], lo[e + 1]) / o[e] - 1) * 100 for e in locked_idx])
    breach = [e for e, v in zip(locked_idx, locked) if v <= -3.0 and e + 2 < n]
    day2 = np.array([(c[e + 2] / o[e] - 1) * 100 for e in breach])

    w_idx = np.arange(max(0, n - 4 - lookback), n - 4)
    mfe = np.array([(h[e:e + 5].max() / o[e] - 1) * 100 for e in w_idx])
    mae = np.array([(lo[e:e + 5].min() / o[e] - 1) * 100 for e in w_idx])

    return {
        "locked_mae_p50": _q(locked, 0.5),
        "locked_mae_p80": _q(locked, 0.2),
        "locked_mae_p90": _q(locked, 0.1),
        "locked_breach_3pct_rate": float((locked <= -3.0).mean() * 100) if len(locked) else float("nan"),
        "day2_ret_given_breach": float(np.median(day2)) if len(day2) else float("nan"),
        "n_locked": int(len(locked)),
        "n_breach": int(len(breach)),
        "mfe5_p50": _q(mfe, 0.5),
        "mae5_p50": _q(mae, 0.5),
        "n_window5": int(len(w_idx)),
        "_locked_mae": locked.tolist(),
    }

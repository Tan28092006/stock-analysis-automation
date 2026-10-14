"""Per-ticker and market indicators, computed causally from bars <= as_of.

Reuses `stock_agent.features.indicators`; RSI(2) reuses `swing_scan._rsi` exactly
(the feature-library `rsi` fills NaN with 50 and is not identical). Percent units for
returns/ratios named `*_pct` or `ret_*`.
"""
from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd

from ..features import indicators as base
from ..features.swing_scan import _rsi
from .excursion import excursion_stats


def _f(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _ret(c: np.ndarray, k: int) -> float:
    return (c[-1] / c[-1 - k] - 1) * 100 if len(c) > k else float("nan")


def _sessions(df: pd.DataFrame, as_of) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"].astype(str).str[:10]).dt.date
    if as_of is not None:
        as_of = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of)[:10])
        out = out[out["date"] <= as_of]
    return out.sort_values("date").reset_index(drop=True)


# Fields whose window can exceed the 650-day snapshot (about 440 sessions) or that the desk
# defines as long-window: taken from the long series when one is supplied (`long_df`).
EXCURSION_FIELDS = ("locked_mae_p50", "locked_mae_p80", "locked_mae_p90", "locked_breach_3pct_rate",
                    "day2_ret_given_breach", "n_locked", "n_breach", "mfe5_p50", "mae5_p50", "n_window5")
LONG_WINDOW_FIELDS = ("sma200", "high_250") + EXCURSION_FIELDS


def long_series(df: pd.DataFrame, long_df: pd.DataFrame | None, as_of=None) -> pd.DataFrame:
    """The series long-window fields use: `long_df` (cut at as_of) when given, else `df` itself.

    The caller guarantees the long series equals `df` over df's sessions (history.compare); here
    only the last session is re-checked so a misaligned series can never be used silently.
    """
    if long_df is None:
        return df
    long = _sessions(long_df, as_of)
    if long.empty or long["date"].iloc[-1] != df["date"].iloc[-1] or len(long) < len(df):
        raise ValueError("long series does not end on the short series' last session")
    return long


def _long_window(df: pd.DataFrame, excursion_lookback: int, short_rows: int | None = None) -> dict:
    """Long-window fields of `df`. Max/min/quantile fields are exact, so a longer series that agrees
    with the snapshot gives bit-identical values; the rolling mean is evaluated over the last
    max(short_rows, 200) closes so it is also bit-identical when the snapshot already covers it."""
    c = df["close"].astype(float)
    if short_rows is not None:
        c = c.tail(max(short_rows, 200))
    out = {"sma200": _f(base.sma(c, 200).iloc[-1]),
           "high_250": _f(df["high"].tail(250).max()) if len(df) >= 250 else float("nan")}
    ex = excursion_stats(df, excursion_lookback)
    ex.pop("_locked_mae", None)
    out.update(ex)
    return out


def compute_ticker(df: pd.DataFrame, as_of=None, *, limit_proxy_pct: float = 6.5,
                   excursion_lookback: int = 250, long_df: pd.DataFrame | None = None) -> dict:
    """Indicators at the last session <= as_of.

    Short-window fields always come from `df` (the verified snapshot). With `long_df` (a longer
    series equal to `df` over df's sessions), LONG_WINDOW_FIELDS come from `long_df`; without it
    the output is exactly the pre-history behaviour.
    """
    df = _sessions(df, as_of)
    if len(df) < 3:
        raise ValueError("at least 3 completed bars required")
    lw = _long_window(long_series(df, long_df, as_of), excursion_lookback, len(df) if long_df is not None else None)
    c = df["close"].astype(float)
    cv = c.to_numpy()
    vol = df["volume"].astype(float)
    bb = base.bollinger(c, 20, 2)
    adx = base.adx(df, 14)
    vsa = base.add_vsa_features(df)
    atr14 = _f(base.atr(df, 14).iloc[-1])
    close = float(cv[-1])
    lower, upper = _f(bb["bb_lower"].iloc[-1]), _f(bb["bb_upper"].iloc[-1])
    rets = np.diff(cv) / cv[:-1]
    streak = 0
    for r in rets[::-1]:
        if r > 0:
            streak += 1
        else:
            break
    high_20 = _f(df["high"].tail(20).max()) if len(df) >= 20 else float("nan")
    ret_prev = _ret(cv[:-1], 1) if len(cv) > 2 else float("nan")
    out = {
        "date": str(df["date"].iloc[-1]),
        "rows": int(len(df)),
        "close": close,
        "prev_close": float(cv[-2]),
        "open": float(df["open"].iloc[-1]),
        "high": float(df["high"].iloc[-1]),
        "low": float(df["low"].iloc[-1]),
        "ret_1d": _ret(cv, 1),
        "ret_3d": _ret(cv, 3),
        "ret_5d": _ret(cv, 5),
        "ret_20d": _ret(cv, 20),
        "gap_open_1d": (float(df["open"].iloc[-1]) / cv[-2] - 1) * 100,
        "sma5": _f(base.sma(c, 5).iloc[-1]),
        "sma20": _f(base.sma(c, 20).iloc[-1]),
        "sma50": _f(base.sma(c, 50).iloc[-1]),
        "sma200": lw["sma200"],
        "ema21": _f(base.ema(c, 21).iloc[-1]),
        "ema50": _f(base.ema(c, 50).iloc[-1]),
        "rsi2": _f(_rsi(c, 2).iloc[-1]),
        "rsi14": _f(base.rsi(c, 14).iloc[-1]),
        "atr14": atr14,
        "atr_pct": atr14 / close * 100 if math.isfinite(atr14) else float("nan"),
        "bb_lower": lower,
        "bb_upper": upper,
        "pct_b": (close - lower) / (upper - lower) if math.isfinite(lower) and upper > lower else float("nan"),
        "adx14": _f(adx["adx14"].iloc[-1]),
        "plus_di14": _f(adx["plus_di14"].iloc[-1]),
        "minus_di14": _f(adx["minus_di14"].iloc[-1]),
        "vol_ratio_20": float(vol.iloc[-1] / vol.tail(20).mean()) if len(vol) >= 20 and vol.tail(20).mean() > 0 else float("nan"),
        "avg_value_20_bn": float((c * vol).tail(20).mean() / 1e9) if len(df) >= 20 else float("nan"),
        "high_20": high_20,
        "low_20": _f(df["low"].tail(20).min()) if len(df) >= 20 else float("nan"),
        "high_250": lw["high_250"],
        "dist_high_20_pct": (high_20 / close - 1) * 100 if math.isfinite(high_20) else float("nan"),
        "limit_up_t": bool(_ret(cv, 1) >= limit_proxy_pct),
        "limit_down_t": bool(_ret(cv, 1) <= -limit_proxy_pct),
        "limit_up_t1": bool(math.isfinite(ret_prev) and ret_prev >= limit_proxy_pct),
        "up_streak": int(streak),
        "vsa_stopping_volume": int(vsa["vsa_stopping_volume"].iloc[-1]),
        "vsa_no_supply": int(vsa["vsa_no_supply"].iloc[-1]),
    }
    out.update({k: lw[k] for k in EXCURSION_FIELDS})
    out["stop_atr_multiple"] = 3.0 / out["atr_pct"] if out["atr_pct"] and math.isfinite(out["atr_pct"]) else float("nan")
    return out


def flow_fields(flows: pd.DataFrame, symbol: str, as_of: str, avg_value_20_bn: float) -> dict:
    """Foreign / proprietary net values (bn VND) from `foreign_flows.load_flows` output.

    Missing values stay NaN (never zero). < 20 sessions -> INSUFFICIENT_HISTORY.
    """
    empty = {"f_net_1d_bn": float("nan"), "f_net_5d_bn": float("nan"), "f_net_20d_bn": float("nan"),
             "f_net_5d_pct_adv": float("nan"), "td_net_5d_bn": float("nan"),
             "flow_history_sessions": 0, "flow_status": "UNVERIFIED"}
    if flows is None or flows.empty:
        return empty
    rows = flows[(flows["symbol"] == symbol) & (flows["date"].astype(str) <= as_of)].sort_values("date")
    if rows.empty:
        return empty
    f = rows["f_net_val"].astype(float)
    td = rows["td_net_val"].astype(float)
    sessions = int(rows["date"].nunique())
    last_is_asof = str(rows["date"].iloc[-1]) == as_of

    def tail_sum(series, k):
        part = series.tail(k)
        return float(part.sum()) if len(part) == k and part.notna().all() else float("nan")

    f5 = tail_sum(f, 5)
    return {
        "f_net_1d_bn": float(f.iloc[-1]) if last_is_asof else float("nan"),
        "f_net_5d_bn": f5,
        "f_net_20d_bn": tail_sum(f, 20),
        "f_net_5d_pct_adv": f5 / (5 * avg_value_20_bn) * 100 if math.isfinite(f5) and avg_value_20_bn else float("nan"),
        "td_net_5d_bn": tail_sum(td, 5),
        "flow_history_sessions": sessions,
        "flow_status": ("OK" if sessions >= 20 else "INSUFFICIENT_HISTORY") if last_is_asof else "STALE",
    }


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

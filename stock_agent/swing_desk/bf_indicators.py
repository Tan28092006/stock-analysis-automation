"""Bottom-fish per-ticker fields (T12 spec §3, BF-doc §8). Pure and causal: bars <= as_of only.

Separate from `indicators.compute_ticker` so swing outputs never change. Percent units as in
`indicators.py`. Missing values: NaN for floats, None for dates/counts/state that do not exist.
"""
from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd

from ..features import indicators as base
from .indicators import _sessions


def tick_size(price: float) -> int:
    """HOSE stock price step: < 10,000 -> 10; 10,000-49,950 -> 50; >= 50,000 -> 100 VND.

    Source: [R4] CTS/VBSE "Quy định giao dịch chứng khoán tại HSX" (KRX, 08/2025), section on
    price steps (bước giá), cited in docs/audits/2026-10-06-swing-desk-signal-research.md §14.
    """
    if price < 10_000:
        return 10
    if price < 50_000:
        return 50
    return 100


def floor_price(ref: float, band_pct: float = 7.0) -> float:
    """Floor price for the next session from reference `ref` (last close).

    The board floor stays inside the ±band: ref x (1 - band) is rounded UP to the tick. The T12
    spec text said "rounded down"; the PNJ prints in BF-doc §10 contradict that (24,750 -> 23,050,
    21,650 -> 20,150, 33,000 -> 30,700 are all rounded up), so the observed rule is used.
    Snapshot prices are provider-adjusted, so this is a proxy on ex-rights days.
    """
    raw = ref * (1 - band_pct / 100)
    tick = tick_size(raw)
    return float(math.ceil(raw / tick - 1e-9) * tick)


def _streak(mask: np.ndarray) -> int:
    n = 0
    for v in mask[::-1]:
        if not v:
            break
        n += 1
    return n


def _nan() -> float:
    return float("nan")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _flows(flows: pd.DataFrame | None, symbol: str | None, as_of: str, volume_t: float) -> dict:
    out = {"f_sell_streak": None, "f_net_turn_positive": None, "f_net_1d_pct_vol": _nan(),
           "flow_status": "UNVERIFIED"}
    if flows is None or symbol is None or flows.empty:
        return out
    rows = flows[(flows["symbol"] == symbol) & (flows["date"].astype(str).str[:10] <= as_of)].sort_values("date")
    if rows.empty or str(rows["date"].iloc[-1])[:10] != as_of:
        return {**out, "flow_status": "STALE" if not rows.empty else "UNVERIFIED"}
    net = rows["f_net_val"].astype(float).to_numpy()
    sell = net < 0
    streak = _streak(sell)
    prev_streak = _streak(sell[:-1])
    nv = float(rows["f_net_vol"].iloc[-1]) if "f_net_vol" in rows else _nan()
    sessions = int(rows["date"].nunique())
    return {"f_sell_streak": streak, "f_net_turn_positive": bool(net[-1] > 0 and prev_streak >= 5),
            "f_net_1d_pct_vol": nv / volume_t * 100 if _finite(nv) and volume_t else _nan(),
            "flow_status": "OK" if sessions >= 20 else "INSUFFICIENT_HISTORY"}


def _index_fields(df: pd.DataFrame, idx: pd.DataFrame | None, cfg: dict, ret_10d: float) -> dict:
    c = cfg["classification"]
    out = {"idx_ret_5d": _nan(), "idx_ret_10d": _nan(), "beta_60": None, "beta_obs": 0,
           "idio_ret_10d": None, "idio_share": None, "index_atr_ratio": _nan(), "index_status": "OK"}
    if idx is None or idx.empty:
        out["index_status"] = "UNVERIFIED: no VNINDEX frame"
        return out
    if idx["date"].iloc[-1] != df["date"].iloc[-1]:
        out["index_status"] = (f"UNVERIFIED: index dates do not align (index {idx['date'].iloc[-1]}, "
                               f"ticker {df['date'].iloc[-1]})")
        return out
    ic = idx["close"].astype(float).to_numpy()
    if len(ic) > 5:
        out["idx_ret_5d"] = (ic[-1] / ic[-6] - 1) * 100
    if len(ic) > 10:
        out["idx_ret_10d"] = (ic[-1] / ic[-11] - 1) * 100
    s_ret = pd.Series(df["close"].astype(float).to_numpy(), index=df["date"]).pct_change()
    i_ret = pd.Series(ic, index=idx["date"]).pct_change()
    gap = int(c.get("beta_gap_sessions", 10))
    if len(df) > gap:
        end = df["date"].iloc[-1 - gap]
        joined = pd.concat([s_ret, i_ret], axis=1, join="inner").dropna()
        joined = joined[joined.index <= end].tail(int(c["beta_window"]))
        out["beta_obs"] = int(len(joined))
        if len(joined) >= int(c["beta_min_obs"]):
            x, y = joined.iloc[:, 1].to_numpy(), joined.iloc[:, 0].to_numpy()
            var = float(np.var(x))
            if var > 0:
                out["beta_60"] = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    if out["beta_60"] is not None and _finite(ret_10d) and _finite(out["idx_ret_10d"]):
        out["idio_ret_10d"] = ret_10d - out["beta_60"] * out["idx_ret_10d"]
        out["idio_share"] = out["idio_ret_10d"] / ret_10d if abs(ret_10d) >= 1 else None
    atr_pct = base.atr(idx, 14) / idx["close"].astype(float) * 100
    window = atr_pct.tail(int(cfg["regime"]["atr_median_window"])).dropna()
    if len(window) and _finite(float(atr_pct.iloc[-1])):
        med = float(window.median())
        out["index_atr_ratio"] = float(atr_pct.iloc[-1]) / med if med > 0 else _nan()
    return out


def compute_bottomfish_fields(df: pd.DataFrame, index_df: pd.DataFrame | None = None, *, cfg: dict,
                              as_of=None, sector_mean_ret_10d: float | None = None,
                              flows: pd.DataFrame | None = None, symbol: str | None = None) -> dict:
    df = _sessions(df, as_of)
    if len(df) < 3:
        raise ValueError("at least 3 completed bars required")
    idx = _sessions(index_df, as_of) if index_df is not None else None
    lim = cfg["limit"]
    o, h, lo, c, v = (df[k].astype(float).to_numpy() for k in ("open", "high", "low", "close", "volume"))
    n = len(df)
    dates = [str(d) for d in df["date"]]
    ret1 = np.full(n, np.nan)
    ret1[1:] = (c[1:] / c[:-1] - 1) * 100
    floor = np.nan_to_num(ret1, nan=0.0) <= lim["floor_proxy_pct"]
    flat = np.isclose(o, h, rtol=1e-9, atol=0) & np.isclose(h, lo, rtol=1e-9, atol=0) & np.isclose(lo, c, rtol=1e-9, atol=0)
    locked = floor & flat
    lookback = int(lim.get("floor_lookback_sessions", 60))
    floor_idx = [i for i in range(max(0, n - lookback), n) if floor[i]]
    last_floor = floor_idx[-1] if floor_idx else None
    since = n - 1 - last_floor if last_floor is not None else None

    ep = cfg["episode"]
    high_roll = pd.Series(h).rolling(int(ep["lookback_high_sessions"]), min_periods=1).max().to_numpy()
    near = np.where(c >= ep["reset_recovery_pct"] * high_roll)[0]
    start = int(near[-1]) + 1 if len(near) else 0
    if start <= n - 1:
        ep_sessions = n - start
        panic_low = float(lo[start:].min())
        depth = (c[-1] / c[start - 1] - 1) * 100 if start >= 1 else _nan()
        new_low_2 = bool(min(lo[-2], lo[-1]) <= panic_low)
        ep_date = dates[start]
    else:
        ep_sessions, panic_low, depth, new_low_2, ep_date = 0, None, 0.0, False, None

    high_60 = float(h[-60:].max())
    dd60 = (c[-1] / high_60 - 1) * 100
    high_250 = float(h[-250:].max()) if n >= 250 else _nan()
    higher_low = bool(lo[-1] > lo[-2])
    stab = cfg["stabilisation"]
    locked_streak = _streak(locked)
    if locked_streak > 0:
        state = "LOCKED"
    elif floor[-1]:
        state = "FALLING"
    elif last_floor is None and dd60 > ep["min_drawdown_pct"]:
        state = "NONE"
    elif since is not None and since < stab["min_sessions_since_floor"]:
        state = "STAB1"
    elif (higher_low or not stab["require_higher_low"]) and \
            (not new_low_2 or not stab["require_no_new_episode_low"]):
        state = "STAB2"
    else:
        state = "WEAK"

    ret_10d = (c[-1] / c[-11] - 1) * 100 if n > 10 else _nan()
    prior = v[-21:-1]
    vr = float(v[-1] / prior.mean()) if n >= 21 and prior.mean() > 0 else _nan()
    floor_today = floor_price(c[-2], lim["band_pct"])
    capit = bool(_finite(vr) and vr >= 3 and (ret1[-1] <= -4 or (o[-1] <= floor_today * 1.001 and h[-1] > o[-1])))
    sma50 = float(base.sma(df["close"].astype(float), 50).iloc[-1])
    atr14 = float(base.atr(df, 14).iloc[-1])
    rng = h[-1] - lo[-1]
    out = {
        "date": dates[-1],
        "close": float(c[-1]),
        "ret_1d": float(ret1[-1]),
        "floor_t": bool(floor[-1]),
        "locked_floor_t": bool(locked[-1]),
        "locked_floor_t1": bool(locked[-2]),
        "floor_streak": _streak(floor),
        "locked_floor_streak": locked_streak,
        "floors_in_last_5": int(floor[-5:].sum()),
        "floors_in_last_10": int(floor[-10:].sum()),
        "last_floor_date": dates[last_floor] if last_floor is not None else None,
        "sessions_since_floor": since,
        "floor_price_today": floor_today,
        "floor_price_next": floor_price(c[-1], lim["band_pct"]),
        "higher_low_t": higher_low,
        "low": float(lo[-1]),
        "prev_low": float(lo[-2]),
        "close_pos_in_range": float((c[-1] - lo[-1]) / rng) if rng > 0 else None,
        "high_60": high_60,
        "drawdown_60_pct": dd60,
        "high_250": high_250,
        "drawdown_250_pct": (c[-1] / high_250 - 1) * 100 if _finite(high_250) else _nan(),
        "episode_start_date": ep_date,
        "episode_sessions": ep_sessions,
        "episode_depth_pct": depth,
        "panic_low": panic_low,
        "new_episode_low_2": new_low_2,
        "stab_state": state,
        "ret_10d": ret_10d,
        "sector_rel_ret_10d": (ret_10d - sector_mean_ret_10d) if _finite(sector_mean_ret_10d) and _finite(ret_10d) else None,
        "vol_ratio_20_prior": vr,
        "capitulation_t": capit,
        "dist_sma50_atr": (c[-1] - sma50) / atr14 if _finite(sma50) and _finite(atr14) and atr14 > 0 else _nan(),
        "avg_value_20_bn": float((c[-20:] * v[-20:]).mean() / 1e9) if n >= 20 else _nan(),
    }
    out.update(_index_fields(df, idx, cfg, ret_10d))
    out.update(_flows(flows, symbol, dates[-1], float(v[-1])))
    return out


def episode_id(ticker: str, fields: dict) -> str | None:
    start = fields.get("episode_start_date")
    return f"{ticker.upper()}-{date.fromisoformat(start):%Y%m%d}" if start else None

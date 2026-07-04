"""Edge-truth experiment: momentum vs mean-reversion vs classic edges.

Goal: confront whether ANY simple, well-known edge has POSITIVE net expectancy
on VN30, especially during downtrends, and whether a VNINDEX EMA50 regime filter
helps avoid "buying the top" (chong du dinh).

Data: data/raw/prices_hist/*.csv (adjusted, VND) fetched 2020->now, + VNINDEX.csv.
Indicators: reuse stock_agent.features.indicators (same math as production).
Execution model (honest, no leakage):
  - Signal on close of day t (uses only data <= t).
  - Enter at OPEN of t+1.
  - T+2 settlement: cannot exit before entry_idx + 2 sessions.
  - Exit: stop-loss / take-profit (ATR-based) / target (mean-revert) / time-stop.
  - Costs: round-trip 0.60% (commission 0.15%x2 + tax 0.1% + slippage 0.1%x2).

Metrics per (strategy, period, regime-filter):
  n trades, win rate, avg net %, median, expectancy(=avg), profit factor,
  and a portfolio proxy = compound 20%-per-trade equity in exit-date order + max DD.
Benchmarks: VNINDEX buy&hold, VN30 equal-weight buy&hold over the same window.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.features.indicators import ema, sma, rsi, macd, bollinger, atr, ichimoku, adx

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
COST_PCT = 0.60          # round-trip, % of notional
MIN_HOLD = 2             # T+2 settlement (sessions after entry before you may sell)
PORTFOLIO_F = 0.20       # fraction of equity per trade in the equity proxy

PERIODS = {
    "2022_downtrend (Apr-Nov)": (date(2022, 4, 1), date(2022, 11, 30)),
    "2025Nov-2026Jan uptrend": (date(2025, 11, 1), date(2026, 1, 31)),
    "2026 Feb-Jun": (date(2026, 2, 1), date(2026, 6, 30)),
    "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
}


# --------------------------------------------------------------------------
# Indicator prep
# --------------------------------------------------------------------------
def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    d["ema9"], d["ema21"], d["ema50"] = ema(c, 9), ema(c, 21), ema(c, 50)
    d["sma5"], d["sma20"], d["sma200"] = sma(c, 5), sma(c, 20), sma(c, 200)
    d["rsi14"], d["rsi2"] = rsi(c, 14), rsi(c, 2)
    d["macd_hist"] = macd(c)["macd_hist"]
    bb = bollinger(c, 20, 2.0)
    d["bb_lower"], d["bb_mid"] = bb["bb_lower"], bb["bb_mid"]
    d["atr"] = atr(d, 14)
    ich = ichimoku(d)
    d["tenkan"], d["kijun"] = ich["tenkan"], ich["kijun"]
    d["cloud_top"] = ich[["senkou_a", "senkou_b"]].max(axis=1)
    d["cloud_bottom"] = ich[["senkou_a", "senkou_b"]].min(axis=1)
    # VSA stopping volume (same formula as production indicators.add_vsa_features)
    spread = (h - l)
    spread_ratio = spread / spread.rolling(20, min_periods=5).mean().replace(0, np.nan)
    close_pos = (c - l) / spread.replace(0, np.nan)
    vr5 = v / v.rolling(20, min_periods=5).mean().replace(0, np.nan)
    d["vsa_stop"] = ((c < d["open"]) & (vr5 >= 1.5) &
                     (spread_ratio <= 1.0) & (close_pos >= 0.4)).astype(int)
    d["vol_ma20"] = v.rolling(20, min_periods=20).mean()
    d["vol_ratio"] = v / d["vol_ma20"]
    d["high20_prior"] = h.rolling(20, min_periods=20).max().shift(1)
    d["prev_close"] = c.shift(1)
    ad = adx(d, 14)
    d["adx14"] = ad["adx14"].values
    d["plus_di14"] = ad["plus_di14"].values
    d["minus_di14"] = ad["minus_di14"].values
    return d


# --------------------------------------------------------------------------
# Strategy entry signals -> boolean Series aligned to df index (signal at t)
# Each returns (entry_mask, exit_spec) where exit_spec configures the simulator.
# --------------------------------------------------------------------------
def strat_momentum(d):
    m = (d["close"] > d["ema50"]) & (d["ema9"] > d["ema21"]) & \
        (d["close"] > d["high20_prior"]) & d["rsi14"].between(50, 72) & (d["vol_ratio"] >= 1.2)
    return m, dict(max_hold=10, stop_atr=1.5, tp_atr=2.5, target=None)

def strat_mean_reversion(d):
    # Bat day: oversold + at/below lower band + up-reversal + volume climax
    m = (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 1.01) & \
        (d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="ema21")

def strat_ichimoku_rsi(d):
    m = (d["close"] > d["cloud_top"]) & (d["tenkan"] > d["kijun"]) & \
        (d["close"] > d["kijun"]) & d["rsi14"].between(45, 70)
    return m, dict(max_hold=10, stop_atr=1.5, tp_atr=2.5, target=None)

def strat_golden_cross(d):
    cross = (d["ema9"] > d["ema21"]) & (d["ema9"].shift(1) <= d["ema21"].shift(1))
    m = cross & (d["close"] > d["ema50"])
    return m, dict(max_hold=15, stop_atr=1.5, tp_atr=3.0, target=None)

def strat_connors_rsi2(d):
    # Classic Connors mean-reversion IN uptrend: close>SMA200 & RSI2<10, exit close>SMA5
    m = (d["close"] > d["sma200"]) & (d["rsi2"] < 10)
    return m, dict(max_hold=10, stop_atr=2.5, tp_atr=None, target="sma5")

def strat_rsi2_dip_nofilter(d):
    # Pure oversold bounce (no trend filter) -> bottom-fishing in any regime
    m = (d["rsi2"] < 5)
    return m, dict(max_hold=10, stop_atr=2.5, tp_atr=None, target="sma5")

def strat_mr_ic(d):
    # IC-informed bat day: rsi2 is the strongest short-term reversal signal (T+2 IC -0.032).
    # Require reversal up-bar + volume climax (capitulation), and EXIT FAST because the
    # bounce is short and in a crash deep dips keep falling over ~10 sessions (falling knife).
    m = (d["rsi2"] < 10) & (d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5) & \
        (d["close"] <= d["bb_lower"] * 1.02)
    return m, dict(max_hold=5, stop_atr=2.0, tp_atr=None, target="ema21")

def _mr_base_mask(d):
    return (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 1.01) & \
           (d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)

def strat_mr_kijun_target(d):
    # Ichimoku essence #1: Kijun (26-session midpoint) as the mean-reversion anchor/target
    m = _mr_base_mask(d)
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="kijun")

def strat_mr_vsa_confirm(d):
    # VSA essence: stopping-volume bar (absorption) as ALTERNATIVE confirmation to
    # reversal+climax -> can enter on the capitulation bar itself (earlier entry)
    base_confirm = (d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)
    m = (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 1.01) & \
        (base_confirm | (d["vsa_stop"] == 1))
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="ema21")

def strat_mr_rr_gate(d):
    # Mode-1's heaviest criterion (risk_reward) transplanted: bounce room to EMA21
    # must be at least the 1.5*ATR stop distance (RR >= 1)
    m = _mr_base_mask(d) & ((d["ema21"] - d["close"]) >= 1.5 * d["atr"])
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="ema21")

def strat_mr_cloud_clear(d):
    # Ichimoku essence #2: cloud overhead = resistance. Reject setups whose path to the
    # EMA21 target is blocked by the cloud (cloud bottom sits between close and target).
    blocked = (d["cloud_bottom"] > d["close"]) & (d["cloud_bottom"] < d["ema21"])
    m = _mr_base_mask(d) & ~blocked
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="ema21")

def strat_mr_hybrid(d):
    # All transplants together: VSA-or-reversal confirm + RR gate + cloud-clear + Kijun target
    blocked = (d["cloud_bottom"] > d["close"]) & (d["cloud_bottom"] < d["kijun"])
    base_confirm = (d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)
    m = (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 1.01) & \
        (base_confirm | (d["vsa_stop"] == 1)) & \
        ((d["kijun"] - d["close"]) >= 1.5 * d["atr"]) & ~blocked
    return m, dict(max_hold=8, stop_atr=1.5, tp_atr=None, target="kijun")

def strat_mom_pullback(d):
    # Momentum engine: BUY PULLBACKS in a confirmed uptrend (IC: momentum works at T+10;
    # the Nov25-Jan26 test showed breakout-chasing loses but dip-buying-in-uptrend wins).
    # Ride the trend, exit on EMA21 break. Longer hold than MR.
    up = (d["close"] > d["ema50"]) & (d["ema21"] > d["ema50"])
    pullback = d["rsi14"].between(40, 58) & (d["close"] > d["prev_close"]) & (d["close"] <= d["ema21"] * 1.03)
    strong = d["adx14"] >= 18
    m = up & pullback & strong
    return m, dict(max_hold=20, stop_atr=2.5, tp_atr=None, target=None, trail_break="ema21")

def strat_mom_breakout_ema(d):
    up = (d["close"] > d["ema50"]) & (d["ema9"] > d["ema21"])
    bo = (d["close"] >= d["high20_prior"]) & (d["adx14"] >= 20) & (d["vol_ratio"] >= 1.2)
    m = up & bo
    return m, dict(max_hold=20, stop_atr=2.5, tp_atr=None, target=None, trail_break="ema21")

STRATEGIES = {
    "mean_reversion(dip)": strat_mean_reversion,
    "MR+hybrid(all)": strat_mr_hybrid,
    "mom_pullback(uptrend)": strat_mom_pullback,
    "mom_breakout_ema": strat_mom_breakout_ema,
}


# --------------------------------------------------------------------------
# Trade simulator for one symbol
# --------------------------------------------------------------------------
def simulate_symbol(d, entry_mask, spec, start, end, regime_ok=None):
    trades = []
    n = len(d)
    i = 0
    while i < n - 1:
        sig_date = d.at[i, "date"]
        if not (start <= sig_date <= end) or not bool(entry_mask.iat[i]):
            i += 1
            continue
        # need valid indicators
        if not np.isfinite(d.at[i, "atr"]) or d.at[i, "atr"] <= 0:
            i += 1
            continue
        # regime filter (VNINDEX EMA50) keyed by signal date
        if regime_ok is not None and not regime_ok.get(sig_date, False):
            i += 1
            continue
        entry_idx = i + 1
        entry = float(d.at[entry_idx, "open"])
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        atr_v = float(d.at[i, "atr"])
        stop = entry - spec["stop_atr"] * atr_v if spec["stop_atr"] else None
        tp = entry + spec["tp_atr"] * atr_v if spec["tp_atr"] else None
        exit_idx = min(entry_idx + spec["max_hold"], n - 1)
        exit_price = float(d.at[exit_idx, "close"])
        reason = "time"
        for j in range(entry_idx, exit_idx + 1):
            if j < entry_idx + MIN_HOLD:      # T+2 lock
                continue
            lo, hi, cl = float(d.at[j, "low"]), float(d.at[j, "high"]), float(d.at[j, "close"])
            if stop is not None and lo <= stop:
                exit_price, reason, exit_idx = stop, "stop", j
                break
            if tp is not None and hi >= tp:
                exit_price, reason, exit_idx = tp, "tp", j
                break
            if spec["target"] is not None and cl >= float(d.at[j, spec["target"]]):
                exit_price, reason, exit_idx = cl, "target", j
                break
            tb = spec.get("trail_break")
            if tb is not None and np.isfinite(d.at[j, tb]) and cl < float(d.at[j, tb]):
                exit_price, reason, exit_idx = cl, "trail_break", j
                break
        gross = (exit_price - entry) / entry * 100.0
        net = gross - COST_PCT
        trades.append({"entry_date": d.at[entry_idx, "date"], "exit_date": d.at[exit_idx, "date"],
                       "net": net, "reason": reason})
        i = exit_idx + 1   # no overlap within a symbol
    return trades


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(trades):
    if not trades:
        return dict(n=0, win=0.0, avg=0.0, med=0.0, pf=0.0, total=0.0, dd=0.0)
    net = np.array([t["net"] for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    # equity proxy: compound PORTFOLIO_F per trade in exit-date order
    order = sorted(trades, key=lambda t: t["exit_date"])
    eq, peak, dd = 1.0, 1.0, 0.0
    for t in order:
        eq *= (1 + PORTFOLIO_F * t["net"] / 100.0)
        peak = max(peak, eq)
        dd = min(dd, (eq - peak) / peak)
    return dict(n=len(net), win=float((net > 0).mean() * 100), avg=float(net.mean()),
                med=float(np.median(net)), pf=float(pf),
                total=float((eq - 1) * 100), dd=float(dd * 100))


def benchmark(index_df, syms_df, start, end):
    def bh(d):
        w = d[(d["date"] >= start) & (d["date"] <= end)]
        if len(w) < 2:
            return None
        return (float(w["close"].iloc[-1]) / float(w["close"].iloc[0]) - 1) * 100
    vni = bh(index_df)
    rets = [bh(d) for d in syms_df.values()]
    rets = [r for r in rets if r is not None]
    vn30 = float(np.mean(rets)) if rets else None
    return vni, vn30


# --------------------------------------------------------------------------
def main():
    files = {p.stem: p for p in DATA.glob("*.csv")}
    if "VNINDEX" not in files:
        print("Missing VNINDEX.csv - run fetch first."); return
    print(f"Loaded {len(files)} files from {DATA}")

    index_raw = pd.read_csv(files["VNINDEX"])
    index_raw["date"] = pd.to_datetime(index_raw["date"]).dt.date
    idx = index_raw.sort_values("date").reset_index(drop=True)
    idx["ema50"] = ema(idx["close"], 50)
    regime_ok = {r["date"]: bool(np.isfinite(r["ema50"]) and r["close"] > r["ema50"])
                 for _, r in idx.iterrows()}

    dfs = {}
    for sym, p in files.items():
        if sym == "VNINDEX":
            continue
        try:
            dfs[sym] = prep(pd.read_csv(p))
        except Exception as e:
            print(f"skip {sym}: {e}")
    print(f"Prepared {len(dfs)} symbols\n")

    for pname, (start, end) in PERIODS.items():
        vni, vn30 = benchmark(idx, dfs, start, end)
        pct_days = sum(1 for dt, ok in regime_ok.items() if start <= dt <= end and ok)
        tot_days = sum(1 for dt in regime_ok if start <= dt <= end)
        regpct = (pct_days / tot_days * 100) if tot_days else 0
        print("=" * 100)
        print(f"PERIOD: {pname}   |  VNINDEX B&H: {vni:+.1f}%   VN30 eq-wt B&H: {vn30:+.1f}%"
              f"   |  days above EMA50: {regpct:.0f}%")
        print("-" * 100)
        print(f"{'strategy':24s} {'filter':7s} {'n':>4s} {'win%':>6s} {'avg%':>7s} "
              f"{'med%':>7s} {'PF':>5s} {'port%':>7s} {'maxDD%':>7s}")
        for sname, sfn in STRATEGIES.items():
            for use_filter in (False, True):
                rk = regime_ok if use_filter else None
                all_trades = []
                for sym, d in dfs.items():
                    em, spec = sfn(d)
                    all_trades += simulate_symbol(d, em, spec, start, end, regime_ok=rk)
                mk = metrics(all_trades)
                pf = "inf" if mk["pf"] == float("inf") else f"{mk['pf']:.2f}"
                tag = "EMA50" if use_filter else "none"
                print(f"{sname:24s} {tag:7s} {mk['n']:>4d} {mk['win']:>6.1f} {mk['avg']:>+7.2f} "
                      f"{mk['med']:>+7.2f} {pf:>5s} {mk['total']:>+7.1f} {mk['dd']:>7.1f}")
        print()


if __name__ == "__main__":
    main()

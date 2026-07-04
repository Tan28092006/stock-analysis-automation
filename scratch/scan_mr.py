"""Scan VN100 with the production mean-reversion (bat day) hybrid mode.

Usage:
  python scratch/scan_mr.py                      # scan as of latest EOD data
  python scratch/scan_mr.py --asof 2026-02-15    # scan as of a historical date
  python scratch/scan_mr.py --sweep 2026-02-01 2026-07-01   # list every BUY_SETUP day

Uses data/raw/prices_hist/*.csv (VN100, adjusted) + configs/rules_mr.json through the
REAL signal engine (score_symbol / score_precomputed_at) — what you see is what the
production scan would decide.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.features.signal_engine import prepare_signal_frame, score_precomputed_at
from stock_agent.features.indicators import ema, sma

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = json.loads(Path(r"D:\Chungkhoan\configs\rules_mr.json").read_text())


def load_frames(asof: str | None):
    frames = {}
    for p in sorted(DATA.glob("*.csv")):
        if p.stem == "VNINDEX":
            continue
        df = pd.read_csv(p)
        df["date"] = df["date"].astype(str).str.slice(0, 10)
        if asof:
            df = df[df["date"] <= asof]
        if len(df) < 90:
            continue
        frames[p.stem] = df.sort_values("date").reset_index(drop=True)
    return frames


def regime_line(asof: str | None, frames):
    idx = pd.read_csv(DATA / "VNINDEX.csv")
    idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    if asof:
        idx = idx[idx["date"] <= asof]
    idx = idx.sort_values("date").reset_index(drop=True)
    e50 = float(ema(idx["close"], 50).iloc[-1])
    close = float(idx["close"].iloc[-1])
    above = tot = 0
    for sym, df in frames.items():
        m = sma(df["close"], 20)
        if np.isfinite(m.iloc[-1]):
            tot += 1
            if float(df["close"].iloc[-1]) > float(m.iloc[-1]):
                above += 1
    breadth = above / tot * 100 if tot else 0
    state = "RISK-ON" if close > e50 and breadth >= 40 else \
            "PANIC/BEAR" if close < e50 else "GRIND (index up, breadth weak)"
    print(f"Market @ {idx['date'].iloc[-1]}: VNINDEX {close:.1f} vs EMA50 {e50:.1f} "
          f"({'above' if close > e50 else 'BELOW'}) | breadth {breadth:.0f}% > SMA20 | => {state}")
    return state


def fmt_signal(sym, sig):
    rp = sig.risk_plan
    gates = {e.name: e.passed for e in sig.evidence}
    confirm = "rev+climax" if (gates.get("Reversal bar") and gates.get("Volume climax")) else \
              ("VSA" if gates.get("VSA stopping volume") else "-")
    return (f"{sym:6s} close={sig.latest_close:>10,.0f}  RSI={next((e.value for e in sig.evidence if e.name=='Oversold RSI'), '?'):>5}  "
            f"confirm={confirm:10s} entry~{rp.entry_reference:>10,.0f} stop={rp.stop_loss:>10,.0f} "
            f"target={rp.take_profit_1:>10,.0f} RR={rp.reward_risk:.2f} hold<={rp.holding_period_days}d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--sweep", nargs=2, metavar=("START", "END"), default=None)
    args = ap.parse_args()

    if args.sweep:
        start, end = args.sweep
        frames = load_frames(None)
        print(f"SWEEP {start} -> {end} | {len(frames)} symbols | mode=mean_reversion hybrid\n")
        hits = []
        for sym, df in frames.items():
            feats = prepare_signal_frame(df, RULES)
            rsi = feats["rsi14"]; bbl = feats["bb_lower"]; cl = feats["close"]
            pre = (rsi < 35) & (cl <= bbl * 1.03)   # loose prefilter, scorer decides
            for i in np.where(pre.fillna(False))[0]:
                d = feats["date"].iloc[i]
                if not (start <= str(d) <= end) or i < 90:
                    continue
                sig = score_precomputed_at(sym, feats, int(i), RULES)
                if sig.decision == "BUY_SETUP":
                    hits.append((str(d), sym, sig))
        hits.sort()
        if not hits:
            print("No BUY_SETUP in this window.")
        for d, sym, sig in hits:
            print(f"{d}  {fmt_signal(sym, sig)}")
        print(f"\ntotal BUY_SETUP: {len(hits)}")
        return

    frames = load_frames(args.asof)
    state = regime_line(args.asof, frames)
    buys, watches = [], []
    for sym, df in frames.items():
        try:
            feats = prepare_signal_frame(df, RULES)
            sig = score_precomputed_at(sym, feats, len(feats) - 1, RULES)
        except Exception:
            continue
        if sig.decision == "BUY_SETUP":
            buys.append((sym, sig))
        elif sig.decision == "WATCH":
            watches.append((sym, sig))
    print(f"\n=== BUY_SETUP ({len(buys)}) ===")
    for sym, sig in sorted(buys, key=lambda x: -x[1].score):
        print(" ", fmt_signal(sym, sig))
    print(f"\n=== WATCH — oversold at band, waiting for confirmation ({len(watches)}) ===")
    for sym, sig in sorted(watches, key=lambda x: -x[1].score):
        rsi = next((e.value for e in sig.evidence if e.name == "Oversold RSI"), "?")
        missing = [e.name for e in sig.evidence if not e.passed]
        print(f"  {sym:6s} close={sig.latest_close:>10,.0f} RSI={rsi:>5} missing: {', '.join(missing) or '-'}")
    if not buys and not watches:
        print("\n(Khong co keo nao — dung thiet ke: bat day chi ban khi co capitulation. "
              "Che do hien tai: " + state + ")")


if __name__ == "__main__":
    main()

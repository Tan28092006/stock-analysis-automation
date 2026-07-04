"""Backtest the MR bottom-fisher YTD 2026 (Jan 1 -> now) with the PRODUCTION config
(T+15 hold, VN30-tuned gate for VN30 symbols, strict for the rest). Prints the full
trade ledger + aggregate + a sized-portfolio proxy, vs VNINDEX YTD.
Costs/slippage/price-limits are the production run_backtest's (tick-aware).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.config import load_json
from stock_agent.features.backtest import BacktestConfig, run_backtest

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = load_json(Path(r"D:\Chungkhoan\configs\rules_mr.json"))
VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--start", default="2026-01-01")
_ap.add_argument("--end", default="2026-07-01")
_a = _ap.parse_args()
START, END = date.fromisoformat(_a.start), date.fromisoformat(_a.end)


def rules_for(sym):
    if sym in VN30 and RULES.get("vn30_mean_reversion"):
        r = dict(RULES); r["mean_reversion"] = {**RULES.get("mean_reversion", {}), **RULES["vn30_mean_reversion"]}
        return r
    return RULES


def main():
    cfg = BacktestConfig.from_rules(RULES)
    rows = []
    for p in sorted(DATA.glob("*.csv")):
        if p.stem == "VNINDEX":
            continue
        df = pd.read_csv(p); df["date"] = pd.to_datetime(df["date"]).dt.date
        r = rules_for(p.stem)
        try:
            res = run_backtest(p.stem, df, r, config=cfg, start=START, end=END)
        except Exception:
            continue
        for t in res.trades:
            rows.append({"symbol": p.stem, "vn30": p.stem in VN30, "signal": t.signal_date,
                         "entry": t.entry_date, "exit": t.exit_date, "reason": t.exit_reason,
                         "entry_px": t.entry_price, "exit_px": t.exit_price, "net": t.net_return_pct})
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    w = idx[(idx["date"] >= str(START)) & (idx["date"] <= str(END))]
    bench = (w["close"].iloc[-1] / w["close"].iloc[0] - 1) * 100
    days_above = None
    if len(w):
        from stock_agent.features.indicators import ema
        e50 = ema(idx["close"], 50)
        wi = idx[(idx["date"] >= str(START)) & (idx["date"] <= str(END))].index
        days_above = float((idx.loc[wi, "close"] > e50.loc[wi]).mean() * 100)

    print(f"MR backtest {START} -> {END}  (T+15, VN30 gate)  | VNINDEX {bench:+.1f}%"
          f"{f' | {days_above:.0f}% ngay tren EMA50' if days_above is not None else ''}\n")
    if not rows:
        print(">>> 0 LENH — dung thiet ke: uptrend khong co capitulation de bat day.")
        print(f"    He ngoi tien mat toan bo giai doan; thi truong {bench:+.1f}%.")
        return
    led = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    print(f"{'ma':6s} {'VN30':5s} {'vao':11s} {'ra':11s} {'reason':13s} {'vao~':>9s} {'ra~':>9s} {'net%':>7s}")
    print("-" * 78)
    for _, t in led.iterrows():
        print(f"{t['symbol']:6s} {'VN30' if t['vn30'] else '':5s} {t['entry']:11s} {t['exit']:11s} "
              f"{t['reason']:13s} {t['entry_px']:>9,.0f} {t['exit_px']:>9,.0f} {t['net']:>+7.2f}")

    net = led["net"].values
    wins = net[net > 0]; losses = net[net <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    # equal-weight 20%/trade compounding proxy in exit order
    eq = 1.0
    for r in led.sort_values("exit")["net"]:
        eq *= (1 + 0.20 * r / 100)
    print("-" * 78)
    print(f"\nLenh: {len(net)} ({int(led['vn30'].sum())} VN30) | win {(net>0).mean()*100:.0f}% | "
          f"trung binh {net.mean():+.2f}%/lenh | PF {pf:.2f}")
    print(f"Portfolio proxy (20%/lenh compound): {(eq-1)*100:+.1f}%")
    print(f"So sanh: VNINDEX buy&hold YTD = {bench:+.1f}%")
    print(f"\nExit mix: {led['reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

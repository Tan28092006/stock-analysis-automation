"""Preview: what can foreign flows (khoi ngoai) actually add? Uses the 12-month monthly
data (covers the Mar-2026 crash + H2-2025 bull — richer than the 21-session daily set).

1. Monthly cross-sectional Rank IC: foreign net-buy (normalized by monthly traded value)
   in month M  vs  stock return in month M+1.  |IC|>0.05 & |t|>2 = usable.
2. Mar-2026 crash case study: were foreigners net buyers or sellers at the bottom, and did
   the stocks they bought most recover better?  (The bottom-fishing edge, if any.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
FDIR = Path(r"D:\Chungkhoan\data\raw\foreign")


def load_monthly(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    df["net"] = df["buy_val"].astype(float) - df["sell_val"].astype(float)   # billions VND
    # period "MM/YYYY" -> sortable YYYY-MM
    df["ym"] = df["period"].str.slice(3, 7) + "-" + df["period"].str.slice(0, 2)
    return df


def monthly_price(sym):
    p = DATA / f"{sym}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p); d["date"] = d["date"].astype(str).str.slice(0, 10)
    d["ym"] = d["date"].str.slice(0, 7)
    g = d.groupby("ym").agg(close=("close", "last"),
                            val=("close", lambda x: 0)).reset_index()
    # monthly traded value (billions) = sum(close*volume)/1e9
    tv = d.assign(v=d["close"] * d["volume"]).groupby("ym")["v"].sum() / 1e9
    g["tradeval"] = g["ym"].map(tv)
    g["ret_next"] = g["close"].shift(-1) / g["close"] - 1
    return g


def main():
    nd = load_monthly(FDIR / "ndtnn_monthly.jsonl")
    syms = sorted(nd["symbol"].unique())
    panel = []
    for sym in syms:
        mp = monthly_price(sym)
        if mp is None:
            continue
        f = nd[nd["symbol"] == sym][["ym", "net"]]
        m = mp.merge(f, on="ym", how="inner")
        m["net_norm"] = m["net"] / m["tradeval"].replace(0, np.nan)   # foreign net / traded value
        m["symbol"] = sym
        panel.append(m)
    panel = pd.concat(panel, ignore_index=True)
    print(f"Monthly panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['ym'].min()} -> {panel['ym'].max()}\n")

    # 1) cross-sectional monthly IC: net_norm[M] vs ret_next[M]
    print("1) MONTHLY IC — khoi ngoai mua rong (chuan hoa) thang M  vs  return thang M+1")
    for feat in ["net_norm", "net"]:
        ics = []
        for ym, g in panel.groupby("ym"):
            gg = g[[feat, "ret_next"]].dropna()
            if len(gg) >= 15 and gg[feat].nunique() > 5:
                ics.append(gg[feat].rank().corr(gg["ret_next"].rank()))
        ics = np.array([x for x in ics if np.isfinite(x)])
        if len(ics) >= 3:
            t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std() > 0 else 0
            print(f"   {feat:9s}: IC {ics.mean():+.3f}  t {t:+.1f}  ({len(ics)} thang, pos {int((ics>0).mean()*100)}%)")
        else:
            print(f"   {feat:9s}: khong du thang")

    # 2) Mar-2026 crash case study
    print("\n2) THANG 3/2026 (crash) — khoi ngoai mua/ban rong + hoi phuc sau do")
    mar = panel[panel["ym"] == "2026-03"].dropna(subset=["net_norm"]).copy()
    if len(mar):
        tot_net = mar["net"].sum()
        print(f"   Tong khoi ngoai thang 3: {tot_net:+.0f} ty VND ({'MUA rong' if tot_net>0 else 'BAN rong'})")
        mar = mar.sort_values("net_norm", ascending=False)
        # forward return: close end-March -> a couple months later
        fwd = {}
        for sym in mar["symbol"]:
            mp = monthly_price(sym)
            if mp is None: continue
            row = mp[mp["ym"] == "2026-03"]
            row5 = mp[mp["ym"] == "2026-05"]
            if len(row) and len(row5):
                fwd[sym] = (float(row5["close"].iloc[0]) / float(row["close"].iloc[0]) - 1) * 100
        mar["fwd_mar_to_may"] = mar["symbol"].map(fwd)
        top = mar.head(10); bot = mar.tail(10)
        print(f"\n   TOP 10 mã khoi ngoai MUA nhieu nhat (thang 3) -> hoi phuc T3->T5:")
        for _, r in top.iterrows():
            print(f"      {r['symbol']:6s} net {r['net']:>+7.0f}ty (norm {r['net_norm']:+.2f}) -> {r.get('fwd_mar_to_may', float('nan')):>+6.1f}%")
        print(f"   BOTTOM 10 mã khoi ngoai BAN nhieu nhat -> hoi phuc T3->T5:")
        for _, r in bot.iterrows():
            print(f"      {r['symbol']:6s} net {r['net']:>+7.0f}ty (norm {r['net_norm']:+.2f}) -> {r.get('fwd_mar_to_may', float('nan')):>+6.1f}%")
        t_avg = top["fwd_mar_to_may"].mean(); b_avg = bot["fwd_mar_to_may"].mean()
        print(f"\n   >>> Hoi phuc TB: nhom NGOAI MUA {t_avg:+.1f}%  vs  nhom NGOAI BAN {b_avg:+.1f}%  "
              f"(chenh {t_avg-b_avg:+.1f}pp)")


if __name__ == "__main__":
    main()

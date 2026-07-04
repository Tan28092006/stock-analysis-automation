"""IC test for foreign (khoi ngoai) + proprietary (tu doanh) flow features.

Same discipline as scratch/feature_ic.py: cross-sectional Rank IC of flow features vs
forward returns on the VN100 panel. Data from data/raw/foreign/*.jsonl (Vietstock
per-symbol charts ~20 sessions + forward snapshots) joined to prices_hist CSVs.

CAVEAT printed with results: with only ~20 sessions the t-stats are preliminary —
directional evidence only. Re-run monthly as the accumulator deepens the panel.

Features:
  f_net_norm   : foreign net buy value / 20d avg traded value (same-day, EOD-known)
  f_net_5d     : 5-session rolling sum of f_net_norm (persistent accumulation)
  td_net_norm  : tu doanh net value / 20d avg traded value
  td_net_5d    : 5-session rolling sum
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PRICES = Path(r"D:\Chungkhoan\data\raw\prices_hist")
DATA_F = Path(r"D:\Chungkhoan\data\raw\foreign")
HORIZONS = [2, 5, 10]


def load_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    nd = load_jsonl(DATA_F / "ndtnn_chart.jsonl")
    td = load_jsonl(DATA_F / "tudoanh_chart.jsonl")
    if nd.empty:
        print("no ndtnn_chart data yet"); sys.exit(1)
    nd["f_net_val"] = nd["buy_val"].astype(float) - nd["sell_val"].astype(float)
    td["td_net_val"] = (td["buy_val"].astype(float) - td["sell_val"].astype(float)) if not td.empty else 0.0

    frames = []
    for sym in sorted(nd["symbol"].unique()):
        pf = DATA_PRICES / f"{sym}.csv"
        if not pf.exists():
            continue
        px = pd.read_csv(pf)
        px["date"] = px["date"].astype(str).str.slice(0, 10)
        px = px.sort_values("date").reset_index(drop=True)
        px["avg_value_20"] = (px["close"] * px["volume"]).rolling(20, min_periods=10).mean() / 1e9  # billions
        for h in HORIZONS:
            px[f"fwd_{h}"] = px["close"].shift(-h) / px["close"] - 1
        f = nd[nd["symbol"] == sym][["date", "f_net_val"]]
        m = px.merge(f, on="date", how="left")
        if not td.empty:
            t = td[td["symbol"] == sym][["date", "td_net_val"]]
            m = m.merge(t, on="date", how="left")
        else:
            m["td_net_val"] = np.nan
        m["symbol"] = sym
        m["f_net_norm"] = m["f_net_val"] / m["avg_value_20"].replace(0, np.nan)
        m["td_net_norm"] = m["td_net_val"] / m["avg_value_20"].replace(0, np.nan)
        m["f_net_5d"] = m["f_net_norm"].rolling(5, min_periods=3).sum()
        m["td_net_5d"] = m["td_net_norm"].rolling(5, min_periods=3).sum()
        frames.append(m[["date", "symbol", "f_net_norm", "f_net_5d", "td_net_norm", "td_net_5d"]
                        + [f"fwd_{h}" for h in HORIZONS]])
    return pd.concat(frames, ignore_index=True)


def rank_ic(panel, feat, fwd):
    ics = []
    for dt, g in panel.groupby("date"):
        gg = g[[feat, fwd]].dropna()
        if len(gg) >= 15 and gg[feat].nunique() > 5:
            ics.append(gg[feat].rank().corr(gg[fwd].rank()))
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 5:
        return None
    mean = ics.mean()
    t = mean / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std() > 0 else 0.0
    return mean, t, len(ics)


def main():
    panel = build_panel()
    have = panel.dropna(subset=["f_net_norm"])
    print(f"Panel: {len(have)} flow rows, {have['symbol'].nunique()} symbols, "
          f"{have['date'].min()} -> {have['date'].max()}")
    n_days = have["date"].nunique()
    print(f"days with flow data: {n_days}")
    if n_days < 60:
        print("*** PRELIMINARY: <60 sessions — directional evidence only, t-stats weak ***")
    print()
    print(f"{'feature':14s} | " + " | ".join(f"T+{h:<2d} IC     t    n" for h in HORIZONS))
    print("-" * 76)
    for feat in ["f_net_norm", "f_net_5d", "td_net_norm", "td_net_5d"]:
        cells = []
        for h in HORIZONS:
            r = rank_ic(panel, feat, f"fwd_{h}")
            cells.append(f"{r[0]:+.3f} {r[1]:+5.1f} {r[2]:3d}" if r else "   n/a        ")
        print(f"{feat:14s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()

"""Verify the mean_reversion mode through the PRODUCTION path (signal_engine + run_backtest).

Confirms the wired-in mode reproduces the experiment's positive bottom-fishing edge,
i.e. the strategy_mode dispatch + risk_plan drive real trades with the same character.
"""
from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.features.backtest import BacktestConfig, run_backtest

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = json.loads(Path(r"D:\Chungkhoan\configs\rules_mr.json").read_text())

PERIODS = {
    "2022 crash (Apr-Nov)": (date(2022, 4, 1), date(2022, 11, 30)),
    "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
}


def agg(all_net):
    net = np.array(all_net)
    if len(net) == 0:
        return "n=0"
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    # equity proxy: 20% per trade compounded in order
    eq, peak, dd = 1.0, 1.0, 0.0
    for r in net:
        eq *= (1 + 0.20 * r / 100.0)
        peak = max(peak, eq); dd = min(dd, (eq - peak) / peak)
    pfs = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={len(net):4d}  win={float((net>0).mean()*100):5.1f}%  "
            f"avg={float(net.mean()):+.2f}%  med={float(np.median(net)):+.2f}%  "
            f"PF={pfs:>5s}  port={float((eq-1)*100):+.1f}%  maxDD={float(dd*100):.1f}%")


def main():
    cfg = BacktestConfig.from_rules(RULES)
    files = sorted(p for p in DATA.glob("*.csv") if p.stem != "VNINDEX")
    print(f"Production mean_reversion mode | {len(files)} symbols | holding_days={cfg.holding_days}\n")
    for pname, (s, e) in PERIODS.items():
        all_net, reasons = [], {}
        for p in files:
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            try:
                res = run_backtest(p.stem, df, RULES, config=cfg, start=s, end=e)
            except Exception as ex:
                print(f"  skip {p.stem}: {ex}"); continue
            for t in res.trades:
                all_net.append(t.net_return_pct)
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"{pname:24s}  {agg(all_net)}")
        print(f"{'':24s}  exits: {reasons}\n")


if __name__ == "__main__":
    main()

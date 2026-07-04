"""Verify the market-regime filter through the PRODUCTION momentum path.

Runs the real signal_engine + run_backtest with market_filter OFF vs ON and shows the
filter improves the momentum branch in downturns (the "chong du dinh" claim), using the
same code that the live scan uses.
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
from stock_agent.features.market_regime import market_regime_map, attach_market_regime

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = json.loads(Path(r"D:\Chungkhoan\configs\rules_t2.json").read_text())
RULES["ml"] = {"enabled": False, "override_enabled": False}  # isolate the filter effect

PERIODS = {
    "2022 crash (Apr-Nov)": (date(2022, 4, 1), date(2022, 11, 30)),
    "2026 Feb-Jun": (date(2026, 2, 1), date(2026, 6, 30)),
    "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
}


def agg(net):
    net = np.array(net)
    if len(net) == 0:
        return "n=   0"
    wins, losses = net[net > 0], net[net <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    eq, peak, dd = 1.0, 1.0, 0.0
    for r in net:
        eq *= (1 + 0.20 * r / 100.0); peak = max(peak, eq); dd = min(dd, (eq - peak) / peak)
    pfs = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={len(net):4d}  win={float((net>0).mean()*100):5.1f}%  avg={float(net.mean()):+.2f}%  "
            f"PF={pfs:>5s}  port={float((eq-1)*100):+7.1f}%  maxDD={float(dd*100):6.1f}%")


def main():
    files = {p.stem: p for p in DATA.glob("*.csv")}
    idx = pd.read_csv(files["VNINDEX"]); idx["date"] = pd.to_datetime(idx["date"]).dt.date
    frames = {}
    for sym, p in files.items():
        if sym == "VNINDEX":
            continue
        df = pd.read_csv(p); df["date"] = pd.to_datetime(df["date"]).dt.date
        frames[sym] = df
    mf = RULES.get("market_filter", {})
    regime = market_regime_map(idx, frames, ema_period=mf.get("ema_period", 50),
                               breadth_ma=mf.get("breadth_ma", 20), breadth_min=mf.get("breadth_min", 0.4))
    cfg = BacktestConfig.from_rules(RULES)
    print(f"Momentum path | {len(frames)} symbols | holding_days={cfg.holding_days} | "
          f"risk-on days={sum(1 for v in regime.values() if v>0)}/{len(regime)}\n")

    for pname, (s, e) in PERIODS.items():
        print(f"PERIOD: {pname}")
        for label, enabled in (("filter OFF", False), ("filter ON ", True)):
            r = dict(RULES)
            r["market_filter"] = {**mf, "enabled": enabled}
            allnet = []
            for sym, df in frames.items():
                d = attach_market_regime(df, regime) if enabled else df
                try:
                    res = run_backtest(sym, d, r, config=cfg, start=s, end=e)
                    allnet += [t.net_return_pct for t in res.trades]
                except Exception:
                    pass
            print(f"   {label}  {agg(allnet)}")
        print()


if __name__ == "__main__":
    main()

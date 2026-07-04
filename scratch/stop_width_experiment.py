"""Stop-width sweep for the MR hybrid: does a wider stop keep 2022 crash protection
while cutting whipsaw (the PC1 case) in choppy/recovering regimes?

Variants: stop 1.5 / 2.0 / 2.5 / 3.0 x ATR / no stop (target+time only), same entry mask
(MR hybrid) and Kijun target, hold<=8, T+2 lock, cost 0.60%.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep, simulate_symbol, metrics, strat_mr_hybrid

PERIODS = {
    "2022 crash (Apr-Nov)": (date(2022, 4, 1), date(2022, 11, 30)),
    "2026 Feb-Jun (grind)": (date(2026, 2, 1), date(2026, 6, 30)),
    "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
}
STOPS = [1.5, 2.0, 2.5, 3.0, None]


def main():
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    dfs = {sym: prep(pd.read_csv(p)) for sym, p in files.items()}
    print(f"{len(dfs)} symbols | mask=MR hybrid | target=kijun | hold<=8 | cost 0.60%\n")
    for pname, (s, e) in PERIODS.items():
        print(f"PERIOD: {pname}")
        print(f"  {'stop':>8s} {'n':>4s} {'win%':>6s} {'avg%':>7s} {'PF':>5s} {'port%':>7s} {'maxDD%':>7s} {'stops':>6s}")
        for stop in STOPS:
            all_trades = []
            for sym, d in dfs.items():
                mask, spec = strat_mr_hybrid(d)
                spec = dict(spec)
                spec["stop_atr"] = stop
                all_trades += simulate_symbol(d, mask, spec, s, e)
            mk = metrics(all_trades)
            n_stop = sum(1 for t in all_trades if t["reason"] == "stop")
            pf = "inf" if mk["pf"] == float("inf") else f"{mk['pf']:.2f}"
            label = f"{stop}xATR" if stop else "NO STOP"
            print(f"  {label:>8s} {mk['n']:>4d} {mk['win']:>6.1f} {mk['avg']:>+7.2f} {pf:>5s} "
                  f"{mk['total']:>+7.1f} {mk['dd']:>7.1f} {n_stop:>6d}")
        print()


if __name__ == "__main__":
    main()

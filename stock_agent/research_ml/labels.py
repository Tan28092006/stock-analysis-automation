"""R3: strategy labels with explicit entry/exit/availability and pending states.

MR15 and momentum labels are separate targets. Entry index ``e`` = first bar after
the signal bar; exit close index ``e + horizon`` (matches the gate: not ``e+horizon-1``).
Pending or unfillable exits are explicit states, never zero or loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.backtest import BacktestConfig, round_trip_cost_pct
from .contracts import VN_TZ

PROTOCOLS = {"mr_fixed15": 15, "mom_5": 5, "mom_21": 21, "mom_63": 63}


@dataclass(frozen=True)
class CostConfig:
    commission_pct: float = 0.15
    sell_tax_pct: float = 0.10
    slippage_pct: float = 0.10

    @property
    def round_trip_pct(self) -> float:
        return self.commission_pct * 2 + self.sell_tax_pct + self.slippage_pct * 2

    @classmethod
    def from_backtest(cls, cfg: BacktestConfig) -> "CostConfig":
        out = cls(cfg.commission_pct, cfg.sell_tax_pct, cfg.slippage_pct)
        assert abs(out.round_trip_pct - round_trip_cost_pct(cfg)) < 1e-12
        return out


def build_labels(bars: pd.DataFrame, signals: pd.DataFrame, *, protocol: str, cost: CostConfig,
                 data_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """``signals``: symbol, signal_date [, selected]. All candidates keep a row."""
    horizon = PROTOCOLS[protocol]
    rows = []
    by_symbol = {s: g.sort_values("session_date").reset_index(drop=True)
                 for s, g in bars.groupby("symbol")}
    for sig in signals.itertuples(index=False):
        g = by_symbol.get(sig.symbol)
        rec = {"symbol": sig.symbol, "signal_date": pd.Timestamp(sig.signal_date).normalize(),
               "protocol": protocol, "horizon": horizon,
               "selected": bool(getattr(sig, "selected", True)), "status": "pending",
               "entry_date": pd.NaT, "exit_date": pd.NaT, "net_return_pct": float("nan"),
               "label_available_at": pd.NaT}
        rows.append(rec)
        if g is None:
            rec["status"] = "no_price_history"
            continue
        hit = g.index[g.session_date == rec["signal_date"]]
        if len(hit) != 1:
            rec["status"] = "signal_bar_missing"
            continue
        e = hit[0] + 1
        if e >= len(g):
            continue  # entry bar not observed yet: pending
        rec["entry_date"] = g.at[e, "session_date"]
        if g.at[e, "volume"] <= 0:
            rec["status"] = "entry_unfillable"
            continue
        x = e + horizon
        if x >= len(g):
            continue  # exit not observed: pending, not a loss
        rec["exit_date"] = g.at[x, "session_date"]
        if g.at[x, "volume"] <= 0:
            rec["status"] = "exit_unfillable"  # suspended/untradable: no fabricated exit
            continue
        gross = (g.at[x, "close"] / g.at[e, "open"] - 1) * 100
        rec.update(status="matured", net_return_pct=gross - cost.round_trip_pct,
                   label_available_at=rec["exit_date"].tz_localize(VN_TZ) + pd.Timedelta(hours=16))
    out = pd.DataFrame(rows)
    out["label_available_at"] = pd.to_datetime(out["label_available_at"], utc=True)
    if data_end is not None:
        out.loc[out.label_available_at > pd.Timestamp(data_end), "status"] = "pending"
    out["win"] = (out.net_return_pct > 0).where(out.status == "matured")
    return out


def trainable(labels: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Only matured labels whose outcome was knowable at ``as_of``."""
    as_of = pd.Timestamp(as_of)
    keep = (labels.status == "matured") & (labels.label_available_at <= as_of)
    return labels[keep].reset_index(drop=True)

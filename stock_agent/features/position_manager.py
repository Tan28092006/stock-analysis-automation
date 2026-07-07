"""Position sizing (#1) + open-position lifecycle & SELL alerts (#3) for the MR engine.

Turns a signal into an actionable order (how many shares at what risk) and, once taken,
watches the position each EOD against its disaster stop / Kijun target / time-stop and
raises a SELL alert. This is the SATELLITE tool: a low-drawdown bottom-fisher you run
alongside a core holding.

Sizing (validated in scratch/portfolio_sim.py): qty = risk_amount / (entry - stop),
capped by max weight, exposure, and cash; disaster stop is the sizing denominator.
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

PRICES_DIR = Path("data/raw/prices_hist")

# Serialize all store mutations — the store is instantiated per-request in a threaded
# server, so concurrent add()/close() on different instances would otherwise interleave
# their load→modify→save and silently drop writes.
_STORE_LOCK = threading.Lock()
POSITIONS_PATH = Path("data/pipeline/mr_positions.json")

DEFAULT_MONEY = {
    "account_nav": 1_000_000_000.0,
    "risk_per_trade_pct": 1.5,
    "max_positions": 4,
    "max_weight_pct": 25.0,
    "max_exposure_pct": 60.0,
    "lot_size": 100,
    "buy_cost_pct": 0.15,
    "sell_cost_pct": 0.25,
}


def money_cfg(rules: dict) -> dict:
    cfg = dict(DEFAULT_MONEY)
    cfg.update(rules.get("money", {}) if isinstance(rules, dict) else {})
    return cfg


def suggest_size(nav: float, entry: float, stop: float, cfg: dict,
                 current_exposure: float = 0.0) -> dict:
    """Position size for one trade. Returns qty (lot-rounded) + context."""
    lot = int(cfg["lot_size"])
    per_share_risk = entry - stop
    if per_share_risk <= 0 or entry <= 0 or nav <= 0:
        return {"qty": 0, "reason": "invalid entry/stop"}
    risk_amount = nav * cfg["risk_per_trade_pct"] / 100.0
    qty = math.floor(risk_amount / per_share_risk / lot) * lot
    # caps: per-position weight, remaining exposure
    qty = min(qty, math.floor(nav * cfg["max_weight_pct"] / 100.0 / entry / lot) * lot)
    room = nav * cfg["max_exposure_pct"] / 100.0 - current_exposure
    qty = min(qty, math.floor(max(0, room) / entry / lot) * lot)
    qty = max(0, int(qty))
    value = qty * entry
    return {
        "qty": qty,
        "position_value": round(value, 0),
        "weight_pct": round(value / nav * 100, 2) if nav else 0,
        "risk_amount": round(risk_amount, 0),
        "risk_pct_of_nav": cfg["risk_per_trade_pct"],
        "est_loss_at_stop": round(qty * per_share_risk, 0),
        "buy_cost": round(value * cfg["buy_cost_pct"] / 100.0, 0),
    }


# ---------------------------------------------------------------- store
class PositionStore:
    def __init__(self, path: Path = POSITIONS_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, rows: list[dict]) -> None:
        # Atomic: write to a temp file then rename, so a crash mid-write can't truncate
        # the positions file (it holds money-at-risk state).
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self, status: str | None = None) -> list[dict]:
        rows = self._load()
        return [r for r in rows if status is None or r.get("status") == status]

    def add(self, symbol: str, entry_date: str, entry_price: float, stop: float,
            target: float, max_hold_days: int, qty: int, kind: str = "mr") -> dict:
        # Unique id (uuid, not list-length) so two open positions can never collide —
        # a length-derived id repeats after any close() and would let close() shut both.
        pos = {
            "id": f"{symbol.upper()}-{entry_date}-{uuid.uuid4().hex[:8]}",
            "symbol": symbol.upper(), "entry_date": entry_date, "kind": kind,
            "entry_price": float(entry_price), "stop": float(stop), "target": float(target),
            "max_hold_days": int(max_hold_days), "qty": int(qty), "status": "OPEN",
        }
        with _STORE_LOCK:
            rows = self._load()
            rows.append(pos)
            self._save(rows)
        return pos

    def close(self, pid: str, exit_price: float | None = None,
              exit_date: str | None = None, reason: str = "manual") -> bool:
        with _STORE_LOCK:
            rows = self._load()
            found = False
            for r in rows:
                if r["id"] == pid and r.get("status") == "OPEN":
                    r["status"] = "CLOSED"
                    r["exit_price"] = float(exit_price) if exit_price is not None else None
                    r["exit_date"] = exit_date or str(date.today())
                    r["exit_reason"] = reason
                    found = True
                    break  # close exactly one, even if a legacy dup id exists
            if found:
                self._save(rows)
        return found


# ------------------------------------------------- daily lifecycle check
def _symbol_frame(symbol: str) -> pd.DataFrame | None:
    p = PRICES_DIR / f"{symbol.upper()}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    return df.sort_values("date").reset_index(drop=True)


def check_positions(store: PositionStore) -> list[dict]:
    """For each OPEN MR position, evaluate against latest EOD data and attach a live status:
    HOLD / SELL(reason) + unrealized P&L. Returns enriched position dicts."""
    out = []
    for pos in store.list(status="OPEN"):
        if pos.get("kind", "mr") != "mr":
            continue
        df = _symbol_frame(pos["symbol"])
        if df is None or df.empty:
            pos = {**pos, "live_status": "NO_DATA"}
            out.append(pos)
            continue
        after = df[df["date"] >= pos["entry_date"]].reset_index(drop=True)
        if after.empty:
            out.append({**pos, "live_status": "PENDING"})
            continue
        entry = pos["entry_price"]
        # replay bar-by-bar from entry (T+2 lock), first of stop/target/time wins —
        # exactly the production exit logic, so the alert = what actually would happen.
        status, reason, exit_px, exit_date = "HOLD", None, None, None
        for k in range(len(after)):
            if k < 2:  # T+2 settlement lock
                continue
            lo, hi = float(after.at[k, "low"]), float(after.at[k, "high"])
            if lo <= pos["stop"]:
                status, reason, exit_px, exit_date = "SELL", "STOP_LOSS", pos["stop"], str(after.at[k, "date"]); break
            if hi >= pos["target"]:
                status, reason, exit_px, exit_date = "SELL", "TARGET_HIT", pos["target"], str(after.at[k, "date"]); break
            if k >= pos["max_hold_days"]:
                status, reason, exit_px, exit_date = "SELL", "TIME_STOP", float(after.at[k, "close"]), str(after.at[k, "date"]); break
        last = after.iloc[-1]
        cur = float(last["close"])
        held = max(0, len(after) - 1)
        ref_px = exit_px if exit_px is not None else cur
        upnl_pct = (ref_px - entry) / entry * 100 if entry else 0.0
        out.append({
            **pos, "current_price": round(cur, 2), "held_days": held,
            "unrealized_pct": round(upnl_pct, 2), "live_status": status,
            "sell_reason": reason, "exit_price": round(exit_px, 2) if exit_px else None,
            "exit_date": exit_date, "as_of": str(last["date"]),
        })
    return out


def check_momentum_positions(store: PositionStore, buffer_symbols: set, risk_on: bool = True) -> list[dict]:
    """Daily lifecycle for CORE quant-momentum holds. Exit rule = the strategy's own:
    a name is held until it drops out of the top-2N momentum BUFFER (monthly rebalance),
    so SELL when it leaves the buffer. (No EMA50/regime kill — validated as harmful.)"""
    out = []
    buf = {s.upper() for s in buffer_symbols}
    for pos in store.list(status="OPEN"):
        if pos.get("kind") != "momentum":
            continue
        df = _symbol_frame(pos["symbol"])
        if df is None or df.empty:
            out.append({**pos, "live_status": "NO_DATA"}); continue
        after = df[df["date"] >= pos["entry_date"]].reset_index(drop=True)
        cur = float(df["close"].iloc[-1])
        held = max(0, len(after) - 1)
        entry = pos["entry_price"]
        upnl = (cur - entry) / entry * 100 if entry else 0.0
        in_buf = pos["symbol"].upper() in buf
        if not in_buf:
            status, reason = "SELL", "Rớt khỏi nhóm momentum (top-2N)"
        else:
            status, reason = "HOLD", None
        out.append({**pos, "current_price": round(cur, 2), "held_days": held,
                    "unrealized_pct": round(upnl, 2), "in_top": in_buf,
                    "live_status": status, "sell_reason": reason, "as_of": str(df["date"].iloc[-1])})
    return out


def alerts(store: PositionStore) -> list[dict]:
    """Positions flagged SELL today (for the EOD alert log / dashboard banner)."""
    return [p for p in check_positions(store) if p.get("live_status") == "SELL"]

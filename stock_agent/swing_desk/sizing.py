"""Position plan: user 3% stop, validated -8% stop, effective-loss sizing, VND at risk.

Sizing denominator is the *effective* loss of the 3% stop: stop + gap allowance + round-trip
cost (default 3 + 1 + 0.6 = 4.6%), because the stop cannot be executed before T+2 afternoon and
stops gap through. Reuses `position_manager.suggest_size` for lot/weight/exposure caps.
"""
from __future__ import annotations

import math

from ..features.position_manager import suggest_size


def effective_loss_pct(cfg: dict, stop_pct: float | None = None) -> float:
    s = cfg["stops"]
    stop = s["user_stop_pct"] if stop_pct is None else stop_pct
    return stop + s["gap_allowance_pct"] + s["round_trip_cost_pct"]


def money(cfg: dict) -> dict:
    m = cfg["money"]
    return {"risk_per_trade_pct": m["risk_per_trade_pct"], "max_weight_pct": m["max_weight_pct"],
            "max_exposure_pct": m["max_exposure_pct"], "lot_size": m["lot_size"],
            "buy_cost_pct": m["buy_cost_pct"], "sell_cost_pct": m["sell_cost_pct"]}


def plan(entry: float, nav: float, cfg: dict, *, current_exposure: float = 0.0,
         amount: float | None = None) -> dict:
    if entry <= 0 or nav <= 0:
        raise ValueError("entry and NAV must be positive")
    s = cfg["stops"]
    lot = int(cfg["money"]["lot_size"])
    eff = effective_loss_pct(cfg)
    eff_validated = effective_loss_pct(cfg, s["validated_stop_pct"])
    m = money(cfg)
    size = suggest_size(nav, entry, entry * (1 - eff / 100), m, current_exposure=current_exposure)
    validated = suggest_size(nav, entry, entry * (1 - eff_validated / 100), m, current_exposure=current_exposure)
    qty = int(size["qty"])
    risk_only = math.floor(nav * m["risk_per_trade_pct"] / 100 / (entry * eff / 100) / lot) * lot
    capped_by = None
    if qty < risk_only:
        weight_cap = math.floor(nav * m["max_weight_pct"] / 100 / entry / lot) * lot
        capped_by = "max_weight" if qty == weight_cap else "max_exposure"
    value = qty * entry
    out = {
        "entry_ref": entry,
        "nav": nav,
        "stop_user": entry * (1 - s["user_stop_pct"] / 100),
        "stop_validated": entry * (1 - s["validated_stop_pct"] / 100),
        "effective_loss_pct": eff,
        "risk_budget_vnd": nav * m["risk_per_trade_pct"] / 100,
        "qty": qty,
        "position_value": value,
        "weight_pct": value / nav * 100,
        "capped_by": capped_by,
        "vnd_at_risk_nominal": value * s["user_stop_pct"] / 100,
        "vnd_at_risk_effective": value * eff / 100,
        "qty_if_validated_stop": int(validated["qty"]),
        "effective_loss_pct_validated": eff_validated,
    }
    if amount is not None:
        rq = int(math.floor(amount / entry / lot) * lot)
        rv = rq * entry
        out["requested"] = {
            "amount": amount, "qty": rq, "value": rv, "weight_pct": rv / nav * 100,
            "vnd_at_risk_nominal": rv * s["user_stop_pct"] / 100,
            "vnd_at_risk_effective": rv * eff / 100,
            "risk_nominal_pct_nav": rv * s["user_stop_pct"] / nav,
            "risk_effective_pct_nav": rv * eff / nav,
            "over_budget": rv * eff / 100 > out["risk_budget_vnd"] + 1e-6,
        }
    return out

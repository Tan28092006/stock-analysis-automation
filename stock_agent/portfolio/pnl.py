from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..constants import LATEST_SCAN_PATH, PORTFOLIO_PATH, TRAINING_EVENTS_PATH
from ..data.repository import append_jsonl, read_json, write_json


@dataclass
class Position:
    symbol: str
    buy_price: float
    quantity: float
    buy_date: str | None = None


class PortfolioStore:
    def __init__(self, path=PORTFOLIO_PATH):
        self.path = path

    def list_positions(self) -> list[dict[str, Any]]:
        payload = read_json(self.path, default={"positions": []})
        return payload.get("positions", [])

    def add_position(self, symbol: str, buy_price: float, quantity: float, buy_date: str | None = None) -> dict:
        position = {
            "symbol": symbol.upper(),
            "buy_price": float(buy_price),
            "quantity": float(quantity),
            "buy_date": buy_date,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = read_json(self.path, default={"positions": []})
        positions = payload.get("positions", [])
        positions.append(position)
        write_json(self.path, {"positions": positions})
        return position

    def clear(self) -> None:
        write_json(self.path, {"positions": []})


def latest_prices_from_scan(path=LATEST_SCAN_PATH) -> dict[str, float]:
    scan = read_json(path, default={})
    prices: dict[str, float] = {}
    for item in scan.get("candidates", []):
        prices[item["symbol"]] = float(item["latest_close"])
    return prices


def calculate_pnl(positions: list[dict[str, Any]], latest_prices: dict[str, float]) -> dict[str, Any]:
    rows = []
    total_cost = 0.0
    priced_cost = 0.0
    unpriced_cost = 0.0
    total_value = 0.0
    missing_prices: list[str] = []
    for pos in positions:
        symbol = pos["symbol"].upper()
        buy_price = float(pos["buy_price"])
        quantity = float(pos["quantity"])
        cost = buy_price * quantity
        current = latest_prices.get(symbol)
        if current is None:
            missing_prices.append(symbol)
            unpriced_cost += cost
            market_value = None
            pnl = None
            pnl_pct = None
        else:
            market_value = current * quantity
            pnl = market_value - cost
            pnl_pct = pnl / cost * 100 if cost else 0.0
            total_value += market_value
            priced_cost += cost
        total_cost += cost
        rows.append(
            {
                "symbol": symbol,
                "buy_price": buy_price,
                "quantity": quantity,
                "current_price": current,
                "cost": round(cost, 2),
                "market_value": round(market_value, 2) if market_value is not None else None,
                "unrealized_pnl": round(pnl, 2) if pnl is not None else None,
                "unrealized_pnl_pct": round(pnl_pct, 3) if pnl_pct is not None else None,
                "buy_date": pos.get("buy_date"),
            }
        )
    total_pnl = total_value - priced_cost
    unpriced_positions = len(set(missing_prices))
    if positions and unpriced_positions == len({pos["symbol"].upper() for pos in positions}):
        pnl_status = "no_prices"
    elif missing_prices:
        pnl_status = "partial"
    else:
        pnl_status = "complete"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "total_cost": round(total_cost, 2),
        "priced_cost": round(priced_cost, 2),
        "unpriced_cost": round(unpriced_cost, 2),
        "unpriced_positions": unpriced_positions,
        "total_market_value": round(total_value, 2),
        "total_unrealized_pnl": round(total_pnl, 2),
        "total_unrealized_pnl_pct": round(total_pnl / priced_cost * 100, 3) if priced_cost else 0.0,
        "pnl_status": pnl_status,
        "missing_prices": sorted(set(missing_prices)),
    }


def log_portfolio_features(pnl_payload: dict[str, Any]) -> None:
    append_jsonl(
        TRAINING_EVENTS_PATH,
        {
            "event_type": "portfolio_pnl_snapshot",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": pnl_payload,
            "label_status": "position_snapshot_not_training_label",
        },
    )

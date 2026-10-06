"""`swing_desk/portfolio.txt` parser: validation with line numbers, FIFO lots, P&L.

Format (UTF-8, `|`-separated, `#` comments, blank lines ignored):
    date|action|ticker|amount_vnd|price|qty|reason|gate|emotion|thesis_id|note
The first five fields are required (may be empty where the action allows it). The parser
is read-only: it never rewrites, normalizes or appends to the journal file.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
import math
from pathlib import Path
import re
from typing import Callable

from ..data.exchange_calendar import is_trading_day
from .calendar_events import session_distance

ACTIONS = {"BUY", "SELL", "THESIS", "CASH", "NAV"}
GATES = {"PASS", "WARN", "BLOCK", "NONE"}
EMOTIONS = {"calm", "confident", "fomo", "fear", "revenge", "hope", "hype", "bored", "other"}
REASONS = {"s1", "s2", "news", "forum", "tip", "kol", "group", "other"}
SELL_REASONS = REASONS | {"target", "stop", "time", "sma5", "invalid", "rebalance"}
FIELDS = ["date", "action", "ticker", "amount_vnd", "price", "qty", "reason", "gate", "emotion", "thesis_id", "note"]
TICKER = re.compile(r"[A-Z0-9]{2,12}")
BUY_COST, SELL_COST = 0.0015, 0.0025
STRESS_ROUND_TRIP = 0.006


@dataclass
class Event:
    line: int
    date: str
    action: str
    ticker: str = ""
    amount: float | None = None
    price: float | None = None
    qty: int | None = None
    reason: str = ""
    gate: str = ""
    emotion: str = ""
    thesis_id: str = ""
    note: str = ""
    price_estimated: bool = False
    gate_source: str = "journal"

    @property
    def scalein(self) -> bool:
        return "plan=scalein" in self.note.replace(" ", "").lower()


@dataclass
class Journal:
    events: list[Event] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    positions: dict = field(default_factory=dict)
    closed: list[dict] = field(default_factory=list)
    buy_cost: float = BUY_COST
    sell_cost: float = SELL_COST

    def nav(self) -> float | None:
        """Last NAV statement + later CASH + later realized P&L; else CASH + P&L; else None."""
        navs = [e for e in self.events if e.action == "NAV"]
        cash = [e for e in self.events if e.action == "CASH"]
        if not navs and not cash:
            return None
        after = navs[-1].date if navs else ""
        base = navs[-1].amount if navs else 0.0
        later = lambda d: d > after if navs else True  # noqa: E731
        base += sum(e.amount for e in cash if later(e.date) or (navs and e.date == after and e.line > navs[-1].line))
        base += sum(t["pnl_vnd"] for t in self.closed if later(t["exit_date"]))
        return base

    def buys(self) -> list[Event]:
        return [e for e in self.events if e.action == "BUY"]

    def theses(self, ticker: str) -> list[Event]:
        return [e for e in self.events if e.action == "THESIS" and e.ticker == ticker]


def _number(text: str, name: str) -> float | None:
    text = text.strip().replace("_", "").replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name}: not a number: {text!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name}: not a finite number")
    return value


def _norm_reason(reason: str) -> str:
    code = reason.upper()
    return code if code in {"S1", "S2"} or re.fullmatch(r"A\d{1,2}", code) else reason.lower()


def _parse_line(no: int, raw: str, price_lookup) -> Event:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) < 5:
        raise ValueError(f"expected at least 5 '|'-separated fields, got {len(parts)}")
    parts += [""] * (len(FIELDS) - len(parts))
    if len(parts) > len(FIELDS):  # extra pipes belong to the note
        parts = parts[:len(FIELDS) - 1] + ["|".join(parts[len(FIELDS) - 1:])]
    d, action, ticker, amount, price, qty, reason, gate, emotion, thesis_id, note = parts
    try:
        day = date.fromisoformat(d)
    except ValueError:
        raise ValueError(f"invalid date {d!r} (use YYYY-MM-DD)") from None
    action = action.upper()
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(ACTIONS)}")
    ticker = ticker.upper()
    if action in {"BUY", "SELL", "THESIS"} and not TICKER.fullmatch(ticker):
        raise ValueError(f"invalid ticker {ticker!r}")
    if gate and gate.upper() not in GATES:
        raise ValueError(f"invalid gate {gate!r}; expected one of {sorted(GATES)}")
    if emotion and emotion.lower() not in EMOTIONS:
        raise ValueError(f"invalid emotion {emotion!r}; expected one of {sorted(EMOTIONS)}")
    allowed = SELL_REASONS if action == "SELL" else REASONS
    if reason and reason.lower() not in allowed and not re.fullmatch(r"[Aa]\d{1,2}", reason):
        raise ValueError(f"invalid reason {reason!r}; expected S1, S2, A-code or one of {sorted(allowed - {'s1', 's2'})}")
    ev = Event(line=no, date=str(day), action=action, ticker=ticker, amount=_number(amount, "amount_vnd"),
               price=_number(price, "price"), reason=_norm_reason(reason),
               gate=gate.upper(), emotion=emotion.lower(), thesis_id=thesis_id, note=note)
    q = _number(qty, "qty")
    if q is not None:
        if q <= 0 or q != int(q):
            raise ValueError("qty: must be a positive whole number")
        ev.qty = int(q)
    if action in {"BUY", "SELL"}:
        if not is_trading_day(day):
            raise ValueError(f"date {day} is not a HOSE trading session")
        if ev.price is not None and ev.price <= 0:
            raise ValueError("price: must be positive")
    if action == "BUY":
        if ev.amount is None and (ev.qty is None or ev.price is None):
            raise ValueError("BUY needs amount_vnd, or qty and price")
        if ev.amount is not None and ev.amount <= 0:
            raise ValueError("amount_vnd: must be positive for BUY")
    if action in {"BUY", "SELL"} and ev.price is None:
        close = price_lookup(ticker, day) if price_lookup else None
        if close is None:
            raise ValueError(f"price missing and no snapshot close available for {ticker} {day}")
        ev.price, ev.price_estimated = float(close), True
    if action == "BUY" and ev.qty is None:
        ev.qty = int(math.floor(ev.amount / ev.price))  # odd lots allowed (lô lẻ)
        if ev.qty <= 0:
            raise ValueError("amount_vnd buys less than one share")
    if action in {"CASH", "NAV"} and ev.amount is None:
        raise ValueError(f"{action} needs amount_vnd")
    if action == "NAV" and ev.amount <= 0:
        raise ValueError("NAV must be positive")
    return ev


def _fill_gate(ev: Event, checks) -> None:
    if ev.action != "BUY" or ev.gate:
        return
    match = [c for c in (checks or []) if c.get("ticker") == ev.ticker and c.get("trade_session") == ev.date]
    if match:
        ev.gate, ev.gate_source = match[-1]["verdict"], "checks_log"
    else:
        ev.gate, ev.gate_source = "NONE", "default"


def parse_journal(text: str, *, price_lookup: Callable | None = None, checks: list[dict] | None = None,
                  buy_cost: float = BUY_COST, sell_cost: float = SELL_COST) -> Journal:
    journal = Journal(buy_cost=buy_cost, sell_cost=sell_cost)
    parsed = []
    for no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            ev = _parse_line(no, stripped, price_lookup)
        except ValueError as exc:
            journal.errors.append({"line": no, "error": str(exc)})
            continue
        _fill_gate(ev, checks)
        parsed.append(ev)
    parsed.sort(key=lambda e: (e.date, e.line))
    lots: dict[str, deque] = {}
    meta: dict[str, dict] = {}
    for ev in parsed:
        if ev.action == "BUY":
            book = lots.setdefault(ev.ticker, deque())
            if not book:
                prior_plan = any(t.scalein for t in parsed if t.action == "THESIS" and t.ticker == ev.ticker
                                 and t.date <= ev.date and t.line != ev.line)
                meta[ev.ticker] = {"first_date": ev.date, "scalein_plan": ev.scalein or prior_plan,
                                   "gate": ev.gate, "emotion": ev.emotion, "reason": ev.reason,
                                   "thesis_id": ev.thesis_id}
            book.append({"date": ev.date, "qty": ev.qty, "price": ev.price, "event": ev})
        elif ev.action == "SELL":
            book = lots.get(ev.ticker) or deque()
            held = sum(l["qty"] for l in book)
            qty = ev.qty if ev.qty is not None else held
            if held == 0 or qty > held:
                journal.errors.append({"line": ev.line, "error": f"SELL {qty} {ev.ticker} exceeds held {held}"})
                continue
            ev.qty = qty
            remaining, matched = qty, []
            while remaining:
                lot = book[0]
                take = min(remaining, lot["qty"])
                matched.append((lot, take))
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] == 0:
                    book.popleft()
            journal.closed.append(_closed_trade(ev, matched, meta.get(ev.ticker, {}), buy_cost, sell_cost))
        journal.events.append(ev)
    for ticker, book in lots.items():
        qty = sum(l["qty"] for l in book)
        if qty:
            cost = sum(l["qty"] * l["price"] for l in book)
            journal.positions[ticker] = {
                "ticker": ticker, "qty": qty, "avg_cost": cost / qty, "cost_value": cost,
                "first_date": book[0]["date"], "last_date": book[-1]["date"],
                "lots": [{"date": l["date"], "qty": l["qty"], "price": l["price"]} for l in book],
                **{k: v for k, v in meta[ticker].items() if k != "first_date"}}
    return journal


def _closed_trade(ev: Event, matched, meta: dict, buy_cost: float, sell_cost: float) -> dict:
    qty = sum(take for _, take in matched)
    basis = sum(lot["price"] * take for lot, take in matched)
    entry_avg = basis / qty
    proceeds = ev.price * qty
    pnl = proceeds * (1 - sell_cost) - basis * (1 + buy_cost)
    first = matched[0][0]["event"]
    entry_date = min(lot["date"] for lot, _ in matched)
    gross = (ev.price / entry_avg - 1) * 100
    return {
        "ticker": ev.ticker, "qty": qty, "entry_date": entry_date, "exit_date": ev.date,
        "entry_avg": entry_avg, "exit_price": ev.price, "exit_price_estimated": ev.price_estimated,
        "gross_pct": gross,
        "net_pct": ((ev.price * (1 - sell_cost)) / (entry_avg * (1 + buy_cost)) - 1) * 100,
        "net_pct_stress": gross - STRESS_ROUND_TRIP * 100,
        "pnl_vnd": pnl, "cost_basis_vnd": basis,
        "holding_sessions": session_distance(date.fromisoformat(entry_date), date.fromisoformat(ev.date)),
        "gate": first.gate or meta.get("gate", "NONE"), "emotion": first.emotion or "unknown",
        "reason": first.reason or "unknown", "exit_reason": ev.reason or "unknown",
        "line": ev.line,
    }


def load_journal(path: Path, **kwargs) -> Journal | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig") as stream:
        text = stream.read()
    return parse_journal(text, **kwargs)

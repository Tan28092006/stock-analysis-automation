"""Session arithmetic for the desk: trade session, T+2.5 sellability, VN30F expiry, horizon."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

from ..data.exchange_calendar import (
    HOSE_HOLIDAYS, VN_TIMEZONE, add_trading_days, is_trading_day, last_trading_day,
    next_trading_day, trading_days_between,
)

MARKET_CLOSE = time(14, 45)
SELL_FROM = "13:00"  # afternoon continuous session of T+2


def trade_session(now: datetime) -> date:
    """Session in which an order placed `now` would execute (VN time)."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(VN_TIMEZONE)
    today = local.date()
    if is_trading_day(today) and local.time() < MARKET_CLOSE:
        return today
    return next_trading_day(today)


def earliest_sell(buy_session: date) -> date:
    """Shares bought on T are sellable from the afternoon session of T+2."""
    return add_trading_days(buy_session, 2)


def session_distance(start: date, end: date) -> int:
    """Number of trading sessions from start to end (0 when the same session)."""
    if end < start:
        return -session_distance(end, start)
    return len(trading_days_between(start, end)) - (1 if is_trading_day(start) else 0)


def vn30f_expiry(year: int, month: int) -> date:
    first = date(year, month, 1)
    thursday = first + timedelta(days=(3 - first.weekday()) % 7)
    third = thursday + timedelta(days=14)
    return last_trading_day(third)


def next_vn30f_expiry(day: date) -> date:
    expiry = vn30f_expiry(day.year, day.month)
    if expiry >= day:
        return expiry
    year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
    return vn30f_expiry(year, month)


def holiday_table_end() -> date:
    return date(max(d.year for d in HOSE_HOLIDAYS), 12, 31)


def horizon_warning(trade_day: date, sessions: int = 10) -> str | None:
    end = add_trading_days(trade_day, sessions)
    limit = holiday_table_end()
    if end > limit:
        return (f"Horizon {sessions} sessions ends {end} after the holiday table end {limit}: "
                "session arithmetic beyond it ignores unknown holidays")
    return None

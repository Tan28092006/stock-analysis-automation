from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

VN_TIMEZONE = timezone(timedelta(hours=7))


def completed_session_date(now: datetime | None = None) -> date:
    """Conservative EOD cutoff: 16:00 Vietnam time, not the host's timezone.

    This is a data-availability buffer, not an assertion about auction close time.
    The maintained HOSE holiday table still needs refreshing for future years.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(VN_TIMEZONE)
    day = local.date() if local.hour >= 16 else local.date() - timedelta(days=1)
    return last_trading_day(day)


HOSE_HOLIDAYS = {
    # 2024
    date(2024, 1, 1),
    date(2024, 2, 8),
    date(2024, 2, 9),
    date(2024, 2, 12),
    date(2024, 2, 13),
    date(2024, 2, 14),
    date(2024, 4, 18),
    date(2024, 4, 30),
    date(2024, 5, 1),
    date(2024, 9, 2),
    date(2024, 9, 3),
    # 2025
    date(2025, 1, 1),
    date(2025, 1, 27),
    date(2025, 1, 28),
    date(2025, 1, 29),
    date(2025, 1, 30),
    date(2025, 1, 31),
    date(2025, 4, 7),
    date(2025, 4, 30),
    date(2025, 5, 1),
    date(2025, 9, 1),
    date(2025, 9, 2),
    # 2026
    # HOSE 2410/TB-SGDHCM (2025-12-25), HNX 5305/TB-SGDHN (2025-12-03).
    # Working Saturdays are not trading sessions. Source links in the audit report.
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 4, 27),
    date(2026, 4, 30),
    date(2026, 5, 1),
    date(2026, 8, 31),
    date(2026, 9, 1),
    date(2026, 9, 2),
}


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in HOSE_HOLIDAYS


def last_trading_day(day: date) -> date:
    current = day
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def next_trading_day(day: date) -> date:
    current = day + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)
    return current


def add_trading_days(day: date, days: int) -> date:
    if days < 0:
        raise ValueError("days must be non-negative")
    current = day
    for _ in range(days):
        current = next_trading_day(current)
    return current


def trading_days_between(start: date, end: date) -> list[date]:
    if start > end:
        return []
    days: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days

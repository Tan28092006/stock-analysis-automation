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
    # 2021-2023: exchange/broker notices, sources and scope in
    # docs/audits/2026-09-30-research-sources.md. Weekends remain excluded below.
    date(2021, 1, 1),
    date(2021, 2, 10), date(2021, 2, 11), date(2021, 2, 12),
    date(2021, 2, 15), date(2021, 2, 16), date(2021, 4, 21),
    date(2021, 4, 30), date(2021, 5, 3),
    date(2021, 9, 2), date(2021, 9, 3),
    date(2022, 1, 3), date(2022, 1, 31),
    date(2022, 2, 1), date(2022, 2, 2), date(2022, 2, 3), date(2022, 2, 4),
    date(2022, 4, 11), date(2022, 5, 2), date(2022, 5, 3),
    date(2022, 9, 1), date(2022, 9, 2),
    date(2023, 1, 2), date(2023, 1, 20), date(2023, 1, 23),
    date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26),
    date(2023, 5, 1), date(2023, 5, 2), date(2023, 5, 3),
    date(2023, 9, 1), date(2023, 9, 4),
    # 2024
    date(2024, 1, 1),
    date(2024, 2, 8),
    date(2024, 2, 9),
    date(2024, 2, 12),
    date(2024, 2, 13),
    date(2024, 2, 14),
    date(2024, 4, 18),
    # HNX holiday-swap notice 1977: https://www.hnx.vn/vi-vn/chi-tiet-lich-nghi-gd-60018631.html
    date(2024, 4, 29),
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
    # HNX notice 5386/TB-SGDHN; holiday swap, no trading on May 2.
    date(2025, 5, 2),
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


# Verified exchange transfers, not missing-price imputation. Primary-source links
# and inclusive suspension boundaries: docs/audits/2026-09-26-market-readiness.md.
SYMBOL_NONTRADING_INTERVALS = {
    # Issuer: last HNX October 5, first HOSE October 11 (2021).
    # https://www.shb.com.vn/shb-chinh-thuc-giao-dich-co-phieu-tren-hose-tu-ngay-11-10/
    "SHB": ((date(2021, 10, 6), date(2021, 10, 10)),),
    "BSR": ((date(2025, 1, 7), date(2025, 1, 16)),),
    "MCH": ((date(2025, 12, 18), date(2025, 12, 24)),),
}


def symbol_trading_days_between(symbol: str, start: date, end: date) -> list[date]:
    intervals = SYMBOL_NONTRADING_INTERVALS.get(symbol, ())
    return [day for day in trading_days_between(start, end)
            if not any(left <= day <= right for left, right in intervals)]

import unittest
from datetime import date

from stock_agent.data.exchange_calendar import (
    add_trading_days,
    is_trading_day,
    last_trading_day,
    next_trading_day,
    trading_days_between,
)


class ExchangeCalendarTests(unittest.TestCase):
    def test_weekend_is_not_trading_day(self):
        self.assertFalse(is_trading_day(date(2026, 5, 23)))
        self.assertEqual(last_trading_day(date(2026, 5, 23)), date(2026, 5, 22))

    def test_known_holidays_are_not_trading_days(self):
        for holiday in [
            date(2026, 2, 17),
            date(2026, 4, 27),
            date(2026, 4, 30),
            date(2026, 5, 1),
            date(2026, 9, 2),
        ]:
            self.assertFalse(is_trading_day(holiday))

    def test_next_trading_day_skips_holiday_and_weekend(self):
        self.assertEqual(next_trading_day(date(2026, 4, 29)), date(2026, 5, 4))

    def test_add_trading_days_for_thursday_entry(self):
        self.assertEqual(add_trading_days(date(2026, 5, 28), 2), date(2026, 6, 1))

    def test_trading_days_between_is_inclusive(self):
        days = trading_days_between(date(2026, 5, 22), date(2026, 5, 26))
        self.assertEqual(days, [date(2026, 5, 22), date(2026, 5, 25), date(2026, 5, 26)])


if __name__ == "__main__":
    unittest.main()

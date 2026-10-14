"""Regressions from replay data validation and causality review."""
from datetime import date

from scripts.period_backtest import _fillable
from stock_agent.data.exchange_calendar import is_trading_day, next_trading_day


def test_2024_april_holiday_swap_is_not_a_missing_price():
    assert not is_trading_day(date(2024, 4, 29))
    assert next_trading_day(date(2024, 4, 26)) == date(2024, 5, 2)
    assert not is_trading_day(date(2024, 5, 4))


def test_open_execution_gate_cannot_look_at_later_high_low_or_volume():
    at_limit = dict(open=107, high=107, low=107, close=107, volume=1000)
    later_recovery = dict(open=107, high=108, low=100, close=105, volume=9999999)
    assert _fillable(at_limit, 100, 'BUY', 6.9) == _fillable(later_recovery, 100, 'BUY', 6.9)
    assert not _fillable(later_recovery, 100, 'BUY', 6.9)

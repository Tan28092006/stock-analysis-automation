"""User-visible research contracts: actual horizons and historical sessions."""
from datetime import date

import pytest

from scripts import period_backtest as bt
from tests.test_period_backtest import prices, rules, order


@pytest.mark.parametrize('day', [
    '2021-02-10', '2021-04-21', '2021-05-03', '2021-09-03',
    '2022-01-03', '2022-01-31', '2022-02-04', '2022-04-11',
    '2022-05-03', '2022-09-01', '2023-01-02', '2023-01-20',
    '2023-01-26', '2023-05-03', '2023-09-04',
])
def test_historical_exchange_holidays_are_not_execution_sessions(day):
    assert not bt.trading_days_between(date.fromisoformat(day), date.fromisoformat(day))


def test_fixed_hold_ignores_stop_target_and_exits_at_original_horizon(rules):
    frame = prices()
    frame.loc[4, ['low', 'high']] = [80, 120]
    signals = {'2025-12-31': [order(hold=4)]}
    bracket = bt.replay_mr({'AAA': frame, 'VNINDEX': prices()}, rules,
                           '2026-01-01', '2026-01-12', signals)
    fixed = bt.replay_mr({'AAA': frame, 'VNINDEX': prices()}, rules,
                         '2026-01-01', '2026-01-12', signals, exit_policy='fixed_hold')
    assert bracket['fills'][0] == fixed['fills'][0]
    assert bracket['fills'][1]['reason'] == 'stop'
    assert fixed['fills'][1]['date'] == '2026-01-09'
    assert fixed['fills'][1]['reason'] == 'time'


def test_fixed_hold_unmatured_lot_stays_open(rules):
    frame = prices()
    frame.loc[4, ['low', 'high']] = [80, 120]
    result = bt.replay_mr({'AAA': frame, 'VNINDEX': prices()}, rules,
                          '2026-01-01', '2026-01-09', {'2025-12-31': [order()]},
                          exit_policy='fixed_hold')
    assert len(result['fills']) == 1
    assert result['open_positions']['AAA'][0]['expiry'] > '2026-01-09'


def test_unknown_exit_policy_fails_before_trading(rules):
    with pytest.raises(ValueError, match='exit_policy'):
        bt.replay_mr({'AAA': prices(), 'VNINDEX': prices()}, rules,
                     '2026-01-01', '2026-01-09', {}, exit_policy='typo')

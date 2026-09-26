"""Causality and accounting contracts for the isolated historical replay."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import period_backtest as bt


@pytest.fixture
def rules():
    return json.loads(Path('configs/rules_mr.json').read_text(encoding='utf-8'))


def prices(dates=None, close=100.0):
    dates = dates or ['2025-12-31', '2026-01-05', '2026-01-06',
                      '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-12']
    return pd.DataFrame(dict(date=dates, open=close, high=close * 1.01,
                             low=close * .99, close=close, volume=1_000_000))


def order(symbol='AAA', stop=90, target=110, hold=15):
    return dict(symbol=symbol, stop=stop, target=target, hold=hold, rr=1.0)


def test_next_open_entry_and_end_mark_not_forced_liquidation(rules):
    frame = prices()
    frame.loc[1, ['open', 'close', 'high']] = [105, 107, 108]
    result = bt.replay_mr({'AAA': frame, 'VNINDEX': prices()}, rules,
                          '2026-01-01', '2026-01-05', {'2025-12-31': [order()]})
    fill = result['fills'][0]
    assert fill['date'] == '2026-01-05'
    assert fill['price'] == pytest.approx(105 * 1.001)
    assert fill['qty'] % 100 == 0
    assert result['metrics']['closed_lots'] == 0
    assert len(result['open_positions']) == 1
    assert result['nav'][-1]['nav'] == pytest.approx(
        1e9 - fill['qty'] * fill['price'] * 1.0015 + fill['qty'] * 107)


def test_t3_lock_stop_first_and_gap_fill(rules):
    frame = prices()
    frame.loc[1:4, ['low', 'high']] = [80, 120]
    frame.loc[4, 'open'] = 96
    result = bt.replay_mr({'AAA': frame, 'VNINDEX': prices()}, rules,
                          '2026-01-01', '2026-01-09', {'2025-12-31': [order(stop=97)]})
    sell = result['fills'][1]
    assert sell['date'] == '2026-01-08'
    assert sell['reason'] == 'stop'
    assert sell['price'] == pytest.approx(96 * .999)
    assert result['metrics']['win_rate_pct'] == 0


def test_intraday_exit_cannot_finance_same_morning_buy(rules):
    rules['money'].update(max_positions=1, max_weight_pct=100, max_exposure_pct=100,
                          risk_per_trade_pct=100)
    a = prices()
    a.loc[4, 'high'] = 112
    signals = {'2025-12-31': [order()], '2026-01-07': [order('BBB')]}
    result = bt.replay_mr({'AAA': a, 'BBB': prices(), 'VNINDEX': prices()}, rules,
                          '2026-01-01', '2026-01-09', signals)
    assert [f['symbol'] for f in result['fills'] if f['side'] == 'BUY'] == ['AAA']


def test_entry_quantity_does_not_depend_on_same_day_close(rules):
    a = prices()
    b = prices()
    signals = {'2025-12-31': [order()], '2026-01-05': [order('BBB')]}
    inputs = {'AAA': a, 'BBB': b, 'VNINDEX': prices()}
    first = bt.replay_mr(inputs, rules, '2026-01-01', '2026-01-09', signals)
    altered = copy.deepcopy(inputs)
    altered['AAA'].loc[2, ['close', 'high']] = [1000, 1001]
    second = bt.replay_mr(altered, rules, '2026-01-01', '2026-01-09', signals)
    assert first['fills'][:2] == second['fills'][:2]


def test_future_bars_do_not_change_replay(rules):
    a = prices()
    inputs = {'AAA': a, 'VNINDEX': a.copy()}
    before = bt.replay_mr(inputs, rules, '2026-01-01', '2026-01-09',
                          {'2025-12-31': [order()]})
    inputs['AAA'].loc[6, ['open', 'high', 'low', 'close']] = [200, 210, 180, 190]
    after = bt.replay_mr(inputs, rules, '2026-01-01', '2026-01-09',
                         {'2025-12-31': [order()]})
    assert before == after


@pytest.mark.parametrize('kind', ['gap', 'duplicate', 'nan'])
def test_invalid_or_missing_prices_fail_closed(kind):
    frame = prices()
    if kind == 'gap':
        frame = frame.drop(index=2)
    elif kind == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    else:
        frame.loc[2, 'close'] = np.nan
    with pytest.raises(ValueError):
        bt.validate_frames({'AAA': frame, 'VNINDEX': prices()}, '2026-01-01', '2026-01-09')


def test_broker_cash_settlement_and_fee_accounting(rules):
    broker = bt.Broker(rules)
    broker.buy('AAA', 100, 100, '2026-01-05')
    assert broker.sell('AAA', 100, 110, '2026-01-07', 'rebalance') == 0
    assert broker.sell('AAA', 100, 110, '2026-01-08', 'rebalance') == 100
    cash = broker.cash
    broker.settle('2026-01-12')
    assert broker.cash == cash
    broker.settle('2026-01-13')
    expected = 1e9 - 100 * 100 * 1.001 * 1.0015 + 100 * 110 * .999 * .9975
    assert broker.cash == pytest.approx(expected)
    assert sum(x['pnl'] for x in broker.closed) == pytest.approx(expected - 1e9)


def test_target_and_time_exits_and_skipped_gap(rules):
    a = prices()
    a.loc[4, 'high'] = 112
    hit = bt.replay_mr({'AAA': a, 'VNINDEX': prices()}, rules, '2026-01-01',
                        '2026-01-09', {'2025-12-31': [order()]})
    assert hit['fills'][1]['reason'] == 'target'
    timed = bt.replay_mr({'AAA': prices(), 'VNINDEX': prices()}, rules, '2026-01-01',
                          '2026-01-09', {'2025-12-31': [order(hold=3)]})
    assert timed['fills'][1]['reason'] == 'time'
    a.loc[1, 'open'] = 120
    skipped = bt.replay_mr({'AAA': a, 'VNINDEX': prices()}, rules, '2026-01-01',
                            '2026-01-09', {'2025-12-31': [order()]})
    assert not skipped['fills']


def test_monthly_rebalance_uses_previous_close_only(monkeypatch, rules):
    dates = ['2026-01-29', '2026-01-30', '2026-02-02', '2026-02-03',
             '2026-02-04', '2026-02-05', '2026-02-06']
    seen = []

    def target(frames, held):
        seen.append(max(frames['AAA']['date']))
        return {'AAA': .5}, []

    monkeypatch.setattr(bt, 'momentum_targets', target)
    result = bt.replay_momentum({'AAA': prices(dates), 'VNINDEX': prices(dates)},
                                rules, '2026-02-01', '2026-02-06')
    assert seen == ['2026-01-30']
    assert result['fills'][0]['date'] == '2026-02-02'
    assert all(r['cash'] >= -1e-5 for r in result['nav'])


def test_momentum_history_gate_and_buffer():
    dates = pd.bdate_range('2024-01-01', periods=300).strftime('%Y-%m-%d').tolist()
    frames = {}
    for i in range(22):
        f = prices(dates)
        f['close'] = 100 * np.exp(np.arange(300) * (i + 1) * .0001 + np.sin(np.arange(300)) * .001)
        frames[f'S{i:02}'] = f
    frames['NEW'] = prices(dates[-100:])
    frames['VNINDEX'] = frames['S10'].copy()
    weights, excluded = bt.momentum_targets(frames, {'S02'})
    assert len(weights) == 10
    assert 'S02' in weights  # rank 20, retained inside top-20 buffer
    assert excluded == ['NEW']
    assert sum(weights.values()) <= 1.00000001


def test_production_signal_prefix_invariance(rules):
    dates = pd.bdate_range('2024-01-01', periods=180).strftime('%Y-%m-%d').tolist()
    f = prices(dates)
    c = 100 + np.sin(np.arange(180) / 3) * 10
    f['close'], f['open'], f['high'], f['low'] = c, c, c + 2, c - 2
    prefix = bt.mr_signals({'AAA': f.iloc[:140]}, rules)
    full = bt.mr_signals({'AAA': f}, rules)
    assert prefix == {d: orders for d, orders in full.items() if d <= dates[139]}


def test_metrics_include_initial_drawdown_and_monthly_compounding():
    nav = [{'date': '2026-01-30', 'nav': 900}, {'date': '2026-02-02', 'nav': 990}]
    metrics = bt.metrics(nav, [], 1000)
    assert metrics['return_pct'] == pytest.approx(-1)
    assert metrics['max_drawdown_pct'] == pytest.approx(-10)
    assert metrics['monthly_pct'] == pytest.approx({'2026-01': -10, '2026-02': 10})

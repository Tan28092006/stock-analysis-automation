"""Preregistered event engine: synthetic causality, cash and activity contracts."""
import copy
import importlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.historical_universe import Timeline
from scripts.momentum_hypotheses import reconcile
from stock_agent.data.exchange_calendar import trading_days_between


def lab():
    return importlib.import_module('scripts.momentum_events')


@pytest.fixture
def market():
    days = [d.isoformat() for d in trading_days_between(date(2024, 6, 3), date(2025, 9, 30))]
    base = pd.DataFrame(dict(date=days, open=10000., high=10100., low=9900.,
                             close=10000., volume=1_000_000.))
    start_i = next(i for i, d in enumerate(days) if d >= '2025-05-05')
    frames = {s: base.copy() for s in ('AAA', 'BBB', 'VNINDEX')}
    timeline = Timeline(dict(coverage_start=days[0], coverage_end=days[-1],
        initial_known_on=days[0], initial_members=['AAA', 'BBB'], member_count=2, changes=[]))
    rules = json.loads(Path('configs/rules_mr.json').read_text())
    return frames, rules, timeline, start_i


def set_bar(frame, i, close, *, opening=None, volume=1_000_000.):
    opening = close if opening is None else opening
    frame.loc[i, ['open', 'high', 'low', 'close', 'volume']] = [
        opening, max(opening, close) + 20, min(opening, close) - 20, close, volume]


def rally(market, *, offset=1, count=20):
    frames, _, _, i = market
    for j in range(count):
        set_bar(frames['AAA'], i + offset + j, 10200 + 75 * j)
    return i + offset


def run(market, variant='daily55', scenario='normal', *, length=30):
    frames, rules, timeline, i = market
    days = frames['VNINDEX'].date
    return lab().replay(frames, rules, timeline, days[i], days[i + length - 1], variant, scenario)


def buys(result):
    return [f for f in result['fills'] if f['side'] == 'BUY']


def test_registered_protocol_is_closed_and_not_live():
    p = lab().load_protocol()
    assert p['live_eligible'] is False and p['holdout'] is False
    assert p['monthly_activity_floor'] is None
    assert len(p['variants']) * len(p['scenarios']) == 18


def test_features_exclude_signal_bar_and_use_seeded_atr(market):
    f = market[0]['AAA'].iloc[:22].copy()
    set_bar(f, 20, 10500., opening=10000., volume=2_000_000.)
    got = lab().features(f, 20)
    assert got.iloc[19].atr == 200
    assert got.iloc[20].atr == pytest.approx((19 * 200 + 540) / 20)
    assert got.iloc[20].entry_high == 10100
    assert got.iloc[20].exit_low == 9900
    assert got.iloc[20].prior_volume == 1_000_000
    assert np.isnan(got.iloc[18].atr)


def test_daily_can_open_midweek_without_rebalance(market):
    signal = rally(market)
    result = run(market)
    assert buys(result)[0]['date'] == market[0]['AAA'].date[signal + 1]
    assert buys(result)[0]['signal_date'] == market[0]['AAA'].date[signal]
    assert buys(result)[0]['intent_kind'] == 'new'
    assert len(buys(result)) == 1
    assert buys(run(market, 'weekly55'))[0]['date'] > buys(result)[0]['date']
    assert reconcile(result, market[0], market[1]['backtest']['initial_capital'])['max_nav_error_vnd'] < 1e-5


def test_delay_freezes_signal_and_quantity(market):
    signal = rally(market)
    normal, delayed = run(market), run(market, scenario='delay_one_session')
    assert buys(delayed)[0]['date'] == market[0]['AAA'].date[signal + 2]
    assert buys(delayed)[0]['signal_date'] == buys(normal)[0]['signal_date']
    assert buys(delayed)[0]['planned_qty'] == buys(normal)[0]['planned_qty']
    assert buys(delayed)[0]['qty'] <= buys(normal)[0]['planned_qty']


@pytest.mark.parametrize('scenario', ['normal', 'double_cost', 'delay_one_session'])
def test_execution_hlcv_and_future_cannot_change_earlier_fills(market, scenario):
    rally(market)
    before = run(market, scenario=scenario)
    day = buys(before)[0]['date']
    changed = copy.deepcopy(market)
    f = changed[0]['AAA']
    j = f.index[f.date == day][0]
    f.loc[j, ['high', 'low', 'close', 'volume']] = [20000, 1000, 19000, 10]
    for k in range(j + 1, len(f)):
        set_bar(f, k, 40000.)
    after = run(changed, scenario=scenario)
    assert [x for x in before['fills'] if x['date'] <= day] == [x for x in after['fills'] if x['date'] <= day]


def test_prefix_invariance(market):
    rally(market)
    short, long = run(market, length=12), run(market, length=50)
    cutoff = short['nav'][-1]['date']
    assert short['nav'] == [r for r in long['nav'] if r['date'] <= cutoff]
    assert short['fills'] == [r for r in long['fills'] if r['date'] <= cutoff]


def test_adds_only_up_not_counted_as_new_and_capped_four_units(market):
    rally(market)
    market[0]['AAA']['volume'] = 100000.
    result = run(market, 'pyramiding55', length=20)
    orders = buys(result)
    assert [f['intent_kind'] for f in orders] == ['new', 'add', 'add', 'add']
    assert all(b['price'] > a['price'] for a, b in zip(orders, orders[1:]))
    assert all(f['qty'] <= orders[0]['qty'] for f in orders)
    assert len({f['date'] for f in orders}) == len(orders)
    assert sum(m['new_positions'] for m in result['activity']['months'].values()) == 1
    assert sum(m['add_on_buys'] for m in result['activity']['months'].values()) == 3


def test_zero_entry_months_are_not_dropped(market):
    result = run(market, length=90)
    activity = result['activity']
    assert len(activity['months']) >= 4
    assert len(activity['zero_new_entry_months']) == len(activity['months'])
    assert activity['longest_quiet_sessions'] == 90
    assert activity['numerical_activity_pass'] is None


@pytest.mark.parametrize('variant', ['market55', 'volume55'])
def test_extra_gates_block_without_changing_baseline(market, variant):
    rally(market)
    assert buys(run(market))
    assert not buys(run(market, variant))
    assert run(market, variant)['funnel'][variant.replace('55', '_gate')] > 0


def test_liquidity_uses_only_lagged_volume_and_board_lots(market):
    signal = rally(market)
    market[0]['AAA'].loc[:signal, 'volume'] = 15000.
    result = run(market, length=4)
    assert buys(result)[0]['qty'] == 100
    assert buys(result)[0]['adv_cap_qty'] == 100


def test_zero_adv_blocks_not_invented_as_liquidity(market):
    signal = rally(market)
    market[0]['AAA'].loc[:signal, 'volume'] = 0.
    result = run(market, length=3)
    assert not buys(result)
    assert result['funnel']['liquidity'] > 0


def test_exit_latches_through_t3_even_after_rebound(market):
    frames, _, _, i = market
    set_bar(frames['AAA'], i, 10200.)
    set_bar(frames['AAA'], i + 1, 9500., opening=10200.)
    for j in range(i + 2, i + 15):
        set_bar(frames['AAA'], j, 10500.)
    result = run(market, length=5)
    sold = [f for f in result['fills'] if f['side'] == 'SELL']
    assert sold and sold[0]['date'] == frames['AAA'].date[i + 4]
    assert sold[0]['signal_date'] == frames['AAA'].date[i + 1]
    assert len(buys(result)) == 1
    reconcile(result, frames, market[1]['backtest']['initial_capital'])


def test_known_effective_membership_rechecked_on_delayed_fill(market):
    frames, rules, timeline, i = market
    rally(market, offset=0)
    data = copy.deepcopy(timeline.data)
    data['initial_members'] = ['AAA', 'CCC']
    frames['CCC'] = frames['BBB'].copy()
    data['changes'] = [dict(known_on=frames['AAA'].date[i + 1],
        effective_on=frames['AAA'].date[i + 2], add=['BBB'], remove=['AAA'])]
    result = run((frames, rules, Timeline(data), i), scenario='delay_one_session', length=6)
    assert not buys(result)
    assert result['funnel']['membership'] > 0


@pytest.mark.parametrize('field,value', [('ml', {'enabled': True}), ('backtest', {'lot_size': 1})])
def test_invalid_rules_fail_closed(market, field, value):
    market[1][field].update(value)
    with pytest.raises(ValueError):
        run(market)


def test_unknown_variant_and_scenario_fail_closed(market):
    with pytest.raises(ValueError):
        run(market, 'optimized_after_seeing_result')
    with pytest.raises(ValueError):
        run(market, scenario='free_fills')


def test_future_listing_does_not_block_or_enter_an_earlier_window(market):
    frames, _, _, i = market
    rally(market)
    frames['BBB'] = frames['BBB'].iloc[i + 35:].reset_index(drop=True)
    result = run(market)
    assert all(f['symbol'] == 'AAA' for f in buys(result))
    assert result['funnel']['history_unavailable'] > 0


def test_partial_exit_retains_latch_and_never_spends_receivable(market):
    frames, _, _, i = market
    set_bar(frames['AAA'], i, 10200.)
    set_bar(frames['AAA'], i + 1, 9500., opening=10200.)
    frames['AAA'].loc[i + 1:, 'volume'] = 20000.
    for j in range(i + 2, i + 12):
        set_bar(frames['AAA'], j, 10200., volume=20000.)
    result = run(market, length=10)
    sells = [f for f in result['fills'] if f['side'] == 'SELL']
    assert len(sells) >= 2
    assert len({f['signal_date'] for f in sells}) == 1
    assert all(f['qty'] <= f['adv_cap_qty'] for f in sells)
    assert len(buys(result)) == 1
    reconcile(result, frames, market[1]['backtest']['initial_capital'])


def test_delayed_add_cancelled_by_new_exit(market):
    frames, _, _, i = market
    rally(market, offset=0)
    frames['AAA']['volume'] = 100000.
    set_bar(frames['AAA'], i + 5, 9000., opening=10575.)
    result = run(market, 'pyramiding55', 'delay_one_session', length=9)
    assert result['funnel'].get('cancelled_by_exit', 0) > 0


def test_buy_caps_are_hard_and_do_not_rebalance_down(market):
    rally(market)
    result = run(market, 'pyramiding55', length=20)
    assert sum(f['qty'] * f['price'] for f in buys(result)) < market[1]['backtest']['initial_capital'] * .21
    assert result['funnel'].get('weight', 0) > 0
    assert not [f for f in result['fills'] if f['side'] == 'SELL']

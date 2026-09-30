"""Locked research extension, not deployment approval."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import period_backtest as bt
from tests.test_momentum_hypotheses import inputs, frame, rules
from scripts.historical_universe import Timeline


def lab():
    return importlib.import_module('scripts.momentum_books')


def test_weekly_cadence_prior_close_holiday_and_default_parity(rules):
    dates = ['2026-04-24','2026-04-27','2026-04-28','2026-04-29','2026-05-04','2026-05-05','2026-05-11']
    data = {s:frame(np.zeros(len(dates)-1),dates) for s in ['AAA','VNINDEX']}
    seen = []
    def target(prefix, held):
        seen.append(prefix['AAA'].date.iloc[-1])
        return {'AAA':.5}, []
    weekly = bt.replay_momentum(data,rules,dates[1],dates[-1],target_fn=target,rebalance='weekly')
    assert seen == ['2026-04-24','2026-04-29','2026-05-05']
    assert len(weekly['rebalances'])==3
    assert bt.replay_momentum(data,rules,dates[1],dates[-1],target_fn=target) == bt.replay_momentum(data,rules,dates[1],dates[-1],target_fn=target,rebalance='monthly')
    with pytest.raises(ValueError,match='rebalance'):
        bt.replay_momentum(data,rules,dates[1],dates[-1],rebalance='hourly')


def test_slope_score_independent_formula_and_continuous_path():
    close=np.exp(np.arange(90)*.003)*100
    assert lab().slope_score(close)==pytest.approx(np.expm1(.003*252))
    assert lab().slope_score(np.ones(90))==0
    with pytest.raises(ValueError): lab().slope_score([1,0])


def test_daily_information_quality_matches_formula_and_skips_last_month():
    f=frame(np.r_[np.full(231,.001),np.full(21,-.03),.001])
    c=f.close.to_numpy()
    r=np.diff(c[-253:-21])/c[-253:-22]
    expected=np.sign(c[-22]/c[-253]-1)*(np.mean(r>0)-np.mean(r<0))
    assert lab().quality_score(c)==pytest.approx(expected)
    edited=c.copy(); edited[-21:]*=3
    assert lab().quality_score(edited)==lab().quality_score(c)


def test_breakout_excludes_signal_bar_from_high_and_volume_baselines():
    f=frame(np.zeros(60)); f.loc[f.index[-1],['close','high','volume']]=[104,110,1.5e6]
    assert lab().breakout_allowed(f,volume=True)
    f.loc[f.index[-1],'volume']=1.49e6
    assert lab().breakout_allowed(f,volume=False)
    assert not lab().breakout_allowed(f,volume=True)
    f.loc[f.index[-2],'high']=104
    assert not lab().breakout_allowed(f)
    assert not lab().breakout_allowed(f.tail(10))


def test_frequency_does_not_count_additions_as_independent_entries():
    fills=[dict(date='a',symbol='A',side='BUY',qty=100),dict(date='b',symbol='A',side='BUY',qty=100),
           dict(date='c',symbol='A',side='SELL',qty=200),dict(date='d',symbol='A',side='BUY',qty=100)]
    result=lab().frequency(dict(fills=fills,rebalances=[{},{}]))
    assert result == dict(buy_fills=3,new_entries=2,additions=1,buy_dates=3,unique_bought_symbols=1,rebalances=2)


@pytest.mark.parametrize('variant',['baseline','weekly','slope90','quality','breakout50','breakout50_volume'])
def test_variant_runs_causally_and_reconciles(variant,rules):
    data=inputs()
    timeline=Timeline(dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
                           member_count=2,initial_members=['AAA','BBB'],changes=[]))
    start,end='2025-01-01','2025-02-03'
    original=lab().run_one(data,rules,timeline,start,end,variant,'normal')
    assert original['reconciliation']['max_nav_error_vnd']<1e-5
    future=copy.deepcopy(data)
    for f in future.values(): f.loc[f.date>end,['open','high','low','close']]*=9
    assert lab().run_one(future,rules,timeline,start,end,variant,'normal')==original
    assert original['summary']['buys']>0 or variant.startswith('breakout')


def test_protocol_retains_windows_controls_and_rejects_unknowns(tmp_path):
    p=lab().load_protocol()
    assert p['family']==27 and p['holdout'] is False
    p['variants'].remove('weekly')
    file=tmp_path/'bad.json'; file.write_text(json.dumps(p))
    with pytest.raises(ValueError): lab().load_protocol(file)
    with pytest.raises(ValueError): lab().targets(inputs(),set(),'typo')


def test_mr_funnel_is_nested_and_matches_registered_bank(rules):
    data=inputs()
    timeline=Timeline(dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
                           member_count=2,initial_members=['AAA','BBB'],changes=[]))
    f=lab().mr_funnel(data,rules,timeline,'2025-01-01','2025-02-03')
    counts=list(f['sequential_pass'].values())
    assert counts[0]>0 and counts==sorted(counts,reverse=True)
    assert f['signals']==sum(len(v) for v in f['signal_days'].values())


def test_quality_ranking_actually_changes_selected_names_not_only_score():
    data={'VNINDEX':frame(.001+np.sin(np.arange(300))*.005)}
    for i in range(20):
        returns=(.004+np.sin(np.arange(300))*.03) if i<10 else (.002+np.cos(np.arange(300))*.0002)
        data[f'S{i:02}']=frame(returns+i*.000001)
    generic,_=lab().targets(data,set(),'weekly')
    quality,_=lab().targets(data,set(),'quality')
    assert set(generic)=={f'S{i:02}' for i in range(10)}
    assert set(quality)=={f'S{i:02}' for i in range(10,20)}
    assert sum(quality.values())<=1.0000000001
    data['NEW']=data['S00'].tail(100)
    assert 'NEW' not in lab().targets(data,set(),'slope90')[0]


def test_weekly_future_changes_do_not_modify_earlier_fills(rules):
    data=inputs()
    timeline=Timeline(dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
                           member_count=2,initial_members=['AAA','BBB'],changes=[]))
    short=lab().run_one(data,rules,timeline,'2025-01-01','2025-01-17','slope90','normal')['replay']
    changed=copy.deepcopy(data)
    for f in changed.values(): f.loc[f.date>'2025-01-17',['open','high','low','close']]*=2
    long=lab().run_one(changed,rules,timeline,'2025-01-01','2025-02-03','slope90','normal')['replay']
    assert short['fills'] and short['fills']==[f for f in long['fills'] if f['date']<='2025-01-17']
    assert short['nav']==[n for n in long['nav'] if n['date']<='2025-01-17']


@pytest.mark.parametrize('scenario',['double_cost','delay_one_session'])
def test_weekly_stresses_reconcile_and_do_not_mutate_rules(scenario,rules):
    original=copy.deepcopy(rules)
    timeline=Timeline(dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
                           member_count=2,initial_members=['AAA','BBB'],changes=[]))
    run=lab().run_one(inputs(),rules,timeline,'2025-01-01','2025-02-03','weekly',scenario)
    assert run['replay']['fills'] and run['reconciliation']['max_nav_error_vnd']<1e-5
    assert rules==original

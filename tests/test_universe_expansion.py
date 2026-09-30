"""Paired universe experiment contracts, no live-state writes."""
import copy
import importlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.historical_universe import Timeline
from stock_agent.data.exchange_calendar import trading_days_between
from tests.test_momentum_hypotheses import inputs, rules


def lab():
    return importlib.import_module('scripts.universe_expansion')


def test_protocol_registered_before_results_and_freezes_paired_design():
    p = lab().load_protocol()
    assert p['start'] == '2026-01-01' and p['end'] == '2026-06-30'
    assert p['family'] == 88 and p['holdout'] is False
    assert p['participation_warning_pct'] == 1.
    names = lab().trial_names()
    assert len(names) == len(set(names)) == 44
    assert 'books/baseline/normal' not in names
    assert 'momentum/universe_equal/normal' in names


@pytest.mark.parametrize('key,value', [('holdout',True),('family',1),('end','2026-07-31'),
                                      ('participation_warning_pct',5),('scenarios',['normal'])])
def test_changed_protocol_requires_version(tmp_path,key,value):
    p = json.loads(Path('configs/research/vn100_h1_v1.json').read_text())
    p[key] = value
    path = tmp_path/'changed.json'
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): lab().load_protocol(path)


def test_bounded_membership_excludes_out_of_period_union_and_does_not_mutate():
    data = dict(coverage_start='2025-01-01',coverage_end='2027-01-01',member_count=1,
        initial_known_on='2025-01-01',initial_members=['AAA'],changes=[
            dict(known_on='2025-10-01',effective_on='2025-10-02',add=['BBB'],remove=['AAA']),
            dict(known_on='2025-12-15',effective_on='2026-02-02',add=['CCC'],remove=['BBB']),
            dict(known_on='2026-07-01',effective_on='2026-08-03',add=['DDD'],remove=['CCC'])])
    before = copy.deepcopy(data)
    t = lab().bounded_timeline(data,'2025-12-31','2026-06-30')
    assert t.all_members == {'BBB','CCC'}
    assert t.members('2025-12-31','2026-02-02') == {'CCC'}
    assert data == before


def test_participation_uses_only_prior_twenty_sessions_and_flags_insufficient_history():
    dates = pd.bdate_range('2026-01-01',periods=23).strftime('%Y-%m-%d')
    frames = {'AAA':pd.DataFrame({'date':dates,'volume':[1000.]*20+[1e9]*3})}
    fills = [dict(date=dates[20],symbol='AAA',qty=100,side='BUY',price=10.,fee=0.),
             dict(date=dates[1],symbol='AAA',qty=100,side='SELL',price=10.,fee=0.)]
    report = lab().execution_diagnostics(fills,frames)
    assert report['fills'][0]['lagged_adv20'] == 1000.
    assert report['fills'][0]['participation_pct'] == 10.
    assert report['above_limit'] == 1 and report['unknown'] == 1
    assert report['capacity_pass'] is False
    frames['AAA'].loc[20:,'volume'] *= 3
    assert lab().execution_diagnostics(fills,frames) == report


def test_paired_comparison_never_calls_losing_less_profit():
    a = dict(return_pct=-10.,index_return_pct=-20.,new_entries=3,additions=2,
             fees_vnd=10.,max_drawdown_pct=-15.)
    b = dict(a,return_pct=-15.,new_entries=2)
    c = lab().compare_metrics(b,a)
    assert c['vn100_minus_vn30_pp'] == 5.
    assert c['vn100_beats_index'] and c['vn100_beats_vn30']
    assert not c['vn100_positive_net_and_beats_both']
    with pytest.raises(ValueError): lab().compare_metrics(b,dict(a,index_return_pct=-21.))


def test_cli_and_ci_cannot_silently_skip_vn100_input():
    workflow = Path('.github/workflows/research-gate.yml').read_text()
    assert 'scripts.universe_expansion' in workflow
    assert 'test_universe_expansion.py' in workflow
    assert 'research-vn100-h1-v1.zip' in workflow
    assert 'research-snapshot-v1.zip' in workflow
    assert 'vars.RESEARCH_SNAPSHOT_RUN_ID' not in workflow


def test_same_universe_produces_identical_replays_and_future_prefix_is_invariant(rules):
    f = inputs()
    start,end = f['AAA'].date.iloc[-15],f['AAA'].date.iloc[-1]
    t = Timeline(dict(coverage_start='2024-01-01',coverage_end=end,member_count=2,
                     initial_known_on='2024-01-01',initial_members=['AAA','BBB'],changes=[]))
    before = copy.deepcopy(f)
    a = lab().run_trial(f,rules,t,{},start,end,'momentum/baseline/normal')
    b = lab().run_trial(f,rules,t,{},start,end,'momentum/baseline/normal')
    assert a == b and a['replay']['fills']
    cutoff = f['AAA'].date.iloc[-5]
    changed = copy.deepcopy(f)
    for frame in changed.values():
        frame.loc[frame.date>cutoff,['open','high','low','close']] *= 9
    c = lab().run_trial(changed,rules,t,{},start,end,'momentum/baseline/normal')
    assert [v for v in a['replay']['nav'] if v['date']<=cutoff] == [v for v in c['replay']['nav'] if v['date']<=cutoff]
    for s in f: pd.testing.assert_frame_equal(f[s],before[s])


def test_complete_synthetic_cli_reconciles_partitions_and_is_immutable(tmp_path,monkeypatch,rules):
    m = lab()
    f = inputs()
    days = [str(d) for d in trading_days_between(date(2024,1,1),date(2026,6,30))]
    for frame in f.values(): frame['date'] = days[:len(frame)]
    start,end = f['AAA'].date.iloc[-15],f['AAA'].date.iloc[-1]
    t = Timeline(dict(coverage_start='2024-01-01',coverage_end=end,member_count=2,
        initial_known_on='2024-01-01',initial_members=['AAA','BBB'],changes=[]))
    p = m.load_protocol()
    p.update(start=start,end=end,event=dict(id='war_2026',kind='event',start=start,end=end))
    monkeypatch.setattr(m,'load_protocol',lambda:p)
    monkeypatch.setattr(m,'load_inputs',lambda *args:(f,{'VN30':t,'VN100':t},rules))
    monkeypatch.setattr(m,'trial_names',lambda:['momentum/baseline/normal'])
    monkeypatch.setattr(m.gate,'mr_signal_bank',lambda *args:{'bracket':{},'fixed15':{}})
    monkeypatch.setattr(m,'verify_snapshot',lambda _:None)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    output = tmp_path/'result'
    m.main(['--manifest',str(manifest),'--output',str(output)])
    r = json.loads((output/'results.json').read_text())
    assert r['status'] == 'universe_research_complete_full_regression_required'
    assert r['partition_reconciled'] and not r['live_eligible']
    assert all(c['vn100_minus_vn30_pp']==0 for b in r['blocks'] for c in b['comparisons']['cash_restart'].values())
    assert r['trial_count'] == 2 + 18*(len(r['blocks'])-1)
    assert (output/'report.html').exists() and (output/'regime_labels.json').exists()
    assert all(not t['sample_adequate'] for t in r['paired_diagnostics'].values())
    with pytest.raises(FileExistsError): m.run_suite(manifest,output)

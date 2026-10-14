"""Causal regimes and honest split-period accounting, not winner selection."""
import copy
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_momentum_hypotheses import inputs, rules
from scripts.historical_universe import Timeline


def lab():
    return importlib.import_module('scripts.research_timeslices')


def market(close):
    return pd.DataFrame({'date': pd.bdate_range('2021-01-01', periods=len(close)).strftime('%Y-%m-%d'),
                         'close': close, 'open': close})


@pytest.mark.parametrize('close,label', [(np.arange(1, 251), 'uptrend'),
    (np.arange(251, 1, -1), 'downtrend'), (np.ones(250), 'transition')])
def test_regime_uses_prior_close_and_sma_slope(close, label):
    f = market(close)
    result = lab().classify_regimes(f, f.date.iloc[220], f.date.iloc[-1])
    assert set(result.regime) == {label}
    first = result.iloc[0]
    assert first.known_on == f.date.iloc[219]
    assert first.sma200 == pytest.approx(f.close.iloc[20:220].mean())
    assert first.sma200_previous == pytest.approx(f.close.iloc[:200].mean())


def test_future_and_same_day_price_cannot_change_current_label():
    f = market(np.arange(1., 301.))
    day = f.date.iloc[240]
    original = lab().classify_regimes(f, f.date.iloc[220], day)
    f.loc[240:, 'close'] = 0.01
    pd.testing.assert_frame_equal(original, lab().classify_regimes(f, f.date.iloc[220], day))


@pytest.mark.parametrize('issue', ['duplicate', 'unsorted', 'nan', 'nonpositive', 'warmup'])
def test_bad_index_fails_closed(issue):
    f = market(np.arange(1., 251.))
    start = f.date.iloc[220]
    if issue == 'duplicate': f.loc[230, 'date'] = f.date.iloc[229]
    if issue == 'unsorted': f = f.iloc[::-1]
    if issue == 'nan': f.loc[100, 'close'] = np.nan
    if issue == 'nonpositive': f.loc[100, 'close'] = 0
    if issue == 'warmup': f = f.iloc[20:]
    with pytest.raises(ValueError): lab().classify_regimes(f, start, f.date.max())


def test_episodes_keep_short_transitions_and_partition_all_sessions_once():
    f = pd.DataFrame({'date':['2026-01-05','2026-01-06','2026-01-07','2026-01-08'],
                      'regime':['uptrend','transition','downtrend','downtrend']})
    episodes = lab().regime_slices(f)
    assert [(x['regime'], x['sessions']) for x in episodes] == [('uptrend',1),('transition',1),('downtrend',2)]
    assert [d for x in episodes for d in f.loc[f.date.between(x['start'],x['end']), 'date']] == f.date.tolist()


def test_half_years_keep_2026_h1_and_mark_incomplete_boundaries():
    slices = lab().calendar_slices('2022-09-01','2026-09-29')
    assert len(slices) == 9
    assert slices[0]['partial'] and slices[-1]['partial']
    first_half = next(s for s in slices if s['id']=='calendar_2026_h1')
    assert first_half == dict(id='calendar_2026_h1', start='2026-01-01', end='2026-06-30',
                              kind='calendar', partial=False)


def replay():
    return dict(nav=[dict(date=d,nav=n,exposure=n,cash=0,receivable=0) for d,n in
                     [('2026-01-05',90.),('2026-01-06',99.),('2026-01-07',108.9)]],
                fills=[dict(date='2026-01-05',symbol='A',side='BUY',qty=100,price=.9,fee=.1),
                       dict(date='2026-01-06',symbol='A',side='BUY',qty=100,price=.5,fee=.2)],
                closed_lots=[])


def test_carry_slice_uses_previous_nav_and_inherited_holdings():
    r = replay()
    benchmark = [dict(date=row['date'],nav=n) for row,n in zip(r['nav'],[80.,84.,92.4])]
    out = lab().attribute_slice(r,benchmark,100.,'2026-01-06','2026-01-07')
    assert out['opening_nav'] == 90
    assert out['pnl_vnd'] == pytest.approx(18.9)
    assert out['return_pct'] == pytest.approx(21.)
    assert out['index_return_pct'] == pytest.approx(15.5)
    assert out['net_excess_return_pp'] == pytest.approx(5.5)
    assert out['new_entries'] == 0 and out['additions'] == 1
    assert out['opening_positions'] == {'A':100}
    assert out['ending_positions'] == {'A':200}
    assert out['fees_vnd'] == .2
    assert out['max_drawdown_pct'] == 0
    assert out['profitable'] and out['beats_index']


def test_slice_includes_first_day_loss_and_losing_less_is_not_profit():
    r = replay()
    benchmark = [dict(date=row['date'],nav=80.) for row in r['nav']]
    out = lab().attribute_slice(r,benchmark,100.,'2026-01-05','2026-01-05')
    assert out['return_pct'] == pytest.approx(-10)
    assert out['max_drawdown_pct'] == pytest.approx(-10)
    assert out['beats_index'] and not out['profitable']
    assert not out['positive_net_and_excess']


def test_attribution_reconciles_returns_and_vnd_across_disjoint_slices():
    r = replay()
    out = [lab().attribute_slice(r,r['nav'],100.,row['date'],row['date']) for row in r['nav']]
    assert sum(x['pnl_vnd'] for x in out) == pytest.approx(8.9)
    assert np.prod([1+x['return_pct']/100 for x in out]) == pytest.approx(1.089)
    with pytest.raises(ValueError): lab().attribute_slice(r,[],100.,'2026-01-05','2026-01-07')


def test_protocol_rejects_lookahead_or_dropping_required_slice(tmp_path):
    protocol = lab().load_protocol()
    for section, key, value in [('regime','lag_sessions',0),('required_calendar_slice','end','2026-05-31')]:
        edited = copy.deepcopy(protocol)
        edited[section][key] = value
        path = tmp_path/'p.json'
        path.write_text(json.dumps(edited),encoding='utf-8')
        with pytest.raises(ValueError): lab().load_protocol(path)


def test_cash_reference_replay_is_separate_and_never_changes_source(rules):
    f = inputs()
    timeline = Timeline(dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
        member_count=2,initial_members=['AAA','BBB'],changes=[]))
    original = copy.deepcopy(f)
    result = lab().cash_reference(f,rules,timeline,{},'2025-01-02','2025-01-03','momentum/baseline','normal')
    assert result['metrics']['opening_nav'] == rules['backtest']['initial_capital']
    assert result['metrics']['opening_positions'] == {}
    assert result['metrics']['new_entries'] > 0
    assert result['reconciliation']['max_nav_error_vnd'] < 1e-5
    for symbol in f: pd.testing.assert_frame_equal(f[symbol],original[symbol])


def test_ci_cannot_skip_timeslices():
    from pathlib import Path
    workflow = Path('.github/workflows/research-gate.yml').read_text(encoding='utf-8')
    assert 'python -m scripts.research_timeslices' in workflow
    assert 'tests/test_research_timeslices.py' in workflow


def parents(tmp_path):
    from pathlib import Path
    from scripts import research_gate as gate, momentum_books as books
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    trial = dict(reconciliation=dict(max_nav_error_vnd=0.), replay=replay())
    blocks = {name: dict(mr={v:copy.deepcopy(trial) for v in gate.MR},
        momentum={v:copy.deepcopy(trial) for v in gate.MOMENTUM}, universe_equal=copy.deepcopy(trial),
        stresses={s:{v:copy.deepcopy(trial) for v in ('mr_bracket','mr_fixed15','momentum_baseline')}
                  for s in books.SCENARIOS[1:]}) for name in gate.load_registry()['blocks']}
    paths = ['scripts/research_gate.py','scripts/period_backtest.py','configs/rules_mr.json',
             'scripts/momentum_books.py','configs/research/research_gate_v1.json']
    common = dict(status='research_complete_live_blocked', manifest_sha256=gate.digest(manifest),
                  source_hashes={p:gate.digest(Path(p)) for p in paths})
    parent = dict(**copy.deepcopy(common), registry=gate.load_registry(), blocks=blocks)
    book = dict(**copy.deepcopy(common), protocol=books.load_protocol(), trial_count=180,
        blocks={name:dict(trials={s:{v:copy.deepcopy(trial) for v in books.VARIANTS}
                                for s in books.SCENARIOS}) for name in blocks})
    return parent, book, manifest


def test_all_44_parent_trials_retained_without_duplicate_book_baseline(tmp_path):
    a,b,m = parents(tmp_path)
    trials = lab().verify_parents(a,b,m)
    assert len(trials) == 44
    assert 'mr/no_confirmation/normal' in trials
    assert 'books/quality/double_cost' in trials
    assert 'momentum/universe_equal/normal' in trials
    assert 'books/baseline/normal' not in trials


@pytest.mark.parametrize('issue', ['registry','status','manifest','source','source_missing','book_block',
                                 'book_count','book_variant','book_scenario'])
def test_parent_missing_or_drifted_evidence_is_blocked(tmp_path,issue):
    a,b,m = parents(tmp_path)
    if issue == 'registry': a['registry']['seed'] = 1
    if issue == 'status': a['status'] = 'running'
    if issue == 'manifest': b['manifest_sha256'] = 'tampered'
    if issue == 'source': a['source_hashes']['scripts/period_backtest.py'] = 'tampered'
    if issue == 'source_missing': b['source_hashes'] = {}
    if issue == 'book_block': del b['blocks']['panic_2022']
    if issue == 'book_count': b['trial_count'] = 179
    if issue == 'book_variant': del b['blocks']['continuous']['trials']['normal']['weekly']
    if issue == 'book_scenario': del b['blocks']['continuous']['trials']['double_cost']
    with pytest.raises(ValueError): lab().verify_parents(a,b,m)


@pytest.mark.parametrize('strategy,scenario', [('typo/baseline','normal'),('mr/bracket','typo')])
def test_cash_runner_rejects_unregistered_input(rules,strategy,scenario):
    with pytest.raises(ValueError): lab().cash_reference({},rules,None,{},'a','b',strategy,scenario)


def test_html_displays_both_starting_modes_and_escapes_untrusted_labels():
    m = lab().attribute_slice(replay(),replay()['nav'],100.,'2026-01-05','2026-01-07')
    block = dict(id='test',kind='regime',regime='<script>alert(1)</script>',start='2026-01-05',end='2026-01-07',
        sessions=3,regime_sessions=dict(uptrend=0,downtrend=3,transition=0),
        carry={s+'/normal':m for s in lab().REFERENCES},
        cash_restart={s+'/normal':dict(metrics=m) for s in lab().REFERENCES})
    report = lab().render_report(dict(slices=[block],protocol=lab().load_protocol()))
    assert '<script>' not in report and '&lt;script&gt;' in report
    assert 'Danh mục liên tục' in report and 'tiền mặt' in report
    assert 'results.json' in report and 'regime_labels.json' in report


def test_cli_end_to_end_on_synthetic_market_preserves_both_modes(tmp_path,monkeypatch,rules):
    """Exercise CLI, real replay/accounting and exports; only data provenance is stubbed."""
    from scripts import research_gate as gate
    module = lab()
    f = inputs()
    from datetime import date
    from stock_agent.data.exchange_calendar import trading_days_between
    days = [d.isoformat() for d in trading_days_between(date(2023,11,1),date(2025,12,31))]
    for frame in f.values(): frame['date'] = days[:len(frame)]
    timeline_data = dict(coverage_start='2024-01-01',coverage_end='2025-12-31',initial_known_on='2024-01-01',
                         member_count=2,initial_members=['AAA','BBB'],changes=[])
    timeline_path = tmp_path/'timeline.json'
    timeline_path.write_text(json.dumps(timeline_data),encoding='utf-8')
    monkeypatch.setattr(gate,'TIMELINE',timeline_path)
    start,end = '2025-01-02','2025-01-10'
    protocol = module.load_protocol()
    protocol.update(start=start,end=end)
    monkeypatch.setattr(module,'load_protocol',lambda:protocol)
    manifest = tmp_path/'manifest.json'
    files = {}
    for symbol,frame in f.items():
        frame.to_csv(tmp_path/(symbol+'.csv'),index=False)
        files[symbol] = dict(path=symbol+'.csv')
    manifest.write_text('{}')
    monkeypatch.setattr(module,'verify_snapshot',lambda _:dict(files=files))
    trial = gate.run_one(f,rules,Timeline(timeline_data),{},start,end,'momentum','baseline')
    parent_trials = {s+'/normal':copy.deepcopy(trial) for s in module.REFERENCES}
    monkeypatch.setattr(module,'verify_parents',lambda *args:parent_trials)
    monkeypatch.setattr(gate,'mr_signal_bank',lambda *args:{'bracket':{},'fixed15':{}})
    parent_path,book_path = tmp_path/'parent.json',tmp_path/'books.json'
    for path in (parent_path,book_path): path.write_text('{}')
    output = tmp_path/'result'
    module.main(['--manifest',str(manifest),'--gate-results',str(parent_path),
                 '--book-results',str(book_path),'--output',str(output)])
    result = json.loads((output/'results.json').read_text())
    assert result['status'] == 'timeslices_complete_not_live_profit'
    assert result['partition_reconciled']
    assert result['cash_trial_count'] == 9*len(result['slices'])
    assert set(b['kind'] for b in result['slices']) == {'calendar','regime'}
    assert (output/'report.html').exists() and (output/'regime_labels.json').exists()
    with pytest.raises(FileExistsError): module.run_suite(manifest,parent_path,book_path,output)

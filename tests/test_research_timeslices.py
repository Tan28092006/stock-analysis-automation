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

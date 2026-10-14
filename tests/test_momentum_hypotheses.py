"""Research-only ablation contracts, including non-vacuous causality checks."""
import copy
import importlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import period_backtest as bt


def research():
    return importlib.import_module('scripts.momentum_hypotheses')


@pytest.fixture
def rules():
    return json.loads(Path('configs/rules_mr.json').read_text(encoding='utf-8'))


def frame(log_returns, dates=None):
    close = 100 * np.exp(np.r_[0, np.cumsum(log_returns)])
    dates = dates or pd.bdate_range('2024-01-01', periods=len(close)).strftime('%Y-%m-%d').tolist()
    return pd.DataFrame(dict(date=dates, open=close, high=close * 1.02,
                             low=close * .98, close=close, volume=1e6))


def inputs():
    rng = np.random.default_rng(7)
    market = rng.normal(.001, .01, 300)
    noise = np.sin(np.arange(300)) * .003
    return {'VNINDEX': frame(market), 'AAA': frame(2 * market + .0002 + noise),
            'BBB': frame(.3 * market + .001 + noise)}


def short_inputs():
    dates = ['2026-01-28', '2026-01-29', '2026-01-30', '2026-02-02',
             '2026-02-03', '2026-02-04', '2026-02-05', '2026-02-06']
    return {s: frame(np.zeros(7), dates) for s in ('AAA', 'VNINDEX')}


def test_market_adjusted_score_matches_independent_formula_and_not_raw_rank():
    rh = research()
    data = inputs()
    ranked = rh.market_adjusted_rank(data)
    assert ranked[0][0] == 'BBB'
    x = np.diff(np.log(data['VNINDEX']['close']))[-252:]
    y = np.diff(np.log(data['BBB']['close']))[-252:]
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    residual = (y - beta * x)[:-21]
    expected = residual.sum() / max(residual.std(ddof=1), 1e-12)
    assert ranked[0][1] == pytest.approx(expected)


def test_market_adjusted_nonconstant_score_and_history_gate():
    rh = research()
    data = inputs()
    data['AAA']['close'] *= np.exp(np.sin(np.arange(301)) * .01)
    data['NEW'] = data['AAA'].tail(100)
    ranked = {r[0]: r for r in rh.market_adjusted_rank(data)}
    x = np.diff(np.log(data['VNINDEX']['close']))[-252:]
    y = np.diff(np.log(data['AAA']['close']))[-252:]
    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    residual = (y - beta * x)[:-21]
    assert ranked['AAA'][1] == pytest.approx(residual.sum() / residual.std(ddof=1))
    assert 'NEW' not in ranked


def test_own_portfolio_vol_includes_covariance_and_caps_exposure():
    rh = research()
    data = inputs()
    data['BBB'] = data['AAA'].copy()
    weights = {'AAA': .5, 'BBB': .5}
    actual = rh.portfolio_volatility(data, weights)
    expected = data['AAA']['close'].pct_change().tail(126).std() * np.sqrt(252)
    assert actual == pytest.approx(expected)
    selected, _ = rh.variant_targets(data, set(), 'own_portfolio_vol')
    assert sum(selected.values()) == pytest.approx(min(1, .2 / expected))
    with pytest.raises(ValueError, match='126'):
        rh.portfolio_volatility({s: f.tail(100) for s, f in data.items()}, weights)


def test_reversal_gate_uses_relative_five_day_move():
    rh = research()
    data = {'VNINDEX': frame(np.full(10, .01)), 'AAA': frame(np.full(10, .005))}
    assert rh.reversal_entry_allowed('AAA', data)
    data['AAA'] = frame(np.full(10, .02))
    assert not rh.reversal_entry_allowed('AAA', data)
    assert not rh.reversal_entry_allowed('AAA', {'AAA': data['AAA'].tail(5), 'VNINDEX': data['VNINDEX']})


def test_delay_freezes_signal_and_quantities_and_default_is_unchanged(monkeypatch, rules):
    data = short_inputs()
    seen = []

    def target(prefixes, held):
        seen.append(prefixes['AAA']['date'].iloc[-1])
        return {'AAA': .5}, []

    monkeypatch.setattr(bt, 'momentum_targets', target)
    old = bt.replay_momentum(data, rules, '2026-02-01', '2026-02-06')
    same = bt.replay_momentum(data, rules, '2026-02-01', '2026-02-06', target_fn=target)
    assert old == same
    delayed = bt.replay_momentum(data, rules, '2026-02-01', '2026-02-06',
                                 target_fn=target, execution_delay=1)
    assert delayed['fills'][0]['date'] == '2026-02-03'
    assert seen == ['2026-01-30'] * 3
    assert delayed['rebalances'][0]['targets'] == old['rebalances'][0]['targets']
    assert delayed['rebalances'][0]['signal_date'] == '2026-01-30'
    with pytest.raises(ValueError):
        bt.replay_momentum(data, rules, '2026-02-01', '2026-02-06', execution_delay=-1)


def test_entry_filter_only_sees_prior_data_and_does_not_reallocate_blocked_cash(rules):
    data = short_inputs()
    data['BBB'] = data['AAA'].copy()
    seen = []

    def gate(symbol, prefixes):
        seen.append(prefixes[symbol]['date'].iloc[-1])
        return symbol == 'BBB'

    result = bt.replay_momentum(data, rules, '2026-02-01', '2026-02-06',
                                target_fn=lambda f, h: ({'AAA': .5, 'BBB': .5}, []), entry_gate=gate)
    assert {f['symbol'] for f in result['fills']} == {'BBB'}
    assert result['fills'][0]['date'] == '2026-02-02'
    assert result['nav'][-1]['exposure'] / result['nav'][-1]['nav'] < .51
    assert max(seen) <= '2026-02-04'
    assert result['skipped_entries']


@pytest.mark.parametrize('variant', ['baseline', 'market_adjusted', 'reversal_entry', 'own_portfolio_vol'])
def test_all_variants_are_future_invariant_with_real_fills(variant, rules):
    rh = research()
    data = inputs()
    start, end = data['VNINDEX'].iloc[275]['date'], data['VNINDEX'].iloc[290]['date']
    first = rh.run_variant(data, rules, start, end, variant, 'normal')
    assert first['fills']
    altered = copy.deepcopy(data)
    for f in altered.values():
        f.loc[f['date'] > end, ['open', 'high', 'low', 'close']] *= 9
    second = rh.run_variant(altered, rules, start, end, variant, 'normal')
    assert first == second
    assert rh.reconcile(first, data, 1e9)['max_nav_error_vnd'] < 1e-5


def test_cost_stress_is_isolated_and_invalid_variant_fails(rules):
    rh = research()
    before = copy.deepcopy(rules)
    data = inputs()
    start, end = data['VNINDEX'].iloc[275]['date'], data['VNINDEX'].iloc[290]['date']
    normal = rh.run_variant(data, rules, start, end, 'baseline', 'normal')
    stress = rh.run_variant(data, rules, start, end, 'baseline', 'double_cost')
    assert stress['metrics']['return_pct'] < normal['metrics']['return_pct']
    assert rules == before
    with pytest.raises(ValueError):
        rh.run_variant(data, rules, start, end, 'oops', 'normal')
    with pytest.raises(ValueError):
        rh.run_variant(data, rules, start, end, 'baseline', 'oops')


def test_statistics_and_bootstrap_are_paired_reproducible_and_zero_safe():
    rh = research()
    control = np.array([.01, -.01, .002, -.003] * 10)
    test = control + .001
    stats = rh.paired_interval(test, control)
    assert stats == rh.paired_interval(test, control)
    assert stats['annualized_mean_difference_pct'] == pytest.approx(25.2)
    assert stats['ci95_pct'] == pytest.approx([25.2, 25.2])
    assert stats['ci_three_variants_pct'] == pytest.approx([25.2, 25.2])
    assert rh.paired_interval(control, control)['ci95_pct'] == [0, 0]
    with pytest.raises(ValueError):
        rh.paired_interval(test, control[:-1])


def test_report_metrics_and_reconciliation_detect_tampering(rules):
    rh = research()
    data = inputs()
    start, end = data['VNINDEX'].iloc[275]['date'], data['VNINDEX'].iloc[290]['date']
    result = rh.run_variant(data, rules, start, end, 'baseline', 'normal')
    benchmark = rh.benchmark_nav(data, start, end, 1e9)
    stats = rh.summarize(result, benchmark, 1e9, rules['backtest']['slippage_pct'] / 100)
    assert stats['net_excess_return_pp'] == pytest.approx(
        result['metrics']['return_pct'] - bt.metrics(benchmark, [], 1e9)['return_pct'])
    assert stats['two_way_turnover_multiple'] > 0
    assert stats['fees_vnd'] == sum(f['fee'] for f in result['fills'])
    result['nav'][0]['nav'] += 10
    with pytest.raises(ValueError, match='reconcil'):
        rh.reconcile(result, data, 1e9)


def test_experiment_cli_writes_trial_registry_and_refuses_overwrite(tmp_path, monkeypatch):
    rh = research()
    dates = [d.isoformat() for d in bt.trading_days_between(date(2024, 1, 1), date(2026, 9, 25))]
    rng = np.random.default_rng(19)
    data = frame(rng.normal(.0005, .01, len(dates) - 1), dates)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    files = {}
    for symbol in ('AAA', 'VNINDEX'):
        data.to_csv(tmp_path / f'{symbol}.csv', index=False)
        files[symbol] = {'path': f'{symbol}.csv'}
    monkeypatch.setattr(rh, 'verify_snapshot', lambda _: {'files': files, 'as_of': '2026-09-25'})
    output = tmp_path / 'experiments.json'
    monkeypatch.setattr(sys, 'argv', ['hypotheses', '--historical-manifest', str(manifest),
                                    '--recent-manifest', str(manifest), '--output', str(output)])
    rh.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['trial_registry']['total_replays'] == 36
    assert len(report['blocks']) == 3
    assert report['live_promotion'] == 'blocked'
    assert report['protocol_sha256']
    assert all(v['reconciliation']['max_nav_error_vnd'] < 1e-5
               for block in report['blocks'].values() for scenario in block['scenarios'].values()
               for v in scenario.values())
    with pytest.raises(FileExistsError):
        rh.main()

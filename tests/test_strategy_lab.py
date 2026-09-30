"""Reusable research backend: independent portfolios, PIT membership, audit trail."""
import copy
import importlib
import json
from pathlib import Path

import pytest

from scripts import historical_universe as hu, period_backtest as bt
from tests.test_market_runtime_hardening import make_snapshot


def lab():
    return importlib.import_module('scripts.strategy_lab')


def request():
    return dict(start='2026-09-01', end='2026-09-21', capital=10000000,
                mr_profiles=['vn30', 'strict'], momentum_variants=['baseline'],
                scenarios=['normal', 'double_cost'], mr_overrides={})


@pytest.fixture
def sources(tmp_path):
    manifest = make_snapshot(tmp_path / 'snapshot')
    timeline = tmp_path / 'timeline.json'
    timeline.write_text(json.dumps(dict(coverage_start='2025-02-03', coverage_end='2026-09-21',
        initial_known_on='2025-02-03', member_count=1, initial_members=['AAA'], changes=[])))
    return manifest, timeline, Path('configs/rules_mr.json')


@pytest.mark.parametrize('key,value', [('capital', 0), ('capital', True), ('capital', float('nan')),
    ('end', '2026-08-01'), ('start', '20260901'), ('mr_profiles', ['oops']),
    ('momentum_variants', ['oops']), ('scenarios', ['delay_one_session']),
    ('mr_profiles', ['vn30', 'vn30']), ('mr_overrides', {'typo': 3}),
    ('mr_overrides', {'rsi_max': 101}), ('mr_overrides', {'max_hold_days': 1.5}),
    ('unexpected', 1)])
def test_invalid_request_rejected(key, value):
    config = request()
    config[key] = value
    with pytest.raises(ValueError):
        lab().validate_request(config)


def test_pit_mr_excludes_future_member_and_keeps_exit_plan():
    t = hu.Timeline(dict(coverage_start='2026-01-01', coverage_end='2026-02-06',
        initial_known_on='2026-01-01', member_count=1, initial_members=['AAA'],
        changes=[dict(known_on='2026-01-29', effective_on='2026-02-03', add=['BBB'], remove=['AAA'])]))
    plan = dict(stop=90., target=110., hold=15, rr=2.)
    raw = {'2026-01-30': [dict(symbol=s, **plan) for s in ('AAA', 'BBB')],
           '2026-02-02': [dict(symbol=s, **plan) for s in ('AAA', 'BBB')]}
    kept, rejected = lab().eligible_mr_signals(raw, t, '2026-02-02', '2026-02-06')
    assert [p['symbol'] for p in kept['2026-01-30']] == ['AAA']
    assert [p['symbol'] for p in kept['2026-02-02']] == ['BBB']
    assert kept['2026-01-30'][0]['stop'] == 90.
    assert len(rejected) == 2
    assert raw['2026-01-30'][1]['symbol'] == 'BBB'


def test_matrix_independent_capital_rules_and_original_momentum_parity(sources):
    import pandas as pd
    module = lab()
    manifest, timeline, rules_path = sources
    before = rules_path.read_bytes()
    result = module.evaluate(request(), manifest, timeline, rules_path)
    assert len(result['runs']) == 6
    assert result['live_approved'] is False
    assert result['trial_count'] == 6
    assert result['source_hashes']
    for run in result['runs'].values():
        assert run['rules']['backtest']['initial_capital'] == request()['capital']
        assert run['reconciliation']['max_nav_error_vnd'] < 1e-5
        assert run['summary']['sample_warning']
    assert result['runs']['mr:strict:normal']['rules']['vn30_mean_reversion'] == {}
    assert result['runs']['mr:vn30:normal']['rules']['vn30_mean_reversion']['rsi_max'] == 35
    assert rules_path.read_bytes() == before
    meta = json.loads(manifest.read_text())
    frames = {s: pd.read_csv(manifest.parent / i['path']) for s, i in meta['files'].items()}
    rules = json.loads(before)
    rules['backtest']['initial_capital'] = request()['capital']
    expected = hu.replay(frames, rules, request()['start'], request()['end'], 'baseline', 'normal',
                         'historical', hu.Timeline(json.loads(timeline.read_text())))
    assert result['runs']['momentum:baseline:normal']['replay'] == expected


def test_artifact_records_request_before_failure_and_refuses_reuse(sources, tmp_path, monkeypatch):
    module = lab()
    monkeypatch.setattr(module, 'evaluate', lambda *a: (_ for _ in ()).throw(ValueError('blocked source')))
    folder = tmp_path / 'runs' / 'trial'
    with pytest.raises(ValueError, match='blocked source'):
        module.run(request(), *sources, folder)
    assert json.loads((folder / 'request.json').read_text())['request'] == request()
    assert json.loads((folder / 'failed.json').read_text())['status'] == 'failed'
    with pytest.raises(FileExistsError):
        module.run(request(), *sources, folder)


def test_cli_end_to_end_and_snapshot_tampering(sources, tmp_path):
    module = lab()
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(request()))
    folder = tmp_path / 'run'
    module.main(['--spec', str(spec), '--manifest', str(sources[0]), '--timeline', str(sources[1]),
                 '--output', str(folder)])
    report = json.loads((folder / 'result.json').read_text())
    assert report['status'] == 'completed_research'
    assert (folder / 'summary.md').exists()
    price = sources[0].parent / 'prices/AAA.csv'
    price.write_text(price.read_text().replace('10000', '10001'))
    # Ensure tamper independent of price representation.
    with price.open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError):
        module.evaluate(request(), *sources)


def test_source_coverage_and_ml_are_fail_closed(sources, tmp_path):
    module = lab()
    config = request()
    config['end'] = '2026-09-25'
    with pytest.raises(ValueError, match='coverage'):
        module.evaluate(config, *sources)
    rules = json.loads(sources[2].read_text())
    rules['ml']['enabled'] = True
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps(rules))
    with pytest.raises(ValueError, match='ML'):
        module.evaluate(request(), sources[0], sources[1], path)

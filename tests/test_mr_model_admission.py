"""Dashboard model admission is independent of a good-looking historical score."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_agent.features import mr_scan as mr
from stock_agent.features import win_probability as wp


@pytest.fixture
def scan_env(monkeypatch):
    rules = copy.deepcopy(mr._load_rules())
    rules['ml'] = {'enabled': True, 'override_enabled': True}
    monkeypatch.setattr(mr, '_load_rules', lambda: rules)
    data = pd.DataFrame(dict(date=['2026-09-25'], close=[100.], rsi14=[25.], bb_lower=[99.]))
    monkeypatch.setattr(mr, '_load_frames', lambda _: {'AAA': data})
    monkeypatch.setattr(mr, '_market_state', lambda *args: {'date': '2026-09-25'})
    monkeypatch.setattr(mr, 'prepare_signal_frame', lambda f, r: f)
    monkeypatch.setattr(mr, 'score_precomputed_at', lambda *a: SimpleNamespace(
        decision='WATCH', risk_plan=None, evidence=[], latest_close=100.))
    monkeypatch.setattr(wp, '_context', lambda p: ({}, {}))
    monkeypatch.setattr(wp, '_breadth_map', lambda f: {})
    monkeypatch.setattr(wp, 'feature_row', lambda *a: {'test': 1})
    monkeypatch.setattr(wp, 'is_candidate', lambda f: True)
    return rules


def artifact(approval=True, temporal=True):
    return SimpleNamespace(meta={'release': {'live_approved': approval, 'paper_eligible': True},
                                 'model_version': 'isolated-test-model'},
                           available_at=lambda day: temporal, predict=lambda f: .9)


@pytest.mark.parametrize('enabled', [False, None, 'true', 1])
def test_disabled_or_malformed_flag_never_loads_artifact(scan_env, monkeypatch, enabled):
    scan_env['ml']['enabled'] = enabled
    monkeypatch.setattr(wp.WinProbModel, 'load', lambda: pytest.fail('disabled ML artifact loaded'))
    result = mr._compute(0, .55, include_positions=False)
    assert result['model']['available'] is False
    assert result['prob_buys'] == []
    assert result['watches'][0]['win_prob'] is None


@pytest.mark.parametrize('approval', [False, None, 'true', 1])
def test_paper_candidate_cannot_emit_probability_buys(scan_env, monkeypatch, approval):
    monkeypatch.setattr(wp.WinProbModel, 'load', lambda: artifact(approval))
    result = mr._compute(0, .55, include_positions=False)
    assert result['model']['available'] is False
    assert result['prob_buys'] == []
    assert result['watches'][0]['win_prob'] is None


def test_enabled_approved_model_still_needs_override_flag_and_asof(scan_env, monkeypatch):
    model = artifact()
    monkeypatch.setattr(wp.WinProbModel, 'load', lambda: model)
    result = mr._compute(0, .55, include_positions=False)
    assert result['model']['available'] is True
    assert result['prob_buys'][0]['win_prob'] == .9
    scan_env['ml']['override_enabled'] = False
    result = mr._compute(0, .55, include_positions=False)
    assert result['prob_buys'] == []
    assert result['watches'][0]['win_prob'] == .9
    model.available_at = lambda day: False
    result = mr._compute(0, .55, include_positions=False)
    assert result['model']['available'] is False
    assert result['watches'][0]['win_prob'] is None


def test_corrupt_artifact_fails_closed_without_losing_rule_scan(scan_env, monkeypatch):
    def bad():
        raise ValueError('invalid artifact')
    monkeypatch.setattr(wp.WinProbModel, 'load', bad)
    result = mr._compute(0, .55, include_positions=False)
    assert result['watches'] and not result['prob_buys']
    assert result['model']['admission'] == 'load_failed'


def test_old_cached_probability_payload_is_not_reused(scan_env, monkeypatch, tmp_path):
    from stock_agent.features import scan_guard
    monkeypatch.setattr(scan_guard, 'readiness', lambda p: {'data_ready': True})
    path = tmp_path / 'cache.json'
    monkeypatch.setattr(mr, 'CACHE_PATH', path)
    monkeypatch.setattr(mr, 'PRICES_DIR', tmp_path)
    monkeypatch.setattr(mr, 'scan_input_snapshot', lambda p: 'same-data')
    path.write_text(json.dumps(dict(rules_hash=mr.compute_rules_hash(scan_env), input_snapshot='same-data',
                                   data_date=None, min_win_prob=.55, prob_buys=[{'symbol': 'UNAPPROVED'}])), encoding='utf-8')
    monkeypatch.setattr(mr, '_compute', lambda *a: {'prob_buys': [], 'data_date': None})
    assert mr.mr_scan()['prob_buys'] == []


def test_explicit_paper_compute_never_loads_model(scan_env, monkeypatch):
    monkeypatch.setattr(wp.WinProbModel, 'load', lambda: pytest.fail('paper model loaded'))
    assert not mr._compute(0, .55, use_model=False, include_positions=False)['model']['available']

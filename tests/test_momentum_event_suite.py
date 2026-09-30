"""Research CLI, complete experiment inventory and fail-closed evidence."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_momentum_events import market, rally


def lab():
    return importlib.import_module('scripts.momentum_event_suite')


def test_exact_trial_inventory_and_required_windows():
    names = lab().trial_names()
    assert len(names) == len(set(names)) == 18
    assert 'pyramiding55/double_cost' in names
    assert 'daily20/delay_one_session' in names
    fixture = Path('data/paper/research_gate/20260930/snapshot_20210824/manifest.json')
    # Source-based inventory does not depend on local private data existing in CI.
    assert lab().PARENT_REPLAYS == 1196


def test_panel_retains_all_trials_and_exact_ledgers(market):
    frames, rules, timeline, i = market
    rally(market)
    days = frames['VNINDEX'].date
    windows = {'continuous': dict(start=days[i], end=days[i + 14], kind='primary'),
               'slice': dict(start=days[i + 5], end=days[i + 9], kind='regime')}
    panel = lab().evaluate_panel(frames, rules, timeline, windows, 'continuous')
    assert len(panel['blocks']) == 2
    for block in panel['blocks'].values():
        assert len(block['trials']) == 18
        for trial in block['trials'].values():
            assert trial['reconciliation']['max_nav_error_vnd'] < 1e-5
            assert trial['activity']['numerical_activity_pass'] is None
            assert trial['summary']['cash_return_pct'] == 0
    carried = panel['carry']['slice']['daily55/normal']
    assert carried['opening_positions']
    assert panel['blocks']['slice']['trials']['daily55/normal']['summary']['opening_positions'] == {}
    assert panel['replay_count'] == 36


def test_bootstrap_family_and_short_sample_cannot_promote():
    result = lab().inference([.001] * 119, [0.] * 119)
    assert result['status'] == 'insufficient_sessions'
    assert all(t['family'] == 184 for t in result['tests'])
    assert lab().inference([.001], [0.])['tests'] == []


def test_cli_requires_both_snapshots_and_parent_results():
    result = subprocess.run([sys.executable, '-m', 'scripts.momentum_event_suite'], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--h1-manifest' in result.stderr and '--gate-results' in result.stderr


def test_protocol_drift_and_absent_manifest_fail_before_output(tmp_path):
    m = lab()
    out = tmp_path / 'output'
    with pytest.raises((ValueError, FileNotFoundError)):
        m.run_suite(tmp_path/'missing.json', tmp_path/'missing-h1.json', tmp_path/'parent.json', out)
    assert not out.exists()


def test_ci_keeps_all_previous_jobs_and_adds_event_contracts():
    lab()
    workflow = Path('.github/workflows/research-gate.yml').read_text()
    assert 'test_momentum_event_suite.py' in workflow and 'test_momentum_events.py' in workflow
    for script in ['research_gate', 'momentum_books', 'research_timeslices', 'universe_expansion', 'momentum_event_suite']:
        assert f'python -m scripts.{script}' in workflow


def test_parent_missing_source_cannot_claim_regression_match(tmp_path):
    m = lab()
    parent = dict(status='research_complete_live_blocked', manifest_sha256='x', source_hashes={})
    path = tmp_path / 'parent.json'
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match='source'):
        m.verify_parent(path, 'x')

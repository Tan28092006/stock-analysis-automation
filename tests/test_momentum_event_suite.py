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


def test_complete_synthetic_suite_accepts_real_parent_schemas_without_overwrite(tmp_path, monkeypatch, market):
    m = lab()
    frames, rules, timeline, i = market
    rally(market)
    files = {}
    for symbol, frame in frames.items():
        frame.to_csv(tmp_path / f'{symbol}.csv', index=False)
        files[symbol] = {'path': f'{symbol}.csv'}
    manifest, h1 = tmp_path/'manifest.json', tmp_path/'h1.json'
    manifest.write_text('{}')
    h1.write_text('{}')
    timeline_path = tmp_path / 'timeline.json'
    timeline_path.write_text(json.dumps(timeline.data))
    monkeypatch.setattr(m.gate, 'TIMELINE', timeline_path)
    monkeypatch.setattr(m, 'verify_snapshot', lambda path: {'files': files})
    real_digest = m.gate.digest
    p = m.events.load_protocol()
    monkeypatch.setattr(m.gate, 'digest', lambda path: p['snapshots']['VN30'] if Path(path) == manifest else
        p['snapshots']['H1'] if Path(path) == h1 else real_digest(path))
    monkeypatch.setattr(m.universe, 'load_inputs', lambda *args: (frames, {'VN30': timeline, 'VN100': timeline}, rules))
    bounds = dict(start=frames['VNINDEX'].date[i], end=frames['VNINDEX'].date[i + 6], kind='calendar')
    monkeypatch.setattr(m, 'windows', lambda index, full: {'continuous' if full else 'h1_2026': bounds})
    sources = list(Path('scripts').glob('*.py')) + list(Path('stock_agent').rglob('*.py')) + [m.gate.RULES, m.gate.REGISTRY]
    hashes = {str(path): real_digest(path) for path in sources}
    parent = dict(status='research_complete_live_blocked', manifest_sha256=p['snapshots']['VN30'],
        source_hashes=hashes, blocks={'continuous': {'momentum': {'baseline': {'summary': {'return_pct': 0.}}}}})
    old_trial = {'metrics': {'return_pct': -1.}}
    old_universes = {u: {'cash_restart': {'momentum/baseline/normal': old_trial}} for u in ['VN30', 'VN100']}
    up = dict(status='universe_research_complete_full_regression_required', manifest_sha256=p['snapshots']['H1'],
        source_hashes=hashes, blocks=[dict(id='calendar_2026_h1', universes=old_universes)])
    parent_path, up_path = tmp_path/'parent.json', tmp_path/'universe.json'
    parent_path.write_text(json.dumps(parent))
    up_path.write_text(json.dumps(up))
    protected = {str(f): real_digest(f) for f in tmp_path.iterdir()}
    result = m.run_suite(manifest, h1, parent_path, tmp_path/'out', up_path)
    assert result['replay_count'] == 54 and result['live_eligible'] is False
    assert result['panels']['H1_VN100']['prior_monthly_control'] == old_trial['metrics']
    assert len(result['paired_universe_tests']) == 18
    assert json.loads((tmp_path/'out/results.json').read_text())['status'] == 'research_complete_live_blocked'
    assert all(real_digest(f) == h for f, h in protected.items())
    with pytest.raises(FileExistsError):
        m.run_suite(manifest, h1, parent_path, tmp_path/'out', up_path)

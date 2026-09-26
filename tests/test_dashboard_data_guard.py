"""Incomplete/stale prices must not become actionable recommendations or caches."""
import importlib
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from tests.test_paper_runner import prices


def guard():
    return importlib.import_module('stock_agent.features.scan_guard')


def test_guard_reports_stale_missing_and_wrong_universe(tmp_path, monkeypatch):
    from stock_agent import config
    monkeypatch.setattr(config, 'load_universe', lambda: {'symbols': ['AAA', 'BBB']})
    prices(tmp_path, 'VNINDEX')
    prices(tmp_path, 'AAA')
    now = datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
    status = guard().readiness(tmp_path, now=now)
    assert status['data_ready'] is False
    assert 'AAA' in status['issues'] and 'BBB' in status['issues']
    prices(tmp_path, 'AAA', end=date(2026, 9, 25))
    prices(tmp_path, 'BBB', end=date(2026, 9, 25))
    prices(tmp_path, 'VNINDEX', end=date(2026, 9, 25))
    assert guard().readiness(tmp_path, now=now)['data_ready'] is True
    prices(tmp_path, 'OUTSIDE', end=date(2026, 9, 25))
    assert 'universe' in guard().readiness(tmp_path, now=now)['issues']


@pytest.mark.parametrize('name', ['mr_scan', 'momentum_scan', 'swing_scan'])
@pytest.mark.parametrize('force', [False, True])
def test_bad_prices_block_before_cache_compute_and_positions(name, force, tmp_path, monkeypatch):
    module = importlib.import_module(f'stock_agent.features.{name}')
    from stock_agent.features import position_manager as pos
    monkeypatch.setattr(module, 'PRICES_DIR', tmp_path)
    cache = tmp_path / 'cache.json'
    old = json.dumps({'picks': [{'symbol': 'STALE'}], 'buys': [{'symbol': 'STALE'}]})
    cache.write_text(old, encoding='utf-8')
    monkeypatch.setattr(module, 'CACHE_PATH', cache)
    monkeypatch.setattr(module, '_compute', lambda *a: pytest.fail('unsafe scan executed'))
    monkeypatch.setattr(pos, 'PositionStore', lambda: pytest.fail('position store accessed'))
    # No files: a real fail-closed check, not a stubbed readiness result.
    result = getattr(module, name)(force=force)
    assert result['status'] == 'blocked_data'
    assert result['live_approved'] is False
    assert not any(result[k] for k in ('picks', 'buys', 'prob_buys', 'sell_alerts'))
    assert result['readiness']['data_ready'] is False
    assert cache.read_text(encoding='utf-8') == old


def test_no_universe_abstains_instead_of_falling_back(tmp_path, monkeypatch):
    from stock_agent import config
    monkeypatch.setattr(config, 'load_universe', lambda: {'symbols': []})
    assert guard().readiness(tmp_path)['data_ready'] is False


def test_blocked_payload_never_calls_missing_signal_a_market_opinion():
    result = guard().blocked({'data_ready': False, 'session': '2026-09-25', 'issues': {'AAA': ['stale']}}, 'mr')
    assert result['market']['state'] == 'UNKNOWN'
    assert result['data_date'] is None  # expected date is not observed data
    assert result['model']['available'] is False
    assert result['active'] is False
    assert result['position_checks_available'] is False

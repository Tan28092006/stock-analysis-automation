import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import research_gate as gate
from scripts.historical_universe import Timeline
from tests.test_period_backtest import prices, rules, order


def test_registry_covers_panic_and_controls_without_claiming_holdout():
    registry = gate.load_registry()
    assert {'continuous', 'recent_6m', 'panic_2022', 'war_2026', 'tariffs_2025'} <= registry['blocks'].keys()
    assert registry['holdout'] is False
    assert len(registry['momentum']) >= 10
    assert len(registry['mr']) >= 12


def test_missing_registered_trial_is_blocked():
    registry = gate.load_registry()
    with pytest.raises(ValueError, match='Missing'):
        gate.validate_evidence({'blocks': {}}, registry)


def test_bootstrap_family_widens_intervals_and_is_deterministic():
    delta = np.random.default_rng(1).normal(.0001, .01, 600)
    one = gate.paired_test(delta, family=1, block=20)
    many = gate.paired_test(delta, family=30, block=20)
    assert many == gate.paired_test(delta, family=30, block=20)
    assert many['ci_family'][0] <= one['ci_family'][0]
    assert many['ci_family'][1] >= one['ci_family'][1]
    assert many['observations'] == 600


@pytest.mark.parametrize('values', [[1, 2], [0.] * 29 + [float('nan')]])
def test_invalid_statistics_fail_closed(values):
    with pytest.raises(ValueError):
        gate.paired_test(np.array(values), family=10, block=20)


def test_exposure_control_uses_previous_close_exposure():
    nav = [dict(date='a', nav=100, exposure=100), dict(date='b', nav=110, exposure=0)]
    assert gate.exposure_control(nav, np.array([.5, .1]), 100).tolist() == [0., .1]


def test_daily_holdings_reconcile_and_do_not_repeat_sales(rules):
    from scripts import period_backtest as bt
    result = bt.replay_mr({'AAA': prices(), 'VNINDEX': prices()}, rules,
                         '2026-01-01', '2026-01-12', {'2025-12-31': [order(hold=4)]}, exit_policy='fixed_hold')
    history = gate.position_history(result, {'AAA': prices()})
    assert history[0]['positions']['AAA']['qty'] > 0
    assert history[-1]['positions'] == {}
    assert all(abs(sum(p['value'] for p in r['positions'].values()) - r['exposure']) < 1e-5 for r in history)


def test_historical_membership_contains_removed_crash_names():
    timeline = Timeline(json.loads(Path(gate.TIMELINE).read_text(encoding='utf-8')))
    assert {'NVL', 'PDR', 'POW', 'KDH'} <= timeline.members('2022-08-31', '2022-09-05')
    assert not {'NVL', 'PDR'} & timeline.members('2023-08-04', '2023-08-07')
    assert timeline.members('2026-09-28', '2026-09-29') == set(json.loads(Path('configs/universe_vn30.json').read_text(encoding='utf-8'))['symbols'])


def test_unknown_variant_is_not_silently_baseline():
    with pytest.raises(ValueError, match='Unknown'):
        gate.momentum_targets({'VNINDEX': prices()}, set(), 'typo')

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


def test_registry_cannot_drop_a_required_crisis_to_turn_green(tmp_path):
    data = gate.load_registry()
    del data['blocks']['panic_2022']
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='required'):
        gate.load_registry(path)


@pytest.mark.parametrize('variant', gate.MOMENTUM + ('universe_equal',))
def test_research_momentum_variants_execute_and_reconcile(variant, rules):
    from tests.test_momentum_hypotheses import inputs
    frames = inputs()
    if variant == 'market_trend':
        frames['VNINDEX']['close'] = np.linspace(100, 200, len(frames['VNINDEX']))
    timeline = Timeline(dict(coverage_start='2024-01-01', coverage_end='2025-12-31',
        initial_known_on='2024-01-01', member_count=2, initial_members=['AAA','BBB'], changes=[]))
    result = gate.run_one(frames, rules, timeline, {}, '2025-01-01', '2025-02-03', 'momentum', variant)
    assert result['summary']['buys'] > 0
    assert result['reconciliation']['max_nav_error_vnd'] < 1e-5


def test_mr_bank_baseline_matches_production_scorer_and_future_invariance(rules):
    from tests.test_momentum_hypotheses import inputs
    from scripts import period_backtest as bt
    frames = inputs()
    # Force an oversold down-bar absorption candidate, with adequate history.
    f = frames['AAA']
    f.loc[280:285, ['open','high','low','close']] *= np.linspace(.98,.7,6)[:,None]
    f.loc[285, 'volume'] = 30e6
    timeline = Timeline(dict(coverage_start='2024-01-01', coverage_end='2025-12-31',
        initial_known_on='2024-01-01', member_count=2, initial_members=['AAA','BBB'], changes=[]))
    bank = gate.mr_signal_bank(frames, rules, timeline)
    baseline = {d: [{k:v for k,v in o.items() if k != 'original_signal_date'} for o in orders]
                for d,orders in bank['bracket'].items()}
    assert baseline == bt.mr_signals(frames, rules)
    assert sum(len(v) for v in bank['no_confirmation'].values()) > 0
    boundary = frames['AAA'].iloc[286]['date']
    shortened = {s:f.loc[f['date'] <= boundary] for s,f in frames.items()}
    prefix_bank = gate.mr_signal_bank(shortened, rules, timeline)
    assert prefix_bank == {v:{d:o for d,o in days.items() if d <= boundary} for v,days in bank.items()}


@pytest.mark.parametrize('scenario', ['normal','double_cost','delay_one_session'])
def test_mr_research_scenarios_do_not_change_frozen_signal_or_cash(scenario, rules):
    timeline = Timeline(dict(coverage_start='2025-12-01', coverage_end='2026-02-01',
        initial_known_on='2025-12-01', member_count=1, initial_members=['AAA'], changes=[]))
    bank = {'fixed15': {'2025-12-31': [order()]}}
    result = gate.run_one({'AAA': prices(), 'VNINDEX': prices()}, rules, timeline, bank,
                         '2026-01-01', '2026-01-12', 'mr', 'fixed15', scenario)
    assert result['reconciliation']['max_nav_error_vnd'] < 1e-5
    assert result['replay']['fills'][0]['date'] == ('2026-01-06' if scenario == 'delay_one_session' else '2026-01-05')


def test_positive_gate_and_market_trend_can_hold_cash():
    from tests.test_momentum_hypotheses import inputs
    frames = inputs()
    for f in frames.values():
        f['close'] = np.exp(np.linspace(5, 4, len(f)))
    assert gate.momentum_targets(frames, set(), 'positive_only')[0] == {}
    assert gate.momentum_targets(frames, set(), 'market_trend')[0] == {}

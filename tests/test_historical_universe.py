"""Point-in-time eligibility contracts; all execution fixtures are isolated."""
import copy
import importlib
import json
from pathlib import Path

import pytest

from scripts import momentum_hypotheses as rh
from tests.test_momentum_hypotheses import rules, short_inputs  # noqa: F401


def module():
    return importlib.import_module('scripts.historical_universe')


def registry():
    return json.loads(Path('configs/research/vn30_membership_2025_2026.json').read_text(encoding='utf-8'))


def toy_registry():
    return dict(coverage_start='2026-01-01', coverage_end='2026-02-06', member_count=1,
                initial_known_on='2026-01-01', initial_members=['AAA'],
                changes=[dict(known_on='2026-01-29', effective_on='2026-02-03',
                              add=['BBB'], remove=['AAA'])])


def test_actual_membership_has_thirty_and_matches_current_config():
    timeline = module().Timeline(registry())
    jan = timeline.members('2025-12-31', '2026-01-05')
    assert {'BCM', 'DGC', 'PLX', 'TPB'} <= jan
    assert not {'BSR', 'VPL', 'MCH', 'TCX'} & jan
    may = timeline.members('2026-05-12', '2026-05-13')
    assert 'BSR' in may and 'DGC' not in may
    current = json.loads(Path('configs/universe_vn30.json').read_text(encoding='utf-8'))['symbols']
    assert timeline.members('2026-09-25', '2026-09-25') == set(current)
    assert len(timeline.all_members) == 35


def test_membership_requires_both_prior_knowledge_and_effective_date():
    t = module().Timeline(toy_registry())
    assert t.members('2026-01-28', '2026-02-03') == {'AAA'}  # not announced
    assert t.members('2026-01-30', '2026-02-02') == {'AAA'}  # not effective
    assert t.members('2026-02-02', '2026-02-03') == {'BBB'}
    for known, execution in [('2025-12-31', '2026-02-02'), ('2026-02-06', '2026-02-09'),
                             ('2026-02-03', '2026-02-02')]:
        with pytest.raises(ValueError):
            t.members(known, execution)


@pytest.mark.parametrize('damage', ['duplicates', 'count', 'unknown_remove', 'existing_add',
                                  'late_knowledge', 'unsorted', 'initial_late', 'bounds'])
def test_invalid_registry_is_rejected(damage):
    data = toy_registry()
    if damage == 'duplicates':
        data['initial_members'] *= 2
    elif damage == 'count':
        data['member_count'] = 2
    elif damage == 'unknown_remove':
        data['changes'][0]['remove'] = ['NOPE']
    elif damage == 'existing_add':
        data['changes'][0]['add'] = ['AAA']
    elif damage == 'late_knowledge':
        data['changes'][0]['known_on'] = '2026-02-04'
    elif damage == 'unsorted':
        data['changes'] *= 2
    elif damage == 'initial_late':
        data['initial_known_on'] = '2026-01-02'
    else:
        data['coverage_end'] = '2025-12-31'
    with pytest.raises(ValueError):
        module().Timeline(data)


def test_retry_cannot_buy_removed_name_and_does_not_reselect_midmonth(monkeypatch, rules):
    hu = module()
    data = short_inputs()
    data['BBB'] = data['AAA'].copy()
    seen = []

    def targets(prefixes, held, variant):
        seen.append(set(prefixes))
        return {'AAA': .5}, []

    monkeypatch.setattr(rh, 'variant_targets', targets)
    t = hu.Timeline(toy_registry())
    result = hu.replay(data, rules, '2026-02-02', '2026-02-06',
                       'baseline', 'delay_one_session', 'historical', t)
    assert seen == [{'AAA', 'VNINDEX'}]
    assert not result['fills']
    assert result['skipped_entries']
    assert result['membership_decisions'][0]['members'] == ['AAA']


@pytest.mark.parametrize('variant', rh.VARIANTS)
def test_fixed_policy_exact_economic_parity_and_future_nonmembers_invariance(variant, monkeypatch, rules):
    hu = module()
    data = short_inputs()
    data['BBB'] = data['AAA'].copy()
    t = hu.Timeline(toy_registry())
    # Frozen end-of-coverage basket is BBB. Target callback is deterministic but real fills run.
    monkeypatch.setattr(rh, 'variant_targets', lambda f, h, v: ({s: .5 for s in f if s != 'VNINDEX'}, []))
    actual = hu.replay(data, rules, '2026-02-02', '2026-02-06', variant, 'normal', 'fixed_current', t)
    expected = rh.run_variant({s: f for s, f in data.items() if s != 'AAA'}, rules,
                              '2026-02-02', '2026-02-06', variant, 'normal')
    assert actual['fills'] and actual['fills'] == expected['fills']
    assert actual['nav'] == expected['nav']
    changed = copy.deepcopy(data)
    changed['AAA'].loc[:, ['open', 'high', 'low', 'close']] *= 9
    repeated = hu.replay(changed, rules, '2026-02-02', '2026-02-06', variant, 'normal', 'fixed_current', t)
    assert actual == repeated


def test_missing_required_history_invalid_policy_and_ml_fail_closed(rules):
    hu = module()
    data = short_inputs()
    t = hu.Timeline(toy_registry())
    with pytest.raises(ValueError, match='Missing'):
        hu.replay(data, rules, '2026-02-02', '2026-02-06', 'baseline', 'normal', 'historical', t)
    data['BBB'] = data['AAA'].copy()
    for variant, scenario, policy in [('x', 'normal', 'historical'),
                                      ('baseline', 'x', 'historical'), ('baseline', 'normal', 'x')]:
        with pytest.raises(ValueError):
            hu.replay(data, rules, '2026-02-02', '2026-02-06', variant, scenario, policy, t)
    rules['ml']['enabled'] = True
    with pytest.raises(ValueError, match='ML'):
        hu.replay(data, rules, '2026-02-02', '2026-02-06', 'baseline', 'normal', 'historical', t)

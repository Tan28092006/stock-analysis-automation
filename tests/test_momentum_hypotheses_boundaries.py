"""Numerical and same-day causality review after the first locked experiment."""
import copy

import pytest

from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from tests.test_momentum_hypotheses import inputs, rules  # noqa: F401


def test_machine_roundoff_is_not_counted_as_an_improving_block():
    blocks = {}
    for block in rh.BLOCKS:
        scenarios = {}
        for scenario in rh.SCENARIOS:
            scenarios[scenario] = {
                variant: dict(summary={'net_excess_return_pp': 1e-14},
                              improvement_over_control_pp=.1 if block == 'h1_2026' else 1e-14,
                              paired_vs_control={'ci_three_variants_pct': [1e-14, 1.0]})
                for variant in rh.VARIANTS[1:]
            }
        blocks[block] = {'scenarios': scenarios}
    checks = rh.decision_summary(blocks)['market_adjusted']['checks']
    assert checks['improves_at_least_two_blocks'] is False
    assert checks['h1_beats_vnindex'] is False
    assert checks['primary_adjusted_interval_positive'] is False


@pytest.mark.parametrize('variant', rh.VARIANTS)
def test_first_open_orders_ignore_that_days_later_prices(variant, rules):
    data = inputs()
    start = data['VNINDEX'].iloc[275]['date']
    end = data['VNINDEX'].iloc[290]['date']
    original = rh.run_variant(data, rules, start, end, variant, 'normal')
    altered = copy.deepcopy(data)
    for frame in altered.values():
        frame.loc[frame['date'] >= start, ['high', 'low', 'close', 'volume']] *= 5
    changed = rh.run_variant(altered, rules, start, end, variant, 'normal')
    original_fills = [f for f in original['fills'] if f['date'] == start]
    assert original_fills
    assert original_fills == [f for f in changed['fills'] if f['date'] == start]


def test_empty_history_flat_market_and_misaligned_benchmark_fail_safely():
    data = inputs()
    assert rh.portfolio_volatility(data, {}) == 0
    data['VNINDEX']['close'] = 100
    assert rh.market_adjusted_rank(data)
    data['VNINDEX'] = data['VNINDEX'].iloc[:-1]
    with pytest.raises(ValueError, match='aligned benchmark'):
        rh.market_adjusted_rank(data)
    assert not rh.reversal_entry_allowed('AAA', data)
    with pytest.raises(ValueError, match='Unknown'):
        rh.variant_targets(data, set(), 'invalid')


def test_reconcile_rejects_negative_cash_unsettled_sales_and_bad_lots(rules):
    data = inputs()
    start, end = data['VNINDEX'].iloc[275]['date'], data['VNINDEX'].iloc[290]['date']
    replay = rh.run_variant(data, rules, start, end, 'baseline', 'normal')
    for kind in ('cash', 'share_settlement', 'lot'):
        broken = copy.deepcopy(replay)
        if kind == 'cash':
            broken['fills'][0]['fee'] = 2e9
        elif kind == 'share_settlement':
            broken['fills'][0]['side'] = 'SELL'
        else:
            broken['fills'][0]['qty'] = 1
        with pytest.raises(ValueError):
            rh.reconcile(broken, data, 1e9)
    benchmark = rh.benchmark_nav(data, start, end, 1e9)
    benchmark[0]['date'] = '1900-01-01'
    with pytest.raises(ValueError, match='dates differ'):
        rh.summarize(replay, benchmark, 1e9, .001)


def test_adjusted_bootstrap_interval_is_wider_and_rejects_nonfinite():
    import numpy as np

    control = np.zeros(100)
    delta = np.random.default_rng(12).normal(0, .01, 100)
    result = rh.paired_interval(delta, control)
    assert result['ci_three_variants_pct'][0] < result['ci95_pct'][0]
    assert result['ci_three_variants_pct'][1] > result['ci95_pct'][1]
    delta[0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        rh.paired_interval(delta, control)

"""Non-vacuous causality and recoverable-failure contracts for the research lab."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import strategy_lab as lab, period_backtest as bt, historical_universe as hu
from tests.test_strategy_lab import request, sources  # noqa: F401


def test_missing_source_is_recorded_as_failed_trial(tmp_path):
    output = tmp_path / 'trial'
    missing = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError):
        lab.run(request(), missing, missing, missing, output)
    assert json.loads((output / 'request.json').read_text())['request'] == request()
    assert json.loads((output / 'failed.json').read_text())['status'] == 'failed'


def test_actual_mr_plans_have_fills_and_ignore_future_prices():
    days = bt.trading_days_between(pd.Timestamp('2025-02-03').date(), pd.Timestamp('2026-01-30').date())
    x = np.arange(len(days))
    close = 100 + np.sin(x / 2) * 10
    frame = pd.DataFrame(dict(date=[d.isoformat() for d in days], open=close,
                              high=close + 1, low=close - 1, close=close, volume=1e6))
    rules = json.loads(Path('configs/rules_mr.json').read_text())
    rules['mean_reversion']['require_cloud_clear'] = False
    config = request()
    config.update(start='2025-09-01', end='2025-12-31', capital=100000,
                  mr_overrides={'rsi_max': 100, 'band_touch_pct': 2, 'vol_climax_min': 0, 'min_rr': 0})
    configured = lab._rules(rules, config, 'strict')
    timeline = hu.Timeline(dict(coverage_start='2025-02-03', coverage_end='2026-01-30',
        initial_known_on='2025-02-03', member_count=1, initial_members=['AAA'], changes=[]))
    raw = bt.mr_signals({'AAA': frame}, configured)
    signals, _ = lab.eligible_mr_signals(raw, timeline, config['start'], config['end'])
    result = bt.replay_mr({'AAA': frame, 'VNINDEX': frame}, configured, config['start'], config['end'], signals)
    assert result['fills'] and result['closed_lots']
    changed = frame.copy()
    changed.loc[changed.date > config['end'], ['open', 'high', 'low', 'close']] *= 10
    other, _ = lab.eligible_mr_signals(bt.mr_signals({'AAA': changed}, configured), timeline,
                                      config['start'], config['end'])
    assert other == signals
    assert result == bt.replay_mr({'AAA': changed, 'VNINDEX': changed}, configured,
                                  config['start'], config['end'], other)


def test_overrides_apply_after_vn30_profile_and_never_modify_input():
    base = json.loads(Path('configs/rules_mr.json').read_text())
    before = copy.deepcopy(base)
    config = request()
    config['mr_overrides'] = {'rsi_max': 27, 'min_rr': 1.5}
    result = lab._rules(base, lab.validate_request(config), 'vn30')
    assert result['mean_reversion']['rsi_max'] == 27
    assert result['vn30_mean_reversion']['rsi_max'] == 27
    assert base == before


@pytest.mark.parametrize('damage', ['no_strategies', 'no_scenarios', 'bad_lot', 'negative_cost'])
def test_other_invalid_inputs(sources, tmp_path, damage):
    config = request()
    rules_path = sources[2]
    if damage == 'no_strategies':
        config['mr_profiles'] = config['momentum_variants'] = []
    elif damage == 'no_scenarios':
        config['scenarios'] = []
    else:
        rules = json.loads(rules_path.read_text())
        rules['backtest']['lot_size' if damage == 'bad_lot' else 'commission_pct'] = -1
        rules_path = tmp_path / 'rules.json'
        rules_path.write_text(json.dumps(rules))
    with pytest.raises(ValueError):
        lab.evaluate(config, sources[0], sources[1], rules_path)

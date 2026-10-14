"""Offline end-to-end output contract; snapshot integrity itself is tested separately."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import period_backtest as bt


def test_snapshot_to_artifact_and_refuse_overwrite(tmp_path, monkeypatch):
    days = bt.trading_days_between(date(2024, 10, 1), date(2026, 1, 9))
    x = np.arange(len(days))
    close = 100 + x * .05 + np.sin(x) * .2
    f = pd.DataFrame(dict(date=[d.isoformat() for d in days], open=close,
                          high=close + 1, low=close - 1, close=close, volume=1e6))
    files = {}
    for symbol in ('AAA', 'VNINDEX'):
        f.to_csv(tmp_path / f'{symbol}.csv', index=False)
        files[symbol] = {'path': f'{symbol}.csv'}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(bt, 'verify_snapshot', lambda _: {'files': files})
    output = tmp_path / 'output.json'
    monkeypatch.setattr(sys, 'argv', ['replay', '--manifest', str(manifest),
                                    '--end', '2026-01-09', '--output', str(output)])
    bt.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['kind'] == 'fixed_current_basket_rule_only_replay'
    assert report['momentum']['fills'][0]['date'] == '2026-01-05'
    assert len(report['mr']['nav']) == 5
    assert report['snapshot_sha256']
    assert report['assumptions']['ml'] is False
    with pytest.raises(FileExistsError):
        bt.main()


def test_production_mr_plan_is_equal_with_future_prices_replaced():
    rules = json.loads(Path('configs/rules_mr.json').read_text(encoding='utf-8'))
    rules['mean_reversion'].update(rsi_max=100, band_touch_pct=100,
                                  vol_climax_min=0, min_rr=0, require_cloud_clear=False)
    rules['vn30_mean_reversion'] = {}
    days = bt.trading_days_between(date(2025, 1, 1), date(2026, 1, 9))
    x = np.arange(len(days))
    close = 100 + np.sin(x / 2) * 10
    f = pd.DataFrame(dict(date=[d.isoformat() for d in days], open=close,
                          high=close + 1, low=close - 1, close=close, volume=1e6))
    cutoff = str(f.iloc[150]['date'])
    expected = bt.mr_signals({'AAA': f.iloc[:151]}, rules)
    assert expected  # Non-vacuous test: actual production BUY_SETUP objects.
    f.loc[151:, ['open', 'high', 'low', 'close']] *= 10
    actual = bt.mr_signals({'AAA': f}, rules)
    assert expected == {day: signals for day, signals in actual.items() if day <= cutoff}

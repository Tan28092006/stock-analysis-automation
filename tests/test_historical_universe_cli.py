"""End-to-end isolated runner artifact and mutation tests, without market I/O."""
import copy
import json
import sys
from datetime import date

import numpy as np
import pytest

from scripts import historical_universe as hu
from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from tests.test_historical_universe import toy_registry
from tests.test_momentum_hypotheses import frame, rules, short_inputs  # noqa: F401


def test_cli_72_replays_hashes_reconciliation_and_output_immutability(tmp_path, monkeypatch):
    registry = toy_registry()
    registry['coverage_end'] = '2026-02-13'
    timeline = tmp_path / 'timeline.json'
    timeline.write_text(json.dumps(registry), encoding='utf-8')
    dates = [d.isoformat() for d in bt.trading_days_between(date(2026, 1, 26), date(2026, 2, 13))]
    data = frame(np.sin(np.arange(len(dates) - 1)) * .001, dates)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    files = {}
    for symbol in ('AAA', 'BBB', 'VNINDEX'):
        data.to_csv(tmp_path / f'{symbol}.csv', index=False)
        files[symbol] = {'path': f'{symbol}.csv'}
    meta = {'files': files, 'as_of': '2026-02-13'}
    monkeypatch.setattr(hu, 'verify_snapshot', lambda _: meta)
    monkeypatch.setattr(rh, 'BLOCKS', {b: ('2026-02-02', '2026-02-13') for b in rh.BLOCKS})
    monkeypatch.setattr(rh, 'variant_targets', lambda f, h, v: ({s: .5 for s in f if s != 'VNINDEX'}, []))
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['historical', '--manifest', str(manifest),
                                    '--timeline', str(timeline), '--output', str(output)])
    hu.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['trial_registry']['total_replays'] == 72
    assert report['live_promotion'] == 'blocked'
    assert report['code_and_config_hashes'][str(timeline)] == rh.sha256(timeline)
    assert report['snapshot_sha256'] == rh.sha256(manifest)
    assert all(v['reconciliation']['max_nav_error_vnd'] < 1e-5
               for p in report['policies'].values() for b in p['blocks'].values()
               for s in b['scenarios'].values() for v in s.values())
    with pytest.raises(FileExistsError):
        hu.main()
    meta['as_of'] = '2026-02-12'
    with pytest.raises(ValueError, match='coverage'):
        hu.run_experiments(manifest, timeline, hu.Path('configs/rules_mr.json'))
    files.pop('BBB')
    with pytest.raises(ValueError, match='union'):
        hu.run_experiments(manifest, timeline, hu.Path('configs/rules_mr.json'))


def test_historical_first_orders_ignore_future_members_prices(monkeypatch, rules):
    data = short_inputs()
    data['BBB'] = data['AAA'].copy()
    timeline = hu.Timeline(toy_registry())
    monkeypatch.setattr(rh, 'variant_targets', lambda f, h, v: ({s: .5 for s in f if s != 'VNINDEX'}, []))
    original = hu.replay(data, rules, '2026-02-02', '2026-02-06', 'baseline', 'normal', 'historical', timeline)
    changed = copy.deepcopy(data)
    changed['BBB'].loc[:, ['open', 'high', 'low', 'close']] *= 10
    changed['AAA'].loc[changed['AAA']['date'] >= '2026-02-02', ['high', 'low', 'close']] *= 5
    repeated = hu.replay(changed, rules, '2026-02-02', '2026-02-06', 'baseline', 'normal', 'historical', timeline)
    assert original['fills'] and original['fills'] == repeated['fills']
    assert original['rebalances'] == repeated['rebalances']

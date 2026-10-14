"""Pinned daily-event experiment, full timeslices and VN100; local JSON only."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import momentum_events as events
from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from scripts import research_gate as gate
from scripts import research_timeslices as ts
from scripts import universe_expansion as universe
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot

PARENT_REPLAYS = 1196


def trial_names() -> list[str]:
    return [f'{v}/{s}' for v in events.VARIANTS for s in events.SCENARIOS]


def inference(values, reference) -> dict:
    p = events.load_protocol()
    x, y = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    if x.ndim != 1 or len(x) != len(y) or not len(x) or not np.isfinite(x - y).all():
        raise ValueError('Unaligned or nonfinite paired returns')
    tests = [gate.paired_test(x - y, family=p['family'], block=b)
             for b in p['bootstrap_blocks'] if len(x) >= b]
    status = 'insufficient_sessions'
    if len(x) >= p['min_sessions']:
        status = ('candidate_development_only' if all(t['ci_family'][0] > 0 and
                  t['p_one_sided_bonferroni'] < p['alpha'] for t in tests) else 'inconclusive_or_no_edge')
    return dict(status=status, tests=tests, live_eligible=False)


def windows(index: pd.DataFrame, *, full: bool) -> dict:
    p = ts.load_protocol() if full else universe.load_protocol()
    slices = ts.regime_slices(ts.classify_regimes(index, p['start'], p['end']))
    if full:
        slices = ts.calendar_slices(p['start'], p['end']) + slices
        result = dict(gate.load_registry()['blocks'])
    else:
        result = {'h1_2026': dict(start=p['start'], end=p['end'], kind='calendar')}
        slices.append(p['event'])
    result.update({s['id']: s for s in slices})
    return result


def evaluate_panel(frames: dict, rules: dict, timeline: Timeline, blocks: dict, primary: str) -> dict:
    initial = float(rules['backtest']['initial_capital'])
    out = dict(blocks={}, carry={}, replay_count=0)
    for name, bounds in blocks.items():
        start, end = bounds['start'], bounds['end']
        benchmark = rh.benchmark_nav(frames, start, end, initial)
        market = rh.daily_returns(benchmark, initial)
        block = dict(**bounds, trials={}, benchmark=benchmark)
        for trial in trial_names():
            variant, scenario = trial.split('/')
            replay = events.replay(frames, rules, timeline, start, end, variant, scenario)
            reconciliation = rh.reconcile(replay, frames, initial)
            slip = rules['backtest']['slippage_pct'] / 100 * (2 if scenario == 'double_cost' else 1)
            summary = ts.attribute_slice(replay, benchmark, initial, start, end, slip=slip)
            exposure_index = gate.exposure_control(replay['nav'], market, initial)
            summary['cash_return_pct'] = 0.
            summary['exposure_matched_index_return_pct'] = float((np.prod(1 + exposure_index) - 1) * 100)
            # Treat microscopic arithmetic residue as zero, never an economic edge.
            summary.update(profitable=summary['return_pct'] > 1e-9,
                beats_index=summary['net_excess_return_pp'] > 1e-9,
                positive_net_and_excess=summary['return_pct'] > 1e-9 and summary['net_excess_return_pp'] > 1e-9)
            item = dict(replay=replay, summary=summary, reconciliation=reconciliation,
                        activity=replay['activity'], funnel=replay['funnel'])
            if name == primary:
                item['portfolio_history'] = gate.position_history(replay, frames)
            if name == primary == 'continuous':
                returns = rh.daily_returns(replay['nav'], initial)
                item['vs_index'] = inference(returns, market)
                item['vs_exposure_index'] = inference(returns, exposure_index)
            block['trials'][trial] = item
            out['replay_count'] += 1
        if name == primary == 'continuous':
            for trial, item in block['trials'].items():
                variant, scenario = trial.split('/')
                if variant != 'daily55':
                    base = block['trials'][f'daily55/{scenario}']['replay']
                    item['vs_daily55'] = inference(rh.daily_returns(item['replay']['nav'], initial),
                                                   rh.daily_returns(base['nav'], initial))
        out['blocks'][name] = block
        print(f'events: {name}: {len(block["trials"])} reconciled paths', flush=True)
    parent = out['blocks'][primary]
    for name, bounds in blocks.items():
        if name == primary:
            continue
        out['carry'][name] = {}
        for trial, item in parent['trials'].items():
            scenario = trial.split('/')[1]
            slip = rules['backtest']['slippage_pct'] / 100 * (2 if scenario == 'double_cost' else 1)
            out['carry'][name][trial] = ts.attribute_slice(item['replay'], parent['benchmark'], initial,
                bounds['start'], bounds['end'], slip=slip)
    return out


def verify_parent(path: Path, manifest_hash: str, *, kind: str = 'gate') -> dict:
    parent = json.loads(Path(path).read_text(encoding='utf-8'))
    status = {'gate': 'research_complete_live_blocked',
              'universe': 'universe_research_complete_full_regression_required'}[kind]
    if parent.get('status') != status or parent.get('manifest_sha256') != manifest_hash:
        raise ValueError('Incomplete or different parent snapshot')
    expected = {p.as_posix() for p in Path('scripts').glob('*.py')} | {
        p.as_posix() for p in Path('stock_agent').rglob('*.py')} | {gate.RULES.as_posix(), gate.REGISTRY.as_posix()}
    hashes = {p.replace('\\', '/'): h for p, h in parent.get('source_hashes', {}).items()}
    if not expected <= hashes.keys() or any(not Path(p).is_file() or gate.digest(p) != h for p, h in hashes.items()):
        raise ValueError('Parent source inventory differs; rerun complete parent suite')
    return parent


def run_suite(manifest_path: Path, h1_path: Path, parent_path: Path, output: Path,
              universe_path: Path | None = None) -> dict:
    p = events.load_protocol()
    manifest = verify_snapshot(manifest_path)
    if gate.digest(manifest_path) != p['snapshots']['VN30']:
        raise ValueError('Not the preregistered full-history snapshot')
    if gate.digest(h1_path) != p['snapshots']['H1']:
        raise ValueError('Not the preregistered H1 snapshot')
    parent = verify_parent(parent_path, p['snapshots']['VN30'])
    if universe_path is None:
        raise ValueError('Paired-universe parent evidence required')
    universe_parent = verify_parent(universe_path, p['snapshots']['H1'], kind='universe')
    timeline = Timeline(json.loads(gate.TIMELINE.read_text(encoding='utf-8')))
    required = timeline.all_members | {'VNINDEX'}
    if not required <= manifest['files'].keys():
        raise ValueError('Missing full-history universe prices')
    frames = {s: pd.read_csv(manifest_path.parent / manifest['files'][s]['path']) for s in sorted(required)}
    h1_frames, h1_timelines, rules = universe.load_inputs(h1_path, universe.load_protocol())
    source_files = sorted(set(Path('stock_agent').rglob('*.py')) | set(Path('scripts').glob('*.py'))
                          | set(Path('configs/research').glob('*.json')) | {gate.RULES})
    hashes = {f.as_posix(): gate.digest(f) for f in source_files}
    result = dict(schema_version=1, status='running', live_eligible=False, holdout=False, protocol=p,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes=hashes, manifest_sha256=p['snapshots'], required_parent_replays=PARENT_REPLAYS,
        parent_results_sha256=dict(gate=gate.digest(parent_path), universe=gate.digest(universe_path)), panels={})
    output.mkdir(parents=True, exist_ok=False)
    panels = [('VN30', frames, timeline, windows(frames['VNINDEX'], full=True), 'continuous')]
    h1_windows = windows(h1_frames['VNINDEX'], full=False)
    panels += [(f'H1_{name}', h1_frames, member, h1_windows, 'h1_2026') for name, member in h1_timelines.items()]
    for name, data, member, bounds, primary in panels:
        panel = evaluate_panel(data, rules, member, bounds, primary)
        # Existing monthly engine is a descriptive multi-mechanism comparator.
        if name == 'VN30':
            panel['prior_monthly_control'] = {k: v['momentum']['baseline']['summary'] for k, v in parent['blocks'].items()}
        else:
            universe_name = name.removeprefix('H1_')
            h1_parent = next(b for b in universe_parent['blocks'] if b['id'] == 'calendar_2026_h1')
            panel['prior_monthly_control'] = h1_parent['universes'][universe_name]['cash_restart']['momentum/baseline/normal']['metrics']
        result['panels'][name] = panel
        (output / f'{name}.json').write_text(json.dumps(panel, allow_nan=False), encoding='utf-8')
    a = result['panels']['H1_VN30']['blocks']['h1_2026']['trials']
    b = result['panels']['H1_VN100']['blocks']['h1_2026']['trials']
    initial = rules['backtest']['initial_capital']
    result['paired_universe_tests'] = {k: inference(rh.daily_returns(b[k]['replay']['nav'], initial),
                                                  rh.daily_returns(a[k]['replay']['nav'], initial)) for k in trial_names()}
    verify_snapshot(manifest_path)
    verify_snapshot(h1_path)
    if any(gate.digest(f) != h for f, h in hashes.items()):
        raise ValueError('Source changed during experiment')
    if gate.digest(manifest_path) != p['snapshots']['VN30'] or gate.digest(h1_path) != p['snapshots']['H1']:
        raise ValueError('Manifest changed during experiment')
    result['status'] = 'research_complete_live_blocked'
    result['replay_count'] = sum(panel['replay_count'] for panel in result['panels'].values())
    result['live_blockers'] = ['No untouched prospective evidence', 'Corporate actions/reference prices not certified',
        'No auction queue or impact validation', 'Numerical activity floor unresolved', 'Research engine not activated in live pipeline']
    (output / 'results.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--h1-manifest', type=Path, required=True)
    parser.add_argument('--gate-results', type=Path, required=True)
    parser.add_argument('--universe-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_suite(args.manifest, args.h1_manifest, args.gate_results, args.output, args.universe_results)
    print(json.dumps(dict(status=result['status'], replays=result['replay_count'], live_eligible=False)))


if __name__ == '__main__':
    main()

"""Offline dual-strategy research backend; no production cache, model or order writes.

Run from the repository root: python -m scripts.strategy_lab --help.
Each output directory is a new immutable trial, including failed attempts.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import historical_universe as hu, momentum_hypotheses as rh, period_backtest as bt
from stock_agent.data.reconciliation import verify_snapshot

FIELDS = {'start', 'end', 'capital', 'mr_profiles', 'momentum_variants', 'scenarios', 'mr_overrides'}
OVERRIDE_BOUNDS = {'rsi_max': (0, 100), 'band_touch_pct': (.5, 2),
                   'vol_climax_min': (0, 20), 'stop_atr_multiple': (.1, 20),
                   'min_rr': (0, 20), 'max_hold_days': (1, 120)}
LIMITATIONS = [
    'Historical research, not untouched out-of-sample or a certificate of live profitability.',
    'Manually reconstructed membership; provider corporate-action adjustments and dividends unverified.',
    'Daily open fills, price-limit proxies and conservative T+3 cannot prove actual execution.',
    'Each strategy/scenario has independent starting capital; returns must not be added together.',
    'VNINDEX is a gross price index, not an investable total-return benchmark.',
    'Open positions are marked to market without hypothetical terminal liquidation fees.',
    'Dashboard daily picks and manual real fills are not this simulated portfolio.',
]


def _number(value: object, low: float, high: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Invalid numeric {field}: expected [{low}, {high}]')


def validate_request(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError(f'Request requires exactly: {sorted(FIELDS)}')
    for key in ('start', 'end'):
        if not isinstance(config[key], str) or date.fromisoformat(config[key]).isoformat() != config[key]:
            raise ValueError(f'{key} must be YYYY-MM-DD')
    if config['start'] > config['end']:
        raise ValueError('Reversed evaluation dates')
    _number(config['capital'], 1, 1e12, 'capital')
    for key, allowed in [('mr_profiles', {'vn30', 'strict'}),
                         ('momentum_variants', set(rh.VARIANTS)), ('scenarios', {'normal', 'double_cost'})]:
        values = config[key]
        if (not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f'Invalid or duplicate {key}')
    if not config['scenarios'] or not (config['mr_profiles'] or config['momentum_variants']):
        raise ValueError('At least one strategy and scenario required')
    overrides = config['mr_overrides']
    if not isinstance(overrides, dict) or set(overrides) - set(OVERRIDE_BOUNDS):
        raise ValueError('Unknown MR override')
    for key, value in overrides.items():
        _number(value, *OVERRIDE_BOUNDS[key], key)
        if key == 'max_hold_days' and not isinstance(value, int):
            raise ValueError('max_hold_days must be an integer')
    return copy.deepcopy(config)


def eligible_mr_signals(signals: dict, timeline: hu.Timeline, start: str, end: str) -> tuple[dict, list]:
    accepted, rejected = {}, []
    for signal_date, plans in sorted(signals.items()):
        # History before the experiment is warm-up, not another membership query.
        if signal_date >= end:
            continue
        execution = bt.add_trading_days(date.fromisoformat(signal_date), 1).isoformat()
        if not start <= execution <= end:
            continue
        members = timeline.members(signal_date, execution)
        accepted[signal_date] = []
        for plan in plans:
            if plan['symbol'] in members:
                accepted[signal_date].append(copy.deepcopy(plan))
            else:
                rejected.append(dict(signal_date=signal_date, execution_date=execution,
                                     symbol=plan['symbol'], reason='not_pit_member', plan=copy.deepcopy(plan)))
    return accepted, rejected


def _rules(base: dict, config: dict, profile: str | None = None) -> dict:
    rules = copy.deepcopy(base)
    rules['backtest']['initial_capital'] = config['capital']
    rules.setdefault('money', {})['account_nav'] = config['capital']
    if profile is not None:
        mr = dict(rules['mean_reversion'])
        if profile == 'vn30':
            mr.update(rules.get('vn30_mean_reversion', {}))
        mr.update(config['mr_overrides'])
        rules['mean_reversion'] = mr
        # Preserve effective overrides for audit; strict explicitly disables them.
        rules['vn30_mean_reversion'] = dict(mr) if profile == 'vn30' else {}
    return rules


def evaluate(config: dict, manifest_path: Path, timeline_path: Path, rules_path: Path) -> dict:
    config = validate_request(config)
    manifest = verify_snapshot(manifest_path)
    timeline = hu.Timeline(json.loads(timeline_path.read_text(encoding='utf-8')))
    base = json.loads(rules_path.read_text(encoding='utf-8'))
    if base.get('ml', {}).get('enabled') is not False:
        raise ValueError('Research requires explicit ML disabled')
    if base['backtest']['lot_size'] != 100:
        raise ValueError('Only 100-share lots supported')
    for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
        _number(base['backtest'][key], 0, 10, key)
    start, end = config['start'], config['end']
    if start < timeline.start or end > min(timeline.end, manifest['as_of']):
        raise ValueError('Insufficient snapshot or membership coverage')
    if set(manifest['files']) != timeline.all_members | {'VNINDEX'}:
        raise ValueError('Snapshot must contain exactly the historical membership union and VNINDEX')
    paths = [manifest_path, timeline_path, rules_path, Path(__file__), Path(bt.__file__),
             Path(rh.__file__), Path(hu.__file__), *sorted(Path('stock_agent/features').glob('*.py')),
             *sorted(Path('stock_agent/data').glob('*.py'))]
    hashes = {str(p): rh.sha256(p) for p in paths}
    frames = {}
    for symbol, item in manifest['files'].items():
        frame = pd.read_csv(manifest_path.parent / item['path'])
        frames[symbol] = frame.loc[frame['date'] <= end].reset_index(drop=True)
    bt.validate_frames({s: f for s, f in frames.items() if not f.empty}, start, end)
    index_days = frames['VNINDEX']['date'].tolist()
    period_days = [d for d in index_days if start <= d <= end]
    first = index_days.index(period_days[0])
    if first == 0:
        raise ValueError('Need a prior close')
    timeline.members(index_days[first - 1], period_days[0])
    initial = config['capital']
    benchmark = rh.benchmark_nav(frames, start, end, initial)
    runs = {}

    def record(key: str, replay: dict, rules: dict) -> None:
        summary = rh.summarize(replay, benchmark, initial, rules['backtest']['slippage_pct'] / 100)
        summary['sample_warning'] = ('Few closed lots; win rate is descriptive, not evidence of an edge.'
                                     if summary['closed_lots'] < 30 else
                                     'Closed lots are dependent observations; no independent significance claim.')
        runs[key] = dict(summary=summary, rules=rules, replay=replay,
                         reconciliation=rh.reconcile(replay, frames, initial))

    for profile in config['mr_profiles']:
        rules = _rules(base, config, profile)
        raw = bt.mr_signals(frames, rules)
        signals, rejected = eligible_mr_signals(raw, timeline, start, end)
        for scenario in config['scenarios']:
            configured = copy.deepcopy(rules)
            if scenario == 'double_cost':
                for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
                    configured['backtest'][key] *= 2
            result = bt.replay_mr(frames, configured, start, end, signals=signals)
            result.update(eligible_signals=signals, membership_rejections=rejected)
            record(f'mr:{profile}:{scenario}', result, configured)
    for variant in config['momentum_variants']:
        rules = _rules(base, config)
        for scenario in config['scenarios']:
            result = hu.replay(frames, rules, start, end, variant, scenario, 'historical', timeline)
            effective = copy.deepcopy(rules)
            if scenario == 'double_cost':
                for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
                    effective['backtest'][key] *= 2
            record(f'momentum:{variant}:{scenario}', result, effective)
    # Detect inputs or source code being modified while a potentially long run executes.
    verify_snapshot(manifest_path)
    if hashes != {str(p): rh.sha256(p) for p in paths}:
        raise ValueError('Source changed during experiment')
    return dict(status='completed_research', schema_version=1, live_approved=False,
                generated_at=datetime.now(timezone.utc).isoformat(), request=config,
                code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_hashes=hashes, timeline=timeline.data, trial_count=len(runs),
                benchmark=dict(metrics=bt.metrics(benchmark, [], initial), nav=benchmark),
                runs=runs, limitations=LIMITATIONS)


def _write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def run(config: dict, manifest_path: Path, timeline_path: Path, rules_path: Path, output: Path) -> dict:
    config = validate_request(config)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'request.json', dict(request=config, created_at=datetime.now(timezone.utc).isoformat(),
        inputs={name: dict(path=str(p.resolve()))
                for name, p in [('manifest', manifest_path), ('timeline', timeline_path), ('rules', rules_path)]}))
    try:
        _write_json(output / 'input_hashes.json', {str(p.resolve()): rh.sha256(p)
                                                for p in (manifest_path, timeline_path, rules_path)})
        result = evaluate(config, manifest_path, timeline_path, rules_path)
        _write_json(output / 'result.json', result)
        lines = ['# Strategy lab — research only', '', f"Period: {config['start']} to {config['end']}",
                 f"Independent starting capital per run: {config['capital']:,.0f} VND", '',
                 '| Strategy / scenario | Net return % | Excess vs VNINDEX pp | Max DD % | Closed lots |',
                 '|---|---:|---:|---:|---:|']
        for key, item in result['runs'].items():
            s = item['summary']
            lines.append(f"| {key} | {s['return_pct']:.4f} | {s['net_excess_return_pp']:.4f} | "
                         f"{s['max_drawdown_pct']:.4f} | {s['closed_lots']} |")
        lines += ['', '## Limitations', '', *[f'- {item}' for item in LIMITATIONS]]
        with (output / 'summary.md').open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        return result
    except Exception as exc:
        _write_json(output / 'failed.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, default=hu.TIMELINE)
    parser.add_argument('--rules', type=Path, default=Path('configs/rules_mr.json'))
    parser.add_argument('--output', type=Path, required=True, help='New directory; never reuses an existing trial')
    args = parser.parse_args(argv)
    result = run(json.loads(args.spec.read_text(encoding='utf-8')), args.manifest, args.timeline, args.rules, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'trial_count': result['trial_count'],
                      'status': result['status'], 'live_approved': False}))


if __name__ == '__main__':
    main()

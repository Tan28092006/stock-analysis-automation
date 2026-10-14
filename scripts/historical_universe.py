"""Locked research-only membership sensitivity. Never writes production state."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from stock_agent.data.reconciliation import verify_snapshot

PROTOCOL = Path('docs/audits/2026-09-26-pit-universe-protocol.md')
TIMELINE = Path('configs/research/vn30_membership_2025_2026.json')
POLICIES = ('fixed_current', 'historical')


def _day(value: str) -> str:
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError('Dates must use YYYY-MM-DD')
    return value


class Timeline:
    """Conservative close-of-day knowledge and separately effective membership."""

    def __init__(self, data: dict):
        self.data = copy.deepcopy(data)
        self.start, self.end = _day(data['coverage_start']), _day(data['coverage_end'])
        known = _day(data['initial_known_on'])
        count = data['member_count']
        current = set(data['initial_members'])
        if (known > self.start or self.start > self.end or count <= 0
                or len(current) != len(data['initial_members']) or len(current) != count):
            raise ValueError('Invalid initial membership or coverage')
        self.initial = current.copy()
        self.all_members = current.copy()
        self.events = []
        previous = self.start
        for event in self.data['changes']:
            known, effective = _day(event['known_on']), _day(event['effective_on'])
            added, removed = set(event['add']), set(event['remove'])
            if (not self.start <= known <= effective <= self.end or effective <= previous
                    or len(added) != len(event['add']) or len(removed) != len(event['remove'])
                    or added & current or not removed <= current or added & removed):
                raise ValueError('Invalid membership change or announcement chronology')
            current = (current - removed) | added
            if len(current) != count:
                raise ValueError('Member count changed')
            self.events.append((known, effective, added, removed))
            self.all_members |= added
            previous = effective

    def members(self, as_of: str, execution: str) -> set[str]:
        as_of, execution = _day(as_of), _day(execution)
        if not self.start <= as_of <= execution <= self.end:
            raise ValueError('Membership date outside documented coverage or reversed')
        current = self.initial.copy()
        for known, effective, added, removed in self.events:
            if known <= as_of and effective <= execution:
                current = (current - removed) | added
        return current


def _next_close_session(prefixes: dict) -> tuple[str, str]:
    prior = str(prefixes['VNINDEX']['date'].iloc[-1])
    return prior, bt.add_trading_days(date.fromisoformat(prior), 1).isoformat()


def replay(frames: dict, rules: dict, start: str, end: str, variant: str,
           scenario: str, policy: str, timeline: Timeline) -> dict:
    if variant not in rh.VARIANTS or scenario not in rh.SCENARIOS or policy not in POLICIES:
        raise ValueError('Unknown locked variant, scenario, or membership policy')
    if rules.get('ml', {}).get('enabled'):
        raise ValueError('Retroactive ML is forbidden')
    required = timeline.all_members | {'VNINDEX'}
    if not required <= set(frames):
        raise ValueError(f'Missing union price files: {sorted(required - set(frames))}')
    configured = copy.deepcopy(rules)
    if scenario == 'double_cost':
        for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
            configured['backtest'][key] *= 2
    fixed = timeline.members(timeline.end, timeline.end)
    active_frames = {s: f for s, f in frames.items()
                     if s in (fixed if policy == 'fixed_current' else timeline.all_members) | {'VNINDEX'}}
    decisions = []

    def eligible(prefixes):
        prior, planned = _next_close_session(prefixes)
        members = fixed if policy == 'fixed_current' else timeline.members(prior, planned)
        return prior, planned, members

    def targets(prefixes, held):
        prior, planned, members = eligible(prefixes)
        subset = {s: f for s, f in prefixes.items() if s in members | {'VNINDEX'}}
        decisions.append(dict(signal_date=prior, planned_session=planned, members=sorted(members)))
        return rh.variant_targets(subset, held, variant)

    def gate(symbol, prefixes):
        _, _, members = eligible(prefixes)
        return symbol in members and (variant != 'reversal_entry' or rh.reversal_entry_allowed(symbol, prefixes))

    result = bt.replay_momentum(
        active_frames, configured, start, end, target_fn=targets,
        entry_gate=gate if policy == 'historical' or variant == 'reversal_entry' else None,
        execution_delay=int(scenario == 'delay_one_session'),
    )
    result['membership_decisions'] = decisions
    if policy == 'historical':
        dates = frames['VNINDEX']['date'].tolist()
        previous = {day: dates[i - 1] for i, day in enumerate(dates) if i}
        for fill in result['fills']:
            if fill['side'] == 'BUY' and fill['symbol'] not in timeline.members(previous[fill['date']], fill['date']):
                raise ValueError('Ineligible historical purchase')
    return result


def run_experiments(manifest_path: Path, timeline_path: Path, rules_path: Path) -> dict:
    manifest = verify_snapshot(manifest_path)
    timeline = Timeline(json.loads(timeline_path.read_text(encoding='utf-8')))
    rules = json.loads(rules_path.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled') or rules['backtest']['lot_size'] != 100:
        raise ValueError('Locked protocol requires ML off and 100-share lots')
    if set(manifest['files']) != timeline.all_members | {'VNINDEX'}:
        raise ValueError('Snapshot must contain exactly the full membership union and VNINDEX')
    full = {s: pd.read_csv(manifest_path.parent / item['path']) for s, item in manifest['files'].items()}
    initial = float(rules['backtest']['initial_capital'])
    policies = {}
    for policy in POLICIES:
        blocks = {}
        for block, (start, end) in rh.BLOCKS.items():
            if end > manifest['as_of'] or end > timeline.end:
                raise ValueError('Snapshot or membership coverage ends before evaluation')
            frames = {s: f.loc[f['date'] <= end].reset_index(drop=True) for s, f in full.items()}
            bt.validate_frames({s: f for s, f in frames.items() if not f.empty}, start, end)
            benchmark = rh.benchmark_nav(frames, start, end, initial)
            scenarios = {}
            for scenario in rh.SCENARIOS:
                variants = {}
                for variant in rh.VARIANTS:
                    result = replay(frames, rules, start, end, variant, scenario, policy, timeline)
                    slip = rules['backtest']['slippage_pct'] / 100 * (2 if scenario == 'double_cost' else 1)
                    variants[variant] = dict(
                        summary=rh.summarize(result, benchmark, initial, slip), replay=result,
                        reconciliation=rh.reconcile(result, frames, initial),
                    )
                    if variant != 'baseline':
                        control = variants['baseline']
                        variants[variant]['improvement_over_control_pp'] = (
                            result['metrics']['return_pct'] - control['summary']['return_pct'])
                        variants[variant]['paired_vs_control'] = rh.paired_interval(
                            rh.daily_returns(result['nav'], initial), rh.daily_returns(control['replay']['nav'], initial))
                scenarios[scenario] = variants
            blocks[block] = dict(start=start, end=end, sessions=len(benchmark),
                                 benchmark=dict(metrics=bt.metrics(benchmark, [], initial), nav=benchmark),
                                 scenarios=scenarios)
        policies[policy] = dict(blocks=blocks, diagnostic_decisions=rh.decision_summary(blocks))
    deltas = {
        block: {scenario: {variant: (
            policies['historical']['blocks'][block]['scenarios'][scenario][variant]['summary']['return_pct']
            - policies['fixed_current']['blocks'][block]['scenarios'][scenario][variant]['summary']['return_pct'])
            for variant in rh.VARIANTS} for scenario in rh.SCENARIOS} for block in rh.BLOCKS
    }
    hashed_paths = [Path(__file__), Path(rh.__file__), Path(bt.__file__), rules_path,
                    Path('stock_agent/features/momentum_scan.py'), Path('stock_agent/data/exchange_calendar.py'),
                    Path('stock_agent/data/reconciliation.py'), timeline_path, PROTOCOL, rh.PROTOCOL]
    return dict(
        schema_version=1, kind='historical_membership_sensitivity',
        generated_at=datetime.now(timezone.utc).isoformat(),
        code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_and_config_hashes={str(p): rh.sha256(p) for p in hashed_paths},
        snapshot_manifest=str(manifest_path.resolve()), snapshot_sha256=rh.sha256(manifest_path),
        timeline=timeline.data, rules=rules,
        trial_registry=dict(total_replays=72, policies=list(POLICIES), blocks=list(rh.BLOCKS),
                            variants=list(rh.VARIANTS), scenarios=list(rh.SCENARIOS), parameter_search=False),
        policies=policies, historical_minus_fixed_return_pp=deltas, live_promotion='blocked',
        limitations=[
            'Manual source-based membership reconstruction; not archived point-in-time market data.',
            'Provider adjustment/corporate-action lineage and dividends unverified.',
            'Development sensitivity on already examined periods, NOT untouched OOS.',
            'Intervals are diagnostic, not a global correction for the expanded trial history.',
            'Daily open fills and T+3 are conservative proxies, not actual execution evidence.',
            'Index removals stop incremental purchases but existing holdings wait for monthly rebalance.',
            'Gross price-only VNINDEX is not an investable total-return benchmark.',
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, default=TIMELINE)
    parser.add_argument('--rules', type=Path, default=Path('configs/rules_mr.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    result = run_experiments(args.manifest, args.timeline, args.rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result['historical_minus_fixed_return_pp'], indent=2))


if __name__ == '__main__':
    main()

"""Versioned, research-only regime/ablation suite. Never changes live state."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot
from stock_agent.features.signal_engine import prepare_signal_frame, _score_mean_reversion_from_features

REGISTRY = Path('configs/research/research_gate_v1.json')
TIMELINE = Path('configs/research/vn30_membership_2022_2026.json')
RULES = Path('configs/rules_mr.json')
MOMENTUM = ('baseline', 'market_adjusted', 'reversal_entry', 'own_portfolio_vol',
            'equal_weight', 'no_buffer', 'positive_only', 'market_trend', 'formation_6_1', 'high_52w')
MR = ('bracket', 'fixed15', 'no_rsi', 'no_band', 'no_confirmation', 'no_rr',
      'no_cloud', 'no_vsa', 'strict', 'fixed5', 'fixed10', 'fixed20')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registry(path=REGISTRY):
    r = json.loads(Path(path).read_text(encoding='utf-8'))
    if (r['holdout'] is not False or tuple(r['momentum']) != MOMENTUM or tuple(r['mr']) != MR
            or r['scenarios'] != ['normal', 'double_cost', 'delay_one_session']
            or r['bootstrap_blocks'] != [20, 40]):
        raise ValueError('Registry and implementation differ; version and tests required')
    for block in r['blocks'].values():
        if date.fromisoformat(block['start']) > date.fromisoformat(block['end']):
            raise ValueError('Reversed block')
    return r


def paired_test(delta, *, family, block, draws=4000):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or len(delta) < block or not np.isfinite(delta).all() or family < 1:
        raise ValueError('Invalid paired observations/family')
    rng = np.random.default_rng(20260930)
    n = len(delta)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    means = delta[indices].mean(axis=1)
    observed = float(delta.mean())
    centered = means - observed
    p = (1 + int(np.sum(centered >= observed))) / (draws + 1)
    return dict(observations=n, block=block, draws=draws, seed=20260930,
                mean_annualized_pp=observed * 25200,
                ci95=(np.quantile(means, [.025, .975]) * 25200).tolist(),
                ci_family=(np.quantile(means, [.05 / (2 * family), 1 - .05 / (2 * family)]) * 25200).tolist(),
                p_one_sided_bonferroni=min(1., p * family), family=family,
                inference='development-only; paired circular blocks; finite bootstrap resolution')


def exposure_control(nav, market_returns, initial):
    previous = [0.] + [r['exposure'] / r['nav'] for r in nav[:-1]]
    return np.asarray(previous) * np.asarray(market_returns)


def position_history(result, frames):
    prices = {s: f.set_index('date')['close'].to_dict() for s, f in frames.items()}
    qty, fills, history = {}, {}, []
    for fill in result['fills']:
        fills.setdefault(fill['date'], []).append(fill)
    for nav in result['nav']:
        for fill in fills.get(nav['date'], []):
            s = fill['symbol']
            qty[s] = qty.get(s, 0) + fill['qty'] * (1 if fill['side'] == 'BUY' else -1)
        positions = {s: dict(qty=q, close=prices[s][nav['date']], value=q * prices[s][nav['date']])
                     for s, q in qty.items() if q}
        history.append({**nav, 'positions': positions})
    return history


def momentum_targets(frames, held, variant):
    if variant in rh.VARIANTS:
        return rh.variant_targets(frames, held, variant)
    if variant not in MOMENTUM and variant != 'universe_equal':
        raise ValueError(f'Unknown variant: {variant}')
    if variant == 'universe_equal':
        symbols = sorted(set(frames) - {'VNINDEX'})
        return {s: 1 / len(symbols) for s in symbols}, []
    ranked = bt._rank({s: f for s, f in frames.items() if s != 'VNINDEX'})
    if variant in ('formation_6_1', 'high_52w'):
        ranked = [(s, (float(frames[s]['close'].iloc[-22]) / float(frames[s]['close'].iloc[-127]) - 1)
                   if variant == 'formation_6_1' else float(frames[s]['close'].iloc[-1]) / float(frames[s]['close'].tail(252).max()),
                   v, c) for s, _, v, c in ranked]
        ranked.sort(key=lambda r: (-r[1], r[0]))
    if variant == 'positive_only':
        ranked = [r for r in ranked if r[1] > 0]
    if variant == 'market_trend':
        index = frames['VNINDEX']['close']
        if index.iloc[-1] <= index.ewm(span=200, adjust=False).mean().iloc[-1]:
            return {}, []
    selected = [] if variant == 'no_buffer' else [r for r in ranked[:20] if r[0] in held][:10]
    for row in ranked:
        if len(selected) >= 10:
            break
        if row[0] not in {r[0] for r in selected}:
            selected.append(row)
    inv = {s: 1. if variant == 'equal_weight' else 1 / max(v, .05) for s, _, v, _ in selected}
    vol = float(frames['VNINDEX']['close'].pct_change().tail(20).std() * math.sqrt(252))
    exposure = min(1., .2 / max(vol, 1e-6))
    total = sum(inv.values()) or 1.
    return {s: w / total * exposure for s, w in inv.items()}, sorted(set(frames) - {'VNINDEX'} - {r[0] for r in ranked})


def mr_signal_bank(frames, rules, timeline):
    """One production scorer per bar; ablate its explicit boolean evidence only."""
    selected = copy.deepcopy(rules)
    selected['mean_reversion'].update(rules.get('vn30_mean_reversion', {}))
    bank = {name: {} for name in MR}
    removed = dict(no_rsi='rsi', no_band='band', no_confirmation='confirmation', no_rr='rr', no_cloud='cloud')
    for symbol, frame in sorted(frames.items()):
        if symbol == 'VNINDEX':
            continue
        prepared = prepare_signal_frame(frame.copy(), selected)
        for i in range(max(1, int(rules.get('min_history_rows', 90)) - 1), len(prepared)):
            day = str(frame.iloc[i]['date'])
            next_day = bt.add_trading_days(date.fromisoformat(day), 1).isoformat()
            if day < timeline.start or next_day > timeline.end or symbol not in timeline.members(day, next_day):
                continue
            sig = _score_mean_reversion_from_features(symbol, prepared.iloc[:i + 1], selected,
                                                      idx=i, include_features=False)
            evidence = {e.name: e.passed for e in sig.evidence}
            gates = dict(rsi=evidence['Oversold RSI'], band=evidence['At lower band'],
                         confirmation=(evidence['Reversal bar'] and evidence['Volume climax']) or evidence.get('VSA stopping volume', False),
                         rr=evidence.get('MR risk reward', True), cloud=evidence.get('Cloud clearance', True))
            for variant in MR:
                checks = gates.copy()
                if variant in removed:
                    checks[removed[variant]] = True
                elif variant == 'no_vsa':
                    checks['confirmation'] = evidence['Reversal bar'] and evidence['Volume climax']
                elif variant == 'strict':
                    checks.update(rsi=float(prepared.iloc[i]['rsi14']) < 30,
                                  band=float(prepared.iloc[i]['close']) <= float(prepared.iloc[i]['bb_lower']) * 1.01)
                if not all(checks.values()):
                    continue
                plan = sig.risk_plan
                hold = int(variant[5:]) if variant.startswith('fixed') else plan.holding_period_days
                bank[variant].setdefault(day, []).append(dict(symbol=symbol, stop=plan.stop_loss,
                    target=plan.take_profit_1, hold=hold, rr=plan.reward_risk, original_signal_date=day))
    return {v: {d: sorted(orders, key=lambda o: (-o['rr'], o['symbol'])) for d, orders in days.items()}
            for v, days in bank.items()}


def run_one(frames, rules, timeline, bank, start, end, kind, variant, scenario='normal'):
    configured = copy.deepcopy(rules)
    if scenario == 'double_cost':
        for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
            configured['backtest'][key] *= 2
    if kind == 'mr':
        signals = bank[variant]
        if scenario == 'delay_one_session':
            shifted = {}
            for day, orders in signals.items():
                prior = bt.add_trading_days(date.fromisoformat(day), 1).isoformat()
                execution = bt.add_trading_days(date.fromisoformat(prior), 1).isoformat()
                if execution <= timeline.end:
                    shifted[prior] = [o for o in orders if o['symbol'] in timeline.members(prior, execution)]
            signals = shifted
        result = bt.replay_mr(frames, configured, start, end, signals,
                              exit_policy='fixed_hold' if variant.startswith('fixed') else 'bracket')
    else:
        def members(prefixes):
            prior = str(prefixes['VNINDEX']['date'].iloc[-1])
            planned = bt.add_trading_days(date.fromisoformat(prior), 1).isoformat()
            return timeline.members(prior, planned)

        def targets(prefixes, held):
            active = members(prefixes)
            return momentum_targets({s: f for s, f in prefixes.items() if s in active | {'VNINDEX'}}, held, variant)

        def eligible(symbol, prefixes):
            return symbol in members(prefixes) and (variant != 'reversal_entry' or rh.reversal_entry_allowed(symbol, prefixes))

        result = bt.replay_momentum(frames, configured, start, end, target_fn=targets, entry_gate=eligible,
                                    execution_delay=int(scenario == 'delay_one_session'))
    initial = float(configured['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames, start, end, initial)
    return dict(summary=rh.summarize(result, benchmark, initial, configured['backtest']['slippage_pct'] / 100),
                reconciliation=rh.reconcile(result, frames, initial), replay=result)


def validate_evidence(result, registry):
    for name in registry['blocks']:
        if name not in result['blocks']:
            raise ValueError(f'Missing required block: {name}')
        block = result['blocks'][name]
        for kind in ('mr', 'momentum'):
            for variant in registry[kind]:
                if variant not in block[kind]:
                    raise ValueError(f'Missing trial: {name}/{kind}/{variant}')
                if block[kind][variant]['reconciliation']['max_nav_error_vnd'] > 1e-5:
                    raise ValueError('Accounting mismatch')
        for scenario in registry['scenarios'][1:]:
            if set(block['stresses'].get(scenario, {})) != {'mr_bracket', 'mr_fixed15', 'momentum_baseline'}:
                raise ValueError(f'Missing stress: {name}/{scenario}')
        if 'universe_equal' not in block:
            raise ValueError('Missing universe control')


def run_suite(manifest_path, output):
    registry = load_registry()
    timeline = Timeline(json.loads(TIMELINE.read_text(encoding='utf-8')))
    rules = json.loads(RULES.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled'):
        raise ValueError('Retroactive ML is forbidden')
    manifest = verify_snapshot(manifest_path)
    required = timeline.all_members | {'VNINDEX'}
    if not required <= set(manifest['files']):
        raise ValueError('Missing historical constituent prices')
    if max(b['end'] for b in registry['blocks'].values()) > min(manifest['as_of'], timeline.end):
        raise ValueError('Insufficient snapshot/membership coverage')
    frames = {s: pd.read_csv(manifest_path.parent / manifest['files'][s]['path']) for s in sorted(required)}
    bt.validate_frames(frames, '2022-09-01', '2026-09-29')
    initial = float(rules['backtest']['initial_capital'])
    source_files = sorted(set(Path('stock_agent').rglob('*.py')) | set(Path('scripts').glob('*.py')))
    source_files += [REGISTRY, TIMELINE, RULES, Path('docs/audits/2026-09-30-research-gate-spec.md')]
    result = dict(schema_version=1, status='running', live_eligible=False,
                  generated_at=datetime.now(timezone.utc).isoformat(), registry=registry,
                  manifest=str(manifest_path.resolve()), manifest_sha256=digest(manifest_path),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_hashes={str(p): digest(p) for p in source_files},
                  ignored_snapshot_symbols=sorted(set(manifest['files']) - required), blocks={})
    print('Computing PIT MR signal bank', flush=True)
    bank = mr_signal_bank(frames, rules, timeline)
    (output / 'signals.json').write_text(json.dumps(bank), encoding='utf-8')
    family = len(MR) - 1 + len(MOMENTUM) - 1 + 2
    for name, block in registry['blocks'].items():
        print(f'Running {name}', flush=True)
        start, end = block['start'], block['end']
        clipped = {s: f.loc[f['date'] <= end].reset_index(drop=True) for s, f in frames.items()}
        clipped = {s: f for s, f in clipped.items() if not f.empty}
        benchmark = rh.benchmark_nav(clipped, start, end, initial)
        market_returns = rh.daily_returns(benchmark, initial)
        out = dict(**block, benchmark=bt.metrics(benchmark, [], initial), mr={}, momentum={}, stresses={})
        for kind in ('mr', 'momentum'):
            for variant in registry[kind]:
                trial = run_one(clipped, rules, timeline, bank, start, end, kind, variant)
                nav = trial['replay']['nav']
                controls = exposure_control(nav, market_returns, initial)
                trial['summary']['exposure_matched_index_return_pct'] = (float(np.prod(1 + controls)) - 1) * 100
                if name in ('recent_6m', 'continuous') and variant in ('bracket', 'fixed15', 'baseline'):
                    trial['portfolio_history'] = position_history(trial['replay'], clipped)
                out[kind][variant] = trial
            control = out[kind][registry[kind][0]]
            for variant, trial in out[kind].items():
                if name == 'continuous':
                    returns = rh.daily_returns(trial['replay']['nav'], initial)
                    reference = market_returns if variant == registry[kind][0] else rh.daily_returns(control['replay']['nav'], initial)
                    trial['paired_tests'] = [paired_test(returns - reference, family=family, block=b) for b in registry['bootstrap_blocks']]
                    sample_ok = len(returns) >= registry['min_sessions'] and (kind != 'mr' or trial['summary']['closed_lots'] >= registry['min_mr_closed_trades'])
                    trial['statistical_status'] = ('candidate_development_only' if sample_ok and all(
                        t['ci_family'][0] > 0 and t['p_one_sided_bonferroni'] < registry['alpha'] for t in trial['paired_tests'])
                        else 'inconclusive_or_no_edge')
        for scenario in registry['scenarios'][1:]:
            out['stresses'][scenario] = {f'{kind}_{variant}': run_one(clipped, rules, timeline, bank, start, end, kind, variant, scenario)
                for kind, variant in [('mr', 'bracket'), ('mr', 'fixed15'), ('momentum', 'baseline')]}
        out['universe_equal'] = run_one(clipped, rules, timeline, bank, start, end, 'momentum', 'universe_equal')
        result['blocks'][name] = out
        (output / f'{name}.json').write_text(json.dumps(out, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(json.dumps({k: {v: round(t['summary']['return_pct'], 3) for v, t in out[k].items()} for k in ('mr', 'momentum')}), flush=True)
    validate_evidence(result, registry)
    result['status'] = 'research_complete_live_blocked'
    result['live_blockers'] = ['No untouched forward holdout', 'Corporate-action/reference-price provenance unverified',
                               'No order-book/queue/impact validation', 'Daily dashboard exits differ from monthly momentum replay',
                               'Foreign legacy availability timestamps missing']
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.validate:
        load_registry()
        Timeline(json.loads(TIMELINE.read_text(encoding='utf-8')))
        print('Protocol valid. This is NOT a full data/research pass.')
        return
    if not args.manifest or not args.output:
        parser.error('--manifest and --output required; no silent data skips')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run_suite(args.manifest, args.output)
    except Exception as exc:
        (args.output / 'failed.json').write_text(json.dumps(dict(status='BLOCKED', error=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

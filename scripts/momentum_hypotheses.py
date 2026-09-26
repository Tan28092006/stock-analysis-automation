"""Run the locked, research-only momentum ablations; never touches live state.

See docs/audits/2026-09-26-momentum-hypotheses-protocol.md before interpreting.
Invoke from the repository root with python -m scripts.momentum_hypotheses.
"""
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

from scripts import period_backtest as bt
from stock_agent.data.reconciliation import verify_snapshot

PROTOCOL = Path('docs/audits/2026-09-26-momentum-hypotheses-protocol.md')
VARIANTS = ('baseline', 'market_adjusted', 'reversal_entry', 'own_portfolio_vol')
SCENARIOS = ('normal', 'double_cost', 'delay_one_session')
BLOCKS = {'h2_2025': ('2025-07-01', '2025-12-31'),
          'h1_2026': ('2026-01-01', '2026-06-30'),
          'h2_2026_partial': ('2026-07-01', '2026-09-25')}


def market_adjusted_rank(frames: dict) -> list:
    """Market-model adaptation, not the original monthly FF3 residual strategy."""
    ranked = bt._rank({s: f for s, f in frames.items() if s != 'VNINDEX'})
    market = frames['VNINDEX'].set_index('date')['close']
    result = []
    for symbol, _, vol, close in ranked:
        stock = frames[symbol].set_index('date')['close'].tail(253)
        aligned = market.reindex(stock.index)
        if aligned.isna().any():
            raise ValueError(f'{symbol}: missing aligned benchmark')
        x, y = np.diff(np.log(aligned)), np.diff(np.log(stock))
        variance = float(np.var(x, ddof=1))
        beta = float(np.cov(y, x, ddof=1)[0, 1] / variance) if variance > 1e-16 else 0.0
        residual = (y - beta * x)[:-21]
        deviation = float(residual.std(ddof=1))
        score = float(residual.sum() / deviation) if deviation > 1e-12 else 0.0
        result.append((symbol, score, vol, close))
    return sorted(result, key=lambda row: (-row[1], row[0]))


def portfolio_volatility(frames: dict, weights: dict) -> float:
    """Historical constant-weight portfolio returns include all covariances."""
    if not weights:
        return 0.0
    closes = pd.concat({s: frames[s].set_index('date')['close'] for s in weights}, axis=1)
    returns = closes.pct_change(fill_method=None).dropna().tail(126)
    if len(returns) != 126:
        raise ValueError('Need 126 common daily returns for own-portfolio volatility')
    combined = returns.mul(pd.Series(weights), axis=1).sum(axis=1)
    return float(combined.std(ddof=1) * math.sqrt(252))


def variant_targets(frames: dict, held: set, variant: str) -> tuple[dict, list]:
    if variant in ('baseline', 'reversal_entry'):
        return bt.momentum_targets(frames, held)
    if variant == 'own_portfolio_vol':
        weights, excluded = bt.momentum_targets(frames, held)
        total = sum(weights.values()) or 1.0
        unscaled = {s: w / total for s, w in weights.items()}
        exposure = min(1.0, .2 / max(portfolio_volatility(frames, unscaled), 1e-6))
        return {s: w * exposure for s, w in unscaled.items()}, excluded
    if variant != 'market_adjusted':
        raise ValueError(f'Unknown variant: {variant}')
    ranked = market_adjusted_rank(frames)
    excluded = sorted(set(frames) - {'VNINDEX'} - {r[0] for r in ranked})
    selected = [r for r in ranked[:bt.DEFAULT_TOP_N * bt.BUFFER_MULT]
                if r[0] in held][:bt.DEFAULT_TOP_N]
    for row in ranked:
        if len(selected) >= bt.DEFAULT_TOP_N:
            break
        if row[0] not in {r[0] for r in selected}:
            selected.append(row)
    inv = {s: 1 / max(v, .05) for s, _, v, _ in selected}
    vol = frames['VNINDEX']['close'].pct_change().tail(20).std() * math.sqrt(252)
    vol = float(vol) if np.isfinite(vol) else .2
    exposure = min(1.0, .2 / max(vol, 1e-6))
    total = sum(inv.values()) or 1.0
    return {s: w / total * exposure for s, w in inv.items()}, excluded


def reversal_entry_allowed(symbol: str, frames: dict) -> bool:
    stock = frames[symbol].set_index('date')['close'].tail(6)
    if len(stock) != 6:
        return False
    market = frames['VNINDEX'].set_index('date')['close'].reindex(stock.index)
    if market.isna().any():
        return False
    relative = math.log(stock.iloc[-1] / stock.iloc[0]) - math.log(market.iloc[-1] / market.iloc[0])
    return relative <= 0


def run_variant(frames: dict, rules: dict, start: str, end: str, variant: str, scenario: str) -> dict:
    if variant not in VARIANTS or scenario not in SCENARIOS:
        raise ValueError('Unknown locked variant/scenario')
    configured = copy.deepcopy(rules)
    if scenario == 'double_cost':
        for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
            configured['backtest'][key] *= 2
    return bt.replay_momentum(
        frames, configured, start, end,
        target_fn=lambda prefixes, held: variant_targets(prefixes, held, variant),
        entry_gate=reversal_entry_allowed if variant == 'reversal_entry' else None,
        execution_delay=int(scenario == 'delay_one_session'),
    )


def benchmark_nav(frames: dict, start: str, end: str, initial: float) -> list:
    index = frames['VNINDEX'].query('@start <= date <= @end')
    first_open = float(index.iloc[0]['open'])
    return [dict(date=r.date, nav=initial * float(r.close) / first_open) for r in index.itertuples()]


def daily_returns(nav: list, initial: float) -> np.ndarray:
    values = np.array([initial] + [r['nav'] for r in nav])
    return values[1:] / values[:-1] - 1


def paired_interval(variant: np.ndarray, control: np.ndarray) -> dict:
    """Fixed circular-block paired bootstrap, not a global data-snooping correction."""
    if len(variant) != len(control) or len(control) < 10:
        raise ValueError('Need aligned samples with at least 10 observations')
    delta = np.asarray(variant) - np.asarray(control)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite paired returns')
    n = len(delta)
    rng = np.random.default_rng(20260926)
    starts = rng.integers(0, n, size=(4000, math.ceil(n / 10)))
    indices = ((starts[:, :, None] + np.arange(10)) % n).reshape(4000, -1)[:, :n]
    samples = delta[indices].mean(axis=1) * 252 * 100
    return dict(observations=n, block_length=10, draws=4000, seed=20260926,
                annualized_mean_difference_pct=float(delta.mean() * 252 * 100),
                ci95_pct=np.quantile(samples, [.025, .975]).tolist(),
                ci_three_variants_pct=np.quantile(samples, [.05 / 6, 1 - .05 / 6]).tolist())


def summarize(result: dict, benchmark: list, initial: float, slip: float) -> dict:
    if [r['date'] for r in result['nav']] != [r['date'] for r in benchmark]:
        raise ValueError('Benchmark dates differ from portfolio dates')
    nav = np.array([initial] + [r['nav'] for r in result['nav']])
    index = np.array([initial] + [r['nav'] for r in benchmark])
    strategy_returns, market_returns = nav[1:] / nav[:-1] - 1, index[1:] / index[:-1] - 1
    active = strategy_returns - market_returns
    deviation = float(active.std(ddof=1))
    variance = float(market_returns.var(ddof=1))
    relative = nav / index
    fills = result['fills']
    return dict(
        **result['metrics'],
        net_excess_return_pp=result['metrics']['return_pct'] - (index[-1] / initial - 1) * 100,
        relative_wealth_drawdown_pct=float((relative / np.maximum.accumulate(relative) - 1).min() * 100),
        beta_to_index=float(np.cov(strategy_returns, market_returns)[0, 1] / variance)
        if variance > 1e-16 else None,
        tracking_error_pct=deviation * math.sqrt(252) * 100,
        information_ratio=float(active.mean() / deviation * math.sqrt(252)) if deviation > 1e-16 else None,
        average_exposure_pct=float(np.mean([r['exposure'] / r['nav'] for r in result['nav']]) * 100),
        two_way_turnover_multiple=sum(f['qty'] * f['price'] for f in fills) / float(nav[1:].mean()),
        fees_vnd=sum(f['fee'] for f in fills),
        modeled_slippage_vnd=sum(f['qty'] * f['price'] / (1 + slip if f['side'] == 'BUY' else 1 - slip) * slip
                                 for f in fills),
        buys=sum(f['side'] == 'BUY' for f in fills), sells=sum(f['side'] == 'SELL' for f in fills),
        screened_purchase_attempts=len(result.get('skipped_entries', [])),
    )


def reconcile(result: dict, frames: dict, initial: float) -> dict:
    """Independent cash-flow/quantity ledger, including T+3 cash and stock checks."""
    cash = float(initial)
    positions, pending, lots = {}, [], {}
    prices = {s: f.set_index('date')['close'].to_dict() for s, f in frames.items()}
    fills_by_day = {}
    for fill in result['fills']:
        fills_by_day.setdefault(fill['date'], []).append(fill)
    max_error = 0.0
    for row in result['nav']:
        day = row['date']
        cash += sum(value for due, value in pending if due <= day)
        pending = [(due, value) for due, value in pending if due > day]
        for fill in fills_by_day.get(day, []):
            symbol, qty = fill['symbol'], fill['qty']
            if qty <= 0 or qty % 100:
                raise ValueError('Invalid board lot in reconciliation')
            due = bt.add_trading_days(date.fromisoformat(day), 3).isoformat()
            if fill['side'] == 'BUY':
                cash -= qty * fill['price'] + fill['fee']
                positions[symbol] = positions.get(symbol, 0) + qty
                lots.setdefault(symbol, []).append([due, qty])
            else:
                available = sum(q for available_date, q in lots.get(symbol, []) if available_date <= day)
                if available < qty:
                    raise ValueError('Share settlement violation in reconciliation')
                remaining = qty
                for lot in lots[symbol]:
                    if lot[0] <= day:
                        sold = min(remaining, lot[1])
                        lot[1] -= sold
                        remaining -= sold
                positions[symbol] -= qty
                pending.append((due, qty * fill['price'] - fill['fee']))
            if cash < -1e-5:
                raise ValueError('Negative cash in reconciliation')
        exposure = sum(qty * prices[symbol][day] for symbol, qty in positions.items() if qty)
        receivable = sum(value for _, value in pending)
        errors = [abs(cash + receivable + exposure - row['nav']), abs(cash - row['cash']),
                  abs(exposure - row['exposure']), abs(receivable - row['unsettled'])]
        max_error = max(max_error, *errors)
    if max_error > 1e-5:
        raise ValueError(f'NAV/cash reconciliation failed: {max_error}')
    return dict(max_nav_error_vnd=max_error, sessions=len(result['nav']),
                checks=['cash', 'receivables', 'holdings', 'NAV', '100-share lots', 'T+3 settlement'])


def decision_summary(blocks: dict) -> dict:
    primary = blocks['h1_2026']['scenarios']
    decisions = {}
    for variant in VARIANTS[1:]:
        normal = primary['normal'][variant]
        checks = dict(
            h1_beats_vnindex=normal['summary']['net_excess_return_pp'] > 0,
            h1_beats_control=normal['improvement_over_control_pp'] > 0,
            both_stresses_improve=all(primary[s][variant]['improvement_over_control_pp'] > 0
                                     for s in SCENARIOS[1:]),
            improves_at_least_two_blocks=sum(b['scenarios']['normal'][variant]['improvement_over_control_pp'] > 0
                                             for b in blocks.values()) >= 2,
            primary_adjusted_interval_positive=normal['paired_vs_control']['ci_three_variants_pct'][0] > 0,
        )
        checks = {name: bool(value) for name, value in checks.items()}
        decisions[variant] = dict(checks=checks, promising_conditional_candidate=all(checks.values()),
                                  live_promotion='blocked')
    return decisions


def run_experiments(historical_manifest: Path, recent_manifest: Path, rules_path: Path) -> dict:
    rules = json.loads(rules_path.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled'):
        raise ValueError('Retroactive ML is forbidden in this rule-only experiment')
    initial = float(rules['backtest']['initial_capital'])
    if rules['backtest']['lot_size'] != 100:
        raise ValueError('Locked protocol requires 100-share lots')
    snapshots = {}
    for path in dict.fromkeys((historical_manifest, recent_manifest)):
        manifest = verify_snapshot(path)
        frames = {s: pd.read_csv(path.parent / f['path']) for s, f in manifest['files'].items()}
        snapshots[path] = (manifest, frames)
    blocks = {}
    for block, (start, end) in BLOCKS.items():
        path = recent_manifest if block == 'h2_2026_partial' else historical_manifest
        manifest, full_frames = snapshots[path]
        if manifest['as_of'] < end:
            raise ValueError(f'{block}: snapshot ends before evaluation period')
        frames = {s: f.loc[f['date'] <= end].reset_index(drop=True) for s, f in full_frames.items()
                  if (f['date'] <= end).any()}
        bt.validate_frames(frames, start, end)
        benchmark = benchmark_nav(frames, start, end, initial)
        scenarios = {}
        for scenario in SCENARIOS:
            variants = {}
            for variant in VARIANTS:
                replay = run_variant(frames, rules, start, end, variant, scenario)
                slip = rules['backtest']['slippage_pct'] / 100 * (2 if scenario == 'double_cost' else 1)
                variants[variant] = dict(
                    summary=summarize(replay, benchmark, initial, slip),
                    reconciliation=reconcile(replay, frames, initial), replay=replay,
                )
                if variant != 'baseline':
                    control = variants['baseline']
                    variants[variant]['improvement_over_control_pp'] = (
                        replay['metrics']['return_pct'] - control['summary']['return_pct'])
                    variants[variant]['paired_vs_control'] = paired_interval(
                        daily_returns(replay['nav'], initial), daily_returns(control['replay']['nav'], initial))
            scenarios[scenario] = variants
        blocks[block] = dict(start=start, end=end, sessions=len(benchmark),
                             snapshot_manifest=str(path.resolve()), snapshot_sha256=sha256(path),
                             universe=sorted(set(frames) - {'VNINDEX'}),
                             benchmark=dict(metrics=bt.metrics(benchmark, [], initial), nav=benchmark), scenarios=scenarios)
    source_files = [Path(__file__), Path(bt.__file__), Path('stock_agent/features/momentum_scan.py'),
                    Path('stock_agent/data/exchange_calendar.py'), Path('stock_agent/data/reconciliation.py'), rules_path]
    return dict(
        schema_version=1, kind='locked_fixed_basket_hypothesis_replay',
        generated_at=datetime.now(timezone.utc).isoformat(),
        code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_and_config_hashes={str(p): sha256(p) for p in source_files},
        protocol=str(PROTOCOL.resolve()), protocol_sha256=sha256(PROTOCOL), rules=rules,
        trial_registry=dict(variants=list(VARIANTS), scenarios=list(SCENARIOS), blocks=list(BLOCKS),
                            total_replays=36, primary_comparisons=3, parameter_search=False),
        blocks=blocks, decisions=decision_summary(blocks), live_promotion='blocked',
        limitations=[
            'Current fixed universe, not point-in-time membership; survivorship/selection bias remains.',
            'Provider-adjusted data retrieved now; corporate-action lineage/dividend cash ledger unverified.',
            'All blocks are development/robustness replays, NOT untouched out-of-sample tests.',
            'Paper-inspired adaptations, not exact original-paper replications.',
            'Daily bars cannot certify queue fills, exact T+2 timing, or nonlinear market impact.',
            'Gross price-only VNINDEX comparator, no investable total-return comparator.',
            'Bootstrap diagnostics do not correct unknown prior searches or prove persistent alpha.',
            'Blocks have separate cash starts and different data vintages; do not concatenate returns.',
        ],
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-manifest', type=Path, required=True)
    parser.add_argument('--recent-manifest', type=Path, required=True)
    parser.add_argument('--rules', type=Path, default=Path('configs/rules_mr.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    result = run_experiments(args.historical_manifest, args.recent_manifest, args.rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result['decisions'], indent=2))


if __name__ == '__main__':
    main()

"""Isolated, rule-only historical portfolio replay; never touches live state.

Daily OHLC cannot identify T+2 afternoon execution. Use conservative T+3
availability for shares and sale proceeds, with no cash advance or margin.
Current fixed-universe/provider-adjusted replay is NOT an untouched OOS test.
Run from repository root: python -m scripts.period_backtest --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.data.exchange_calendar import (
    add_trading_days, symbol_trading_days_between, trading_days_between,
)
from stock_agent.data.reconciliation import verify_snapshot
from stock_agent.features.momentum_scan import (
    BUFFER_MULT, DEFAULT_TOP_N, TARGET_VOL, _rank,
)
from stock_agent.features.position_manager import money_cfg
from stock_agent.features.signal_engine import prepare_signal_frame, score_precomputed_at


def validate_frames(frames: dict, start: str, end: str) -> None:
    """Fail closed on malformed, stale, or internally missing observed history."""
    if start > end or 'VNINDEX' not in frames:
        raise ValueError('Invalid period or missing VNINDEX')
    for symbol, frame in frames.items():
        if frame.empty or frame['date'].duplicated().any() or not frame['date'].is_monotonic_increasing:
            raise ValueError(f'{symbol}: empty, duplicate, or unsorted history')
        numeric = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
        if not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any():
            raise ValueError(f'{symbol}: invalid OHLCV')
        if ((frame['high'] < frame[['open', 'close', 'low']].max(axis=1)).any()
                or (frame['low'] > frame[['open', 'close', 'high']].min(axis=1)).any()):
            raise ValueError(f'{symbol}: inconsistent OHLC')
        first = date.fromisoformat(str(frame['date'].iloc[0]))
        expected = {d.isoformat() for d in symbol_trading_days_between(symbol, first, date.fromisoformat(end))}
        observed = set(frame.loc[frame['date'] <= end, 'date'])
        if expected != observed:
            raise ValueError(f'{symbol}: calendar gaps/extra dates: {sorted(expected ^ observed)[:8]}')
    expected_period = {d.isoformat() for d in trading_days_between(date.fromisoformat(start), date.fromisoformat(end))}
    if not expected_period or not expected_period.issubset(set(frames['VNINDEX']['date'])):
        raise ValueError('Incomplete benchmark period')


def mr_signals(frames: dict, rules: dict) -> dict:
    """Same BUY_SETUP scorer/large-cap override as the current MR scan, no ML."""
    selected = {**rules, 'mean_reversion': {
        **rules['mean_reversion'], **rules.get('vn30_mean_reversion', {}),
    }}
    signals: dict[str, list] = {}
    for symbol, frame in sorted(frames.items()):
        if symbol == 'VNINDEX':
            continue
        prepared = prepare_signal_frame(frame.copy(), selected)
        for i in range(max(1, int(rules.get('min_history_rows', 90)) - 1), len(prepared)):
            # Give the scorer a prefix too: auxiliary snapshots cannot see later bars.
            sig = score_precomputed_at(symbol, prepared.iloc[:i + 1], i, selected)
            if sig.decision == 'BUY_SETUP' and sig.risk_plan:
                plan = sig.risk_plan
                signals.setdefault(str(frame.iloc[i]['date']), []).append({
                    'symbol': symbol, 'stop': plan.stop_loss, 'target': plan.take_profit_1,
                    'hold': plan.holding_period_days, 'rr': plan.reward_risk,
                })
    return {day: sorted(orders, key=lambda o: (-o['rr'], o['symbol']))
            for day, orders in signals.items()}


def momentum_targets(frames: dict, held: set) -> tuple[dict, list]:
    """12-1/inverse-vol/top-20 buffer, fixed production N=10 and vol target."""
    stocks = {s: f for s, f in frames.items() if s != 'VNINDEX'}
    ranked = _rank(stocks)
    excluded = sorted(set(stocks) - {r[0] for r in ranked})
    retained = [r for r in ranked[:DEFAULT_TOP_N * BUFFER_MULT] if r[0] in held]
    selected = retained[:DEFAULT_TOP_N]
    for row in ranked:
        if len(selected) >= DEFAULT_TOP_N:
            break
        if row[0] not in {r[0] for r in selected}:
            selected.append(row)
    vol = frames['VNINDEX']['close'].pct_change().tail(20).std() * math.sqrt(252)
    vol = float(vol) if np.isfinite(vol) else TARGET_VOL
    exposure = min(1.0, TARGET_VOL / max(vol, 1e-6))
    inv = {s: 1.0 / max(v, .05) for s, _, v, _ in selected}
    total = sum(inv.values()) or 1.0
    return {s: v / total * exposure for s, v in inv.items()}, excluded


class Broker:
    """FIFO lots, fees, finite cash, delayed settlements, auditable cash flows."""

    def __init__(self, rules: dict):
        cfg = rules['backtest']
        self.initial = float(cfg['initial_capital'])
        self.cash = self.initial
        self.buy_fee = float(cfg['commission_pct']) / 100
        self.sell_fee = (float(cfg['commission_pct']) + float(cfg['sell_tax_pct'])) / 100
        self.slip = float(cfg['slippage_pct']) / 100
        self.lot = int(cfg['lot_size'])
        self.positions: dict[str, list] = {}
        self.receivables: list[dict] = []
        self.fills: list[dict] = []
        self.closed: list[dict] = []

    def round_qty(self, value: float) -> int:
        return int(max(0.0, value) // self.lot) * self.lot

    def quantity(self, symbol: str) -> int:
        return sum(p['qty'] for p in self.positions.get(symbol, []))

    def settle(self, day: str) -> None:
        self.cash += sum(r['amount'] for r in self.receivables if r['due'] <= day)
        self.receivables = [r for r in self.receivables if r['due'] > day]

    def buy(self, symbol: str, qty: int, raw_price: float, day: str, **metadata) -> int:
        px = raw_price * (1 + self.slip)
        qty = min(self.round_qty(qty), self.round_qty(self.cash / (px * (1 + self.buy_fee))))
        if qty == 0:
            return 0
        cost = qty * px * (1 + self.buy_fee)
        self.cash -= cost
        self.positions.setdefault(symbol, []).append({
            'qty': qty, 'cost': cost, 'entry': px, 'entry_date': day,
            'available': add_trading_days(date.fromisoformat(day), 3).isoformat(), **metadata,
        })
        self.fills.append(dict(date=day, symbol=symbol, side='BUY', qty=qty,
                               price=px, fee=qty * px * self.buy_fee, reason='entry'))
        return qty

    def sell(self, symbol: str, qty: int, raw_price: float, day: str, reason: str) -> int:
        px = raw_price * (1 - self.slip)
        remaining = qty
        for position in self.positions.get(symbol, []):
            if remaining <= 0 or position['available'] > day:
                continue
            sold = min(position['qty'], remaining)
            cost = position['cost'] * sold / position['qty']
            proceeds = sold * px * (1 - self.sell_fee)
            position['qty'] -= sold
            position['cost'] -= cost
            remaining -= sold
            self.receivables.append({
                'due': add_trading_days(date.fromisoformat(day), 3).isoformat(), 'amount': proceeds,
            })
            self.closed.append(dict(symbol=symbol, entry_date=position['entry_date'],
                                    exit_date=day, qty=sold, cost=cost, pnl=proceeds - cost, reason=reason))
        sold = qty - remaining
        if sold:
            self.fills.append(dict(date=day, symbol=symbol, side='SELL', qty=sold,
                                   price=px, fee=sold * px * self.sell_fee, reason=reason))
        if symbol in self.positions:
            self.positions[symbol] = [p for p in self.positions[symbol] if p['qty']]
            if not self.positions[symbol]:
                del self.positions[symbol]
        return sold

    def mark(self, bars: dict, day: str) -> dict:
        exposure = sum(self.quantity(s) * float(bars[s]['close']) for s in self.positions)
        unsettled = sum(r['amount'] for r in self.receivables)
        return dict(date=day, nav=self.cash + unsettled + exposure, cash=self.cash,
                    unsettled=unsettled, exposure=exposure, positions=len(self.positions))


def metrics(nav: list, closed: list, initial: float) -> dict:
    values = np.array([initial] + [r['nav'] for r in nav], dtype=float)
    changes = values[1:] / values[:-1] - 1
    monthly_ends = {r['date'][:7]: r['nav'] for r in nav}
    monthly = {}
    previous = initial
    for month, value in monthly_ends.items():
        monthly[month] = (value / previous - 1) * 100
        previous = value
    return dict(
        final_nav=values[-1], pnl=values[-1] - initial,
        return_pct=(values[-1] / initial - 1) * 100,
        max_drawdown_pct=float((values / np.maximum.accumulate(values) - 1).min() * 100),
        sharpe_daily=float(changes.mean() / changes.std(ddof=1) * math.sqrt(252))
        if len(changes) > 1 and changes.std(ddof=1) > 0 else None,
        closed_lots=len(closed),
        win_rate_pct=sum(r['pnl'] > 0 for r in closed) / len(closed) * 100 if closed else None,
        realized_pnl=sum(r['pnl'] for r in closed), monthly_pct=monthly,
    )


def _inputs(frames: dict, start: str, end: str) -> tuple:
    # Exclude the future before computing any feature, rank, price map or calendar.
    frames = {s: f.loc[f['date'] <= end].reset_index(drop=True).copy() for s, f in frames.items()}
    index_dates = frames['VNINDEX']['date'].tolist()
    days = [d for d in index_dates if start <= d <= end]
    if not days or index_dates.index(days[0]) == 0:
        raise ValueError('Need period sessions and a prior close')
    previous = {d: index_dates[i - 1] for i, d in enumerate(index_dates) if i}
    rows = {s: f.set_index('date').to_dict('index') for s, f in frames.items()}
    return frames, days, previous, rows


def _fillable(bar: dict, previous_close: float, side: str, limit_pct: float) -> bool:
    # Conservative opening-price gate; NEVER inspect the later high/low/volume.
    # Adjusted previous close is only a proxy for official reference price.
    change = float(bar['open']) / previous_close - 1
    return not ((side == 'BUY' and change >= limit_pct / 100)
                or (side == 'SELL' and change <= -limit_pct / 100))


def _result(broker: Broker, nav: list, **extra) -> dict:
    return dict(metrics=metrics(nav, broker.closed, broker.initial), nav=nav,
                fills=broker.fills, closed_lots=broker.closed, open_positions=broker.positions,
                receivables=broker.receivables, **extra)


def replay_mr(frames: dict, rules: dict, start: str, end: str, signals: dict | None = None) -> dict:
    frames, days, previous, rows = _inputs(frames, start, end)
    signals = mr_signals(frames, rules) if signals is None else signals
    broker = Broker(rules)
    cfg = money_cfg(rules)
    nav = []
    skipped = []
    prior_nav = broker.initial
    for day in days:
        broker.settle(day)
        prior = previous[day]
        bars = {s: data[day] for s, data in rows.items() if day in data}
        # Buy FIRST: today's high/low/close and intraday sale proceeds are unavailable.
        for signal in signals.get(prior, []):
            symbol = signal['symbol']
            if symbol in broker.positions or len(broker.positions) >= cfg['max_positions']:
                continue
            if symbol not in bars or prior not in rows[symbol]:
                continue
            bar = bars[symbol]
            px = float(bar['open']) * (1 + broker.slip)
            if not signal['stop'] < px < signal['target'] or not _fillable(
                    bar, rows[symbol][prior]['close'], 'BUY', rules['backtest']['price_limit_pct']):
                skipped.append(dict(date=day, symbol=symbol, reason='entry_outside_plan_or_unfillable'))
                continue
            exposure = sum(broker.quantity(s) * float(bars[s]['open']) for s in broker.positions)
            risk_qty = prior_nav * cfg['risk_per_trade_pct'] / 100 / (px - signal['stop'])
            weight_qty = prior_nav * cfg['max_weight_pct'] / 100 / px
            exposure_qty = (prior_nav * cfg['max_exposure_pct'] / 100 - exposure) / px
            qty = broker.round_qty(min(risk_qty, weight_qty, exposure_qty))
            broker.buy(symbol, qty, float(bar['open']), day, signal_date=prior,
                       stop=signal['stop'], target=signal['target'],
                       expiry=add_trading_days(date.fromisoformat(day), signal['hold']).isoformat())
        for symbol in list(broker.positions):
            position = broker.positions[symbol][0]
            bar = bars[symbol]
            if position['available'] > day or not _fillable(
                    bar, rows[symbol][prior]['close'], 'SELL', rules['backtest']['price_limit_pct']):
                continue
            raw_px, reason = None, None
            if float(bar['low']) <= position['stop']:
                raw_px, reason = min(position['stop'], float(bar['open'])), 'stop'
            elif float(bar['high']) >= position['target']:
                raw_px, reason = position['target'], 'target'
            elif day >= position['expiry']:
                raw_px, reason = float(bar['close']), 'time'
            if raw_px is not None:
                broker.sell(symbol, broker.quantity(symbol), raw_px, day, reason)
        nav.append(broker.mark(bars, day))
        prior_nav = nav[-1]['nav']
    used_dates = {previous[d] for d in days}
    return _result(broker, nav, skipped=skipped, signal_count=sum(len(v) for d, v in signals.items() if d in used_dates))


def replay_momentum(frames: dict, rules: dict, start: str, end: str) -> dict:
    frames, days, previous, rows = _inputs(frames, start, end)
    broker = Broker(rules)
    nav, rebalances = [], []
    targets: dict[str, int] = {}
    retry_until = ''
    prior_nav = broker.initial
    for i, day in enumerate(days):
        broker.settle(day)
        prior = previous[day]
        bars = {s: data[day] for s, data in rows.items() if day in data}
        if i == 0 or day[:7] != days[i - 1][:7]:
            prefixes = {s: f.loc[f['date'] <= prior].copy() for s, f in frames.items()
                        if prior in rows[s]}
            weights, excluded = momentum_targets(prefixes, set(broker.positions))
            targets = {s: broker.round_qty(prior_nav * w / float(rows[s][prior]['close']))
                       for s, w in weights.items()}
            retry_until = add_trading_days(date.fromisoformat(day), 3).isoformat()
            rebalances.append(dict(signal_date=prior, first_execution=day, weights=weights,
                                   targets=targets.copy(), ineligible=excluded))
        if day <= retry_until:
            for symbol in list(broker.positions):
                surplus = broker.quantity(symbol) - targets.get(symbol, 0)
                if surplus > 0 and _fillable(bars[symbol], rows[symbol][prior]['close'],
                                            'SELL', rules['backtest']['price_limit_pct']):
                    broker.sell(symbol, surplus, float(bars[symbol]['open']), day, 'rebalance')
            needed = {s: max(0, qty - broker.quantity(s)) for s, qty in targets.items()
                      if s in bars and _fillable(bars[s], rows[s][prior]['close'],
                                                'BUY', rules['backtest']['price_limit_pct'])}
            cost = sum(qty * float(bars[s]['open']) * (1 + broker.slip) * (1 + broker.buy_fee)
                       for s, qty in needed.items())
            scale = min(1.0, broker.cash / cost) if cost else 0.0
            for symbol, qty in needed.items():
                broker.buy(symbol, broker.round_qty(qty * scale), float(bars[symbol]['open']), day)
        nav.append(broker.mark(bars, day))
        prior_nav = nav[-1]['nav']
    return _result(broker, nav, rebalances=rebalances)


def run_snapshot(manifest_path: Path, rules_path: Path, start: str, end: str) -> dict:
    manifest = verify_snapshot(manifest_path)
    frames = {s: pd.read_csv(manifest_path.parent / item['path'])
              for s, item in manifest['files'].items()}
    frames = {s: f.loc[f['date'] <= end].reset_index(drop=True) for s, f in frames.items()}
    validate_frames(frames, start, end)
    rules = json.loads(rules_path.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled'):
        raise ValueError('This replay supports the rule-only configuration, not retroactive ML')
    mr = replay_mr(frames, rules, start, end)
    momentum = replay_momentum(frames, rules, start, end)
    idx = frames['VNINDEX'].query('@start <= date <= @end')
    initial = rules['backtest']['initial_capital']
    first_open = float(idx.iloc[0]['open'])
    benchmark = [dict(date=r.date, nav=initial * float(r.close) / first_open)
                 for r in idx.itertuples()]
    source_files = [Path(__file__), Path('stock_agent/features/signal_engine.py'),
                    Path('stock_agent/features/indicators.py'), Path('stock_agent/features/momentum_scan.py'),
                    Path('stock_agent/data/exchange_calendar.py'), rules_path]
    return dict(
        schema_version=1, kind='fixed_current_basket_rule_only_replay', start=start, end=end,
        generated_at=datetime.now(timezone.utc).isoformat(),
        code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_and_config_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        snapshot_manifest=str(manifest_path.resolve()),
        snapshot_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        universe=sorted(set(frames) - {'VNINDEX'}), rules=rules,
        assumptions=dict(ml=False, retrained=False, tuned_on_period=False,
                         settlement='T+3 daily conservative; no advance or margin',
                         execution='prior-close orders; next open; MR intraday stop before target',
                         momentum_rebalance='first session monthly; prior-close fixed quantities; T+3 cash retry',
                         ending_positions='marked at final close, NOT forcibly sold',
                         mr_caps='money config: 4 positions, 25% each, 60% gross, 1.5% risk',
                         benchmark='VNINDEX first session OPEN to final CLOSE, price-only gross, not investable'),
        limitations=[
            'Current fixed universe, not historical point-in-time membership: selection/survivorship bias remains.',
            'Rules previously researched on historical data; this is not untouched out-of-sample performance.',
            manifest.get('adjustment_basis', 'provider adjustment lineage unverified'),
            'Daily OHLC cannot prove intraday path, queue priority, fill liquidity, or exact T+2 afternoon availability.',
            'No dividends/corporate-action cash ledger; no market impact model; open holdings have no hypothetical exit fees.',
        ],
        mr=mr, momentum=momentum, benchmark=dict(metrics=metrics(benchmark, [], initial), nav=benchmark),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--rules', type=Path, default=Path('configs/rules_mr.json'))
    parser.add_argument('--start', default='2026-01-01')
    parser.add_argument('--end', default='2026-06-30')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    result = run_snapshot(args.manifest, args.rules, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k]['metrics'] for k in ('mr', 'momentum', 'benchmark')}, indent=2))


if __name__ == '__main__':
    main()

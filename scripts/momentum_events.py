"""Preregistered close-confirmed event momentum. Research only; no live state."""
from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import period_backtest as bt
from scripts import research_gate as gate
from scripts.historical_universe import Timeline

PROTOCOL = Path('configs/research/momentum_events_v1.json')
VARIANTS = ('daily55', 'weekly55', 'daily20', 'pyramiding55', 'market55', 'volume55')
SCENARIOS = ('normal', 'double_cost', 'delay_one_session')


def load_protocol(path: Path = PROTOCOL) -> dict:
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = dict(schema_version=1, hypothesis_id='momentum_events_v1_20260930',
        holdout=False, live_eligible=False, variants=list(VARIANTS), scenarios=list(SCENARIOS),
        comparators={v: 'daily55' for v in VARIANTS[1:]}, atr_sessions=20, exit_sessions=20,
        risk_per_unit_pct=.5, max_positions=10, max_weight_pct=20, max_exposure_pct=100,
        max_stop_risk_pct=6, stop_atr=2, add_atr=.5, max_units=4, max_chase_atr=1,
        adv_sessions=20, participation_pct=1, volume_multiple=1.5, market_sma=200,
        monthly_activity_floor=None, bootstrap_blocks=[20, 40], family=184,
        min_sessions=252, seed=20260930, alpha=.05,
        snapshots=dict(VN30='21b2deff6694a2a7218c861a681524b91f311617fc1e84de0598fc8001e155d0',
                       H1='c9d046ff5d1c336284c2fc8a1bd07abc2e63c84d2493ce508a8d16bf132d7148'))
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Event protocol changed; explicitly version before evaluating')
    if not any(r['id'] == p['hypothesis_id'] and r['protocol'] == PROTOCOL.as_posix()
               for r in gate.load_registry().get('event_hypotheses', [])):
        raise ValueError('Event hypothesis not registered')
    return p


def features(frame: pd.DataFrame, entry_sessions: int) -> pd.DataFrame:
    """Only backward-looking features; N seeded with SMA, then alpha=1/20."""
    if entry_sessions not in (20, 55):
        raise ValueError('Unregistered entry channel')
    f = frame.reset_index(drop=True).copy()
    tr = pd.concat([f.high - f.low, (f.high - f.close.shift()).abs(),
                    (f.low - f.close.shift()).abs()], axis=1).max(axis=1).to_numpy(float)
    n = np.full(len(f), np.nan)
    if len(f) >= 20:
        n[19] = tr[:20].mean()
        for i in range(20, len(f)):
            n[i] = (19 * n[i - 1] + tr[i]) / 20
    f['atr'] = n
    f['entry_high'] = f.high.shift().rolling(entry_sessions).max()
    f['exit_low'] = f.low.shift().rolling(20).min()
    f['prior_volume'] = f.volume.shift().rolling(20).mean()
    f['adv'] = f.volume.rolling(20).mean()
    return f


def activity(result: dict) -> dict:
    months = {r['date'][:7]: dict(new_positions=0, add_on_buys=0, entry_days=0)
              for r in result['nav']}
    quantities, entry_days = {}, set()
    for fill in result['fills']:
        symbol, day = fill['symbol'], fill['date']
        held = quantities.get(symbol, 0)
        if fill['side'] == 'BUY':
            months[day[:7]]['new_positions' if held == 0 else 'add_on_buys'] += 1
            if held == 0:
                entry_days.add(day)
        quantities[symbol] = held + fill['qty'] * (1 if fill['side'] == 'BUY' else -1)
    quiet, longest = 0, 0
    for row in result['nav']:
        day = row['date']
        quiet = 0 if day in entry_days else quiet + 1
        longest = max(longest, quiet)
    for month, stats in months.items():
        stats['entry_days'] = sum(day.startswith(month) for day in entry_days)
    first = result['nav'][0]['date']
    return dict(months=months, distinct_entry_days=len(entry_days),
        first_session_new_positions=sum(f['side'] == 'BUY' and f['date'] == first
                                        and f['intent_kind'] == 'new' for f in result['fills']),
        zero_new_entry_months=[m for m, v in months.items() if not v['new_positions']],
        longest_quiet_sessions=longest, numerical_activity_pass=None)


def _buy_quantity(broker, plan, campaigns, bars, prior_bars, prior_nav, px, p):
    """Execution can shrink frozen intent; no current close/high/low/volume."""
    s = plan['symbol']
    stop = plan['stop']
    if plan['intent_kind'] == 'add':
        stop = max(campaigns[s]['stop'], px - p['stop_atr'] * campaigns[s]['n'])
    exposure = sum(broker.quantity(k) * bars[k]['open'] for k in broker.positions)
    risk = sum(broker.quantity(k) * max(0, max(bars[k]['open'], prior_bars[k]['close'])
                - (stop if k == s else campaigns[k]['stop'])) for k in broker.positions)
    capacities = dict(
        planned=plan['planned_qty'],
        liquidity=min(plan['adv_cap_qty'], broker.round_qty(prior_bars[s]['adv'] * p['participation_pct'] / 100)),
        cash=broker.cash / (px * (1 + broker.buy_fee)),
        weight=(prior_nav * p['max_weight_pct'] / 100 - broker.quantity(s) * bars[s]['open']) / px,
        exposure=(prior_nav * p['max_exposure_pct'] / 100 - exposure) / px,
        unit_risk=plan['risk_budget'] / (px - stop),
        portfolio_risk=(prior_nav * p['max_stop_risk_pct'] / 100 - risk) / (px - stop))
    binding = min(capacities, key=capacities.get)
    return broker.round_qty(capacities[binding]), stop, binding, capacities['liquidity']


def replay(frames: dict, rules: dict, timeline: Timeline, start: str, end: str,
           variant: str = 'daily55', scenario: str = 'normal') -> dict:
    p = load_protocol()
    if variant not in VARIANTS or scenario not in SCENARIOS:
        raise ValueError('Unregistered event variant/scenario')
    if rules.get('ml', {}).get('enabled') or rules['backtest']['lot_size'] != 100:
        raise ValueError('Requires ML off and 100-share lots')
    required = timeline.all_members | {'VNINDEX'}
    if not required <= set(frames):
        raise ValueError('Missing PIT union price history')
    frames, days, previous, _ = bt._inputs({s: frames[s] for s in sorted(required)}, start, end)
    # A later IPO in the historical union must not invalidate an earlier window.
    # Its membership is still evaluated; unavailable pre-listing history cannot buy.
    frames = {s: f for s, f in frames.items() if not f.empty}
    bt.validate_frames(frames, start, end)
    rows = {s: features(f, 20 if variant == 'daily20' else 55).set_index('date').to_dict('index')
            for s, f in frames.items()}
    market_sma = frames['VNINDEX'].set_index('date').close.rolling(p['market_sma']).mean().to_dict()
    configured = copy.deepcopy(rules)
    if scenario == 'double_cost':
        for key in ('commission_pct', 'sell_tax_pct', 'slippage_pct'):
            configured['backtest'][key] *= 2
    broker = bt.Broker(configured)
    delay = int(scenario == 'delay_one_session')
    limit = rules['backtest']['price_limit_pct']
    campaigns, pending, exits = {}, {}, {}
    funnel, decisions, nav, skips = Counter(), [], [], []
    prior_nav, seen_week = broker.initial, None

    def skip(day, symbol, reason):
        funnel[reason] += 1
        skips.append(dict(date=day, symbol=symbol, reason=reason))

    for i, day in enumerate(days):
        broker.settle(day)
        prior = previous[day]
        bars = {s: r[day] for s, r in rows.items() if day in r}
        past = {s: r[prior] for s, r in rows.items() if prior in r}
        members = timeline.members(prior, day)
        week = date.fromisoformat(day).isocalendar()[:2]
        buy_day = variant != 'weekly55' or week != seen_week
        seen_week = week

        # Exit decisions use only information from the last completed session.
        for s in sorted(broker.positions):
            if s not in past or s not in bars:
                raise ValueError(f'Missing held-symbol session: {s}/{day}')
            if s not in exits:
                reason = ('membership_exit' if s not in members else
                          'close_stop' if past[s]['close'] <= campaigns[s]['stop'] else
                          'channel_exit' if past[s]['close'] < past[s]['exit_low'] else None)
                if reason:
                    exits[s] = dict(symbol=s, signal_date=prior, due=i + delay, reason=reason)
                    decisions.append(dict(side='SELL', **exits[s]))
            if s in exits and s in pending:
                pending.pop(s)
                skip(day, s, 'cancelled_by_exit')
        exiting_today = set(exits)

        if buy_day:
            for s in sorted(members):
                if s not in past or s not in bars:
                    skip(day, s, 'history_unavailable')
                    continue
                f = past[s]
                if not all(np.isfinite(f[k]) for k in ('atr', 'entry_high', 'exit_low', 'adv')) or f['atr'] <= 0:
                    skip(day, s, 'warmup')
                    continue
                held = s in campaigns
                if held:
                    if variant != 'pyramiding55' or campaigns[s]['units'] >= p['max_units']:
                        continue
                    n = campaigns[s]['n']
                    trigger = campaigns[s]['last_fill'] + p['add_atr'] * n
                    if f['close'] < trigger:
                        continue
                    kind, stop = 'add', campaigns[s]['stop']
                else:
                    if f['close'] <= f['entry_high']:
                        continue
                    n, trigger = f['atr'], f['entry_high']
                    kind, stop = 'new', f['close'] - p['stop_atr'] * f['atr']
                funnel['candidate_' + kind] += 1
                if s in exits or s in pending:
                    skip(day, s, 'exit_pending' if s in exits else 'buy_pending')
                    continue
                if variant == 'market55' and not past['VNINDEX']['close'] > market_sma[prior]:
                    skip(day, s, 'market_gate')
                    continue
                if variant == 'volume55' and not f['volume'] >= p['volume_multiple'] * f['prior_volume']:
                    skip(day, s, 'volume_gate')
                    continue
                budget = prior_nav * p['risk_per_unit_pct'] / 100
                qty = broker.round_qty(budget / (p['stop_atr'] * n))
                if held:
                    qty = min(qty, campaigns[s]['first_qty'])
                plan = dict(symbol=s, signal_date=prior, due=i + delay, intent_kind=kind,
                    n=n, stop=stop, trigger=trigger, signal_close=f['close'],
                    priority=(f['close'] - trigger) / n, planned_qty=qty, risk_budget=budget,
                    adv_cap_qty=broker.round_qty(f['adv'] * p['participation_pct'] / 100))
                pending[s] = plan
                decisions.append(dict(side='BUY', **plan))

        # Sells release holdings, never same-day spendable cash (shared T+3 broker).
        for s, intent in sorted(list(exits.items())):
            if intent['due'] > i:
                continue
            if not bt._fillable(bars[s], past[s]['close'], 'SELL', limit):
                skip(day, s, 'sell_limit')
                continue
            cap = broker.round_qty(past[s]['adv'] * p['participation_pct'] / 100)
            available = sum(lot['qty'] for lot in broker.positions[s] if lot['available'] <= day)
            qty = min(broker.quantity(s), available, cap)
            if not qty:
                skip(day, s, 'sell_settlement' if not available else 'sell_liquidity')
                continue
            broker.sell(s, qty, bars[s]['open'], day, intent['reason'])
            broker.fills[-1].update(signal_date=intent['signal_date'], adv_cap_qty=cap)
            if not broker.quantity(s):
                del exits[s], campaigns[s]

        due = sorted((v for v in pending.values() if v['due'] <= i),
                     key=lambda v: (v['intent_kind'] != 'new', -v['priority'], v['symbol']))
        for plan in due:
            s = plan['symbol']
            del pending[s]  # One attempt, including a failed or partial fill.
            if s not in members:
                skip(day, s, 'membership')
                continue
            if s in exiting_today or (plan['intent_kind'] == 'add' and s not in campaigns):
                skip(day, s, 'exit_pending')
                continue
            if s not in bars or s not in past:
                raise ValueError(f'Missing intent execution session: {s}/{day}')
            if plan['intent_kind'] == 'new' and len(broker.positions) >= p['max_positions']:
                skip(day, s, 'position_limit')
                continue
            px = bars[s]['open'] * (1 + broker.slip)
            if (px <= plan['stop'] or px < plan['trigger']
                    or px > plan['signal_close'] + p['max_chase_atr'] * plan['n']):
                skip(day, s, 'gap_outside_plan')
                continue
            if not bt._fillable(bars[s], past[s]['close'], 'BUY', limit):
                skip(day, s, 'buy_limit')
                continue
            qty, stop, binding, cap = _buy_quantity(broker, plan, campaigns, bars, past, prior_nav, px, p)
            if not qty:
                skip(day, s, binding)
                continue
            actual = broker.buy(s, qty, bars[s]['open'], day)
            broker.fills[-1].update(signal_date=plan['signal_date'], intent_kind=plan['intent_kind'],
                planned_qty=plan['planned_qty'], adv_cap_qty=cap, nominal_stop=stop, binding_cap=binding)
            if s not in campaigns:
                campaigns[s] = dict(n=plan['n'], units=0, first_qty=actual)
            campaigns[s].update(units=campaigns[s]['units'] + 1, stop=stop, last_fill=px)
            funnel['filled_' + plan['intent_kind']] += 1
        nav.append(broker.mark(bars, day))
        prior_nav = nav[-1]['nav']
    result = bt._result(broker, nav, decisions=decisions, skipped_entries=skips,
                        pending_buys=pending, pending_exits=exits, campaigns=campaigns,
                        funnel=dict(sorted(funnel.items())), live_eligible=False)
    result['activity'] = activity(result)
    return result

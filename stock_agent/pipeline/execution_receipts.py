"""Validate declared order/fill evidence; no broker connection or fill simulation.

Hashes bind supplied bytes/content, not their truth or broker authenticity. The
single-plan resource check is not a portfolio-wide reservation engine.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stock_agent.data.exchange_calendar import is_trading_day

VN = timezone(timedelta(hours=7))
PLAN_FIELDS = set('schema_version strategy_id signal_id symbol side order_type quantity '
    'limit_price_vnd lot_size price_basis source_sha256 signal_available_at created_at '
    'valid_from valid_until market_reference resources'.split())
REFERENCE_FIELDS = set('session_date reference_price_vnd floor_price_vnd ceiling_price_vnd '
    'tick_size_vnd observed_at source_sha256'.split())
RESOURCE_FIELDS = set('as_of cash_available_vnd shares_available buy_fee_reserve_pct'.split())
EVENT_FIELDS = set('event_id sequence order_id plan_sha256 origin kind occurred_at '
    'observed_at source_sha256'.split())
KINDS = {'ACCEPTED', 'FILL', 'CANCEL_REQUEST', 'CANCELLED', 'REJECTED', 'EXPIRED'}
TERMINAL = {'filled', 'cancelled', 'rejected', 'expired'}


def _fields(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'{label}: missing or unsupported fields')


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(
            ord(c) < 32 for c in value) or not value.strip():
        raise ValueError(f'{label}: invalid identifier')


def _hash(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise ValueError('Invalid SHA-256 provenance digest')


def _time(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError as exc:
        raise ValueError('Invalid timestamp') from exc
    if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Timestamp requires explicit timezone')
    return parsed.astimezone(VN)


def _number(value: object, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected a JSON number')
    result = Decimal(str(value))
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError('Expected finite nonnegative number')
    return result


def _integer(value: object, *, positive: bool = True) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError('Invalid integer quantity or sequence')
    return value


def _price(value: object, reference: dict) -> Decimal:
    price = _number(value, positive=True)
    floor = _number(reference['floor_price_vnd'], positive=True)
    ceiling = _number(reference['ceiling_price_vnd'], positive=True)
    tick = _number(reference['tick_size_vnd'], positive=True)
    if not floor <= price <= ceiling or price % tick:
        raise ValueError('Price outside declared band/tick')
    return price


def validate_plan(plan: dict) -> dict:
    """Return a detached validated plain LO plan; never synthesize missing data."""
    _fields(plan, PLAN_FIELDS, 'plan')
    if type(plan['schema_version']) is not int or plan['schema_version'] != 1:
        raise ValueError('Unsupported plan schema')
    for key in ('strategy_id', 'signal_id', 'symbol'):
        _text(plan[key], key)
    if not re.fullmatch(r'[A-Z][A-Z0-9]{2,9}', plan['symbol']):
        raise ValueError('Invalid stock symbol')
    if plan['side'] not in ('BUY', 'SELL') or plan['order_type'] != 'LO':
        raise ValueError('Only plain BUY/SELL LO supported')
    if plan['price_basis'] != 'unadjusted_vnd' or type(plan['lot_size']) is not int or plan['lot_size'] != 100:
        raise ValueError('Requires raw VND and 100-share lots')
    qty = _integer(plan['quantity'])
    if qty % 100:
        raise ValueError('Quantity is not a board lot')
    _hash(plan['source_sha256'])
    signal, created, start, end = [_time(plan[k]) for k in
        ('signal_available_at', 'created_at', 'valid_from', 'valid_until')]
    if not signal <= created <= start < end or signal.date() >= start.date() or start.date() != end.date():
        raise ValueError('Invalid signal/plan/session chronology')
    if not is_trading_day(start.date()):
        raise ValueError('Plan validity is not a trading session')
    reference, resources = plan['market_reference'], plan['resources']
    _fields(reference, REFERENCE_FIELDS, 'reference')
    _fields(resources, RESOURCE_FIELDS, 'resources')
    if reference['session_date'] != start.date().isoformat():
        raise ValueError('Reference belongs to another session')
    _hash(reference['source_sha256'])
    observed = _time(reference['observed_at'])
    resource_time = _time(resources['as_of'])
    if (observed > created or observed.date() != start.date() or resource_time > created
            or resource_time.date() != created.date()):
        raise ValueError('Unavailable or wrong-session reference/resources')
    # The declared tick is a contract input, not a hardcoded historical exchange rule.
    _price(reference['reference_price_vnd'], reference)
    price = _price(plan['limit_price_vnd'], reference)
    cash = _number(resources['cash_available_vnd'])
    shares = _integer(resources['shares_available'], positive=False)
    reserve = _number(resources['buy_fee_reserve_pct'])
    if plan['side'] == 'BUY' and qty * price * (1 + reserve / 100) > cash:
        raise ValueError('Insufficient unreserved cash at limit plus fee reserve')
    if plan['side'] == 'SELL' and qty > shares:
        raise ValueError('Insufficient unreserved sellable shares')
    return copy.deepcopy(plan)


def plan_digest(plan: dict) -> str:
    validated = validate_plan(plan)
    data = json.dumps(validated, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def audit_receipts(plan: dict, events: list[dict], *, as_of: str) -> dict:
    """Audit supplied receipts as of observation time, without certifying origin."""
    plan = validate_plan(plan)
    digest = plan_digest(plan)
    cutoff, created = _time(as_of), _time(plan['created_at'])
    start, end = _time(plan['valid_from']), _time(plan['valid_until'])
    if cutoff < created or not isinstance(events, list):
        raise ValueError('Invalid audit cutoff or event list')
    unique, sequences, order_ids = {}, set(), set()
    duplicates = 0
    for event in events:
        if (not isinstance(event, dict) or not isinstance(event.get('kind'), str)
                or event['kind'] not in KINDS):
            raise ValueError('Unsupported receipt kind')
        extra = {'quantity', 'price_vnd', 'fee_vnd'} if event['kind'] == 'FILL' else set()
        if event['kind'] in ('CANCELLED', 'REJECTED', 'EXPIRED'):
            extra = {'cumulative_filled_quantity'}
            if not extra <= event.keys():
                raise ValueError('Terminal receipt missing cumulative filled quantity')
        _fields(event, EVENT_FIELDS | extra, 'receipt')
        for key in ('event_id', 'order_id'):
            _text(event[key], key)
        _hash(event['source_sha256'])
        seq = _integer(event['sequence'])
        if event['origin'] != 'broker_execution_report' or event['plan_sha256'] != digest:
            raise ValueError('Receipt is not bound to this plan/broker-report contract')
        occurred, observed = _time(event['occurred_at']), _time(event['observed_at'])
        if not created <= occurred <= observed <= cutoff:
            raise ValueError('Unavailable or inconsistent receipt timestamps')
        identity = event['event_id']
        if identity in unique:
            if unique[identity] != event:
                raise ValueError('Conflicting duplicate event ID')
            duplicates += 1
            continue
        if seq in sequences:
            raise ValueError('Conflicting duplicate sequence')
        unique[identity] = event
        sequences.add(seq)
        order_ids.add(event['order_id'])
    if len(order_ids) > 1:
        raise ValueError('More than one broker order for a fixed plan')
    state, filled, fees, cashflow = 'unconfirmed', 0, Decimal(0), Decimal(0)
    last_time = created
    for event in sorted(unique.values(), key=lambda e: e['sequence']):
        when, kind = _time(event['occurred_at']), event['kind']
        if when < last_time or state in TERMINAL:
            raise ValueError('Out-of-order or post-terminal event')
        last_time = when
        if kind in ('CANCELLED', 'REJECTED', 'EXPIRED'):
            cumulative = _integer(event['cumulative_filled_quantity'], positive=False)
            if cumulative != filled:
                raise ValueError('Terminal cumulative quantity disagrees with supplied fills')
        if kind == 'ACCEPTED':
            if state != 'unconfirmed' or when > end:
                raise ValueError('Invalid acceptance transition')
            state = 'accepted'
        elif kind == 'REJECTED':
            if state != 'unconfirmed':
                raise ValueError('Rejection after exchange acceptance')
            state = 'rejected'
        elif kind == 'EXPIRED':
            if when < end:
                raise ValueError('Premature expiry receipt')
            state = 'expired'
        else:
            if state == 'unconfirmed':
                raise ValueError('Receipt requires prior exchange acceptance')
            if kind == 'CANCEL_REQUEST':
                if state == 'cancel_pending':
                    raise ValueError('Duplicate cancellation request')
                state = 'cancel_pending'
            elif kind == 'CANCELLED':
                state = 'cancelled'
            elif kind == 'FILL':
                qty = _integer(event['quantity'])
                px = _price(event['price_vnd'], plan['market_reference'])
                fee = _number(event['fee_vnd'])
                if qty % 100 or filled + qty > plan['quantity'] or not start <= when <= end:
                    raise ValueError('Invalid fill quantity/validity')
                limit = _number(plan['limit_price_vnd'], positive=True)
                if (plan['side'] == 'BUY' and px > limit) or (plan['side'] == 'SELL' and px < limit):
                    raise ValueError('Fill violates fixed limit price')
                filled += qty
                fees += fee
                cashflow += qty * px * (-1 if plan['side'] == 'BUY' else 1) - fee
                if plan['side'] == 'BUY' and -cashflow > _number(plan['resources']['cash_available_vnd']):
                    raise ValueError('Recorded fills overdraw declared available cash')
                state = 'filled' if filled == plan['quantity'] else (
                    'cancel_pending' if state == 'cancel_pending' else 'partially_filled')
    return dict(schema_version=1, plan_sha256=digest, as_of=as_of, state=state,
        recorded_filled_quantity=filled, unconfirmed_remaining_quantity=plan['quantity'] - filled,
        recorded_fee_vnd=float(fees), recorded_cashflow_vnd=float(cashflow),
        unique_records=len(unique), duplicate_records=duplicates, fully_reconciled=state in TERMINAL,
        broker_authenticity_verified=False, live_eligible=False,
        limitations=['Supplied provenance is not independently authenticated',
                     'Single-plan audit is not portfolio solvency or funded-account P&L',
                     'No orders submitted and no daily-bar fills inferred'])

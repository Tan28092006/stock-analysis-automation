"""Order-plan and receipt evidence, never broker-order submission."""
import copy
import importlib

import pytest


def lab():
    return importlib.import_module('stock_agent.pipeline.execution_receipts')


@pytest.fixture
def plan():
    return dict(schema_version=1, strategy_id='test-only', signal_id='signal-1',
        symbol='AAA', side='BUY', order_type='LO', quantity=200, limit_price_vnd=10500,
        lot_size=100, price_basis='unadjusted_vnd', source_sha256='a'*64,
        signal_available_at='2026-09-30T15:01:00+07:00',
        created_at='2026-10-01T08:30:00+07:00',
        valid_from='2026-10-01T09:00:00+07:00', valid_until='2026-10-01T14:45:00+07:00',
        market_reference=dict(session_date='2026-10-01', reference_price_vnd=10000,
            floor_price_vnd=9300, ceiling_price_vnd=10700, tick_size_vnd=50,
            observed_at='2026-10-01T08:00:00+07:00', source_sha256='b'*64),
        resources=dict(as_of='2026-10-01T08:20:00+07:00',
            cash_available_vnd=3_000_000, shares_available=0, buy_fee_reserve_pct=.2))


def event(plan, seq, kind, **changes):
    row = dict(event_id=f'e{seq}', sequence=seq, order_id='broker-1',
        plan_sha256=lab().plan_digest(plan), origin='broker_execution_report',
        kind=kind, occurred_at=f'2026-10-01T09:{seq:02}:00+07:00',
        observed_at=f'2026-10-01T09:{seq:02}:01+07:00', source_sha256='c'*64)
    if kind == 'FILL':
        row.update(quantity=100, price_vnd=10400, fee_vnd=1560)
    row.update(changes)
    return row


def audit(plan, events, cutoff='2026-10-01T16:00:00+07:00'):
    return lab().audit_receipts(plan, events, as_of=cutoff)


def test_plan_digest_is_canonical_and_validation_does_not_mutate(plan):
    before = copy.deepcopy(plan)
    validated = lab().validate_plan(plan)
    assert validated == before and plan == before
    assert lab().plan_digest(dict(reversed(list(plan.items())))) == lab().plan_digest(plan)


@pytest.mark.parametrize('field,value', [
    ('price_basis', 'adjusted_vnd'), ('order_type', 'ATO'), ('side', 'SHORT'),
    ('quantity', 150), ('quantity', True), ('lot_size', 1),
    ('limit_price_vnd', 10501), ('limit_price_vnd', 10800), ('limit_price_vnd', float('nan')),
    ('created_at', '2026-10-01T08:30:00'),
    ('created_at', '2026-10-01T09:01:00+07:00'),
    ('signal_available_at', '2026-10-01T08:00:00+07:00'),
    ('valid_until', '2026-10-02T14:45:00+07:00'), ('source_sha256', 'unverified')])
def test_invalid_or_unavailable_plan_fails(plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        lab().validate_plan(plan)


@pytest.mark.parametrize('section,field,value', [
    ('resources','cash_available_vnd',100), ('resources','shares_available',-1),
    ('resources','buy_fee_reserve_pct',float('inf')),
    ('resources','as_of','2026-10-01T09:00:00+07:00'),
    ('market_reference','observed_at','2026-10-01T09:00:00+07:00'),
    ('market_reference','floor_price_vnd',11000),
    ('market_reference','session_date','2026-10-02')])
def test_bad_reference_or_resources_fail(plan, section, field, value):
    plan[section][field] = value
    with pytest.raises(ValueError):
        lab().validate_plan(plan)


def test_lower_execution_gate_cannot_be_disguised_as_plain_limit(plan):
    plan['minimum_execution_price_vnd'] = 10300
    with pytest.raises(ValueError):
        lab().validate_plan(plan)


def test_sell_requires_sellable_shares_and_does_not_require_buy_cash(plan):
    plan['side'] = 'SELL'
    plan['resources']['cash_available_vnd'] = 0
    with pytest.raises(ValueError):
        lab().validate_plan(plan)
    plan['resources']['shares_available'] = 200
    assert lab().validate_plan(plan)['side'] == 'SELL'


def test_no_receipts_is_unknown_not_expired_or_proven_no_fills(plan):
    r = audit(plan, [])
    assert r['state'] == 'unconfirmed' and not r['fully_reconciled']
    assert r['live_eligible'] is False and r['broker_authenticity_verified'] is False
    assert r['recorded_filled_quantity'] == 0 and r['unconfirmed_remaining_quantity'] == 200
    assert 'profit' not in r and 'pnl' not in r


def test_partial_fill_cashflow_and_exact_duplicate_are_idempotent(plan):
    a, f = event(plan,1,'ACCEPTED'), event(plan,2,'FILL')
    r = audit(plan,[f,a,f])
    assert r['state'] == 'partially_filled' and r['recorded_filled_quantity'] == 100
    assert r['recorded_cashflow_vnd'] == -1041560
    assert r['duplicate_records'] == 1 and r['recorded_fee_vnd'] == 1560


def test_cancel_request_does_not_prevent_subsequent_fill_until_ack(plan):
    records = [event(plan,1,'ACCEPTED'), event(plan,2,'CANCEL_REQUEST'),
               event(plan,3,'FILL'), event(plan,4,'CANCELLED')]
    r = audit(plan, records)
    assert r['state'] == 'cancelled' and r['fully_reconciled']
    assert r['recorded_filled_quantity'] == 100
    assert audit(plan, records[:2])['state'] == 'cancel_pending'
    with pytest.raises(ValueError):
        audit(plan, records + [event(plan,5,'FILL')])


@pytest.mark.parametrize('change', [dict(price_vnd=10600), dict(quantity=300),
    dict(quantity=10), dict(fee_vnd=-1), dict(price_vnd=10401),
    dict(origin='daily_ohlc_simulation'), dict(plan_sha256='d'*64),
    dict(order_id='another'), dict(sequence=True),
    dict(observed_at='2026-10-01T09:01:00+07:00')])
def test_bad_fill_cannot_pass_as_execution_evidence(plan, change):
    records=[event(plan,1,'ACCEPTED'),event(plan,2,'FILL',**change)]
    with pytest.raises(ValueError):
        audit(plan, records)


def test_fill_requires_acceptance_and_valid_time(plan):
    with pytest.raises(ValueError):
        audit(plan, [event(plan,2,'FILL')])
    for bad in ['2026-10-01T08:59:00+07:00','2026-10-01T14:46:00+07:00']:
        with pytest.raises(ValueError):
            audit(plan,[event(plan,1,'ACCEPTED'),event(plan,2,'FILL',occurred_at=bad,
                  observed_at='2026-10-01T15:00:00+07:00')])


def test_conflicting_duplicate_id_and_sequence_fail(plan):
    a, f = event(plan,1,'ACCEPTED'), event(plan,2,'FILL')
    for bad in [dict(f,fee_vnd=100), dict(f,event_id='other')]:
        with pytest.raises(ValueError):
            audit(plan,[a,f,bad])


def test_full_fill_and_fees_not_reinterpreted_as_profit(plan):
    r = audit(plan,[event(plan,1,'ACCEPTED'),event(plan,2,'FILL'),event(plan,3,'FILL')])
    assert r['state'] == 'filled' and r['fully_reconciled']
    assert r['recorded_cashflow_vnd'] == -2083120
    with pytest.raises(ValueError):
        audit(plan,[event(plan,1,'ACCEPTED'),event(plan,2,'FILL',quantity=200),event(plan,3,'FILL')])


def test_plan_tampering_is_detected(plan):
    records=[event(plan,1,'ACCEPTED')]
    plan['quantity']=100
    with pytest.raises(ValueError):
        audit(plan,records)


def test_cutoff_rejects_future_observation_even_if_fill_occurred_before(plan):
    with pytest.raises(ValueError):
        audit(plan,[event(plan,1,'ACCEPTED')],cutoff='2026-10-01T09:01:00+07:00')


def test_rejection_and_expiry_require_explicit_receipt(plan):
    assert audit(plan,[event(plan,1,'REJECTED')])['state']=='rejected'
    with pytest.raises(ValueError):
        audit(plan,[event(plan,1,'EXPIRED')])
    e=event(plan,2,'EXPIRED',occurred_at=plan['valid_until'],
            observed_at='2026-10-01T14:45:01+07:00')
    assert audit(plan,[event(plan,1,'ACCEPTED'),e])['state']=='expired'


def test_sell_limit_direction_and_cashflow(plan):
    plan['side']='SELL'
    plan['resources']['shares_available']=200
    good=[event(plan,1,'ACCEPTED'),event(plan,2,'FILL',price_vnd=10600)]
    assert audit(plan,good)['recorded_cashflow_vnd']==1_058_440
    with pytest.raises(ValueError):
        audit(plan,[event(plan,1,'ACCEPTED'),event(plan,2,'FILL',price_vnd=10400)])

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_agent.data import foreign_flows as ff


NOW = datetime(2026, 9, 30, 10, 5, tzinfo=timezone.utc)


def payload(day='2026-09-30', buy=1.25):
    epoch = int(datetime.fromisoformat(day + 'T00:00:00+07:00').timestamp() * 1000)
    return json.dumps([[{'StockID': 890}], [dict(TradingDate=f'/Date({epoch})/', BuyVal=buy,
        SellVal=.5, BuyVol=1000, SellVol=400, TradingMonthYear=None, Quarter=None)]]).encode()


def test_chart_values_units_and_vietnam_session_are_explicit():
    rows, rejected = ff.normalize_chart(payload(), 'MBB', 'foreign', NOW)
    assert rejected == []
    assert rows[0]['date'] == '2026-09-30'
    assert rows[0]['buy_value_vnd'] == 1_250_000_000
    assert rows[0]['net_value_vnd'] == 750_000_000
    assert rows[0]['available_at'] == NOW.isoformat()
    assert rows[0]['trade_scope'] == 'provider_chart_aggregate'


def test_intraday_future_and_holiday_rows_never_become_eod():
    morning = datetime(2026, 9, 30, 3, tzinfo=timezone.utc)
    rows, rejected = ff.normalize_chart(payload(), 'MBB', 'foreign', morning)
    assert not rows and rejected[0]['reason'] == 'incomplete_or_future_session'
    rows, rejected = ff.normalize_chart(payload('2026-09-02'), 'MBB', 'foreign', NOW)
    assert not rows and rejected[0]['reason'] == 'nontrading_session'


@pytest.mark.parametrize('bad', [None, float('nan'), -1, 'garbage'])
def test_missing_nonfinite_negative_values_are_not_zero(bad):
    rows, rejected = ff.normalize_chart(payload(buy=bad), 'MBB', 'foreign', NOW)
    assert not rows and rejected


def test_schema_unknown_symbol_path_and_naive_time_fail_closed():
    with pytest.raises(ValueError):
        ff.normalize_chart(b'{}', 'MBB', 'foreign', NOW)
    with pytest.raises(ValueError):
        ff.normalize_chart(payload(), '../MBB', 'foreign', NOW)
    with pytest.raises(ValueError):
        ff.normalize_chart(payload(), 'MBB', 'foreign', NOW.replace(tzinfo=None))


def test_store_preserves_revisions_and_filters_asof_before_dedup(tmp_path):
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k: payload(), clock=lambda: NOW, sleep=lambda _: None)
    later = NOW.replace(hour=11)
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k: payload(buy=2.), clock=lambda: later, sleep=lambda _: None)
    old = ff.load_flows(root=tmp_path, as_of=NOW)
    new = ff.load_flows(root=tmp_path, as_of=later)
    assert len(old) == len(new) == 1
    assert old.iloc[0]['f_buy_val'] == 1.25
    assert new.iloc[0]['f_buy_val'] == 2.
    assert ff.load_flows(root=tmp_path, as_of=NOW.replace(hour=9)).empty
    assert len(list((tmp_path/'runs').glob('*/manifest.json'))) == 2


def test_partial_collection_is_visible_and_never_zero_fills(tmp_path):
    def fetch(symbol, kind):
        if symbol == 'AAA':
            raise TimeoutError('do not log tokens')
        return payload()
    result = ff.collect_flows(['MBB', 'AAA'], root=tmp_path, fetcher=fetch, clock=lambda: NOW, sleep=lambda _: None)
    assert result['status'] == 'partial'
    assert result['missing_latest']['foreign'] == ['AAA']
    assert set(ff.load_flows(root=tmp_path)['symbol']) == {'MBB'}
    assert 'do not log tokens' not in json.dumps(result)


def test_tampered_payload_and_concurrent_writers_block(tmp_path):
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k: payload(), clock=lambda: NOW, sleep=lambda _: None)
    raw = next((tmp_path/'runs').glob('*/raw/*.json'))
    raw.write_bytes(b'{}')
    with pytest.raises(ValueError, match='hash'):
        ff.load_flows(root=tmp_path)
    (tmp_path/'.collect-lock').mkdir()
    with pytest.raises(ValueError, match='Concurrent'):
        ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k: payload(), clock=lambda: NOW, sleep=lambda _: None)


def test_legacy_migration_keeps_raw_hash_and_never_enters_default_panel(tmp_path):
    legacy = tmp_path/'legacy'; legacy.mkdir()
    source = legacy/'ndtnn.jsonl'
    row = dict(date='2026-07-01', TradingDate='/Date(1782925200000)/', StockCode='AAA',
               BuyVal=1000., SellVal=250., BuyVol=100, SellVol=25)
    source.write_text(json.dumps(row), encoding='utf-8')
    before = source.read_bytes()
    report = ff.migrate_legacy(legacy, tmp_path/'store', now=NOW)
    assert source.read_bytes() == before
    assert report['normalized_rows'] == 1
    migrated = json.loads(Path(report['rows_path']).read_text().splitlines()[0])
    assert migrated['date'] == '2026-07-02'
    assert migrated['buy_value_vnd'] == 1e9
    assert migrated['available_at'] is None
    assert ff.load_flows(root=tmp_path/'store').empty

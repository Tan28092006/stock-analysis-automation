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
    assert set(ff.load_flows(root=tmp_path, as_of=NOW)['symbol']) == {'MBB'}
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


def test_health_does_not_call_stale_or_incomplete_universe_ready(tmp_path):
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k:payload(), clock=lambda:NOW, sleep=lambda _:None)
    assert ff.health(['MBB'], root=tmp_path, now=NOW)['status'] == 'ready'
    report = ff.health(['MBB','ACB'], root=tmp_path, now=NOW)
    assert report['status'] == 'partial'
    assert report['missing_latest']['foreign'] == ['ACB']
    assert ff.health(['MBB'], root=tmp_path, now=NOW.replace(day=1, month=10))['status'] != 'ready'


def test_health_detects_history_gap_between_successful_downloads(tmp_path):
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k:payload('2026-09-28'), clock=lambda:NOW.replace(day=28), sleep=lambda _:None)
    ff.collect_flows(['MBB'], root=tmp_path, fetcher=lambda s,k:payload(), clock=lambda:NOW, sleep=lambda _:None)
    report = ff.health(['MBB'], root=tmp_path, now=NOW)
    assert '2026-09-29' in report['gaps']['foreign:MBB']
    assert report['status'] == 'partial'


def test_response_observed_later_than_commit_is_invalid(tmp_path):
    times = iter([NOW, NOW.replace(hour=12), NOW.replace(hour=12), NOW])
    with pytest.raises(ValueError, match='Clock'):
        ff.collect_flows(['MBB'],root=tmp_path,fetcher=lambda s,k:payload(),clock=lambda:next(times),sleep=lambda _:None)


def test_identical_rerun_is_panel_idempotent_but_revisions_retained(tmp_path):
    for _ in range(2):
        ff.collect_flows(['MBB'],root=tmp_path,fetcher=lambda s,k:payload(),clock=lambda:NOW,sleep=lambda _:None)
    assert len(ff.load_flows(root=tmp_path,as_of=NOW)) == 1
    assert len(ff.verified_records(tmp_path)) == 4


def test_conflicting_duplicate_session_is_quarantined_entirely():
    data = json.loads(payload()); second = dict(data[1][0]); second['BuyVal']=2
    data[1].append(second)
    rows, rejected = ff.normalize_chart(json.dumps(data).encode(),'MBB','foreign',NOW)
    assert rows == [] and rejected[0]['reason']=='duplicate_session'


def test_transient_failure_retries_and_empty_payload_is_not_success(tmp_path):
    calls=[]
    def fetch(s,k):
        calls.append(k)
        if len(calls)==1:
            raise TimeoutError()
        return payload()
    result=ff.collect_flows(['MBB'],root=tmp_path/'retry',fetcher=fetch,clock=lambda:NOW,sleep=lambda _:None)
    assert result['status']=='ready' and len(calls)==3
    result=ff.collect_flows(['MBB'],root=tmp_path/'empty',fetcher=lambda s,k:b'[[],[]]',clock=lambda:NOW,sleep=lambda _:None)
    assert result['status']=='blocked' and result['rows']==0 and len(result['errors'])==2


def test_unfinished_run_is_visible_as_blocked_health(tmp_path):
    (tmp_path/'runs'/'interrupted'/'raw').mkdir(parents=True)
    report = ff.health(['MBB'], root=tmp_path, now=NOW)
    assert report['status']=='blocked' and report['incomplete_runs']==['interrupted']


def test_unexpected_html_is_not_archived_as_market_raw(tmp_path):
    ff.collect_flows(['MBB'],root=tmp_path,fetcher=lambda s,k:b'<html>private-session-token</html>',clock=lambda:NOW,sleep=lambda _:None)
    assert not list((tmp_path/'runs').glob('*/raw/*.json'))


def test_market_schema_rejects_aggregates_fractional_volume_and_unknown_kind():
    data=json.loads(payload()); data[1][0]['Quarter']=3
    rows,bad=ff.normalize_chart(json.dumps(data).encode(),'MBB','foreign',NOW)
    assert not rows and bad[0]['reason']=='not_daily'
    data=json.loads(payload()); data[1][0]['BuyVol']=100.5
    assert not ff.normalize_chart(json.dumps(data).encode(),'MBB','foreign',NOW)[0]
    with pytest.raises(ValueError):
        ff.normalize_chart(payload(),'MBB','unknown',NOW)


def test_manifest_canonical_and_path_tamper_block(tmp_path):
    result=ff.collect_flows(['MBB'],root=tmp_path,fetcher=lambda s,k:payload(),clock=lambda:NOW,sleep=lambda _:None)
    manifest=Path(result['manifest_path']); data=json.loads(manifest.read_text())
    data['sources'][0]['path']='../../outside.json'; manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='path'):
        ff.load_flows(root=tmp_path,as_of=NOW)


def test_zero_activity_is_explicit_and_differs_from_missing_data():
    data=json.loads(payload())
    data[1][0].update(BuyVal=0,SellVal=0,BuyVol=0,SellVol=0)
    rows,bad=ff.normalize_chart(json.dumps(data).encode(),'MBB','foreign',NOW)
    assert rows[0]['net_value_vnd']==0 and bad==[]


def test_legacy_chart_keeps_unknown_session_and_bad_json_quarantined(tmp_path):
    legacy=tmp_path/'legacy'; legacy.mkdir()
    (legacy/'ndtnn_chart.jsonl').write_text(json.dumps(dict(date='2026-09-20',symbol='MBB',buy_val=1,sell_val=.5,buy_vol=100,sell_vol=50))+'\ninvalid')
    report=ff.migrate_legacy(legacy,tmp_path/'store',now=NOW)
    assert report['normalized_rows']==1 and report['quarantined_rows']==1
    row=json.loads(Path(report['rows_path']).read_text().strip())
    assert row['date'] is None and row['research_only'] is True

from datetime import date
import json

import pytest

from scripts import foreign_flow_research as flow


def test_source_epoch_uses_vietnam_date_not_utc_date():
    assert flow.source_session('/Date(1782925200000)/') == '2026-07-02'


def test_detailed_million_to_billion_and_no_false_pit_eligibility():
    row = dict(date='2026-07-01', TradingDate='/Date(1782925200000)/',
               StockCode='AAA', BuyVal=9127.315, SellVal=1000, BuyVol=10, SellVol=5)
    actual = flow.normalize('ndtnn', row)
    assert actual['session'] == '2026-07-02'
    assert actual['net_bn'] == pytest.approx(8.127315)
    assert actual['eligible'] is False
    assert actual['date_quality'] == 'source_epoch'


def test_chart_date_is_not_silently_repaired():
    row = dict(date='2026-09-20', symbol='ACB', buy_val=1., sell_val=2., buy_vol=1, sell_vol=2)
    actual = flow.normalize('ndtnn_chart', row)
    assert actual['session'] is None
    assert actual['eligible'] is False
    assert actual['stored_date'] == '2026-09-20'


def test_missing_and_nan_values_are_quarantined_not_zero():
    with pytest.raises(ValueError):
        flow.normalize('price_board_snapshots', dict(date='2026-09-21', symbol='ACB', f_buy_val=float('nan'), f_sell_val=1))


def test_monthly_is_not_daily_and_requires_period_key():
    with pytest.raises(ValueError, match='daily'):
        flow.normalize('ndtnn_monthly', dict(period='2026-06', symbol='ACB', buy_val=1, sell_val=0))


def test_inventory_report_is_strict_json_and_uses_period_key(tmp_path):
    detail = dict(date='2026-07-01', TradingDate='/Date(1782925200000)/', StockCode='AAA',
                  BuyVal=1000, SellVal=500, BuyVol=100, SellVol=50)
    chart = dict(date='2026-07-01', symbol='AAA', buy_val=1, sell_val=.5, buy_vol=100, sell_vol=50)
    (tmp_path / 'ndtnn.jsonl').write_text(json.dumps(detail))
    (tmp_path / 'ndtnn_chart.jsonl').write_text(json.dumps(chart))
    (tmp_path / 'ndtnn_monthly.jsonl').write_text('\n'.join(json.dumps(dict(symbol='AAA', period=p)) for p in ['2026-06','2026-07']))
    files, rows, units = flow.inventory(tmp_path)
    assert files['ndtnn_monthly']['duplicate_rows'] == 0
    assert units['matched_volume_and_unit_converted_value'] == 1
    json.dumps(dict(files=files, units=units), allow_nan=False)


def test_aggregate_period_bounds_are_chronological(tmp_path):
    (tmp_path / 'ndtnn_monthly.jsonl').write_text('\n'.join(json.dumps(dict(symbol='AAA', period=p)) for p in ['12/2025','01/2026']))
    files, _, _ = flow.inventory(tmp_path)
    assert files['ndtnn_monthly']['first'] == '12/2025'
    assert files['ndtnn_monthly']['last'] == '01/2026'

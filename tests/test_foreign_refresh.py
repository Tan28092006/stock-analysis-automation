import json
from datetime import datetime, timezone

from stock_agent.pipeline import foreign_refresh as job


def test_scheduled_universe_keeps_collection_and_current_trading_names():
    symbols = job.collection_symbols()
    assert len(symbols) == 105
    assert {'MCH','TAL','TCX','VCK','VPX','ACB','ANV','VTP','DXS','HDC','IMP','SCS','SZC'} <= set(symbols)


def test_job_returns_nonzero_on_partial_and_status_is_visible(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(job.ff, 'collect_flows', lambda *a,**kw: dict(status='partial',rows=1,missing_latest={'foreign':['MBB']}))
    assert job.main(['--root',str(tmp_path),'--symbols','MBB']) == 2
    assert 'partial' in capsys.readouterr().out


def test_exception_is_nonzero_and_does_not_expose_source_secrets(tmp_path, monkeypatch, capsys):
    def fail(*a,**kw):
        raise RuntimeError('secret-token-123')
    monkeypatch.setattr(job.ff, 'collect_flows', fail)
    assert job.main(['--root',str(tmp_path),'--symbols','MBB']) == 2
    assert 'secret-token-123' not in capsys.readouterr().out


def test_old_pipeline_harvest_routes_to_v2_not_legacy_files(monkeypatch):
    from stock_agent.pipeline import eod_update
    monkeypatch.setattr(job.ff,'collect_flows',lambda symbols: {'status':'ready','symbols':symbols})
    result = eod_update.harvest_foreign()
    assert result['status'] == 'ready' and 'MCH' in result['symbols']

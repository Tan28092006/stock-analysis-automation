"""Bounded late-publication recovery; no real network or real sleeps."""
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stock_agent.pipeline import foreign_refresh as job
from tests.test_foreign_flows_v2 import payload


NOW = datetime(2026, 9, 30, 10, 5, tzinfo=timezone.utc)


def state(missing=(), **extra):
    return dict(status='partial' if missing else 'ready', expected_session='2026-09-30',
                missing_latest={'foreign': [], 'proprietary': list(missing)}, gaps={},
                errors=[], quarantine_rows=0, quarantined_intraday_rows=0,
                incomplete_runs=[], collection_in_progress=False, **extra)


def setup_batches(monkeypatch, batches, healths=None):
    calls, sleeps = [], []
    queue = iter(batches)
    current = {}
    def collect(symbols, **kwargs):
        calls.append(list(symbols))
        current.clear()
        current.update(copy.deepcopy(next(queue)))
        return {**current, 'manifest_path': f'batch-{len(calls)}/manifest.json'}
    health_queue = iter(healths) if healths is not None else None
    def health(symbols, **kwargs):
        assert symbols == ['AAA', 'BBB']
        return copy.deepcopy(next(health_queue) if health_queue else current)
    monkeypatch.setattr(job.ff, 'collect_flows', collect)
    monkeypatch.setattr(job.ff, 'health', health)
    return calls, sleeps


def test_ready_first_batch_never_waits_or_fetches_again(monkeypatch, tmp_path):
    calls, waits = setup_batches(monkeypatch, [state()])
    result = job.refresh_with_retries(['BBB', 'AAA'], root=tmp_path, clock=lambda: NOW, sleep=waits.append)
    assert result['status'] == 'ready' and result['attempts'] == 1
    assert calls == [['AAA', 'BBB']] and waits == []


def test_late_publication_retries_only_missing_names_and_checks_full_basket(monkeypatch, tmp_path):
    calls, waits = setup_batches(monkeypatch, [state(['BBB']), state()])
    result = job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: NOW, sleep=waits.append)
    assert calls == [['AAA', 'BBB'], ['BBB']] and sum(waits) == 60
    assert result['status'] == 'ready' and result['attempts'] == 2
    assert result['attempt_manifest_paths'] == ['batch-1/manifest.json', 'batch-2/manifest.json']


def test_persistent_staleness_exhausts_three_retries_without_success(monkeypatch, tmp_path):
    calls, waits = setup_batches(monkeypatch, [state(['BBB']) for _ in range(4)])
    result = job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: NOW, sleep=waits.append)
    assert result['status'] == 'partial' and result['retry_exhausted'] is True
    assert len(calls) == 4 and sum(waits) == 1260
    assert result['missing_latest']['proprietary'] == ['BBB']


@pytest.mark.parametrize('problem', [
    {'errors': [{'error': 'ValueError'}]}, {'quarantine_rows': 1},
    {'gaps': {'proprietary:BBB': ['2026-09-28']}},
    {'incomplete_runs': ['unfinished']}, {'collection_in_progress': True},
])
def test_nonretryable_evidence_is_not_hidden_by_retries(monkeypatch, tmp_path, problem):
    broken = state(['BBB']); broken.update(problem)
    calls, waits = setup_batches(monkeypatch, [broken])
    result = job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: NOW, sleep=waits.append)
    assert result['status'] != 'ready' and len(calls) == 1 and not waits


def test_retry_subset_ready_does_not_hide_historical_gap_elsewhere(monkeypatch, tmp_path):
    global_bad = state(); global_bad.update(status='partial', gaps={'foreign:AAA': ['2026-09-28']})
    calls, waits = setup_batches(monkeypatch, [state(['BBB']), state()], [state(['BBB']), global_bad])
    result = job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: NOW, sleep=waits.append)
    assert result['status'] == 'partial' and result['gaps'] == global_bad['gaps']
    assert len(calls) == 2


@pytest.mark.parametrize('later', [NOW - timedelta(seconds=1), NOW + timedelta(days=1)])
def test_clock_reversal_or_session_rollover_aborts_before_next_fetch(monkeypatch, tmp_path, later):
    calls, _ = setup_batches(monkeypatch, [state(['BBB'])])
    current = [NOW]
    def wait(seconds):
        current[0] = later
    with pytest.raises(ValueError):
        job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: current[0], sleep=wait)
    assert len(calls) == 1


def test_real_store_keeps_old_missing_value_and_new_availability(monkeypatch, tmp_path):
    actual_collect = job.ff.collect_flows
    current = [NOW]
    count = [0]
    first_manifest = []
    def collect(symbols, **kwargs):
        count[0] += 1
        def fetch(symbol, kind):
            stale = count[0] == 1 and symbol == 'BBB' and kind == 'proprietary'
            return payload('2026-09-29' if stale else '2026-09-30')
        result = actual_collect(symbols, root=tmp_path, fetcher=fetch,
                                clock=lambda: current[0], sleep=lambda _: None)
        if count[0] == 1:
            p = Path(result['manifest_path'])
            first_manifest.extend([p, p.read_bytes()])
        return result
    def wait(seconds):
        current[0] += timedelta(seconds=seconds)
    monkeypatch.setattr(job.ff, 'collect_flows', collect)
    result = job.refresh_with_retries(['AAA', 'BBB'], root=tmp_path, clock=lambda: current[0], sleep=wait)
    assert result['status'] == 'ready' and count[0] == 2
    assert first_manifest[0].read_bytes() == first_manifest[1]
    old = job.ff.load_flows(root=tmp_path, as_of=NOW)
    new = job.ff.load_flows(root=tmp_path, as_of=current[0])
    assert old.loc[(old.symbol == 'BBB') & (old.date.astype(str) == '2026-09-30'), 'td_net_val'].isna().all()
    assert new.loc[(new.symbol == 'BBB') & (new.date.astype(str) == '2026-09-30'), 'td_net_val'].notna().all()


def test_cli_retry_flag_routes_to_recovery_without_live_scan(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(job, 'refresh_with_retries', lambda symbols, **kwargs: calls.append(symbols) or state(), raising=False)
    assert job.main(['--root', str(tmp_path), '--symbols', 'AAA', '--retry-stale']) == 0
    assert calls == [['AAA']] and json.loads(capsys.readouterr().out)['status'] == 'ready'


def test_default_collection_retains_every_previous_name():
    old = set('ACB ANV BAF BCM BID BMP BSI BSR BVH BWE CII CMG CTD CTG CTR CTS DBC DCM DGW DIG DPM DSE DXG DXS EIB EVF FPT FRT FTS GAS GEE GEX GMD GVR HAG HCM HDB HDC HDG HHV HPG HSG HT1 IMP KBC KDC KDH KOS LPB MBB MSB MSN MWG NAB NKG NLG NT2 NVL OCB PAN PC1 PDR PHR PLX PNJ POW PVD PVT REE SAB SBT SCS SHB SIP SJS SSB SSI STB SZC TCB TCH TPB VCB VCG VCI VGC VHC VHM VIB VIC VIX VJC VND VNM VPB VPI VPL VRE VSC VTP'.split())
    config = json.loads(Path('configs/foreign_flows.json').read_text())
    assert len(old) == 100
    assert set(config['symbols']) == old | {'MCH', 'TAL', 'TCX', 'VCK', 'VPX'}


def test_launcher_opts_into_bounded_freshness_retry():
    launcher = Path('run_eod_update.bat').read_text()
    assert '-m stock_agent.pipeline.foreign_refresh --retry-stale' in launcher

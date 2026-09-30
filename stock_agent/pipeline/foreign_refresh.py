"""Unattended flow collection/migration/status. No strategy or ledger writes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import load_universe
from ..data import foreign_flows as ff


def collection_symbols():
    config = json.loads(Path('configs/foreign_flows.json').read_text(encoding='utf-8'))
    return ff.symbols_contract(sorted(set(config['symbols']) | set(load_universe()['symbols'])))


def refresh_with_retries(symbols, *, root=ff.STORE, clock=None, sleep=time.sleep):
    """Recover delayed valid EOD rows without weakening source/store checks."""
    symbols = ff.symbols_contract(symbols)
    clock = clock or ff.aware
    last_time = ff.aware(clock())
    expected = str(ff.completed_session_date(last_time))
    delays = (60, 300, 900)
    targets, manifests = symbols, []

    def checkpoint():
        nonlocal last_time
        now = ff.aware(clock())
        if now < last_time or str(ff.completed_session_date(now)) != expected:
            raise ValueError('Clock reversed or completed session changed during refresh')
        last_time = now
        return now

    for attempt in range(1 + len(delays)):
        checkpoint()
        batch = ff.collect_flows(targets, root=root, clock=clock)
        current = ff.health(symbols, root=root, now=checkpoint())
        if batch['expected_session'] != expected or current['expected_session'] != expected:
            raise ValueError('Batch or health belongs to another completed session')
        manifests.append(batch['manifest_path'])
        missing = {kind: sorted(set(batch.get('missing_latest', {}).get(kind, [])) |
                                set(current.get('missing_latest', {}).get(kind, [])))
                   for kind in ff.KINDS}
        gaps = {key: sorted(set(batch.get('gaps', {}).get(key, [])) |
                            set(current.get('gaps', {}).get(key, [])))
                for key in set(batch.get('gaps', {})) | set(current.get('gaps', {}))}
        invalid = (bool(batch.get('errors')) or
                   batch.get('quarantine_rows', 0) > batch.get('quarantined_intraday_rows', 0) or
                   bool(current.get('incomplete_runs')) or current.get('collection_in_progress', False))
        ready = batch['status'] == current['status'] == 'ready' and not invalid
        result = dict(current, status='ready' if ready else (
            'blocked' if 'blocked' in (batch['status'], current['status']) else 'partial'),
            missing_latest=missing, gaps=gaps, errors=batch.get('errors', []),
            quarantine_rows=batch.get('quarantine_rows', 0),
            quarantined_intraday_rows=batch.get('quarantined_intraday_rows', 0),
            requested_symbols=symbols, attempts=attempt + 1,
            attempt_manifest_paths=list(manifests), retry_exhausted=False)
        if ready:
            return result
        targets = sorted({s for group in missing.values() for s in group})
        if not set(targets) <= set(symbols):
            raise ValueError('Missing-symbol report escaped requested basket')
        if (invalid or batch['status'] not in ('ready', 'partial') or not targets or
                any(day != expected for days in gaps.values() for day in days)):
            result['retry_reason'] = 'nonretryable_source_or_store_evidence'
            return result
        if attempt == len(delays):
            result['retry_exhausted'] = True
            return result
        remaining = delays[attempt]
        while remaining:
            step = min(60, remaining)
            sleep(step)
            checkpoint()
            remaining -= step


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ff.STORE)
    p.add_argument('--symbols', nargs='+')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--migrate-legacy', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--retry-stale', action='store_true', help='Bounded retries for delayed valid EOD rows')
    args = p.parse_args(argv)
    try:
        if args.migrate_legacy:
            result = ff.migrate_legacy(root=args.root)
        elif args.status:
            result = ff.health(args.symbols or collection_symbols(), root=args.root)
        elif args.retry_stale:
            result = refresh_with_retries(args.symbols or collection_symbols(), root=args.root)
        else:
            result = ff.collect_flows(args.symbols or collection_symbols(), root=args.root)
            if result['status'] == 'ready':
                current = ff.health(args.symbols or collection_symbols(), root=args.root)
                result['status'] = current['status']
                result['gaps'] = current['gaps']
                result['missing_latest'] = current['missing_latest']
        # Keep terminal/log output short; detailed errors, dates and hashes are on disk.
        keys = ('status','expected_session','rows','normalized_rows','quarantined_rows','date_recovered_rows',
                'missing_latest','gaps','manifest_path','rows_path','errors','quarantine_rows','quarantined_intraday_rows',
                'requested_symbols','attempts','attempt_manifest_paths','retry_exhausted','retry_reason')
        print(json.dumps({k:result[k] for k in keys if k in result}, ensure_ascii=False), flush=True)
        return 0 if result['status'] in ('ready','research_only') else 2
    except Exception as exc:
        print(json.dumps(dict(status='blocked', error=type(exc).__name__,
                              reason='See run artifacts; no fallback to unverified legacy data')), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

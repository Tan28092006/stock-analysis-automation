"""Unattended flow collection/migration/status. No strategy or ledger writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_universe
from ..data import foreign_flows as ff


def collection_symbols():
    config = json.loads(Path('configs/foreign_flows.json').read_text(encoding='utf-8'))
    return ff.symbols_contract(sorted(set(config['symbols']) | set(load_universe()['symbols'])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ff.STORE)
    p.add_argument('--symbols', nargs='+')
    p.add_argument('--migrate-legacy', action='store_true')
    p.add_argument('--status', action='store_true')
    args = p.parse_args(argv)
    try:
        if args.migrate_legacy:
            result = ff.migrate_legacy(root=args.root)
        elif args.status:
            result = ff.health(args.symbols or collection_symbols(), root=args.root)
        else:
            result = ff.collect_flows(args.symbols or collection_symbols(), root=args.root)
            if result['status'] == 'ready':
                current = ff.health(args.symbols or collection_symbols(), root=args.root)
                result['status'] = current['status']
                result['gaps'] = current['gaps']
                result['missing_latest'] = current['missing_latest']
        # Keep terminal/log output short; detailed errors, dates and hashes are on disk.
        keys = ('status','expected_session','rows','normalized_rows','quarantined_rows','date_recovered_rows',
                'missing_latest','gaps','manifest_path','rows_path','errors','quarantine_rows','quarantined_intraday_rows')
        print(json.dumps({k:result[k] for k in keys if k in result}, ensure_ascii=False), flush=True)
        return 0 if result['status'] in ('ready','research_only') else 2
    except Exception as exc:
        print(json.dumps(dict(status='blocked', error=type(exc).__name__,
                              reason='See run artifacts; no fallback to unverified legacy data')), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

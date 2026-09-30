"""Local receipt-file audit only. Never connects to a broker or creates orders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stock_agent.data import exchange_calendar
from stock_agent.pipeline import execution_receipts

MAX_INPUT_BYTES = 8 * 1024 * 1024


def _object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError('Nonfinite JSON number')


def _read(path: Path) -> tuple[object, str]:
    with path.open('rb') as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError('Input exceeds size limit')
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    return value, hashlib.sha256(raw).hexdigest()


def audit_files(plan_path: Path, receipts_path: Path, output: Path, *, as_of: str) -> dict:
    """Read supplied data once; hash the exact parsed bytes; never overwrite."""
    if output.exists():
        raise FileExistsError('Audit output already exists')
    plan, plan_hash = _read(plan_path)
    receipts, receipt_hash = _read(receipts_path)
    result = execution_receipts.audit_receipts(plan, receipts, as_of=as_of)
    result['input_sha256'] = {'plan': plan_hash, 'receipts': receipt_hash}
    paths = [Path(__file__), Path(execution_receipts.__file__), Path(exchange_calendar.__file__)]
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--as-of', required=True, help='Explicit timezone-aware observation cutoff')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_files(args.plan, args.receipts, args.output, as_of=args.as_of)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Supplied private fields and raw exception content never enter console logs.
        parser.exit(2, 'BLOCKED: invalid/missing evidence or output already exists; inputs preserved.\n')
    print(json.dumps({k: result[k] for k in ('state', 'recorded_filled_quantity', 'live_eligible')}))


if __name__ == '__main__':
    main()

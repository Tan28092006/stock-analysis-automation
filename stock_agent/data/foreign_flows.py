"""Versioned foreign/proprietary flow data with conservative availability.

Legacy data never enters the default panel. Readiness is not trading approval.
Raw responses contain public market data only, never cookies/request tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from .exchange_calendar import VN_TIMEZONE, completed_session_date, is_trading_day, trading_days_between

FOREIGN_DIR = Path('data/raw/foreign')
STORE = Path('data/foreign_flows_v2')
KINDS = ('foreign', 'proprietary')
ENDPOINTS = {
    'foreign': 'https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock',
    'proprietary': 'https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhChartByStock',
}
MAX_RESPONSE = 5 * 1024 * 1024


def aware(value=None):
    value = value or datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return value.astimezone(timezone.utc)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_time(value):
    match = re.fullmatch(r'/Date\((\d+)(?:[+-]\d{4})?\)/', str(value))
    if not match:
        raise ValueError('invalid_source_timestamp')
    return datetime.fromtimestamp(int(match.group(1)) / 1000, timezone.utc)


def symbols_contract(symbols):
    if not symbols or len(set(symbols)) != len(symbols) or any(
            not isinstance(s, str) or not re.fullmatch(r'[A-Z0-9]{2,12}', s) or s == 'VNINDEX' for s in symbols):
        raise ValueError('Distinct valid stock symbols required')
    return sorted(symbols)


def number(value, scale=1):
    try:
        n = Decimal(str(value)) * scale
    except (InvalidOperation, TypeError) as exc:
        raise ValueError('invalid_numeric') from exc
    if not n.is_finite() or n < 0 or n > Decimal('1e18'):
        raise ValueError('invalid_numeric')
    return float(n)


def values(row, scale):
    buy, sell = number(row['BuyVal'], scale), number(row['SellVal'], scale)
    bv, sv = number(row['BuyVol']), number(row['SellVol'])
    if bv != int(bv) or sv != int(sv) or (bv == 0) != (buy == 0) or (sv == 0) != (sell == 0):
        raise ValueError('inconsistent_value_volume')
    return dict(buy_value_vnd=buy, sell_value_vnd=sell, net_value_vnd=buy-sell,
                buy_volume=int(bv), sell_volume=int(sv), net_volume=int(bv-sv))


def normalize_chart(raw, symbol, kind, observed_at, *, cutoff=None):
    symbols_contract([symbol])
    observed_at = aware(observed_at)
    if kind not in KINDS or not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE:
        raise ValueError('Unknown source/category or oversized response')
    cutoff = cutoff or completed_session_date(observed_at)
    if cutoff > completed_session_date(observed_at):
        raise ValueError('Cutoff after completed session')
    payload = json.loads(raw.decode('utf-8-sig'))
    if (not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[0], list)
            or not isinstance(payload[1], list) or not payload[1]):
        raise ValueError('Empty or changed chart schema')
    rows, rejected, seen, duplicates = [], [], set(), set()
    for index, row in enumerate(payload[1]):
        session = None
        try:
            session = source_time(row['TradingDate']).astimezone(VN_TIMEZONE).date()
            if not is_trading_day(session):
                raise ValueError('nontrading_session')
            if session > cutoff:
                raise ValueError('incomplete_or_future_session')
            if row.get('TradingMonthYear') is not None or row.get('Quarter') is not None:
                raise ValueError('not_daily')
            if str(session) in seen:
                duplicates.add(str(session))
                raise ValueError('duplicate_session')
            seen.add(str(session))
            rows.append(dict(schema_version=2, date=str(session), symbol=symbol, investor=kind,
                source='vietstock_chart', trade_scope='provider_chart_aggregate', units='VND/shares',
                source_timestamp=row['TradingDate'], observed_at=observed_at.isoformat(),
                available_at=observed_at.isoformat(), raw_sha256=sha(raw), **values(row, 1_000_000_000)))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            reason = str(exc) if str(exc) in {'nontrading_session','incomplete_or_future_session','not_daily','duplicate_session',
                                             'invalid_source_timestamp','invalid_numeric','inconsistent_value_volume'} else 'invalid_row_schema'
            rejected.append(dict(row=index, date=str(session) if session else None, reason=reason))
    return [r for r in rows if r['date'] not in duplicates], rejected


class VietstockClient:
    """Anonymous public session. CSRF tokens remain in memory, never in artifacts."""
    def __init__(self):
        import requests
        from html.parser import HTMLParser
        self.session = requests.Session()
        self.referer = 'https://finance.vietstock.vn/MBB/thong-ke-giao-dich.htm'
        self.session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': self.referer})
        response = self.session.get(self.referer, timeout=(10, 25))
        response.raise_for_status()

        class TokenParser(HTMLParser):
            in_form = False
            token = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'form':
                    self.in_form = attrs.get('id') == '__CHART_AjaxAntiForgeryForm'
                if tag == 'input' and self.in_form and attrs.get('name') == '__RequestVerificationToken':
                    self.token = attrs.get('value')

            def handle_endtag(self, tag):
                if tag == 'form':
                    self.in_form = False

        parser = TokenParser()
        parser.feed(response.text)
        if not parser.token:
            raise ValueError('Public chart session unavailable')
        self.token = parser.token

    def fetch(self, symbol, kind):
        symbols_contract([symbol])
        data = dict(stockCode=symbol, type='1', __RequestVerificationToken=self.token)
        if kind == 'foreign':
            data['isRealTime'] = 'false'
        response = self.session.post(ENDPOINTS[kind], data=data,
            headers={'X-Requested-With':'XMLHttpRequest'}, timeout=(10,25))
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE:
            raise ValueError('Oversized source response')
        return response.content

    def close(self):
        self.session.close()


def collect_flows(symbols, *, root=STORE, fetcher=None, clock=None, sleep=time.sleep):
    symbols = symbols_contract(symbols)
    clock = clock or (lambda: datetime.now(timezone.utc))
    started = aware(clock())
    cutoff = completed_session_date(started)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.collect-lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError('Concurrent or interrupted collector; investigate lock') from exc
    client = None
    try:
        run = root / 'runs' / (started.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8])
        (run/'raw').mkdir(parents=True)
        records, sources, errors, quarantine = [], [], [], []
        failures = 0
        for kind in KINDS:
            for symbol in symbols:
                if failures >= 5:
                    errors.append(dict(symbol=symbol, investor=kind, error='circuit_open'))
                    continue
                for attempt in range(3):
                    try:
                        if fetcher is None:
                            if client is None:
                                client = VietstockClient()
                            raw = client.fetch(symbol, kind)
                        else:
                            raw = fetcher(symbol, kind)
                        observed = aware(clock())
                        if observed < started:
                            raise ValueError('Clock moved backwards')
                        raw_path = f'raw/{kind}_{symbol}.json'
                        if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE:
                            raise ValueError('Invalid response')
                        (run/raw_path).write_bytes(raw)
                        rows, bad = normalize_chart(raw, symbol, kind, observed, cutoff=cutoff)
                        sources.append(dict(symbol=symbol, investor=kind, path=raw_path, sha256=sha(raw),
                                            observed_at=observed.isoformat()))
                        records.extend(rows)
                        quarantine.extend(dict(symbol=symbol, investor=kind, **r) for r in bad)
                        failures = 0
                        break
                    except Exception as exc:
                        if client is not None:
                            client.close()
                            client = None
                        if attempt == 2:
                            failures += 1
                            errors.append(dict(symbol=symbol, investor=kind, error=type(exc).__name__, attempts=3))
                        else:
                            sleep(2 ** (attempt + 1))
                sleep(.65)
        completed = aware(clock())
        for row in records:
            row['available_at'] = max(aware(row['observed_at']), completed).isoformat()
        missing = {k: sorted(set(symbols) - {r['symbol'] for r in records if r['investor'] == k and r['date'] == str(cutoff)}) for k in KINDS}
        gaps, coverage = {}, {}
        for kind in KINDS:
            for symbol in symbols:
                days = sorted({r['date'] for r in records if r['investor'] == kind and r['symbol'] == symbol})
                if days:
                    key = kind + ':' + symbol
                    absent = sorted({str(d) for d in trading_days_between(date.fromisoformat(days[0]), cutoff)} - set(days))
                    coverage[key] = dict(first=days[0], last=days[-1], sessions=len(days))
                    if absent:
                        gaps[key] = absent
        bad_eod = any(r['reason'] != 'incomplete_or_future_session' for r in quarantine)
        status = 'ready' if not errors and not any(missing.values()) and not gaps and not bad_eod else ('partial' if records else 'blocked')
        canonical = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records).encode()
        (run/'records.jsonl').write_bytes(canonical)
        atomic_json(run/'quarantine.json', quarantine)
        manifest = dict(schema_version=2, status=status, started_at=started.isoformat(), completed_at=completed.isoformat(),
                        expected_session=str(cutoff), symbols=symbols, sources=sources, records_path='records.jsonl',
                        records_sha256=sha(canonical), rows=len(records), missing_latest=missing,
                        gaps=gaps, coverage=coverage, errors=errors, quarantine_rows=len(quarantine),
                        quarantined_intraday_rows=sum(r['reason']=='incomplete_or_future_session' for r in quarantine),
                        strategy_approved=False, collector_sha256=sha(Path(__file__).read_bytes()))
        atomic_json(run/'manifest.json', manifest)
        summary = {**manifest, 'manifest_path':str((run/'manifest.json').resolve()), 'manifest_sha256':sha((run/'manifest.json').read_bytes())}
        atomic_json(root/'latest_status.json', summary)
        return summary
    finally:
        if client is not None:
            client.close()
        lock.rmdir()


def _bound_file(folder, relative, expected):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError('Invalid flow artifact path')
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('Flow artifact hash mismatch')
    return raw


def verified_records(root=STORE):
    records = []
    for path in sorted((Path(root)/'runs').glob('*/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 2:
            raise ValueError('Unsupported flow manifest schema')
        expected = []
        for source in manifest['sources']:
            raw = _bound_file(path.parent, source['path'], source['sha256'])
            parsed, _ = normalize_chart(raw, source['symbol'], source['investor'], aware(source['observed_at']),
                                        cutoff=date.fromisoformat(manifest['expected_session']))
            for row in parsed:
                row['available_at'] = max(aware(row['observed_at']), aware(manifest['completed_at'])).isoformat()
            expected.extend(parsed)
        raw = _bound_file(path.parent, manifest['records_path'], manifest['records_sha256'])
        actual = [json.loads(line) for line in raw.splitlines() if line]
        if actual != expected or len(actual) != manifest['rows']:
            raise ValueError('Canonical rows do not match raw source/provenance')
        records.extend(actual)
    return records


def load_flows(*, root=STORE, as_of=None):
    """Wide bn-VND compatibility view. Historical features MUST pass as_of.

    Without as_of this is a latest-data view, not a PIT backtest. No legacy or
    matched-only source mixing. Missing foreign/TD values stay NaN, never zero.
    """
    cutoff = aware(as_of)
    rows = [r for r in verified_records(root) if aware(r['available_at']) <= cutoff]
    columns = ['date','symbol','f_buy_val','f_sell_val','f_net_val','f_buy_vol','f_sell_vol','f_net_vol',
               'td_buy_val','td_sell_val','td_net_val','f_available_at','td_available_at','source','trade_scope']
    if not rows:
        return pd.DataFrame(columns=columns)
    panel = pd.DataFrame(rows).sort_values(['available_at','observed_at','raw_sha256'])
    panel = panel.drop_duplicates(['date','symbol','investor','source','trade_scope'], keep='last')
    wide = {}
    for r in panel.to_dict('records'):
        key = (r['date'], r['symbol'])
        item = wide.setdefault(key, dict(date=date.fromisoformat(r['date']), symbol=r['symbol'], source=r['source'], trade_scope=r['trade_scope']))
        prefix = 'f' if r['investor']=='foreign' else 'td'
        for side in ('buy','sell','net'):
            item[f'{prefix}_{side}_val'] = r[f'{side}_value_vnd'] / 1e9
            item[f'{prefix}_{side}_vol'] = r[f'{side}_volume']
        item[f'{prefix}_available_at'] = r['available_at']
    return pd.DataFrame(wide.values()).reindex(columns=columns + ['td_buy_vol','td_sell_vol','td_net_vol']).sort_values(['symbol','date']).reset_index(drop=True)


def snapshot_today(symbols):
    """Deprecated wrapper: dated chart responses instead of host-stamped board."""
    result = collect_flows(symbols)
    if result['status'] != 'ready':
        raise ValueError('Foreign collection incomplete; see data/foreign_flows_v2/latest_status.json')
    return result['rows']


def migrate_legacy(legacy=FOREIGN_DIR, root=STORE, *, now=None):
    """Derived research-only normalization/inventory. Does not overwrite raw."""
    now = aware(now)
    legacy, root = Path(legacy), Path(root)
    folder = root/'legacy'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    normalized, rejected, files = [], [], {}
    names = ['ndtnn','ndtnn_chart','tudoanh_chart','price_board_snapshots',
             'ndtnn_monthly','ndtnn_quarterly','tudoanh_monthly','tudoanh_quarterly']
    for name in names:
        path = legacy/f'{name}.jsonl'
        if not path.exists():
            continue
        raw = path.read_bytes()
        files[name] = dict(sha256=sha(raw), bytes=len(raw))
        for index, line in enumerate(raw.decode('utf-8-sig').splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                symbol = row.get('StockCode',row.get('symbol'))
                symbols_contract([symbol])
                session = source_time(row['TradingDate']).astimezone(VN_TIMEZONE).date() if row.get('TradingDate') else None
                if session and not is_trading_day(session):
                    raise ValueError('nontrading_session')
                if name == 'ndtnn':
                    numeric = values(row, 1_000_000)
                elif name in ('ndtnn_chart','tudoanh_chart'):
                    numeric = values(dict(BuyVal=row['buy_val'],SellVal=row['sell_val'],BuyVol=row['buy_vol'],SellVol=row['sell_vol']),1_000_000_000)
                else:
                    raise ValueError('not_traceable_daily_source')
                normalized.append(dict(schema_version=2, date=str(session) if session else None,
                    stored_date=row['date'], symbol=symbol, investor='proprietary' if name.startswith('tudoanh') else 'foreign',
                    source=name, source_file_sha256=files[name]['sha256'], source_line=index+1,
                    trade_scope='matched_only' if name=='ndtnn' else 'provider_chart_aggregate',
                    observed_at=None, available_at=None, research_only=True,
                    quality='legacy_unknown_availability' if session else 'legacy_unknown_session_and_availability', **numeric))
            except (ValueError, TypeError, KeyError):
                rejected.append(dict(source=name, line=index+1, reason='unverified_or_invalid_legacy'))
    rows_path = folder/'research_only.jsonl'
    rows_path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in normalized), encoding='utf-8')
    atomic_json(folder/'quarantine.json', rejected)
    report = dict(status='research_only', migrated_at=now.isoformat(), normalized_rows=len(normalized), quarantined_rows=len(rejected),
                  files=files, rows_path=str(rows_path.resolve()), rows_sha256=sha(rows_path.read_bytes()),
                  date_recovered_rows=sum(r['date'] is not None for r in normalized), prospective_eligible_rows=0)
    atomic_json(folder/'manifest.json', report)
    return report

"""CI inputs contain only immutable public OHLCV, never application state."""
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from stock_agent.data.reconciliation import verify_snapshot

FIXTURES = Path('tests/fixtures/research')


@pytest.mark.parametrize('name,manifest_hash', [
    ('research-snapshot-v1.zip','21b2deff6694a2a7218c861a681524b91f311617fc1e84de0598fc8001e155d0'),
    ('research-vn100-h1-v1.zip','c9d046ff5d1c336284c2fc8a1bd07abc2e63c84d2493ce508a8d16bf132d7148'),
])
def test_pinned_archive_hashes_source_lineage_and_safe_inventory(tmp_path,name,manifest_hash):
    sums = dict(line.split(maxsplit=1)[::-1] for line in (FIXTURES/'SHA256SUMS').read_text().splitlines())
    archive = FIXTURES/name
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == sums[archive.as_posix()]
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert all('\\' not in n and not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        assert hashlib.sha256(z.read('manifest.json')).hexdigest() == manifest_hash
        manifest = json.loads(z.read('manifest.json'))
        allowed = {'manifest.json'} | {entry[key] for entry in manifest['files'].values() for key in ('path','raw_path')}
        assert {n for n in names if not n.endswith('/')} == allowed
        z.extractall(tmp_path)
    assert verify_snapshot(tmp_path/'manifest.json')['status'] == 'verified'


def test_real_vn100_input_loader_uses_bounded_paired_unions(tmp_path):
    from scripts import universe_expansion as u
    with zipfile.ZipFile(FIXTURES/'research-vn100-h1-v1.zip') as z:
        z.extractall(tmp_path)
    frames,timelines,rules = u.load_inputs(tmp_path/'manifest.json',u.load_protocol())
    assert len(frames) == 105
    assert len(timelines['VN30'].all_members) == 32
    assert len(timelines['VN100'].all_members) == 104
    assert all(frame.date.max() == '2026-06-30' for frame in frames.values())
    assert not rules['ml']['enabled']

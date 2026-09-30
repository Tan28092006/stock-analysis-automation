"""Complete local-file CLI flow; no real account data or orders."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_execution_receipts import event, plan


def lab():
    return importlib.import_module('scripts.execution_audit')


def inputs(tmp_path, plan):
    p, e = tmp_path/'plan.json', tmp_path/'receipts.json'
    p.write_text(json.dumps(plan))
    e.write_text(json.dumps([event(plan,1,'ACCEPTED'),event(plan,2,'FILL')]))
    return p, e


def test_file_audit_hashes_inputs_and_preserves_existing_outputs(tmp_path, plan):
    p, e = inputs(tmp_path,plan)
    before = (p.read_bytes(),e.read_bytes())
    out = tmp_path/'audit.json'
    r=lab().audit_files(p,e,out,as_of='2026-10-01T16:00:00+07:00')
    assert r['state']=='partially_filled' and len(r['input_sha256'])==2
    assert r['source_sha256'] and r['broker_authenticity_verified'] is False
    assert (p.read_bytes(),e.read_bytes())==before
    original=out.read_bytes()
    with pytest.raises(FileExistsError):
        lab().audit_files(p,e,out,as_of='2026-10-01T16:00:00+07:00')
    assert out.read_bytes()==original


@pytest.mark.parametrize('payload', ['{"quantity":100,"quantity":200}', 'NaN', '[]'])
def test_bad_json_never_creates_audit_output(tmp_path,plan,payload):
    p,e=inputs(tmp_path,plan)
    p.write_text(payload)
    out=tmp_path/'audit.json'
    with pytest.raises(ValueError):
        lab().audit_files(p,e,out,as_of='2026-10-01T16:00:00+07:00')
    assert not out.exists()


def test_excessive_input_size_is_rejected_before_parsing(tmp_path,plan):
    p,e=inputs(tmp_path,plan)
    p.write_text(' '*(lab().MAX_INPUT_BYTES+1))
    with pytest.raises(ValueError, match='size'):
        lab().audit_files(p,e,tmp_path/'out.json',as_of='2026-10-01T16:00:00+07:00')


def test_cli_end_to_end_and_malformed_input_exit(tmp_path,plan):
    p,e=inputs(tmp_path,plan)
    out=tmp_path/'audit.json'
    args=[sys.executable,'-m','scripts.execution_audit','--plan',str(p),'--receipts',str(e),
          '--as-of','2026-10-01T16:00:00+07:00','--output',str(out)]
    r=subprocess.run(args,capture_output=True,text=True)
    assert r.returncode==0, r.stderr
    assert json.loads(out.read_text())['state']=='partially_filled'
    assert json.loads(r.stdout)['live_eligible'] is False
    assert subprocess.run(args,capture_output=True,text=True).returncode==2


def test_ci_retains_every_replay_and_adds_receipt_tests():
    workflow=Path('.github/workflows/research-gate.yml').read_text()
    assert 'test_execution_receipts.py' in workflow
    assert 'test_execution_audit_cli.py' in workflow
    for name in ['research_gate','momentum_books','research_timeslices','universe_expansion','momentum_event_suite']:
        assert 'python -m scripts.'+name in workflow

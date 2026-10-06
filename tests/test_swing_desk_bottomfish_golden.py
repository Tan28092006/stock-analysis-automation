"""Swing-mode invariance (golden): adding the bottom-fish mode must not change swing outputs.

The golden file was recorded from the base revision 08dc727 (before the bottom-fish mode) by
running this module with SWING_DESK_WRITE_GOLDEN=1 against an export of that revision.
Ignored fields: `desk_code_hash` (hashes every module in the package, so it changes when modules
are added) and `snapshot_manifest_sha256` / the gate-table sha prefix (manifest.json embeds the
temporary directory path).
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from tests.swing_desk_support import NOW_EOD
from tests.test_swing_desk_check import ready, setup_desk

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "swing_desk_golden_v1.json"
VOLATILE = {"desk_code_hash", "snapshot_manifest_sha256"}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _text(text: str) -> str:
    return re.sub(r"snapshot sha [0-9a-f]{12}", "snapshot sha <sha>", text)


def collect(tmp_path, monkeypatch) -> dict:
    from stock_agent.swing_desk import render, service
    from stock_agent.swing_desk.config import load_config
    out = {}
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    kw = dict(now=NOW_EOD, s2_override=[], record=False)
    cases = {
        "aaa_pass": dict(ticker="AAA"),
        "aaa_price_amount": dict(ticker="AAA", price=None, amount=300_000_000),
        "bbb_warn_ack": dict(ticker="BBB", ack=True),
        "vcb_reason_forum": dict(ticker="VCB", reason="forum", emotion="fomo"),
        "xyz_out_of_scope": dict(ticker="XYZ"),
    }
    for name, args in cases.items():
        ticker = args.pop("ticker")
        r = service.check(paths, ticker, **kw, **args)
        out[name] = {"json": _clean(r), "text": _text(render.render_check(r))}
    close = out["aaa_pass"]["json"]["indicators"]["close"]
    for name, price in (("aaa_price_plus2", round(close * 1.02)), ("aaa_price_plus6", round(close * 1.06))):
        r = service.check(paths, "AAA", price=price, amount=5_000_000, reason="S1", emotion="calm", **kw)
        out[name] = {"json": _clean(r), "text": _text(render.render_check(r))}
    later = datetime(2026, 9, 22, 17, 0, tzinfo=VN_TIMEZONE)
    r = service.check(paths, "AAA", now=later, s2_override=[], record=False)
    out["stale_g0"] = {"json": _clean(r), "text": _text(render.render_check(r))}
    paths.portfolio.write_text("2026-09-01|NAV||1000000000||||||\n2026-09-17|BUY|BBB||20000|1000|S1|PASS|calm||\n"
                               "2026-09-21|SELL|BBB||19000|1000|stop||fear||\n", encoding="utf-8")
    r = service.check(paths, "AAA", **kw)
    out["aaa_revenge_block"] = {"json": _clean(r), "text": _text(render.render_check(r))}
    paths.portfolio.unlink()
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    out["scan_table"] = _text(service.gate_table(state))
    out["desk_v1_sha256"] = hashlib.sha256((ROOT / "configs/swing_desk/desk_v1.json").read_bytes()).hexdigest()
    out["prospective_v1_sha256"] = hashlib.sha256((ROOT / "configs/swing_desk/prospective_v1.json").read_bytes()).hexdigest()
    out["params_hash"] = load_config(ROOT / "configs/swing_desk/desk_v1.json")["params_hash"]
    return json.loads(json.dumps(out, ensure_ascii=False, sort_keys=True, default=str))


def test_swing_outputs_match_base_revision(tmp_path, monkeypatch):
    got = collect(tmp_path, monkeypatch)
    if os.environ.get("SWING_DESK_WRITE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden written")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(got) == set(want)
    for key in sorted(want):
        assert got[key] == want[key], key


def test_swing_check_default_mode_equals_explicit_swing_mode(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    a = service.check(paths, "AAA", now=NOW_EOD, s2_override=[], record=False)
    b = service.check(paths, "AAA", now=NOW_EOD, s2_override=[], record=False, mode="swing")
    assert a == b

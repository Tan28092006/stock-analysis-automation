from datetime import date, datetime
import json

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE


def test_append_builds_chain_and_verifies(tmp_path):
    from stock_agent.swing_desk import reclog
    log = tmp_path / "log.jsonl"
    a = reclog.append(log, {"id": "a", "x": 1})
    b = reclog.append(log, {"id": "b", "x": 2})
    assert a["prev_sha256"] == reclog.GENESIS and b["prev_sha256"] == a["row_sha256"]
    assert reclog.verify_chain(log) == {"ok": True, "rows": 2, "errors": []}
    assert [r["id"] for r in reclog.read_rows(log)] == ["a", "b"]


def test_chain_detects_edit_deletion_and_truncation(tmp_path):
    from stock_agent.swing_desk import reclog
    log = tmp_path / "log.jsonl"
    for i in range(3):
        reclog.append(log, {"id": str(i), "net": -1.0})
    lines = log.read_text(encoding="utf-8").splitlines()

    edited = json.loads(lines[1]); edited["net"] = 5.0
    log.write_text("\n".join([lines[0], json.dumps(edited), lines[2]]) + "\n", encoding="utf-8")
    assert not reclog.verify_chain(log)["ok"]

    log.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")       # delete middle row
    assert not reclog.verify_chain(log)["ok"]

    log.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")                  # drop the last row
    res = reclog.verify_chain(log)
    assert not res["ok"] and "head" in " ".join(res["errors"])


def test_append_refuses_broken_chain(tmp_path):
    from stock_agent.swing_desk import reclog
    log = tmp_path / "log.jsonl"
    reclog.append(log, {"id": "a"})
    log.write_text(log.read_text(encoding="utf-8").replace('"a"', '"z"'), encoding="utf-8")
    with pytest.raises(ValueError, match="chain"):
        reclog.append(log, {"id": "b"})


def _row(**over):
    row = {"ticker": "HPG", "action": "BUY", "setup": ["S1"], "gate": "PASS", "gate_failed": [],
           "entry": {"type": "limit", "limit_price": 27400, "valid_session": "2026-09-22"},
           "stop": 26578, "stop_basis": "user_3pct", "target": 28800, "max_hold_sessions": 8,
           "earliest_sell": "2026-09-24 PM", "size": {"qty": 1800, "risk_vnd": 2268000, "risk_pct_nav": 1.0},
           "p_win_stated": 0.55, "sentiment": {"score": 1, "confidence": "low", "citations": ["S-01"]},
           "invalidation": "VNINDEX close < EMA50", "data_as_of": "2026-09-21", "snapshot_sha256": "ab" * 32}
    row.update(over)
    return row


META = {"snapshot_manifest_sha256": "ab" * 32, "prompt_hash": "p" * 16, "model_id": "test-model",
        "n_rollouts": 4, "adjudication": {"base": "a1", "open_forks": 0}}


def test_freeze_refuses_after_0900_of_target_session(tmp_path):
    from stock_agent.swing_desk import reclog
    log = tmp_path / "rec.jsonl"
    ok = datetime(2026, 9, 22, 8, 59, tzinfo=VN_TIMEZONE)
    late = datetime(2026, 9, 22, 9, 0, tzinfo=VN_TIMEZONE)
    rows = reclog.freeze(log, [_row()], source="desk", target_session=date(2026, 9, 22),
                         session_as_of="2026-09-21", now=ok, meta=META)
    assert rows[0]["source"] == "desk" and rows[0]["frozen_before"].startswith("2026-09-22T09:00")
    with pytest.raises(ValueError, match="09:00"):
        reclog.freeze(log, [_row()], source="desk", target_session=date(2026, 9, 22),
                      session_as_of="2026-09-21", now=late, meta=META)
    assert reclog.verify_chain(log)["rows"] == 1


@pytest.mark.parametrize("missing", ["ticker", "stop", "p_win_stated", "entry", "max_hold_sessions"])
def test_freeze_enforces_required_fields_for_analyst_arms(tmp_path, missing):
    from stock_agent.swing_desk import reclog
    row = _row(); row.pop(missing)
    with pytest.raises(ValueError, match=missing):
        reclog.freeze(tmp_path / "r.jsonl", [row], source="single", target_session=date(2026, 9, 22),
                      session_as_of="2026-09-21", now=datetime(2026, 9, 21, 20, tzinfo=VN_TIMEZONE), meta=META)
    assert not (tmp_path / "r.jsonl").exists()


def test_freeze_rejects_unknown_source_and_stale_session(tmp_path):
    from stock_agent.swing_desk import reclog
    now = datetime(2026, 9, 21, 20, tzinfo=VN_TIMEZONE)
    with pytest.raises(ValueError, match="source"):
        reclog.freeze(tmp_path / "r.jsonl", [_row()], source="oracle", target_session=date(2026, 9, 22),
                      session_as_of="2026-09-21", now=now, meta=META)
    with pytest.raises(ValueError, match="data_as_of"):
        reclog.freeze(tmp_path / "r.jsonl", [_row(data_as_of="2026-09-18")], source="desk",
                      target_session=date(2026, 9, 22), session_as_of="2026-09-21", now=now, meta=META)

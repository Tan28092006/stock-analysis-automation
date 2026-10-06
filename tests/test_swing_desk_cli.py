import json

from tests.swing_desk_support import NOW_EOD
from tests.test_swing_desk_check import ready, setup_desk

NOW = NOW_EOD.isoformat()


def cli(paths, *args):
    from stock_agent.swing_desk.__main__ import main
    return main(["--repo", str(paths.repo), "--now", NOW, *args])


def test_cli_check_render_and_exit_code(tmp_path, monkeypatch, capsys):
    from stock_agent.swing_desk import data as dl
    monkeypatch.setattr(dl, "s2_symbols", lambda *a, **k: ([], "OK"))
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    code = cli(paths, "check", "AAA")
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("AAA — PASS")
    assert cli(paths, "check", "XYZ", "--no-log") == 2


def test_cli_refresh_scan_freeze_score_and_verify(tmp_path, monkeypatch, capsys):
    from stock_agent.swing_desk import data as dl, refresh
    monkeypatch.setattr(dl, "s2_symbols", lambda *a, **k: ([], "OK"))
    monkeypatch.setattr(refresh, "run_process", lambda cmd, cwd: (_ for _ in ()).throw(AssertionError("no network")))
    paths, _ = setup_desk(tmp_path, monkeypatch)
    assert cli(paths, "refresh") == 0
    assert json.loads(capsys.readouterr().out)["snapshot_action"] == "reused"
    assert cli(paths, "scan") == 0
    assert "| AAA | PASS | S1 |" in capsys.readouterr().out
    assert cli(paths, "freeze", "--source", "baseline") == 0
    assert json.loads(capsys.readouterr().out)["baseline"] == ["AAA"]
    assert cli(paths, "score") == 0
    assert "NOT PROVEN" in capsys.readouterr().out
    assert cli(paths, "verify-log") == 0


def test_cli_status_is_read_only(tmp_path, monkeypatch, capsys):
    paths, _ = setup_desk(tmp_path, monkeypatch)
    before = sorted(p for p in paths.repo.rglob("*"))
    assert cli(paths, "status") == 0
    st = json.loads(capsys.readouterr().out)
    assert st["snapshot"]["ok"] and st["snapshot"]["as_of"] == "2026-09-21"
    assert st["foreign_flows"]["status"] == "UNVERIFIED"
    assert sorted(p for p in paths.repo.rglob("*")) == before


def test_cli_events_marks_g10(tmp_path, monkeypatch, capsys):
    paths, _ = setup_desk(tmp_path, monkeypatch)
    assert cli(paths, "events", "AAA", "--session", "2026-09-22", "--event", "2026-09-23:ex_dividend",
               "--source", "https://example.invalid/cbtt") == 0
    data = json.loads(paths.events.read_text(encoding="utf-8"))
    assert data["checked"]["AAA"] == "2026-09-22" and data["events"][0]["type"] == "ex_dividend"

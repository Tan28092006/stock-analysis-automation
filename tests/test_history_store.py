"""10-year history store: quarantine, immutability, verification, pacing, status.

Fakes only: no network. Every HTTP call goes through an injected `post`.
"""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_agent.data import history_store as hs
from stock_agent.data import reconciliation as rec

NOW = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc)  # 17:00 VN, completed session 2026-09-21
WEEK = ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21")


def bars(symbol="FPT", dates=WEEK, close=101000):
    return [{"symbol": symbol, "t": [str(int(pd.Timestamp(d, tz="UTC").timestamp())) for d in dates],
             "o": [100000] * len(dates), "h": [102000] * len(dates),
             "l": [99000] * len(dates), "c": [close] * len(dates), "v": [1000] * len(dates)}]


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakePost:
    """Return per-symbol payloads or HTTP statuses; record each call time."""

    def __init__(self, payloads, statuses=None, clock=None, interrupt_on=None):
        self.payloads, self.statuses = payloads, dict(statuses or {})
        self.clock, self.interrupt_on = clock, interrupt_on
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        symbol = json["symbols"][0]
        self.calls.append((symbol, self.clock.now if self.clock else None))
        if symbol == self.interrupt_on:
            raise KeyboardInterrupt
        queue = self.statuses.get(symbol)
        if queue:
            status = queue.pop(0)
            if status != 200:
                return SimpleNamespace(status_code=status, content=b'{"error":"x"}')
        body = self.payloads[symbol]
        content = body if isinstance(body, bytes) else __import__("json").dumps(body).encode()
        return SimpleNamespace(status_code=200, content=content)


def build(tmp_path, symbols, payloads, statuses=None, store_id="t1", start=date(2026, 9, 14),
          as_of=date(2026, 9, 21), **kwargs):
    clock = kwargs.pop("clock", None) or FakeClock()
    post = kwargs.pop("post", None) or FakePost(payloads, statuses, clock)
    store, manifest = hs.build_store(symbols, tmp_path / "history", start=start, as_of=as_of, now=NOW,
                                     post=post, clock=clock.clock, sleep=clock.sleep, store_id=store_id, **kwargs)
    return store, manifest, post, clock


# --- non-strict parse ---------------------------------------------------------------------------

def test_quarantine_keeps_good_rows_and_never_repairs():
    data = bars(dates=WEEK + ("2026-09-19",))  # Saturday row -> non_session
    data[0]["o"][1] = 110000  # open above high -> inconsistent
    data[0]["c"][3] = 0  # non-positive close
    frame, issues = hs.parse_history_rows(data, "FPT", date(2026, 9, 14), date(2026, 9, 21))
    assert [str(d) for d in frame.date] == ["2026-09-14", "2026-09-16", "2026-09-18", "2026-09-21"]
    assert frame.close.tolist() == [101000] * 4 and frame.open.tolist() == [100000] * 4
    by_date = {q["date"]: q for q in issues["quarantined"]}
    assert set(by_date) == {"2026-09-15", "2026-09-17", "2026-09-19"}
    assert "high_below_ohlc" in by_date["2026-09-15"]["reasons"]
    assert by_date["2026-09-15"]["open"] == 110000  # raw value preserved, not fixed
    assert "non_positive_price" in by_date["2026-09-17"]["reasons"]
    assert by_date["2026-09-19"]["reasons"] == ["non_session"]
    assert not rec._invalid(frame).any()


def test_duplicate_dates_and_bad_values_are_quarantined_not_dropped_silently():
    data = bars(dates=("2026-09-14", "2026-09-14", "2026-09-15", "2026-09-16"))
    data[0]["v"][2] = None
    data[0]["t"].append("not-a-time")
    for key in "ohlcv":
        data[0][key].append(1)
    frame, issues = hs.parse_history_rows(data, "FPT", date(2026, 9, 14), date(2026, 9, 21))
    assert [str(d) for d in frame.date] == ["2026-09-16"]
    reasons = sorted(r for q in issues["quarantined"] for r in q["reasons"])
    assert reasons == ["bad_timestamp", "duplicate_date", "duplicate_date", "non_finite"]


def test_clean_payload_matches_strict_parse_byte_for_byte():
    data = bars()
    strict = rec.parse_vci_history(data, "FPT", date(2026, 9, 14), date(2026, 9, 21))
    frame, issues = hs.parse_history_rows(data, "FPT", date(2026, 9, 14), date(2026, 9, 21))
    assert issues["quarantined"] == []
    assert hs.csv_bytes(frame) == strict.to_csv(index=False, lineterminator="\n").encode("utf-8")


def test_out_of_window_rows_are_counted_not_quarantined():
    frame, issues = hs.parse_history_rows(bars(), "FPT", date(2026, 9, 15), date(2026, 9, 18))
    assert len(frame) == 4 and issues["quarantined"] == [] and issues["excluded_out_of_window"] == 2


@pytest.mark.parametrize("kind", ["wrong_symbol", "empty", "ragged", "all_bad"])
def test_structural_or_empty_response_fails_the_symbol(kind):
    data = bars()
    if kind == "wrong_symbol": data[0]["symbol"] = "OTHER"
    if kind == "empty": data = []
    if kind == "ragged": data[0]["c"].pop()
    if kind == "all_bad": data[0]["c"] = [0] * len(WEEK)
    with pytest.raises(ValueError):
        hs.parse_history_rows(data, "FPT", date(2026, 9, 14), date(2026, 9, 21))


# --- store build ---------------------------------------------------------------------------------

def test_store_marks_partial_symbol_and_hashes_every_file(tmp_path):
    vnindex = bars("VNINDEX", close=1650.5)
    for key, value in (("o", 1640.0), ("h", 1660.0), ("l", 1630.0)):
        vnindex[0][key] = [value] * len(WEEK)
    vnindex[0]["h"][2] = 1645.0  # high < close: invalid row in the middle: only it is lost
    store, manifest, _, _ = build(tmp_path, ["FPT", "VNINDEX"], {"FPT": bars(), "VNINDEX": vnindex})
    assert store.parent == tmp_path / "history" and store.name.endswith("_t1")
    assert manifest["kind"] == "price_history_store" and manifest["status"] == "partial"
    assert manifest["counts"] == {"ok": 1, "partial": 1, "failed": 0}
    fpt, idx = manifest["symbols"]["FPT"], manifest["symbols"]["VNINDEX"]
    assert fpt["status"] == "ok" and fpt["rows"] == 6 and fpt["quarantined_rows"] == 0
    assert idx["status"] == "partial" and idx["rows"] == 5 and idx["quarantined_rows"] == 1
    assert idx["quarantine_reasons"] == {"high_below_ohlc": 1} and idx["unit"] == "index_points"
    assert (idx["first_date"], idx["last_date"]) == ("2026-09-14", "2026-09-21")
    for entry in manifest["symbols"].values():
        for path_key, hash_key in (("raw_path", "raw_sha256"), ("path", "sha256"), ("issues_path", "issues_sha256")):
            assert hashlib.sha256((store / entry[path_key]).read_bytes()).hexdigest() == entry[hash_key]
        assert entry["request"]["symbols"] and entry["fetched_at"]
    assert "unverified" in manifest["adjustment_basis"] and "survivorship" in manifest["universe_policy"]
    assert json.loads((store / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert hs.verify_store(store)["status"] == "partial"


def test_store_directory_is_new_and_never_reused(tmp_path):
    build(tmp_path, ["FPT"], {"FPT": bars()})
    with pytest.raises(FileExistsError):
        build(tmp_path, ["FPT"], {"FPT": bars()})


def test_as_of_after_completed_session_is_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="completed"):
        build(tmp_path, ["FPT"], {"FPT": bars()}, as_of=date(2026, 9, 22))
    assert not (tmp_path / "history").exists()


def test_client_error_fails_only_that_symbol_without_retry(tmp_path):
    store, manifest, post, _ = build(tmp_path, ["AAA", "FPT"], {"FPT": bars()}, statuses={"AAA": [404]})
    assert [c[0] for c in post.calls] == ["AAA", "FPT"]
    assert manifest["symbols"]["AAA"]["status"] == "failed" and "404" in manifest["symbols"]["AAA"]["error"]
    assert manifest["symbols"]["FPT"]["status"] == "ok" and manifest["status"] == "partial"
    hs.verify_store(store)


def test_rate_limit_is_not_retried_and_stops_further_calls(tmp_path):
    _, manifest, post, _ = build(tmp_path, ["AAA", "FPT", "VNINDEX"], {}, statuses={"AAA": [429]})
    assert [c[0] for c in post.calls] == ["AAA"]
    assert all(e["status"] == "failed" for e in manifest["symbols"].values())
    assert "not attempted" in manifest["symbols"]["VNINDEX"]["error"]
    assert manifest["status"] == "failed"


def test_transient_server_error_is_retried_with_bounded_attempts(tmp_path):
    _, manifest, post, _ = build(tmp_path, ["FPT"], {"FPT": bars()}, statuses={"FPT": [503, 200]})
    assert [c[0] for c in post.calls] == ["FPT", "FPT"]
    assert manifest["symbols"]["FPT"]["status"] == "ok" and manifest["symbols"]["FPT"]["attempts"] == 2
    _, manifest, post, _ = build(tmp_path, ["FPT"], {"FPT": bars()}, statuses={"FPT": [503, 503, 503, 503]},
                                 store_id="t2")
    assert len(post.calls) == 3 and manifest["symbols"]["FPT"]["status"] == "failed"


def test_pacing_is_at_least_3_1_seconds_between_every_call(tmp_path):
    clock = FakeClock()
    symbols = ["AAA", "BBB", "CCC", "FPT"]
    payloads = {s: bars(s) for s in symbols}
    _, manifest, post, _ = build(tmp_path, symbols, payloads, statuses={"BBB": [502, 200]}, clock=clock)
    times = [t for _, t in post.calls]
    assert len(times) == 5 and manifest["counts"]["ok"] == 4
    assert all(b - a >= hs.PACING_SECONDS - 1e-9 for a, b in zip(times, times[1:]))
    assert hs.PACING_SECONDS >= 3.1


def test_manifest_is_written_last_and_atomically(tmp_path, monkeypatch):
    written = []
    original = hs._write_bytes
    monkeypatch.setattr(hs, "_write_bytes", lambda path, data: (written.append(Path(path).name), original(path, data)))
    store, _, _, _ = build(tmp_path, ["FPT", "VNINDEX"], {"FPT": bars(), "VNINDEX": bars("VNINDEX")})
    assert written[-1] == "manifest.json.tmp" and written.count("manifest.json.tmp") == 1
    assert sorted(p.name for p in store.iterdir()) == ["issues", "manifest.json", "prices", "raw"]


def test_crash_mid_run_leaves_no_manifest_and_status_ignores_it(tmp_path):
    clock = FakeClock()
    post = FakePost({"FPT": bars()}, clock=clock, interrupt_on="VNINDEX")
    with pytest.raises(KeyboardInterrupt):
        build(tmp_path, ["FPT", "VNINDEX"], {}, post=post, clock=clock)
    (store,) = (tmp_path / "history").iterdir()
    assert (store / "raw" / "FPT.json").exists()
    assert not (store / "manifest.json").exists() and not (store / "manifest.json.tmp").exists()
    report = hs.store_status(tmp_path / "history", now=NOW)
    assert report["store"] is None and report["incomplete_dirs"] == [store.name] and report["stale"] is True


def test_failed_atomic_replace_leaves_no_manifest(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(hs.os, "replace", boom)
    with pytest.raises(OSError):
        build(tmp_path, ["FPT"], {"FPT": bars()})
    (store,) = (tmp_path / "history").iterdir()
    assert not (store / "manifest.json").exists() and not (store / "manifest.json.tmp").exists()


# --- verify ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["prices/FPT.csv", "raw/FPT.json", "issues/FPT.json"])
def test_verify_detects_tampered_files(tmp_path, target):
    store, _, _, _ = build(tmp_path, ["FPT"], {"FPT": bars()})
    assert hs.verify_store(store / "manifest.json")["symbols_checked"] == 1
    path = store / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        hs.verify_store(store)


def test_verify_detects_csv_rewritten_with_matching_manifest_hash(tmp_path):
    store, manifest, _, _ = build(tmp_path, ["FPT"], {"FPT": bars()})
    csv = store / "prices/FPT.csv"
    forged = csv.read_bytes().replace(b"101000", b"105000")
    csv.write_bytes(forged)
    manifest["symbols"]["FPT"]["sha256"] = hashlib.sha256(forged).hexdigest()
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        hs.verify_store(store)


def test_verify_rejects_paths_escaping_the_store(tmp_path):
    store, manifest, _, _ = build(tmp_path, ["FPT"], {"FPT": bars()})
    manifest["symbols"]["FPT"]["path"] = "../../outside.csv"
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escaped"):
        hs.verify_store(store)


# --- coverage --------------------------------------------------------------------------------------

def test_coverage_separates_verified_calendar_from_unverified_pre_2021():
    present = [date(2020, 12, 30), date(2020, 12, 31), date(2021, 1, 4), date(2021, 1, 6)]
    report = hs.coverage_report("FPT", present, date(2021, 1, 6))
    verified, pre = report["verified_calendar"], report["unverified_pre_calendar"]
    assert verified["start"] == "2021-01-01" and verified["expected_sessions"] == 3
    assert verified["present_sessions"] == 2 and verified["missing_dates_sample"] == ["2021-01-05"]
    assert pre["status"] == "UNVERIFIED" and pre["present_rows"] == 2 and pre["weekdays"] == 2
    assert pre["end"] == "2020-12-31"


# --- universe ----------------------------------------------------------------------------------------

def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_universe_union_is_deduplicated_and_labelled(tmp_path):
    _write(tmp_path / hs.CURRENT_VN30, {"symbols": ["ACB", "FPT"]})
    _write(tmp_path / hs.VN30_HISTORY_CONFIGS[0], {"initial_members": ["ACB", "FPT", "KDH"],
                                                    "changes": [{"add": ["BCM"], "remove": ["KDH"]}]})
    _write(tmp_path / hs.VN30_HISTORY_CONFIGS[1], {"initial_members": ["ACB", "BCM"], "changes": []})
    _write(tmp_path / hs.VN100_CONFIG, {"initial_members": ["ACB", "PNJ"], "changes": [{"add": ["NVL"], "remove": []}]})
    symbols, labels = hs.resolve_universe(["all"], extra=["pnj", "tpb", "FPT"], repo=tmp_path)
    assert symbols == ["ACB", "BCM", "FPT", "KDH", "NVL", "PNJ", "TPB", "VNINDEX"]
    assert len(symbols) == len(set(symbols))
    assert labels["FPT"] == ["extra", "vn30", "vn30-history"] and labels["VNINDEX"] == ["vnindex"]
    assert labels["PNJ"] == ["extra", "vn100"]
    assert hs.resolve_universe(["vn30", "vnindex"], repo=tmp_path)[0] == ["ACB", "FPT", "VNINDEX"]
    with pytest.raises(ValueError):
        hs.resolve_universe(["all"], extra=["BAD SYMBOL"], repo=tmp_path)
    with pytest.raises(ValueError):
        hs.resolve_universe(["sp500"], repo=tmp_path)


def test_real_repo_universe_contains_current_vn30_history_vn100_and_index():
    symbols, labels = hs.resolve_universe(["all"], extra=["PNJ", "TPB"])
    current = json.loads((hs.REPO / hs.CURRENT_VN30).read_text(encoding="utf-8"))["symbols"]
    assert set(current) <= set(symbols) and "VNINDEX" in symbols
    assert {"NVL", "PDR", "KDH", "POW"} <= set(symbols)  # ever-members from 2022 configs
    assert len(symbols) == len(set(symbols)) == len(labels)


# --- status ------------------------------------------------------------------------------------------

def test_status_reports_newest_store_and_staleness(tmp_path):
    payloads = {"FPT": bars(dates=WEEK[:-1])}
    build(tmp_path, ["FPT"], payloads, store_id="old", as_of=date(2026, 9, 17))
    build(tmp_path, ["FPT"], payloads, store_id="new", as_of=date(2026, 9, 18))
    report = hs.store_status(tmp_path / "history", now=NOW)
    assert report["store"].endswith("_new") and report["as_of"] == "2026-09-18"
    assert report["completed_session"] == "2026-09-21" and report["sessions_behind"] == 1
    assert report["stale"] is True and report["counts"] == {"ok": 1, "partial": 0, "failed": 0}
    fresh = hs.store_status(tmp_path / "history", now=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc))
    assert fresh["stale"] is False and fresh["sessions_behind"] == 0


def test_status_with_no_store_is_stale(tmp_path):
    report = hs.store_status(tmp_path / "missing", now=NOW)
    assert report["store"] is None and report["stale"] is True


# --- CLI ---------------------------------------------------------------------------------------------

def test_cli_dry_run_makes_no_network_call(tmp_path, capsys, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network used")
    monkeypatch.setattr(hs.requests, "post", no_network)
    code = hs.main(["fetch", "--years", "10", "--universe", "all", "--symbols", "PNJ", "TPB",
                    "--out", str(tmp_path / "h"), "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["dry_run"] is True and "VNINDEX" in out["symbols"]
    assert out["estimated_seconds_min"] == pytest.approx(len(out["symbols"]) * hs.PACING_SECONDS)
    assert not (tmp_path / "h").exists()


# --- git hygiene -------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("path,ignored", [
    ("data/history/20261006T100000Z_ab12cd34/manifest.json", True),
    ("data/history/20261006T100000Z_ab12cd34/prices/FPT.csv", True),
    ("stock_agent/data/history_store.py", False),
])
def test_history_store_output_is_git_ignored(path, ignored):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo)
    assert (result.returncode == 0) is ignored, path

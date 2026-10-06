"""Desk long-window history: store selection, consistency check, fallback and provenance (offline)."""
from dataclasses import replace
import json

import numpy as np
import pytest

from tests.swing_desk_history_support import history_desk, tail, write_store
from tests.swing_desk_support import AS_OF, market_frames

AS = str(AS_OF)


def _resolver(paths):
    from stock_agent.swing_desk import history as hist
    hist.clear_cache()
    return hist.HistoryResolver(paths, AS)


# ------------------------------------------------------------------ selection

def test_store_series_used_when_verified_matching_and_consistent(tmp_path, monkeypatch):
    paths, _, long, snap, store = history_desk(tmp_path, monkeypatch)
    df, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert prov["history_source"] == "history_store" and prov["history_mode"] == "on"
    assert prov["history_store_id"] == store.name
    assert prov["history_store_manifest_sha256"] == __import__("hashlib").sha256(
        (store / "manifest.json").read_bytes()).hexdigest()
    assert prov["history_rows"] == 600 and prov["history_first_date"] == str(long["AAA"]["date"].iloc[0])
    assert prov["history_mismatch"] is False and prov["history_data_issues"] == []
    assert not [n for n in prov["history_notes"] if n["level"] == "WARN"]
    assert len(df) == 600 and list(df["date"].tail(150)) == list(snap["AAA"]["date"])


def test_newest_verified_store_with_matching_as_of_wins(tmp_path, monkeypatch):
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, store_id="old",
                                           completed_at="2026-09-21T10:00:00+00:00")
    write_store(paths.history_root, long, store_id="new", completed_at="2026-09-21T11:00:00+00:00")
    other = {s: df.iloc[:-1].reset_index(drop=True) for s, df in long.items()}   # as_of 2026-09-18
    write_store(paths.history_root, other, store_id="otherday", as_of=other["AAA"]["date"].iloc[-1],
                completed_at="2026-09-21T12:00:00+00:00")
    _, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert prov["history_store_id"].endswith("_new")


def test_store_that_fails_verify_is_skipped_and_reported(tmp_path, monkeypatch):
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, store_id="old",
                                           completed_at="2026-09-21T10:00:00+00:00")
    bad = write_store(paths.history_root, long, store_id="new", completed_at="2026-09-21T11:00:00+00:00")
    csv = bad / "prices" / "AAA.csv"
    csv.write_bytes(csv.read_bytes().replace(b"\n", b"\n ", 1))           # tamper: hash mismatch
    res = _resolver(paths)
    _, prov = res.series("BBB", snap["BBB"])
    assert prov["history_store_id"].endswith("_old")
    skipped = res.summary({"BBB": prov})["skipped_stores"]
    assert [s["store_id"] for s in skipped] == [bad.name] and "hash mismatch" in skipped[0]["reason"]


def test_incomplete_store_without_manifest_is_ignored(tmp_path, monkeypatch):
    paths, _, _, snap, store = history_desk(tmp_path, monkeypatch)
    (paths.history_root / "20260921T110000Z_running" / "prices").mkdir(parents=True)
    _, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert prov["history_store_id"] == store.name


# ------------------------------------------------------------------ fallback

def _warn(prov):
    return [n["text"] for n in prov["history_notes"] if n["level"] == "WARN"]


def test_no_store_falls_back_to_snapshot_with_warn(tmp_path, monkeypatch):
    paths, _, _, snap, _ = history_desk(tmp_path, monkeypatch, store=False)
    df, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert df is None and prov["history_source"] == "snapshot_650d" and prov["history_store_id"] is None
    assert prov["history_rows"] == 150 and prov["history_first_date"] == str(snap["AAA"]["date"].iloc[0])
    assert _warn(prov) and "no verified history store" in _warn(prov)[0]


def test_store_with_other_as_of_is_not_used(tmp_path, monkeypatch):
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, store=False)
    other = {s: df.iloc[:-1].reset_index(drop=True) for s, df in long.items()}
    write_store(paths.history_root, other, as_of=other["AAA"]["date"].iloc[-1])
    df, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert df is None and prov["history_source"] == "snapshot_650d" and _warn(prov)


def test_failed_symbol_falls_back_with_warn_others_use_store(tmp_path, monkeypatch):
    paths, _, _, snap, store = history_desk(tmp_path, monkeypatch, fail=("AAA",))
    res = _resolver(paths)
    df_a, a = res.series("AAA", snap["AAA"])
    df_b, b = res.series("BBB", snap["BBB"])
    assert df_a is None and a["history_source"] == "snapshot_650d" and "failed" in _warn(a)[0]
    assert b["history_source"] == "history_store" and df_b is not None
    summary = res.summary({"AAA": a, "BBB": b})
    assert summary["per_ticker"]["AAA"]["history_source"] == "snapshot_650d"
    assert summary["history_source"] == "per_ticker" and summary["stores"] == {store.name: b["history_store_manifest_sha256"]}


def test_symbol_absent_from_store_falls_back(tmp_path, monkeypatch):
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, store=False)
    write_store(paths.history_root, {s: df for s, df in long.items() if s != "VCB"})
    df, prov = _resolver(paths).series("VCB", snap["VCB"])
    assert df is None and prov["history_source"] == "snapshot_650d" and "not in" in _warn(prov)[0]


def test_older_store_is_used_for_a_symbol_that_failed_in_the_newest(tmp_path, monkeypatch):
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, store_id="old",
                                           completed_at="2026-09-21T10:00:00+00:00")
    write_store(paths.history_root, long, store_id="new", fail=("AAA",), completed_at="2026-09-21T11:00:00+00:00")
    res = _resolver(paths)
    assert res.series("AAA", snap["AAA"])[1]["history_store_id"].endswith("_old")
    assert res.series("BBB", snap["BBB"])[1]["history_store_id"].endswith("_new")


def test_quarantined_rows_stay_out_and_are_listed(tmp_path, monkeypatch):
    def edit(sym, body):
        if sym == "BBB":
            body[0]["h"][10] = round(body[0]["c"][10] * 0.999, 2)   # high below close -> quarantined
        return body
    paths, _, long, snap, _ = history_desk(tmp_path, monkeypatch, edit=edit)
    df, prov = _resolver(paths).series("BBB", snap["BBB"])
    bad_day = str(long["BBB"]["date"].iloc[10])
    assert prov["history_source"] == "history_store" and prov["history_rows"] == 599
    assert bad_day not in {str(d) for d in df["date"]}
    assert prov["history_data_issues"] == [{"date": bad_day, "reasons": ["high_below_ohlc"]}]


def test_history_off_never_reads_the_store(tmp_path, monkeypatch):
    paths, _, _, snap, _ = history_desk(tmp_path, monkeypatch, history="off")
    df, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert df is None and prov["history_mode"] == "off" and prov["history_source"] == "snapshot_650d"
    assert not _warn(prov)


# ------------------------------------------------------------------ consistency

def test_readjusted_store_series_is_a_mismatch_not_stitched(tmp_path, monkeypatch):
    def edit(sym, body):
        if sym == "BBB":
            for k in ("o", "h", "l", "c"):
                body[0][k] = [round(v * 0.98, 2) for v in body[0][k]]   # corporate-action re-adjustment
        return body
    paths, _, _, snap, _ = history_desk(tmp_path, monkeypatch, edit=edit)
    df, prov = _resolver(paths).series("BBB", snap["BBB"])
    assert df is None and prov["history_source"] == "snapshot_650d" and prov["history_mismatch"] is True
    assert "history_mismatch" in _warn(prov)[0]
    assert prov["history_mismatch_detail"]["value_diffs"] > 0


def test_one_tick_difference_is_a_mismatch(tmp_path, monkeypatch):
    def edit(sym, body):
        if sym == "AAA":
            body[0]["v"][-3] = body[0]["v"][-3] + 100
        return body
    paths, _, _, snap, _ = history_desk(tmp_path, monkeypatch, edit=edit)
    _, prov = _resolver(paths).series("AAA", snap["AAA"])
    assert prov["history_mismatch"] is True
    assert prov["history_mismatch_detail"]["first_diff"]["field"] == "volume"


def test_missing_session_inside_the_overlap_is_a_mismatch(tmp_path, monkeypatch):
    long = market_frames(n=600)
    gap = {s: (df.drop(index=len(df) - 20).reset_index(drop=True) if s == "VCB" else df) for s, df in long.items()}
    paths, _, _, snap, _ = history_desk(tmp_path, monkeypatch, store=False)
    write_store(paths.history_root, gap)
    _, prov = _resolver(paths).series("VCB", snap["VCB"])
    assert prov["history_mismatch"] is True
    assert prov["history_mismatch_detail"]["sessions_missing_in_store"] == [str(long["VCB"]["date"].iloc[-20])]

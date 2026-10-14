import sqlite3

import pandas as pd
import pytest

from stock_agent.research_ml.catalog import Catalog, CatalogError, Fault

TZ = "Asia/Ho_Chi_Minh"
SESSIONS = ["2026-01-05", "2026-01-06"]


def frame(close=(10, 11), sessions=SESSIONS):
    return pd.DataFrame([dict(symbol="AAA", session_date=s, provider="p", vintage=1, open=10, high=12,
                              low=9, close=c, volume=100, available_at=f"{s} 16:00:00+07:00")
                         for s, c in zip(sessions, close)])


def pub(cat, f=None, **kw):
    args = dict(request_key="k", provider="p", scope="AAA", fetched_at="t", committed_at="t",
                expected_sessions=SESSIONS)
    args.update(kw)
    return cat.publish_prices(frame() if f is None else f, **args)


def cut(day):
    return pd.Timestamp(f"{day} 23:00", tz=TZ)


def test_idempotent_duplicate_and_correction_vintage(tmp_path):
    cat = Catalog(tmp_path)
    assert pub(cat)["new_vintages"] == 2
    again = pub(cat)
    assert again["new_vintages"] == 0 and again["duplicates"] == 2
    fixed = frame(close=(10, 10.5))
    fixed["available_at"] = "2026-01-08 10:00:00+07:00"
    assert pub(cat, fixed)["new_vintages"] == 1
    assert cat.read_prices(cut("2026-01-07")).close.tolist() == [10, 11]  # old knowledge preserved
    last = cat.read_prices(cut("2026-01-09")).sort_values("vintage")
    assert last.vintage.tolist() == [1, 1, 2] and last.close.iloc[-1] == 10.5


def test_partial_generation_is_not_ready_and_hidden(tmp_path):
    cat = Catalog(tmp_path)
    out = pub(cat, frame(close=(10,), sessions=SESSIONS[:1]))
    assert out["ready"] is False and out["missing_sessions"] == ["2026-01-06"]
    assert cat.read_prices(cut("2026-01-09")).empty
    assert len(cat.read_prices(cut("2026-01-09"), require_ready=False)) == 1
    assert cat.coverage().status.iloc[0] == "partial"


@pytest.mark.parametrize("fault", ["after_raw", "before_commit"])
def test_crash_before_commit_leaves_nothing_visible(tmp_path, fault):
    cat = Catalog(tmp_path)
    with pytest.raises(Fault):
        pub(cat, fault=fault)
    assert cat.read_prices(cut("2026-01-09"), require_ready=False).empty
    assert cat.coverage().empty
    assert pub(cat)["new_vintages"] == 2  # retry succeeds, orphan raw is reused


def test_missing_or_corrupt_raw_after_commit_fails_closed(tmp_path):
    cat = Catalog(tmp_path)
    pub(cat)
    raw = next((tmp_path / "raw").iterdir())
    raw.write_bytes(b"tampered")
    with pytest.raises(CatalogError, match="corrupt"):
        cat.read_prices(cut("2026-01-09"))
    raw.unlink()
    with pytest.raises(CatalogError, match="missing"):
        cat.read_prices(cut("2026-01-09"))


def test_rebuild_projection_matches_original(tmp_path):
    cat = Catalog(tmp_path)
    pub(cat)
    fixed = frame(close=(10, 10.5))
    fixed["available_at"] = "2026-01-08 10:00:00+07:00"
    pub(cat, fixed)
    before = cat.read_prices(cut("2026-02-01")).drop(columns="generation_id")
    with sqlite3.connect(cat.db_path) as c:
        c.execute("DELETE FROM price_bar_vintage")
    assert cat.rebuild_projection() == 3
    after = cat.read_prices(cut("2026-02-01")).drop(columns="generation_id")
    pd.testing.assert_frame_equal(before.sort_values(["session_date", "vintage"]).reset_index(drop=True),
                                  after.sort_values(["session_date", "vintage"]).reset_index(drop=True))


def test_backup_restore_round_trip_and_detects_damage(tmp_path):
    cat = Catalog(tmp_path / "live")
    pub(cat)
    cat.backup(tmp_path / "bak")
    restored = Catalog.restore(tmp_path / "bak", tmp_path / "restored")
    assert len(restored.read_prices(cut("2026-01-09"))) == 2
    next((tmp_path / "bak" / "raw").iterdir()).write_bytes(b"x")
    with pytest.raises(CatalogError):
        Catalog.restore(tmp_path / "bak", tmp_path / "restored2")


def test_concurrent_writer_lock_does_not_corrupt(tmp_path):
    cat = Catalog(tmp_path)
    blocker = sqlite3.connect(cat.db_path)
    blocker.execute("BEGIN IMMEDIATE")
    cat2 = Catalog(tmp_path)
    cat2._conn = lambda: sqlite3.connect(cat.db_path, timeout=0.1)  # type: ignore[method-assign]
    with pytest.raises(sqlite3.OperationalError):
        pub(cat2)
    blocker.rollback(); blocker.close()
    assert pub(cat)["new_vintages"] == 2

"""Tests for the Nhóm-2 security/correctness hardening: PositionStore id uniqueness +
concurrency + atomic save + close-exactly-one, chat symbol validation (path traversal),
and the HTTP Basic auth gate."""
import base64
import threading
from urllib.parse import urlparse

import pytest


# ---- PositionStore -------------------------------------------------------------------
def _store(tmp_path):
    from stock_agent.features.position_manager import PositionStore
    return PositionStore(path=tmp_path / "pos.json")


def _add(store, sym="PNJ"):
    return store.add(symbol=sym, entry_date="2026-07-06", entry_price=50.0, stop=47.0,
                     target=55.0, max_hold_days=15, qty=100)


def test_ids_are_unique(tmp_path):
    s = _store(tmp_path)
    ids = {_add(s)["id"] for _ in range(50)}
    assert len(ids) == 50  # uuid-based -> never collide (old len-based id repeated)


def test_close_closes_exactly_one_even_with_dup_id(tmp_path):
    s = _store(tmp_path)
    # craft two OPEN rows sharing an id (as the old len-based scheme could produce)
    import json
    dup = [
        {"id": "X", "symbol": "AAA", "entry_date": "2026-07-06", "kind": "mr",
         "entry_price": 1, "stop": 0.9, "target": 1.1, "max_hold_days": 15, "qty": 100, "status": "OPEN"},
        {"id": "X", "symbol": "BBB", "entry_date": "2026-07-06", "kind": "mr",
         "entry_price": 1, "stop": 0.9, "target": 1.1, "max_hold_days": 15, "qty": 100, "status": "OPEN"},
    ]
    (tmp_path / "pos.json").write_text(json.dumps(dup), encoding="utf-8")
    assert s.close("X") is True
    open_after = s.list(status="OPEN")
    assert len(open_after) == 1  # only ONE closed, not both


def test_concurrent_adds_dont_lose_writes(tmp_path):
    s = _store(tmp_path)
    threads = [threading.Thread(target=_add, args=(s,)) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(s.list()) == 25  # lock serializes load->append->save; none dropped


def test_save_is_atomic_no_tmp_left(tmp_path):
    s = _store(tmp_path)
    _add(s)
    assert (tmp_path / "pos.json").exists()
    assert not (tmp_path / "pos.json.tmp").exists()  # temp renamed away


# ---- chat symbol validation (path traversal) -----------------------------------------
@pytest.mark.parametrize("bad", ["../../../etc/passwd", "../secret", "a/b", "TOOLONGSYM", "", "  "])
def test_get_symbol_context_rejects_bad_symbols(bad):
    from stock_agent.ai.chat import get_symbol_context
    out = get_symbol_context(bad)
    assert out == {"candidate": None, "recent_prices": []}


def test_get_symbol_context_accepts_valid_shape():
    from stock_agent.ai.chat import _SYMBOL_RE
    assert _SYMBOL_RE.match("FPT") and _SYMBOL_RE.match("VN30") and _SYMBOL_RE.match("A9")
    assert not _SYMBOL_RE.match("../X") and not _SYMBOL_RE.match("fpt")


# ---- HTTP Basic auth gate ------------------------------------------------------------
class _FakeHandler:
    def __init__(self, path, auth=None):
        self.path = path
        self.headers = {"Authorization": auth} if auth else {}
        self.status = None

    def send_response(self, s):
        self.status = s

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass


def _basic(user, pw):
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


def test_auth_open_when_no_password(monkeypatch):
    import stock_agent.app as app
    monkeypatch.setattr(app, "_AUTH_PASS", "")
    assert app._auth_ok(_FakeHandler("/api/mr/scan")) is True


def test_auth_health_exempt(monkeypatch):
    import stock_agent.app as app
    monkeypatch.setattr(app, "_AUTH_PASS", "secret")
    assert app._auth_ok(_FakeHandler("/api/health")) is True


def test_auth_rejects_missing_and_wrong(monkeypatch):
    import stock_agent.app as app
    monkeypatch.setattr(app, "_AUTH_PASS", "secret")
    monkeypatch.setattr(app, "_AUTH_USER", "admin")
    h_missing = _FakeHandler("/")
    assert app._auth_ok(h_missing) is False and h_missing.status == 401
    assert app._auth_ok(_FakeHandler("/", _basic("admin", "wrong"))) is False
    assert app._auth_ok(_FakeHandler("/", _basic("wrong", "secret"))) is False


def test_auth_accepts_correct(monkeypatch):
    import stock_agent.app as app
    monkeypatch.setattr(app, "_AUTH_PASS", "secret")
    monkeypatch.setattr(app, "_AUTH_USER", "admin")
    assert app._auth_ok(_FakeHandler("/", _basic("admin", "secret"))) is True

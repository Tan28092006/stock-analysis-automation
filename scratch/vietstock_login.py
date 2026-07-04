"""Vietstock login + immediate deep-history verification.

Opens a VISIBLE browser window on the Vietstock foreign-trade page. Log in there
(Google/Facebook/email). The script detects the logged-in session, saves it to
data/raw/foreign/vietstock_session.json, then verifies IN THE SAME SESSION whether
old dates (2022/2020) are actually served (checks TradingDate INSIDE rows — the
anonymous tier silently falls back to the latest day).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from playwright.sync_api import sync_playwright

SESSION_PATH = Path(r"D:\Chungkhoan\data\raw\foreign\vietstock_session.json")
PAGE_URL = "https://finance.vietstock.vn/ket-qua-giao-dich?tab=gd-khop-lenh-nn"
EP_NN = "https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNPaging"
EP_TD = "https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhPaging"
LOGIN_WAIT_S = 420  # 7 minutes to log in


def ms2date(s):
    m = re.search(r"(-?\d+)", s or "")
    if not m:
        return None
    ms = int(m.group(1))
    if ms < 0:
        return None
    return (datetime(1970, 1, 1) + timedelta(milliseconds=ms)).date()


def in_page_post(page, ep, token, **kw):
    form = {"page": "1", "pageSize": "50", "catID": "1",
            "__RequestVerificationToken": token}
    form.update({k: str(v) for k, v in kw.items()})
    js = """async ([ep, form]) => {
      const body = new URLSearchParams(form).toString();
      const r = await fetch(ep, {method:'POST', body,
        headers: {'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                  'X-Requested-With':'XMLHttpRequest'},
        credentials:'include'});
      return [r.status, (await r.text()).slice(0, 500000)];
    }"""
    status, text = page.evaluate(js, [ep, form])
    try:
        j = json.loads(text)
    except Exception:
        return status, None, []
    hdr = j[0][0] if j and j[0] else {}
    rows = j[1] if len(j) > 1 else []
    return status, hdr, rows


AUTH_HINTS = ("aspxauth", "vts", "usr", "user", "token", "login", "auth", "member", "acc")


def is_logged_in(page, baseline: set) -> bool:
    try:
        cookies = page.context.cookies()
        for c in cookies:
            name = c["name"].lower()
            val = c.get("value") or ""
            if name in baseline:
                continue
            if len(val) > 24 and any(h in name for h in AUTH_HINTS):
                print(f"  [detect] new auth cookie: {c['name']} (len {len(val)})", flush=True)
                return True
        # DOM signals: logout link, or an account/username element appears
        dom = page.evaluate("""() => {
          const t = (document.body && document.body.innerText) || '';
          if (/Đăng xuất|Thoát|Tài khoản của tôi|My Account/i.test(t)) return 'logout-text';
          if (document.querySelector('.user-name, .username, .member-name, #hypLoginName, .vst-user, a[href*="thoat"], a[href*="logout"]')) return 'user-el';
          return '';
        }""")
        if dom:
            print(f"  [detect] DOM signal: {dom}", flush=True)
            return True
        return False
    except Exception:
        return False


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False)
        ctx = b.new_context(viewport={"width": 1400, "height": 900}, locale="vi-VN")
        page = ctx.new_page()
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        # open the login modal for convenience
        try:
            btn = page.query_selector("a.btnlogin-link")
            if btn:
                btn.click()
        except Exception:
            pass

        baseline = {c["name"].lower() for c in ctx.cookies()}
        print("=" * 64)
        print(">>> CUA SO TRINH DUYET DA MO — HAY DANG NHAP VIETSTOCK <<<")
        print(">>> (Google / Facebook / email deu duoc) <<<")
        print(f">>> Cho toi da {LOGIN_WAIT_S // 60} phut...")
        print("=" * 64, flush=True)

        t0 = time.time()
        logged = False
        last_dump = 0.0
        while time.time() - t0 < LOGIN_WAIT_S:
            if is_logged_in(page, baseline):
                logged = True
                break
            # every ~30s, save a session snapshot + dump cookie names as a fallback,
            # so even if auto-detect misses, we still capture the logged-in state.
            if time.time() - last_dump > 30:
                last_dump = time.time()
                try:
                    ctx.storage_state(path=str(SESSION_PATH))
                    names = sorted({c["name"] for c in ctx.cookies()} - baseline)
                    if names:
                        print(f"  [snapshot] saved; new cookies so far: {names}", flush=True)
                except Exception:
                    pass
            time.sleep(3)
        if not logged:
            print("TIMEOUT: chua phat hien dang nhap. Chay lai script khi san sang.")
            b.close()
            return

        print("\nDang nhap OK — luu session...", flush=True)
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        ctx.storage_state(path=str(SESSION_PATH))
        print(f"Session -> {SESSION_PATH}")

        # reload the data page to get a fresh token under the logged-in session
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        token = page.evaluate("""() => {
          const el = document.querySelector('input[name=__RequestVerificationToken]');
          return el ? el.value : null;
        }""")
        print("token:", bool(token))

        print("\n--- VERIFY: lich su sau co mo khoa khong? (soi ngay TRONG du lieu) ---")
        verdict_ok = 0
        for d in ["2022-06-10", "2020-03-24"]:
            status, hdr, rows = in_page_post(page, EP_NN, token, date=d)
            served = ms2date((hdr or {}).get("TrDate"))
            row_dates = sorted({str(ms2date(r.get("TradingDate"))) for r in rows[:10]})
            ok = bool(rows) and any(str(d)[:4] in rd for rd in row_dates)
            verdict_ok += int(ok)
            print(f"NN  date={d}: status={status} rows={len(rows)} served={served} "
                  f"row_dates={row_dates} -> {'UNLOCKED' if ok else 'fallback/blocked'}")
        # page-2 + tu doanh
        status, hdr, rows = in_page_post(page, EP_NN, token, date="2022-06-10", page=2)
        syms = [r.get("StockCode") for r in rows[:3]]
        print(f"NN  page=2: rows={len(rows)} first={syms}")
        status, hdr, rows = in_page_post(page, EP_TD, token, date="2022-06-10")
        row_dates = sorted({str(ms2date(r.get("TradingDate"))) for r in rows[:5]})
        print(f"TD  date=2022-06-10: rows={len(rows)} row_dates={row_dates}")

        print("\n==> KET LUAN:", "LOGIN MO KHOA LICH SU SAU — co the backfill!" if verdict_ok
              else "van bi chan — tier free khong du, can goi tra phi.")
        b.close()


if __name__ == "__main__":
    main()

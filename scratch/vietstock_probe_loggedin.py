"""Reuse the saved Vietstock session (headless) and probe what login actually unlocks.
Decides whether deep daily backfill (2020-2025) is possible on the free account."""
import io, sys, re, json
from datetime import datetime, timedelta
sys.path.insert(0, r"D:\Chungkhoan")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from playwright.sync_api import sync_playwright

SESSION = r"D:\Chungkhoan\data\raw\foreign\vietstock_session.json"
PAGE_URL = "https://finance.vietstock.vn/ket-qua-giao-dich?tab=gd-khop-lenh-nn"
EP_NN = "https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNPaging"
EP_CHART = "https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock"

def ms2date(s):
    m = re.search(r"(-?\d+)", s or "")
    if not m: return None
    ms = int(m.group(1))
    return (datetime(1970,1,1)+timedelta(milliseconds=ms)).date() if ms >= 0 else None

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(storage_state=SESSION, locale="vi-VN")
    page = ctx.new_page()
    page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3500)
    logged = page.evaluate("""() => /Đăng xuất|Tài khoản của tôi/i.test(document.body.innerText) ||
        !!document.querySelector('.user-name,.username,#hypLoginName')""")
    token = page.evaluate("() => { const e=document.querySelector('input[name=__RequestVerificationToken]'); return e?e.value:null; }")
    print(f"session reused | logged_in_dom={logged} | token={bool(token)}\n")

    def post(ep, **kw):
        form = {"__RequestVerificationToken": token}
        form.update({k: str(v) for k, v in kw.items()})
        js = """async ([ep, form]) => {
          const r = await fetch(ep, {method:'POST', body:new URLSearchParams(form).toString(),
            headers:{'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8','X-Requested-With':'XMLHttpRequest'},
            credentials:'include'});
          return [r.status, (await r.text()).slice(0,600000)]; }"""
        st, txt = page.evaluate(js, [ep, form])
        try: return st, json.loads(txt)
        except Exception: return st, None

    print("--- 1) paging: does date work now, and page-2? ---")
    for d, pg in [("2026-07-02",1),("2026-07-02",2),("2026-07-02",3),("2022-06-10",1),("2020-03-24",1)]:
        st, j = post(EP_NN, page=pg, pageSize=50, catID=1, date=d)
        rows = j[1] if j and len(j)>1 else []
        served = ms2date((j[0][0] if j and j[0] else {}).get("TrDate"))
        first = [r.get("StockCode") for r in rows[:2]]
        print(f"  date={d} page={pg}: rows={len(rows)} served={served} first={first}")

    print("\n--- 2) per-symbol chart depth (logged-in) type=1..3 ---")
    for t in ["1","2","3"]:
        st, j = post(EP_CHART, stockCode="SSI", type=t, isRealTime="false")
        series = j[1] if j and len(j)>1 else []
        if series:
            d0,d1 = ms2date(series[0].get("TradingDate")), ms2date(series[-1].get("TradingDate"))
            print(f"  type={t}: rows={len(series)} range={d0} -> {d1}")
        else:
            print(f"  type={t}: rows=0")

    print("\n--- 3) chart with explicit date-range params (guess names) ---")
    for extra in [{"fromDate":"2022-01-01","toDate":"2022-12-31"},
                  {"from":"2022-01-01","to":"2022-12-31"},
                  {"type":"1","fromDate":"2022-01-01","toDate":"2022-12-31"}]:
        st, j = post(EP_CHART, stockCode="SSI", isRealTime="false", **extra)
        series = j[1] if j and len(j)>1 else []
        d0 = ms2date(series[0].get("TradingDate")) if series else None
        d1 = ms2date(series[-1].get("TradingDate")) if series else None
        print(f"  {list(extra)}: rows={len(series)} range={d0}->{d1}")
    b.close()

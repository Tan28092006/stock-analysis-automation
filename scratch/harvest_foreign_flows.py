"""Harvest daily foreign (NDTNN) + proprietary (tu doanh) flows from Vietstock.

Free anonymous tier serves ~9 months back (cutoff ~2025-10). For each calendar date in
range, POST the paging endpoints discovered via Playwright network sniffing:
  /data/KQGDGiaoDichNDTNNPaging   (khoi ngoai: buy/sell matched + put-through, room, own%)
  /data/KQGDGiaoDichTuDoanhPaging (tu doanh:  buy/sell matched + put-through + totals)

Notes
-----
- The server keys responses by its own TrDate (requesting date d returns the trading day
  BEFORE d due to timezone) -> we key rows by the served TradingDate and dedupe, so the
  calendar sweep is self-correcting.
- pageSize is server-capped at 50 -> page until a short page arrives.
- Output: data/raw/foreign/ndtnn.jsonl and tudoanh.jsonl (one row per symbol-day),
  resume-safe (already-served dates are skipped on rerun).
"""
from __future__ import annotations

import io, sys, re, json, time, random
import urllib.request, urllib.parse, http.cookiejar
from datetime import date, datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\Chungkhoan\data\raw\foreign")
OUT.mkdir(parents=True, exist_ok=True)
START = date(2025, 10, 1)          # free-tier cutoff (probed)
END = date(2026, 7, 2)
PAGE_SIZE = 50
THROTTLE = (0.6, 1.0)              # polite random sleep between requests

ENDPOINTS = {
    "ndtnn": "https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNPaging",
    "tudoanh": "https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhPaging",
}
REFERER = "https://finance.vietstock.vn/ket-qua-giao-dich?tab=gd-khop-lenh-nn"


class Session:
    def __init__(self):
        self.renew()

    def renew(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
        html = self.op.open(REFERER, timeout=30).read().decode("utf-8", "replace")
        self.token = re.search(r"name=__RequestVerificationToken[^>]*value=([^ >]+)", html).group(1)

    def post(self, ep: str, params: dict):
        d = {"__RequestVerificationToken": self.token}
        d.update(params)
        req = urllib.request.Request(
            ep, data=urllib.parse.urlencode(d).encode(),
            headers={"X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Referer": REFERER, "User-Agent": "Mozilla/5.0"})
        raw = self.op.open(req, timeout=30).read().decode("utf-8-sig", "replace")
        return json.loads(raw)


def ms2date(s: str):
    m = re.search(r"(-?\d+)", s or "")
    if not m:
        return None
    return (datetime(1970, 1, 1) + timedelta(milliseconds=int(m.group(1)))).date()


def served_dates(path: Path) -> set:
    got = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    got.add(json.loads(line)["date"])
                except Exception:
                    pass
    return got


def harvest(kind: str, ses: Session) -> None:
    ep = ENDPOINTS[kind]
    out_path = OUT / f"{kind}.jsonl"
    done = served_dates(out_path)
    print(f"[{kind}] resume: {len(done)} dates already harvested")
    d = END
    consecutive_errors = 0
    while d >= START:
        req_d = d + timedelta(days=1)      # server serves the trading day BEFORE `date`
        if d.weekday() >= 5:               # skip weekends quickly
            d -= timedelta(days=1)
            continue
        if str(d) in done:
            d -= timedelta(days=1)
            continue
        try:
            rows_all, page = [], 1
            trdate = None
            while True:
                j = ses.post(ep, {"page": str(page), "pageSize": str(PAGE_SIZE),
                                  "catID": "1", "date": str(req_d)})
                hdr = j[0][0] if j and j[0] else {}
                trdate = ms2date(hdr.get("TrDate")) or trdate
                rows = j[1] if len(j) > 1 else []
                rows_all += rows
                if len(rows) < PAGE_SIZE:
                    break
                page += 1
                time.sleep(random.uniform(*THROTTLE))
            # fallback guard: server returns latest day for dates beyond the free tier
            if trdate is None or abs((trdate - d).days) > 5:
                print(f"[{kind}] {d} -> served {trdate} (out of free range?) stop-or-skip")
                if trdate and (trdate - d).days > 30:
                    print(f"[{kind}] free-tier cutoff reached at {d}; stopping")
                    break
                d -= timedelta(days=1)
                continue
            if str(trdate) in done:
                d -= timedelta(days=1)
                continue
            with out_path.open("a", encoding="utf-8") as f:
                for r in rows_all:
                    rec = {"date": str(trdate), "kind": kind}
                    rec.update({k: r.get(k) for k in r if k not in ("FinanceURL", "StockNameEn")})
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            done.add(str(trdate))
            print(f"[{kind}] {trdate}: {len(rows_all)} rows ({page} pages)", flush=True)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"[{kind}] {d} ERR {repr(e)[:120]} (#{consecutive_errors})", flush=True)
            if consecutive_errors >= 3:
                try:
                    ses.renew()
                    print(f"[{kind}] session renewed")
                    consecutive_errors = 0
                except Exception as e2:
                    print(f"[{kind}] renew failed {repr(e2)[:100]}; sleeping 60s")
                    time.sleep(60)
            time.sleep(5)
            continue   # retry same date once after error
        d -= timedelta(days=1)
        time.sleep(random.uniform(*THROTTLE))


def main():
    t0 = time.time()
    ses = Session()
    print("session ok, token acquired")
    for kind in ("ndtnn", "tudoanh"):
        harvest(kind, ses)
    print(f"DONE in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()

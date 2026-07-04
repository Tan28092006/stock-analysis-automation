"""Harvest last ~20 daily sessions of foreign (NDTNN) + proprietary (tu doanh) flows
for every VN100 symbol via Vietstock per-symbol chart endpoints (anonymous tier).

Anonymous limits (probed exhaustively):
  - market-wide paging: 50 alphabetical rows/day only -> useless for VN100
  - per-symbol chart:   last ~20 daily sessions        -> THIS is what we harvest
Run daily/weekly and the JSONL accumulates a growing panel (dedup on date+symbol).
"""
from __future__ import annotations

import io, sys, re, json, time, random
import urllib.request, urllib.parse, http.cookiejar
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\Chungkhoan\data\raw\foreign")
OUT.mkdir(parents=True, exist_ok=True)

VN100 = ["ACB","ANV","BAF","BCM","BID","BMP","BSI","BSR","BVH","BWE","CII","CMG",
         "CTD","CTG","CTR","CTS","DBC","DCM","DGW","DIG","DPM","DSE","DXG","DXS",
         "EIB","EVF","FPT","FRT","FTS","GAS","GEE","GEX","GMD","GVR","HAG","HCM",
         "HDB","HDC","HDG","HHV","HPG","HSG","HT1","IMP","KBC","KDC","KDH","KOS",
         "LPB","MBB","MSB","MSN","MWG","NAB","NKG","NLG","NT2","NVL","OCB","PAN",
         "PC1","PDR","PHR","PLX","PNJ","POW","PVD","PVT","REE","SAB","SBT","SCS",
         "SHB","SIP","SJS","SSB","SSI","STB","SZC","TCB","TCH","TPB","VCB","VCG",
         "VCI","VGC","VHC","VHM","VIB","VIC","VIX","VJC","VND","VNM","VPB","VPI",
         "VPL","VRE","VSC","VTP"]

ENDPOINTS = {
    "ndtnn_chart": ("https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock",
                    {"type": "1", "isRealTime": "false"}),
    "tudoanh_chart": ("https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhChartByStock",
                      {"type": "1"}),
}
REFERER = "https://finance.vietstock.vn/MBB/thong-ke-giao-dich.htm"


def ms2date(s):
    m = re.search(r"(-?\d+)", s or "")
    if not m:
        return None
    ms = int(m.group(1))
    if ms < 0:
        return None
    return (datetime(1970, 1, 1) + timedelta(milliseconds=ms)).date()


class Session:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
        html = self.op.open(REFERER, timeout=30).read().decode("utf-8", "replace")
        self.token = re.search(
            r"__CHART_AjaxAntiForgeryForm[^>]*>.*?name=__RequestVerificationToken[^>]*value=([^ >]+)",
            html, re.S).group(1)

    def post(self, ep, params):
        d = {"__RequestVerificationToken": self.token}
        d.update(params)
        req = urllib.request.Request(ep, data=urllib.parse.urlencode(d).encode(),
            headers={"X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Referer": REFERER, "User-Agent": "Mozilla/5.0"})
        return json.loads(self.op.open(req, timeout=30).read().decode("utf-8-sig", "replace"))


def existing_keys(path):
    keys = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["date"], r["symbol"]))
                except Exception:
                    pass
    return keys


def main():
    t0 = time.time()
    ses = Session()
    print("session ok")
    for kind, (ep, base) in ENDPOINTS.items():
        out_path = OUT / f"{kind}.jsonl"
        seen = existing_keys(out_path)
        added = 0
        for i, sym in enumerate(VN100):
            for attempt in range(3):
                try:
                    j = ses.post(ep, {**base, "stockCode": sym})
                    series = j[1] if isinstance(j, list) and len(j) > 1 else []
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in series:
                            d = ms2date(r.get("TradingDate"))
                            if d is None or (str(d), sym) in seen:
                                continue
                            rec = {"date": str(d), "symbol": sym,
                                   "buy_vol": r.get("BuyVol"), "buy_val": r.get("BuyVal"),
                                   "sell_vol": r.get("SellVol"), "sell_val": r.get("SellVal")}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            seen.add((str(d), sym))
                            added += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[{kind}] {sym} FAIL {repr(e)[:90]}", flush=True)
                    else:
                        time.sleep(4)
                        try:
                            ses = Session()
                        except Exception:
                            pass
            if (i + 1) % 20 == 0:
                print(f"[{kind}] {i+1}/{len(VN100)} symbols, +{added} rows", flush=True)
            time.sleep(random.uniform(0.4, 0.8))
        print(f"[{kind}] DONE +{added} rows -> {out_path}", flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

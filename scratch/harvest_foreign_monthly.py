"""Harvest MONTHLY (type=2, ~12mo) + QUARTERLY (type=3, ~12q) foreign + prop flows per
VN100 symbol from Vietstock (anonymous — these aggregates are not date-gated).

These feed the slow REGIME signal (foreign accumulation/outflow trend over months),
not the T+2 entry gate. Sum across the universe -> market-wide foreign flow by month.
Output: data/raw/foreign/{ndtnn,tudoanh}_{monthly,quarterly}.jsonl
"""
from __future__ import annotations
import io, sys, re, json, time, random
import urllib.request, urllib.parse, http.cookiejar
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(r"D:\Chungkhoan\data\raw\foreign"); OUT.mkdir(parents=True, exist_ok=True)
REFERER = "https://finance.vietstock.vn/MBB/thong-ke-giao-dich.htm"
VN100 = [p.stem for p in Path(r"D:\Chungkhoan\data\raw\prices_hist").glob("*.csv") if p.stem != "VNINDEX"]
ENDPOINTS = {
    "ndtnn": "https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock",
    "tudoanh": "https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhChartByStock",
}
PERIODS = {"monthly": "2", "quarterly": "3"}


class Ses:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
        html = self.op.open(REFERER, timeout=30).read().decode("utf-8", "replace")
        self.token = re.search(r"__CHART_AjaxAntiForgeryForm[^>]*>.*?name=__RequestVerificationToken[^>]*value=([^ >]+)", html, re.S).group(1)

    def post(self, ep, params):
        d = {"__RequestVerificationToken": self.token}; d.update(params)
        req = urllib.request.Request(ep, data=urllib.parse.urlencode(d).encode(),
            headers={"X-Requested-With": "XMLHttpRequest", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Referer": REFERER, "User-Agent": "Mozilla/5.0"})
        return json.loads(self.op.open(req, timeout=30).read().decode("utf-8-sig", "replace"))


def main():
    ses = Ses(); print("session ok")
    for kind, ep in ENDPOINTS.items():
        for pname, ptype in PERIODS.items():
            out = OUT / f"{kind}_{pname}.jsonl"
            seen = set()
            if out.exists():
                for line in out.open("r", encoding="utf-8"):
                    try:
                        r = json.loads(line); seen.add((r["period"], r["symbol"]))
                    except Exception: pass
            added = 0
            for sym in VN100:
                for attempt in range(3):
                    try:
                        j = ses.post(ep, {"stockCode": sym, "type": ptype, "isRealTime": "false"})
                        series = j[1] if isinstance(j, list) and len(j) > 1 else []
                        with out.open("a", encoding="utf-8") as f:
                            for r in series:
                                period = r.get("TradingMonthYear") or r.get("Quarter")
                                if not period or (period, sym) in seen:
                                    continue
                                f.write(json.dumps({"period": period, "symbol": sym, "kind": kind, "freq": pname,
                                    "buy_vol": r.get("BuyVol"), "buy_val": r.get("BuyVal"),
                                    "sell_vol": r.get("SellVol"), "sell_val": r.get("SellVal")}, ensure_ascii=False) + "\n")
                                seen.add((period, sym)); added += 1
                        break
                    except Exception:
                        time.sleep(3)
                        try: ses = Ses()
                        except Exception: pass
                time.sleep(random.uniform(0.3, 0.6))
            print(f"[{kind}_{pname}] +{added} rows -> {out}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

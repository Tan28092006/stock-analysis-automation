import json
from pathlib import Path

p = Path("d:/Chungkhoan/data/processed/latest_scan.json")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in data.get("candidates", []):
        if c.get("symbol") == "ACB":
            print("ACB Candidate Details:")
            print("Decision:", c.get("decision"))
            print("Latest Close:", c.get("latest_close"))
            print("Latest Date:", c.get("latest_date"))
            print("Data Quality Status:", c.get("data_quality", {}).get("status"))
            print("Data Quality Cross Check:", c.get("data_quality", {}).get("cross_check_status"))
            print("Provider Audits:")
            for pa in c.get("data_quality", {}).get("provider_audits", []):
                print(f"  Provider: {pa.get('provider')}, Status: {pa.get('status')}, Rows: {pa.get('rows')}, Latest Date: {pa.get('latest_date')}, Latest Close: {pa.get('latest_close')}, Error: {pa.get('error')}")
            break
else:
    print("Not found")

import json
from pathlib import Path

p = Path("d:/Chungkhoan/data/processed/latest_scan.json")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Scan ID:", data.get("scan_id"))
    print("Candidates:", len(data.get("candidates", [])))
    
    # Check data quality for the first few candidates
    for idx, c in enumerate(data.get("candidates", [])[:5]):
        dq = c.get("data_quality", {})
        print(f"{idx+1}. {c.get('symbol')} - Decision: {c.get('decision')} - Status: {dq.get('status')} - Cross Check: {dq.get('cross_check_status')} - Warnings: {dq.get('warnings')}")
else:
    print("Not found")

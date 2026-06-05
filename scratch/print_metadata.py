import json
from pathlib import Path

p = Path("d:/Chungkhoan/data/processed/latest_scan.json")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Scan ID:", data.get("scan_id"))
    print("Created At:", data.get("created_at"))
    print("Candidates in list:", len(data.get("candidates", [])))
    print("Rejected count:", data.get("rejected_count"))
else:
    print("Not found")

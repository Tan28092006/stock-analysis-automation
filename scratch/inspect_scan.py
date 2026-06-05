import json
from pathlib import Path

latest_scan_path = Path("d:/Chungkhoan/data/processed/latest_scan.json")
if latest_scan_path.exists():
    with open(latest_scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Total symbols scanned:", data.get("symbols_scanned"))
    print("Rejected count in summary:", data.get("rejected_count"))
    candidates = data.get("candidates", [])
    print("Candidates in list:", len(candidates))
    decisions = {}
    for c in candidates:
        dec = c.get("decision")
        decisions[dec] = decisions.get(dec, 0) + 1
    print("Decisions count in candidate list:", decisions)
    print("Candidates symbols:", [c.get("symbol") for c in candidates])
else:
    print("File not found")

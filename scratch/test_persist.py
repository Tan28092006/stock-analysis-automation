import sys
sys.path.append(r'd:\Chungkhoan')
from stock_agent.agents.orchestrator import run_scan
import json
from pathlib import Path

print("Running direct scan with persist=True...")
res = run_scan(demo=False, persist=True)
print("Symbols scanned:", res.symbols_scanned)
print("Rejected count in result:", res.rejected_count)
print("Result candidates length:", len(res.candidates))

latest_scan_path = Path("d:/Chungkhoan/data/processed/latest_scan.json")
with open(latest_scan_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Saved file candidates length:", len(data.get("candidates", [])))
print("Saved file decisions:", [c.get("decision") for c in data.get("candidates", [])])

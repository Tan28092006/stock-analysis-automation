import sys
sys.path.append(r'd:\Chungkhoan')
from stock_agent.agents.orchestrator import run_scan
from collections import Counter

print("Running direct scan...")
res = run_scan(demo=False, persist=False)
print("Symbols scanned:", res.symbols_scanned)
print("Rejected count:", res.rejected_count)
print("Candidates list length:", len(res.candidates))
decisions = Counter(c.decision for c in res.candidates)
print("Decisions in candidates:", decisions)
print("All candidate symbols:", [c.symbol for c in res.candidates])

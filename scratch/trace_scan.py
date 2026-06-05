import sys
sys.path.append(r'd:\Chungkhoan')
from stock_agent.agents.orchestrator import run_scan
import logging

logging.basicConfig(level=logging.INFO)
print("Starting loop logging test...")
res = run_scan(demo=False, persist=False)
print("Symbols scanned:", res.symbols_scanned)
print("Candidates length:", len(res.candidates))
for idx, c in enumerate(res.candidates):
    print(f"{idx+1}. Symbol: {c.symbol}, Decision: {c.decision}")

import sys
sys.path.append(r'd:\Chungkhoan')

# Mock yfinance and vnstock imports or behavior to simulate failures
import stock_agent.data.providers as providers

# Backup original methods
orig_yf_history = providers.YahooProvider.history
orig_vn_history = providers.VnStockProvider.history

# Define mock methods that fail
def mock_yf_history(self, symbol, start, end):
    from stock_agent.schemas import ProviderAudit
    from stock_agent.data.validation import ProviderFrame
    return ProviderFrame(
        self.name,
        None,
        ProviderAudit(self.name, "error", error="unable to open database file", source_type=self.source_type)
    )

def mock_vn_history(self, symbol, start, end):
    from stock_agent.schemas import ProviderAudit
    from stock_agent.data.validation import ProviderFrame
    return ProviderFrame(
        self.name,
        None,
        ProviderAudit(self.name, "missing", error="vnstock is not installed or has an unsupported API", source_type=self.source_type)
    )

providers.YahooProvider.history = mock_yf_history
providers.VnStockProvider.history = mock_vn_history

from stock_agent.agents.orchestrator import run_scan
from collections import Counter

print("Running scan with simulated failures...")
res = run_scan(demo=False, persist=False)
print("Symbols scanned:", res.symbols_scanned)
print("Rejected count:", res.rejected_count)
print("Candidates list length:", len(res.candidates))
decisions = Counter(c.decision for c in res.candidates)
print("Decisions in candidates:", decisions)
print("Candidates list:")
for idx, c in enumerate(res.candidates):
    print(f"{idx+1}. {c.symbol}: {c.decision}")

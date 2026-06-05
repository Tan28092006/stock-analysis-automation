import sys
sys.path.append(r'd:\Chungkhoan')
from stock_agent.agents.orchestrator import run_scan
import inspect

print("run_scan path:", inspect.getfile(run_scan))
# Print the last few lines of run_scan function
source = inspect.getsource(run_scan)
print("Source length:", len(source))
# Check if "candidates" is filtered or if only WATCH/BUY_SETUP are appended
# Look at the end of the source
print("End of run_scan:")
print("\n".join(source.splitlines()[-30:]))

import urllib.request
import json

url = "http://127.0.0.1:8000/api/scan/trigger"
data = json.dumps({"demo": False}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Triggering full scan...")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode())
except Exception as e:
    print(f"Error: {e}")
    raise

print(f"\nScan ID: {result['scan_id']}")
print(f"Mode: {result['mode']}")
print(f"Symbols scanned: {result['symbols_scanned']}")
print(f"Total candidates: {len(result['candidates'])}")

# Summarise cross-check statuses
status_counts = {}
source_counts = {}
for c in result["candidates"]:
    dq = c.get("data_quality", {})
    cs = dq.get("cross_check_status", "unknown")
    status_counts[cs] = status_counts.get(cs, 0) + 1
    ps = dq.get("primary_provider", "unknown")
    source_counts[ps] = source_counts.get(ps, 0) + 1

print(f"\n--- Cross-Check Status Distribution ---")
for k, v in sorted(status_counts.items()):
    print(f"  {k}: {v}")

print(f"\n--- Primary Provider Distribution ---")
for k, v in sorted(source_counts.items()):
    print(f"  {k}: {v}")

# Show each candidate briefly
print(f"\n--- All Candidates ---")
print(f"{'Symbol':<8} {'Decision':<12} {'Score':>6} {'Price':>12} {'Cross-Check':<20} {'Provider':<15} {'Latest Date':<12}")
print("-" * 90)
for c in result["candidates"]:
    dq = c.get("data_quality", {})
    print(f"{c['symbol']:<8} {c['decision']:<12} {c['score']:>6.0f} {c['latest_close']:>12,.0f} {dq.get('cross_check_status','?'):<20} {dq.get('primary_provider','?'):<15} {c.get('latest_date','?'):<12}")

if result.get("warnings"):
    print(f"\n--- Scan Warnings ---")
    for w in result["warnings"]:
        print(f"  ⚠ {w}")

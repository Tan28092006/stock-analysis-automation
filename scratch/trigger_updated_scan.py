import urllib.request
import json

url = "http://127.0.0.1:8000/api/scan/trigger"
data = json.dumps({"demo": False}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Triggering web scan on updated server...")
try:
    with urllib.request.urlopen(req, timeout=300) as res:
        response_data = json.loads(res.read().decode("utf-8"))
    
    print("Symbols scanned:", response_data.get("symbols_scanned"))
    print("Rejected count:", response_data.get("rejected_count"))
    candidates = response_data.get("candidates", [])
    print("Candidates in response:", len(candidates))
    decisions = {}
    for c in candidates:
        dec = c.get("decision")
        decisions[dec] = decisions.get(dec, 0) + 1
    print("Decisions:", decisions)
    
    # Show first 5 candidates with date and cross-check details
    print("\nTop candidates details:")
    for c in candidates[:8]:
        dq = c.get("data_quality", {})
        providers = [f"{pa.get('provider')}({pa.get('status')}, {pa.get('latest_date')})" 
                     for pa in dq.get("provider_audits", [])]
        print(f"  {c.get('symbol'):5s} | {c.get('decision'):10s} | close={c.get('latest_close'):>10} | date={c.get('latest_date')} | xcheck={dq.get('cross_check_status'):15s} | providers: {', '.join(providers)}")
except Exception as e:
    print("Error:", e)

import urllib.request
import json

url = "http://127.0.0.1:8000/api/scan/trigger"
data = json.dumps({"demo": False}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Triggering web scan...")
try:
    with urllib.request.urlopen(req) as res:
        response_data = json.loads(res.read().decode("utf-8"))
    
    print("Web scan complete!")
    print("Symbols scanned:", response_data.get("symbols_scanned"))
    print("Rejected count:", response_data.get("rejected_count"))
    candidates = response_data.get("candidates", [])
    print("Candidates in response:", len(candidates))
    decisions = {}
    for c in candidates:
        dec = c.get("decision")
        decisions[dec] = decisions.get(dec, 0) + 1
    print("Decisions in candidates:", decisions)
except Exception as e:
    print("Error:", e)

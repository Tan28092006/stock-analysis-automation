# Output contract — `recommendations.json`

A JSON array: at most 3 new ideas (`BUY`) plus one entry per held position (`HOLD`/`SELL`).
`AVOID` rows are allowed for tickers you explicitly reject. `freeze` validates these fields and
refuses the file after 09:00 VN of the target session.

```json
{"ticker": "HPG", "action": "BUY|HOLD|SELL|AVOID",
 "setup": ["S1"], "gate": "PASS|WARN", "gate_failed": [],
 "entry": {"type": "limit|ato", "limit_price": 27400, "valid_session": "2026-10-07"},
 "stop": 26578, "stop_basis": "user_3pct|validated_8pct",
 "target": 28800, "max_hold_sessions": 8, "earliest_sell": "2026-10-09 13:00",
 "size": {"qty": 1800, "risk_vnd": 2268000, "risk_pct_nav": 1.0},
 "p_win_stated": 0.55, "sentiment": {"score": 1, "confidence": "low|med|high", "citations": ["S-01"]},
 "invalidation": "VNINDEX close < EMA50 or close < stop",
 "data_as_of": "2026-10-06", "snapshot_sha256": "<manifest sha256 from workspace/manifest.json>"}
```

Required for every row: `ticker`, `action`, `entry`, `stop`, `target`, `max_hold_sessions`,
`data_as_of`, `p_win_stated`, `stop_basis`, `earliest_sell`, `invalidation`, `sentiment`.

Rules checked by `freeze` and by the verifiers:

- `data_as_of` equals the state session; `snapshot_sha256` equals the state's manifest sha.
- BUY: `gate` is never BLOCK; `stop < entry price < target`; limit entries carry `limit_price`.
- `p_win_stated` in [0, 1].
- `earliest_sell` is the afternoon of target session + 2 trading sessions (copy it from the ticker JSON).
- Every `sentiment.citations` id exists in `sentiment.md` with an opened URL; unopened = UNVERIFIED, weight 0.
- Values that cannot be traced to the workspace or a cited source stay empty; never fill gaps by guess.

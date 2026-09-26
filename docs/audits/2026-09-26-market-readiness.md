# Market readiness — 26/09/2026

## Verdict

**Working: prospective EOD paper observations from real public market data. Not approved: real-money execution or claims that the legacy model/history are leakage-free.**

The first source-verified paper record for the completed session **25/09/2026** was actually generated on **26/09/2026 at 09:30:55 Vietnam time**, before the next session on 28/09. It is not represented as a prediction issued on Friday. No broker order, account change, active model replacement, raw-history repair or legacy-ledger rewrite occurred.

## Actual run

- Snapshot: `data/paper/snapshots/20260926T022757171239Z/manifest.json`, VCI, 31/31 symbols, zero source failures. Fetch completed at 02:30:52 UTC. Raw responses, canonical CSVs, source/request timestamps and SHA-256 hashes are retained locally.
- Session readiness: all 30 configured VN30 constituents plus VNINDEX, completed daily bars only, coherent dates, valid OHLCV, positive latest volume, no unexplained internal session gaps.
- Scan: 30 stocks; MR rule BUY = 0, WATCH = 1. No requirement to invent a trade. Momentum = 10 informational picks. TCX is explicitly excluded from momentum because its history is shorter than the 254-bar minimum, not silently omitted from MR.
- Immutable record: `data/paper/runs/2026-09-25/paper.json`.
- First-record format note: its embedded `status` retained the pre-commit `preview_only` scan label. The commit receipt in `data/paper/latest.json`, actual `recorded_at`, `record_identity` and verified scorer establish that it was recorded. Final code now persists `status=recorded`; the original evidence is intentionally not rewritten to cosmetically correct that old field.
- Initial outcome report: `data/paper/scores/latest.json`: 10 pending, 0 resolved, 0 errors. These are observations, not ten funded positions. No forward performance is available yet.
- Prior failures are preserved in `data/pipeline/eod_update.log`: missing calendar date/symbol transfer handling; transient VNINDEX timeout on 25/09. A failed refresh now returns failure rather than publishing success.

## Architecture of the operational path

`run_eod_update.bat` → new isolated snapshot → verify every source/CSV hash → calendar/universe/OHLCV/readiness gates → rule-only MR + momentum scan → immutable pre-entry paper record → separate delayed outcome report.

The existing Windows task `StockAgent_EOD_Update` was not reconfigured. At inspection its next run was **28/09/2026 17:05**, with previous result 2 (failure). Its BAT action now runs the path above and preserves Python exit codes. It no longer stages, commits, rebases or pushes files. The machine must be awake, network available, and its task account/interpreter usable; future runs are not guaranteed merely by this successful manual run.

Legacy dashboard caches, swing scan, daily-run training and positions are separate paths and were not migrated. The trusted output of this rollout is `data/paper/`, not the old dashboard or old aggregate win-rate panel.

## Data gap corrections and source evidence

No synthetic OHLC bars, forward-fills or guessed prices were inserted.

- **02/05/2025 is an exchange holiday**, omitted from the old calendar. [Government publication covering HOSE/HNX](https://xaydungchinhsach.chinhphu.vn/lich-nghi-giao-dich-chung-khoan-dip-le-30-4-va-1-5-119250425104410084.htm).
- **BSR 07–16/01/2025**: exchange transfer, last UPCoM trading day 06/01 and first HOSE day 17/01. [Issuer resolution](https://bsr.com.vn/c/document_library/get_file?download=true&groupId=37629&uuid=74c428f4-ecb2-60ce-b214-4d230721ddfc), [issuer listing announcement](https://bsr.com.vn/w/bsr-vung-buoc-chuyen-minh-nang-tam-vi-the).
- **MCH 18–24/12/2025**: UPCoM deregistration then HOSE listing. [HNX notice](https://hnx.vn/vi-vn/m-tin-tuc-hnx/Ngay%2018122025%20ngay%20huy%20DKGD%20co%20phieu%20cua%20CTCP%20Hang%20tieu%20dung%20Masan-589962-1.html), [issuer annual report, corporate history](https://masanconsumer.com/wp-content/uploads/2026/03/Preview260326_MCH_AR25_Final-Full-Book_VIE-Final.pdf).

Only those exact symbol/date intervals are exempted. Other gaps still block recording. The calendar is maintained through 2026 and requires updating before 2027. Provider-adjusted prices and current VN30 membership remain explicitly **not** historical point-in-time evidence.

## One frozen model experiment (not deployed)

Candidate: `data/models/candidates/20260926T022835-b9bcb9cc6e/win_prob_mr.pkl`; trained 26/09 from the independently stored 21/09 snapshot. The production artifact was not replaced. The loader rechecks its release, artifact and source hashes. It rejects interpreting this newly trained model as available for a 25/09 signal.

1,220 mature candidates; purged fit 521, calibration 170, test 376. Latest fit label 09/03, calibration starts 10/03; calibration label end 08/06, test starts 09/06. Threshold .65 and promotion gates were fixed before fitting; test results were not used to tune them.

| Metric | Candidate | Comparator/gate |
|---|---:|---:|
| Test AUC | 0.63015 | >= 0.52 |
| Test Brier (lower better) | 0.23994 | Train-prior baseline 0.24302 |
| Test calibration error | 0.09308 | <= 0.10 |
| 2026Q2 AUC / Brier | 0.48554 / 0.26375 | Baseline Brier 0.23015 |
| 2026Q3 AUC / Brier | 0.69516 / 0.23381 | Baseline Brier 0.24634 |

Aggregate gates return `paper_eligible=true`, `live_approved=false`. Q2 is worse than baseline and calibration varies by quarter, so **the automated paper runner deliberately stays rule-only**. Offline trade labels are simplified daily-bar simulations, not execution evidence. Do not interpret selected win rates or returns as a deployable trading edge. Do not repeatedly tune this test set; a prospective period is required for the next decision.

## Operator runbook

From the repo root, using the existing environment:

```powershell
# Scheduled action / a genuinely new session, no orders:
C:/Users/acer/anaconda3/python.exe -m stock_agent.pipeline.paper_runner --refresh --run

# Verify/preview an existing snapshot, no paper record:
C:/Users/acer/anaconda3/python.exe -m stock_agent.pipeline.paper_runner --manifest data/paper/snapshots/20260926T022757171239Z/manifest.json --check

# Re-score records only; retain the original scan and paper records:
C:/Users/acer/anaconda3/python.exe -m stock_agent.pipeline.paper_runner --manifest data/paper/snapshots/20260926T022757171239Z/manifest.json --score
```

Exit codes: 0 success; 2 invalid/stale/unverified source, scan/score conflict or runtime failure; 3 `--run` requested outside the ex-ante window (preview only). The window opens at 16:00 VN on the completed session and closes at 09:00 VN next trading session. No retroactive records after that next open. `VN30_PYTHON` overrides the launcher interpreter if needed.

One record per session. Identical inputs/decisions retry safely; changed code/rules/data/decisions or a corrupted record cause a conflict, never overwrite. If a process is interrupted leaving `.write-lock`, inspect process state and the record first; do not automatically clear it. After software changes, use `--check` or `--score` for an already recorded session and let the next new session use the new version.

Scoring re-verifies original and outcome snapshots, rejects historical-price revisions, duplicates, corrupt records and missing future sessions. MR uses next-open entry, a two-bar settlement lock, stop-first daily-bar replay and 0.6 percentage-point modeled round-trip cost; a gap-down stop uses the worse opening price. Invalid next-open risk bounds skip entry; zero-volume/locked exit bars remain unverified. Daily OHLC cannot establish intraday settlement availability, queue fills, slippage or actual execution. Momentum 21-session close returns are informational, not portfolio P&L. Pending outcomes never count as wins/losses.

## Remaining gates / rollback

- Legacy history still has 76 invalid rows across 12 files; preserved-source probes confirm at least one upstream error. It remains excluded from this paper/training path. See `2026-09-22-reconciliation/source-probes.json`.
- No proven historical membership/adjustment lineage; no prospective sample with mature results; no validated broker/fill/risk integration. **No blanket “data clean/no leakage” or live-trading approval.**
- Keep source snapshots, candidates and paper records together for reproducibility; they are local runtime files ignored by Git, not a backup service. Do not prune snapshots referenced by a record or candidate.
- Safe fallback is no new paper record on failure. Retain evidence, fix the source/contract, then retry within the valid window. Do not fall back to the old auto-push BAT or overwrite an old prediction to recover a failed day.
- Before considering real money: review prospective outcomes and quarterly instability, verify corporate-action/PIT lineage, validate execution/risk limits, and obtain an explicit deployment decision.

## Verification

- Full suite: 264 passed, 23 existing dependency/deprecation warnings. No skipped tests. Final status-format correction also passed all 43 candidate/paper integration checks.
- New-path branch-inclusive coverage: reconciliation 91%, candidate release 91%, paper runner 87%, paper scorer 81%; combined 88%. This is scoped new-path coverage, not a claim of 88% for the entire repository.
- Evidence: `2026-09-26-market-junit.xml`, `2026-09-26-market-coverage.json`, and `2026-09-26-market-evidence.json`. The last file hashes the real source manifest, first immutable paper record and staged candidate.
- Actual CLI checks: refresh/run succeeded; separate scorer returned 10 pending/0 resolved/0 errors; final-source preview remained data-ready and source-verified. Candidate loading succeeded and `available_at('2026-09-25')` correctly returned false.
- Scoped Git checks: no whitespace errors; no changes to tracked legacy `data/raw`, `data/models` or `data/pipeline` files. Two pre-existing untracked dashboard/log files were left alone.
- MLE/TDD workflow kept source checks, temporal guards, fixed candidate gates, RED/GREEN checkpoints, and the rule-only fallback explicit. Passing tests or offline gates is not a live-money approval.

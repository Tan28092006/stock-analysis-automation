# Agent workflow — local backend / evidence first

This is the onboarding router, not a second set of trading rules. Root
[`AGENTS.md`](../AGENTS.md) owns authority and acceptance. Read this guide once,
then follow only the route relevant to the user's task. Do not load every audit,
skill, data file or experiment report into context.

## Five fresh-session answers

1. **System:** a local Vietnamese EOD research backend, with MR and momentum
   rules, optional ML research and paper observations. No broker execution is
   authorized by startup or a green test. User operation is CLI/chat-first.
2. **Structure:** module owners are in the table below; runtime `data/` artifacts
   are not source code and are generally ignored by Git.
3. **Run:** establish checkout/runtime with the discovery commands below. Run
   signals only when asked, using the documented paper/CLI lane; data refresh
   and scheduled collection are separate state-changing workflows.
4. **Verify:** use focused tests for the owned module. Strategy/data changes also
   require all registered pinned replays, books and timeslices; protocol
   validation alone is not a backtest. No CI status implies live approval.
5. **Progress / next step:** the [data/ML roadmap](plans/2026-10-04-data-ml-research-plan.md)
   owns its task ledger and dependencies. The [04/10 system review](audits/2026-10-04-system-design-review.md)
   records operational limitations. Dates and revision hashes bound evidence;
   recheck current state rather than treating a document as a live dashboard.

## Safe discovery from repository root

Each line is a separate command, compatible with PowerShell. No network,
refresh, training, scheduling, signal generation or broker calls are needed.

```text
git status --short
git branch --show-current
git rev-parse HEAD
python --version
python -B -m scripts.research_gate --help
python -B -m scripts.momentum_books --help
python -B -m scripts.research_timeslices --help
python -B -m stock_agent.pipeline.paper_runner --help
```

Use `rg --files stock_agent scripts tests configs docs` for targeted discovery,
then `rg -n <term> <owned-path>`. Do not read `.env`, cookies or credentials into
agent output. Do not recursively scan `data/` during startup. Do not install
dependencies, initialize a new worktree or alter MCP/plugin setup implicitly.

## On-demand signal route — not startup

When the user explicitly requests local signals, inspect the [operator runbook](audits/2026-09-26-market-readiness.md#operator-runbook)
and current `stock_agent/pipeline/paper_runner.py`. For a verified, appropriately
fresh snapshot and a new isolated preview directory, the documented lane is:

```text
python -m stock_agent.pipeline.paper_runner --manifest <verified-manifest> --check --output-dir <new-preview-directory>
```

**Effects:** scans and writes preview `latest.json`; `--check` means no immutable
paper-session record, not no filesystem writes. `--run` records only within its
ex-ante window; `--refresh` fetches new data; `--score` writes scores. Do not add
those flags during discovery or silently overwrite existing output. This guide
checks command discoverability/help, not a fresh app boot or production scan.
No broker orders, automatic ML promotion or refresh schedule changes are implied.

## Ownership and reference routes

| Task | Existing code / configuration | Read next | Focused verification |
|---|---|---|---|
| Prices, snapshots, source verification | `stock_agent/data/reconciliation.py`, `validation.py` | Snapshot fixture README; data/ML plan §3 | `tests/test_reconciliation.py`, `test_validation.py`, `test_pinned_market_fixtures.py` |
| Foreign flow ingestion / provenance | `stock_agent/data/foreign_flows.py`, `stock_agent/pipeline/foreign_refresh.py` | [foreign runbook](FOREIGN-FLOWS.md) | `test_foreign_flows_v2.py`, `test_foreign_refresh.py`, `test_foreign_refresh_retry.py` |
| PIT universe / calendar | `scripts/historical_universe.py`, `stock_agent/data/exchange_calendar.py`, `configs/research/` | Required research audits in root instructions | `test_historical_universe.py`, `test_exchange_calendar.py`, `test_vn100_membership.py` |
| Features, labels, split, ML candidate | `stock_agent/features/`, notably `temporal_validation.py`, `ensemble_model.py`, `model_release.py` | Data/ML plan §4–8; [training remediation](audits/2026-09-22-training-remediation.md) | `test_training_integrity.py`, `test_ensemble.py`, `test_model_release.py`, `test_mr_model_admission.py` |
| MR / momentum research | `scripts/research_gate.py`, `momentum_books.py`, `research_timeslices.py`, `period_backtest.py` | Required research audits; `configs/research/research_gate_v1.json` | Causality/accounting tests **and complete pinned suites** below |
| Paper, receipts, operational readiness | `stock_agent/pipeline/`, `scripts/execution_audit.py`, `run_eod_update.bat` | [system review](audits/2026-10-04-system-design-review.md) | `test_paper_runner.py`, `test_paper_scoring.py`, `test_execution_receipts.py`, `test_market_runtime_hardening.py` |
| Agent packaging / onboarding only | `AGENTS.md`, this file, [packaging](agent-packaging.md) | Relevant skill instructions; current request | Link/path validation, startup CLI smoke, `test_pytest_collection_safety.py`; no trading gate unless runtime/contracts change |

Test filenames in the last column are under `tests/`; module filenames in a row
share the indicated package unless a full path is shown. Inspect a file before
editing it. Legacy HTTP/dashboard and mutable position lanes are not automatically
equivalent to the tested research portfolio.

## Verification by effect

**Documentation / routing only:** validate changed links and referenced paths;
run startup help commands and collection safety test. Preserve unrelated dirty
files. Do not claim agent latency or task-success improvements without a measured
before/after task evaluation.

**Behavior change:** write a failing, isolated test, make the minimal change,
run owned tests and broader integration tests proportional to the boundary.
Record the exact commands, counts and failures. Never collect scratch programs;
`pytest.ini` deliberately restricts collection to `tests/`.

**Research / data-contract change:** preregister before observing new results.
Use verified manifests from immutable fixtures, never a fresh provider fetch as
an interchangeable CI input. In addition to focused tests, preserve the current
full workflow in [research-gate CI](../.github/workflows/research-gate.yml).
For new, nonexisting output paths:

```text
python -m scripts.research_gate --validate
python -m scripts.research_gate --manifest <verified-manifest> --output <new-run-directory>
python -m scripts.momentum_books --manifest <verified-manifest> --output <new-book-directory>
python -m scripts.research_timeslices --manifest <verified-manifest> --gate-results <new-run-directory>/results.json --book-results <new-book-directory>/results.json --output <new-slice-directory>
```

These placeholders must be resolved, verified and reported, not copied as a
passing run. Retain the VN100 and event jobs already in CI. An ML extension must
produce its own predictions and executable portfolio evidence in addition to,
not as a substitute for, the existing rule regression suite.

## Handoff contract

Report outcome, scoped changed paths, revision and uncommitted-source hashes
when relevant, exact checks and artifacts, failed/blocked/unrun checks, and one
next bounded action. Separate implemented capability, planned work, historical
development findings and prospective evidence. Do not mark a plan, `--validate`,
model AUC, paper picks or missing fills as profitable real execution. Pushing,
promotion, schedules and orders require appropriate user authority.

# Unified agent instructions for this workspace

## Operating principles
- Prefer evidence-backed, minimal changes.
- Keep edits scoped to the user request.
- Verify with tests when behavior changes.
- Avoid touching the trading pipeline unless the task explicitly asks for it.

## Strategy research regression gate
- For MR/momentum signals, selection, exits, sizing, costs, calendar, or data-contract changes, register the hypothesis before examining new results in `configs/research/research_gate_v1.json` (version the protocol when changing its design).
- Run the causality/accounting tests and the complete pinned-snapshot suite: `python -m scripts.research_gate --manifest <verified-manifest> --output <new-directory>`.
- Also retain the book suite and run `python -m scripts.research_timeslices --manifest <verified-manifest> --gate-results <gate-results.json> --book-results <book-results.json> --output <new-directory>`. Report causal VNINDEX up/down/transition episodes and fixed half-years, including 2026-01-01 through 2026-06-30. Do not use the pooled 2022-2026 return alone as acceptance; distinguish cash restart from inherited-portfolio attribution and positive net profit from merely losing less than the index.
- Preserve all required crisis/calendar blocks, controls, one-rule ablations, cost/delay stresses and source hashes. Missing evidence is BLOCKED; `--validate` alone is not a backtest pass.
- Read `docs/audits/2026-09-30-research-gate-spec.md` and `docs/audits/2026-09-30-strategy-reassessment.md` before interpreting results. Already inspected history is development data, never an untouched holdout.
- A green engineering run does not approve trading or deployment. Require separate prospective evidence and execution/data readiness; legacy foreign rows without provenance remain quarantined.

## Workspace conventions
- The repo is a VN30 stock-agent platform; agent-packaging files live alongside it without changing runtime behavior.
- Use `AGENTS.md` for durable instruction context.
- Use skill files for reusable workflows.
- Use MCP for external state or memory, not prompt stuffing.

## Agent packaging targets
- Codex should consume this file plus any `SKILL.md` adapters and MCP wiring.
- Antigravity should consume `.agents/skills/**/SKILL.md` and plugin manifests.
- Hermes-style skill format may be borrowed, but Hermes runtime is not a dependency.


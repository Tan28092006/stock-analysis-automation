# Unified agent instructions for this workspace

## Operating principles
- Prefer evidence-backed, minimal changes.
- Keep edits scoped to the user request.
- Verify with tests when behavior changes.
- Avoid touching the trading pipeline unless the task explicitly asks for it.
- User-facing operation is backend/CLI-first: do not build a UI unless explicitly requested. When the user asks to see signals, run the code locally and report results in chat. Keep automatic data collection separate from on-demand signal viewing; this preference does not authorize broker orders or changing existing data-refresh schedules.

## Strategy research regression gate
- For MR/momentum signals, selection, exits, sizing, costs, calendar, or data-contract changes, register the hypothesis before examining new results in `configs/research/research_gate_v1.json` (version the protocol when changing its design).
- Run the causality/accounting tests and the complete pinned-snapshot suite: `python -m scripts.research_gate --manifest <verified-manifest> --output <new-directory>`.
- Also retain the book suite and run `python -m scripts.research_timeslices --manifest <verified-manifest> --gate-results <gate-results.json> --book-results <book-results.json> --output <new-directory>`. Report causal VNINDEX up/down/transition episodes and fixed half-years, including 2026-01-01 through 2026-06-30. Do not use the pooled 2022-2026 return alone as acceptance; distinguish cash restart from inherited-portfolio attribution and positive net profit from merely losing less than the index.
- Preserve all required crisis/calendar blocks, controls, one-rule ablations, cost/delay stresses and source hashes. Missing evidence is BLOCKED; `--validate` alone is not a backtest pass.
- Activity is a separate user acceptance requirement, alongside positive net profit and excess over VNINDEX. Report monthly flat-to-long entries, add-on buys and distinct entry days separately; identify first-session portfolio construction and zero-new-entry months. Do not hide sparse opportunities inside total fills or a half-year average. Declare a numerical activity floor before a new experiment once agreed with the user; never force orders or loosen gates merely to meet a quota.
- Read `docs/audits/2026-09-30-research-gate-spec.md` and `docs/audits/2026-09-30-strategy-reassessment.md` before interpreting results. Already inspected history is development data, never an untouched holdout.
- A green engineering run does not approve trading or deployment. Require separate prospective evidence and execution/data readiness; legacy foreign rows without provenance remain quarantined.

## Workspace conventions
- The repo is a VN30 stock-agent platform; agent-packaging files live alongside it without changing runtime behavior.
- Use `AGENTS.md` for durable instruction context.
- Use skill files for reusable workflows.
- Use MCP for external state or memory, not prompt stuffing.

## Agent startup and task routing
- Start with `docs/agent-workflow.md`: repository ownership map, safe discovery commands, test routes and handoff contract. Startup must not refresh data, scan signals, train models or change schedules.
- For the current data/preprocessing/ensemble research roadmap, use `docs/plans/2026-10-04-data-ml-research-plan.md`. Its work ledger is the source of progress for that roadmap; planned tasks are not implemented capabilities.
- For operational/system boundaries, consult `docs/audits/2026-10-04-system-design-review.md`; for foreign collection, consult `docs/FOREIGN-FLOWS.md`.
- `docs/scrum_development_plan.md` is historical T+2 planning, not current MR/momentum acceptance. Runtime code, dated evidence and the registered research protocol take precedence over legacy README examples or test counts.
- Load only the task-relevant references. Before handing off, record changed paths, exact verification commands/results, source revision and dirty-tree limitations, unresolved blockers and the next bounded action. Do not infer permission to push, promote or place orders from a successful check.

## Agent packaging targets
- Codex should consume this file plus any `SKILL.md` adapters and MCP wiring.
- Antigravity should consume `.agents/skills/**/SKILL.md` and plugin manifests.
- Hermes-style skill format may be borrowed, but Hermes runtime is not a dependency.
- Swing desk (advisory pre-trade check, bottom-fish check, desk run, journal review; universe VN100 by default, `--universe vn30` selectable): `.claude/skills/swing-desk/SKILL.md` (Claude Code) and `.codex/skills/swing-desk/SKILL.md` (Codex); code `stock_agent/swing_desk/`, folder `swing_desk/`.


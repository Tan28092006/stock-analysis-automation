# Unified agent instructions for this workspace

## Operating principles
- Prefer evidence-backed, minimal changes.
- Keep edits scoped to the user request.
- Verify with tests when behavior changes.
- Avoid touching the trading pipeline unless the task explicitly asks for it.

## Workspace conventions
- The repo is a VN30 stock-agent platform; agent-packaging files live alongside it without changing runtime behavior.
- Use `AGENTS.md` for durable instruction context.
- Use skill files for reusable workflows.
- Use MCP for external state or memory, not prompt stuffing.

## Agent packaging targets
- Codex should consume this file plus any `SKILL.md` adapters and MCP wiring.
- Antigravity should consume `.agents/skills/**/SKILL.md` and plugin manifests.
- Hermes-style skill format may be borrowed, but Hermes runtime is not a dependency.


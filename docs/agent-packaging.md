# Agent Packaging

This repository now includes a portable skill bundle that can be mapped to Codex and Antigravity.

Runtime-independent onboarding and verification are documented in [agent workflow](agent-workflow.md). Start from the root `AGENTS.md`; load skill adapters only when relevant to the task. The [current data/ML roadmap](plans/2026-10-04-data-ml-research-plan.md) owns progress for that work, rather than creating a second feature tracker here.

## Components
- `AGENTS.md` for durable repo instructions.
- `.agents/skills/karpathy-guidelines/SKILL.md` for Karpathy-style operating principles.
- `.agents/skills/agent-workflows/SKILL.md` for standardized workflow execution.
- `.agents/skills/mempalace-mcp/SKILL.md` for memory via MCP.
- `.agents/plugin.json` for Antigravity-style plugin bundling.
- `.codex-plugin/plugin.json` for Codex-facing packaging metadata.

## Notes
- Hermes is treated as a format reference only.
- MemPalace is wired as external memory, not as prompt text.


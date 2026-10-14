"""Skill/adapters and tracked desk files stay consistent (no drift between hosts)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = ROOT / ".claude/skills/swing-desk/SKILL.md"
CODEX = ROOT / ".codex/skills/swing-desk/SKILL.md"


def frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, text[match.end():]


def test_claude_and_codex_skills_share_frontmatter_and_body():
    a, body_a = frontmatter(CLAUDE)
    b, body_b = frontmatter(CODEX)
    assert a["name"] == b["name"] == "swing-desk"
    assert set(a) == set(b) == {"name", "description"}
    assert a["description"] == b["description"]
    assert body_b.startswith(body_a.rstrip("\n"))
    assert "## Codex notes" in body_b[len(body_a.rstrip()):]


def test_skill_is_host_neutral_with_delegation_section_and_modes():
    fields, body = frontmatter(CLAUDE)
    for word in ("pre-trade check", "daily desk run", "review"):
        assert word in fields["description"]
    assert "## Running the 4-analyst flow" in body
    assert "Claude Code" in body and "Codex" in body and "not isolated" in body


def test_agents_md_points_to_both_skill_paths():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert ".claude/skills/swing-desk/SKILL.md" in text and ".codex/skills/swing-desk/SKILL.md" in text


def test_example_journal_parses_and_prompts_exist():
    from stock_agent.swing_desk.journal import parse_journal
    text = (ROOT / "swing_desk/portfolio.example.txt").read_text(encoding="utf-8")
    journal = parse_journal(text, price_lookup=lambda t, d: 40000.0)
    assert journal.errors == [] and len(journal.closed) == 1
    for name in ("ANALYST.md", "OUTPUT_CONTRACT.md", "SENTIMENT_RUBRIC.md"):
        assert (ROOT / "swing_desk/prompts" / name).is_file()

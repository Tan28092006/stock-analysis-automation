from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestAgentPackaging(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_required_files_exist(self) -> None:
        required = [
            self.root / "AGENTS.md",
            self.root / ".agents" / "skills" / "karpathy-guidelines" / "SKILL.md",
            self.root / ".agents" / "skills" / "agent-workflows" / "SKILL.md",
            self.root / ".agents" / "skills" / "mempalace-mcp" / "SKILL.md",
            self.root / ".agents" / "plugin.json",
            self.root / ".codex-plugin" / "plugin.json",
            self.root / "docs" / "agent-packaging.md",
        ]
        missing = [str(path) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Missing packaging files: {missing}")

    def test_antigravity_plugin_manifest_is_well_formed(self) -> None:
        manifest_path = self.root / ".agents" / "plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "unified-skill-pack")
        self.assertIn(".agents/skills/karpathy-guidelines/SKILL.md", manifest["skills"])
        self.assertIn("mcpServers", manifest)
        self.assertIn("mempalace", manifest["mcpServers"])

    def test_codex_plugin_manifest_points_at_agent_context(self) -> None:
        manifest_path = self.root / ".codex-plugin" / "plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "unified-skill-pack")
        self.assertIn("AGENTS.md", manifest["skills"])
        self.assertIn(".agents/skills/mempalace-mcp/SKILL.md", manifest["skills"])


if __name__ == "__main__":
    unittest.main()

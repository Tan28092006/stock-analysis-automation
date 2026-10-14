"""Desk hashes must not depend on checkout line endings (Windows autocrlf vs LF)."""
import shutil
from pathlib import Path

from stock_agent.swing_desk import config


def test_code_hash_is_line_ending_independent(tmp_path, monkeypatch):
    pkg = Path(config.__file__).parent
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for target, eol in ((lf, b"\n"), (crlf, b"\r\n")):
        target.mkdir()
        for src in pkg.glob("*.py"):
            text = src.read_bytes().replace(b"\r\n", b"\n")
            (target / src.name).write_bytes(text.replace(b"\n", eol))
    hashes = []
    for target in (lf, crlf):
        monkeypatch.setattr(config, "__file__", str(target / "config.py"))
        hashes.append(config.code_hash())
    assert hashes[0] == hashes[1]

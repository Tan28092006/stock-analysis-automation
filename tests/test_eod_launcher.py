"""The scheduled launcher must only run the isolated paper pipeline.

Runtime checks copy it to a temporary directory and substitute a local Python
stub; they never invoke the real provider, modify a ledger, or publish Git data.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "run_eod_update.bat"


def _commands() -> list[str]:
    return [line.strip().lower() for line in LAUNCHER.read_text().splitlines()
            if line.strip() and not line.strip().lower().startswith("rem ")]


def test_scheduled_launcher_routes_to_isolated_paper_runner():
    commands = _commands()
    assert any("-m stock_agent.pipeline.paper_runner --refresh --run" in line for line in commands)
    assert not any("-m stock_agent.pipeline.eod_update" in line for line in commands)


def test_scheduled_launcher_never_mutates_git_or_publishes():
    assert not any(line.startswith("git ") for line in _commands())


def _run_stub(tmp_path: Path, exit_code: int):
    # Fail before execution on the old launcher; never run the production job.
    test_scheduled_launcher_routes_to_isolated_paper_runner()
    test_scheduled_launcher_never_mutates_git_or_publishes()
    root = tmp_path / "paper workspace"
    module_dir = root / "stock_agent" / "pipeline"
    module_dir.mkdir(parents=True)
    (module_dir / "paper_runner.py").write_text(
        "import os, sys\n"
        "print('STUB_CWD=' + os.getcwd())\n"
        "print('STUB_ARGS=' + ' '.join(sys.argv[1:]))\n"
        "print('STUB_STDERR', file=sys.stderr)\n"
        "raise SystemExit(int(os.environ['VN30_TEST_EXIT']))\n",
        encoding="utf-8",
    )
    launcher = root / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    env = {**os.environ, "VN30_PYTHON": sys.executable, "VN30_TEST_EXIT": str(exit_code)}
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher)], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=20, check=False,
    )
    log = (root / "data" / "pipeline" / "eod_update.log").read_text()
    return result, log, root


@pytest.mark.skipif(os.name != "nt", reason="Windows scheduled launcher integration")
def test_scheduled_launcher_preserves_runner_failure_exit_code(tmp_path):
    result, log, _ = _run_stub(tmp_path, 7)
    assert result.returncode == 7
    assert "FAILED exit=7" in log
    assert "SUCCESS" not in log


@pytest.mark.skipif(os.name != "nt", reason="Windows scheduled launcher integration")
def test_scheduled_launcher_success_logs_output_and_uses_own_directory(tmp_path):
    result, log, root = _run_stub(tmp_path, 0)
    assert result.returncode == 0
    assert "SUCCESS" in log
    assert f"STUB_CWD={root}" in log
    assert "STUB_ARGS=--refresh --run" in log
    assert "STUB_STDERR" in log


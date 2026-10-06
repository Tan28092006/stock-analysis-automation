"""Suite-wide isolation for runtime side outputs.

EOD entry points write per-step status records (stock_agent.pipeline.eod_status).
Tests must never write them into the repository's data/ directory.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_eod_status_dir(tmp_path_factory):
    from stock_agent.pipeline.eod_status import ENV
    previous = os.environ.get(ENV)
    os.environ[ENV] = str(tmp_path_factory.mktemp("eod_status"))
    yield
    if previous is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = previous

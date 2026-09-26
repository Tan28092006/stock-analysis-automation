"""Default test collection must not execute ad-hoc, stateful scratch programs."""
from pathlib import Path

import pytest


def test_default_collection_is_scoped_to_the_isolated_test_directory():
    config = pytest.Config.fromdictargs({}, [])
    assert config.getini('testpaths') == ['tests']
    assert (Path(config.rootpath) / 'tests').is_dir()

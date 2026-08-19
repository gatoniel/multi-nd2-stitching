import copy

import pytest
import yaml

from multi_nd2_stitching.config import loads_config

MINIMAL = {
    "files": ["a.nd2", "b.nd2"],
    "grid_spacing": 55,
    "grid_spacing_error": 5,
    "positions": {
        "tile_a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "tile_b": {"start": [0, 0]},
    },
}


@pytest.fixture
def cfg_dict():
    """A minimal *valid* config. Mutate exactly one thing per test."""
    return copy.deepcopy(MINIMAL)


@pytest.fixture
def parse():
    return lambda d: loads_config(yaml.safe_dump(d))

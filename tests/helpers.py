"""Pure test helpers. Imported directly -- unlike conftest.py, which pytest loads."""

import numpy as np
import yaml

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.layout import build_layout
from multi_nd2_stitching.metadata import FileMeta, Metadata


def make_meta(
    n_files=2,
    nt=5,
    tiles=("tile_a", "tile_b"),
    spacing=55.0,
    axis="x",
    nz=80,
    ny=724,
    nx=724,
    voxel=0.1,
    paths=None,
):
    """Synthetic ND2 metadata: `tiles` laid out in a line, `spacing` um apart.

    This is the whole point of splitting metadata from geometry -- the layout
    layer is exercised without a single .nd2 file.
    """
    stage = [
        (i * spacing, 0.0) if axis == "x" else (0.0, i * spacing)
        for i, _ in enumerate(tiles)
    ]

    for i, _ in enumerate(tiles):
        stage.append((i * spacing, 0.0) if axis == "x" else (0.0, i * spacing))
    return Metadata(
        tuple(
            FileMeta(
                path=(paths[i] if paths else f"f{i}.nd2"),
                nt=nt,
                nz=nz,
                ny=ny,
                nx=nx,
                position_names=tuple(tiles),
                stage_um=tuple(stage),
                voxel_x_um=voxel,
            )
            for i in range(n_files)
        )
    )


def build(d, **meta_kw):
    return build_layout(loads_config(yaml.safe_dump(d)), make_meta(**meta_kw))


class FakeReader:
    """Volumes with a known translation, so correlation results are checkable.

    Each VolumeRef gets a deterministic random volume shifted by `shifts[ref]`.
    """

    def __init__(self, shape=(16, 64, 64), shifts=None, seed=0):
        self.shape = shape
        self.shifts = shifts or {}
        self.base = np.random.default_rng(seed).random(shape)
        self.reads = []

    def read(self, ref):
        self.reads.append(ref)
        dz, dy, dx = self.shifts.get(ref, (0, 0, 0))
        return np.roll(self.base, (dz, dy, dx), axis=(0, 1, 2))

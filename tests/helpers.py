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
    stage = [
        (i * spacing, 0.0) if axis == "x" else (0.0, i * spacing)
        for i, _ in enumerate(tiles)
    ]
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


def stub_files(tmp_path, n=2, prefix="f", size=100):
    """Create `n` placeholder .nd2 files and return their paths as strings.

    build_plan stamps each file's (path, size, mtime) into its cache keys, so
    the paths in a config have to exist even when the ND2 content is faked.
    Sizes differ per file so the stamps do too.
    """
    out = []
    for i in range(n):
        p = tmp_path / f"{prefix}{i}.nd2"
        p.write_bytes(b"x" * (size + i))
        out.append(str(p))
    return out


def grid_meta(coords, paths, nt=3, nz=4, ny=8, nx=8, voxel=0.1):
    """Metadata with explicit stage positions, so grids and rings are possible.

    `coords` maps tile name -> (x_um, y_um). make_meta only lays tiles out in a
    line, which cannot produce a cycle.
    """
    names = tuple(coords)
    stage = tuple(coords[n] for n in names)
    return Metadata(
        tuple(
            FileMeta(
                path=paths[i],
                nt=nt,
                nz=nz,
                ny=ny,
                nx=nx,
                position_names=names,
                stage_um=stage,
                voxel_x_um=voxel,
            )
            for i in range(len(paths))
        )
    )

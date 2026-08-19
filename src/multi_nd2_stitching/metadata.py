"""Everything read from the ND2 headers. No pixel data is touched here.

This is the *only* layer that talks to nd2. Splitting it out means the geometry
layer below can be tested with hand-written metadata and no microscopy files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import attrs
import cattrs
from cattrs.preconf.json import make_converter

logger = logging.getLogger(__name__)


@attrs.frozen
class FileMeta:
    """Header facts about one .nd2 file."""

    path: str
    nt: int
    nz: int
    ny: int
    nx: int
    position_names: tuple[str, ...]
    stage_um: tuple[tuple[float, float], ...]  # (x, y) per position, same order
    voxel_x_um: float

    def position_of(self, names: tuple[str, ...]) -> int | None:
        """Index of the single position matching any of `names`, or None."""
        hits = [i for i, n in enumerate(self.position_names) if n in names]
        if not hits:
            return None
        if len(hits) > 1:
            raise ValueError(
                f"{self.path}: names {names} match several positions {hits}"
            )
        return hits[0]


@attrs.frozen
class Metadata:
    files: tuple[FileMeta, ...]

    def __getitem__(self, i: int) -> FileMeta:
        return self.files[i]

    def __len__(self) -> int:
        return len(self.files)

    @property
    def nts(self) -> tuple[int, ...]:
        return tuple(f.nt for f in self.files)


def read_metadata(paths) -> Metadata:
    """Open each ND2 just long enough to read its headers."""
    import nd2

    metas = []
    for path in paths:
        with nd2.ND2File(str(path)) as f:
            points = f.experiment[1].parameters.points
            metas.append(
                FileMeta(
                    path=str(path),
                    nt=f.sizes.get("T", 1),
                    nz=f.sizes["Z"],
                    ny=f.sizes["Y"],
                    nx=f.sizes["X"],
                    position_names=tuple(p.name for p in points),
                    stage_um=tuple(
                        (p.stagePositionUm.x, p.stagePositionUm.y) for p in points
                    ),
                    voxel_x_um=f.voxel_size().x,
                )
            )
    return Metadata(tuple(metas))


# --- the one thing worth caching at this layer --------------------------------
# Reading headers means opening every file over a network mount. The result is a
# few hundred bytes, keyed by (path, size, mtime), so a moved or rewritten file
# invalidates itself. Deleting the cache changes nothing but runtime.

converter = make_converter()


@attrs.frozen
class FileStamp:
    """Identity of a file on disk, cheap to compute. attrs gives us __eq__."""

    path: str
    size: int
    mtime: int

    @classmethod
    def of(cls, path) -> FileStamp:
        st = Path(path).stat()
        return cls(path=str(path), size=st.st_size, mtime=int(st.st_mtime))


@attrs.frozen
class MetadataCache:
    stamp: tuple[FileStamp, ...]
    metadata: Metadata


def load_metadata(paths, cache: Path | None = None) -> Metadata:
    paths = [str(p) for p in paths]
    if cache is None:
        return read_metadata(paths)

    cache = Path(cache)
    stamp = tuple(FileStamp.of(p) for p in paths)
    if cache.exists():
        try:
            blob = converter.loads(cache.read_text(), MetadataCache)
        except (json.JSONDecodeError, cattrs.errors.ClassValidationError):
            # Corrupt file, or a FileMeta field added since it was written.
            # A stale cache is never a reason to fail -- just re-read.
            logger.debug(
                "ignoring unreadable metadata cache at %s", cache, exc_info=True
            )
        else:
            if blob.stamp == stamp:
                return blob.metadata

    meta = read_metadata(paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        converter.dumps(MetadataCache(stamp=stamp, metadata=meta), indent=1)
    )
    return meta

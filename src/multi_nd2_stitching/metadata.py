"""Everything read from the ND2 headers. No pixel data is touched here.

This is the *only* layer that talks to nd2. Splitting it out means the geometry
layer below can be tested with hand-written metadata and no microscopy files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import attrs
from cattrs.errors import BaseValidationError
from cattrs.preconf.json import make_converter

# A cache file we cannot read: decode error, wrong type, or a schema that has
# moved on since it was written. Never fatal -- but never silent either.
_PARSE_ERRORS = (OSError, ValueError, KeyError, TypeError, BaseValidationError)


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
    real_time_s: tuple[float | None, ...] = ()  # per local t; None if unrecorded

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


# --- absolute per-timepoint time -----------------------------------------------
# A Julian Day Number is already an absolute clock (unlike the elapsed-seconds
# columns nd2 also exposes), so it is what lets two different files be placed on
# one real-world timeline. JDN 2440587.5 == 1970-01-01T00:00:00Z.
_JDN_UNIX_EPOCH = 2440587.5

# A skipped/aborted timepoint means `loop_indices` names a frame that was never
# actually written. Reading its metadata is expected to fail; that failure is
# the signal, not a reason to give up on the rest of the file. Exact type is a
# best guess reasoned from nd2's public API, not observed against a real
# aborted acquisition -- narrow or widen this if a real one raises otherwise.
_FRAME_READ_ERRORS = (IndexError, KeyError, ValueError)


def _jdn_to_unix_s(jdn: float) -> float:
    """Julian Day Number -> POSIX seconds."""
    return (jdn - _JDN_UNIX_EPOCH) * 86400.0


def _first_seq_index_per_t(loop_indices) -> dict[int, int]:
    """For each T value, the seq_index of its first frame.

    `loop_indices` enumerates every *nominal* frame combination in nd2's own
    loop order (T, Z, C, P, ...). Regardless of that order, the first frame at
    which any given T value appears always has every other axis still at its
    initial 0 -- a property of the underlying itertools.product, not of any
    particular loop nesting -- so this is always (T=t, Z=0, C=0, P=0) without
    needing to know the loop order.
    """
    out: dict[int, int] = {}
    for i, idx in enumerate(loop_indices):
        out.setdefault(idx.get("T", 0), i)
    return out


def _read_real_times(f, nt: int, path: str) -> tuple[float | None, ...]:
    """One representative absolute timestamp per T value, None where missing."""
    t_to_seq = _first_seq_index_per_t(f.loop_indices)
    times: list[float | None] = []
    for t in range(nt):
        seq = t_to_seq.get(t)
        if seq is None:
            times.append(None)
            continue
        try:
            jdn = f.frame_metadata(seq).channels[0].time.absoluteJulianDayNumber
        except _FRAME_READ_ERRORS as e:
            warnings.warn(
                f"{path}: timepoint {t} has no recorded frame metadata "
                f"({type(e).__name__}); its real time is unknown",
                RuntimeWarning,
                stacklevel=3,
            )
            times.append(None)
            continue
        times.append(_jdn_to_unix_s(jdn))
    return tuple(times)


def read_metadata(paths) -> Metadata:
    """Open each ND2 just long enough to read its headers."""
    import nd2

    metas = []
    for path in paths:
        with nd2.ND2File(str(path)) as f:
            points = f.experiment[1].parameters.points
            nt = f.sizes.get("T", 1)
            metas.append(
                FileMeta(
                    path=str(path),
                    nt=nt,
                    nz=f.sizes["Z"],
                    ny=f.sizes["Y"],
                    nx=f.sizes["X"],
                    position_names=tuple(p.name for p in points),
                    stage_um=tuple(
                        (p.stagePositionUm.x, p.stagePositionUm.y) for p in points
                    ),
                    voxel_x_um=f.voxel_size().x,
                    real_time_s=_read_real_times(f, nt, str(path)),
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
        except _PARSE_ERRORS as e:
            # Corrupt file, or a FileMeta field added since it was written.
            # A stale cache is never a reason to fail -- just re-read.
            warnings.warn(
                f"{cache}: ignoring unreadable metadata cache "
                f"({type(e).__name__}); re-reading the ND2 headers",
                RuntimeWarning,
                stacklevel=2,
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

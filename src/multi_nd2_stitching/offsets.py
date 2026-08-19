"""What has to be computed, and how each unit of work is identified.

The cache key for a task is a hash of *exactly* the inputs that determine its
value -- the bytes on disk, the position index, the local timepoint, the crop.
Never the tile's name, never its index in a sorted list, never the global
timepoint. Rename a tile or prepend a file and every cached value stays valid,
because none of those change which pixels get correlated.
"""

from __future__ import annotations

import hashlib
import json

import attrs

from .config import clamp_z
from .layout import Layout
from .metadata import FileStamp, Metadata


def _digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@attrs.frozen
class Crop:
    """A 3D crop as plain ints, so it can go into a key."""

    z: tuple[int | None, int | None]
    y: tuple[int | None, int | None]
    x: tuple[int | None, int | None]

    @classmethod
    def of(cls, slices, nz: int) -> Crop:
        z, y, x = clamp_z(slices, nz)
        return cls(z=(z.start, z.stop), y=(y.start, y.stop), x=(x.start, x.stop))

    def as_slices(self):
        return (slice(*self.z), slice(*self.y), slice(*self.x))

    def key(self):
        return [list(self.z), list(self.y), list(self.x)]


@attrs.frozen
class VolumeRef:
    """Identifies one 3D volume by its content, not by anyone's naming scheme."""

    file: str  # short hash of the file's (path, size, mtime)
    position: int  # stage position index inside that file
    local_t: int  # timepoint inside that file
    nz: int  # number of z frames read

    def key(self):
        return [self.file, self.position, self.local_t, self.nz]


@attrs.frozen
class TimeTask:
    """Drift of one anchor tile between two consecutive global timepoints."""

    name: str
    t_from: int
    t_to: int
    src: VolumeRef
    dst: VolumeRef
    crop: Crop
    realign: bool = False

    kind = "time"

    @property
    def key(self) -> str:
        return _digest(["time", self.src.key(), self.dst.key(), self.crop.key()])

    def describe(self) -> str:
        tag = " (realign)" if self.realign else ""
        return f"time {self.name} {self.t_from}->{self.t_to}{tag}"


@attrs.frozen
class PairTask:
    """Offset between two overlapping tiles at one global timepoint."""

    a: str
    b: str
    axis: int
    t: int
    src: VolumeRef
    dst: VolumeRef
    crop: Crop
    shift_px: int

    kind = "pair"

    @property
    def key(self) -> str:
        return _digest(
            [
                "pair",
                self.src.key(),
                self.dst.key(),
                self.axis,
                self.shift_px,
                self.crop.key(),
            ]
        )

    def describe(self) -> str:
        return f"pair {self.a}|{self.b} axis={self.axis} t={self.t}"


@attrs.frozen
class Plan:
    time_tasks: tuple[TimeTask, ...]
    pair_tasks: tuple[PairTask, ...]

    @property
    def tasks(self) -> tuple:
        return self.time_tasks + self.pair_tasks

    def pending(self, store) -> tuple:
        """The whole point: what still has to be run."""
        return tuple(t for t in self.tasks if t.key not in store)

    def at(self, t: int) -> tuple:
        return tuple(
            task
            for task in self.tasks
            if (task.t_to if isinstance(task, TimeTask) else task.t) == t
        )

    def between(self, t0: int, t1: int) -> Plan:
        """Restrict to a timepoint window, for 'why is t=21 bad'."""
        return Plan(
            time_tasks=tuple(x for x in self.time_tasks if t0 <= x.t_to < t1),
            pair_tasks=tuple(x for x in self.pair_tasks if t0 <= x.t < t1),
        )


def _file_keys(meta: Metadata) -> tuple[str, ...]:
    return tuple(_digest(attrs.asdict(FileStamp.of(f.path))) for f in meta.files)


def build_plan(layout: Layout, meta: Metadata) -> Plan:
    """Enumerate every offset the config implies. Pure; touches no pixel data."""
    cfg = layout.config
    fkeys = _file_keys(meta)
    crop = Crop.of(cfg.slices, layout.nz)
    realign_crop = Crop.of(cfg.realignment_slices, layout.nz)

    def ref(name: str, t: int) -> VolumeRef:
        file_i, pos = layout.frame_index(name, t)
        _, local_t = layout.locate(t)
        return VolumeRef(fkeys[file_i], pos, local_t, layout.nz)

    time_tasks = []
    for i, name in enumerate(layout.tiles):
        for t in range(1, layout.nt):
            if not layout.is_anchor[t, i]:
                continue
            if not layout.tile_alive[t - 1, i]:
                continue  # nothing to drift from; check_layout reports this
            realign = name in cfg.realigned_at(t)
            time_tasks.append(
                TimeTask(
                    name=name,
                    t_from=t - 1,
                    t_to=t,
                    src=ref(name, t - 1),
                    dst=ref(name, t),
                    crop=realign_crop if realign else crop,
                    realign=realign,
                )
            )

    pair_tasks = []
    for t in range(layout.nt):
        for p in layout.pairs_at(t):
            pair_tasks.append(
                PairTask(
                    a=p.a,
                    b=p.b,
                    axis=p.axis,
                    t=t,
                    src=ref(p.a, t),
                    dst=ref(p.b, t),
                    crop=crop,
                    shift_px=layout.shift_px,
                )
            )

    return Plan(tuple(time_tasks), tuple(pair_tasks))

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

_AXIS_NAME = {0: "z", 1: "y", 2: "x"}


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

    def free_axis(self, axis: int) -> Crop:
        """Drop the restriction on one lateral axis.

        A neighbour correlation lives on the overlap strip at the tile edge
        along `axis`; cropping there removes exactly the signal it needs.
        The z restriction and the *other* lateral axis still apply -- those
        cut planes and rows that are bad in both tiles alike.
        """
        return attrs.evolve(self, **{_AXIS_NAME[axis]: (None, None)})


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
class SpectrumRef:
    """One rfftn input, identified exactly.

    A time task feeds the whole cropped volume; a pair task feeds the overlap
    strip, which differs depending on whether the tile is the parent or the
    child of the pair -- so `axis`/`side`/`shift_px` are part of the identity.
    """

    volume: VolumeRef
    crop: Crop
    precision: str
    axis: int | None = None
    side: str | None = None
    shift_px: int = 0


@attrs.frozen
class TimeTask:
    """Drift of one anchor tile between two consecutive global timepoints."""

    name: str
    t_from: int
    t_to: int
    src: VolumeRef
    dst: VolumeRef
    crop: Crop
    precision: str = "float32"
    realign: bool = False
    shaped_peak: bool = False

    kind = "time"

    @property
    def key(self) -> str:
        return _digest(
            [
                "time",
                self.src.key(),
                self.dst.key(),
                self.crop.key(),
                self.precision,
                self.shaped_peak,
            ]
        )

    def spectrum_refs(self) -> tuple[SpectrumRef, SpectrumRef]:
        return (
            SpectrumRef(self.src, self.crop, self.precision),
            SpectrumRef(self.dst, self.crop, self.precision),
        )

    def describe(self) -> str:
        tag = " (realign)" if self.realign else ""
        tag += " (shaped)" if self.shaped_peak else ""
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
    precision: str = "float32"
    shaped_peak: bool = False

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
                self.precision,
                self.shaped_peak,
            ]
        )

    def spectrum_refs(self) -> tuple[SpectrumRef, SpectrumRef]:
        return (
            SpectrumRef(
                self.src, self.crop, self.precision, self.axis, "parent", self.shift_px
            ),
            SpectrumRef(
                self.dst, self.crop, self.precision, self.axis, "child", self.shift_px
            ),
        )

    def describe(self) -> str:
        tag = " (shaped)" if self.shaped_peak else ""
        return f"pair {self.a}|{self.b} axis={self.axis} t={self.t}{tag}"


@attrs.frozen
class Plan:
    time_tasks: tuple[TimeTask, ...]
    pair_tasks: tuple[PairTask, ...]

    @property
    def tasks(self) -> tuple:
        """Every task, ordered by timepoint.

        Ordering is not cosmetic: a volume is needed by the pair tasks at t and
        by the time tasks at t and t+1. Grouping by timepoint bounds the working
        set at roughly two timepoints, which is what makes VolumeCache viable.
        Running all time tasks before all pair tasks would evict and re-read
        every volume twice.
        """
        return tuple(
            sorted(
                self.time_tasks + self.pair_tasks,
                key=lambda x: (
                    x.t_to if isinstance(x, TimeTask) else x.t,
                    0 if isinstance(x, TimeTask) else 1,
                ),
            )
        )

    def spectrum_uses(self, tasks=None):
        """How many times each rfftn result is needed.

        Time tasks are where this pays: an anchor's spectrum at t is the `dst`
        of the (t-1 -> t) task and the `src` of the (t -> t+1) one, so caching
        halves the transforms. Pair spectra are almost always single-use,
        because a tile is parent to one neighbour and child to another --
        different strips, different transforms. The counter sorts that out
        without anyone having to special-case it.

        Pass `tasks` when only part of the plan will run -- counting the whole
        plan overstates the remaining uses of every spectrum, so nothing ever
        reaches zero and the cache never releases anything.
        """
        from collections import Counter

        c = Counter()
        for task in self.tasks if tasks is None else tasks:
            for ref in task.spectrum_refs():
                c[ref] += 1
        return c

    def volume_uses(self, tasks=None):
        """How many times each volume is needed. Drives the reader's eviction."""
        from collections import Counter

        c = Counter()
        for task in self.tasks if tasks is None else tasks:
            c[task.src] += 1
            c[task.dst] += 1
        return c

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


def file_keys(meta: Metadata) -> tuple[str, ...]:
    return tuple(_digest(attrs.asdict(FileStamp.of(f.path))) for f in meta.files)


def build_plan(layout: Layout, meta: Metadata, precision: str = "float32") -> Plan:
    """Enumerate every offset the config implies. Pure; touches no pixel data."""
    cfg = layout.config
    fkeys = file_keys(meta)
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
                    precision=precision,
                    realign=realign,
                    shaped_peak=name in cfg.shaped_peak_at(t),
                )
            )

    pair_tasks = []
    for t in range(layout.nt):
        shaped_at_t = cfg.shaped_peak_at(t)
        for p in layout.pairs_at(t):
            pair_tasks.append(
                PairTask(
                    a=p.a,
                    b=p.b,
                    axis=p.axis,
                    t=t,
                    src=ref(p.a, t),
                    dst=ref(p.b, t),
                    crop=crop.free_axis(p.axis),
                    shift_px=layout.shift_px,
                    precision=precision,
                    shaped_peak=(
                        f"{p.a},{p.b}" in shaped_at_t or f"{p.b},{p.a}" in shaped_at_t
                    ),
                )
            )

    return Plan(tuple(time_tasks), tuple(pair_tasks))

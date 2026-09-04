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
from .layout import Layout, corner_direction
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
    """Drift of one anchor tile between two consecutive *compacted*
    timepoints -- `t_from`/`t_to` are always exactly one apart in this
    numbering, even when exclude_at has cut raw timepoints out of the gap
    between them. `raw_gap` records how many raw timepoints that really was
    (1 when nothing was excluded there); it is purely informational -- not
    part of `key`, since it is derived from t alone and not from the pixels
    being correlated, the same reason a comment never belongs in the key.
    """

    name: str
    t_from: int
    t_to: int
    src: VolumeRef
    dst: VolumeRef
    crop: Crop
    precision: str = "float32"
    realign: bool = False
    shaped_peak: bool = False
    near: tuple[int, int, int] | None = None
    raw_gap: int = 1

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
                self.near,
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
        tag += f" (jump over {self.raw_gap - 1} excluded)" if self.raw_gap > 1 else ""
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
    near: tuple[int, int, int] | None = None
    realign: bool = False

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
                self.near,
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
        tag = " (realign)" if self.realign else ""
        tag += " (shaped)" if self.shaped_peak else ""
        return f"pair {self.a}|{self.b} axis={self.axis} t={self.t}{tag}"


@attrs.frozen
class CornerTask:
    """Offset between two diagonally-adjacent tiles at one global timepoint.

    Unlike `PairTask`, there is no single axis to free -- a diagonal pair's
    overlap is a rectangle near the corner, not a full-length edge strip, so
    both `crop_a` and `crop_b` restrict *both* lateral axes, each to a
    `shift_px`-narrower band -- the same `n - shift_px` overlap-strip
    convention `crop_for_alignment`/`trim_for` already use for a `Pair`,
    just applied on both axes independently instead of one.

    `nominal` is the mirror of `PairTask.shift_px`'s `offset[axis] +=
    shift_px`: the crop only ever lets the correlation measure the
    *correction* to the nominal `corner_direction` guess, not the tiles'
    true origin-to-origin offset, so `run_task` adds `nominal` back.
    """

    a: str
    b: str
    t: int
    src: VolumeRef
    dst: VolumeRef
    crop_a: Crop
    crop_b: Crop
    nominal: tuple[int, int, int]
    precision: str = "float32"

    kind = "corner"

    @property
    def key(self) -> str:
        return _digest(
            [
                "corner",
                self.src.key(),
                self.dst.key(),
                self.crop_a.key(),
                self.crop_b.key(),
                list(self.nominal),
                self.precision,
            ]
        )

    def spectrum_refs(self) -> tuple[SpectrumRef, SpectrumRef]:
        return (
            SpectrumRef(self.src, self.crop_a, self.precision),
            SpectrumRef(self.dst, self.crop_b, self.precision),
        )

    def describe(self) -> str:
        return f"corner {self.a}|{self.b} t={self.t}"


@attrs.frozen
class Plan:
    time_tasks: tuple[TimeTask, ...]
    pair_tasks: tuple[PairTask, ...]
    corner_tasks: tuple[CornerTask, ...] = ()

    @property
    def tasks(self) -> tuple:
        """Every task, ordered by timepoint.

        Ordering is not cosmetic: a volume is needed by the pair tasks at t and
        by the time tasks at t and t+1. Grouping by timepoint bounds the working
        set at roughly two timepoints, which is what makes VolumeCache viable.
        Running all time tasks before all pair tasks would evict and re-read
        every volume twice. Corner tasks sort with pair tasks -- same reasoning,
        both correlate already-alive tiles at t rather than driving t forward.
        """
        return tuple(
            sorted(
                self.time_tasks + self.pair_tasks + self.corner_tasks,
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
            corner_tasks=tuple(x for x in self.corner_tasks if t0 <= x.t < t1),
        )


def file_keys(meta: Metadata) -> tuple[str, ...]:
    return tuple(_digest(attrs.asdict(FileStamp.of(f.path))) for f in meta.files)


def _corner_crop(
    cfg, layout, meta, a: str, b: str, base: Crop
) -> tuple[Crop, Crop, tuple[int, int, int]]:
    """`a`'s and `b`'s own crops for a CornerTask, and the nominal offset to
    add back to the raw correlation (see `CornerTask`'s docstring).

    Both lateral axes get the same `n - shift_px` overlap-strip treatment
    `crop_for_alignment`/`trim_for` already give a `Pair`'s one axis, just
    independently on y and x: whichever side of `a` faces `b` keeps
    `[shift_px, n)` (its `shift_px` near the boundary cut away), and `b`'s
    matching side keeps `[0, n - shift_px)` -- the same physical strip, seen
    from each tile's own local origin. z is untouched -- both tiles are
    assumed nominally aligned there, same as an edge `Pair` never adjusts z.
    """
    dy_sign, dx_sign = corner_direction(cfg, meta, layout.tile, a, b)
    ny, nx, s = layout.ny, layout.nx, layout.shift_px

    def side(sign: int, extent: int) -> tuple[int, int]:
        return (s, extent) if sign > 0 else (0, extent - s)

    crop_a = attrs.evolve(base, y=side(dy_sign, ny), x=side(dx_sign, nx))
    crop_b = attrs.evolve(base, y=side(-dy_sign, ny), x=side(-dx_sign, nx))
    nominal = (0, dy_sign * s, dx_sign * s)
    return crop_a, crop_b, nominal


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

    # Override.at (and thus everything cfg.*_at reads) is in raw global-
    # timepoint numbering; `t` below is the compacted one layout.nt loops
    # over, so every lookup against the config translates through raw_t
    # first -- see StitchingConfig.exclude_at.
    time_tasks = []
    for i, name in enumerate(layout.tiles):
        for t in range(1, layout.nt):
            if not layout.is_anchor[t, i]:
                continue
            if not layout.tile_alive[t - 1, i]:
                continue  # nothing to drift from; check_layout reports this
            raw_t = layout.raw_t[t]
            realign = name in cfg.realigned_at(raw_t)
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
                    shaped_peak=name in cfg.shaped_peak_at(raw_t),
                    near=cfg.near_hint(name, raw_t),
                    raw_gap=raw_t - layout.raw_t[t - 1],
                )
            )

    pair_tasks = []
    for t in range(layout.nt):
        raw_t = layout.raw_t[t]
        shaped_at_t = cfg.shaped_peak_at(raw_t)
        realign_at_t = cfg.realigned_at(raw_t)
        for p in layout.pairs_at(t):
            realign = f"{p.a},{p.b}" in realign_at_t or f"{p.b},{p.a}" in realign_at_t
            pair_tasks.append(
                PairTask(
                    a=p.a,
                    b=p.b,
                    axis=p.axis,
                    t=t,
                    src=ref(p.a, t),
                    dst=ref(p.b, t),
                    crop=(realign_crop if realign else crop).free_axis(p.axis),
                    shift_px=layout.shift_px,
                    precision=precision,
                    shaped_peak=(
                        f"{p.a},{p.b}" in shaped_at_t or f"{p.b},{p.a}" in shaped_at_t
                    ),
                    near=(
                        cfg.near_hint(f"{p.a},{p.b}", raw_t)
                        or cfg.near_hint(f"{p.b},{p.a}", raw_t)
                    ),
                    realign=realign,
                )
            )

    corner_tasks = []
    corner_crop_cache: dict[
        tuple[str, str], tuple[Crop, Crop, tuple[int, int, int]]
    ] = {}
    for t in range(layout.nt):
        corner_at_t = cfg.corner_at(layout.raw_t[t])
        for c in layout.corners_at(t):
            if f"{c.a},{c.b}" not in corner_at_t and f"{c.b},{c.a}" not in corner_at_t:
                continue
            key = (c.a, c.b)
            if key not in corner_crop_cache:
                corner_crop_cache[key] = _corner_crop(cfg, layout, meta, c.a, c.b, crop)
            crop_a, crop_b, nominal = corner_crop_cache[key]
            corner_tasks.append(
                CornerTask(
                    a=c.a,
                    b=c.b,
                    t=t,
                    src=ref(c.a, t),
                    dst=ref(c.b, t),
                    crop_a=crop_a,
                    crop_b=crop_b,
                    nominal=nominal,
                    precision=precision,
                )
            )

    return Plan(tuple(time_tasks), tuple(pair_tasks), tuple(corner_tasks))

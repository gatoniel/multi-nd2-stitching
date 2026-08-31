"""Compositing tiles onto a canvas, one timepoint at a time.

The canvas geometry is fixed when the canvas is created and stored beside it.
Later runs place tiles into *that* frame rather than recomputing it, because
the required extent is a min/max over every timepoint: recomputing one offset
can move it, and a moved origin would silently shift every timepoint already
written. An oversized canvas is harmless -- it costs disk, and you shrink it by
deleting the canvas and blending again.

Each timepoint is recorded as done, so a crashed write costs one timepoint
rather than the run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import attrs
import numpy as np

from .coordinates import Coordinates


class CanvasMismatch(Exception):
    pass


# What a malformed JSON sidecar or log line can actually raise: a read error, a
# decode error, a missing key, or a field of the wrong type. Anything outside
# this set is a bug in our own code and should not be swallowed.
_PARSE_ERRORS = (OSError, ValueError, KeyError, TypeError)


@attrs.frozen
class CanvasGeometry:
    """Where world coordinates land in the canvas. Immutable once written."""

    origin: tuple[int, int, int]  # world zyx of canvas index (0, 0, 0)
    shape: tuple[int, int, int, int]  # (t, z, y, x)
    dtype: str

    @classmethod
    def required(
        cls,
        coords: Coordinates,
        tile_shape,
        nt: int,
        dtype: str,
        t0: int | None = None,
        t1: int | None = None,
        pad=0,
    ):
        """Tight frame around the placed tiles, optionally padded.

        `pad` buys room for timepoints not yet computed: without it, a canvas
        created from a prefix will almost certainly be too small once the rest
        of the run drifts, forcing --recreate. It may be one number (applied to
        y and x only) or three (z, y, x). Padding z is almost never wanted --
        the stack is shallow and drifts little, so it mostly multiplies the
        canvas volume.
        """
        pads = (
            (0, int(pad), int(pad)) if np.isscalar(pad) else tuple(int(v) for v in pad)
        )
        ext = coords.extent(tile_shape, t0, t1)
        origin = tuple(int(v) - p for v, p in zip(ext[:, 0], pads, strict=False))
        size = tuple(
            int(v) + 2 * p for v, p in zip(ext[:, 1] - ext[:, 0], pads, strict=False)
        )
        return cls(origin=origin, shape=(nt, *size), dtype=dtype)

    @property
    def spatial(self) -> tuple[int, int, int]:
        return self.shape[1:]

    def offset_of(self, world_zyx) -> np.ndarray:
        return (np.asarray(world_zyx) - np.array(self.origin)).astype(int)

    def violations(self, coords: Coordinates, tile_shape, t0: int, t1: int):
        """Tiles that would fall outside this canvas, if any."""
        out = []
        for t in range(t0, t1):
            for name, world in coords.at(t).items():
                lo = self.offset_of(world)
                hi = lo + np.array(tile_shape)
                if (lo < 0).any() or (hi > np.array(self.spatial)).any():
                    out.append(
                        f"t={t} '{name}' at canvas {tuple(int(v) for v in lo)}"
                        f"..{tuple(int(v) for v in hi)} is outside {self.spatial}"
                    )
        return out

    def slack(
        self,
        coords: Coordinates,
        tile_shape,
        nt: int,
        t0: int | None = None,
        t1: int | None = None,
    ) -> tuple[int, ...]:
        """How much bigger this canvas is than the coordinates now need."""
        need = CanvasGeometry.required(coords, tile_shape, nt, self.dtype, t0, t1)
        return tuple(
            int(a - b) for a, b in zip(self.spatial, need.spatial, strict=False)
        )

    # --- persistence ------------------------------------------------------
    @staticmethod
    def path_for(output) -> Path:
        return Path(str(output) + ".geometry.json")

    def save(self, output) -> None:
        p = self.path_for(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(attrs.asdict(self)))

    @classmethod
    def load(cls, output) -> CanvasGeometry | None:
        """The canvas's own frame, or None if it has never been written.

        A sidecar that exists but cannot be read is an error, not a miss.
        Returning None there would derive a fresh frame and silently remap every
        timepoint already on the canvas -- exactly the failure this file exists
        to prevent.
        """
        p = cls.path_for(output)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            return cls(
                origin=tuple(d["origin"]), shape=tuple(d["shape"]), dtype=d["dtype"]
            )
        except _PARSE_ERRORS as e:
            raise CanvasMismatch(
                f"{p} exists but cannot be read ({type(e).__name__}: {e}). "
                "The canvas frame is unknown, so writing to it would misplace "
                "the timepoints already there. Fix the file, or use --recreate "
                "to discard the canvas and start again."
            ) from e


def resolve_geometry(
    output, coords, tile_shape, nt, dtype, t0, t1, recreate=False, pad=0
):
    """Reuse the canvas's own geometry if it has one; otherwise derive a new one."""
    existing = None if recreate else CanvasGeometry.load(output)
    if existing is None:
        return CanvasGeometry.required(
            coords, tile_shape, nt, dtype, t0, t1, pad=pad
        ), True
    if existing.dtype != dtype:
        raise CanvasMismatch(
            f"canvas at {output} was written as {existing.dtype}, not {dtype}. "
            "Pass --dtype to match, a new --output, or --recreate."
        )
    if existing.shape[0] != nt:
        raise CanvasMismatch(
            f"canvas at {output} holds {existing.shape[0]} timepoints, "
            f"this config has {nt}. Use a new --output or --recreate."
        )
    bad = existing.violations(coords, tile_shape, t0, t1)
    if bad:
        raise CanvasMismatch(
            f"{len(bad)} tile(s) no longer fit the existing canvas at {output}:\n"
            + "\n".join(f"  - {b}" for b in bad[:5])
            + (f"\n  ... and {len(bad) - 5} more" if len(bad) > 5 else "")
            + "\nThe canvas frame is fixed once created. Use --recreate "
            "(discards what is written) or a new --output."
        )
    return existing, False


def _weights_key(name, t, coords, pairs, corners, tile_shape):
    here = coords.at(t)
    parts = []
    for pair in pairs:
        if name not in (pair.a, pair.b) or pair.a not in here or pair.b not in here:
            continue
        length = int((here[pair.b] - here[pair.a])[pair.axis])
        parts.append((pair.axis, pair.a == name, length))
    corner_parts = []
    for corner in corners:
        if (
            name not in (corner.a, corner.b)
            or corner.a not in here
            or corner.b not in here
        ):
            continue
        other = corner.b if corner.a == name else corner.a
        dy, dx = (int(v) for v in (here[other] - here[name])[1:])
        corner_parts.append((dy, dx))
    return (tile_shape, tuple(sorted(parts)), tuple(sorted(corner_parts)))


def _corner_taper(dy: int, dx: int, tile_shape):
    """The 2D ramp for one diagonal neighbour, and where it lands.

    Unlike an edge `Pair`'s ramp -- uniform across the whole perpendicular
    axis, correct because the tiles overlap along its *entire* length -- a
    diagonal neighbour only overlaps a small rectangle near the corner. This
    has to be a genuine 2D patch confined to that rectangle, or it would
    taper regions of the tile the neighbour never actually reaches.

    `dy`/`dx` are the neighbour's position minus `name`'s, straight out of
    `coords.at(t)`. The ramp length is `abs(dy)`/`abs(dx)` themselves -- the
    raw separation, exactly like a `Pair`'s `length` -- not
    `tile_shape - separation`: a region the neighbour's frame never actually
    reaches is never covered by anything else either, so the caller's
    divide-by-accumulated-weight makes any taper there invisible regardless
    of its exact width (see blend_weights' docstring). What matters is
    tapering correctly at the tile's own edge, same as every edge Pair does.
    """
    oy = abs(dy)
    ox = abs(dx)
    if not (0 < oy < tile_shape[1] and 0 < ox < tile_shape[2]):
        return None  # degenerate placement; leave this corner flat
    ry = np.linspace(0.01, 0.99, oy, dtype=np.float32)
    rx = np.linspace(0.01, 0.99, ox, dtype=np.float32)
    if dy > 0:  # neighbour is below -- name's *bottom* band faces it
        ry = ry[::-1]
        y_sl = slice(tile_shape[1] - oy, tile_shape[1])
    else:  # neighbour is above -- name's *top* band faces it
        y_sl = slice(0, oy)
    if dx > 0:  # neighbour is to the right -- name's *right* band faces it
        rx = rx[::-1]
        x_sl = slice(tile_shape[2] - ox, tile_shape[2])
    else:  # neighbour is to the left -- name's *left* band faces it
        x_sl = slice(0, ox)
    return y_sl, x_sl, ry[:, np.newaxis] * rx[np.newaxis, :]


# Divisor floor for uncovered voxels. Must stay far below the float32 ulp of
# the smallest ramp weight (0.01, ulp ~1e-9) so it perturbs nothing real.
EPS = 1e-12

WEIGHTS_CACHE_MAX = 24


def blend_weights(name, t, coords: Coordinates, pairs, corners, tile_shape, cache=None):
    """Linear ramps across every edge that has a neighbour, plus a 2D taper
    in the corner rectangle for every diagonal neighbour.

    Only the relative shape matters: the caller divides by the accumulated
    weight, so a solo region comes out unchanged whatever its weight was.

    The result depends only on the tile size and the ramp lengths, so an
    optional cache keyed on those turns this into a lookup while the placement
    holds still. Callers must not mutate it.

    The cache is BOUNDED, and that is not optional. A ramp length is the
    measured separation between two tiles, which drifts by a pixel or two from
    timepoint to timepoint, so distinct keys accumulate for as long as the run
    lasts -- at a 724x724 tile each entry is 2 MB, and an unbounded cache grows
    without limit. When it fills it is dropped whole: consecutive timepoints
    are what share keys, so recent entries are the only ones worth keeping.
    """
    key = _weights_key(name, t, coords, pairs, corners, tile_shape)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
        if len(cache) >= WEIGHTS_CACHE_MAX:
            cache.clear()

    here = coords.at(t)
    w = [
        np.ones(tile_shape[1], dtype=np.float32),
        np.ones(tile_shape[2], dtype=np.float32),
    ]
    for pair in pairs:
        if name not in (pair.a, pair.b) or pair.a not in here or pair.b not in here:
            continue
        length = int((here[pair.b] - here[pair.a])[pair.axis])
        if not 0 < length < tile_shape[pair.axis]:
            continue  # degenerate placement; leave this edge flat
        ramp = np.linspace(0.01, 0.99, length, dtype=np.float32)
        if pair.a == name:
            w[pair.axis - 1][-length:] = ramp[::-1]
        else:
            w[pair.axis - 1][:length] = ramp
    out = w[0][:, np.newaxis] * w[1][np.newaxis, :]

    # A diagonal neighbour never gets an edge Pair (it doesn't share a full
    # edge), so blend_weights would otherwise leave the corner rectangle at
    # full weight regardless of the real overlap. `min` rather than another
    # multiply: each neighbour only ever imposes an upper bound on how much
    # weight `name` may claim near it, so this is a no-op wherever the edge
    # ramps above already tapered that rectangle down this far or further --
    # exactly the case when a third tile completes the square normally.
    for corner in corners:
        if (
            name not in (corner.a, corner.b)
            or corner.a not in here
            or corner.b not in here
        ):
            continue
        other = corner.b if corner.a == name else corner.a
        dy, dx = (int(v) for v in (here[other] - here[name])[1:])
        patch = _corner_taper(dy, dx, tile_shape)
        if patch is None:
            continue
        y_sl, x_sl, ramp2d = patch
        out[y_sl, x_sl] = np.minimum(out[y_sl, x_sl], ramp2d)

    if cache is not None:
        cache[key] = out
    return out


def tile_boxes(t, layout, coords: Coordinates, geometry: CanvasGeometry) -> dict:
    """Where each tile lands on the canvas, as (lo, hi) in zyx.

    Known before a single voxel is read, which is the point: the covered region
    never has to be discovered by scanning a mask.
    """
    tile_shape = np.array((layout.nz, layout.ny, layout.nx))
    out = {}
    for name in layout.tiles_at(t):
        if name not in coords.at(t):
            continue
        lo = geometry.offset_of(coords[t, name])
        out[name] = (
            tuple(int(v) for v in lo),
            tuple(int(v) for v in lo + tile_shape),
        )
    return out


def boxes_bbox(boxes) -> list[list[int]] | None:
    """Union of tile boxes -- the analytic equivalent of bbox_of(inds)."""
    if not boxes:
        return None
    lo = np.array([b[0] for b in boxes.values()])
    hi = np.array([b[1] for b in boxes.values()])
    return [[int(lo[:, a].min()), int(hi[:, a].max())] for a in range(3)]


def hits_a_box(sl, boxes) -> bool:
    return any(
        all(sl[a].start < hi[a] and lo[a] < sl[a].stop for a in range(3))
        for lo, hi in boxes.values()
    )


def load_timepoint(reader, refs, t, names) -> dict:
    """Every tile of one timepoint, read as a single batch where possible."""
    wanted = {name: refs[(t, name)] for name in names if (t, name) in refs}
    if not wanted:
        return {}
    read_many = getattr(reader, "read_many", None)
    if read_many is None:
        return {name: reader.read(ref) for name, ref in wanted.items()}
    arrays = read_many(list(wanted.values()))
    return {name: arrays[ref] for name, ref in wanted.items()}


def z_segments(boxes, region_z):
    """Split the region's z range where the set of covering tiles changes.

    `counts` -- the sum of tile weights at a voxel -- depends on z only through
    *which tiles span that z*. Tiles do have z offsets, but only a handful of
    distinct ones, so a few 2D planes describe the whole 3D array. Each plane
    is a few hundred KB and stays in cache; the 3D counts array was the size of
    the canvas and had to be accumulated tile by tile.
    """
    edges = {0, int(region_z)}
    for lo, hi in boxes.values():
        for z in (lo[0], hi[0]):
            if 0 < z < region_z:
                edges.add(int(z))
    marks = sorted(edges)
    return [
        (za, zb, [n for n, (lo, hi) in boxes.items() if lo[0] <= za and zb <= hi[0]])
        for za, zb in itertools.pairwise(marks)
    ]


def compose_timepoint(
    t,
    layout,
    coords: Coordinates,
    volumes: dict,
    geometry: CanvasGeometry,
    region,
    boxes,
    dtype="uint16",
    buffer=None,
    weights_cache=None,
):
    """Blend one timepoint into an array covering `region`, in the canvas dtype.

    Buffers are sized to the written region, not to the canvas, so padding and
    empty space cost nothing -- not even the page faults that a whole-canvas
    mask scan used to force.

    Two things here look redundant and are not; both were measured:

    * The tile is cast to float32 in the reuse buffer and scaled in place
      rather than `multiply(uint16_3d, float32_2d)` in one call. numpy's
      mixed-dtype broadcast loop runs at roughly an eighth of the bandwidth of
      the same-dtype one, so two fast passes beat one slow pass by ~2x.
    * The normalise divides straight into the output dtype. Dividing and then
      converting is two full passes over the region plus a second region-sized
      allocation; fused it is one pass, measured ~3.8x faster.
    """
    origin = np.array([r[0] for r in region])
    shape = tuple(int(r[1] - r[0]) for r in region)
    accum = np.zeros(shape, dtype=np.float32)
    tile_shape = (layout.nz, layout.ny, layout.nx)
    pairs = layout.pairs_at(t)
    corners = layout.corners_at(t)
    if buffer is None:
        buffer = np.empty(tile_shape, dtype=np.float32)

    placed = {}
    weights_by_tile = {}
    for name, frame in volumes.items():
        if name not in boxes:
            continue
        z0, y0, x0 = (int(v) for v in np.array(boxes[name][0]) - origin)
        placed[name] = (z0, y0, x0)
        weights = blend_weights(
            name, t, coords, pairs, corners, tile_shape, weights_cache
        )
        weights_by_tile[name] = weights
        np.copyto(buffer, frame, casting="unsafe")
        np.multiply(buffer, weights, out=buffer)
        accum[z0 : z0 + layout.nz, y0 : y0 + layout.ny, x0 : x0 + layout.nx] += buffer

    image = np.empty(shape, dtype=dtype)
    plane = np.empty(shape[1:], dtype=np.float32)
    local_boxes = {
        name: ((z, y, x), (z + layout.nz, y + layout.ny, x + layout.nx))
        for name, (z, y, x) in placed.items()
    }
    for za, zb, covering in z_segments(local_boxes, shape[0]):
        # Seed with the floor rather than zeroing and clamping afterwards. Any
        # weight a tile contributes is at least 0.01, whose float32 ulp is ~1e-9,
        # so EPS is far below the rounding of every covered voxel and adds
        # nothing to them. Uncovered voxels keep EPS, and their accum is exactly
        # zero, so the divide yields zero rather than 0/0. One pass over the
        # plane instead of three.
        plane.fill(EPS)
        for name in covering:
            _, y0, x0 = placed[name]
            plane[y0 : y0 + layout.ny, x0 : x0 + layout.nx] += weights_by_tile[name]
        np.divide(accum[za:zb], plane, out=image[za:zb], casting="unsafe")
    return image


def bbox_of(inds) -> list[list[int]] | None:
    """Tight bounding box of the covered region, or None if nothing is covered."""
    if not inds.any():
        return None
    out = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        hit = np.where(inds.any(axis=others))[0]
        out.append([int(hit[0]), int(hit[-1]) + 1])
    return out


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [[min(a[i][0], b[i][0]), max(a[i][1], b[i][1])] for i in range(3)]


def snap_to_chunks(region, chunk_zyx, shape_zyx):
    """Grow a bbox out to chunk boundaries.

    Writing a slice that starts mid-chunk forces zarr to read-modify-write every
    partial chunk, and neighbouring writes then hit the same chunk repeatedly.
    Measured at ~3x slower than aligned writes even without compression.
    """
    if region is None:
        return None
    out = []
    for axis in range(3):
        lo, hi = region[axis]
        step = chunk_zyx[axis]
        out.append([(lo // step) * step, min(-(-hi // step) * step, shape_zyx[axis])])
    return out


def overlaps(sl, bbox) -> bool:
    if bbox is None:
        return False
    return all(sl[a].start < bbox[a][1] and bbox[a][0] < sl[a].stop for a in range(3))


# --- the resume log -----------------------------------------------------------
class BlendLog:
    """Which timepoints are on the canvas, under which placement.

    The key covers the canvas origin and the tile positions at that timepoint --
    NOT the global extent. A change elsewhere in the run that only widens the
    extent must not invalidate timepoints whose tiles have not moved.
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        self._done: dict[int, str] = {}
        self._bbox: dict[int, list] = {}
        self.skipped = 0
        if self.path is not None and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._done[rec["t"]] = rec["key"]
                    self._bbox[rec["t"]] = rec.get("bbox")
                except _PARSE_ERRORS:
                    # A killed process can leave a half-written final line.
                    # Dropping it is right; doing so silently is not -- a run
                    # that quietly re-blends work would look like a hang.
                    self.skipped += 1
            if self.skipped:
                warnings.warn(
                    f"{self.path}: skipped {self.skipped} unreadable line(s); "
                    "those timepoints will be blended again",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def key(self, t: int, coords: Coordinates, geometry: CanvasGeometry) -> str:
        payload = {
            "origin": list(geometry.origin),
            "dtype": geometry.dtype,
            "coords": sorted((n, [int(v) for v in c]) for n, c in coords.at(t).items()),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def is_done(self, t: int, key: str) -> bool:
        return self._done.get(t) == key

    def written(self, t: int) -> bool:
        return t in self._done

    def bbox(self, t: int):
        return self._bbox.get(t)

    def mark(self, t: int, key: str, bbox) -> None:
        self._done[t] = key
        self._bbox[t] = bbox
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"t": t, "key": key, "bbox": bbox}) + "\n")
            f.flush()


def write_with_retry(fn, attempts: int = 4, delay: float = 2.0):
    """Network mounts drop writes. Retry before giving up on a timepoint."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            last = e
            if i + 1 < attempts:
                time.sleep(delay * (i + 1))
    raise last


def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MB. 0.0 where unavailable."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return round(peak / (2**20 if peak > 2**32 else 2**10), 1)


@attrs.define
class Timings:
    """Where the wall clock went. Read, compose and write have opposite fixes."""

    read: float = 0.0
    compose: float = 0.0
    write: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        """Phase times plus the wall clock.

        When pipelining, the phases run concurrently, so they sum to more than
        the total; the excess is reported as `overlap_s` and is exactly what the
        pipeline saved. `other_s` appears instead when the phases sum to less
        than the wall clock -- that is time going somewhere none of them cover.
        """
        slack = self.total - self.read - self.compose - self.write
        out = {
            "read_s": round(self.read, 1),
            "compose_s": round(self.compose, 1),
            "write_s": round(self.write, 1),
        }
        if slack < 0:
            out["overlap_s"] = round(-slack, 1)
        else:
            out["other_s"] = round(slack, 1)
        out["total_s"] = round(self.total, 1)
        out["peak_mb"] = peak_rss_mb()
        return out


def _write_region(canvas, t, image, region, boxes, stale, chunk_zyx, pool):
    """Push one timepoint's chunks. Distinct zarr chunks are independent files,
    so they can go out concurrently -- on a network share the round trips
    dominate and serialising them wastes the link."""
    origin = np.array([r[0] for r in region])
    (z0, z1), (y0, y1), (x0, x1) = region
    slices = []
    for z in range(z0, z1, chunk_zyx[0]):
        for y in range(y0, y1, chunk_zyx[1]):
            for x in range(x0, x1, chunk_zyx[2]):
                sl = (
                    slice(z, min(z + chunk_zyx[0], z1)),
                    slice(y, min(y + chunk_zyx[1], y1)),
                    slice(x, min(x + chunk_zyx[2], x1)),
                )
                # Skip chunks no tile touches, unless a previous pass wrote
                # there and they now need clearing.
                if hits_a_box(sl, boxes) or overlaps(sl, stale):
                    slices.append(sl)

    def put_one(sl):
        local = tuple(
            slice(sl[a].start - origin[a], sl[a].stop - origin[a]) for a in range(3)
        )
        canvas[(t, *sl)] = image[local]

    if pool is None or len(slices) < 2:
        for sl in slices:
            put_one(sl)
    else:
        list(pool.map(put_one, slices))


def blend(
    layout,
    coords: Coordinates,
    reader,
    refs,
    output,
    log: BlendLog,
    geometry: CanvasGeometry,
    t0: int = 0,
    t1: int | None = None,
    chunk=(1, 32, 512, 512),
    force: bool = False,
    progress=None,
    attempts: int = 4,
    writers: int = 4,
    pipeline: bool = True,
    timings: Timings | None = None,
) -> int:
    """Write timepoints [t0, t1) into an existing geometry. Returns how many.

    Read, compose and write are pipelined: the next timepoint's tiles are being
    fetched while this one composes, and this one is being written while the
    next composes. Serialised, the disk idles through every compose and the CPU
    idles through every write.

    Memory, roughly, at steady state:

        2 x (tiles of one timepoint)     one batch in flight, one being used
        1 x (region as float32) x 2      accum + counts, while composing
        2 x (region as canvas dtype)     one written, one just composed

    The region grows with the mosaic, so a run whose object spreads over time
    uses more memory late than early -- that is the working set, not a leak.
    It is bounded by the canvas. `pipeline=False` removes roughly half of it.

    `progress` is a factory taking the total number of tiles and returning an
    object with `update(n)` and `close()` -- a tqdm, in practice. Note this is
    a different contract from run_plan's, which wraps an iterable.
    """
    import zarr

    t1 = layout.nt if t1 is None else t1
    canvas = zarr.open(
        str(output), mode="a", shape=geometry.shape, chunks=chunk, dtype=geometry.dtype
    )
    if tuple(canvas.shape) != tuple(geometry.shape):
        raise CanvasMismatch(
            f"canvas at {output} has shape {tuple(canvas.shape)} but its geometry "
            f"says {tuple(geometry.shape)}. Use --recreate or a new --output."
        )
    geometry.save(output)

    todo = [
        t
        for t in range(t0, t1)
        if force or not log.is_done(t, log.key(t, coords, geometry))
    ]
    if not todo:
        return 0
    spatial = geometry.spatial
    chunk_zyx = chunk[1:]
    tile_shape = (layout.nz, layout.ny, layout.nx)
    buffer = np.empty(tile_shape, dtype=np.float32)
    weights_cache: dict = {}
    timings = Timings() if timings is None else timings
    started = time.perf_counter()

    plans = {}
    for t in todo:
        boxes = tile_boxes(t, layout, coords, geometry)
        stale = log.bbox(t) if log.written(t) else None
        region = snap_to_chunks(
            union_bbox(boxes_bbox(boxes), stale), chunk_zyx, spatial
        )
        plans[t] = (boxes, stale, region)

    # Advance the bar by tiles, not timepoints. Early timepoints hold one or
    # two tiles and late ones the whole mosaic, so a per-timepoint bar
    # under-predicts at the start and over-predicts at the end. Every tile
    # count is known from `plans` before any work happens.
    total_tiles = sum(len(plans[t][0]) for t in todo)
    bar = progress(total_tiles) if progress is not None else None

    def load(t):
        clock = time.perf_counter()
        boxes = plans[t][0]
        out = load_timepoint(reader, refs, t, list(boxes))
        timings.read += time.perf_counter() - clock
        return out

    def store(t, image):
        clock = time.perf_counter()
        boxes, stale, region = plans[t]
        if region is not None:
            write_with_retry(
                lambda: _write_region(
                    canvas,
                    t,
                    image,
                    region,
                    boxes,
                    stale,
                    chunk_zyx,
                    write_pool,
                ),
                attempts=attempts,
            )
        log.mark(t, log.key(t, coords, geometry), boxes_bbox(boxes))
        timings.write += time.perf_counter() - clock

    written = 0
    stack = ExitStack()
    with stack:
        # Three separate pools on purpose. `store` runs on store_pool and then
        # fans its chunks out over write_pool; sharing one pool would have a
        # task waiting on tasks queued behind itself.
        write_pool = (
            stack.enter_context(ThreadPoolExecutor(max_workers=writers))
            if writers > 1
            else None
        )
        loader = (
            stack.enter_context(ThreadPoolExecutor(max_workers=1)) if pipeline else None
        )
        store_pool = (
            stack.enter_context(ThreadPoolExecutor(max_workers=1)) if pipeline else None
        )

        ahead = loader.submit(load, todo[0]) if loader else None
        pending = None

        # `it` is iterated lazily so a tqdm bar advances as work completes
        for i, t in enumerate(todo):
            volumes = ahead.result() if ahead is not None else load(t)
            if loader is not None and i + 1 < len(todo):
                ahead = loader.submit(load, todo[i + 1])
            elif loader is not None:
                ahead = None

            clock = time.perf_counter()
            boxes, _, region = plans[t]
            if region is None:
                image = None
            else:
                image = compose_timepoint(
                    t,
                    layout,
                    coords,
                    volumes,
                    geometry,
                    region,
                    boxes,
                    dtype=geometry.dtype,
                    buffer=buffer,
                    weights_cache=weights_cache,
                )
            del volumes
            timings.compose += time.perf_counter() - clock

            if pending is not None:
                pending.result()
            if image is None:
                store(t, None)
                pending = None
            elif store_pool is not None:
                pending = store_pool.submit(store, t, image)
            else:
                store(t, image)
                pending = None
            written += 1
            if bar is not None:
                bar.update(len(plans[t][0]))

        if pending is not None:
            pending.result()

    if bar is not None:
        bar.close()
    timings.total = time.perf_counter() - started
    return written

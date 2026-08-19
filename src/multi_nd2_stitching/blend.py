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
import json
import time
from pathlib import Path

import attrs
import numpy as np

from .coordinates import Coordinates


class CanvasMismatch(Exception):
    pass


@attrs.frozen
class CanvasGeometry:
    """Where world coordinates land in the canvas. Immutable once written."""

    origin: tuple[int, int, int]  # world zyx of canvas index (0, 0, 0)
    shape: tuple[int, int, int, int]  # (t, z, y, x)
    dtype: str

    @classmethod
    def required(cls, coords: Coordinates, tile_shape, nt: int, dtype: str):
        ext = coords.extent(tile_shape)
        origin = tuple(int(v) for v in ext[:, 0])
        size = tuple(int(v) for v in (ext[:, 1] - ext[:, 0]))
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

    def slack(self, coords: Coordinates, tile_shape, nt: int) -> tuple[int, ...]:
        """How much bigger this canvas is than the coordinates now need."""
        need = CanvasGeometry.required(coords, tile_shape, nt, self.dtype)
        return tuple(int(a - b) for a, b in zip(self.spatial, need.spatial))

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
        p = cls.path_for(output)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            return cls(
                origin=tuple(d["origin"]), shape=tuple(d["shape"]), dtype=d["dtype"]
            )
        except Exception:
            return None


def resolve_geometry(output, coords, tile_shape, nt, dtype, t0, t1, recreate=False):
    """Reuse the canvas's own geometry if it has one; otherwise derive a new one."""
    existing = None if recreate else CanvasGeometry.load(output)
    if existing is None:
        return CanvasGeometry.required(coords, tile_shape, nt, dtype), True
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


def blend_weights(name: str, t: int, coords: Coordinates, pairs, tile_shape):
    """Linear ramps across every edge that has a neighbour.

    Only the relative shape matters: the caller divides by the accumulated
    weight, so a solo region comes out unchanged whatever its weight was.
    """
    here = coords.at(t)
    w = [
        np.ones(tile_shape[1], dtype=np.float32),
        np.ones(tile_shape[2], dtype=np.float32),
    ]

    for p in pairs:
        if name not in (p.a, p.b) or p.a not in here or p.b not in here:
            continue
        length = int((here[p.b] - here[p.a])[p.axis])
        if not 0 < length < tile_shape[p.axis]:
            continue  # degenerate placement; leave this edge flat
        ramp = np.linspace(0.01, 0.99, length, dtype=np.float32)
        if p.a == name:
            w[p.axis - 1][-length:] = ramp[::-1]
        else:
            w[p.axis - 1][:length] = ramp
    return w[0][:, np.newaxis] * w[1][np.newaxis, :]


def compose_timepoint(t, layout, coords, reader, geometry: CanvasGeometry, refs):
    """Blend one timepoint into a float32 array. Returns (image, mask)."""
    out_shape = geometry.spatial
    accum = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(out_shape, dtype=np.float32)
    inds = np.zeros(out_shape, dtype=bool)
    tile_shape = (layout.nz, layout.ny, layout.nx)
    pairs = layout.pairs_at(t)

    for name in layout.tiles_at(t):
        if name not in coords.at(t):
            continue
        frame = reader.read(refs[(t, name)])
        z0, y0, x0 = geometry.offset_of(coords[t, name])
        sls = (
            slice(z0, z0 + layout.nz),
            slice(y0, y0 + layout.ny),
            slice(x0, x0 + layout.nx),
        )
        weights = blend_weights(name, t, coords, pairs, tile_shape)
        accum[sls] += frame * weights
        counts[sls] += weights
        inds[sls] = True

    np.divide(accum, counts, out=accum, where=inds)
    return accum, inds


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
        if self.path is not None and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._done[rec["t"]] = rec["key"]
                    self._bbox[rec["t"]] = rec.get("bbox")
                except Exception:
                    pass

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
) -> int:
    """Write timepoints [t0, t1) into an existing geometry. Returns how many."""
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
    it = progress(todo) if progress is not None else todo

    written = 0
    for t in it:
        image, inds = compose_timepoint(t, layout, coords, reader, geometry, refs)
        block = image.astype(geometry.dtype)
        new_bbox = bbox_of(inds)
        # Rewriting: also cover whatever the previous pass wrote, so stale pixels
        # outside the new footprint are erased rather than left behind.
        region = union_bbox(new_bbox, log.bbox(t) if log.written(t) else None)

        def put(t=t, block=block, inds=inds, region=region):
            if region is None:
                return
            (z0, z1), (y0, y1), (x0, x1) = region
            for z in range(z0, z1, chunk[1]):
                for y in range(y0, y1, chunk[2]):
                    for x in range(x0, x1, chunk[3]):
                        sl = (
                            slice(z, min(z + chunk[1], z1)),
                            slice(y, min(y + chunk[2], y1)),
                            slice(x, min(x + chunk[3], x1)),
                        )
                        canvas[(t, *sl)] = block[sl]

        write_with_retry(put, attempts=attempts)
        log.mark(t, log.key(t, coords, geometry), new_bbox)
        written += 1
    return written

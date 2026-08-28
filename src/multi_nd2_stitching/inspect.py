"""Exports for looking at a single neighbour pair by eye.

Writes plain zarr arrays that napari opens directly. Nothing here feeds back
into the pipeline -- it exists so a bad offset can be seen rather than guessed
at.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .compute import (
    SHAPED_PEAK_CANDIDATES,
    axis_profile,
    candidate_peaks,
    correlation_surface,
    spectrum,
    to_signed_shift,
    trim_for,
)

_AXIS_NAME = ("z", "y", "x")
_PROFILE_RADIUS = 8


def _place(a, b, shift_zyx, tile_shape):
    """Two tiles on a shared canvas, b displaced from a by shift_zyx."""
    shift = np.asarray(shift_zyx, dtype=int)
    lo = np.minimum(np.zeros(3, dtype=int), shift)
    hi = np.maximum(np.array(tile_shape), shift + np.array(tile_shape))
    size = tuple(int(v) for v in (hi - lo))

    out = np.zeros((2, *size), dtype=np.float32)
    for layer, (img, at) in enumerate(((a, np.zeros(3, dtype=int)), (b, shift))):
        z, y, x = at - lo
        out[
            layer, z : z + tile_shape[0], y : y + tile_shape[1], x : x + tile_shape[2]
        ] = img
    return out


def inspect_pair(
    task, offset, reader, out_dir, *, response: bool = True, nominal: bool = True
) -> Path:
    """Export one pair task for visual inspection.

    Arrays written (all zyx, layer 0 = `a`, layer 1 = `b`):
      measured  both tiles placed using the offset that was computed
      nominal   both tiles placed at the bare grid spacing, no correction
      overlap   only the two strips the correlation actually saw
      response  the phase-correlation surface, peak-centred

    In napari the useful move is to load `measured` and flip layer 1 on and off:
    a good offset makes the seam disappear. `response` tells you whether the
    correlation was confident -- one sharp peak, or a smear with rivals.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import zarr

    sl = task.crop.as_slices()
    vol_a = reader.read(task.src)[sl]
    vol_b = reader.read(task.dst)[sl]
    tile_shape = vol_a.shape
    measured = np.array([offset.dz, offset.dy, offset.dx], dtype=int)

    def save(name, arr, **attrs):
        z = zarr.open(
            str(out_dir / f"{name}.zarr"),
            mode="w",
            shape=arr.shape,
            chunks=tuple(min(c, 64) for c in arr.shape),
            dtype=arr.dtype,
        )
        z[:] = arr
        for k, v in attrs.items():
            z.attrs[k] = v

    save(
        "measured",
        _place(vol_a, vol_b, measured, tile_shape),
        offset=[int(v) for v in measured],
        a=task.a,
        b=task.b,
        axis=task.axis,
        t=task.t,
    )

    if nominal:
        flat = np.zeros(3, dtype=int)
        flat[task.axis] = task.shift_px
        save(
            "nominal",
            _place(vol_a, vol_b, flat, tile_shape),
            offset=[int(v) for v in flat],
            correction=[int(v) for v in (measured - flat)],
        )

    ref_a, ref_b = task.spectrum_refs()
    strip_a = trim_for(vol_a, ref_a)
    strip_b = trim_for(vol_b, ref_b)
    save(
        "overlap",
        np.stack([strip_a, strip_b]).astype(np.float32),
        note="the two strips the correlation compared, before alignment",
    )

    if response:
        f0 = spectrum(strip_a, precision=task.precision)
        f1 = spectrum(strip_b, precision=task.precision)
        surface = correlation_surface(f0, f1, strip_a.shape)
        save(
            "response",
            np.fft.fftshift(surface).astype(np.float32),
            note="peak-centred: a shift of zero sits at the array centre",
            centre=[int(v) // 2 for v in strip_a.shape],
        )

        nominal_shift = np.zeros(3, dtype=int)
        nominal_shift[task.axis] = task.shift_px
        _write_candidate_csvs(out_dir, surface, strip_a.shape, nominal_shift, measured)

    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "pair": [task.a, task.b],
                "axis": task.axis,
                "t": task.t,
                "shift_px": task.shift_px,
                "crop": {
                    "z": list(task.crop.z),
                    "y": list(task.crop.y),
                    "x": list(task.crop.x),
                },
                "measured_offset": [int(v) for v in measured],
                "shaped_peak": task.shaped_peak,
                "tile_shape": [int(v) for v in tile_shape],
                "strip_shape": [int(v) for v in strip_a.shape],
            },
            indent=1,
        )
    )
    return out_dir


def _write_candidate_csvs(out_dir, surface, shape, nominal_shift, measured) -> None:
    """candidates.csv + profiles.csv: what the response actually offered, not
    just what got picked.

    `candidates.csv` lists the same `SHAPED_PEAK_CANDIDATES` points the real
    `shaped_peak` override would consider -- so this is exactly what that
    override sees, not a separately-tuned debug view. `profiles.csv` gives
    each candidate's drop-off curve along every axis, to plot "does this
    actually look like a peak" without needing a plot exported for you.
    """
    cands = candidate_peaks(surface, candidates=SHAPED_PEAK_CANDIDATES)
    shifts = [np.array(to_signed_shift(c.index, shape)) + nominal_shift for c in cands]

    with (out_dir / "candidates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "dz", "dy", "dx", "value", "decay", "taken"])
        for rank, (c, shift) in enumerate(zip(cands, shifts, strict=True), start=1):
            taken = int(np.array_equal(shift, measured))
            w.writerow([rank, *(int(v) for v in shift), c.value, c.decay, taken])

    with (out_dir / "profiles.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "axis", "step", "value"])
        steps = range(-_PROFILE_RADIUS, _PROFILE_RADIUS + 1)
        for rank, c in enumerate(cands, start=1):
            for axis, name in enumerate(_AXIS_NAME):
                profile = axis_profile(surface, c.index, axis, radius=_PROFILE_RADIUS)
                for step, value in zip(steps, profile, strict=True):
                    w.writerow([rank, name, step, float(value)])


def _project(vol, axis: int):
    return vol.max(axis=axis)


def inspect_drift(
    name,
    tasks,
    store,
    reader,
    out_dir,
    *,
    size: int | None = 256,
    response: bool = True,
    full: bool = False,
    progress=None,
) -> Path:
    """Export one tile's drift over time.

    `tasks` are that tile's TimeTasks in ascending order. Arrays written:

      aligned_xy  (T, y, x)  drift-corrected, z-projected -- the sample should
                             sit still while you scrub time. Anything that
                             lurches is a bad step.
      raw_xy      (T, y, x)  the same frames uncorrected, for comparison
      aligned_zx  (T, z, x)  y-projected, so z drift is visible too
      response    (T-1, y, x) each step's correlation surface, peak-centred:
                             the centre is "no shift". A step that jumps shows
                             its peak far out, or shows rivals.
      offsets.csv            per-step dz/dy/dx and the running total

    A full (T, z, y, x) stack is written instead of the projections with
    `full=True`; at real tile sizes that is tens of GB, hence the default.
    """
    import zarr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tasks:
        raise ValueError(f"no drift steps for '{name}'")

    steps = [(t, store.get(t.key)) for t in tasks]
    missing = [t.describe() for t, o in steps if o is None]
    if missing:
        raise ValueError(f"{len(missing)} drift offset(s) not computed: {missing[:3]}")

    # running position, and the canvas it needs
    times = [tasks[0].t_from] + [t.t_to for t in tasks]
    cum = [np.zeros(3, dtype=int)]
    for _, off in steps:
        cum.append(cum[-1] + np.array([off.dz, off.dy, off.dx], dtype=int))
    cum = np.array(cum)
    lo, hi = cum.min(axis=0), cum.max(axis=0)

    def crop_middle(vol):
        if size is None:
            return vol
        out = vol
        for axis in (1, 2):
            n = out.shape[axis]
            if n > size:
                start = (n - size) // 2
                sl = [slice(None)] * 3
                sl[axis] = slice(start, start + size)
                out = out[tuple(sl)]
        return out

    probe = crop_middle(reader.read(tasks[0].src)[tasks[0].crop.as_slices()])
    tile = np.array(probe.shape)
    span = tuple(int(v) for v in (hi - lo + tile))
    nt = len(times)

    arrays = {}

    def make(key, shape, dtype="float32"):
        shape = tuple(int(v) for v in shape)
        a = zarr.open(
            str(out_dir / f"{key}.zarr"),
            mode="w",
            shape=shape,
            chunks=tuple(min(c, 128) for c in shape),
            dtype=dtype,
        )
        arrays[key] = a
        return a

    if full:
        make("aligned", (nt, *span), probe.dtype.name)
    else:
        make("aligned_xy", (nt, span[1], span[2]), probe.dtype.name)
        make("aligned_zx", (nt, span[0], span[2]), probe.dtype.name)
    make("raw_xy", (nt, tile[1], tile[2]), probe.dtype.name)
    if response:
        make("response", (nt - 1, tile[1], tile[2]))

    refs = [tasks[0].src] + [t.dst for t in tasks]
    it = progress(range(nt)) if progress is not None else range(nt)
    prev_strip = None
    for k in it:
        task = tasks[min(k, len(tasks) - 1)]
        vol = crop_middle(reader.read(refs[k])[task.crop.as_slices()])
        z, y, x = cum[k] - lo
        if full:
            canvas = np.zeros(span, dtype=vol.dtype)
            canvas[z : z + tile[0], y : y + tile[1], x : x + tile[2]] = vol
            arrays["aligned"][k] = canvas
        else:
            flat = np.zeros((span[1], span[2]), dtype=vol.dtype)
            flat[y : y + tile[1], x : x + tile[2]] = _project(vol, 0)
            arrays["aligned_xy"][k] = flat
            side = np.zeros((span[0], span[2]), dtype=vol.dtype)
            side[z : z + tile[0], x : x + tile[2]] = _project(vol, 1)
            arrays["aligned_zx"][k] = side
        arrays["raw_xy"][k] = _project(vol, 0)

        if response:
            if prev_strip is not None:
                f0 = spectrum(prev_strip, precision=task.precision)
                f1 = spectrum(vol, precision=task.precision)
                surf = correlation_surface(f0, f1, vol.shape)
                arrays["response"][k - 1] = np.fft.fftshift(_project(surf, 0)).astype(
                    np.float32
                )
            prev_strip = vol

    rows = ["t,dz,dy,dx,cum_z,cum_y,cum_x,magnitude,realign,shaped_peak"]
    for i, (task, off) in enumerate(steps):
        mag = float(np.linalg.norm([off.dz, off.dy, off.dx]))
        rows.append(
            f"{task.t_to},{off.dz},{off.dy},{off.dx},"
            f"{cum[i + 1][0]},{cum[i + 1][1]},{cum[i + 1][2]},"
            f"{mag:.2f},{int(task.realign)},{int(task.shaped_peak)}"
        )
    (out_dir / "offsets.csv").write_text("\n".join(rows) + "\n")

    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "tile": name,
                "steps": len(steps),
                "timepoints": [int(times[0]), int(times[-1])],
                "crop_size": size,
                "tile_shape": [int(v) for v in tile],
                "canvas": [int(v) for v in span],
                "total_drift": [int(v) for v in cum[-1]],
                "largest_step": max(
                    (float(np.linalg.norm([o.dz, o.dy, o.dx])) for _, o in steps),
                    default=0.0,
                ),
            },
            indent=1,
        )
    )
    return out_dir

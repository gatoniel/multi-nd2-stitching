"""A PNG of a wide overview image with the fine tile positions marked on it.

Two tiers, same split as everywhere else in this codebase: the pixel-math
(`marker_positions`, `to_pixel`) is pure and unit-tested against fake
`FileMeta`s; reading the actual overview plane and drawing on it needs nd2 and
Pillow and is exercised manually against a real overview.nd2, the same way
`metadata.read_metadata` itself is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import StitchingConfig
from .metadata import FileMeta, Metadata


def tile_positions_um(cfg: StitchingConfig, meta: Metadata) -> dict[str, tuple]:
    """Each tile's stage position (x, y) um, from the file it first appears in."""
    out = {}
    for name, pos in cfg.positions.items():
        file_i = pos.start[0]
        fm = meta[file_i]
        idx = fm.position_of((name, *pos.aliases))
        if idx is None:
            raise ValueError(f"'{name}' has no position in file {file_i}")
        out[name] = fm.stage_um[idx]
    return out


def to_pixel(
    stage_um: tuple,
    overview_um: tuple,
    voxel_x_um: float,
    image_shape: tuple,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple:
    """A tile's (x, y) stage position -> (row, col) pixel in the overview image.

    Stage y increases "up"; image rows increase downward, hence the sign flip
    on the row axis. `flip_x`/`flip_y` reuse the same config flags that
    already resolve this mismatch for the fine stitching -- unverified for the
    overview image specifically until checked against a real file.

    `voxel_x_um <= 0` (an uncalibrated or broken overview file) would divide
    by zero -- unlike numpy, plain Python floats *raise* on that rather than
    returning inf/nan, which would otherwise surface as an opaque
    `ZeroDivisionError` deep inside marker placement. Returning nan instead
    lets the caller detect and skip it the same way as any other non-finite
    result.
    """
    if voxel_x_um <= 0:
        return (float("nan"), float("nan"))
    dx = (stage_um[0] - overview_um[0]) / voxel_x_um
    dy = (stage_um[1] - overview_um[1]) / voxel_x_um
    if flip_x:
        dx = -dx
    if flip_y:
        dy = -dy
    h, w = image_shape
    return h / 2 - dy, w / 2 + dx


def marker_positions(
    cfg: StitchingConfig, meta: Metadata, overview_meta: FileMeta, channel: str
) -> dict[str, tuple]:
    """tile name -> (row, col) pixel position in the overview image."""
    idx = overview_meta.position_of((channel,))
    if idx is None:
        raise ValueError(
            f"'{channel}' is not a position in {overview_meta.path}; "
            f"known: {overview_meta.position_names}"
        )
    overview_um = overview_meta.stage_um[idx]
    shape = (overview_meta.ny, overview_meta.nx)
    tiles = tile_positions_um(cfg, meta)
    return {
        name: to_pixel(
            um, overview_um, overview_meta.voxel_x_um, shape, cfg.flip_x, cfg.flip_y
        )
        for name, um in tiles.items()
    }


def downsample_plane(arr: np.ndarray, max_dim: int = 2000) -> tuple[np.ndarray, float]:
    """Shrink a plane so its longer side is at most `max_dim`.

    An overview scan can be tens of thousands of pixels per side, and this is
    meant to become a small PNG (a few MB), not a full-resolution copy of the
    sensor.

    Returns the shrunk array and the scale factor applied, so pixel positions
    computed against the full-resolution stage geometry (see `to_pixel`) can
    be scaled down to match.

    History, because the obvious ways to write this both crash: a numpy
    stride (`arr[::step, ::step]`) forced contiguous with
    `np.ascontiguousarray` segfaulted inside numpy's `_multiarray_umath`; a
    Pillow `Image.resize()` on the same array segfaulted inside Pillow's own
    `_imaging` instead. Two independent C libraries, same failure mode, both
    confirmed via the kernel's crash log (not a Python exception -- nothing in
    this codebase could have caught either one) -- on a real
    ~38000x14000 uint16 plane, numpy 2.5.2 / Pillow / CPython 3.14.7, all very
    new, on WSL2. The common thread isn't which library, it's that both read
    the *entire* multi-hundred-MB buffer in one bulk vectorized/SIMD sweep;
    `nd2` itself reads the same plane one row at a time and does not crash.
    So this does too: never touch more than one source row in any single
    numpy call, at the cost of a Python-level loop instead of one fused
    vectorized op.
    """
    longest = max(arr.shape)
    if longest <= max_dim:
        return arr, 1.0
    step = -(-longest // max_dim)  # ceil division
    scale = 1.0 / step
    rows = [np.array(arr[r, ::step]) for r in range(0, arr.shape[0], step)]
    return np.stack(rows, axis=0), scale


def normalize_to_uint8(arr: np.ndarray, lo_pct: float = 1, hi_pct: float = 99):
    """Percentile-stretch to the 0..255 range a PNG can hold."""
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((arr.astype(np.float64) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


# --- the nd2/Pillow-touching half -----------------------------------------------


def _find_seq_index(loop_indices, **want) -> int | None:
    """First frame whose loop indices match `want`; axes not in `want` are free."""
    for i, idx in enumerate(loop_indices):
        if all(idx.get(axis, 0) == value for axis, value in want.items()):
            return i
    return None


def _first_plane(frame: np.ndarray, components_per_channel: int) -> np.ndarray:
    """Reduce an `ND2File.read_frame()` result to a single 2D plane.

    `read_frame` reshapes to (channelCount, height, width, componentsPerChannel)
    and then squeezes away whatever axes are size 1 -- so an extra axis can be
    *leading* (several fluorescence channels bundled in one frame) or
    *trailing* (RGB/RGBA components, common for an overview taken on a colour
    camera). Those need opposite handling: stripping a trailing colour axis
    with the same "take index 0 while ndim > 2" logic used for a leading
    channel axis would grab the frame's first *row* instead, silently
    producing a garbled sliver instead of a plane -- so the colour axis is
    collapsed first, explicitly, using the file's own component count rather
    than guessing from shape alone.
    """
    if (
        components_per_channel > 1
        and frame.ndim >= 2
        and frame.shape[-1] == components_per_channel
    ):
        frame = frame.mean(axis=-1)
    while frame.ndim > 2:
        frame = frame[0]
    return frame


def read_overview_plane(
    path, position_index: int, z: int = 0, t: int = 0
) -> np.ndarray:
    """One 2D plane for the given position: channel 0, a specific z and t.

    Overview scans are usually a single plane with no Z or T loop at all, in
    which case `sizes` (and `loop_indices`) simply omit that axis and `z`/`t`
    are irrelevant -- `_find_seq_index` treats a missing axis as always at 0,
    so the defaults just work. They only matter once the file genuinely has
    more than one z-slice or timepoint.
    """
    import nd2

    with nd2.ND2File(str(path)) as f:
        seq = _find_seq_index(f.loop_indices, P=position_index, Z=z, T=t, C=0)
        if seq is None:
            raise ValueError(
                f"{path}: no frame found for position {position_index}, z={z}, t={t}"
            )
        frame = f.read_frame(seq)
        return _first_plane(frame, f.components_per_channel)


def render_overview(
    image_u8: np.ndarray,
    markers: dict[str, tuple],
    out_path: Path,
    label: bool = True,
    marker_px: int = 6,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.fromarray(image_u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    for name, (row, col) in markers.items():
        draw.line(
            (col - marker_px, row - marker_px, col + marker_px, row + marker_px),
            fill=(255, 0, 0),
            width=2,
        )
        draw.line(
            (col - marker_px, row + marker_px, col + marker_px, row - marker_px),
            fill=(255, 0, 0),
            width=2,
        )
        if label:
            draw.text((col + marker_px, row - marker_px), name, fill=(255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    if not out_path.exists():
        # img.save() raises on a real failure; this is a last-resort check
        # against a silent one (a network mount that acknowledges a write it
        # hasn't actually persisted, for instance) -- if this ever fires, it
        # turns "nothing happened" into a traceback that says so.
        raise RuntimeError(
            f"Image.save() returned without error, but {out_path} does not "
            "exist afterwards -- check disk space, permissions, and whether "
            "this path is on a network mount that buffers writes"
        )

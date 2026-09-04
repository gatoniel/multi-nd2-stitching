"""A PNG of a wide overview image with the fine tile positions marked on it.

Two tiers, same split as everywhere else in this codebase: the pixel-math
(`marker_positions`, `to_pixel`, `block_reduce`, `_want_axes`) is pure and
unit-tested against fake `FileMeta`s and synthetic arrays; reading the actual
overview plane and drawing on it needs nd2 and Pillow and is exercised
manually against a real overview.nd2, the same way `metadata.read_metadata`
itself is.

The overview file is not a tile file: it may have no `P` axis (a single
scan), no `C` axis, and never has a `T` or a real z-stack worth aligning on --
just a representative mid-plane. `read_overview_meta`/`read_overview_plane`
handle that directly against `f.sizes`, rather than assuming the same shape
`metadata.read_metadata` requires of tile files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Overview, StitchingConfig
from .metadata import FileMeta, Metadata

# Peak extra memory `block_reduce` is allowed per chunk, not counting the
# input array itself (already in memory -- nd2 has no partial-frame read) or
# the small output. Bounds the downsample step independent of image size,
# which is the actual fix for the OS OOM-killing the process on a strided
# slice of a huge array under Python 3.14.7 -- there was no Traceback, only
# `dmesg`, because the kill happens below the interpreter.
BLOCK_REDUCE_BUDGET_BYTES = 256 * 1024 * 1024


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
    """
    dx = (stage_um[0] - overview_um[0]) / voxel_x_um
    dy = (stage_um[1] - overview_um[1]) / voxel_x_um
    if flip_x:
        dx = -dx
    if flip_y:
        dy = -dy
    h, w = image_shape
    return h / 2 - dy, w / 2 + dx


def marker_positions(
    cfg: StitchingConfig,
    meta: Metadata,
    overview_meta: FileMeta,
    channel: int | None,
    *,
    pixel_size_um: float | None = None,
    downsample: int = 1,
) -> dict[str, tuple]:
    """tile name -> (row, col) pixel position in the (possibly downsampled)
    overview image.

    `channel` is a direct index into `overview_meta.stage_um` -- `None` means
    "the file's one implicit position" (index 0), valid only when there is
    exactly one. `pixel_size_um` overrides a wrong ND2 header; `downsample`
    is the block-reduce factor the plane was (or will be) shrunk by, so
    markers land in the same pixel space as the shrunk image.
    """
    idx = 0 if channel is None else channel
    n = len(overview_meta.stage_um)
    if not 0 <= idx < n:
        raise ValueError(
            f"channel {channel} is out of range for {overview_meta.path} "
            f"({n} position(s))"
        )
    overview_um = overview_meta.stage_um[idx]
    voxel = overview_meta.voxel_x_um if pixel_size_um is None else pixel_size_um
    eff_voxel = voxel * downsample
    shape = (overview_meta.ny // downsample, overview_meta.nx // downsample)
    tiles = tile_positions_um(cfg, meta)
    return {
        name: to_pixel(um, overview_um, eff_voxel, shape, cfg.flip_x, cfg.flip_y)
        for name, um in tiles.items()
    }


def normalize_to_uint8(arr: np.ndarray, lo_pct: float = 1, hi_pct: float = 99):
    """Percentile-stretch to the 0..255 range a PNG can hold.

    Only ever called on the already-downsampled image (see `build_overview`)
    -- a percentile scan plus a float64 cast over the raw, full-size plane is
    exactly the whole-array pass that made the original crash worse.
    """
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((arr.astype(np.float64) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _want_axes(sizes, position_index: int | None) -> dict:
    """Which loop-index constraints actually make sense for this file.

    Only constrain an axis that exists. A bare `(Y, X)` overview file has no
    `P` (and often no `C`); asking `_find_seq_index` to match `P=0` against
    loop indices that never carry a `P` key at all previously "worked" only
    by accident (`idx.get("P", 0) == 0`), and was simply wrong for any other
    index.
    """
    want: dict[str, int] = {}
    if "P" in sizes and position_index is not None:
        want["P"] = position_index
    if "Z" in sizes:
        want["Z"] = sizes["Z"] // 2
    if "C" in sizes:
        want["C"] = 0
    return want


def block_reduce(
    arr: np.ndarray, factor: int, method: str = "mean", *, progress=None
) -> np.ndarray:
    """Shrink `arr` by `factor` per axis via block mean or median.

    Processed row-block by row-block, each block sized to stay under
    `BLOCK_REDUCE_BUDGET_BYTES` regardless of the input's total size -- never
    a whole-array reduction, and never a strided slice (`arr[::factor]`)
    anywhere. Both were implicated in the earlier crash; block-reducing is
    also just correct (no aliasing), unlike picking every Nth pixel.

    Trims to a whole multiple of `factor` per axis first (drop the remainder
    edge, don't pad) via an ordinary step-1 slice -- a view, not a copy.

    `progress`, if given, wraps the block iterable (same convention as
    `compute.run_plan`'s `progress=`) -- so a slow or hung run is visible,
    not silent.
    """
    if method not in ("mean", "median"):
        raise ValueError(f"unknown reduction method {method!r}")
    if factor <= 1:
        return arr.astype(np.float64, copy=False)

    ny, nx = arr.shape
    ny2, nx2 = (ny // factor) * factor, (nx // factor) * factor
    arr = arr[:ny2, :nx2]
    out_h, out_w = ny2 // factor, nx2 // factor
    out = np.empty((out_h, out_w), dtype=np.float64)

    itemsize = arr.dtype.itemsize
    rows_per_block = max(
        factor,
        (BLOCK_REDUCE_BUDGET_BYTES // (nx2 * itemsize) // factor) * factor,
    )
    reduce_fn = np.mean if method == "mean" else np.median

    starts = range(0, ny2, rows_per_block)
    if progress is not None:
        starts = progress(starts)
    for r0 in starts:
        r1 = min(r0 + rows_per_block, ny2)
        rows_out = (r1 - r0) // factor
        block = arr[r0:r1].reshape(rows_out, factor, out_w, factor)
        oy0 = r0 // factor
        out[oy0 : oy0 + rows_out] = reduce_fn(block, axis=(1, 3))
    return out


# --- the nd2/Pillow-touching half -----------------------------------------------


def _find_seq_index(loop_indices, **want) -> int | None:
    """First frame whose loop indices match `want`; axes not in `want` are free."""
    for i, idx in enumerate(loop_indices):
        if all(idx.get(axis, 0) == value for axis, value in want.items()):
            return i
    return None


def read_overview_meta(path) -> FileMeta:
    """Header facts for an overview file -- `P` and `C` axes are optional.

    When there is no `P` axis, `position_names` is empty and `stage_um` holds
    exactly the one implicit position, read from per-frame metadata (which
    exists regardless of whether a multipoint loop does) rather than
    `f.experiment[1].parameters.points` (what `metadata.read_metadata` uses,
    correctly, for tile files that always have one).
    """
    import nd2

    with nd2.ND2File(str(path)) as f:
        sizes = f.sizes
        voxel_x_um = f.voxel_size().x
        if "P" in sizes:
            points = f.experiment[1].parameters.points
            position_names = tuple(p.name for p in points)
            stage_um = tuple((p.stagePositionUm.x, p.stagePositionUm.y) for p in points)
        else:
            stage = f.frame_metadata(0).channels[0].position.stagePositionUm
            position_names = ()
            stage_um = ((stage.x, stage.y),)
        return FileMeta(
            path=str(path),
            nt=1,
            nz=sizes.get("Z", 1),
            ny=sizes["Y"],
            nx=sizes["X"],
            position_names=position_names,
            stage_um=stage_um,
            voxel_x_um=voxel_x_um,
        )


def read_overview_plane(path, position_index: int | None) -> np.ndarray:
    """One representative 2D plane: the given position, channel 0, middle z."""
    import nd2

    with nd2.ND2File(str(path)) as f:
        want = _want_axes(f.sizes, position_index)
        seq = _find_seq_index(f.loop_indices, **want)
        if seq is None:
            raise ValueError(f"{path}: no frame found for {want}")
        frame = f.read_frame(seq)
    # A bundled multichannel frame carries a leading channel axis; take the first.
    while frame.ndim > 2:
        frame = frame[0]
    return frame


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


def build_overview(
    cfg: StitchingConfig,
    meta: Metadata,
    ov: Overview,
    out_path: Path,
    *,
    progress=None,
) -> dict[str, tuple]:
    """Read, downsample, mark, and save one overview -- the full pipeline.

    Returns the markers actually drawn (tile name -> (row, col)), so the
    caller can report what landed without recomputing it. Kept out of
    `cli.py` so it stays unit-testable without going through argparse, per
    this module's own pixel-math/nd2-touching split.
    """
    overview_meta = read_overview_meta(ov.file)
    n = len(overview_meta.stage_um)
    if ov.channel is None and n > 1:
        raise ValueError(f"{ov.file} has {n} positions; overview.channel must pick one")
    if ov.channel is not None and not 0 <= ov.channel < n:
        raise ValueError(f"channel {ov.channel} is out of range for {ov.file} ({n})")
    plane = read_overview_plane(ov.file, ov.channel)
    ny, nx = plane.shape
    factor = max(1, -(-max(ny, nx) // ov.max_output_px))
    small = block_reduce(plane, factor, ov.reduction, progress=progress)
    image = normalize_to_uint8(small)
    markers = marker_positions(
        cfg,
        meta,
        overview_meta,
        ov.channel,
        pixel_size_um=ov.pixel_size_um,
        downsample=factor,
    )
    render_overview(image, markers, out_path, label=ov.label)
    return markers

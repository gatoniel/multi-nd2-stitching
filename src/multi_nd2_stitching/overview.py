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


def read_overview_plane(path, position_index: int) -> np.ndarray:
    """One representative 2D plane for the given position: channel 0, middle z."""
    import nd2

    with nd2.ND2File(str(path)) as f:
        mid_z = f.sizes.get("Z", 1) // 2
        seq = _find_seq_index(f.loop_indices, P=position_index, Z=mid_z, C=0)
        if seq is None:
            raise ValueError(
                f"{path}: no frame found for position {position_index}, z={mid_z}"
            )
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

import math

import numpy as np
import pytest
import yaml
from helpers import make_meta

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.overview import (
    _find_seq_index,
    _first_plane,
    downsample_plane,
    marker_positions,
    normalize_to_uint8,
    render_overview,
    tile_positions_um,
    to_pixel,
)


def _cfg(cfg_dict, **overrides):
    d = dict(cfg_dict, **overrides)
    return loads_config(yaml.safe_dump(d))


# --- tile_positions_um ----------------------------------------------------------
def test_tile_positions_um_reads_the_start_file(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"), spacing=55.0, axis="x")
    positions = tile_positions_um(cfg, meta)
    assert positions == {"tile_a": (0.0, 0.0), "tile_b": (55.0, 0.0)}


def test_tile_positions_um_follows_aliases(cfg_dict):
    cfg_dict["positions"]["tile_b"]["aliases"] = ["renamed_b"]
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "renamed_b"), spacing=55.0, axis="x")
    positions = tile_positions_um(cfg, meta)
    assert positions["tile_b"] == (55.0, 0.0)


# --- to_pixel ---------------------------------------------------------------------
def test_to_pixel_centers_on_the_overview_position():
    row, col = to_pixel((100.0, 100.0), (100.0, 100.0), 1.0, (200, 300))
    assert (row, col) == (100.0, 150.0)


def test_to_pixel_x_moves_right_and_y_moves_up():
    row, col = to_pixel((110.0, 105.0), (100.0, 100.0), 1.0, (200, 300))
    assert col == 160.0  # +10 um in x, 1um/px -> +10 px right
    assert row == 95.0  # +5 um in y -> 5 px *up* (smaller row)


def test_to_pixel_respects_voxel_size():
    _row, col = to_pixel((110.0, 100.0), (100.0, 100.0), 2.0, (200, 300))
    assert col == 155.0  # 10um / 2um-per-px = 5px


@pytest.mark.parametrize("voxel_x_um", [0.0, -1.0])
def test_to_pixel_zero_or_negative_voxel_is_nan_not_a_crash(voxel_x_um):
    """Plain Python floats raise ZeroDivisionError on x/0.0, unlike numpy --
    this must come back as a detectable nan instead of an opaque crash."""
    row, col = to_pixel((110.0, 105.0), (100.0, 100.0), voxel_x_um, (200, 300))
    assert math.isnan(row)
    assert math.isnan(col)


@pytest.mark.parametrize(
    "flip_x,flip_y,expected",
    [
        (True, False, (95.0, 140.0)),
        (False, True, (105.0, 160.0)),
        (True, True, (105.0, 140.0)),
    ],
)
def test_to_pixel_flips(flip_x, flip_y, expected):
    assert (
        to_pixel((110.0, 105.0), (100.0, 100.0), 1.0, (200, 300), flip_x, flip_y)
        == expected
    )


# --- marker_positions -------------------------------------------------------------
def test_marker_positions_end_to_end(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"), spacing=55.0, axis="x")
    overview_meta = make_meta(n_files=1, tiles=("wide1",), spacing=0.0, voxel=1.0)[0]
    markers = marker_positions(cfg, meta, overview_meta, "wide1")
    assert set(markers) == {"tile_a", "tile_b"}
    # tile_a sits exactly on the overview position -> dead center.
    assert markers["tile_a"] == (overview_meta.ny / 2, overview_meta.nx / 2)


def test_marker_positions_unknown_channel_is_loud(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"))
    overview_meta = make_meta(n_files=1, tiles=("wide1",))[0]
    with pytest.raises(ValueError, match="not a position"):
        marker_positions(cfg, meta, overview_meta, "nope")


# --- _first_plane -------------------------------------------------------------------
def test_first_plane_leaves_grayscale_alone():
    frame = np.arange(6 * 5).reshape(6, 5)
    out = _first_plane(frame, components_per_channel=1)
    assert out.shape == (6, 5)
    assert np.array_equal(out, frame)


def test_first_plane_strips_a_leading_channel_axis():
    """Several fluorescence channels bundled into one frame: (C, H, W)."""
    frame = np.stack([np.full((6, 5), c) for c in range(3)])
    assert frame.shape == (3, 6, 5)
    out = _first_plane(frame, components_per_channel=1)
    assert out.shape == (6, 5)
    assert np.all(out == 0)  # channel 0


def test_first_plane_collapses_a_trailing_rgb_axis():
    """An RGB overview camera: (H, W, 3) -- the extra axis is trailing, not
    leading, so naively taking index 0 would grab the first *row*."""
    frame = np.zeros((6, 5, 3))
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    out = _first_plane(frame, components_per_channel=3)
    assert out.shape == (6, 5)
    assert np.allclose(out, 20)  # mean of 10, 20, 30


def test_first_plane_handles_both_axes_together():
    """Multiple channels, each RGB: (C, H, W, 3)."""
    frame = np.zeros((2, 6, 5, 3))
    frame[0] = 9  # channel 0, every component
    frame[1] = 99  # channel 1 -- must not be picked
    out = _first_plane(frame, components_per_channel=3)
    assert out.shape == (6, 5)
    assert np.allclose(out, 9)


# --- downsample_plane --------------------------------------------------------------
def test_downsample_plane_leaves_a_small_image_alone():
    arr = np.zeros((100, 200))
    out, scale = downsample_plane(arr, max_dim=2000)
    assert scale == 1.0
    assert out.shape == arr.shape
    assert np.shares_memory(out, arr)


def test_downsample_plane_shrinks_a_big_image():
    arr = (np.arange(4000 * 3000) % 65535).astype(np.uint16).reshape(4000, 3000)
    out, scale = downsample_plane(arr, max_dim=1000)
    assert max(out.shape) <= 1000
    assert scale == pytest.approx(1 / 4)  # step = ceil(4000/1000) = 4


def test_downsample_plane_result_is_contiguous():
    """A numpy stride over the huge original array (`arr[::step, ::step]`,
    even forced contiguous afterwards) reproducibly segfaulted inside numpy's
    own C extension on a real, very large plane -- shrinking with Pillow
    instead must leave a genuine, independent, contiguous copy behind."""
    arr = (np.arange(4000 * 3000) % 65535).astype(np.uint16).reshape(4000, 3000)
    out, _scale = downsample_plane(arr, max_dim=1000)
    assert out.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(out, arr)


def test_downsample_plane_scale_maps_positions_back_consistently():
    """A marker near pixel (r, c) in the full image should land near
    (r*scale, c*scale) in the shrunk one -- not exact (Pillow resamples on
    its own continuous grid, it doesn't just pick every Nth source pixel),
    but close. A block, not a single pixel, so nearest-neighbour subsampling
    can't just step over it entirely."""
    arr = np.zeros((4000, 3000), dtype=np.uint8)
    arr[780:820, 1180:1220] = 255  # a 40x40 block centred on (800, 1200)
    out, scale = downsample_plane(arr, max_dim=1000)
    rows, cols = np.nonzero(out)
    assert rows.size > 0
    assert abs(rows.mean() - 800 * scale) <= 2
    assert abs(cols.mean() - 1200 * scale) <= 2


# --- normalize_to_uint8 -----------------------------------------------------------
def test_normalize_to_uint8_stretches_the_range():
    arr = np.array([0, 25, 50, 75, 100])
    out = normalize_to_uint8(arr, lo_pct=0, hi_pct=100)
    assert out.dtype == np.uint8
    assert out[0] == 0
    assert out[-1] == 255


def test_normalize_to_uint8_handles_a_flat_image():
    arr = np.full((4, 4), 7)
    out = normalize_to_uint8(arr)
    assert out.dtype == np.uint8
    assert np.all(out == 0)  # lo == hi -> everything clips to the bottom


# --- _find_seq_index ---------------------------------------------------------------
def test_find_seq_index_matches_requested_axes():
    loop_indices = [
        {"P": 0, "Z": 0, "C": 0},
        {"P": 0, "Z": 0, "C": 1},
        {"P": 1, "Z": 0, "C": 0},
    ]
    assert _find_seq_index(loop_indices, P=1, C=0) == 2


def test_find_seq_index_missing_returns_none():
    assert _find_seq_index([{"P": 0}], P=5) is None


def test_find_seq_index_defaults_to_zero_when_an_axis_has_no_loop():
    """A single-plane overview has no Z or T loop at all -- loop_indices then
    simply omits those keys, and z=0/t=0 must still match."""
    loop_indices = [{"P": 0}, {"P": 1}]
    assert _find_seq_index(loop_indices, P=1, Z=0, T=0, C=0) == 1


def test_find_seq_index_picks_a_specific_z_and_t():
    loop_indices = [
        {"P": 0, "Z": 0, "T": 0},
        {"P": 0, "Z": 1, "T": 0},
        {"P": 0, "Z": 0, "T": 1},
        {"P": 0, "Z": 1, "T": 1},
    ]
    assert _find_seq_index(loop_indices, P=0, Z=1, T=1, C=0) == 3


# --- render_overview -----------------------------------------------------------------
def test_render_overview_writes_the_file(tmp_path):
    image = np.zeros((20, 30), dtype=np.uint8)
    out = tmp_path / "sub" / "overview.png"
    render_overview(image, {"tile_a": (5.0, 5.0)}, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_overview_is_loud_if_save_does_not_actually_write(tmp_path, monkeypatch):
    """The post-save existence check: if Image.save ever returns without
    error but nothing lands on disk, that must not pass as success."""
    from PIL import Image

    monkeypatch.setattr(Image.Image, "save", lambda self, *a, **kw: None)
    image = np.zeros((4, 4), dtype=np.uint8)
    out = tmp_path / "overview.png"
    with pytest.raises(RuntimeError, match="does not exist afterwards"):
        render_overview(image, {}, out)

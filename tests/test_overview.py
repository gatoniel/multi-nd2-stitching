import numpy as np
import pytest
import yaml
from helpers import make_meta

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.overview import (
    _find_seq_index,
    marker_positions,
    normalize_to_uint8,
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

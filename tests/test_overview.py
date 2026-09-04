import numpy as np
import pytest
import yaml
from helpers import make_meta

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.overview import (
    _find_seq_index,
    _want_axes,
    block_reduce,
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
    markers = marker_positions(cfg, meta, overview_meta, 0)
    assert set(markers) == {"tile_a", "tile_b"}
    # tile_a sits exactly on the overview position -> dead center.
    assert markers["tile_a"] == (overview_meta.ny / 2, overview_meta.nx / 2)


def test_marker_positions_channel_none_uses_the_one_implicit_position(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"), spacing=55.0, axis="x")
    overview_meta = make_meta(n_files=1, tiles=("wide1",), spacing=0.0, voxel=1.0)[0]
    markers = marker_positions(cfg, meta, overview_meta, None)
    assert markers["tile_a"] == (overview_meta.ny / 2, overview_meta.nx / 2)


def test_marker_positions_out_of_range_channel_is_loud(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"))
    overview_meta = make_meta(n_files=1, tiles=("wide1",))[0]
    with pytest.raises(ValueError, match="out of range"):
        marker_positions(cfg, meta, overview_meta, 5)


def test_marker_positions_pixel_size_override(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"), spacing=55.0, axis="x")
    overview_meta = make_meta(n_files=1, tiles=("wide1",), spacing=0.0, voxel=1.0)[0]
    default = marker_positions(cfg, meta, overview_meta, 0)
    overridden = marker_positions(cfg, meta, overview_meta, 0, pixel_size_um=2.0)
    # Half the pixel size in um/px -> tile_b's offset is twice as many px.
    dcol_default = default["tile_b"][1] - default["tile_a"][1]
    dcol_overridden = overridden["tile_b"][1] - overridden["tile_a"][1]
    assert dcol_overridden == pytest.approx(dcol_default / 2)


def test_marker_positions_downsample_scales_coordinates(cfg_dict):
    cfg = _cfg(cfg_dict)
    meta = make_meta(n_files=2, tiles=("tile_a", "tile_b"), spacing=55.0, axis="x")
    overview_meta = make_meta(
        n_files=1, tiles=("wide1",), spacing=0.0, voxel=1.0, ny=800, nx=800
    )[0]
    full = marker_positions(cfg, meta, overview_meta, 0)
    small = marker_positions(cfg, meta, overview_meta, 0, downsample=4)
    assert small["tile_b"][1] == pytest.approx(full["tile_b"][1] / 4)
    assert small["tile_a"] == (100.0, 100.0)  # (ny/4/2, nx/4/2)


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


# --- _want_axes ---------------------------------------------------------------------
def test_want_axes_full_shape():
    assert _want_axes({"P": 3, "Z": 5, "C": 2, "Y": 10, "X": 10}, 1) == {
        "P": 1,
        "Z": 2,
        "C": 0,
    }


def test_want_axes_no_p_axis_ignores_position_index():
    # A bare (Y, X) file -- must not accidentally constrain P at all, unlike
    # the old `idx.get("P", 0) == 0` trick which only worked for index 0.
    assert _want_axes({"Y": 10, "X": 10}, 3) == {}


def test_want_axes_position_index_none_never_constrains_p():
    assert "P" not in _want_axes({"P": 3, "Y": 10, "X": 10}, None)


def test_want_axes_no_z_or_c_axis():
    assert _want_axes({"P": 2, "Y": 10, "X": 10}, 0) == {"P": 0}


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


# --- block_reduce -------------------------------------------------------------------
def test_block_reduce_mean_on_a_known_array():
    arr = np.array(
        [
            [0, 0, 4, 4],
            [0, 0, 4, 4],
            [8, 8, 12, 12],
            [8, 8, 12, 12],
        ],
        dtype=np.float64,
    )
    out = block_reduce(arr, 2, "mean")
    np.testing.assert_array_equal(out, [[0, 4], [8, 12]])


def test_block_reduce_median_on_a_known_array():
    arr = np.array(
        [
            [0, 0, 100, 4],
            [0, 0, 4, 4],
            [8, 8, 12, 12],
            [8, 8, 12, 12],
        ],
        dtype=np.float64,
    )
    out = block_reduce(arr, 2, "median")
    # top-right block is [100, 4, 4, 4] -> median 4
    np.testing.assert_array_equal(out, [[0, 4], [8, 12]])


def test_block_reduce_trims_a_remainder_edge():
    arr = np.zeros((5, 5), dtype=np.float64)
    arr[:4, :4] = 1.0
    out = block_reduce(arr, 2, "mean")
    assert out.shape == (2, 2)
    assert np.all(out == 1.0)


def test_block_reduce_factor_one_is_a_no_op():
    arr = np.arange(9, dtype=np.float64).reshape(3, 3)
    out = block_reduce(arr, 1, "mean")
    np.testing.assert_array_equal(out, arr)


def test_block_reduce_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown reduction"):
        block_reduce(np.zeros((4, 4)), 2, "max")


@pytest.mark.parametrize("method", ["mean", "median"])
def test_block_reduce_chunked_matches_a_one_shot_reference(method):
    # Force many small chunks via a tiny byte budget, and confirm the result
    # is identical to reducing the whole array in one call -- proves the
    # row-block loop doesn't change the answer, only how it gets there.
    import multi_nd2_stitching.overview as overview_mod

    rng = np.random.default_rng(0)
    arr = rng.random((240, 60)).astype(np.float64)
    factor = 4

    reduce_fn = np.mean if method == "mean" else np.median
    ny2, nx2 = (240 // factor) * factor, (60 // factor) * factor
    reference = reduce_fn(
        arr[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor),
        axis=(1, 3),
    )

    original_budget = overview_mod.BLOCK_REDUCE_BUDGET_BYTES
    overview_mod.BLOCK_REDUCE_BUDGET_BYTES = 64  # forces rows_per_block == factor
    try:
        chunked = block_reduce(arr, factor, method)
    finally:
        overview_mod.BLOCK_REDUCE_BUDGET_BYTES = original_budget

    np.testing.assert_allclose(chunked, reference)


class _GuardedArray(np.ndarray):
    """Raises if ever indexed with a step other than 1 -- the shape of the
    original `arr[::N]` crash, guarded against creeping back in."""

    def __getitem__(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        for part in parts:
            if isinstance(part, slice) and part.step not in (None, 1):
                raise AssertionError(f"strided slice used: {key}")
        return super().__getitem__(key)


def test_block_reduce_never_uses_a_strided_slice():
    arr = np.arange(64, dtype=np.float64).reshape(8, 8).view(_GuardedArray)
    block_reduce(arr, 2, "mean")


def test_block_reduce_progress_wraps_the_block_iterable():
    seen = []

    def progress(xs):
        seen.append(list(xs))
        return seen[-1]

    arr = np.zeros((4, 4), dtype=np.float64)
    block_reduce(arr, 2, "mean", progress=progress)
    assert seen  # progress was actually called with the block starts

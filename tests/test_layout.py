import numpy as np
import pytest
import yaml
from helpers import build, make_meta

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.layout import build_layout


# --- timeline -----------------------------------------------------------------
def test_timeline_concatenates_files(cfg_dict):
    lay = build(cfg_dict, n_files=2, nt=5)
    assert (lay.nt, lay.file_start) == (10, (0, 5))


@pytest.mark.parametrize(
    "t,expected", [(0, (0, 0)), (4, (0, 4)), (5, (1, 0)), (9, (1, 4))]
)
def test_locate(cfg_dict, t, expected):
    assert build(cfg_dict, n_files=2, nt=5).locate(t) == expected


@pytest.mark.parametrize("t", [-1, 10])
def test_locate_out_of_range(cfg_dict, t):
    with pytest.raises(IndexError):
        build(cfg_dict, n_files=2, nt=5).locate(t)


def test_ragged_files(cfg_dict):
    """Files need not have equal length; file_start must follow the real lengths."""
    from multi_nd2_stitching.metadata import Metadata

    meta = make_meta(n_files=3, nt=4)
    meta = Metadata(
        tuple(
            m.__class__(
                **{**{f.name: getattr(m, f.name) for f in m.__attrs_attrs__}, "nt": n}
            )
            for m, n in zip(meta.files, [3, 7, 2])
        )
    )
    lay = build_layout(
        loads_config(
            yaml.safe_dump(
                {
                    **cfg_dict,
                    "files": ["a", "b", "c"],
                    "positions": {
                        "tile_a": {"start": [0, 0], "reference_in_files": [0, 1, 2]},
                        "tile_b": {"start": [0, 0]},
                    },
                }
            )
        ),
        meta,
    )
    assert (lay.nt, lay.file_start) == (12, (0, 3, 10))
    assert lay.locate(10) == (2, 0)


# --- tiles --------------------------------------------------------------------
def test_end_is_exclusive(cfg_dict):
    cfg_dict["files"] = [f"f{i}" for i in range(3)]
    cfg_dict["positions"]["tile_a"]["reference_in_files"] = [0, 1, 2]
    cfg_dict["positions"]["tile_b"] = {"start": [0, 0], "end": 2}
    lay = build(cfg_dict, n_files=3, nt=5)
    assert lay.tile["tile_b"].last_t == 10  # files 0 and 1 only
    assert lay.tile_alive[9, lay.ti("tile_b")]
    assert not lay.tile_alive[10, lay.ti("tile_b")]


def test_start_timepoint_within_file(cfg_dict):
    cfg_dict["positions"]["tile_b"] = {"start": [1, 2]}
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.tile["tile_b"].first_t == 7
    assert not lay.tile_alive[6, lay.ti("tile_b")]
    assert lay.tile_alive[7, lay.ti("tile_b")]


def test_missing_position_name_is_loud(cfg_dict):
    cfg_dict["positions"]["ghost"] = {"start": [0, 0]}
    with pytest.raises(ValueError, match="no matching position"):
        build(cfg_dict)


def test_alias_resolves_a_renamed_position(cfg_dict):
    cfg_dict["positions"]["renamed"] = {"start": [0, 0], "aliases": ["tile_b"]}
    del cfg_dict["positions"]["tile_b"]
    lay = build(cfg_dict)
    assert lay.tile["renamed"].position == (1, 1)


# --- pair discovery -----------------------------------------------------------
def test_pairs_found_along_x(cfg_dict):
    lay = build(cfg_dict, axis="x")
    assert [(p.a, p.b, p.axis) for p in lay.pairs] == [("tile_b", "tile_a", 2)]


def test_pairs_found_along_y(cfg_dict):
    lay = build(cfg_dict, axis="y")
    assert [p.axis for p in lay.pairs] == [1]


def test_no_pairs_when_spacing_is_wrong(cfg_dict):
    assert build(cfg_dict, spacing=200.0).pairs == ()


def test_flip_x_reverses_only_x_pairs(cfg_dict):
    a = build(cfg_dict, axis="x").pairs[0]
    cfg_dict["flip_x"] = True
    b = build(cfg_dict, axis="x").pairs[0]
    assert (b.a, b.b) == (a.b, a.a)


def test_pairs_are_deterministic(cfg_dict):
    assert build(cfg_dict).pairs == build(cfg_dict).pairs


# --- shift_px -----------------------------------------------------------------
def test_shift_px_derived_from_voxel_size(cfg_dict):
    assert build(cfg_dict, voxel=0.1).shift_px == 550


def test_shift_px_explicit_wins(cfg_dict):
    cfg_dict["shift_px"] = 42
    assert build(cfg_dict, voxel=0.1).shift_px == 42


# --- overrides ----------------------------------------------------------------
def test_drop_removes_tile_at_that_timepoint_only(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    lay = build(cfg_dict)
    i = lay.ti("tile_b")
    assert not lay.tile_alive[3, i]
    assert lay.tile_alive[2, i] and lay.tile_alive[4, i]


def test_drop_also_kills_its_pairs(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    lay = build(cfg_dict)
    assert lay.pairs_at(3) == []
    assert len(lay.pairs_at(2)) == 1


def test_anchor_adds_a_single_timepoint(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "anchor": ["tile_b"]}]
    lay = build(cfg_dict)
    assert lay.anchors_at(3) == ["tile_a", "tile_b"]
    assert lay.anchors_at(2) == ["tile_a"]


def test_a_dropped_tile_is_never_an_anchor(cfg_dict):
    """drop wins over reference_in_files, so the masks stay consistent."""
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_a"], "anchor": ["tile_b"]}]
    lay = build(cfg_dict)
    assert lay.anchors_at(3) == ["tile_b"]


# --- sanity -------------------------------------------------------------------
def test_layout_is_a_pure_function(cfg_dict):
    """Same inputs -> identical masks. Nothing is carried over between builds."""
    a, b = build(cfg_dict), build(cfg_dict)
    assert np.array_equal(a.tile_alive, b.tile_alive)
    assert np.array_equal(a.is_anchor, b.is_anchor)
    assert np.array_equal(a.pair_alive, b.pair_alive)


def test_tile_size_mismatch_is_loud(cfg_dict):
    from multi_nd2_stitching.metadata import Metadata

    m = make_meta(n_files=2)
    bad = Metadata(
        (
            m[0],
            type(m[1])(
                **{
                    **{f.name: getattr(m[1], f.name) for f in m[1].__attrs_attrs__},
                    "nx": 512,
                }
            ),
        )
    )
    with pytest.raises(ValueError, match="tile size differs"):
        build_layout(loads_config(yaml.safe_dump(cfg_dict)), bad)

import numpy as np
import pytest
import yaml
from helpers import build, grid_meta, make_meta, stub_files

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.layout import build_layout, corner_direction
from multi_nd2_stitching.validate import (
    check_corner,
    check_layout,
    check_realign,
    check_shaped_peak,
)

# A 2x2 grid, one step = 55um, same spacing test_placement.py's SQUARE uses.
SQUARE = {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (0.0, 55.0), "d": (55.0, 55.0)}
# The gap case this all exists for: b and c are diagonal, but nothing sits at
# the (55, 55) corner to "complete the square" via two edge Pairs instead.
ELL = {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (0.0, 55.0)}


def _grid(tmp_path, coords, nt=3):
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {n: {"start": [0, 0]} for n in coords},
    }
    return build_layout(
        loads_config(yaml.safe_dump(cfg)), grid_meta(coords, files, nt=nt)
    )


# --- timeline -----------------------------------------------------------------
def test_timeline_concatenates_files(cfg_dict):
    lay = build(cfg_dict, n_files=2, nt=5)
    assert (lay.nt, lay.file_start) == (10, (0, 5))


# --- stop_at --------------------------------------------------------------------
def test_stop_at_absent_leaves_nt_equal_to_raw_nt(cfg_dict):
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.nt == lay.raw_nt == 10


def test_stop_at_truncates_nt(cfg_dict):
    cfg_dict["stop_at"] = 7
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.nt == 7
    assert lay.raw_nt == 10


def test_stop_at_beyond_the_real_timeline_is_a_noop(cfg_dict):
    cfg_dict["stop_at"] = 1000
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.nt == lay.raw_nt == 10


def test_stop_at_shapes_the_masks(cfg_dict):
    cfg_dict["stop_at"] = 7
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.tile_alive.shape[0] == 7
    assert lay.is_anchor.shape[0] == 7
    assert lay.pair_alive.shape[0] == 7


def test_stop_at_tile_starting_past_it_is_never_alive(cfg_dict):
    cfg_dict["stop_at"] = 3
    cfg_dict["positions"]["tile_b"] = {"start": [1, 0]}  # global t=5
    lay = build(cfg_dict, n_files=2, nt=5)
    assert not any(lay.tile_alive[t, lay.ti("tile_b")] for t in range(lay.nt))


def test_stop_at_open_ended_tile_stops_at_the_truncation(cfg_dict):
    """last_t = nt for an open-ended tile -- it should end at the truncated
    nt, not the file-derived total."""
    cfg_dict["stop_at"] = 7
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.tile["tile_a"].last_t == 7


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
            for m, n in zip(meta.files, [3, 7, 2], strict=False)
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


# --- exclude_at -----------------------------------------------------------------
def test_exclude_at_absent_leaves_the_timeline_untouched(cfg_dict):
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.excluded == ()
    assert lay.raw_t == tuple(range(10))
    assert lay.nt == lay.stop_nt == lay.raw_nt == 10


def test_exclude_at_shrinks_nt_and_compacts_raw_t(cfg_dict):
    cfg_dict["exclude_at"] = [3, 4]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.nt == 8
    assert lay.stop_nt == lay.raw_nt == 10
    assert lay.raw_t == (0, 1, 2, 5, 6, 7, 8, 9)
    assert lay.excluded == (3, 4)


def test_exclude_at_shapes_the_masks(cfg_dict):
    cfg_dict["exclude_at"] = [3, 4]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.tile_alive.shape[0] == 8
    assert lay.is_anchor.shape[0] == 8
    assert lay.pair_alive.shape[0] == 8


def test_exclude_at_locate_skips_the_gap(cfg_dict):
    """Compacted t=3 is raw t=5 -- file 1, local t=0 -- not raw t=3."""
    cfg_dict["exclude_at"] = [3, 4]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.locate(2) == (0, 2)
    assert lay.locate(3) == (1, 0)


def test_exclude_at_out_of_range_entries_are_a_noop(cfg_dict):
    cfg_dict["exclude_at"] = [99999]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.excluded == ()
    assert lay.nt == 10


def test_exclude_at_raw_to_t_omits_excluded_entries(cfg_dict):
    cfg_dict["exclude_at"] = [3, 4]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert 3 not in lay.raw_to_t and 4 not in lay.raw_to_t
    assert lay.raw_to_t[5] == 3


def test_exclude_at_beyond_stop_at_is_a_noop(cfg_dict):
    """stop_at truncates first; an exclude past that boundary has nothing to
    remove, same as an override there would."""
    cfg_dict["stop_at"] = 5
    cfg_dict["exclude_at"] = [7]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.excluded == ()
    assert lay.nt == 5


def test_exclude_at_within_stop_at_still_compacts(cfg_dict):
    cfg_dict["stop_at"] = 7
    cfg_dict["exclude_at"] = [3, 4]
    lay = build(cfg_dict, n_files=2, nt=5)
    assert lay.stop_nt == 7
    assert lay.excluded == (3, 4)
    assert lay.nt == 5
    assert lay.raw_t == (0, 1, 2, 5, 6)


def test_exclude_at_check_layout_is_fine_across_the_gap(cfg_dict):
    """The whole point: a tile alive before and after an excluded stretch
    still routes fine, drifting straight across the gap."""
    cfg_dict["exclude_at"] = [3, 4]
    assert check_layout(build(cfg_dict, n_files=2, nt=5)) == []


# build_plan's raw_gap/TimeTask behavior across an excluded stretch is
# tested in test_offsets.py, alongside the rest of build_plan.


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


# --- position_in_files: resolving a position by index, not name ---------------
def _blank_names(meta, file_i):
    """Same metadata, but file_i's positions have no names -- as if nd2
    never recorded one, which is exactly what position_in_files is for."""
    from multi_nd2_stitching.metadata import Metadata

    files = list(meta.files)
    m = files[file_i]
    files[file_i] = m.__class__(
        **{
            **{f.name: getattr(m, f.name) for f in m.__attrs_attrs__},
            "position_names": tuple("" for _ in m.position_names),
        }
    )
    return Metadata(tuple(files))


def _meta_with_extra_position(n_files=2, nt=5):
    """Like make_meta's default line, but each file has a third, unclaimed
    position -- lets a test force an override onto an index no configured
    tile would ever name-match, with no collision risk."""
    from multi_nd2_stitching.metadata import FileMeta, Metadata

    return Metadata(
        tuple(
            FileMeta(
                path=f"f{i}.nd2",
                nt=nt,
                nz=80,
                ny=724,
                nx=724,
                position_names=("tile_a", "tile_b", "extra"),
                stage_um=((0.0, 0.0), (55.0, 0.0), (110.0, 0.0)),
                voxel_x_um=0.1,
            )
            for i in range(n_files)
        )
    )


def test_position_in_files_resolves_without_any_name_match(cfg_dict):
    meta = _blank_names(make_meta(n_files=2, nt=5), 0)
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 0}
    cfg_dict["positions"]["tile_b"]["position_in_files"] = {0: 1}
    lay = build_layout(loads_config(yaml.safe_dump(cfg_dict)), meta)
    assert lay.tile["tile_a"].position[0] == 0
    assert lay.tile["tile_b"].position[0] == 1


def test_position_in_files_mixes_with_name_resolution_across_files(cfg_dict):
    """File 0 has no names (needs the override); file 1 does (still resolves
    by name) -- both for the same tile."""
    meta = _blank_names(make_meta(n_files=2, nt=5), 0)
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 0}
    cfg_dict["positions"]["tile_b"]["position_in_files"] = {0: 1}
    lay = build_layout(loads_config(yaml.safe_dump(cfg_dict)), meta)
    assert lay.tile["tile_a"].position == (0, 0)
    assert lay.tile["tile_b"].position == (1, 1)


def test_position_in_files_overrides_even_when_a_name_would_match(cfg_dict):
    meta = _meta_with_extra_position()
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 2}
    lay = build_layout(loads_config(yaml.safe_dump(cfg_dict)), meta)
    assert lay.tile["tile_a"].position[0] == 2  # not 0, which the name matches
    assert lay.tile["tile_b"].position[0] == 1  # untouched, still resolves by name


def test_position_in_files_out_of_range_is_loud(cfg_dict):
    meta = make_meta(n_files=2, nt=5)  # 2 named positions per file
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 5}
    with pytest.raises(ValueError, match="out of range"):
        build_layout(loads_config(yaml.safe_dump(cfg_dict)), meta)


def test_position_in_files_collision_with_a_name_match_is_loud(cfg_dict):
    """The general safety net tier 1 can't provide: an override colliding
    with a *different* tile's name-matched position."""
    meta = make_meta(n_files=2, nt=5)
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 1}  # tile_b's slot
    with pytest.raises(ValueError, match="both resolve to position 1 in file 0"):
        build_layout(loads_config(yaml.safe_dump(cfg_dict)), meta)


# --- missing_in_files: a gap in the middle of an otherwise-contiguous range ---
def test_missing_in_files_gap_resolves_to_none_without_raising(cfg_dict):
    cfg_dict["files"] = [f"f{i}" for i in range(3)]
    cfg_dict["positions"]["tile_a"]["reference_in_files"] = [0, 1, 2]
    cfg_dict["positions"]["tile_b"] = {"start": [0, 0], "missing_in_files": [1]}
    lay = build(cfg_dict, n_files=3, nt=5)
    assert lay.tile["tile_b"].position[0] is not None
    assert lay.tile["tile_b"].position[1] is None
    assert lay.tile["tile_b"].position[2] is not None


def test_missing_in_files_clears_tile_alive_for_the_gap_only(cfg_dict):
    cfg_dict["files"] = [f"f{i}" for i in range(3)]
    cfg_dict["positions"]["tile_a"]["reference_in_files"] = [0, 1, 2]
    cfg_dict["positions"]["tile_b"] = {"start": [0, 0], "missing_in_files": [1]}
    lay = build(cfg_dict, n_files=3, nt=5)
    i = lay.ti("tile_b")
    assert lay.tile_alive[4, i]  # file 0's last t
    assert not lay.tile_alive[5, i]  # file 1's first t: the gap
    assert not lay.tile_alive[9, i]  # file 1's last t: the gap
    assert lay.tile_alive[10, i]  # file 2's first t: alive again


def test_missing_in_files_reappearance_via_an_existing_anchor_is_fine(cfg_dict):
    """The realistic case: tile_b never needed its own anchor -- it hangs off
    tile_a, which stays alive (and a reference) straight through the gap."""
    cfg_dict["files"] = [f"f{i}" for i in range(3)]
    cfg_dict["positions"]["tile_a"]["reference_in_files"] = [0, 1, 2]
    cfg_dict["positions"]["tile_b"] = {"start": [0, 0], "missing_in_files": [1]}
    lay = build(cfg_dict, n_files=3, nt=5)
    assert check_layout(lay) == []


def test_missing_in_files_reappearance_without_an_anchor_is_still_flagged(cfg_dict):
    """A lone tile reappearing after a gap needs a fresh anchor, same as any
    other tile starting mid-run -- check_layout catches it unchanged."""
    cfg_dict["files"] = [f"f{i}" for i in range(3)]
    cfg_dict["positions"] = {
        "tile_a": {
            "start": [0, 0],
            "reference_in_files": [0],
            "missing_in_files": [1],
        }
    }
    lay = build(cfg_dict, n_files=3, nt=5)
    problems = check_layout(lay)
    assert any("no anchor" in p for p in problems), problems


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


# --- corner discovery -----------------------------------------------------------
def test_corners_found_diagonally(tmp_path):
    lay = _grid(tmp_path, SQUARE)
    assert {(c.a, c.b) for c in lay.corners} == {("a", "d"), ("b", "c")}


def test_edge_neighbours_are_not_also_corners(tmp_path):
    lay = _grid(tmp_path, SQUARE)
    edges = {(p.a, p.b) for p in lay.pairs} | {(p.b, p.a) for p in lay.pairs}
    corners = {(c.a, c.b) for c in lay.corners} | {(c.b, c.a) for c in lay.corners}
    assert edges.isdisjoint(corners)


def test_no_corners_in_a_line(cfg_dict):
    """make_meta only lays tiles out in a line -- nothing is ever diagonal."""
    assert build(cfg_dict).corners == ()


def test_corner_survives_a_missing_third_tile(tmp_path):
    """The whole point: b and c are diagonal even though nothing occupies the
    (55, 55) corner to connect them via two edge Pairs instead."""
    lay = _grid(tmp_path, ELL)
    assert {(c.a, c.b) for c in lay.corners} == {("b", "c")}


def test_corners_are_deterministic(tmp_path):
    assert _grid(tmp_path, SQUARE).corners == _grid(tmp_path, SQUARE).corners


def test_corner_alive_tracks_both_tiles(tmp_path):
    lay = _grid(tmp_path, ELL, nt=3)
    k = next(k for k, c in enumerate(lay.corners) if {c.a, c.b} == {"b", "c"})
    assert lay.corner_alive[:, k].all()


def test_dropping_either_tile_kills_the_corner(tmp_path):
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {n: {"start": [0, 0]} for n in ELL},
        "overrides": [{"at": 1, "drop": ["b"]}],
    }
    lay = build_layout(loads_config(yaml.safe_dump(cfg)), grid_meta(ELL, files, nt=3))
    assert lay.corners_at(1) == []
    assert len(lay.corners_at(0)) == 1


# --- corner_direction: the nominal crop direction a CornerTask needs ----------
def _square_layout(tmp_path, flip_x=False, flip_y=False):
    files = stub_files(tmp_path, 1)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "positions": {n: {"start": [0, 0]} for n in SQUARE},
    }
    meta = grid_meta(SQUARE, files, nt=1)
    return build_layout(loads_config(yaml.safe_dump(cfg)), meta), meta


@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
def test_corner_direction_matches_the_pair_convention(tmp_path, flip_x, flip_y):
    """Empirically tied to _discover_pairs' own a/b choice, across every
    flip_x/flip_y combination: whichever direction a real Pair independently
    gives for each axis, corner_direction gives the same sign for that axis."""
    lay, meta = _square_layout(tmp_path, flip_x, flip_y)
    px = next(p for p in lay.pairs if {p.a, p.b} == {"a", "b"} and p.axis == 2)
    py = next(p for p in lay.pairs if {p.a, p.b} == {"a", "c"} and p.axis == 1)
    x_dir_from_a = 1 if px.a == "a" else -1
    y_dir_from_a = 1 if py.a == "a" else -1
    dy_sign, dx_sign = corner_direction(lay.config, meta, lay.tile, "a", "d")
    assert (dy_sign, dx_sign) == (y_dir_from_a, x_dir_from_a)


def test_corner_direction_reverses_with_the_arguments(tmp_path):
    lay, meta = _square_layout(tmp_path)
    forward = corner_direction(lay.config, meta, lay.tile, "a", "d")
    backward = corner_direction(lay.config, meta, lay.tile, "d", "a")
    assert backward == (-forward[0], -forward[1])


# x, y: diagonal to each other only -- no edge Pair connects them at all.
DIAGONAL = {"x": (0.0, 0.0), "y": (55.0, 55.0)}


def test_check_layout_flags_a_diagonal_only_component_without_corner(tmp_path):
    """Without a corner override, y has no route to x's anchor at all --
    check_layout must see that, not silently ignore it."""
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {
            "x": {"start": [0, 0], "reference_in_files": [0, 1]},
            "y": {"start": [0, 0]},
        },
    }
    lay = build_layout(
        loads_config(yaml.safe_dump(cfg)), grid_meta(DIAGONAL, files, nt=1)
    )
    assert lay.pairs == ()  # confirms there really is no edge route
    problems = check_layout(lay)
    assert any("no anchor" in p and "'y'" in p for p in problems), problems


def test_check_layout_is_fine_once_corner_connects_the_component(tmp_path):
    """The fix this pins: check_layout has to see the same enabled-corner
    edge pool plan_placement builds, or it would wrongly flag a component a
    `corner` override genuinely connects."""
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {
            "x": {"start": [0, 0], "reference_in_files": [0, 1]},
            "y": {"start": [0, 0]},
        },
        "overrides": [{"at": [0, 1], "corner": ["x,y"]}],
    }
    lay = build_layout(
        loads_config(yaml.safe_dump(cfg)), grid_meta(DIAGONAL, files, nt=1)
    )
    assert check_layout(lay) == []


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


# --- connectivity -------------------------------------------------------------
def test_clean_config_is_connected(cfg_dict):
    assert check_layout(build(cfg_dict)) == []


def test_detects_component_without_anchor(cfg_dict):
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [0, 0]},
    }
    cfg_dict["overrides"] = [{"at": 7, "drop": ["a1"]}]
    problems = check_layout(build(cfg_dict, tiles=("a", "a1", "a2")))
    assert any("no anchor at t=7" in p for p in problems), problems


def test_replacement_anchor_fixes_it(cfg_dict):
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [0, 0]},
    }
    cfg_dict["overrides"] = [{"at": 7, "drop": ["a1"], "anchor": ["a2"]}]
    assert check_layout(build(cfg_dict, tiles=("a", "a1", "a2"))) == []


def test_detects_anchor_with_no_predecessor(cfg_dict):
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [1, 2]},
    }
    cfg_dict["overrides"] = [{"at": 7, "drop": ["a1"], "anchor": ["a2"]}]
    problems = check_layout(build(cfg_dict, tiles=("a", "a1", "a2")))
    assert any("no coordinate to drift from" in p for p in problems), problems


def test_ranges_are_collapsed_in_messages(cfg_dict):
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [0, 0]},
    }
    cfg_dict["overrides"] = [{"at": [3, 4, 5, 8], "drop": ["a1"]}]
    problems = check_layout(build(cfg_dict, tiles=("a", "a1", "a2")))
    assert any("t=3-5, 8" in p for p in problems), problems


# --- shaped_peak pairs ----------------------------------------------------------
def test_shaped_peak_pair_alive_throughout_is_fine(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "shaped_peak": ["tile_a,tile_b"]}]
    assert check_shaped_peak(build(cfg_dict)) == []


def test_shaped_peak_pair_reversed_order_is_fine(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "shaped_peak": ["tile_b,tile_a"]}]
    assert check_shaped_peak(build(cfg_dict)) == []


def test_shaped_peak_bare_tile_name_is_not_this_checks_concern(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "shaped_peak": ["tile_a"]}]
    assert check_shaped_peak(build(cfg_dict)) == []


def test_shaped_peak_flags_a_pair_that_never_exists(cfg_dict):
    """a and a2 are never adjacent (a1 sits between them), so 'a,a2' never
    names a real neighbour edge at any timepoint."""
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [0, 0]},
    }
    cfg_dict["overrides"] = [{"at": 3, "shaped_peak": ["a,a2"]}]
    problems = check_shaped_peak(build(cfg_dict, tiles=("a", "a1", "a2")))
    assert any("not a discovered neighbour pair" in p for p in problems), problems


def test_shaped_peak_flags_a_pair_not_alive_at_that_time(cfg_dict):
    cfg_dict["overrides"] = [
        {"at": 3, "drop": ["tile_b"], "shaped_peak": ["tile_a,tile_b"]}
    ]
    problems = check_shaped_peak(build(cfg_dict))
    assert any("not alive at t=3" in p for p in problems), problems


def test_shaped_peak_pair_alive_at_some_but_not_all_named_timepoints(cfg_dict):
    cfg_dict["overrides"] = [
        {"at": [3, 4, 5], "drop": ["tile_b"]},
        {"at": [2, 3, 4, 5, 6], "shaped_peak": ["tile_a,tile_b"]},
    ]
    problems = check_shaped_peak(build(cfg_dict))
    assert any("not alive at t=3-5" in p for p in problems), problems


def test_shaped_peak_pair_dead_ranges_are_collapsed_in_the_message(cfg_dict):
    cfg_dict["overrides"] = [
        {"at": [3, 4, 5, 8], "drop": ["tile_b"], "shaped_peak": ["tile_a,tile_b"]}
    ]
    problems = check_shaped_peak(build(cfg_dict))
    assert any("t=3-5, 8" in p for p in problems), problems


# --- realign pairs ----------------------------------------------------------
# Realignment_slices must differ from slices on the pair's *other* lateral
# axis (y, here -- the default line is along x) so the axis-aware no-op
# check (below) doesn't fire and contaminate the "this part is fine" cases.
_REALIGN_DIFFERS = {"y": [1, 2]}


def test_realign_pair_alive_throughout_is_fine(cfg_dict):
    cfg_dict["realignment_slices"] = _REALIGN_DIFFERS
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    assert check_realign(build(cfg_dict)) == []


def test_realign_pair_reversed_order_is_fine(cfg_dict):
    cfg_dict["realignment_slices"] = _REALIGN_DIFFERS
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_b,tile_a"]}]
    assert check_realign(build(cfg_dict)) == []


def test_realign_bare_tile_name_is_not_this_checks_concern(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_a"]}]
    assert check_realign(build(cfg_dict)) == []


def test_realign_flags_a_pair_that_never_exists(cfg_dict):
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "a1": {"start": [0, 0]},
        "a2": {"start": [0, 0]},
    }
    cfg_dict["realignment_slices"] = _REALIGN_DIFFERS
    cfg_dict["overrides"] = [{"at": 3, "realign": ["a,a2"]}]
    problems = check_realign(build(cfg_dict, tiles=("a", "a1", "a2")))
    assert any("not a discovered neighbour pair" in p for p in problems), problems


def test_realign_flags_a_pair_not_alive_at_that_time(cfg_dict):
    cfg_dict["realignment_slices"] = _REALIGN_DIFFERS
    cfg_dict["overrides"] = [
        {"at": 3, "drop": ["tile_b"], "realign": ["tile_a,tile_b"]}
    ]
    problems = check_realign(build(cfg_dict))
    assert any("not alive at t=3" in p for p in problems), problems


def test_realign_pair_flags_when_only_its_own_axis_differs(cfg_dict):
    """The addition specific to realign: realignment_slices differs from
    slices only along the pair's own (always-freed) axis, so the crop
    realign actually uses is identical to the plain one -- a silent no-op
    _check_realign's whole-volume comparison can't see."""
    cfg_dict["slices"] = {"x": [0, 10]}
    cfg_dict["realignment_slices"] = {"x": [5, 15]}  # differs only on x
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    problems = check_realign(build(cfg_dict))
    assert any("always freed for a pair" in p for p in problems), problems


def test_realign_pair_is_fine_when_the_other_axis_differs(cfg_dict):
    cfg_dict["slices"] = {"x": [0, 10]}
    cfg_dict["realignment_slices"] = {"x": [5, 15], **_REALIGN_DIFFERS}
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    assert check_realign(build(cfg_dict)) == []


# --- check_corner ---------------------------------------------------------------
def _diagonal_layout(tmp_path, overrides=None, nt=1):
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {n: {"start": [0, 0]} for n in DIAGONAL},
    }
    if overrides:
        cfg["overrides"] = overrides
    return build_layout(
        loads_config(yaml.safe_dump(cfg)), grid_meta(DIAGONAL, files, nt=nt)
    )


def test_corner_pair_alive_throughout_is_fine(tmp_path):
    lay = _diagonal_layout(tmp_path, overrides=[{"at": [0, 1], "corner": ["x,y"]}])
    assert check_corner(lay) == []


def test_corner_pair_reversed_order_is_fine(tmp_path):
    lay = _diagonal_layout(tmp_path, overrides=[{"at": [0, 1], "corner": ["y,x"]}])
    assert check_corner(lay) == []


def test_corner_flags_a_pair_that_never_exists(cfg_dict):
    """tile_a/tile_b are edge-adjacent (a line), never diagonal."""
    cfg_dict["overrides"] = [{"at": 3, "corner": ["tile_a,tile_b"]}]
    problems = check_corner(build(cfg_dict))
    assert any("not a discovered neighbour pair" in p for p in problems), problems


def test_corner_flags_a_pair_not_alive_at_that_time(tmp_path):
    lay = _diagonal_layout(
        tmp_path,
        overrides=[
            {"at": [1], "drop": ["y"]},
            {"at": [0, 1], "corner": ["x,y"]},
        ],
    )
    problems = check_corner(lay)
    assert any("not alive at t=1" in p for p in problems), problems


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


# --- unanchor -----------------------------------------------------------------
def test_unanchor_removes_the_anchor_but_keeps_the_tile(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "unanchor": ["tile_a"]}]
    lay = build(cfg_dict)
    i = lay.ti("tile_a")
    assert lay.tile_alive[3, i], "the tile must still be placed"
    assert not lay.is_anchor[3, i]
    assert lay.is_anchor[2, i] and lay.is_anchor[4, i]


def test_unanchor_keeps_the_pairs_alive(cfg_dict):
    """Unlike drop: the tile still takes part in the neighbour graph."""
    cfg_dict["overrides"] = [{"at": 3, "unanchor": ["tile_a"]}]
    lay = build(cfg_dict)
    assert len(lay.pairs_at(3)) == 1


def test_drop_and_unanchor_differ(cfg_dict):
    dropped = build({**cfg_dict, "overrides": [{"at": 3, "drop": ["tile_a"]}]})
    unanchored = build({**cfg_dict, "overrides": [{"at": 3, "unanchor": ["tile_a"]}]})
    assert "tile_a" not in dropped.tiles_at(3)
    assert "tile_a" in unanchored.tiles_at(3)
    assert dropped.anchors_at(3) == unanchored.anchors_at(3) == []


def test_handover_leaves_exactly_one_anchor(cfg_dict):
    """The whole point: unanchor one tile as another takes over."""
    cfg_dict["positions"]["tile_b"]["reference_in_files"] = [0, 1]
    cfg_dict["overrides"] = [{"at": 3, "unanchor": ["tile_a"], "anchor": ["tile_b"]}]
    lay = build(cfg_dict)
    assert lay.anchors_at(3) == ["tile_b"]


def test_unanchor_wins_over_reference_in_files(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "unanchor": ["tile_a"]}]
    assert build(cfg_dict).anchors_at(3) == []


def test_override_order_does_not_matter(cfg_dict):
    """Blocks are applied in passes, not in the order they are written."""
    a = build(
        {
            **cfg_dict,
            "overrides": [
                {"at": 3, "unanchor": ["tile_a"]},
                {"at": 3, "anchor": ["tile_b"]},
            ],
        }
    )
    b = build(
        {
            **cfg_dict,
            "overrides": [
                {"at": 3, "anchor": ["tile_b"]},
                {"at": 3, "unanchor": ["tile_a"]},
            ],
        }
    )
    assert a.anchors_at(3) == b.anchors_at(3) == ["tile_b"]


def test_a_dropped_tile_is_still_never_an_anchor(cfg_dict):
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_a"], "anchor": ["tile_b"]}]
    lay = build(cfg_dict)
    assert lay.anchors_at(3) == ["tile_b"]

from pathlib import Path

import attrs
import pytest
import yaml
from helpers import build, grid_meta, make_meta, stub_files

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.layout import build_layout, corner_direction
from multi_nd2_stitching.offsets import Crop, build_plan
from multi_nd2_stitching.store import Offset, OffsetStore

# a, b (x-neighbour of a), c (y-neighbour of a) -- b and c are diagonal to
# each other, with no fourth tile to connect them via edge Pairs instead.
ELL = {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (0.0, 55.0)}


def _corner_plan(tmp_path, corner=None, nt=3):
    files = stub_files(tmp_path, 2)
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "shift_px": 3,
        "positions": {n: {"start": [0, 0]} for n in ELL},
    }
    if corner is not None:
        cfg["overrides"] = [{"at": list(range(nt)), "corner": [corner]}]
    meta = grid_meta(ELL, files, nt=nt)
    lay = build_layout(loads_config(yaml.safe_dump(cfg)), meta)
    return build_plan(lay, meta), lay, meta


@pytest.fixture
def plan(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    meta = make_meta(n_files=2, nt=5, paths=files)
    return (
        build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta),
        cfg_dict,
        meta,
    )


# --- what gets planned --------------------------------------------------------
def test_pair_task_per_alive_pair_per_timepoint(plan):
    p, _, _ = plan
    assert len(p.pair_tasks) == 10  # 1 pair x 10 timepoints


def test_time_task_only_for_anchors_with_a_predecessor(plan):
    p, _, _ = plan
    assert len(p.time_tasks) == 9  # t=0 has nothing to drift from
    assert {t.name for t in p.time_tasks} == {"tile_a"}
    assert p.time_tasks[0].t_from == 0 and p.time_tasks[0].t_to == 1


def test_dropped_tile_removes_its_tasks(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    assert [t.t for t in p.pair_tasks].count(3) == 0
    assert len(p.pair_tasks) == 9


def test_realign_flag_and_crop(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["slices"] = {"z": [0, 20]}
    cfg_dict["realignment_slices"] = {"z": [5, 15]}
    cfg_dict["overrides"] = [{"at": 4, "realign": ["tile_a"]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    realigned = [t for t in p.time_tasks if t.realign]
    assert [t.t_to for t in realigned] == [4]
    assert realigned[0].crop.z == (5, 15)
    assert next(t for t in p.time_tasks if not t.realign).crop.z == (0, 20)


@pytest.mark.parametrize("pair", ["tile_a,tile_b", "tile_b,tile_a"])
def test_realign_pair_flags_either_order(cfg_dict, tmp_path, pair):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["realignment_slices"] = {"y": [100, 200]}
    cfg_dict["overrides"] = [{"at": 3, "realign": [pair]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    flagged = [t for t in p.pair_tasks if t.realign]
    assert [t.t for t in flagged] == [3]


def test_realign_pair_frees_its_own_axis_regardless_of_realignment_slices(
    cfg_dict, tmp_path
):
    """The correction this whole feature exists for: realignment_slices must
    never restrict the axis actually being correlated, even though it's free
    to restrict z and the *other* lateral axis."""
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["slices"] = {"z": [0, 20]}
    cfg_dict["realignment_slices"] = {"z": [5, 15], "y": [100, 200]}
    cfg_dict["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)

    realigned = next(t for t in p.pair_tasks if t.realign)
    assert realigned.axis == 2  # the default line is along x
    assert realigned.crop.z == (5, 15)
    assert realigned.crop.y == (100, 200)
    assert realigned.crop.x == (None, None)  # its own axis: always freed

    plain = next(t for t in p.pair_tasks if not t.realign)
    assert plain.crop.z == (0, 20)
    assert plain.crop.y == (None, None)  # `slices` never touched y
    assert plain.crop.x == (None, None)


def test_realign_pair_changes_the_key_via_the_crop(plan):
    """Unlike shaped_peak/near, `realign` itself isn't in PairTask.key() --
    it doesn't need to be: a real realignment_slices always changes the crop,
    and the crop is already part of the key (same as TimeTask)."""
    p, cfg, meta = plan
    cfg2 = {
        **cfg,
        "realignment_slices": {"y": [1, 2]},
        "overrides": [{"at": 3, "realign": ["tile_a,tile_b"]}],
    }
    q = build_plan(build(cfg2, n_files=2, nt=5, paths=cfg["files"]), meta)
    realigned = next(t for t in q.pair_tasks if t.t == 3)
    plain = next(t for t in p.pair_tasks if t.t == 3)
    assert realigned.realign and not plain.realign
    assert realigned.key != plain.key
    others_p = {t.key for t in p.pair_tasks if t.t != 3}
    others_q = {t.key for t in q.pair_tasks if t.t != 3}
    assert others_p == others_q


def test_shaped_peak_tile_name_flags_only_that_timepoint(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [{"at": 4, "shaped_peak": ["tile_a"]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    flagged = [t for t in p.time_tasks if t.shaped_peak]
    assert [t.t_to for t in flagged] == [4]


@pytest.mark.parametrize("pair", ["tile_a,tile_b", "tile_b,tile_a"])
def test_shaped_peak_pair_flags_either_order(cfg_dict, tmp_path, pair):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [{"at": 3, "shaped_peak": [pair]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    flagged = [t for t in p.pair_tasks if t.shaped_peak]
    assert [t.t for t in flagged] == [3]


def test_near_hint_flows_to_time_task(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [
        {"at": 4, "shaped_peak": ["tile_a"], "near": {"tile_a": [0, 5, -3]}}
    ]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    flagged = next(t for t in p.time_tasks if t.t_to == 4)
    assert flagged.near == (0, 5, -3)
    assert all(t.near is None for t in p.time_tasks if t.t_to != 4)


@pytest.mark.parametrize("pair", ["tile_a,tile_b", "tile_b,tile_a"])
def test_near_hint_flows_to_pair_task_either_order(cfg_dict, tmp_path, pair):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [
        {"at": 3, "shaped_peak": [pair], "near": {pair: [1, -2, 3]}}
    ]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    flagged = next(t for t in p.pair_tasks if t.t == 3)
    assert flagged.near == (1, -2, 3)


def test_near_hint_changes_the_key(plan):
    """Same reasoning as shaped_peak: a different search needs a different slot."""
    p, cfg, meta = plan
    cfg2 = {
        **cfg,
        "overrides": [
            {
                "at": 3,
                "shaped_peak": ["tile_a,tile_b"],
                "near": {"tile_a,tile_b": [0, 4, -1]},
            }
        ],
    }
    q = build_plan(build(cfg2, n_files=2, nt=5, paths=cfg["files"]), meta)
    hinted = next(t for t in q.pair_tasks if t.t == 3)
    plain = next(t for t in p.pair_tasks if t.t == 3)
    assert hinted.near is not None and plain.near is None
    assert hinted.key != plain.key
    others_p = {t.key for t in p.pair_tasks if t.t != 3}
    others_q = {t.key for t in q.pair_tasks if t.t != 3}
    assert others_p == others_q


def test_shaped_peak_changes_the_key(plan):
    """The whole design: a different computation needs a different cache slot."""
    p, cfg, meta = plan
    cfg2 = {**cfg, "overrides": [{"at": 3, "shaped_peak": ["tile_a,tile_b"]}]}
    q = build_plan(build(cfg2, n_files=2, nt=5, paths=cfg["files"]), meta)
    shaped = next(t for t in q.pair_tasks if t.t == 3)
    plain = next(t for t in p.pair_tasks if t.t == 3)
    assert shaped.shaped_peak and not plain.shaped_peak
    assert shaped.key != plain.key
    # every other pair task at every other timepoint is untouched
    others_p = {t.key for t in p.pair_tasks if t.t != 3}
    others_q = {t.key for t in q.pair_tasks if t.t != 3}
    assert others_p == others_q


# --- key behaviour: this is the whole design ----------------------------------
def test_key_is_stable_across_rebuilds(plan):
    p, cfg, meta = plan
    q = build_plan(build(cfg, n_files=2, nt=5, paths=cfg["files"]), meta)
    assert [t.key for t in p.tasks] == [t.key for t in q.tasks]


def test_renaming_a_tile_does_not_change_keys(plan):
    """The key names pixels, not tiles. A rename must not cost a recompute."""
    p, cfg, meta = plan
    cfg2 = {
        **cfg,
        "positions": {
            "renamed": {**cfg["positions"]["tile_a"], "aliases": ["tile_a"]},
            "tile_b": cfg["positions"]["tile_b"],
        },
    }
    q = build_plan(build(cfg2, n_files=2, nt=5, paths=cfg["files"]), meta)
    assert sorted(t.key for t in q.pair_tasks) == sorted(t.key for t in p.pair_tasks)


def test_changing_slices_changes_keys(plan):
    p, cfg, meta = plan
    cfg2 = {**cfg, "slices": {"z": [5, 40]}}
    q = build_plan(build(cfg2, n_files=2, nt=5, paths=cfg["files"]), meta)
    assert {t.key for t in q.tasks}.isdisjoint(t.key for t in p.tasks)


def test_unrelated_config_change_keeps_keys(plan):
    """Adding a tile that forms no pair must not invalidate anything."""
    p, cfg, _meta = plan
    cfg2 = {**cfg, "positions": {**cfg["positions"], "far": {"start": [0, 0]}}}
    meta2 = make_meta(
        n_files=2,
        nt=5,
        tiles=("tile_a", "tile_b", "far"),
        spacing=55.0,
        paths=cfg["files"],
    )
    q = build_plan(
        build(
            cfg2, n_files=2, nt=5, tiles=("tile_a", "tile_b", "far"), paths=cfg["files"]
        ),
        meta2,
    )
    assert {t.key for t in p.pair_tasks} <= {t.key for t in q.pair_tasks}


def test_rewriting_a_file_invalidates_its_keys(plan, tmp_path):
    p, cfg, meta = plan
    Path(cfg["files"][0]).write_bytes(b"y" * 9999)
    q = build_plan(build(cfg, n_files=2, nt=5, paths=cfg["files"]), meta)
    changed = {t.key for t in q.tasks} - {t.key for t in p.tasks}
    assert changed, "a rewritten ND2 must invalidate the offsets read from it"


def test_time_and_pair_keys_never_collide(plan):
    p, _, _ = plan
    keys = [t.key for t in p.tasks]
    assert len(set(keys)) == len(keys)


# --- windowing ----------------------------------------------------------------
def test_between_restricts_the_window(plan):
    p, _, _ = plan
    w = p.between(3, 6)
    assert {t.t for t in w.pair_tasks} == {3, 4, 5}
    assert {t.t_to for t in w.time_tasks} == {3, 4, 5}


def test_at_returns_both_kinds(plan):
    p, _, _ = plan
    assert len(p.at(4)) == 2


# --- pending ------------------------------------------------------------------
def test_pending_is_everything_when_store_is_empty(plan):
    p, _, _ = plan
    assert len(p.pending(OffsetStore())) == len(p.tasks)


def test_pending_shrinks_as_results_land(plan):
    p, _, _ = plan
    store = OffsetStore()
    store.put(p.tasks[0], Offset(0, 1, 2))
    assert len(p.pending(store)) == len(p.tasks) - 1


# --- Crop ---------------------------------------------------------------------
def test_crop_clamps_z_to_stack_depth():
    assert Crop.of((slice(5, 100), slice(None), slice(None)), 60).z == (5, 60)


def test_crop_roundtrips_to_slices():
    c = Crop.of((slice(5, 40), slice(1, 2), slice(None)), 80)
    assert c.as_slices() == (slice(5, 40), slice(1, 2), slice(None))


# --- CornerTask -----------------------------------------------------------------
def test_corner_task_not_created_without_an_override(tmp_path):
    p, _lay, _meta = _corner_plan(tmp_path)
    assert p.corner_tasks == ()


def test_corner_task_created_when_enabled_and_alive(tmp_path):
    p, _lay, _meta = _corner_plan(tmp_path, corner="b,c", nt=3)
    assert {t.t for t in p.corner_tasks} == {0, 1, 2}
    assert {(t.a, t.b) for t in p.corner_tasks} == {("b", "c")}


@pytest.mark.parametrize("corner", ["b,c", "c,b"])
def test_corner_task_enabled_either_order(tmp_path, corner):
    p, _lay, _meta = _corner_plan(tmp_path, corner=corner, nt=1)
    assert len(p.corner_tasks) == 1


def test_corner_task_crop_is_the_overlap_strip(tmp_path):
    """Same `n - shift_px` overlap-strip length crop_for_alignment/trim_for
    already use for a Pair's one axis, here on both lateral axes at once --
    not a `shift_px`-sized sliver at the tile's tip."""
    p, lay, _meta = _corner_plan(tmp_path, corner="b,c", nt=1)
    task = p.corner_tasks[0]
    s = lay.shift_px
    for crop in (task.crop_a, task.crop_b):
        assert crop.y[1] - crop.y[0] == lay.ny - s
        assert crop.x[1] - crop.x[0] == lay.nx - s
        assert crop.z == (None, lay.nz)  # full stack, no slices restriction given


def test_corner_task_crops_are_the_same_physical_strip(tmp_path):
    """a's crop faces b, b's crop faces a -- the same overlap strip, seen
    from each tile's own local origin."""
    p, lay, meta = _corner_plan(tmp_path, corner="b,c", nt=1)
    task = p.corner_tasks[0]
    dy_sign, dx_sign = corner_direction(lay.config, meta, lay.tile, task.a, task.b)
    s = lay.shift_px
    expect_a_y = (s, lay.ny) if dy_sign > 0 else (0, lay.ny - s)
    expect_b_y = (0, lay.ny - s) if dy_sign > 0 else (s, lay.ny)
    expect_a_x = (s, lay.nx) if dx_sign > 0 else (0, lay.nx - s)
    expect_b_x = (0, lay.nx - s) if dx_sign > 0 else (s, lay.nx)
    assert task.crop_a.y == expect_a_y
    assert task.crop_b.y == expect_b_y
    assert task.crop_a.x == expect_a_x
    assert task.crop_b.x == expect_b_x
    assert task.nominal == (0, dy_sign * s, dx_sign * s)


def test_corner_task_key_changes_with_the_crop(tmp_path):
    p, lay, meta = _corner_plan(tmp_path, corner="b,c", nt=1)
    task = p.corner_tasks[0]
    other = build_plan(
        build_layout(
            loads_config(
                yaml.safe_dump(
                    {
                        "files": lay.config.files,
                        "grid_spacing": 55,
                        "grid_spacing_error": 5,
                        "shift_px": 5,  # different -> different crop
                        "positions": {n: {"start": [0, 0]} for n in ELL},
                        "overrides": [{"at": [0], "corner": ["b,c"]}],
                    }
                )
            ),
            meta,
        ),
        meta,
    ).corner_tasks[0]
    assert task.key != other.key


def test_corner_task_key_changes_with_nominal_alone(tmp_path):
    """Even holding the crops fixed, a different nominal reconstruction is a
    different result -- it has to be its own key ingredient, not just
    implied by the crop bounds."""
    p, _lay, _meta = _corner_plan(tmp_path, corner="b,c", nt=1)
    task = p.corner_tasks[0]
    other = attrs.evolve(task, nominal=tuple(v + 1 for v in task.nominal))
    assert task.key != other.key

import pytest
from helpers import build, make_meta

from multi_nd2_stitching.offsets import Crop, build_plan
from multi_nd2_stitching.store import Offset, OffsetStore


@pytest.fixture
def plan(cfg_dict, tmp_path):
    files = []
    for i in range(2):
        p = tmp_path / f"f{i}.nd2"
        p.write_bytes(b"x" * (100 + i))
        files.append(str(p))
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
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    meta = make_meta(n_files=2, nt=5, paths=files)
    p = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    assert [t.t for t in p.pair_tasks].count(3) == 0
    assert len(p.pair_tasks) == 9


def test_realign_flag_and_crop(cfg_dict, tmp_path):
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
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
    open(cfg["files"][0], "wb").write(b"y" * 9999)
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

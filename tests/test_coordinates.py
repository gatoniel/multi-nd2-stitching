import numpy as np
import pytest
from helpers import build, make_meta, stub_files

from multi_nd2_stitching.coordinates import (
    MissingOffsets,
    build_coordinates,
)
from multi_nd2_stitching.offsets import build_plan
from multi_nd2_stitching.store import Offset, OffsetStore


@pytest.fixture
def setup(cfg_dict, tmp_path):
    def _make(tiles=("tile_a", "tile_b"), positions=None, overrides=None, nt=3):
        files = stub_files(tmp_path, 2)
        cfg_dict["files"] = files
        cfg_dict["shift_px"] = 3
        if positions:
            cfg_dict["positions"] = positions
        if overrides:
            cfg_dict["overrides"] = overrides
        meta = make_meta(n_files=2, nt=nt, tiles=tiles, paths=files)
        lay = build(cfg_dict, n_files=2, nt=nt, tiles=tiles, paths=files)
        return lay, build_plan(lay, meta)

    return _make


def fill(plan, store, time_off=(0, 0, 0), pair_off=(0, 0, 5)):
    for t in plan.time_tasks:
        store.put(t, Offset(*time_off))
    for p in plan.pair_tasks:
        store.put(p, Offset(*pair_off))
    return store


def test_first_anchor_is_the_origin(setup):
    lay, plan = setup()
    coords = build_coordinates(lay, plan, fill(plan, OffsetStore()))
    assert np.array_equal(coords[0, "tile_a"], np.zeros(3))


def test_neighbour_offset_places_the_second_tile(setup):
    lay, plan = setup()
    coords = build_coordinates(lay, plan, fill(plan, OffsetStore()))
    p = lay.pairs[0]
    assert np.array_equal(coords[0, p.b] - coords[0, p.a], np.array([0, 0, 5]))


def test_drift_accumulates_over_time(setup):
    lay, plan = setup()
    store = fill(plan, OffsetStore(), time_off=(0, 2, 0))
    coords = build_coordinates(lay, plan, store)
    assert coords[0, "tile_a"][1] == 0
    assert coords[1, "tile_a"][1] == 2
    assert coords[2, "tile_a"][1] == 4


def test_every_alive_tile_gets_placed(setup):
    lay, plan = setup()
    coords = build_coordinates(lay, plan, fill(plan, OffsetStore()))
    for t in range(lay.nt):
        assert set(coords.at(t)) == set(lay.tiles_at(t))


def test_missing_offsets_are_named_not_a_keyerror(setup):
    lay, plan = setup()
    with pytest.raises(MissingOffsets) as e:
        build_coordinates(lay, plan, OffsetStore())
    assert e.value.missing
    assert any("pair" in m or "time" in m for m in e.value.missing)


def test_a_disconnected_component_terminates(setup):
    """The original looped forever here. This must return, not hang."""
    lay, plan = setup(
        tiles=("a", "b", "c"),
        positions={
            "a": {"start": [0, 0], "reference_in_files": [0, 1]},
            "b": {"start": [0, 0]},
            "c": {"start": [0, 0]},
        },
        overrides=[{"at": 1, "drop": ["b"]}],
    )
    coords = build_coordinates(lay, plan, fill(plan, OffsetStore()))
    assert "c" not in coords.at(1)  # orphaned, but we got here
    assert "a" in coords.at(1)


def test_extent_covers_every_tile(setup):
    lay, plan = setup()
    coords = build_coordinates(lay, plan, fill(plan, OffsetStore()))
    ext = coords.extent((lay.nz, lay.ny, lay.nx))
    assert ext.shape == (3, 2)
    assert ext[2, 1] - ext[2, 0] == lay.nx + 5  # one tile plus the x offset


def test_sign_of_the_offset_does_not_change_the_span(setup):
    """tile_a is the anchor; whether it is `a` or `b` of the pair, the canvas
    has to be the same size."""
    lay, plan = setup()
    spans = []
    for off in [(0, 0, 5), (0, 0, -5)]:
        coords = build_coordinates(lay, plan, fill(plan, OffsetStore(), pair_off=off))
        ext = coords.extent((lay.nz, lay.ny, lay.nx))
        spans.append(int(ext[2, 1] - ext[2, 0]))
    assert spans == [lay.nx + 5, lay.nx + 5]


def test_drift_moves_the_extent(setup):
    lay, plan = setup()
    store = fill(plan, OffsetStore(), time_off=(0, 3, 0))
    coords = build_coordinates(lay, plan, store)
    ext = coords.extent((lay.nz, lay.ny, lay.nx))
    assert int(ext[1, 1] - ext[1, 0]) == lay.ny + 3 * (lay.nt - 1)


# --- partial timelines --------------------------------------------------------
def fill_until(plan, store, upto, time_off=(0, 0, 0), pair_off=(0, 0, 5)):
    """Simulate `stitch offsets --between 0 upto`."""
    for t in plan.time_tasks:
        if t.t_to < upto:
            store.put(t, Offset(*time_off))
    for p in plan.pair_tasks:
        if p.t < upto:
            store.put(p, Offset(*pair_off))
    return store


def test_a_prefix_can_be_placed_without_the_rest(setup):
    lay, plan = setup(nt=4)
    half = lay.nt // 2
    store = fill_until(plan, OffsetStore(), half)
    coords = build_coordinates(lay, plan, store, 0, half)
    assert coords.window == (0, half)
    for t in range(half):
        assert set(coords.at(t)) == set(lay.tiles_at(t))


def test_the_full_range_still_demands_everything(setup):
    lay, plan = setup(nt=4)
    store = fill_until(plan, OffsetStore(), lay.nt // 2)
    with pytest.raises(MissingOffsets):
        build_coordinates(lay, plan, store)


def test_extent_covers_only_the_window(setup):
    """A prefix must not be sized by tiles nobody asked about."""
    lay, plan = setup(nt=4)
    store = fill_until(plan, OffsetStore(), lay.nt, time_off=(0, 5, 0))
    tile = (lay.nz, lay.ny, lay.nx)
    early = build_coordinates(lay, plan, store, 0, 2).extent(tile)
    whole = build_coordinates(lay, plan, store).extent(tile)
    assert (early[1, 1] - early[1, 0]) < (whole[1, 1] - whole[1, 0])


def test_a_later_window_still_needs_the_drift_chain(setup):
    """Drift is absolute, so t=6 cannot be placed without steps 1..6."""
    lay, plan = setup(nt=4)
    store = OffsetStore()
    for p in plan.pair_tasks:
        if p.t >= 6:
            store.put(p, Offset(0, 0, 5))
    with pytest.raises(MissingOffsets) as e:
        build_coordinates(lay, plan, store, 6, 8)
    assert all("time" in m for m in e.value.missing)


def test_a_later_window_works_once_drift_exists(setup):
    lay, plan = setup(nt=4)
    store = OffsetStore()
    for t in plan.time_tasks:
        store.put(t, Offset(0, 1, 0))
    for p in plan.pair_tasks:
        if p.t >= 6:
            store.put(p, Offset(0, 0, 5))
    coords = build_coordinates(lay, plan, store, 6, 8)
    assert set(coords.at(6)) == set(lay.tiles_at(6))
    assert coords[6, "tile_a"][1] == 6, "drift accumulated from t=0"


def test_missing_message_names_the_command(setup):
    lay, plan = setup(nt=4)
    with pytest.raises(MissingOffsets, match="stitch offsets --between 0 3"):
        build_coordinates(lay, plan, OffsetStore(), 0, 3)


def test_frames_outside_the_window_stay_empty(setup):
    lay, plan = setup(nt=4)
    store = fill_until(plan, OffsetStore(), 3)
    coords = build_coordinates(lay, plan, store, 0, 3)
    assert len(coords.by_time) == lay.nt
    assert coords.at(lay.nt - 1) == {}

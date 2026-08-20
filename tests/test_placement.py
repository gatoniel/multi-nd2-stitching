"""Placement routing: one route per tile, and a flag when there is more."""

import pytest
import yaml
from helpers import grid_meta, stub_files

from multi_nd2_stitching.config import loads_config
from multi_nd2_stitching.layout import build_layout
from multi_nd2_stitching.placement import (
    DRIFT,
    ORIGIN,
    PAIR,
    group_runs,
    placements_for,
    plan_placement,
    render,
)

ANCHOR = {"reference_in_files": [0, 1]}
LINE = {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (110.0, 0.0)}
SQUARE = {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (0.0, 55.0), "d": (55.0, 55.0)}


@pytest.fixture
def scene(tmp_path):
    def _make(positions, coords, overrides=None, nt=3):
        files = stub_files(tmp_path, 2)
        cfg = {
            "files": files,
            "grid_spacing": 55,
            "grid_spacing_error": 5,
            "shift_px": 3,
            "positions": positions,
        }
        if overrides:
            cfg["overrides"] = overrides
        return build_layout(
            loads_config(yaml.safe_dump(cfg)), grid_meta(coords, files, nt=nt)
        )

    return _make


def chain(scene, **kw):
    return scene(
        {
            "a": {"start": [0, 0], **ANCHOR},
            "b": {"start": [0, 0]},
            "c": {"start": [0, 0]},
        },
        LINE,
        **kw,
    )


# --- the happy path -----------------------------------------------------------
def test_every_alive_tile_gets_exactly_one_step(scene):
    lay = chain(scene)
    for t in range(lay.nt):
        p = plan_placement(lay, t)
        assert [s.tile for s in p.steps] == sorted({s.tile for s in p.steps})
        assert {s.tile for s in p.steps} == set(lay.tiles_at(t))


def test_t0_seeds_the_origin(scene):
    p = plan_placement(chain(scene), 0)
    assert p.seeds[0].kind == ORIGIN
    assert p.seeds[0].tile == "a"


def test_later_timepoints_drift(scene):
    p = plan_placement(chain(scene), 1)
    assert p.seeds[0].kind == DRIFT


def test_neighbours_hang_off_the_seed(scene):
    p = plan_placement(chain(scene), 0).by_tile
    assert p["b"].kind == PAIR and p["b"].via == "a"
    assert p["c"].via == "b", "c is two hops out, not adjacent to the anchor"


def test_a_clean_chain_is_unambiguous(scene):
    lay = chain(scene)
    assert not any(plan_placement(lay, t).ambiguous for t in range(lay.nt))


def test_route_to_returns_the_whole_chain(scene):
    route = plan_placement(chain(scene), 0).route_to("c")
    assert [s.tile for s in route] == ["a", "b", "c"]


def test_route_to_an_unplaced_tile_is_empty(scene):
    lay = chain(scene)
    assert plan_placement(lay, 0).route_to("nope") == []


# --- ambiguity ----------------------------------------------------------------
def test_a_cycle_is_flagged(scene):
    """In a 2x2 grid the fourth tile is reachable two ways."""
    lay = scene(
        {n: {"start": [0, 0], **(ANCHOR if n == "a" else {})} for n in "abcd"},
        SQUARE,
    )
    p = plan_placement(lay, 0)
    assert p.ambiguous
    assert len(p.redundant) == 1
    assert set(p.redundant[0][:2]) == {"c", "d"}


def test_a_cycle_still_places_every_tile(scene):
    lay = scene(
        {n: {"start": [0, 0], **(ANCHOR if n == "a" else {})} for n in "abcd"},
        SQUARE,
    )
    p = plan_placement(lay, 0)
    assert {s.tile for s in p.steps} == set("abcd")
    assert not p.unplaced


def test_two_anchors_in_one_component_are_flagged(scene):
    lay = scene(
        {
            "a": {"start": [0, 0], **ANCHOR},
            "b": {"start": [0, 0]},
            "c": {"start": [0, 0], **ANCHOR},
        },
        LINE,
    )
    p = plan_placement(lay, 1)
    assert p.over_anchored == (("a", "c"),)
    assert p.ambiguous


def test_two_anchors_in_separate_components_are_fine(scene):
    """Two chains far apart each need their own anchor; that is not ambiguity."""
    lay = scene(
        {
            "a": {"start": [0, 0], **ANCHOR},
            "b": {"start": [0, 0]},
            "c": {"start": [0, 0], **ANCHOR},
        },
        {"a": (0.0, 0.0), "b": (55.0, 0.0), "c": (500.0, 0.0)},
    )
    p = plan_placement(lay, 1)
    assert not p.over_anchored
    assert not p.ambiguous


def test_a_dropped_tile_orphans_its_far_side(scene):
    lay = chain(scene, overrides=[{"at": 1, "drop": ["b"]}])
    p = plan_placement(lay, 1)
    assert p.unplaced == ("c",)


# --- grouping over time -------------------------------------------------------
def test_identical_timepoints_collapse(scene):
    lay = chain(scene)
    runs = group_runs(placements_for(lay))
    assert len(runs) == 2, "t=0 is the origin; everything after it drifts"
    assert runs[0][:2] == (0, 0)
    assert runs[1][0] == 1


def test_a_change_in_topology_starts_a_new_run(scene):
    lay = chain(scene, overrides=[{"at": 3, "drop": ["c"]}])
    runs = group_runs(placements_for(lay))
    assert any(a == b == 3 for a, b, _ in runs)


# --- rendering ----------------------------------------------------------------
def test_render_marks_ambiguous_runs(scene):
    lay = scene(
        {n: {"start": [0, 0], **(ANCHOR if n == "a" else {})} for n in "abcd"},
        SQUARE,
    )
    text = "\n".join(render(placements_for(lay)))
    assert "[AMBIGUOUS]" in text
    assert "redundant edge" in text


def test_render_is_quiet_when_all_is_well(scene):
    text = "\n".join(render(placements_for(chain(scene))))
    assert "AMBIGUOUS" not in text and "!" not in text


def test_render_draws_the_tree(scene):
    text = "\n".join(render(placements_for(chain(scene))))
    assert "[origin]" in text and "x→ b" in text


def test_render_tile_mode_shows_one_chain(scene):
    text = "\n".join(render(placements_for(chain(scene)), tile="c"))
    assert "x→ c" in text
    assert "└─" not in text, "tile mode is a single line, not a tree"


# --- the graph must match what build_coordinates actually does ----------------
def test_placement_matches_the_coordinates_that_get_built(scene):
    from multi_nd2_stitching.coordinates import build_coordinates
    from multi_nd2_stitching.offsets import build_plan
    from multi_nd2_stitching.store import Offset, OffsetStore

    lay = scene(
        {n: {"start": [0, 0], **(ANCHOR if n == "a" else {})} for n in "abcd"},
        SQUARE,
    )
    plan = build_plan(lay, grid_meta(SQUARE, lay.config.files, nt=3))
    store = OffsetStore()
    for t in plan.time_tasks:
        store.put(t, Offset(0, 0, 1))
    for p in plan.pair_tasks:
        store.put(p, Offset(0, 0, 5))

    coords = build_coordinates(lay, plan, store)
    for t in range(lay.nt):
        assert set(coords.at(t)) == {s.tile for s in plan_placement(lay, t).steps}


# --- handover between anchors -------------------------------------------------
def test_two_anchors_at_a_handover_are_ambiguous_without_unanchor(scene):
    lay = scene(
        {"a": {"start": [0, 0], **ANCHOR}, "b": {"start": [0, 0]}},
        {"a": (0.0, 0.0), "b": (55.0, 0.0)},
        overrides=[{"at": 1, "anchor": ["b"]}],
    )
    assert plan_placement(lay, 1).ambiguous


def test_unanchor_resolves_the_handover(scene):
    lay = scene(
        {"a": {"start": [0, 0], **ANCHOR}, "b": {"start": [0, 0]}},
        {"a": (0.0, 0.0), "b": (55.0, 0.0)},
        overrides=[{"at": 1, "unanchor": ["a"], "anchor": ["b"]}],
    )
    p = plan_placement(lay, 1)
    assert not p.ambiguous
    assert [s.tile for s in p.seeds] == ["b"]
    assert p.by_tile["a"].via == "b", "a is now placed off its neighbour"


def test_the_handover_is_a_single_timepoint(scene):
    """b anchors only at t=1; a takes the drift back afterwards."""
    lay = scene(
        {"a": {"start": [0, 0], **ANCHOR}, "b": {"start": [0, 0]}},
        {"a": (0.0, 0.0), "b": (55.0, 0.0)},
        overrides=[{"at": 1, "unanchor": ["a"], "anchor": ["b"]}],
    )
    assert [s.tile for s in plan_placement(lay, 1).seeds] == ["b"]
    assert [s.tile for s in plan_placement(lay, 2).seeds] == ["a"]
    assert plan_placement(lay, 2).by_tile["b"].via == "a"

import attrs
import numpy as np
import pytest
import zarr
from helpers import build, make_meta

from multi_nd2_stitching.blend import (
    BlendLog,
    CanvasGeometry,
    CanvasMismatch,
    bbox_of,
    blend,
    blend_weights,
    compose_timepoint,
    resolve_geometry,
    union_bbox,
    write_with_retry,
)
from multi_nd2_stitching.coordinates import build_coordinates
from multi_nd2_stitching.offsets import VolumeRef, build_plan
from multi_nd2_stitching.store import Offset, OffsetStore


class FlatReader:
    """Each tile is a constant plane, so blended values are easy to reason about."""

    def __init__(self, value_by_pos=None, shape=(2, 8, 8)):
        self.shape = shape
        self.value_by_pos = value_by_pos or {0: 100.0, 1: 200.0}
        self.reads = 0

    def read(self, ref):
        self.reads += 1
        return np.full(self.shape, self.value_by_pos[ref.position], dtype=np.float32)


@pytest.fixture
def scene(cfg_dict, tmp_path):
    """Two 8x8 tiles overlapping by 3 px in x, offsets already in the store."""
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=2, nz=2, ny=8, nx=8, paths=files)
    lay = build(cfg_dict, n_files=2, nt=2, nz=2, ny=8, nx=8, paths=files)
    plan = build_plan(lay, meta)
    store = OffsetStore()
    for t in plan.time_tasks:
        store.put(t, Offset(0, 0, 0))
    for p in plan.pair_tasks:
        store.put(p, Offset(0, 0, 5))
    coords = build_coordinates(lay, plan, store)
    refs = {
        (t, n): VolumeRef(
            "f0", lay.tile[n].position[lay.locate(t)[0]], lay.locate(t)[1], lay.nz
        )
        for t in range(lay.nt)
        for n in coords.at(t)
    }
    geom = CanvasGeometry.required(coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16")
    return lay, coords, refs, tmp_path, geom


# --- weights ------------------------------------------------------------------
def test_weights_are_flat_without_neighbours(scene):
    lay, coords, _, _, _geom = scene
    w = blend_weights("tile_a", 0, coords, [], (lay.nz, lay.ny, lay.nx))
    assert np.all(w == 1.0)


def test_weights_ramp_down_towards_the_neighbour(scene):
    lay, coords, _, _, _geom = scene
    p = lay.pairs[0]
    w = blend_weights(p.a, 0, coords, lay.pairs_at(0), (lay.nz, lay.ny, lay.nx))
    row = w[0]
    assert row[-1] < row[0], "the edge nearest the neighbour must weigh least"


def test_the_two_sides_ramp_oppositely(scene):
    lay, coords, _, _, _geom = scene
    p = lay.pairs[0]
    shape = (lay.nz, lay.ny, lay.nx)
    wa = blend_weights(p.a, 0, coords, lay.pairs_at(0), shape)[0]
    wb = blend_weights(p.b, 0, coords, lay.pairs_at(0), shape)[0]
    assert wa[-1] < wa[0] and wb[0] < wb[-1]


def test_degenerate_placement_leaves_weights_flat(scene):
    """A nonsense offset must not raise or produce a negative-length ramp."""
    lay, coords, _, _, _geom = scene
    bad = type(coords)(
        ({"tile_a": np.zeros(3), "tile_b": np.array([0, 0, 999])},) * lay.nt
    )
    w = blend_weights("tile_a", 0, bad, lay.pairs_at(0), (lay.nz, lay.ny, lay.nx))
    assert np.all(w == 1.0)


# --- composing ----------------------------------------------------------------
def test_solo_region_keeps_its_value(scene):
    """Normalisation means a pixel covered by one tile is unchanged."""
    lay, coords, refs, _, geom = scene
    img, _inds = compose_timepoint(0, lay, coords, FlatReader(), geom, refs)
    # tile_b (position 1, value 200) sits at x=-5, so it owns the left edge
    assert img[0, 0, 0] == pytest.approx(200.0)
    assert img[0, 0, -1] == pytest.approx(100.0)


def test_overlap_lies_between_the_two_values(scene):
    lay, coords, refs, _, geom = scene
    img, _ = compose_timepoint(0, lay, coords, FlatReader(), geom, refs)
    overlap = img[0, 0, 5:8]  # world x 0..2, covered by both tiles
    assert np.all(overlap >= 100.0) and np.all(overlap <= 200.0)
    assert not np.all(overlap == overlap[0]), "the overlap should be a gradient"


def test_mask_marks_only_covered_pixels(scene):
    lay, coords, refs, _, geom = scene
    _, inds = compose_timepoint(0, lay, coords, FlatReader(), geom, refs)
    assert inds.all(), "two tiles 5px apart should tile the whole canvas"


# --- the resume log -----------------------------------------------------------
def test_log_records_and_reloads(tmp_path):
    p = tmp_path / "c.blended"
    BlendLog(p).mark(3, "abc", [[0, 1], [0, 2], [0, 3]])
    reloaded = BlendLog(p)
    assert reloaded.is_done(3, "abc")
    assert not reloaded.is_done(3, "different")
    assert reloaded.bbox(3) == [[0, 1], [0, 2], [0, 3]]


def test_log_key_changes_with_coordinates(scene):
    _lay, coords, _, _tmp_path, geom = scene
    log = BlendLog(None)
    moved = type(coords)(
        tuple(
            {n: c + np.array([0, 0, 1]) for n, c in frame.items()}
            for frame in coords.by_time
        )
    )
    assert log.key(0, coords, geom) != log.key(0, moved, geom)


def test_log_key_changes_with_dtype(scene):
    _lay, coords, _, _, geom = scene
    other = attrs.evolve(geom, dtype="float32")
    assert BlendLog(None).key(0, coords, geom) != BlendLog(None).key(0, coords, other)


def test_log_key_ignores_a_wider_canvas(scene):
    """The whole point: growing the canvas must not invalidate placed timepoints."""
    _lay, coords, _, _, geom = scene
    wider = attrs.evolve(
        geom,
        shape=(geom.shape[0], geom.shape[1], geom.shape[2] + 500, geom.shape[3] + 500),
    )
    assert BlendLog(None).key(0, coords, geom) == BlendLog(None).key(0, coords, wider)


def test_log_key_changes_with_origin(scene):
    """A moved origin remaps every pixel, so it must invalidate."""
    _lay, coords, _, _, geom = scene
    shifted = attrs.evolve(geom, origin=(0, 0, geom.origin[2] - 10))
    assert BlendLog(None).key(0, coords, geom) != BlendLog(None).key(0, coords, shifted)


# --- retries ------------------------------------------------------------------
def test_retry_succeeds_after_transient_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("mount went away")
        return "ok"

    assert write_with_retry(flaky, attempts=4, delay=0) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_and_reraises():
    def always():
        raise OSError("gone for good")

    with pytest.raises(OSError, match="gone for good"):
        write_with_retry(always, attempts=2, delay=0)


# --- writing ------------------------------------------------------------------
def test_blend_writes_a_canvas(scene):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    n = blend(lay, coords, FlatReader(), refs, out, BlendLog(None), geom)
    assert n == lay.nt
    arr = zarr.open(str(out), mode="r")
    assert arr.shape == (lay.nt, lay.nz, lay.ny, lay.nx + 5)
    assert arr[0, 0, 0, 0] == 200
    assert arr[0, 0, 0, -1] == 100


def test_second_blend_skips_finished_timepoints(scene):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    log = BlendLog(tmp_path / "c.blended")
    assert blend(lay, coords, FlatReader(), refs, out, log, geom) == lay.nt
    reader = FlatReader()
    assert (
        blend(lay, coords, reader, refs, out, BlendLog(tmp_path / "c.blended"), geom)
        == 0
    )
    assert reader.reads == 0


def test_force_rewrites(scene):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    log = BlendLog(tmp_path / "c.blended")
    blend(lay, coords, FlatReader(), refs, out, log, geom)
    assert (
        blend(
            lay,
            coords,
            FlatReader(),
            refs,
            out,
            BlendLog(tmp_path / "c.blended"),
            geom,
            force=True,
        )
        == lay.nt
    )


def test_between_writes_only_that_window(scene):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    log = BlendLog(tmp_path / "c.blended")
    assert blend(lay, coords, FlatReader(), refs, out, log, geom, t0=1, t1=2) == 1
    assert log.is_done(1, log.key(1, coords, geom))
    assert not log.is_done(0, log.key(0, coords, geom))


def test_a_crash_costs_one_timepoint(scene, monkeypatch):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    logpath = tmp_path / "c.blended"

    calls = {"n": 0}
    real = compose_timepoint

    def boom(t, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("mount died")
        return real(t, *a, **kw)

    monkeypatch.setattr("multi_nd2_stitching.blend.compose_timepoint", boom)
    with pytest.raises(OSError):
        blend(lay, coords, FlatReader(), refs, out, BlendLog(logpath), geom)
    assert BlendLog(logpath).is_done(0, BlendLog(logpath).key(0, coords, geom))


# --- geometry is fixed once created -------------------------------------------
def test_geometry_is_persisted_and_reused(scene):
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    blend(lay, coords, FlatReader(), refs, out, BlendLog(None), geom)
    again, is_new = resolve_geometry(
        out, coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", 0, lay.nt
    )
    assert not is_new
    assert again == geom


def test_a_wider_extent_reuses_the_existing_origin(scene):
    """Recomputing an offset can widen the extent. The origin must not move,
    or every timepoint already written silently shifts."""
    lay, coords, _refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    big = attrs.evolve(
        geom,
        shape=(geom.shape[0], geom.shape[1], geom.shape[2] + 20, geom.shape[3] + 20),
    )
    big.save(out)
    resolved, is_new = resolve_geometry(
        out, coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", 0, lay.nt
    )
    assert not is_new
    assert resolved.origin == big.origin
    assert resolved.spatial == big.spatial, "the oversized frame is kept"


def test_an_oversized_canvas_reports_its_slack(scene):
    lay, coords, _, _, geom = scene
    big = attrs.evolve(
        geom,
        shape=(geom.shape[0], geom.shape[1] + 4, geom.shape[2] + 8, geom.shape[3] + 16),
    )
    assert big.slack(coords, (lay.nz, lay.ny, lay.nx), lay.nt) == (4, 8, 16)


def test_tiles_outside_the_existing_canvas_are_refused(scene):
    lay, coords, _refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    moved = type(coords)(
        tuple(
            {
                n: (c + np.array([0, 0, 500]) if n == "tile_b" else c)
                for n, c in frame.items()
            }
            for frame in coords.by_time
        )
    )
    with pytest.raises(CanvasMismatch, match="no longer fit"):
        resolve_geometry(
            out, moved, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", 0, lay.nt
        )


def test_recreate_derives_a_fresh_tight_geometry(scene):
    lay, coords, _refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    attrs.evolve(geom, shape=(geom.shape[0], 99, 99, 99)).save(out)
    fresh, is_new = resolve_geometry(
        out,
        coords,
        (lay.nz, lay.ny, lay.nx),
        lay.nt,
        "uint16",
        0,
        lay.nt,
        recreate=True,
    )
    assert is_new
    assert fresh == geom


def test_dtype_change_is_refused(scene):
    lay, coords, _, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    with pytest.raises(CanvasMismatch, match="was written as uint16"):
        resolve_geometry(
            out, coords, (lay.nz, lay.ny, lay.nx), lay.nt, "float32", 0, lay.nt
        )


# --- erase on rewrite ---------------------------------------------------------
def test_bbox_of_a_partial_mask():
    inds = np.zeros((4, 8, 8), dtype=bool)
    inds[1:3, 2:5, 6:7] = True
    assert bbox_of(inds) == [[1, 3], [2, 5], [6, 7]]


def test_bbox_of_nothing_is_none():
    assert bbox_of(np.zeros((2, 2, 2), dtype=bool)) is None


def test_union_bbox():
    assert union_bbox([[0, 2], [0, 2], [0, 2]], [[1, 5], [0, 1], [3, 4]]) == [
        [0, 5],
        [0, 2],
        [0, 4],
    ]
    assert union_bbox(None, [[1, 2], [1, 2], [1, 2]]) == [[1, 2], [1, 2], [1, 2]]


def test_rewrite_erases_the_old_footprint(scene):
    """A tile that moves must not leave its old pixels behind."""
    import zarr

    lay, coords, refs, tmp_path, _ = scene
    out = tmp_path / "canvas.zarr"
    logpath = tmp_path / "c.blended"
    # a canvas wide enough for the tile to move within
    wide = CanvasGeometry(
        origin=(0, 0, -5), shape=(lay.nt, lay.nz, lay.ny, 30), dtype="uint16"
    )
    only_a = type(coords)(tuple({"tile_a": np.zeros(3)} for _ in coords.by_time))
    blend(lay, only_a, FlatReader(), refs, out, BlendLog(logpath), wide)
    arr = zarr.open(str(out), mode="r")
    assert arr[0, 0, 0, 5] == 100  # tile_a occupies canvas x 5..12

    moved = type(coords)(
        tuple({"tile_a": np.array([0, 0, 15])} for _ in coords.by_time)
    )
    blend(lay, moved, FlatReader(), refs, out, BlendLog(logpath), wide)
    arr = zarr.open(str(out), mode="r")
    assert arr[0, 0, 0, 20] == 100  # new home
    assert arr[0, 0, 0, 5] == 0, "the old footprint must have been cleared"

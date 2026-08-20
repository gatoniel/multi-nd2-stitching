import attrs
import numpy as np
import pytest
import zarr
from helpers import build, make_meta, stub_files

from multi_nd2_stitching.blend import (
    BlendLog,
    CanvasGeometry,
    CanvasMismatch,
    bbox_of,
    blend,
    blend_weights,
    boxes_bbox,
    compose_timepoint,
    hits_a_box,
    load_timepoint,
    resolve_geometry,
    snap_to_chunks,
    tile_boxes,
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
    files = stub_files(tmp_path, 2)
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
def _compose(t, lay, coords, reader, geom, chunk=(1, 1, 1)):
    """Drive the new compose the way blend() does."""
    boxes = tile_boxes(t, lay, coords, geom)
    region = snap_to_chunks(boxes_bbox(boxes), chunk, geom.spatial)
    if region is None:
        return None, boxes, None
    volumes = load_timepoint(reader, {(t, n): n for n in boxes}, t, list(boxes))
    image = compose_timepoint(t, lay, coords, volumes, geom, region, boxes)
    return image, boxes, region


class NamedReader(FlatReader):
    """load_timepoint passes tile names as refs here, so map on the name."""

    def read(self, ref):
        self.reads += 1
        return np.full(self.shape, self.value_by_name[ref], dtype=np.float32)

    def __init__(self, value_by_name=None, shape=(2, 8, 8)):
        super().__init__(shape=shape)
        self.value_by_name = value_by_name or {"tile_a": 100.0, "tile_b": 200.0}


def test_solo_region_keeps_its_value(scene):
    """Normalisation means a pixel covered by one tile is unchanged."""
    lay, coords, _refs, _, geom = scene
    img, _boxes, _region = _compose(0, lay, coords, NamedReader(), geom)
    # tile_b (200) sits at x=-5, so it owns the left edge
    assert img[0, 0, 0] == pytest.approx(200.0)
    assert img[0, 0, -1] == pytest.approx(100.0)


def test_overlap_lies_between_the_two_values(scene):
    lay, coords, _refs, _, geom = scene
    img, _, _ = _compose(0, lay, coords, NamedReader(), geom)
    overlap = img[0, 0, 5:8]  # world x 0..2, covered by both tiles
    assert np.all(overlap >= 100.0) and np.all(overlap <= 200.0)
    assert not np.all(overlap == overlap[0]), "the overlap should be a gradient"


def test_boxes_cover_the_whole_canvas_here(scene):
    lay, coords, _refs, _, geom = scene
    _, boxes, _ = _compose(0, lay, coords, NamedReader(), geom)
    assert boxes_bbox(boxes) == [[0, lay.nz], [0, lay.ny], [0, lay.nx + 5]]


def test_uncovered_voxels_stay_zero(scene):
    """No mask array any more -- the divisor is clamped instead."""
    lay, coords, _refs, _, _geom = scene
    wide = CanvasGeometry(
        origin=(0, 0, -5), shape=(lay.nt, lay.nz, lay.ny, 40), dtype="uint16"
    )
    only_a = type(coords)(tuple({"tile_a": np.zeros(3)} for _ in coords.by_time))
    # a chunk grid coarser than the tile, so the region extends past it
    img, _boxes, region = _compose(
        0, lay, only_a, NamedReader(), wide, chunk=(2, 8, 32)
    )
    assert region[2] == [0, 32], "region snapped out beyond the tile"
    assert img[0, 0, 0] == 0.0, "before the tile: never covered, stays zero"
    assert img[0, 0, -1] == 0.0, "past the tile: same"
    assert img[0, 0, 6] == pytest.approx(100.0), "inside the tile"


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


# --- write layout -------------------------------------------------------------
def test_snap_to_chunks_grows_to_boundaries():
    from multi_nd2_stitching.blend import snap_to_chunks

    assert snap_to_chunks(
        [[5, 40], [7, 600], [13, 20]], (32, 512, 512), (64, 1024, 1024)
    ) == [[0, 64], [0, 1024], [0, 512]]


def test_snap_to_chunks_clips_to_the_canvas():
    from multi_nd2_stitching.blend import snap_to_chunks

    assert snap_to_chunks(
        [[0, 50], [0, 10], [0, 10]], (32, 512, 512), (50, 10, 10)
    ) == [
        [0, 50],
        [0, 10],
        [0, 10],
    ]


def test_snap_to_chunks_passes_none_through():
    from multi_nd2_stitching.blend import snap_to_chunks

    assert snap_to_chunks(None, (32, 512, 512), (64, 64, 64)) is None


def test_overlaps():
    from multi_nd2_stitching.blend import overlaps

    sl = (slice(0, 10), slice(0, 10), slice(0, 10))
    assert overlaps(sl, [[5, 8], [5, 8], [5, 8]])
    assert not overlaps(sl, [[20, 30], [0, 5], [0, 5]])
    assert not overlaps(sl, None)


def test_every_write_starts_on_a_chunk_boundary(scene, monkeypatch):
    """Unaligned writes force zarr into read-modify-write on partial chunks."""
    lay, coords, refs, tmp_path, geom = scene
    out = tmp_path / "canvas.zarr"
    chunk = (1, 2, 4, 4)
    starts = []
    import zarr

    real = zarr.open

    class Spy:
        def __init__(self, inner):
            self._inner = inner
            self.shape = inner.shape

        def __setitem__(self, key, value):
            starts.append(tuple(s.start for s in key[1:]))
            self._inner[key] = value

    monkeypatch.setattr(zarr, "open", lambda *a, **k: Spy(real(*a, **k)))
    blend(lay, coords, FlatReader(), refs, out, BlendLog(None), geom, chunk=chunk)
    assert starts
    for z, y, x in starts:
        assert (z % chunk[1], y % chunk[2], x % chunk[3]) == (0, 0, 0), (z, y, x)


def test_empty_chunks_are_skipped(scene, monkeypatch):
    lay, coords, refs, tmp_path, _ = scene
    out = tmp_path / "canvas.zarr"
    # a canvas far wider than the tiles, so most chunks are empty
    wide = CanvasGeometry(
        origin=(0, 0, -5), shape=(lay.nt, lay.nz, lay.ny, 64), dtype="uint16"
    )
    only_a = type(coords)(tuple({"tile_a": np.zeros(3)} for _ in coords.by_time))
    writes = []
    import zarr

    real = zarr.open

    class Spy:
        def __init__(self, inner):
            self._inner = inner
            self.shape = inner.shape

        def __setitem__(self, key, value):
            writes.append(key)
            self._inner[key] = value

    monkeypatch.setattr(zarr, "open", lambda *a, **k: Spy(real(*a, **k)))
    blend(
        lay, only_a, FlatReader(), refs, out, BlendLog(None), wide, chunk=(1, 2, 4, 4)
    )
    per_t = len(writes) / lay.nt
    assert per_t < (2 * 2 * 16), (
        f"wrote {per_t} chunks/timepoint on a mostly empty canvas"
    )


# --- cost is proportional to the data, not the canvas -------------------------
def test_buffers_are_sized_to_the_region_not_the_canvas(scene):
    """A padded canvas must not make the per-timepoint work bigger."""
    lay, coords, _refs, _, _geom = scene
    wide = CanvasGeometry(
        origin=(0, 0, -5), shape=(lay.nt, lay.nz, lay.ny, 400), dtype="uint16"
    )
    img, _, region = _compose(0, lay, coords, NamedReader(), wide)
    assert img.size < lay.nz * lay.ny * 400, "allocated the whole canvas"
    assert img.shape == tuple(int(r[1] - r[0]) for r in region)


def test_the_box_is_computed_without_touching_pixels(scene):
    """tile_boxes is analytic; nothing is read to find the covered region."""
    lay, coords, _refs, _, geom = scene
    reader = NamedReader()
    boxes = tile_boxes(0, lay, coords, geom)
    assert reader.reads == 0
    assert set(boxes) == set(lay.tiles_at(0))


def test_an_empty_timepoint_has_no_box(scene):
    lay, coords, _refs, _, geom = scene
    empty = type(coords)(tuple({} for _ in coords.by_time))
    assert tile_boxes(0, lay, empty, geom) == {}
    assert boxes_bbox({}) is None


def test_hits_a_box(scene):
    boxes = {"a": ((0, 0, 0), (2, 4, 4))}
    assert hits_a_box((slice(0, 1), slice(0, 2), slice(0, 2)), boxes)
    assert not hits_a_box((slice(0, 1), slice(8, 9), slice(0, 2)), boxes)
    assert not hits_a_box((slice(0, 1), slice(0, 2), slice(0, 2)), {})


def test_weights_cache_returns_the_same_array(scene):
    lay, coords, _, _, _geom = scene
    cache = {}
    shape = (lay.nz, lay.ny, lay.nx)
    a = blend_weights("tile_a", 0, coords, lay.pairs_at(0), shape, cache)
    b = blend_weights("tile_a", 1, coords, lay.pairs_at(1), shape, cache)
    assert a is b, "same placement, same weights -- should be a lookup"
    assert len(cache) == 1


# --- padding ------------------------------------------------------------------
def test_scalar_pad_leaves_z_alone(scene):
    """Padding z multiplies the canvas volume for no benefit: drift in z is tiny."""
    lay, coords, _, _, geom = scene
    padded = CanvasGeometry.required(
        coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", pad=10
    )
    assert padded.spatial[0] == geom.spatial[0]
    assert padded.spatial[1] == geom.spatial[1] + 20
    assert padded.spatial[2] == geom.spatial[2] + 20
    assert padded.origin[0] == geom.origin[0]


def test_three_values_pad_each_axis(scene):
    lay, coords, _, _, geom = scene
    padded = CanvasGeometry.required(
        coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", pad=(1, 2, 3)
    )
    assert padded.spatial == tuple(
        a + 2 * b for a, b in zip(geom.spatial, (1, 2, 3), strict=False)
    )
    assert padded.origin == tuple(
        a - b for a, b in zip(geom.origin, (1, 2, 3), strict=False)
    )


def test_no_padding_by_default(scene):
    """The default frame is exactly the extent of the placed tiles."""
    lay, coords, _, _, _geom = scene
    tile = (lay.nz, lay.ny, lay.nx)
    ext = coords.extent(tile)
    default = CanvasGeometry.required(coords, tile, lay.nt, "uint16")
    assert default.origin == tuple(int(v) for v in ext[:, 0])
    assert default.spatial == tuple(int(v) for v in (ext[:, 1] - ext[:, 0]))
    assert default.slack(coords, tile, lay.nt) == (0, 0, 0)


# --- malformed sidecars -------------------------------------------------------
def test_a_corrupt_geometry_sidecar_is_fatal(scene, tmp_path):
    """Returning None here would derive a fresh frame and remap what is written."""
    _lay, _coords, _refs, _, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    CanvasGeometry.path_for(out).write_text("{ not json")
    with pytest.raises(CanvasMismatch, match="cannot be read"):
        CanvasGeometry.load(out)


@pytest.mark.parametrize(
    "body",
    [
        '{"origin": [0, 0]}',  # missing keys
        '{"origin": 5, "shape": 5, "dtype": "u2"}',  # wrong types
    ],
)
def test_a_malformed_geometry_sidecar_is_fatal(scene, tmp_path, body):
    _lay, _coords, _refs, _, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    CanvasGeometry.path_for(out).write_text(body)
    with pytest.raises(CanvasMismatch):
        CanvasGeometry.load(out)


def test_a_missing_sidecar_is_not_an_error(tmp_path):
    assert CanvasGeometry.load(tmp_path / "never-written.zarr") is None


def test_resolve_geometry_surfaces_a_corrupt_sidecar(scene, tmp_path):
    lay, coords, _refs, _, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    CanvasGeometry.path_for(out).write_text("garbage")
    with pytest.raises(CanvasMismatch, match="--recreate"):
        resolve_geometry(
            out, coords, (lay.nz, lay.ny, lay.nx), lay.nt, "uint16", 0, lay.nt
        )


def test_recreate_ignores_a_corrupt_sidecar(scene, tmp_path):
    """--recreate is the documented escape hatch, so it must not trip over it."""
    lay, coords, _refs, _, geom = scene
    out = tmp_path / "canvas.zarr"
    geom.save(out)
    CanvasGeometry.path_for(out).write_text("garbage")
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
    assert is_new and fresh == geom


def test_a_torn_blend_log_line_warns_and_is_skipped(tmp_path):
    p = tmp_path / "c.blended"
    log = BlendLog(p)
    log.mark(0, "abc", [[0, 1], [0, 1], [0, 1]])
    with p.open("a") as f:
        f.write('{"t": 1, "ke')
    with pytest.warns(RuntimeWarning, match="skipped 1 unreadable line"):
        reloaded = BlendLog(p)
    assert reloaded.is_done(0, "abc")
    assert reloaded.skipped == 1
    assert not reloaded.written(1)


def test_a_clean_blend_log_warns_about_nothing(tmp_path, recwarn):
    p = tmp_path / "c.blended"
    BlendLog(p).mark(0, "abc", None)
    BlendLog(p)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --- pipelining ---------------------------------------------------------------
def test_pipelined_and_serial_agree(scene, tmp_path):
    """Overlapping the phases must not change a single voxel."""
    lay, coords, refs, _, geom = scene
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    blend(lay, coords, FlatReader(), refs, a, BlendLog(None), geom, pipeline=False)
    blend(lay, coords, FlatReader(), refs, b, BlendLog(None), geom, pipeline=True)
    assert np.array_equal(
        zarr.open(str(a), mode="r")[:], zarr.open(str(b), mode="r")[:]
    )


@pytest.mark.parametrize("writers", [1, 2, 8])
def test_parallel_writers_agree(scene, tmp_path, writers):
    lay, coords, refs, _, geom = scene
    ref = tmp_path / "ref.zarr"
    out = tmp_path / f"w{writers}.zarr"
    blend(lay, coords, FlatReader(), refs, ref, BlendLog(None), geom, writers=1)
    blend(lay, coords, FlatReader(), refs, out, BlendLog(None), geom, writers=writers)
    assert np.array_equal(
        zarr.open(str(ref), mode="r")[:], zarr.open(str(out), mode="r")[:]
    )


def test_timings_are_recorded(scene, tmp_path):
    from multi_nd2_stitching.blend import Timings

    lay, coords, refs, _, geom = scene
    timings = Timings()
    blend(
        lay,
        coords,
        FlatReader(),
        refs,
        tmp_path / "c.zarr",
        BlendLog(None),
        geom,
        timings=timings,
    )
    d = timings.as_dict()
    assert d["total_s"] >= 0.0
    assert {"read_s", "compose_s", "write_s", "total_s"} <= set(d)
    # exactly one of the two slack keys, never both
    assert ("overlap_s" in d) != ("other_s" in d)


def test_every_timepoint_is_marked_even_when_pipelined(scene, tmp_path):
    lay, coords, refs, _, geom = scene
    logpath = tmp_path / "c.blended"
    blend(
        lay,
        coords,
        FlatReader(),
        refs,
        tmp_path / "c.zarr",
        BlendLog(logpath),
        geom,
        pipeline=True,
    )
    reloaded = BlendLog(logpath)
    for t in range(lay.nt):
        assert reloaded.is_done(t, reloaded.key(t, coords, geom))


def test_a_write_failure_still_propagates_when_pipelined(scene, tmp_path):
    lay, coords, refs, _, geom = scene

    class Boom(FlatReader):
        def read(self, ref):
            raise OSError("mount died")

    with pytest.raises(OSError, match="mount died"):
        blend(
            lay,
            coords,
            Boom(),
            refs,
            tmp_path / "c.zarr",
            BlendLog(None),
            geom,
            pipeline=True,
        )

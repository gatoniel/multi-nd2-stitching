import numpy as np
import pytest
import scipy.fft as spfft
from helpers import FakeReader

from multi_nd2_stitching import compute as C
from multi_nd2_stitching.compute import (
    Spectra,
    crop_for_alignment,
    fft_translation_3d,
    phase_corr_from_ffts,
    run_plan,
    run_task,
    to_signed_shift,
)
from multi_nd2_stitching.offsets import CornerTask, Crop, PairTask, TimeTask, VolumeRef
from multi_nd2_stitching.store import Offset, OffsetStore


def vol(seed=0, shape=(16, 64, 64)):
    return np.random.default_rng(seed).random(shape)


# --- the correlation actually recovers a known shift --------------------------
@pytest.mark.parametrize("shift", [(0, 0, 0), (0, 5, 0), (0, 0, 7), (2, -3, 4)])
def test_recovers_a_known_translation(shift):
    a = vol()
    b = np.roll(a, shift, axis=(0, 1, 2))
    assert tuple(fft_translation_3d(a, b)) == tuple(-s for s in shift)


def test_identical_volumes_give_zero():
    a = vol()
    assert tuple(fft_translation_3d(a, a)) == (0, 0, 0)


# --- an exact-zero cross-power bin must not poison the whole correlation ------
def test_survives_a_zeroed_spectrum_bin(recwarn):
    """One exactly-zero bin (e.g. a flat/blank patch in the crop) used to turn
    the *entire* correlation into NaN and silently return (0, 0, 0)."""
    a = vol()
    shift = (0, 2, 3)
    b = np.roll(a, shift, axis=(0, 1, 2))
    fft0 = spfft.rfftn(a)
    fft1 = spfft.rfftn(b)
    fft0[1, 2, 3] = 0  # an otherwise-clean pair with one dead bin

    result = tuple(phase_corr_from_ffts(fft0, fft1, a.shape))

    assert result == tuple(-s for s in shift)
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


# --- shaped_peak: reject artifact peaks by shape, not raw height --------------
def test_neighbour_decay_isotropic_bump_scores_positive_in_every_axis():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    arr[3, 4, 4] = arr[5, 4, 4] = 3.0
    arr[4, 3, 4] = arr[4, 5, 4] = 3.0
    arr[4, 4, 3] = arr[4, 4, 5] = 3.0
    assert C._neighbour_decay(arr, (4, 4, 4)) == pytest.approx(2.0)


def test_neighbour_decay_line_artifact_scores_zero():
    """Bright the whole length of the z axis at one (y, x) -- a stuck-column-
    style defect. It doesn't decay along its own axis at all."""
    arr = np.zeros((8, 8, 8))
    arr[:, 4, 4] = 10.0
    assert C._neighbour_decay(arr, (4, 4, 4)) == pytest.approx(0.0)


def test_neighbour_decay_wraps_around():
    """The response is a circular correlation surface -- index 0's neighbour
    on one side is index -1, not out of bounds."""
    arr = np.zeros((8, 8, 8))
    arr[0, 4, 4] = 5.0
    arr[7, 4, 4] = arr[1, 4, 4] = 3.0  # the wrapped z-neighbours of index 0
    assert C._neighbour_decay(arr, (0, 4, 4)) == pytest.approx(2.0)


def test_neighbour_decay_uses_the_brighter_neighbour_not_the_average():
    """A point at the edge of a streak (background on one side, more streak
    on the other) must not look decayed just because the average of a bright
    and a dark neighbour happens to be low -- the streak side alone should
    count."""
    arr = np.zeros((8, 8, 8))
    arr[1:4, 2, 2] = 8.0  # a 3-long ridge
    # (1, 2, 2) is the ridge's edge: z-neighbours are 0 (background) and 8.0
    # (more ridge). The *average* would suggest some z-decay; it must not.
    assert C._neighbour_decay(arr, (1, 2, 2)) == pytest.approx(0.0)


def test_shaped_peak_index_prefers_the_isotropic_bump_over_a_taller_ridge():
    """The scenario the override exists for: an artifact taller than the real
    peak, but shaped like a streak rather than a point."""
    arr = np.zeros((8, 8, 8))
    arr[1:4, 2, 2] = 8.0  # taller than the real peak, but a flat ridge
    arr[5, 5, 5] = 6.0  # the real, isotropically-decaying peak
    arr[4, 5, 5] = arr[6, 5, 5] = 4.0
    arr[5, 4, 5] = arr[5, 6, 5] = 4.0
    arr[5, 5, 4] = arr[5, 5, 6] = 4.0

    # Sanity: naive argmax lands on the ridge, not the real peak.
    assert np.unravel_index(np.argmax(arr), arr.shape) in {
        (1, 2, 2),
        (2, 2, 2),
        (3, 2, 2),
    }
    assert C._shaped_peak_index(arr, candidates=5) == (5, 5, 5)


def test_shaped_peak_index_with_no_artifact_still_finds_the_true_peak():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    arr[3, 4, 4] = arr[5, 4, 4] = 3.0
    arr[4, 3, 4] = arr[4, 5, 4] = 3.0
    arr[4, 4, 3] = arr[4, 4, 5] = 3.0
    assert C._shaped_peak_index(arr) == (4, 4, 4)


def test_phase_corr_from_ffts_shaped_mode_still_recovers_a_clean_translation():
    """No artifact present: shaped mode must agree with the default path."""
    a = vol()
    shift = (1, -2, 3)
    b = np.roll(a, shift, axis=(0, 1, 2))
    fft0, fft1 = spfft.rfftn(a), spfft.rfftn(b)
    plain = tuple(phase_corr_from_ffts(fft0, fft1, a.shape, shaped=False))
    shaped = tuple(phase_corr_from_ffts(fft0, fft1, a.shape, shaped=True))
    assert plain == shaped == tuple(-s for s in shift)


def test_phase_corr_from_ffts_shaped_true_uses_the_shaped_index(monkeypatch):
    """Wiring check: shaped=True actually calls the shaped selector, and its
    result goes through the same wrap-to-signed-shift arithmetic as argmax."""
    calls = []

    def fake_shaped_peak_index(inverse, candidates=C.SHAPED_PEAK_CANDIDATES):
        calls.append(inverse.shape)
        return (0, 0, 0)

    monkeypatch.setattr(C, "_shaped_peak_index", fake_shaped_peak_index)
    a = vol()
    fft0, fft1 = spfft.rfftn(a), spfft.rfftn(a)
    result = tuple(phase_corr_from_ffts(fft0, fft1, a.shape, shaped=True))
    assert result == (0, 0, 0)
    assert calls == [a.shape]


# --- near: restrict the search to a window around a manual hint ---------------
def test_windowed_peak_index_finds_the_hinted_peak_ignoring_a_taller_one_outside():
    arr = np.zeros((8, 8, 8))
    arr[1:4, 2, 2] = 8.0  # taller artifact, well outside the window below
    arr[5, 5, 5] = 6.0
    near = C.to_signed_shift((5, 5, 5), arr.shape)
    assert C._windowed_peak_index(arr, near, radius=2) == (5, 5, 5)


def test_windowed_peak_index_prefers_tallest_within_window_regardless_of_shape():
    """Unlike _shaped_peak_index, this is a plain argmax once windowed -- no
    shape filtering. A sharp spike beats a broader, more isotropic bump if
    it's simply taller, as long as both are inside the window."""
    arr = np.zeros((8, 8, 8))
    arr[5, 5, 5] = 6.0  # isotropic bump
    arr[4, 4, 4] = 4.0
    arr[6, 4, 4] = arr[4, 6, 4] = arr[4, 4, 6] = 3.0
    arr[3, 3, 3] = 9.0  # sharp, unshaped spike, still inside a radius-3 window
    near = C.to_signed_shift((5, 5, 5), arr.shape)
    assert C._windowed_peak_index(arr, near, radius=3) == (3, 3, 3)


def test_windowed_peak_index_wraps_around():
    arr = np.zeros((8, 8, 8))
    arr[7, 4, 4] = 5.0
    near = C.to_signed_shift((0, 4, 4), arr.shape)  # hint centred on index 0
    assert C._windowed_peak_index(arr, near, radius=2) == (7, 4, 4)


def test_phase_corr_from_ffts_near_overrides_shaped_and_argmax(monkeypatch):
    """The tmp_profiles finding, reproduced directly: a taller, better-shaped
    artifact beats the real (shorter, less isotropic) peak under both plain
    argmax and the decay-based `shaped` pick. Only a `near` hint pointed at
    the real peak, far enough away that the window excludes the artifact
    entirely, recovers it."""
    shape = (48, 48, 48)
    arr = np.zeros(shape)
    arr[6, 6, 6] = 8.0  # taller AND better-shaped artifact
    arr[5, 6, 6] = arr[7, 6, 6] = 5.0
    arr[6, 5, 6] = arr[6, 7, 6] = 5.0
    arr[6, 6, 5] = arr[6, 6, 7] = 5.0
    arr[30, 30, 30] = 5.0  # the real peak: shorter, decays less cleanly
    arr[29, 30, 30] = arr[31, 30, 30] = 4.0
    arr[30, 29, 30] = arr[30, 31, 30] = 4.0
    arr[30, 30, 29] = arr[30, 30, 31] = 4.0

    monkeypatch.setattr(C, "correlation_surface", lambda *a, **k: arr)
    assert np.unravel_index(np.argmax(arr), shape) == (6, 6, 6)
    assert C._shaped_peak_index(arr) == (6, 6, 6)  # even shaped is fooled

    near = to_signed_shift((30, 30, 30), shape)
    result = phase_corr_from_ffts(None, None, shape, shaped=True, near=near)
    assert tuple(result) == tuple(to_signed_shift((30, 30, 30), shape))


# --- candidate_peaks / axis_profile: the data stitch inspect exports ----------
def test_candidate_peaks_returns_the_requested_count_sorted_by_value():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    arr[1, 1, 1] = 4.0
    arr[2, 2, 2] = 3.0
    arr[3, 3, 3] = 2.0
    arr[6, 6, 6] = 1.0
    cands = C.candidate_peaks(arr, candidates=3)
    assert len(cands) == 3
    assert [c.value for c in cands] == [5.0, 4.0, 3.0]
    assert cands[0].index == (4, 4, 4)


def test_candidate_peaks_decay_matches_neighbour_decay():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    arr[3, 4, 4] = arr[5, 4, 4] = 3.0
    arr[4, 3, 4] = arr[4, 5, 4] = 3.0
    arr[4, 4, 3] = arr[4, 4, 5] = 3.0
    cands = C.candidate_peaks(arr, candidates=1)
    assert cands[0].decay == pytest.approx(C._neighbour_decay(arr, (4, 4, 4)))


def test_shaped_peak_index_agrees_with_max_decay_candidate():
    """A refactor, not a behaviour change: _shaped_peak_index is exactly
    max(candidate_peaks(...), key=decay)."""
    arr = np.zeros((8, 8, 8))
    arr[1:4, 2, 2] = 8.0
    arr[5, 5, 5] = 6.0
    arr[4, 5, 5] = arr[6, 5, 5] = 4.0
    arr[5, 4, 5] = arr[5, 6, 5] = 4.0
    arr[5, 5, 4] = arr[5, 5, 6] = 4.0
    best = max(C.candidate_peaks(arr, candidates=5), key=lambda c: c.decay)
    assert C._shaped_peak_index(arr, candidates=5) == best.index


def test_axis_profile_length_and_centre():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    profile = C.axis_profile(arr, (4, 4, 4), axis=0, radius=3)
    assert len(profile) == 7  # 2*radius + 1
    assert profile[3] == 5.0  # the centre step (0)


def test_axis_profile_wraps_around():
    arr = np.zeros((8, 8, 8))
    arr[0, 4, 4] = 5.0
    arr[7, 4, 4] = 2.0  # wrapped neighbour "before" index 0
    profile = C.axis_profile(arr, (0, 4, 4), axis=0, radius=1)
    assert list(profile) == [2.0, 5.0, 0.0]  # step -1, 0, +1


def test_axis_profile_only_moves_along_the_requested_axis():
    arr = np.zeros((8, 8, 8))
    arr[4, 4, 4] = 5.0
    arr[4, 4, 5] = 9.0  # x-neighbour; must not leak into the z profile
    profile = C.axis_profile(arr, (4, 4, 4), axis=0, radius=1)
    assert list(profile) == [0.0, 5.0, 0.0]


# --- correlation_surface: the shared, zero-bin-safe response ------------------
def test_correlation_surface_matches_phase_corr_peak():
    a = vol()
    shift = (0, 3, -2)
    b = np.roll(a, shift, axis=(0, 1, 2))
    fft0, fft1 = spfft.rfftn(a), spfft.rfftn(b)
    surface = C.correlation_surface(fft0, fft1, a.shape)
    idx = np.unravel_index(np.argmax(surface), a.shape)
    assert tuple(C.to_signed_shift(idx, a.shape)) == tuple(-s for s in shift)


def test_correlation_surface_survives_a_zeroed_bin(recwarn):
    """The zero-bin fix lives here now; phase_corr_from_ffts is just a caller."""
    a = vol()
    fft0, fft1 = spfft.rfftn(a), spfft.rfftn(a)
    fft0[1, 2, 3] = 0
    surface = C.correlation_surface(fft0, fft1, a.shape)
    assert np.all(np.isfinite(surface))
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


def test_all_zero_volumes_do_not_warn(recwarn):
    z = np.zeros((4, 8, 8))
    fft_translation_3d(z, z)
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


# --- cropping -----------------------------------------------------------------
def test_crop_for_alignment_shapes_match():
    a, b = vol(), vol(1)
    ca, cb = crop_for_alignment(a, b, 2, 20)
    assert ca.shape == cb.shape == (16, 64, 44)


@pytest.mark.parametrize("shift_px", [0, 64, 200, -5])
def test_crop_rejects_impossible_shift(shift_px):
    with pytest.raises(ValueError, match="outside tile extent"):
        crop_for_alignment(vol(), vol(1), 2, shift_px)


# --- run_task -----------------------------------------------------------------
def _refs():
    return VolumeRef("f0", 0, 0, 16), VolumeRef("f0", 1, 0, 16)


def test_time_task_recovers_drift():
    src, dst = _refs()
    reader = FakeReader(shifts={dst: (0, 3, -2)})
    task = TimeTask(
        name="a",
        t_from=0,
        t_to=1,
        src=src,
        dst=dst,
        crop=Crop((None, None), (None, None), (None, None)),
    )
    assert run_task(task, Spectra(reader)) == Offset(0, -3, 2)


@pytest.mark.parametrize("extra", [0, 3, -4])
def test_pair_task_total_offset(extra):
    """b sits shift_px + `extra` to the right of a.

    crop_for_alignment lines up the nominal overlap, so the correlation only
    sees `extra`; shift_px is added back to give the total displacement.
    """
    src, dst = _refs()
    reader = FakeReader(shifts={dst: (0, 0, -(20 + extra))})
    task = PairTask(
        a="a",
        b="b",
        axis=2,
        t=0,
        src=src,
        dst=dst,
        crop=Crop((None, None), (None, None), (None, None)),
        shift_px=20,
    )
    out = run_task(task, Spectra(reader))
    assert (out.dz, out.dy, out.dx) == (0, 0, 20 + extra)


def test_crop_is_applied_before_correlation():
    src, dst = _refs()
    reader = FakeReader(shifts={dst: (0, 4, 0)})
    task = TimeTask(
        name="a",
        t_from=0,
        t_to=1,
        src=src,
        dst=dst,
        crop=Crop((2, 10), (None, None), (None, None)),
    )
    out = run_task(task, Spectra(reader))
    assert out.dy == -4  # y shift still recovered from the z-cropped stack


def test_run_task_converts_near_from_measured_to_raw_space_for_pair_task(monkeypatch):
    """`near` on a PairTask is measured (post shift_px) space; run_task must
    subtract shift_px back out before it means anything to the windowed
    search -- the mirror of the `+=` it does to the result."""
    captured = {}

    def fake_phase_corr(fft0, fft1, shape, workers=-1, shaped=False, near=None):
        captured["near"] = tuple(near) if near is not None else None
        return np.zeros(3, dtype=int)

    monkeypatch.setattr(C, "phase_corr_from_ffts", fake_phase_corr)
    src, dst = _refs()
    task = PairTask(
        a="a",
        b="b",
        axis=2,
        t=0,
        src=src,
        dst=dst,
        crop=Crop((None, None), (None, None), (None, None)),
        shift_px=20,
        near=(1, 2, 23),
    )
    run_task(task, Spectra(FakeReader()))
    assert captured["near"] == (1, 2, 3)


def test_run_task_leaves_near_unconverted_for_time_task(monkeypatch):
    """A TimeTask has no shift_px -- its `near` is already in raw space."""
    captured = {}

    def fake_phase_corr(fft0, fft1, shape, workers=-1, shaped=False, near=None):
        captured["near"] = tuple(near) if near is not None else None
        return np.zeros(3, dtype=int)

    monkeypatch.setattr(C, "phase_corr_from_ffts", fake_phase_corr)
    src, dst = _refs()
    task = TimeTask(
        name="a",
        t_from=0,
        t_to=1,
        src=src,
        dst=dst,
        crop=Crop((None, None), (None, None), (None, None)),
        near=(1, 2, 3),
    )
    run_task(task, Spectra(FakeReader()))
    assert captured["near"] == (1, 2, 3)


def test_run_task_near_none_passes_none_through(monkeypatch):
    captured = {}

    def fake_phase_corr(fft0, fft1, shape, workers=-1, shaped=False, near=None):
        captured["near"] = near
        return np.zeros(3, dtype=int)

    monkeypatch.setattr(C, "phase_corr_from_ffts", fake_phase_corr)
    src, dst = _refs()
    task = TimeTask(
        name="a",
        t_from=0,
        t_to=1,
        src=src,
        dst=dst,
        crop=Crop((None, None), (None, None), (None, None)),
    )
    run_task(task, Spectra(FakeReader()))
    assert captured["near"] is None


def test_corner_task_crop_and_reconstruction_recover_a_known_offset():
    """The CornerTask equivalent of test_pair_task_total_offset -- a real
    crop on both lateral axes, not (None, None, None). This is exactly the
    test whose absence let a crop/reconstruction bug through undetected."""
    rng = np.random.default_rng(0)
    nz, ny, nx = 8, 40, 40
    shift_px = 12
    true_dy, true_dx = 12, -13  # dy_sign=+1, dx_sign=-1
    dy_sign, dx_sign = 1, -1

    # A large shared field; a and b are two overlapping windows into it,
    # offset from each other by the true (known) corner displacement.
    field = rng.random((nz, ny + 60, nx + 60))
    ay0, ax0 = 30, 30
    by0, bx0 = ay0 + true_dy, ax0 + true_dx

    class SlicedReader:
        def read(self, origin):
            y0, x0 = origin
            return field[:, y0 : y0 + ny, x0 : x0 + nx]

    def side(sign, extent, s):
        return (s, extent) if sign > 0 else (0, extent - s)

    crop_a = Crop(
        (None, None), side(dy_sign, ny, shift_px), side(dx_sign, nx, shift_px)
    )
    crop_b = Crop(
        (None, None), side(-dy_sign, ny, shift_px), side(-dx_sign, nx, shift_px)
    )
    task = CornerTask(
        a="a",
        b="b",
        t=0,
        src=(ay0, ax0),
        dst=(by0, bx0),
        crop_a=crop_a,
        crop_b=crop_b,
        nominal=(0, dy_sign * shift_px, dx_sign * shift_px),
    )
    out = run_task(task, Spectra(SlicedReader()))
    assert (out.dz, out.dy, out.dx) == (0, true_dy, true_dx)


def test_unknown_task_type_is_loud():
    with pytest.raises(TypeError):
        run_task(object(), Spectra(FakeReader()))


# --- run_plan -----------------------------------------------------------------
class TinyPlan:
    def __init__(self, tasks):
        self._tasks = tuple(tasks)

    @property
    def tasks(self):
        return self._tasks

    def pending(self, store):
        return tuple(t for t in self._tasks if t.key not in store)


def _tasks(n):
    return [
        TimeTask(
            name=f"t{i}",
            t_from=i,
            t_to=i + 1,
            src=VolumeRef("f0", 0, i, 16),
            dst=VolumeRef("f0", 0, i + 1, 16),
            crop=Crop((None, None), (None, None), (None, None)),
        )
        for i in range(n)
    ]


def test_run_plan_runs_everything_once(tmp_path):
    plan = TinyPlan(_tasks(3))
    store = OffsetStore(tmp_path / "off.jsonl")
    assert run_plan(plan, store, Spectra(FakeReader())) == 3
    assert len(store) == 3


def test_run_plan_is_a_noop_the_second_time(tmp_path):
    plan = TinyPlan(_tasks(3))
    store = OffsetStore(tmp_path / "off.jsonl")
    run_plan(plan, store, Spectra(FakeReader()))
    reader = FakeReader()
    assert run_plan(plan, store, Spectra(reader)) == 0
    assert reader.reads == [], "a cached task must not touch the reader"


def test_run_plan_resumes_after_an_interruption(tmp_path):
    plan = TinyPlan(_tasks(5))
    path = tmp_path / "off.jsonl"
    store = OffsetStore(path)
    for task in plan.tasks[:2]:
        store.put(task, Offset(0, 0, 0))
    assert run_plan(plan, OffsetStore(path), Spectra(FakeReader())) == 3


def test_results_survive_a_crash_mid_run(tmp_path):
    """Each result is appended before the next task starts."""
    path = tmp_path / "off.jsonl"
    store = OffsetStore(path)
    tasks = _tasks(4)

    class Boom(FakeReader):
        def read(self, ref):
            if ref.local_t >= 2:
                raise KeyboardInterrupt
            return super().read(ref)

    with pytest.raises(KeyboardInterrupt):
        run_plan(TinyPlan(tasks), store, Spectra(Boom()))
    # task 0 reads local_t 0 and 1; task 1 reads local_t 2 and dies.
    assert len(OffsetStore(path)) == 1


class Bar:
    """A tqdm-shaped stand-in that records whether it was closed."""

    def __init__(self, xs):
        self.xs = list(xs)
        self.closed = False
        self.updates = 0

    def __iter__(self):
        return iter(self.xs)

    def update(self, n):
        self.updates += n

    def close(self):
        self.closed = True


@pytest.mark.parametrize("concurrency", [1, 3])
def test_progress_bar_is_closed_on_failure(tmp_path, concurrency):
    """An unclosed tqdm finalises at interpreter shutdown and hides the real error."""
    bars = []

    def progress(xs):
        bars.append(Bar(xs))
        return bars[-1]

    class Boom(FakeReader):
        def read(self, ref):
            raise RuntimeError("mount died")

    with pytest.raises(RuntimeError, match="mount died"):
        run_plan(
            TinyPlan(_tasks(4)),
            OffsetStore(),
            Spectra(Boom()),
            concurrency=concurrency,
            progress=progress,
        )
    assert bars and bars[0].closed


@pytest.mark.parametrize("concurrency", [1, 3])
def test_progress_bar_is_closed_on_success(tmp_path, concurrency):
    bars = []

    def progress(xs):
        bars.append(Bar(xs))
        return bars[-1]

    run_plan(
        TinyPlan(_tasks(4)),
        OffsetStore(),
        Spectra(FakeReader()),
        concurrency=concurrency,
        progress=progress,
    )
    assert bars[0].closed

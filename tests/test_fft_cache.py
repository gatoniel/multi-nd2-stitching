from collections import Counter

import numpy as np
import pytest

from multi_nd2_stitching.compute import (
    Spectra,
    phase_corr_from_ffts,
    run_plan,
    spectrum,
)
from multi_nd2_stitching.offsets import Crop, SpectrumRef, VolumeRef, build_plan
from multi_nd2_stitching.reader import SpectrumCache
from multi_nd2_stitching.store import OffsetStore

from helpers import FakeReader, build, make_meta

FULL = Crop((None, None), (None, None), (None, None))


class CountingReader(FakeReader):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.loads = Counter()

    def read(self, ref):
        self.loads[ref] += 1
        return super().read(ref)


# --- fftshift removal is an exact identity ------------------------------------
@pytest.mark.parametrize("shift", [(0, 0, 0), (0, 5, 0), (1, -3, 7), (0, 0, -1)])
@pytest.mark.parametrize("shape", [(8, 16, 16), (7, 15, 17)])
def test_matches_the_fftshift_formulation(shift, shape):
    """The old code did fftshift then argmax. This must agree exactly."""
    import scipy.fft as spfft

    a = np.random.default_rng(0).random(shape)
    b = np.roll(a, shift, axis=(0, 1, 2))
    f0, f1 = spfft.rfftn(a), spfft.rfftn(b)

    mult = f0 * np.conjugate(f1)
    mult /= np.abs(mult)
    inverse = spfft.irfftn(mult, s=shape, axes=[0, 1, 2])
    old_peak = np.array(np.unravel_index(np.argmax(np.fft.fftshift(inverse)), shape))
    old = old_peak - np.array(shape) // 2

    new = phase_corr_from_ffts(spfft.rfftn(a), spfft.rfftn(b), shape)
    assert tuple(new) == tuple(old)


# --- precision ----------------------------------------------------------------
@pytest.mark.parametrize(
    "precision,dtype",
    [
        ("float32", np.complex64),
        ("float64", np.complex128),
    ],
)
def test_precision_controls_the_spectrum_dtype(precision, dtype):
    a = np.random.default_rng(0).random((8, 16, 16))
    assert spectrum(a, precision=precision).dtype == dtype


def test_float32_finds_the_same_peak():
    a = np.random.default_rng(0).random((8, 32, 32))
    b = np.roll(a, (1, -4, 6), axis=(0, 1, 2))
    lo = phase_corr_from_ffts(
        spectrum(a, precision="float32"), spectrum(b, precision="float32"), a.shape
    )
    hi = phase_corr_from_ffts(
        spectrum(a, precision="float64"), spectrum(b, precision="float64"), a.shape
    )
    assert tuple(lo) == tuple(hi)


def test_precision_is_part_of_the_key(cfg_dict, tmp_path):
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    meta = make_meta(n_files=2, nt=5, paths=files)
    lay = build(cfg_dict, n_files=2, nt=5, paths=files)
    a = build_plan(lay, meta, precision="float64")
    b = build_plan(lay, meta, precision="float32")
    assert set(t.key for t in a.tasks).isdisjoint(t.key for t in b.tasks)


# --- the cache ----------------------------------------------------------------
def _sref(pos, t, precision="float64"):
    return SpectrumRef(VolumeRef("f0", pos, t, 4), FULL, precision)


def test_repeated_spectrum_is_transformed_once():
    reader = CountingReader(shape=(4, 8, 8))
    ref = _sref(0, 0)
    cache = SpectrumCache(reader, Counter({ref: 3}))
    a, _ = cache.get(ref)
    for _ in range(2):
        cache.get(ref)
    assert reader.loads[ref.volume] == 1
    assert cache.stats()["hits"] == 2


def test_last_use_frees_the_spectrum():
    cache = SpectrumCache(CountingReader(shape=(4, 8, 8)), Counter({_sref(0, 0): 2}))
    cache.get(_sref(0, 0))
    assert cache.resident == 1
    cache.get(_sref(0, 0))
    assert cache.resident == 0


def test_different_crops_are_different_entries():
    reader = CountingReader(shape=(4, 8, 8))
    a = SpectrumRef(VolumeRef("f0", 0, 0, 4), FULL, "float64")
    b = SpectrumRef(
        VolumeRef("f0", 0, 0, 4), Crop((1, 3), (None, None), (None, None)), "float64"
    )
    cache = SpectrumCache(reader, Counter({a: 2, b: 2}))
    fa, sa = cache.get(a)
    fb, sb = cache.get(b)
    assert sa != sb


def test_parent_and_child_strips_are_different_entries():
    reader = CountingReader(shape=(4, 8, 8))
    v = VolumeRef("f0", 0, 0, 4)
    p = SpectrumRef(v, FULL, "float64", axis=2, side="parent", shift_px=3)
    c = SpectrumRef(v, FULL, "float64", axis=2, side="child", shift_px=3)
    cache = SpectrumCache(reader, Counter({p: 2, c: 2}))
    (fp, sp), (fc, sc) = cache.get(p), cache.get(c)
    assert sp == sc == (4, 8, 5)
    assert not np.array_equal(fp, fc)


def test_spectrum_cache_returns_correct_results(cfg_dict, tmp_path):
    """Caching must not change a single offset."""
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=5, paths=files)
    plan = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)

    plain = OffsetStore()
    run_plan(plan, plain, Spectra(FakeReader(shape=(4, 8, 8))))

    cached = OffsetStore()
    reader = FakeReader(shape=(4, 8, 8))
    run_plan(plan, cached, SpectrumCache(reader, plan.spectrum_uses()))

    assert {k: plain[k] for k in plain._data} == {k: cached[k] for k in cached._data}


def test_time_tasks_reuse_anchor_spectra(cfg_dict, tmp_path):
    """The payoff: each anchor volume is transformed once, not twice."""
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=5, paths=files)
    plan = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    uses = plan.spectrum_uses()
    reused = [r for r, n in uses.items() if n > 1]
    assert reused, "anchor spectra should be shared between consecutive drift tasks"
    assert all(r.axis is None for r in reused), "only time spectra are reusable"


# --- concurrency --------------------------------------------------------------
@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_concurrency_gives_identical_results(cfg_dict, tmp_path, concurrency):
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=5, paths=files)
    plan = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)

    ref_store = OffsetStore()
    run_plan(plan, ref_store, Spectra(FakeReader(shape=(4, 8, 8))))

    store = OffsetStore(tmp_path / f"c{concurrency}.jsonl")
    n = run_plan(
        plan,
        store,
        SpectrumCache(FakeReader(shape=(4, 8, 8)), plan.spectrum_uses()),
        concurrency=concurrency,
    )
    assert n == len(plan.tasks)
    assert {k: store[k] for k in ref_store._data} == {
        k: ref_store[k] for k in ref_store._data
    }


def test_every_task_is_run_exactly_once(cfg_dict, tmp_path):
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=5, paths=files)
    plan = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    path = tmp_path / "off.jsonl"
    run_plan(
        plan, OffsetStore(path), Spectra(FakeReader(shape=(4, 8, 8))), concurrency=4
    )
    assert len(path.read_text().strip().splitlines()) == len(plan.tasks)


def test_an_exception_propagates_out_of_the_pool(cfg_dict, tmp_path):
    class Boom(FakeReader):
        def read(self, ref):
            raise RuntimeError("disk on fire")

    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    for f in files:
        open(f, "wb").write(b"x")
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3
    meta = make_meta(n_files=2, nt=5, paths=files)
    plan = build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)
    with pytest.raises(RuntimeError, match="disk on fire"):
        run_plan(plan, OffsetStore(), Spectra(Boom()), concurrency=4)

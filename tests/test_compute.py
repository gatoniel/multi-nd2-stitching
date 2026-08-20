import numpy as np
import pytest
from helpers import FakeReader

from multi_nd2_stitching.compute import (
    Spectra,
    crop_for_alignment,
    fft_translation_3d,
    run_plan,
    run_task,
)
from multi_nd2_stitching.offsets import Crop, PairTask, TimeTask, VolumeRef
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

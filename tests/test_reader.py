from collections import Counter

import numpy as np
import pytest
from helpers import FakeReader, build, make_meta, stub_files

from multi_nd2_stitching.compute import Spectra, run_plan
from multi_nd2_stitching.offsets import VolumeRef, build_plan
from multi_nd2_stitching.reader import VolumeCache
from multi_nd2_stitching.store import OffsetStore


def refs(n=3):
    return [VolumeRef("f0", i, 0, 4) for i in range(n)]


class CountingSource:
    def __init__(self, shape=(4, 8, 8)):
        self.shape = shape
        self.loads = Counter()

    def read(self, ref):
        self.loads[ref] += 1
        return np.full(self.shape, ref.position, dtype=np.uint16)


# --- the core promise ---------------------------------------------------------
def test_a_volume_is_loaded_once_however_often_it_is_used():
    a, b, _ = refs()
    src = CountingSource()
    cache = VolumeCache(src, Counter({a: 3, b: 1}))
    for _ in range(3):
        cache.read(a)
    cache.read(b)
    assert src.loads[a] == 1
    assert cache.stats()["hits"] == 2


def test_last_use_frees_it_immediately():
    a, _, _ = refs()
    cache = VolumeCache(CountingSource(), Counter({a: 2}))
    cache.read(a)
    assert cache.resident == 1
    cache.read(a)
    assert cache.resident == 0, "the final use must not leave it in memory"


def test_single_use_volume_is_never_cached():
    a, _, _ = refs()
    cache = VolumeCache(CountingSource(), Counter({a: 1}))
    cache.read(a)
    assert cache.resident == 0


def test_unknown_volume_passes_through_uncached():
    """An ad-hoc read outside the plan must not leak."""
    a, _, _ = refs()
    cache = VolumeCache(CountingSource(), Counter())
    cache.read(a)
    cache.read(a)
    assert cache.resident == 0


def test_returns_the_same_data_as_the_source():
    a, _, _ = refs()
    src = CountingSource()
    cache = VolumeCache(src, Counter({a: 2}))
    assert np.array_equal(cache.read(a), cache.read(a))


# --- memory bound -------------------------------------------------------------
def test_max_bytes_evicts_the_least_needed_first():
    a, b, c = refs()
    src = CountingSource(shape=(4, 8, 8))  # 512 bytes each
    cache = VolumeCache(src, Counter({a: 5, b: 2, c: 2}), max_bytes=1100)
    cache.read(a)
    cache.read(b)
    cache.read(c)  # must evict b (fewest left)
    assert cache.evicted_early == 1
    assert cache.resident == 2


def test_max_bytes_never_stalls_progress():
    a, b, _ = refs()
    src = CountingSource(shape=(4, 8, 8))
    cache = VolumeCache(src, Counter({a: 3, b: 3}), max_bytes=1)
    cache.read(a)
    cache.read(b)
    assert cache.resident == 0  # nothing fits, still works
    assert np.array_equal(cache.read(a), src.read(a))


# --- integration with a real plan ---------------------------------------------
@pytest.fixture
def real_plan(cfg_dict, tmp_path):
    files = stub_files(tmp_path, 2)
    cfg_dict["files"] = files
    cfg_dict["shift_px"] = 3  # tiny, to match the 8x8 fake volumes
    meta = make_meta(n_files=2, nt=5, paths=files)
    return build_plan(build(cfg_dict, n_files=2, nt=5, paths=files), meta)


def test_plan_is_ordered_by_timepoint(real_plan):
    """Ordering is what makes refcounting bound the working set."""
    times = [t.t_to if hasattr(t, "t_to") else t.t for t in real_plan.tasks]
    assert times == sorted(times)


def test_volume_uses_counts_every_reference(real_plan):
    uses = real_plan.volume_uses()
    assert sum(uses.values()) == 2 * len(real_plan.tasks)


def test_cache_cuts_reads_on_a_real_plan(real_plan):
    src = CountingSource(shape=(4, 8, 8))
    cache = VolumeCache(src, real_plan.volume_uses())
    for task in real_plan.tasks:
        cache.read(task.src)
        cache.read(task.dst)
    assert sum(src.loads.values()) < 2 * len(real_plan.tasks)
    assert cache.resident == 0, "every volume must be released by the end"


def test_working_set_stays_small(real_plan):
    """With timepoint ordering the cache should hold ~2 timepoints, not the run."""
    src = CountingSource(shape=(4, 8, 8))
    cache = VolumeCache(src, real_plan.volume_uses())
    peak = 0
    for task in real_plan.tasks:
        cache.read(task.src)
        cache.read(task.dst)
        peak = max(peak, cache.resident)
    assert peak <= 4


def test_run_plan_through_the_cache(real_plan, tmp_path):
    cache = VolumeCache(FakeReader(shape=(4, 8, 8)), real_plan.volume_uses())
    store = OffsetStore(tmp_path / "off.jsonl")
    assert run_plan(real_plan, store, Spectra(cache)) == len(real_plan.tasks)
    assert cache.resident == 0

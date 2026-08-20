"""Nd2Reader against a stub nd2 module -- concurrency behaviour, no real files."""

import sys
import threading
import types

import numpy as np
import pytest

from multi_nd2_stitching.offsets import VolumeRef
from multi_nd2_stitching.reader import Nd2Reader


class StubND2:
    """Minimal stand-in. Records concurrent use so races are visible."""

    opened = 0
    live = 0
    max_live = 0
    _lock = threading.Lock()

    def __init__(self, path, nt=4, nz=4, ny=8, nx=8, has_t=True):
        self.path = path
        self.sizes = {"T": nt, "P": 2, "Z": nz, "Y": ny, "X": nx}
        if not has_t:
            del self.sizes["T"]
        self.dtype = np.dtype("uint16")
        self._nz = nz
        self._ny, self._nx = ny, nx
        type(self).opened += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def _seq_index_from_coords(self, coords):
        return int(
            np.ravel_multi_index(
                coords,
                [self.sizes.get("T", 1), self.sizes["P"], self.sizes["Z"]][
                    -len(coords) :
                ],
            )
        )

    def read_frame(self, index):
        with type(self)._lock:
            type(self).live += 1
            type(self).max_live = max(type(self).max_live, type(self).live)
        try:
            return np.full((self._ny, self._nx), index % 1000, dtype=np.uint16)
        finally:
            with type(self)._lock:
                type(self).live -= 1


@pytest.fixture
def stub_nd2(monkeypatch):
    StubND2.opened = StubND2.live = StubND2.max_live = 0
    mod = types.ModuleType("nd2")
    mod.ND2File = StubND2
    monkeypatch.setitem(sys.modules, "nd2", mod)
    return StubND2


def make_reader(**kw):
    return Nd2Reader(["a.nd2", "b.nd2"], ["k0", "k1"], nz=4, ny=8, nx=8, **kw)


def test_reads_a_volume(stub_nd2):
    with make_reader(threads=2) as r:
        arr = r.read(VolumeRef("k0", 1, 2, 4))
    assert arr.shape == (4, 8, 8)
    assert arr.dtype == np.uint16


def test_frames_are_consecutive(stub_nd2):
    with make_reader(threads=2) as r:
        arr = r.read(VolumeRef("k0", 0, 0, 4))
    first = int(arr[0, 0, 0])
    assert [int(arr[z, 0, 0]) for z in range(4)] == list(range(first, first + 4))


def test_requires_a_context_manager(stub_nd2):
    with pytest.raises(RuntimeError, match="context manager"):
        make_reader().read(VolumeRef("k0", 0, 0, 4))


# --- the bug: peeking into the pool while every handle is checked out ---------
def test_concurrent_reads_do_not_exhaust_the_pool(stub_nd2):
    """Reproduces `IndexError: deque index out of range`.

    With several correlation threads reading at once, every pooled handle can
    be out. Any code that peeks at pool.queue[0] blows up exactly here.
    """
    errors = []

    def hammer(r, pos):
        try:
            for t in range(4):
                r.read(VolumeRef("k0", pos % 2, t, 4))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with make_reader(threads=4, handles=2) as r:
        threads = [threading.Thread(target=hammer, args=(r, i)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors, errors


def test_more_handles_than_pooled_are_never_needed(stub_nd2):
    """Even with a single pooled handle, reads must serialise rather than fail."""
    with make_reader(threads=4, handles=1) as r:
        arr = r.read(VolumeRef("k1", 0, 1, 4))
    assert arr.shape == (4, 8, 8)


def test_index_lookup_does_not_borrow_a_pooled_handle(stub_nd2):
    with make_reader(threads=2, handles=3) as r:
        r.read(VolumeRef("k0", 0, 0, 4))  # opens file 0
        pool = r._pools[0]
        borrowed = [pool.get() for _ in range(3)]  # drain the pool
        try:
            assert r._first_frame_index(0, VolumeRef("k0", 1, 2, 4)) >= 0
        finally:
            for h in borrowed:
                pool.put(h)


def test_handles_default_to_the_thread_count(stub_nd2):
    r = make_reader(threads=6)
    assert r._handles >= 6


def test_files_without_a_time_axis(stub_nd2, monkeypatch):
    mod = sys.modules["nd2"]
    monkeypatch.setattr(mod, "ND2File", lambda path: StubND2(path, has_t=False))
    with make_reader(threads=2) as r:
        assert r.read(VolumeRef("k0", 1, 0, 4)).shape == (4, 8, 8)


def test_handles_are_released_on_exit(stub_nd2):
    with make_reader(threads=2) as r:
        r.read(VolumeRef("k0", 0, 0, 4))
    assert r._pools == {}
    assert r._index_handle == {}


# --- lazy opening -------------------------------------------------------------
def test_no_files_are_opened_until_first_read(stub_nd2):
    with make_reader(threads=2, handles=3) as r:
        assert stub_nd2.opened == 0
        r.read(VolumeRef("k0", 0, 0, 4))
        assert 0 < stub_nd2.opened <= 5


def test_only_the_files_in_use_stay_open(stub_nd2):
    with make_reader(threads=2, handles=3, max_open_files=1) as r:
        r.read(VolumeRef("k0", 0, 0, 4))
        r.read(VolumeRef("k1", 0, 0, 4))
        assert list(r._pools) == [1], "file 0 should have been closed"


def test_revisiting_a_file_reopens_it(stub_nd2):
    with make_reader(threads=2, handles=2, max_open_files=1) as r:
        a = r.read(VolumeRef("k0", 1, 2, 4))
        r.read(VolumeRef("k1", 0, 0, 4))
        b = r.read(VolumeRef("k0", 1, 2, 4))
    assert np.array_equal(a, b)


def test_a_file_in_use_is_never_closed(stub_nd2):
    """Eviction must not pull a handle out from under a running read."""
    errors = []

    def hammer(r, key):
        try:
            for _ in range(3):
                for t in range(4):  # the stub has 4 timepoints
                    r.read(VolumeRef(key, 0, t, 4))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    with make_reader(threads=4, handles=2, max_open_files=1) as r:
        ts = [threading.Thread(target=hammer, args=(r, k)) for k in ("k0", "k1")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert not errors, errors


def test_everything_is_closed_on_exit(stub_nd2):
    with make_reader(threads=2, handles=2) as r:
        r.read(VolumeRef("k0", 0, 0, 4))
        r.read(VolumeRef("k1", 0, 0, 4))
    assert r._pools == {} and r._index_handle == {}

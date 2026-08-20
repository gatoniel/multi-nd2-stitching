"""Reading volumes, and keeping the useful ones in memory.

The plan is known up front, so the cache never has to guess: it counts how many
times each volume is needed, and drops one the moment its last use is done. No
LRU, no heuristics, no tuning -- exact reference counting against a static plan.
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from queue import Queue
from typing import Self

import numpy as np

from .offsets import SpectrumRef, VolumeRef


class Nd2Reader:
    """Reads one volume at a time from a set of open ND2 files.

    Frames are fetched through a pool of handles so that several threads can
    read from the same file at once -- the SDK handle is not thread-safe, but
    the read releases the GIL, so this is real parallelism on a slow mount.
    """

    def __init__(
        self,
        paths,
        file_keys,
        nz: int,
        ny: int,
        nx: int,
        handles: int | None = None,
        threads: int = 10,
        max_open_files: int = 2,
    ):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.threads = threads
        self._by_key = {k: i for i, k in enumerate(file_keys)}
        self._paths = list(paths)
        # One handle per reader thread, plus slack. Fewer handles than threads
        # only makes threads queue up behind each other.
        self._handles = threads + 2 if handles is None else handles
        self._stack: ExitStack | None = None
        self._pools: dict[int, Queue] = {}
        self._dtype: dict[int, object] = {}
        self._has_t: dict[int, bool] = {}
        # A handle reserved for coordinate lookups, never in the pool. Peeking
        # into the pool is not safe: with concurrency > 1 every handle can be
        # checked out, and `pool.queue[0]` then raises IndexError.
        self._index_handle: dict[int, object] = {}
        self._index_lock = threading.Lock()
        self._open_lock = threading.RLock()
        self._lru: list[int] = []
        self._active: Counter = Counter()
        self.max_open_files = max_open_files
        self._executor: ThreadPoolExecutor | None = None

    # --- lifecycle: explicit, so no __del__ runs at interpreter shutdown ---
    def __enter__(self) -> Self:
        self._stack = ExitStack()
        self._executor = self._stack.enter_context(
            ThreadPoolExecutor(max_workers=self.threads)
        )
        return self

    def _acquire(self, file_i: int) -> None:
        """Open one file's handles on first use and mark it in use.

        Opening and claiming happen under one lock acquisition: doing them
        separately leaves a window in which another thread can evict the file
        between the two, and the caller then reads from handles that are gone.

        Opening every file up front means handles x files descriptors live at
        once -- for nine files that was over a hundred open ND2 objects, each
        with its own buffers, competing for page cache on a network share. Work
        is ordered by timepoint, so only one or two files are ever in play.
        """
        import nd2

        with self._open_lock:
            if file_i in self._pools:
                self._lru.remove(file_i)
                self._lru.append(file_i)
                self._active[file_i] += 1
                return
            path = self._paths[file_i]
            probe = nd2.ND2File(str(path))
            pool: Queue = Queue()
            for _ in range(self._handles):
                pool.put(nd2.ND2File(str(path)))
            self._pools[file_i] = pool
            self._index_handle[file_i] = probe
            self._dtype[file_i] = probe.dtype
            self._has_t[file_i] = "T" in probe.sizes
            self._lru.append(file_i)
            self._active[file_i] += 1
            self._evict_locked()

    def _release(self, file_i: int) -> None:
        with self._open_lock:
            self._active[file_i] -= 1

    def _evict_locked(self) -> None:
        """Close files nobody is reading from, oldest first."""
        while len(self._pools) > self.max_open_files:
            victim = next((f for f in self._lru if not self._active[f]), None)
            if victim is None:
                return  # all in use; try again next time
            self._lru.remove(victim)
            pool = self._pools.pop(victim)
            while not pool.empty():
                pool.get().close()
            self._index_handle.pop(victim).close()
            self._dtype.pop(victim, None)
            self._has_t.pop(victim, None)

    def __exit__(self, *exc) -> None:
        with self._open_lock:
            for pool in self._pools.values():
                while not pool.empty():
                    pool.get().close()
            for handle in self._index_handle.values():
                handle.close()
            self._pools.clear()
            self._index_handle.clear()
            self._lru.clear()
        self._stack.close()
        self._stack = None

    # --- reading ----------------------------------------------------------
    def _frame(self, file_i: int, index: int):
        pool = self._pools[file_i]
        handle = pool.get()
        try:
            return handle.read_frame(index)
        finally:
            pool.put(handle)

    def _first_frame_index(self, file_i: int, ref: VolumeRef) -> int:
        coords = (
            (ref.local_t, ref.position, 0) if self._has_t[file_i] else (ref.position, 0)
        )
        with self._index_lock:
            return self._index_handle[file_i]._seq_index_from_coords(coords)

    def read(self, ref: VolumeRef) -> np.ndarray:
        if self._stack is None:
            raise RuntimeError("Nd2Reader must be used as a context manager")
        file_i = self._by_key[ref.file]
        self._acquire(file_i)
        try:
            start = self._first_frame_index(file_i, ref)
            arr = np.empty((ref.nz, self.ny, self.nx), dtype=self._dtype[file_i])
            futures = [
                self._executor.submit(self._frame, file_i, start + z)
                for z in range(ref.nz)
            ]
            for z, fut in enumerate(futures):
                arr[z] = fut.result()
            return arr
        finally:
            self._release(file_i)


class VolumeCache:
    """Refcounted volume cache in front of any reader.

    `uses` comes from Plan.volume_uses(). Each read decrements; at zero the
    volume is dropped. A volume the plan never mentions is passed through
    uncached, so an ad-hoc read can't leak.
    """

    def __init__(self, source, uses: Counter, max_bytes: int | None = None):
        self.source = source
        self.remaining = Counter(uses)
        self.max_bytes = max_bytes
        self._cache: dict[VolumeRef, np.ndarray] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0
        self.evicted_early = 0

    def _drop(self, ref: VolumeRef) -> None:
        arr = self._cache.pop(ref, None)
        if arr is not None:
            self._bytes -= arr.nbytes

    def read(self, ref: VolumeRef) -> np.ndarray:
        with self._lock:
            arr = self._cache.get(ref)
            if arr is not None:
                self.hits += 1
            else:
                self.misses += 1
        if arr is None:
            arr = self.source.read(ref)  # outside the lock: this is the slow part
        with self._lock:
            return self._settle(ref, arr)

    def _settle(self, ref: VolumeRef, arr):
        left = self.remaining[ref] - 1
        self.remaining[ref] = max(left, 0)

        if left > 0:
            if ref not in self._cache:
                # Make room by dropping the volume with the fewest uses left.
                while (
                    self.max_bytes is not None
                    and self._bytes + arr.nbytes > self.max_bytes
                    and self._cache
                ):
                    victim = min(self._cache, key=lambda r: self.remaining[r])
                    self._drop(victim)
                    self.evicted_early += 1
                if self.max_bytes is None or arr.nbytes <= self.max_bytes:
                    self._cache[ref] = arr
                    self._bytes += arr.nbytes
                    self.peak_bytes = max(self.peak_bytes, self._bytes)
        else:
            self._drop(ref)  # last use -- let it go immediately
        return arr

    @property
    def resident(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "reads": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "peak_mb": round(self.peak_bytes / 2**20, 1),
            "resident": self.resident,
            "evicted_early": self.evicted_early,
        }


class _RefCounted:
    """Shared machinery: hold an entry until its last planned use is done."""

    def __init__(self, uses: Counter, max_bytes: int | None = None):
        self.remaining = Counter(uses)
        self.max_bytes = max_bytes
        self._cache: dict = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0
        self.evicted_early = 0

    def _drop(self, key) -> None:
        entry = self._cache.pop(key, None)
        if entry is not None:
            self._bytes -= self._sizeof(entry)

    def _sizeof(self, entry) -> int:
        raise NotImplementedError

    def _fetch(self, key):
        raise NotImplementedError

    def _get(self, key):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self.hits += 1
                self._settle(key, entry)
                return entry
            self.misses += 1

        entry = self._fetch(key)  # outside the lock: this is the slow part

        with self._lock:
            self._settle(key, entry)
        return entry

    def _settle(self, key, entry) -> None:
        left = self.remaining[key] - 1
        self.remaining[key] = max(left, 0)
        if left <= 0:
            self._drop(key)
            return
        if key in self._cache:
            return
        size = self._sizeof(entry)
        while (
            self.max_bytes is not None
            and self._bytes + size > self.max_bytes
            and self._cache
        ):
            victim = min(self._cache, key=lambda k: self.remaining[k])
            self._drop(victim)
            self.evicted_early += 1
        if self.max_bytes is None or size <= self.max_bytes:
            self._cache[key] = entry
            self._bytes += size
            self.peak_bytes = max(self.peak_bytes, self._bytes)

    @property
    def resident(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "reads": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "peak_mb": round(self.peak_bytes / 2**20, 1),
            "resident": self.resident,
            "evicted_early": self.evicted_early,
        }


class SpectrumCache(_RefCounted):
    """Refcounted rfftn cache in front of a volume reader.

    An anchor's spectrum is used twice -- as `dst` of the (t-1 -> t) drift task
    and as `src` of the (t -> t+1) one -- so this halves the transforms on the
    time side. Pair spectra are single-use and pass straight through, because a
    tile is parent to one neighbour and child to another: different strips.

    Note the size: an rfftn of a full tile is ~4x the uint16 volume at float64,
    ~2x at float32. Pass max_bytes.
    """

    def __init__(
        self, reader, uses: Counter, workers: int = -1, max_bytes: int | None = None
    ):
        super().__init__(uses, max_bytes)
        self.reader = reader
        self.workers = workers

    def _sizeof(self, entry) -> int:
        return entry[0].nbytes

    def _fetch(self, ref: SpectrumRef):
        from .compute import spectrum, trim_for

        arr = trim_for(self.reader.read(ref.volume)[ref.crop.as_slices()], ref)
        return spectrum(arr, self.workers, ref.precision), arr.shape

    def get(self, ref: SpectrumRef):
        return self._get(ref)

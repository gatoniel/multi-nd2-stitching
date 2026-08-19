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
        handles: int = 10,
        threads: int = 10,
    ):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.threads = threads
        self._by_key = {k: i for i, k in enumerate(file_keys)}
        self._paths = list(paths)
        self._handles = handles
        self._stack: ExitStack | None = None
        self._pools: dict[int, Queue] = {}
        self._dtype: dict[int, object] = {}
        self._executor: ThreadPoolExecutor | None = None

    # --- lifecycle: explicit, so no __del__ runs at interpreter shutdown ---
    def __enter__(self) -> Self:
        import nd2

        self._stack = ExitStack()
        self._executor = self._stack.enter_context(
            ThreadPoolExecutor(max_workers=self.threads)
        )
        for i, path in enumerate(self._paths):
            pool: Queue = Queue()
            for _ in range(self._handles):
                pool.put(self._stack.enter_context(nd2.ND2File(str(path))))
            self._pools[i] = pool
            probe = pool.queue[0]
            self._dtype[i] = probe.dtype
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()
        self._stack = None
        self._pools.clear()

    # --- reading ----------------------------------------------------------
    def _frame(self, file_i: int, index: int):
        pool = self._pools[file_i]
        handle = pool.get()
        try:
            return handle.read_frame(index)
        finally:
            pool.put(handle)

    def read(self, ref: VolumeRef) -> np.ndarray:
        if self._stack is None:
            raise RuntimeError("Nd2Reader must be used as a context manager")
        file_i = self._by_key[ref.file]
        handle = self._pools[file_i].queue[0]
        if "T" in handle.sizes:
            start = handle._seq_index_from_coords((ref.local_t, ref.position, 0))
        else:
            start = handle._seq_index_from_coords((ref.position, 0))

        arr = np.empty((ref.nz, self.ny, self.nx), dtype=self._dtype[file_i])
        futures = [
            self._executor.submit(self._frame, file_i, start + z) for z in range(ref.nz)
        ]
        for z, fut in enumerate(futures):
            arr[z] = fut.result()
        return arr


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

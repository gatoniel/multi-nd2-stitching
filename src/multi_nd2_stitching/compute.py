"""Phase correlation and the loop that runs a plan."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Protocol

import numpy as np
import scipy.fft as spfft

from .offsets import PairTask, SpectrumRef, TimeTask, VolumeRef
from .store import Offset, OffsetStore

PRECISION = {"float32": np.float32, "float64": np.float64}


# --- pure correlation ---------------------------------------------------------
def phase_corr_from_ffts(fft0, fft1, shape, workers: int = -1):
    """Cross-power spectrum -> peak, as a signed shift.

    No fftshift: the peak index is wrapped arithmetically instead. fftshift on
    a 100 MB+ array is a full single-threaded copy, and it buys nothing that
    `(i + n//2) % n - n//2` doesn't.
    """
    mult = fft0 * np.conjugate(fft1)
    np.divide(mult, np.abs(mult), out=mult)
    inverse = spfft.irfftn(mult, s=shape, workers=workers, axes=list(range(fft0.ndim)))
    peak = np.array(np.unravel_index(np.argmax(inverse), shape=shape))
    half = np.array(shape) // 2
    return (peak + half) % np.array(shape) - half


def spectrum(img, workers: int = -1, precision: str = "float64"):
    """rfftn at the requested precision.

    float32 halves both the transform time and the ~400 MB the spectrum of a
    full tile occupies. It is part of the cache key, so switching precision
    recomputes rather than silently mixing results.
    """
    return spfft.rfftn(np.asarray(img, dtype=PRECISION[precision]), workers=workers)


def fft_translation_3d(img0, img1, workers: int = -1, precision: str = "float64"):
    return phase_corr_from_ffts(
        spectrum(img0, workers, precision),
        spectrum(img1, workers, precision),
        np.shape(img0),
        workers=workers,
    )


def crop_for_alignment(parent, child, axis: int, shift_px: int):
    """Slice off the non-overlapping margin before correlating."""
    n = parent.shape[axis]
    if not 0 < shift_px < n:
        raise ValueError(f"shift_px={shift_px} outside tile extent {n} on axis {axis}")
    sl_p = [slice(None)] * 3
    sl_c = [slice(None)] * 3
    sl_p[axis] = slice(shift_px, None)
    sl_c[axis] = slice(None, n - shift_px)
    return parent[tuple(sl_p)], child[tuple(sl_c)]


def trim_for(arr, ref: SpectrumRef):
    """The sub-array a SpectrumRef designates."""
    if ref.axis is None:
        return arr
    n = arr.shape[ref.axis]
    if not 0 < ref.shift_px < n:
        raise ValueError(
            f"shift_px={ref.shift_px} outside tile extent {n} on axis {ref.axis}"
        )
    sl = [slice(None)] * 3
    sl[ref.axis] = (
        slice(ref.shift_px, None)
        if ref.side == "parent"
        else slice(None, n - ref.shift_px)
    )
    return arr[tuple(sl)]


# --- providers ----------------------------------------------------------------
class VolumeReader(Protocol):
    def read(self, ref: VolumeRef) -> np.ndarray: ...


class Spectra:
    """Uncached spectrum provider: read, crop, trim, transform."""

    def __init__(self, reader: VolumeReader, workers: int = -1):
        self.reader = reader
        self.workers = workers

    def get(self, ref: SpectrumRef):
        arr = trim_for(self.reader.read(ref.volume)[ref.crop.as_slices()], ref)
        return spectrum(arr, self.workers, ref.precision), arr.shape


# --- running ------------------------------------------------------------------
def run_task(task, spectra, workers: int = -1) -> Offset:
    """Execute one task against a spectrum provider."""
    if not isinstance(task, (TimeTask, PairTask)):
        raise TypeError(f"unknown task type {type(task).__name__}")

    s0, s1 = task.spectrum_refs()
    fft0, shape = spectra.get(s0)
    fft1, _ = spectra.get(s1)
    offset = phase_corr_from_ffts(fft0, fft1, shape, workers=workers)
    if isinstance(task, PairTask):
        offset[task.axis] += task.shift_px
    return Offset.of(offset)


def run_plan(
    plan,
    store: OffsetStore,
    spectra,
    workers: int = -1,
    concurrency: int = 1,
    limit: int | None = None,
    progress=None,
) -> int:
    """Run only what is missing. Returns the number of tasks executed.

    `concurrency` runs several correlations at once. scipy.fft and the numpy
    reductions release the GIL, so threads give real parallelism here -- and
    they must be threads, not processes, because the whole point is a shared
    volume/spectrum cache holding arrays far too big to pickle.

    Tasks are submitted in a bounded window, never all at once: the plan is
    ordered by timepoint so the cache's working set stays small, and a wide-open
    submission would defeat that.
    """
    pending = plan.pending(store)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return 0
    if concurrency <= 1:
        it = progress(pending) if progress is not None else pending
        done = 0
        try:
            for task in it:
                store.put(task, run_task(task, spectra, workers=workers))
                done += 1
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()
        return done

    lock = threading.Lock()
    done = 0
    tracker = progress(pending) if progress is not None else None

    def work(task):
        offset = run_task(task, spectra, workers=workers)
        with lock:
            store.put(task, offset)
        return task

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            queue = deque(pending)
            running = {
                ex.submit(work, queue.popleft())
                for _ in range(min(concurrency, len(queue)))
            }
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    fut.result()  # re-raise inside the caller's thread
                    done += 1
                    if tracker is not None:
                        tracker.update(1)
                while queue and len(running) < concurrency:
                    running.add(ex.submit(work, queue.popleft()))
    finally:
        # Close the bar here, or tqdm's __del__ runs during interpreter
        # shutdown and buries the real traceback under an ImportError.
        close = getattr(tracker, "close", None)
        if close is not None:
            close()
    return done

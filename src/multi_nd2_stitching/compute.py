"""Phase correlation and the loop that runs a plan."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import NamedTuple, Protocol

import numpy as np
import scipy.fft as spfft

from .offsets import PairTask, SpectrumRef, TimeTask, VolumeRef
from .store import Offset, OffsetStore

PRECISION = {"float32": np.float32, "float64": np.float64}


# --- pure correlation ---------------------------------------------------------
# How many of the response's highest-value points get considered as candidate
# peaks in `shaped` mode. Small on purpose: this only has to beat out a single
# artifact spike, not do a general search.
SHAPED_PEAK_CANDIDATES = 25


def _neighbour_decay(inverse, idx: tuple) -> float:
    """How much `inverse` drops from `idx` to its neighbours, in the axis that
    drops *least*.

    The response is a circular correlation surface, so neighbours wrap
    (`np.roll`-style modular indexing). A genuine correlation peak falls off
    in every one of the 3 axes; a sensor-defect streak (bright along the
    whole length of one row/column/line) does not fall off along that one
    axis, so its minimum here is near zero or negative regardless of how tall
    the spike is -- that's what distinguishes it from a real peak, without
    assuming any particular curve shape.

    Each axis compares against the *brighter* of its two neighbours, not
    their average: a point sitting right at the edge of a streak (background
    on one side, more streak on the other) still has one neighbour as bright
    as itself, and averaging that against the dark side would understate how
    little it actually decays along the streak's own axis -- exactly the
    point most likely to otherwise be mistaken for a real peak.
    """
    center = inverse[idx]
    shape = inverse.shape
    worst = np.inf
    for axis, n in enumerate(shape):
        plus = list(idx)
        plus[axis] = (idx[axis] + 1) % n
        minus = list(idx)
        minus[axis] = (idx[axis] - 1) % n
        brighter_neighbour = max(inverse[tuple(plus)], inverse[tuple(minus)])
        worst = min(worst, center - brighter_neighbour)
    return worst


class Candidate(NamedTuple):
    """One point in a response surface, considered as a possible peak."""

    index: tuple  # raw (unwrapped) array index
    value: float
    decay: float  # `_neighbour_decay` at this point


def candidate_peaks(inverse, candidates: int = SHAPED_PEAK_CANDIDATES) -> list:
    """The `candidates` highest-value points in `inverse`, brightest first.

    Shared by `_shaped_peak_index` (which just picks the healthiest-decay
    entry) and by `inspect.py`'s CSV export, so a debugging session sees
    exactly the same candidate set the real `shaped_peak` override considers
    -- not a separately-tuned view.
    """
    flat = inverse.ravel()
    k = min(candidates, flat.size)
    top = np.argpartition(flat, -k)[-k:]
    out = [
        Candidate(
            index=(idx := np.unravel_index(i, inverse.shape)),
            value=float(inverse[idx]),
            decay=float(_neighbour_decay(inverse, idx)),
        )
        for i in top
    ]
    out.sort(key=lambda c: c.value, reverse=True)
    return out


def _shaped_peak_index(inverse, candidates: int = SHAPED_PEAK_CANDIDATES) -> tuple:
    """The best-shaped point among the `candidates` highest values.

    "Best-shaped" is whichever has the healthiest `_neighbour_decay` -- not
    necessarily the tallest of the candidates.
    """
    return max(candidate_peaks(inverse, candidates), key=lambda c: c.decay).index


def to_signed_shift(idx, shape) -> np.ndarray:
    """A raw array index -> a signed shift, wrapping arithmetically.

    No fftshift: this does the same job (`(i + n//2) % n - n//2`) without the
    full single-threaded copy fftshift would cost on a 100 MB+ array.
    """
    peak = np.array(idx)
    shape = np.array(shape)
    half = shape // 2
    return (peak + half) % shape - half


def axis_profile(inverse, idx: tuple, axis: int, radius: int = 8) -> np.ndarray:
    """`2*radius+1` values of `inverse`, centred on `idx`, stepping along one
    axis and wrapping -- the drop-off curve `_neighbour_decay` only samples
    two points of.
    """
    n = inverse.shape[axis]
    steps = np.arange(-radius, radius + 1)
    out = np.empty(steps.shape, dtype=inverse.dtype)
    for k, step in enumerate(steps):
        pos = list(idx)
        pos[axis] = (idx[axis] + int(step)) % n
        out[k] = inverse[tuple(pos)]
    return out


def correlation_surface(fft0, fft1, shape, workers: int = -1) -> np.ndarray:
    """Cross-power spectrum -> the unshifted (raw-index) response surface.

    The one place that knows how to build a phase-correlation surface --
    `phase_corr_from_ffts` and every `inspect.py` export all go through this,
    so there is exactly one copy of the zero-bin fix below to keep correct.
    """
    mult = fft0 * np.conjugate(fft1)
    # A cross-power bin is exactly zero only where fft0 or fft1 is exactly
    # zero -- most often a flat/blank patch in the correlated crop. Dividing
    # there is 0/0 -> NaN, and one NaN in `mult` turns the whole inverse FFT
    # (and thus the peak) into garbage. Leave those bins at their already-zero
    # value instead: "no information here" rather than "poison everything".
    mag = np.abs(mult)
    np.divide(mult, mag, out=mult, where=mag != 0)
    return spfft.irfftn(mult, s=shape, workers=workers, axes=list(range(fft0.ndim)))


def phase_corr_from_ffts(fft0, fft1, shape, workers: int = -1, shaped: bool = False):
    """Cross-power spectrum -> peak, as a signed shift.

    `shaped=True` (the `shaped_peak` override) picks the best-shaped point
    among the top candidates instead of the single tallest one -- see
    `_shaped_peak_index`. Default is the plain, unchanged `argmax`, so every
    offset computed without the override is bit-for-bit what it always was.
    """
    inverse = correlation_surface(fft0, fft1, shape, workers=workers)
    if shaped:
        idx = _shaped_peak_index(inverse)
    else:
        idx = np.unravel_index(np.argmax(inverse), shape=shape)
    return to_signed_shift(idx, shape)


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
    offset = phase_corr_from_ffts(
        fft0, fft1, shape, workers=workers, shaped=task.shaped_peak
    )
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

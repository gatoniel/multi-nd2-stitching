"""The expensive part: phase correlation, and the loop that runs a plan.

The correlation functions are pure array -> array. The runner takes a reader
protocol, so the whole pipeline can be exercised on synthetic volumes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import scipy.fft as spfft

from .offsets import PairTask, TimeTask, VolumeRef
from .store import Offset, OffsetStore


# --- pure correlation ---------------------------------------------------------
def phase_corr_from_ffts(fft0, fft1, shape, workers: int = -1):
    mult = fft0 * np.conjugate(fft1)
    mult /= np.abs(mult)
    inverse = spfft.irfftn(mult, s=shape, workers=workers, axes=list(range(fft0.ndim)))
    peak = np.array(np.unravel_index(np.argmax(np.fft.fftshift(inverse)), shape=shape))
    return peak - np.array(shape) // 2


def fft_translation_3d(img0, img1, workers: int = -1):
    return phase_corr_from_ffts(
        spfft.rfftn(img0, workers=workers),
        spfft.rfftn(img1, workers=workers),
        img0.shape,
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


# --- reading ------------------------------------------------------------------
class VolumeReader(Protocol):
    def read(self, ref: VolumeRef) -> np.ndarray: ...


def run_task(task, reader: VolumeReader, workers: int = -1) -> Offset:
    """Execute one task. No caching, no I/O beyond the reader."""
    # Dispatch before reading: an unknown task must fail instantly, not after
    # pulling two volumes off a network mount.
    if not isinstance(task, (TimeTask, PairTask)):
        raise TypeError(f"unknown task type {type(task).__name__}")

    sl = task.crop.as_slices()
    src = reader.read(task.src)[sl]
    dst = reader.read(task.dst)[sl]

    if isinstance(task, TimeTask):
        return Offset.of(fft_translation_3d(src, dst, workers=workers))

    crop0, crop1 = crop_for_alignment(src, dst, task.axis, task.shift_px)
    offset = fft_translation_3d(crop0, crop1, workers=workers)
    offset[task.axis] += task.shift_px
    return Offset.of(offset)


def run_plan(
    plan,
    store: OffsetStore,
    reader: VolumeReader,
    workers: int = -1,
    progress=None,
) -> int:
    """Run only what is missing. Returns the number of tasks executed.

    Safe to interrupt: each result is appended before the next task starts.
    """
    pending = plan.pending(store)
    it = progress(pending) if progress is not None else pending
    done = 0
    for task in it:
        store.put(task, run_task(task, reader, workers=workers))
        done += 1
    return done

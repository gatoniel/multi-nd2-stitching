"""Turning offsets into absolute tile positions.

Pure: takes a layout, a plan and a store, returns coordinates. Touches no
pixels and writes nothing. If an offset is missing it says which one, rather
than raising KeyError or looping forever.
"""

from __future__ import annotations

from collections import deque

import attrs
import numpy as np


class MissingOffsets(Exception):
    def __init__(self, missing: list[str], hint: str = ""):
        self.missing = missing
        super().__init__(
            f"{len(missing)} offset(s) not computed yet:\n"
            + "\n".join(f"  - {m}" for m in missing[:10])
            + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
            + (f"\n{hint}" if hint else "")
        )


@attrs.frozen
class Coordinates:
    """(t, tile) -> zyx position, in a frame whose origin is arbitrary.

    `window` is the timepoint range that was actually asked for. Frames outside
    it may be present (the drift chain has to be walked from t=0 regardless) but
    they are placed best-effort and are excluded from `extent`, so blending a
    prefix does not size the canvas from tiles nobody asked about.
    """

    by_time: tuple[dict[str, np.ndarray], ...]
    window: tuple[int, int] = (0, 0)

    def at(self, t: int) -> dict[str, np.ndarray]:
        return self.by_time[t]

    def __getitem__(self, key):
        t, name = key
        return self.by_time[t][name]

    def extent(
        self, tile_shape, t0: int | None = None, t1: int | None = None
    ) -> np.ndarray:
        """(3, 2) array of min/max over placed tiles, in zyx."""
        t0 = self.window[0] if t0 is None else t0
        t1 = self.window[1] if t1 is None else t1
        placed = [c for frame in self.by_time[t0:t1] for c in frame.values()]
        if not placed:
            raise ValueError(f"no tiles placed in t={t0}..{t1 - 1}")
        arr = np.array(placed)
        return np.stack(
            (arr.min(axis=0), arr.max(axis=0) + np.array(tile_shape)), axis=1
        )


def _index_tasks(plan):
    time_by = {(t.name, t.t_to): t for t in plan.time_tasks}
    pair_by = {(p.a, p.b, p.axis, p.t): p for p in plan.pair_tasks}
    return time_by, pair_by


def build_coordinates(
    layout, plan, store, t0: int = 0, t1: int | None = None
) -> Coordinates:
    """Place tiles for t in [t0, t1).

    The drift chain is absolute: an anchor's position at t is its position at
    t-1 plus that step's offset, so timepoints before the window still have to
    be walked. Their *time* offsets are therefore required; their pair offsets
    are not, and a tile that cannot be placed before the window is simply left
    out rather than reported. Inside the window everything is required.
    """
    t1 = layout.nt if t1 is None else t1
    time_by, pair_by = _index_tasks(plan)
    frames: list[dict[str, np.ndarray]] = []
    missing: list[str] = []

    for t in range(t1):
        in_window = t >= t0
        here: dict[str, np.ndarray] = {}

        # --- seeds: anchors carry their own position forward through drift ---
        for name in layout.anchors_at(t):
            if t == 0 or name not in frames[t - 1]:
                if not here:
                    here[name] = np.zeros(3)  # first anchor defines the origin
                continue
            task = time_by.get((name, t))
            if task is None:
                continue
            offset = store.get(task.key)
            if offset is None:
                missing.append(task.describe())  # drift is needed either way
                continue
            here[name] = frames[t - 1][name] + offset.as_array()

        if not here and layout.tiles_at(t):
            # nothing anchored: fall back to the first alive tile at the origin
            here[layout.tiles_at(t)[0]] = np.zeros(3)

        # --- flood fill along the neighbour graph ---------------------------
        pairs = deque(layout.pairs_at(t))
        stalled = 0
        while pairs and stalled <= len(pairs):
            p = pairs.popleft()
            task = pair_by.get((p.a, p.b, p.axis, t))
            offset = store.get(task.key) if task is not None else None
            if offset is None:
                if task is not None and in_window:
                    missing.append(task.describe())
                stalled += 1
                continue
            arr = offset.as_array()
            if p.a in here and p.b not in here:
                here[p.b] = here[p.a] + arr
                stalled = 0
            elif p.b in here and p.a not in here:
                here[p.a] = here[p.b] - arr
                stalled = 0
            elif p.a in here and p.b in here:
                stalled += 1  # already placed, consistent or not
            else:
                pairs.append(p)  # neither end placed yet; try again later
                stalled += 1
        frames.append(here)

    frames.extend({} for _ in range(layout.nt - len(frames)))

    if missing:
        raise MissingOffsets(
            sorted(set(missing)),
            hint=f"run `stitch offsets --between {t0} {t1}` "
            "(drift also needs every step from t=0)",
        )
    return Coordinates(tuple(frames), window=(t0, t1))

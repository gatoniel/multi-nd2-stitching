"""Turning offsets into absolute tile positions.

Pure: takes a layout, a plan and a store, returns coordinates. Touches no
pixels and writes nothing. If an offset is missing it says which one, rather
than raising KeyError or looping forever.
"""

from __future__ import annotations

import attrs
import numpy as np

from .placement import DRIFT, ORIGIN, plan_placement


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

    def restrict(self, keep: dict[int, tuple]) -> Coordinates:
        """A view holding only `keep[t]` at each timepoint.

        Everything downstream -- extent, canvas geometry, the blend log key --
        reads the coordinates, so narrowing them here narrows all of it without
        any of those needing to know why. Positions are untouched, so a
        restricted canvas stays comparable with a full one.
        """
        return Coordinates(
            by_time=tuple(
                {n: c for n, c in frame.items() if n in set(keep.get(t, ()))}
                for t, frame in enumerate(self.by_time)
            ),
            window=self.window,
        )

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
        # The routing comes from placement.plan_placement, so the graph printed
        # by `stitch graph` is exactly the one walked here.
        for step in plan_placement(layout, t).steps:
            if step.kind == ORIGIN:
                here[step.tile] = np.zeros(3)
                continue
            if step.kind == DRIFT:
                task = time_by.get((step.tile, t))
                if task is None or step.tile not in frames[t - 1]:
                    continue
                offset = store.get(task.key)
                if offset is None:
                    missing.append(task.describe())  # drift is needed either way
                    continue
                here[step.tile] = frames[t - 1][step.tile] + offset.as_array()
                continue

            # a neighbour edge: the parent must already be placed
            forward = pair_by.get((step.via, step.tile, step.axis, t))
            backward = pair_by.get((step.tile, step.via, step.axis, t))
            task = forward or backward
            if task is None or step.via not in here:
                continue
            offset = store.get(task.key)
            if offset is None:
                if in_window:
                    missing.append(task.describe())
                continue
            arr = offset.as_array()
            here[step.tile] = here[step.via] + (arr if forward else -arr)

        frames.append(here)

    frames.extend({} for _ in range(layout.nt - len(frames)))

    if missing:
        raise MissingOffsets(
            sorted(set(missing)),
            hint=f"run `stitch offsets --between {t0} {t1}` "
            "(drift also needs every step from t=0)",
        )
    return Coordinates(tuple(frames), window=(t0, t1))

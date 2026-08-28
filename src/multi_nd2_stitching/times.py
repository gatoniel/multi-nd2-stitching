"""Real-world time per global timepoint of the stitched canvas.

Pure and free, like `coordinates.py`: everything here comes straight out of
the (cached) metadata, so it costs nothing to rebuild and needs no cache of
its own. No pixel data, no nd2 files touched.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import attrs
import numpy as np


@attrs.frozen
class TimeRow:
    t: int  # global timepoint
    file: int  # index into cfg.files
    local_t: int  # timepoint within that file
    real_time_s: float | None  # POSIX seconds, UTC; None if unrecorded
    skipped: bool  # real_time_s is None

    @property
    def real_time_iso(self) -> str | None:
        if self.real_time_s is None:
            return None
        return datetime.fromtimestamp(self.real_time_s, tz=UTC).isoformat()


def build_time_table(layout, meta) -> list[TimeRow]:
    """One row per global timepoint, in order."""
    rows = []
    for t in range(layout.nt):
        file_i, local_t = layout.locate(t)
        times = meta[file_i].real_time_s
        real_time_s = times[local_t] if local_t < len(times) else None
        rows.append(
            TimeRow(
                t=t,
                file=file_i,
                local_t=local_t,
                real_time_s=real_time_s,
                skipped=real_time_s is None,
            )
        )
    return rows


def write_csv(rows: list[TimeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "file", "local_t", "real_time_iso", "real_time_s", "skipped"])
        for r in rows:
            w.writerow(
                [
                    r.t,
                    r.file,
                    r.local_t,
                    r.real_time_iso or "",
                    r.real_time_s if r.real_time_s is not None else "",
                    r.skipped,
                ]
            )


def write_npy(rows: list[TimeRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("t", "i8"),
            ("file", "i8"),
            ("local_t", "i8"),
            ("real_time_s", "f8"),
            ("skipped", "?"),
        ]
    )
    arr = np.array(
        [
            (
                r.t,
                r.file,
                r.local_t,
                np.nan if r.real_time_s is None else r.real_time_s,
                r.skipped,
            )
            for r in rows
        ],
        dtype=dtype,
    )
    np.save(path, arr)


def write_parquet(rows: list[TimeRow], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "writing parquet needs pyarrow, which is not installed. "
            "Run `uv add pyarrow` (or pick --format csv/npy instead)."
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "t": [r.t for r in rows],
            "file": [r.file for r in rows],
            "local_t": [r.local_t for r in rows],
            "real_time_iso": [r.real_time_iso for r in rows],
            "real_time_s": [r.real_time_s for r in rows],
            "skipped": [r.skipped for r in rows],
        }
    )
    pq.write_table(table, path)


WRITERS = {"csv": write_csv, "npy": write_npy, "parquet": write_parquet}

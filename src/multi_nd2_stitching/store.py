"""Append-only offset cache.

One JSON object per line. Appending means an interrupted run loses nothing --
every result is durable the moment it is computed, so you can Ctrl-C at any
point and resume. Last line wins for a repeated key.

The file is greppable on purpose: when t=21 looks wrong, you want to read the
record, not decode an .npy.
"""

from __future__ import annotations

import json
from pathlib import Path

import attrs


@attrs.frozen
class Offset:
    """Result of one correlation, in zyx pixels."""

    dz: int
    dy: int
    dx: int

    @classmethod
    def of(cls, arr) -> Offset:
        z, y, x = (int(v) for v in arr)
        return cls(z, y, x)

    def as_array(self):
        import numpy as np

        return np.array([self.dz, self.dy, self.dx], dtype=int)


class OffsetStore:
    """dict-like, backed by a JSONL file."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self._data: dict[str, Offset] = {}
        self._meta: dict[str, dict] = {}
        if self.path is not None and self.path.exists():
            self._read()

    def _read(self) -> None:
        skipped = 0
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = Offset(**rec["offset"])
                    self._meta[rec["key"]] = rec.get("task", {})
                except Exception:
                    skipped += 1  # torn final line from a killed process
        self.skipped = skipped

    # --- mapping-ish ------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Offset:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def describe(self, key: str) -> dict:
        """The human-readable record for a key, for debugging."""
        return self._meta.get(key, {})

    # --- writing ----------------------------------------------------------
    def put(self, task, offset: Offset) -> None:
        self._data[task.key] = offset
        self._meta[task.key] = {"kind": task.kind, "what": task.describe()}
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "key": task.key,
                        "kind": task.kind,
                        "task": {"kind": task.kind, "what": task.describe()},
                        "offset": attrs.asdict(offset),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            f.flush()

    def compact(self) -> int:
        """Rewrite without superseded lines. Never required for correctness."""
        if self.path is None or not self.path.exists():
            return 0
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            for key, off in self._data.items():
                f.write(
                    json.dumps(
                        {
                            "key": key,
                            "kind": self._meta.get(key, {}).get("kind", ""),
                            "task": self._meta.get(key, {}),
                            "offset": attrs.asdict(off),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        tmp.replace(self.path)
        return len(self._data)

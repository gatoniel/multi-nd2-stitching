"""Where a run keeps its files.

Everything derived from `path/to/ch6.yaml` lives in `path/to/ch6/`. The config
is the only input; the folder holds nothing that cannot be rebuilt from it.
Deleting the folder costs runtime and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import attrs


@attrs.frozen
class Workspace:
    config_path: Path

    @classmethod
    def of(cls, config_path) -> Workspace:
        return cls(Path(config_path).expanduser().resolve())

    @property
    def root(self) -> Path:
        return self.config_path.parent / self.config_path.stem

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def offsets(self) -> Path:
        return self.root / "offsets.jsonl"

    @property
    def canvas(self) -> Path:
        return self.root / "canvas.zarr"

    def create(self) -> Workspace:
        self.root.mkdir(parents=True, exist_ok=True)
        return self

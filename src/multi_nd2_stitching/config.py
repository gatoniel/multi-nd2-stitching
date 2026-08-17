from dataclasses import dataclass, field
from typing import Optional
import yaml


import desert


def get_slices(slices, min_nz):
    if slices.stop[0] is None:
        z_stop = min_nz
    else:
        z_stop = min(min_nz, slices.stop[0])

    return (
        slice(
            slices.start[0],
            z_stop,
        ),
        slice(
            slices.start[1],
            slices.stop[1],
        ),
        slice(
            slices.start[2],
            slices.stop[2],
        ),
    )


@dataclass
class Position:
    start: tuple[int, int]
    aliases: Optional[list[str]]
    end: Optional[int]


@dataclass
class Slices:
    start: tuple[Optional[int], Optional[int], Optional[int]]
    stop: tuple[Optional[int], Optional[int], Optional[int]]


@dataclass
class StitchingConfig:
    files: list[str]
    names: dict[str, Position]
    manual_realignment_time: dict[str, list[int]]
    flip_x: bool
    flip_y: bool
    grid_spacing: float
    grid_spacing_error: float
    shift_px: Optional[int]
    slices: Optional[Slices]
    realignment_slices: Optional[Slices]
    start_names: dict[str, list[int]]
    ignore_timepoints: Optional[dict[str, list[int]]] = field(default_factory=dict)
    start_names_manual: Optional[dict[str, list[int]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.ignore_timepoints is None:
            self.ignore_timepoints = {}
        if self.start_names_manual is None:
            self.start_names_manual = {}


def load_config(file):
    with open(file) as f:
        schema = desert.schema(StitchingConfig)
        config = schema.load(yaml.safe_load(f))

    if config.slices is None:
        config.slices = Slices(start=(None, None, None), stop=(None, None, None))
    if config.realignment_slices is None:
        config.realignment_slices = Slices(
            start=(None, None, None), stop=(None, None, None)
        )
    return config

import attrs
from cattrs.preconf.pyyaml import make_converter

Slices3D = tuple[slice, slice, slice]
_AXES = ("z", "y", "x")


def _empty_slices() -> Slices3D:
    return (slice(None), slice(None), slice(None))


def _structure_slices(data, _) -> Slices3D:
    data = {} if data is None else data
    return tuple(slice(*data.get(axis, (None, None))) for axis in _AXES)


def _structure_at(data, _) -> tuple[int, ...]:
    """`at: 143` and `at: [4, 16]` both mean a set of timepoints."""
    if data is None:
        return ()
    if isinstance(data, int):
        return (data,)
    return tuple(data)


def clamp_z(slices: Slices3D, min_nz: int) -> Slices3D:
    """Cap the z-axis stop at the smallest stack depth across tiles."""
    z, y, x = slices
    z_stop = min_nz if z.stop is None else min(z.stop, min_nz)
    return (slice(z.start, z_stop), y, x)


Timepoints = tuple[int, ...]

converter = make_converter()
converter.register_structure_hook_func(lambda t: t is Slices3D, _structure_slices)
converter.register_structure_hook_func(lambda t: t is Timepoints, _structure_at)

# NOTE: the annotation must admit None, or cattrs raises while structuring and
# the converter never runs. `= None` alone is not enough.
_none_to_list = attrs.converters.default_if_none(factory=list)


@attrs.define(kw_only=True)
class Position:
    """A tile, identified by name, tracked across a contiguous run of files.

    start: (file index, timepoint within that file) where the tile first appears
    end:   file index at which it stops existing, EXCLUSIVE. None = until the end.
           `start: [3, 20], end: 8` means alive in files 3..7.
    """

    start: tuple[int, int]
    end: int | None = None
    aliases: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    reference_in_files: list[int] | None = attrs.field(
        factory=list, converter=_none_to_list
    )

    def alive_in_file(self, file_i: int, n_files: int) -> bool:
        end = n_files if self.end is None else self.end
        return self.start[0] <= file_i < end

    def last_file(self, n_files: int) -> int:
        return (n_files if self.end is None else self.end) - 1


@attrs.define(kw_only=True)
class Override:
    """One manual intervention at one or more timepoints.

    Groups the edits that belong together. Dropping a tile can disconnect the
    neighbour graph, which then needs a fresh anchor at the same timepoint --
    those two edits are one decision and live in one block.

    The four verbs:
      drop      the tile is not there at all: no coordinate, no neighbour edges
      unanchor  the tile stays and is still placed, but not by drifting from
                t-1 -- it hangs off a neighbour instead
      anchor    the tile is placed by drifting from t-1, in addition to
                whatever reference_in_files says
      realign   recompute this tile's drift step using realignment_slices

    `unanchor` is what you want when a tile should keep its place in the mosaic
    while some *other* tile carries the drift across a timepoint. Two anchors in
    one connected component over-determine it, so handing over means unanchoring
    one as you anchor the other.
    """

    at: Timepoints
    reason: str | None = None
    drop: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    unanchor: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    anchor: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    realign: list[str] | None = attrs.field(factory=list, converter=_none_to_list)

    @property
    def names(self) -> set[str]:
        return (
            set(self.drop) | set(self.unanchor) | set(self.anchor) | set(self.realign)
        )


@attrs.define(kw_only=True)
class StitchingConfig:
    files: list[str]
    positions: dict[str, Position]
    grid_spacing: float
    grid_spacing_error: float
    overrides: list[Override] | None = attrs.field(
        factory=list, converter=_none_to_list
    )
    flip_x: bool = False
    flip_y: bool = False
    shift_px: int | None = None
    slices: Slices3D = attrs.field(factory=_empty_slices)
    realignment_slices: Slices3D = attrs.field(factory=_empty_slices)

    @property
    def n_files(self) -> int:
        return len(self.files)

    def references_for_file(self, file_i: int) -> list[str]:
        return [n for n, p in self.positions.items() if file_i in p.reference_in_files]

    def dropped_at(self, t: int) -> set[str]:
        return {n for o in self.overrides if t in o.at for n in o.drop}

    def unanchored_at(self, t: int) -> set[str]:
        return {n for o in self.overrides if t in o.at for n in o.unanchor}

    def anchored_at(self, t: int) -> set[str]:
        return {n for o in self.overrides if t in o.at for n in o.anchor}

    def realigned_at(self, t: int) -> set[str]:
        return {n for o in self.overrides if t in o.at for n in o.realign}


def loads_config(text: str) -> StitchingConfig:
    return converter.loads(text, StitchingConfig)


def load_config(path) -> StitchingConfig:
    with open(path) as f:
        return loads_config(f.read())

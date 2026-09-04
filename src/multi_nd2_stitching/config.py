import re

import attrs
from cattrs.preconf.pyyaml import make_converter

Slices3D = tuple[slice, slice, slice]
_AXES = ("z", "y", "x")


def _empty_slices() -> Slices3D:
    return (slice(None), slice(None), slice(None))


def _structure_slices(data, _) -> Slices3D:
    data = {} if data is None else data
    return tuple(slice(*data.get(axis, (None, None))) for axis in _AXES)


_AT_RANGE = re.compile(r"(\d+)-(\d+)")


def _expand_at_range(s: str) -> range:
    m = _AT_RANGE.fullmatch(s)
    if not m:
        raise ValueError(
            f"'{s}' is not a valid 'at' entry: expected an integer or 'start-end'"
        )
    lo, hi = int(m[1]), int(m[2])
    if hi < lo:
        raise ValueError(f"'{s}': range end {hi} is before start {lo}")
    return range(lo, hi + 1)  # inclusive


def _structure_at(data, _) -> tuple[int, ...]:
    """`at: 143`, `at: "20-40"`, and a list mixing either -- e.g.
    `at: [4, 16, "20-40"]` -- all mean a set of timepoints. A range string is
    inclusive on both ends."""
    if data is None:
        return ()
    if isinstance(data, (int, str)):
        data = [data]
    out: list[int] = []
    for entry in data:
        out.extend(_expand_at_range(entry) if isinstance(entry, str) else [entry])
    return tuple(out)


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
_none_to_dict = attrs.converters.default_if_none(factory=dict)


@attrs.define(kw_only=True)
class Position:
    """A tile, identified by name, tracked across a contiguous run of files.

    start: (file index, timepoint within that file) where the tile first appears
    end:   file index at which it stops existing, EXCLUSIVE. None = until the end.
           `start: [3, 20], end: 8` means alive in files 3..7.
    position_in_files: {3: 2} forces file 3's position 2 for this tile,
           bypassing name matching there entirely -- for files whose positions
           were never named (nd2 forbids adding one after acquisition, so
           there is no other way to resolve them). Files not listed still
           resolve by name as before; a tile can mix both across its files.
    missing_in_files: file indices, within [start, end), where this tile has
           no position at all -- an expected gap, not an error. The tile can
           still be alive again in a later file; `drop` cannot express this,
           since it only removes an already-resolved tile from a timepoint,
           and resolution here would fail first.
    """

    start: tuple[int, int]
    end: int | None = None
    aliases: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    reference_in_files: list[int] | None = attrs.field(
        factory=list, converter=_none_to_list
    )
    position_in_files: dict[int, int] | None = attrs.field(
        factory=dict, converter=_none_to_dict
    )
    missing_in_files: list[int] | None = attrs.field(
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

    The five verbs:
      drop         the tile is not there at all: no coordinate, no neighbour edges
      unanchor     the tile stays and is still placed, but not by drifting from
                   t-1 -- it hangs off a neighbour instead
      anchor       the tile is placed by drifting from t-1, in addition to
                   whatever reference_in_files says
      realign      recompute using realignment_slices instead of slices --
                   a bare tile name (its drift step) or an "a,b" pair (a
                   neighbour correlation, same convention as shaped_peak)
      shaped_peak  pick the correlation peak by shape instead of raw height --
                   see StitchingConfig.shaped_peak_at
      corner       fit and use a diagonal Corner as a placement edge -- see
                   StitchingConfig.corner_at

    `unanchor` is what you want when a tile should keep its place in the mosaic
    while some *other* tile carries the drift across a timepoint. Two anchors in
    one connected component over-determine it, so handing over means unanchoring
    one as you anchor the other.

    `shaped_peak` entries name either a bare tile name (a drift step) or an
    `"a,b"` pair (a neighbour correlation, the same convention `stitch inspect
    --pair` uses) -- unlike the other three verbs, it never changes which tiles
    exist or how they're connected, only how one correlation's peak is picked,
    so it is deliberately not part of `names`. `realign` names either form too,
    for the identical reason: recomputing with a different crop doesn't change
    graph membership either, so it is also excluded from `names`.

    `near` gives a rough manual estimate of the true offset for a
    `shaped_peak` entry -- `{"a,b": [dz, dy, dx]}`, in the same units already
    shown in `candidates.csv`'s `dz,dy,dx` columns or `offsets.csv`'s drift
    columns. When present, the peak search for that name is restricted to a
    window around this point instead of ranked by shape at all -- see
    `compute._windowed_peak_index`. Every key must also appear in
    `shaped_peak`; a hint with nothing to attach to is an error, not a
    silent no-op.

    `corner` promotes a diagonal (`layout.Corner`) relationship into a real
    placement edge, fitted by its own FFT correlation -- for a component with
    no edge-adjacent path to an anchor at all. Always an `"a,b"` pair (a
    corner has no drift-step equivalent, unlike `shaped_peak`/`realign`), and
    -- for the same reason those are excluded from `names` -- so is this: it
    adds a route, but names/drops/anchors nothing.
    """

    at: Timepoints
    reason: str | None = None
    drop: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    unanchor: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    anchor: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    realign: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    shaped_peak: list[str] | None = attrs.field(factory=list, converter=_none_to_list)
    near: dict[str, list[int]] | None = attrs.field(
        factory=dict, converter=_none_to_dict
    )
    corner: list[str] | None = attrs.field(factory=list, converter=_none_to_list)

    @property
    def names(self) -> set[str]:
        return set(self.drop) | set(self.unanchor) | set(self.anchor)


@attrs.define(kw_only=True)
class Overview:
    """A wide-field image to orient the stitched tiles within.

    `channel` is a direct index into overview.nd2's own P axis -- unlike tile
    positions, which are matched by name, the overview file's positions are
    picked by number. `None` when the file has no P axis at all, or exactly
    one. Not a fluorescence channel.
    """

    file: str
    channel: int | None = None
    label: bool = True
    # Overrides overview.nd2's own (possibly wrong) header pixel size.
    pixel_size_um: float | None = None
    # Longer side of the exported PNG, in px; the downsample factor is derived
    # from this, never picked by hand.
    max_output_px: int = 2000
    # "mean" or "median" block-reduce; validated in validate.py.
    reduction: str = "mean"


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
    overview: Overview | None = None
    # The global timeline ends here, EXCLUSIVE (same convention as
    # Position.end) -- timepoints from stop_at on are simply not part of the
    # run, as if the files were shorter. For the tail after an experiment
    # ends but imaging continues; unset, the whole timeline is used.
    stop_at: int | None = None
    # Raw global timepoints (same numbering as stop_at and Override.at --
    # i.e. counted straight off the concatenated files, unaffected by any
    # exclusion) that are cut out of the timeline entirely: no tile, no
    # canvas frame, no `times` row. Unlike `drop`, which removes one tile
    # from an otherwise-real timepoint, this removes the timepoint itself --
    # for the case where *nothing* at it is worth keeping. Everything
    # downstream is renumbered around the gap, so `layout.nt` (and every t
    # a command like `stitch graph --at` or `stitch blend --between`
    # addresses) counts only the timepoints that survive; a tile's drift
    # step across the gap correlates directly against the last surviving
    # timepoint rather than the missing ones, which is the point -- a jump
    # is preferable to correlating against blank frames. `at:` in overrides
    # stays in this same raw numbering regardless of what exclude_at
    # removes, so edits to one never renumber the other.
    exclude_at: Timepoints = attrs.field(factory=tuple)

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

    def shaped_peak_at(self, t: int) -> set[str]:
        """Tile names and/or 'a,b' pair strings, whichever this override names."""
        return {n for o in self.overrides if t in o.at for n in o.shaped_peak}

    def near_hint(self, name: str, t: int) -> tuple[int, int, int] | None:
        """The rough (dz, dy, dx) estimate for `name` at `t`, if one was given."""
        for o in self.overrides:
            if t in o.at and name in o.near:
                return tuple(o.near[name])
        return None

    def corner_at(self, t: int) -> set[str]:
        """'a,b' pair strings naming a Corner to fit and use as a placement
        edge at this timepoint."""
        return {n for o in self.overrides if t in o.at for n in o.corner}


def loads_config(text: str) -> StitchingConfig:
    return converter.loads(text, StitchingConfig)


def load_config(path) -> StitchingConfig:
    with open(path) as f:
        return loads_config(f.read())

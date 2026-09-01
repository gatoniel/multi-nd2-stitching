"""Geometry and timeline: a pure function of (config, metadata).

Nothing here is expensive and nothing here is cached. It is rebuilt from scratch
on every run, which is what makes the result depend only on the YAML.
"""

from __future__ import annotations

from itertools import combinations

import attrs
import numpy as np

from .config import StitchingConfig
from .metadata import Metadata


@attrs.frozen(order=True)
class Pair:
    """Two tiles that overlap along one axis. `axis` is 1 (y) or 2 (x) in zyx."""

    a: str
    b: str
    axis: int


@attrs.frozen(order=True)
class Corner:
    """Two tiles diagonally adjacent in the nominal grid -- one step away in
    BOTH y and x at once, not just one axis.

    `_discover_pairs` only ever connects edge-adjacent tiles, so a diagonal
    relationship like this never becomes a `Pair` and `blend_weights` never
    tapers for it -- even though the tiles genuinely overlap, in the small
    rectangle near the corner. This is true whether or not some other tile
    occupies the corner itself; `a`/`b` are just the two names in sorted
    order (unlike `Pair`, direction is not baked in here -- `blend_weights`
    reads the real, possibly-drifted signed offset straight out of
    `coords.at(t)`, same as it already does for a `Pair`'s ramp length)."""

    a: str
    b: str


@attrs.frozen
class Tile:
    name: str
    aliases: tuple[str, ...]
    first_t: int  # global timepoint it appears
    last_t: int  # global timepoint it disappears, EXCLUSIVE
    position: tuple[int | None, ...]  # position index per file, None where absent


@attrs.define
class Layout:
    config: StitchingConfig
    tiles: tuple[str, ...]
    tile: dict[str, Tile]
    pairs: tuple[Pair, ...]
    corners: tuple[Corner, ...]
    nts: tuple[int, ...]
    file_start: tuple[int, ...]  # global t at which each file begins
    nt: int
    raw_nt: int  # nt before any stop_at truncation; nt itself for reporting only
    nz: int  # min stack depth across files
    ny: int
    nx: int
    shift_px: int
    tile_alive: np.ndarray  # (nt, n_tiles) bool
    pair_alive: np.ndarray  # (nt, n_pairs) bool
    corner_alive: np.ndarray  # (nt, n_corners) bool
    is_anchor: np.ndarray  # (nt, n_tiles) bool

    # --- lookups ---------------------------------------------------------
    def ti(self, name: str) -> int:
        return self.tiles.index(name)

    def locate(self, t: int) -> tuple[int, int]:
        """Global timepoint -> (file index, timepoint within that file)."""
        if not 0 <= t < self.nt:
            raise IndexError(f"t={t} outside timeline 0..{self.nt - 1}")
        file_i = int(np.searchsorted(self.file_start, t, side="right") - 1)
        return file_i, t - self.file_start[file_i]

    def tiles_at(self, t: int) -> list[str]:
        return [n for i, n in enumerate(self.tiles) if self.tile_alive[t, i]]

    def pairs_at(self, t: int) -> list[Pair]:
        return [p for k, p in enumerate(self.pairs) if self.pair_alive[t, k]]

    def corners_at(self, t: int) -> list[Corner]:
        return [c for k, c in enumerate(self.corners) if self.corner_alive[t, k]]

    def anchors_at(self, t: int) -> list[str]:
        return [n for i, n in enumerate(self.tiles) if self.is_anchor[t, i]]

    def frame_index(self, name: str, t: int) -> tuple[int, int]:
        """(file index, position index) for a tile at a global timepoint."""
        file_i, _ = self.locate(t)
        pos = self.tile[name].position[file_i]
        if pos is None:
            raise ValueError(f"'{name}' has no position in file {file_i} (t={t})")
        return file_i, pos


def _canvas_after(stage_delta: float, flip: bool) -> bool:
    """Is the point this delta was measured *from* canvas-positioned after
    the point it's relative *to*, along one axis? (`stage_delta = here[source]
    - here[ref]`.) Mirrors exactly the rule `_discover_pairs` already uses to
    decide `Pair.a`/`.b`: unflipped, a *smaller* raw stage delta means
    canvas-after (proven by `coordinates.py`: `Pair.b` is always canvas-after
    `Pair.a`, and `_discover_pairs` picks whichever tile has the *larger* raw
    stage coordinate as `a`). `flip_x`/`flip_y` each invert that for their own
    axis, which is the entire reason they exist.
    """
    return (stage_delta < 0) != flip


def _discover_pairs(
    cfg: StitchingConfig, meta: Metadata, tile: dict[str, Tile]
) -> tuple[Pair, ...]:
    """Infer adjacency from stage coordinates, unioned over all files."""
    lo = cfg.grid_spacing - cfg.grid_spacing_error
    hi = cfg.grid_spacing + cfg.grid_spacing_error
    tol = cfg.grid_spacing_error

    found: set[Pair] = set()
    for file_i, fm in enumerate(meta.files):
        here = {
            name: np.array(fm.stage_um[t.position[file_i]])
            for name, t in tile.items()
            if t.position[file_i] is not None
        }
        for n_i, n_j in combinations(sorted(here), 2):
            dx, dy = here[n_i] - here[n_j]
            if abs(dx) < tol:
                if lo < dy < hi:
                    found.add(Pair(n_i, n_j, 1))
                elif -hi < dy < -lo:
                    found.add(Pair(n_j, n_i, 1))
            if abs(dy) < tol:
                if lo < dx < hi:
                    found.add(Pair(n_i, n_j, 2))
                elif -hi < dx < -lo:
                    found.add(Pair(n_j, n_i, 2))

    pairs = list(found)
    if cfg.flip_x:
        pairs = [Pair(p.b, p.a, p.axis) if p.axis == 2 else p for p in pairs]
    if cfg.flip_y:
        pairs = [Pair(p.b, p.a, p.axis) if p.axis == 1 else p for p in pairs]
    return tuple(sorted(pairs))


def _discover_corners(
    cfg: StitchingConfig, meta: Metadata, tile: dict[str, Tile]
) -> tuple[Corner, ...]:
    """Grid-diagonal adjacency: one step away in BOTH x and y at once.

    Unlike `_discover_pairs`, direction is not baked into `a`/`b` here (they
    are just sorted names) -- `blend_weights` reads the real signed offset
    straight out of `coords.at(t)` at blend time, so no flip_x/flip_y
    handling is needed: whichever way the placement pipeline actually
    resolves the two tiles is what gets used.
    """
    lo = cfg.grid_spacing - cfg.grid_spacing_error
    hi = cfg.grid_spacing + cfg.grid_spacing_error

    found: set[Corner] = set()
    for file_i, fm in enumerate(meta.files):
        here = {
            name: np.array(fm.stage_um[t.position[file_i]])
            for name, t in tile.items()
            if t.position[file_i] is not None
        }
        for n_i, n_j in combinations(sorted(here), 2):
            dx, dy = here[n_i] - here[n_j]
            if lo < abs(dx) < hi and lo < abs(dy) < hi:
                found.add(Corner(n_i, n_j))
    return tuple(sorted(found))


def corner_direction(
    cfg: StitchingConfig, meta: Metadata, tile: dict[str, Tile], a: str, b: str
) -> tuple[int, int]:
    """Nominal canvas (dy_sign, dx_sign) of `b` relative to `a`, from stage
    coordinates alone.

    `blend.py`'s corner taper can read the real, drifted offset out of
    `coords.at(t)` because both tiles are already placed by blend time. This
    is for the opposite situation -- building a `CornerTask`'s *crop*, before
    either tile necessarily has a coordinate at all (that's the point of
    fitting a corner: there is no other route yet) -- so it has to come from
    the nominal grid instead, the same way `_discover_pairs` already fixes a
    `Pair`'s direction from `flip_x`/`flip_y`. `Corner.a`/`.b` themselves stay
    untouched (still just sorted names -- blend.py doesn't need this and
    nothing else should start depending on a direction baked in there).
    """
    for file_i, fm in enumerate(meta.files):
        pa, pb = tile[a].position[file_i], tile[b].position[file_i]
        if pa is None or pb is None:
            continue
        dx, dy = np.array(fm.stage_um[pb]) - np.array(fm.stage_um[pa])
        return (
            1 if _canvas_after(dy, cfg.flip_y) else -1,
            1 if _canvas_after(dx, cfg.flip_x) else -1,
        )
    raise ValueError(f"'{a}' and '{b}' are never both alive in the same file")


def build_layout(cfg: StitchingConfig, meta: Metadata) -> Layout:
    n_files = len(meta)
    if n_files != cfg.n_files:
        raise ValueError(f"config lists {cfg.n_files} files, metadata has {n_files}")

    nys = {f.ny for f in meta.files}
    nxs = {f.nx for f in meta.files}
    if len(nys) != 1 or len(nxs) != 1:
        raise ValueError(f"tile size differs across files: ny={nys}, nx={nxs}")

    nts = meta.nts
    file_start = tuple(int(s) for s in np.concatenate([[0], np.cumsum(nts)[:-1]]))
    raw_nt = int(sum(nts))
    # An experiment can end before the acquisition does; stop_at truncates the
    # *effective* timeline right here, so nothing downstream -- masks below,
    # build_plan, coordinates, blend -- ever sees a timepoint past it. They
    # all already loop over `layout.nt`, never raw metadata, so this one
    # change is enough; no per-module truncation logic needed anywhere else.
    nt = raw_nt if cfg.stop_at is None else min(raw_nt, cfg.stop_at)

    # --- tiles ------------------------------------------------------------
    tiles = tuple(sorted(cfg.positions))
    tile: dict[str, Tile] = {}
    # (file, position index) -> tile name, across every tile resolved so far --
    # an override index is easy to typo into another tile's slot in a way a
    # name match structurally could not, so this is checked regardless of how
    # either tile's position got resolved.
    seen_positions: dict[tuple[int, int], str] = {}
    for name in tiles:
        pos = cfg.positions[name]
        names = (name, *(a for a in pos.aliases if a != name))
        end_file = n_files if pos.end is None else pos.end
        for i, pi in pos.position_in_files.items():
            if pos.start[0] <= i < end_file and not 0 <= pi < len(
                meta[i].position_names
            ):
                raise ValueError(
                    f"'{name}'.position_in_files[{i}] = {pi} is out of range "
                    f"(file {i} has {len(meta[i].position_names)} position(s))"
                )
        position = tuple(
            (
                None
                if i in pos.missing_in_files
                else pos.position_in_files[i]
                if i in pos.position_in_files
                else meta[i].position_of(names)
            )
            if pos.start[0] <= i < end_file
            else None
            for i in range(n_files)
        )
        missing = [
            i
            for i in range(pos.start[0], end_file)
            if position[i] is None and i not in pos.missing_in_files
        ]
        if missing:
            raise ValueError(
                f"'{name}' should be alive in files {pos.start[0]}..{end_file - 1} "
                f"but no matching position exists in files {missing}"
            )
        for i, pi in enumerate(position):
            if pi is None:
                continue
            key = (i, pi)
            if key in seen_positions:
                raise ValueError(
                    f"'{name}' and '{seen_positions[key]}' both resolve to "
                    f"position {pi} in file {i}"
                )
            seen_positions[key] = name
        tile[name] = Tile(
            name=name,
            aliases=names,
            first_t=file_start[pos.start[0]] + pos.start[1],
            last_t=nt
            if pos.end is None
            else file_start[end_file - 1] + nts[end_file - 1],
            position=position,
        )

    pairs = _discover_pairs(cfg, meta, tile)
    corners = _discover_corners(cfg, meta, tile)

    # --- masks ------------------------------------------------------------
    tile_alive = np.zeros((nt, len(tiles)), dtype=bool)
    is_anchor = np.zeros((nt, len(tiles)), dtype=bool)
    for i, name in enumerate(tiles):
        tile_alive[tile[name].first_t : tile[name].last_t, i] = True
        # A gap cut out of that otherwise-contiguous range -- the tile can be
        # alive again in a later file, so this is not the same as end.
        for f in cfg.positions[name].missing_in_files:
            tile_alive[file_start[f] : file_start[f] + nts[f], i] = False
        for f in cfg.positions[name].reference_in_files:
            is_anchor[file_start[f] : file_start[f] + nts[f], i] = True

    # Overrides in three passes, so the result does not depend on the order the
    # blocks happen to be written in: everything is dropped, then unanchored,
    # then anchored. A tile listed in both unanchor and anchor is contradictory
    # and is caught by validate, not silently resolved here.
    for o in cfg.overrides:
        for t in o.at:
            for name in o.drop:
                tile_alive[t, tiles.index(name)] = False
    for o in cfg.overrides:
        for t in o.at:
            for name in o.unanchor:
                is_anchor[t, tiles.index(name)] = False
    for o in cfg.overrides:
        for t in o.at:
            for name in o.anchor:
                is_anchor[t, tiles.index(name)] = True
    is_anchor &= tile_alive

    pair_alive = np.zeros((nt, len(pairs)), dtype=bool)
    for k, p in enumerate(pairs):
        pair_alive[:, k] = (
            tile_alive[:, tiles.index(p.a)] & tile_alive[:, tiles.index(p.b)]
        )

    corner_alive = np.zeros((nt, len(corners)), dtype=bool)
    for k, c in enumerate(corners):
        corner_alive[:, k] = (
            tile_alive[:, tiles.index(c.a)] & tile_alive[:, tiles.index(c.b)]
        )

    shift_px = (
        cfg.shift_px
        if cfg.shift_px is not None
        else int(cfg.grid_spacing / meta[0].voxel_x_um)
    )

    return Layout(
        config=cfg,
        tiles=tiles,
        tile=tile,
        pairs=pairs,
        corners=corners,
        nts=nts,
        file_start=file_start,
        nt=nt,
        raw_nt=raw_nt,
        nz=min(f.nz for f in meta.files),
        ny=nys.pop(),
        nx=nxs.pop(),
        shift_px=shift_px,
        tile_alive=tile_alive,
        pair_alive=pair_alive,
        corner_alive=corner_alive,
        is_anchor=is_anchor,
    )

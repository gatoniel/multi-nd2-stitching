"""Semantic checks cattrs cannot do.

Two tiers:
  check(cfg)                  -- config only, no ND2 files touched. Fast, always run.
  check(cfg, timeline=...)    -- adds checks that need nts (timepoints per file).
Anything phrased in *global* timepoints (overrides) is only fully checkable in
tier 2, because the global time axis is the concatenation of the per-file ones.
"""

from pathlib import Path

from .config import StitchingConfig


class ConfigError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            f"{len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems)
        )


_FULL_VOLUME = (slice(None), slice(None), slice(None))


def _check_realign(cfg: StitchingConfig, p: list[str]) -> None:
    """A realign step must actually crop differently, or it does nothing.

    `realign` recomputes a drift offset using `realignment_slices` instead of
    `slices`. The crop is part of the cache key, so if the two crops are equal
    the recomputed offset has the same key and the same value as the original:
    the override is silently a no-op, and the run looks like it honoured it.
    """
    names = sorted({n for o in cfg.overrides for n in o.realign})
    if not names:
        return
    where = f"realign is used for {names}, but "
    if cfg.realignment_slices == _FULL_VOLUME:
        p.append(
            where + "realignment_slices is not set, so those steps would be "
            "recomputed over the whole volume. Set realignment_slices to the "
            "region that should be correlated instead."
        )
    elif cfg.realignment_slices == cfg.slices:
        p.append(
            where + "realignment_slices is identical to slices, so those steps "
            "would recompute to exactly the same offsets. Make them differ, or "
            "drop the realign."
        )


def _check_positions(cfg: StitchingConfig, p: list[str]) -> None:
    n = cfg.n_files
    for name, pos in cfg.positions.items():
        where = f"positions.{name}"
        f0, t0 = pos.start
        if not 0 <= f0 < n:
            p.append(f"{where}.start: file index {f0} out of range (0..{n - 1})")
        if t0 < 0:
            p.append(f"{where}.start: negative timepoint {t0}")
        if pos.end is not None:
            if not 0 < pos.end <= n:
                p.append(f"{where}.end: {pos.end} out of range (1..{n}, exclusive)")
            elif pos.end <= f0:
                p.append(
                    f"{where}.end ({pos.end}) <= start file ({f0}): tile is never alive"
                )
        if name in pos.aliases:
            p.append(f"{where}.aliases: contains its own name (redundant)")
        clash = set(pos.aliases) & (set(cfg.positions) - {name})
        if clash:
            p.append(f"{where}.aliases: {sorted(clash)} is also a position name")

        # A reference must be alive in the file it anchors.
        for f in sorted(set(pos.reference_in_files)):
            if not 0 <= f < n:
                p.append(f"{where}.reference_in_files: file index {f} out of range")
            elif not pos.alive_in_file(f, n):
                p.append(
                    f"{where}.reference_in_files: anchors file {f} but is only alive "
                    f"in files {f0}..{pos.last_file(n)}"
                )
        if len(set(pos.reference_in_files)) != len(pos.reference_in_files):
            p.append(f"{where}.reference_in_files: contains duplicates")


def _check_coverage(cfg: StitchingConfig, p: list[str]) -> None:
    uncovered = [f for f in range(cfg.n_files) if not cfg.references_for_file(f)]
    if uncovered:
        p.append(
            f"files {uncovered} have no reference position: the drift chain breaks there"
        )


def _check_overrides(cfg: StitchingConfig, p: list[str]) -> None:
    known = set(cfg.positions)
    seen: dict[int, int] = {}
    for i, o in enumerate(cfg.overrides):
        where = f"overrides[{i}]"
        if not o.at:
            p.append(f"{where}.at: empty")
        for t in o.at:
            if t < 0:
                p.append(f"{where}.at: negative timepoint {t}")
            if t in seen and seen[t] != i:
                p.append(
                    f"{where}.at: timepoint {t} also handled by overrides[{seen[t]}]; "
                    "merge them into one block"
                )
            seen.setdefault(t, i)
        unknown = sorted(o.names - known)
        if unknown:
            p.append(f"{where}: unknown position(s) {unknown}")
        if not o.names:
            p.append(f"{where}: no drop/anchor/realign, block does nothing")
        both = sorted(set(o.drop) & (set(o.anchor) | set(o.realign)))
        if both:
            p.append(f"{where}: {both} dropped and used in the same block")
        contradictory = sorted(set(o.unanchor) & set(o.anchor))
        if contradictory:
            p.append(f"{where}: {contradictory} both unanchored and anchored; pick one")
        redundant = sorted(set(o.unanchor) & set(o.drop))
        if redundant:
            p.append(
                f"{where}: {redundant} unanchored and dropped; dropping already "
                "removes the anchor, so the unanchor does nothing"
            )
        if 0 in o.at and o.realign:
            p.append(f"{where}: realign at t=0, which has no predecessor")
        # The coupling this schema exists to make visible.
        dropped_refs = [
            n
            for n in list(o.drop) + list(o.unanchor)
            if n in known and cfg.positions[n].reference_in_files
        ]
        if dropped_refs and not o.anchor:
            p.append(
                f"{where}: removes the anchor from {dropped_refs} without naming "
                "a replacement; the graph may be unanchored at these timepoints"
            )


def _check_timeline(cfg: StitchingConfig, nts, p: list[str]) -> None:
    if len(nts) != cfg.n_files:
        p.append(f"timeline: got {len(nts)} file lengths for {cfg.n_files} files")
        return
    starts = [sum(nts[:i]) for i in range(cfg.n_files)]
    nt = sum(nts)

    for name, pos in cfg.positions.items():
        f0, t0 = pos.start
        if 0 <= f0 < cfg.n_files and t0 >= nts[f0]:
            p.append(
                f"positions.{name}.start: timepoint {t0} beyond file {f0} (has {nts[f0]})"
            )

    for i, o in enumerate(cfg.overrides):
        for t in o.at:
            if t >= nt:
                p.append(f"overrides[{i}].at: timepoint {t} beyond timeline (nt={nt})")
                continue
            file_i = max(f for f in range(cfg.n_files) if starts[f] <= t)
            for name in sorted(o.names):
                pos = cfg.positions.get(name)
                if pos is None:
                    continue
                if not pos.alive_in_file(file_i, cfg.n_files):
                    p.append(
                        f"overrides[{i}]: t={t} is in file {file_i}, where "
                        f"'{name}' is not alive"
                    )
                elif t < starts[file_i] + pos.start[1] and file_i == pos.start[0]:
                    p.append(f"overrides[{i}]: t={t} is before '{name}' starts")


def check(cfg: StitchingConfig, nts=None, check_files: bool = False) -> list[str]:
    """Return every problem found. Empty list == config is internally consistent."""
    p: list[str] = []
    if cfg.n_files == 0:
        p.append("files: empty")
    if len(set(cfg.files)) != cfg.n_files:
        p.append("files: contains duplicates")
    if not cfg.positions:
        p.append("positions: empty")
    if cfg.grid_spacing_error <= 0:
        p.append("grid_spacing_error must be > 0")
    elif cfg.grid_spacing_error >= cfg.grid_spacing / 2:
        p.append(
            f"grid_spacing_error ({cfg.grid_spacing_error}) >= half of grid_spacing "
            f"({cfg.grid_spacing}): neighbour detection windows overlap"
        )
    if check_files:
        for f in cfg.files:
            if not Path(f).exists():
                p.append(f"files: {f} does not exist")

    _check_positions(cfg, p)
    _check_coverage(cfg, p)
    _check_overrides(cfg, p)
    _check_realign(cfg, p)
    if nts is not None:
        _check_timeline(cfg, list(nts), p)
    return p


def validate(cfg: StitchingConfig, nts=None, check_files: bool = False):
    problems = check(cfg, nts=nts, check_files=check_files)
    if problems:
        raise ConfigError(problems)
    return cfg


# =============================================================================
# Tier 3: needs metadata (neighbour graph). Still no pixel data.
# =============================================================================


def _ranges(ts) -> str:
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    ts = sorted(ts)
    out, start, prev = [], ts[0], ts[0]
    for t in [*ts[1:], None]:
        if t != prev + 1:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = t
        prev = t
    return ", ".join(out)


def _components(names, pairs):
    """Union-find over the alive tiles."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if p.a in parent and p.b in parent:
            ra, rb = find(p.a), find(p.b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def check_layout(layout) -> list[str]:
    """The check that catches a drop disconnecting the graph.

    At every timepoint, every connected component of alive tiles must contain an
    anchor -- otherwise final_coordinates cannot place it and its flood-fill loop
    never terminates.
    """
    p: list[str] = []
    unanchored: dict[tuple[str, ...], list[int]] = {}
    empty: list[int] = []
    late_anchor: dict[str, list[int]] = {}

    for t in range(layout.nt):
        alive = layout.tiles_at(t)
        if not alive:
            empty.append(t)
            continue
        anchors = set(layout.anchors_at(t))
        for comp in _components(alive, layout.pairs_at(t)):
            if not anchors & set(comp):
                unanchored.setdefault(tuple(sorted(comp)), []).append(t)
        if t > 0:
            for a in anchors:
                if not layout.tile_alive[t - 1, layout.ti(a)]:
                    late_anchor.setdefault(a, []).append(t)

    if empty:
        p.append(f"no tiles alive at timepoints {_ranges(empty)}")
    for comp, ts in unanchored.items():
        p.append(
            f"tiles {list(comp)} form a component with no anchor at "
            f"t={_ranges(ts)}: final_coordinates cannot place them"
        )
    for name, ts in late_anchor.items():
        p.append(
            f"'{name}' becomes an anchor at t={_ranges(ts)} but is not alive at the "
            "previous timepoint, so it has no coordinate to drift from"
        )
    return p

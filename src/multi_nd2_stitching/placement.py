"""How each tile gets its coordinate, and whether that route is unique.

A tile is placed either by drifting forward from its own position at t-1 (an
anchor) or by hanging off an already-placed neighbour. Every tile should have
exactly one such route. It does not when the neighbour graph contains a cycle,
or when one connected component holds two anchors: then the flood fill picks
whichever edge it happens to reach first, and the result depends on traversal
order rather than on the data. Those cases are flagged, not silently resolved.

This module owns the traversal. build_coordinates walks the same steps, so the
printed graph is the placement that actually happened, not a reconstruction.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import attrs

ORIGIN = "origin"
DRIFT = "drift"
PAIR = "pair"
CORNER = "corner"  # a fitted diagonal Corner, opted into by the `corner` override


class Edge(NamedTuple):
    """One placement edge for the flood fill: a real neighbour Pair, or a
    Corner opted into by the `corner` override at this timepoint. Unifying
    them into one pool is what makes the existing cycle/multiple-route
    detection below cover corners too, with no extra logic."""

    a: str
    b: str
    axis: int | None  # None for a corner -- it has none
    kind: str


@attrs.frozen
class Step:
    """One placement. `via` is None for a seed."""

    tile: str
    kind: str
    via: str | None = None
    axis: int | None = None  # None for ORIGIN/DRIFT/CORNER -- a corner has no axis
    shaped: bool = False  # this step's offset used the shaped_peak override

    @property
    def label(self) -> str:
        if self.kind == ORIGIN:
            return "origin"
        if self.kind == DRIFT:
            return "drift from t-1"
        if self.kind == CORNER:
            return f"corner from {self.via}"
        return f"{'y' if self.axis == 1 else 'x'} from {self.via}"


@attrs.frozen
class Placement:
    """The full routing at one timepoint."""

    t: int
    steps: tuple[Step, ...]
    redundant: tuple[tuple[str, str, int | None], ...] = ()
    unplaced: tuple[str, ...] = ()
    over_anchored: tuple[tuple[str, ...], ...] = ()

    @property
    def by_tile(self) -> dict[str, Step]:
        return {s.tile: s for s in self.steps}

    @property
    def seeds(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.via is None)

    @property
    def ambiguous(self) -> bool:
        return bool(self.redundant or self.over_anchored)

    @property
    def has_shaped(self) -> bool:
        return any(s.shaped for s in self.steps)

    def children(self) -> dict[str | None, list[Step]]:
        out: dict[str | None, list[Step]] = {}
        for s in self.steps:
            out.setdefault(s.via, []).append(s)
        return out

    def signature(self):
        """Topology (plus shaped_peak status) -- used to collapse runs of
        identical timepoints. shaped_peak doesn't change the topology, but a
        run turning it on or off is exactly the kind of change this exists to
        surface, so it breaks the run like anything else would."""
        return (
            tuple((s.tile, s.kind, s.via, s.axis, s.shaped) for s in self.steps),
            self.redundant,
            self.unplaced,
            self.over_anchored,
        )

    def route_to(self, tile: str) -> list[Step]:
        """The chain of steps that places `tile`, from its seed outwards."""
        by_tile = self.by_tile
        chain: list[Step] = []
        cur: str | None = tile
        seen = set()
        while cur is not None and cur in by_tile and cur not in seen:
            seen.add(cur)
            step = by_tile[cur]
            chain.append(step)
            cur = step.via
        return list(reversed(chain))


def plan_placement(layout, t: int, seeded=None) -> Placement:
    """Work out the routing at one timepoint. Pure topology -- no offsets read.

    `seeded` overrides which anchors can act as seeds; by default an anchor
    seeds if it was alive at t-1 (so it has something to drift from), and at
    t=0 the first anchor defines the origin.
    """
    alive = layout.tiles_at(t)
    if not alive:
        return Placement(t=t, steps=())

    # shaped_peak names either a bare tile (a drift step) or an "a,b" pair (a
    # neighbour edge, either order) -- see StitchingConfig.shaped_peak_at.
    shaped_names = layout.config.shaped_peak_at(t)

    def pair_shaped(a: str, b: str) -> bool:
        return f"{a},{b}" in shaped_names or f"{b},{a}" in shaped_names

    # corner names an "a,b" pair too, either order -- see corner_at.
    corner_names = layout.config.corner_at(t)

    def corner_enabled(a: str, b: str) -> bool:
        return f"{a},{b}" in corner_names or f"{b},{a}" in corner_names

    anchors = [n for n in layout.anchors_at(t) if seeded is None or n in seeded]
    steps: list[Step] = []
    placed: set[str] = set()

    if t == 0:
        first = anchors[0] if anchors else alive[0]
        steps.append(Step(first, ORIGIN))
        placed.add(first)
    else:
        for name in anchors:
            if layout.tile_alive[t - 1, layout.ti(name)]:
                steps.append(Step(name, DRIFT, shaped=name in shaped_names))
                placed.add(name)
        if not placed:
            steps.append(Step(alive[0], ORIGIN))
            placed.add(alive[0])

    # Which seeds share a component? More than one means the component is
    # over-determined: two independent chains fix the same tiles.
    seeds = set(placed)

    pairs = list(layout.pairs_at(t))
    edges = [Edge(p.a, p.b, p.axis, PAIR) for p in pairs]
    edges += [
        Edge(c.a, c.b, None, CORNER)
        for c in layout.corners_at(t)
        if corner_enabled(c.a, c.b)
    ]
    redundant: list[tuple[str, str, int | None]] = []
    queue = deque(sorted(edges, key=lambda e: (e.a, e.b, e.kind, e.axis or 0)))
    stalled = 0
    while queue and stalled <= len(queue):
        e = queue.popleft()
        if e.a in placed and e.b in placed:
            # Both ends already fixed: this edge closes a cycle, or joins two
            # separately seeded chains. Either way the placement is not unique.
            redundant.append((e.a, e.b, e.axis))
            stalled += 1
        elif e.a in placed:
            steps.append(
                Step(e.b, e.kind, via=e.a, axis=e.axis, shaped=pair_shaped(e.a, e.b))
            )
            placed.add(e.b)
            stalled = 0
        elif e.b in placed:
            steps.append(
                Step(e.a, e.kind, via=e.b, axis=e.axis, shaped=pair_shaped(e.a, e.b))
            )
            placed.add(e.a)
            stalled = 0
        else:
            queue.append(e)
            stalled += 1

    over = []
    if len(seeds) > 1:
        for group in _components(alive, edges):
            hit = sorted(seeds & set(group))
            if len(hit) > 1:
                over.append(tuple(hit))

    return Placement(
        t=t,
        steps=tuple(steps),
        redundant=tuple(redundant),
        unplaced=tuple(n for n in alive if n not in placed),
        over_anchored=tuple(over),
    )


def _components(names, pairs):
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


def _arrow(axis: int | None) -> str:
    """The edge label rendering uses: y/x for a Pair, "corner" for a Corner
    (axis=None)."""
    if axis is None:
        return "corner"
    return "y" if axis == 1 else "x"


# --- rendering ----------------------------------------------------------------
def group_runs(placements):
    """Collapse consecutive timepoints with identical topology.

    Over hundreds of timepoints the routing is usually constant for long
    stretches; the interesting thing is where it changes.
    """
    runs = []
    for p in placements:
        if runs and runs[-1][2].signature() == p.signature():
            runs[-1][1] = p.t
        else:
            runs.append([p.t, p.t, p])
    return [(a, b, p) for a, b, p in runs]


def render_tree(placement: Placement, indent: str = "  ") -> list[str]:
    """One ASCII tree per seed: down the page is depth, across is neighbours."""
    children = placement.children()
    lines: list[str] = []

    def walk(step: Step, prefix: str, last: bool, root: bool):
        tag = "  [shaped_peak]" if step.shaped else ""
        if root:
            lines.append(f"{indent}{step.tile}  [{step.label}]{tag}")
            new_prefix = indent
        else:
            arm = "└─" if last else "├─"
            lines.append(f"{prefix}{arm}{_arrow(step.axis)}→ {step.tile}{tag}")
            new_prefix = prefix + ("   " if last else "│  ")
        kids = sorted(children.get(step.tile, []), key=lambda s: (s.axis or -1, s.tile))
        for i, kid in enumerate(kids):
            walk(kid, new_prefix, i == len(kids) - 1, False)

    for seed in placement.seeds:
        walk(seed, indent, True, True)
    return lines


def render(placements, tile: str | None = None, indent: str = "  ") -> list[str]:
    lines: list[str] = []
    for t0, t1, p in group_runs(placements):
        span = f"t={t0}" if t0 == t1 else f"t={t0}..{t1}"
        count = "" if t0 == t1 else f"  ({t1 - t0 + 1} timepoints)"
        flags = [
            f
            for f, on in (("AMBIGUOUS", p.ambiguous), ("SHAPED_PEAK", p.has_shaped))
            if on
        ]
        flag = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"{span}{count}{flag}")
        if tile is not None:
            chain = p.route_to(tile)
            if not chain:
                lines.append(f"{indent}{tile}: not placed")
            else:
                lines.append(
                    indent
                    + " → ".join(
                        (
                            f"{s.tile}[{s.label}]"
                            if s.via is None
                            else f"{_arrow(s.axis)}→ {s.tile}"
                        )
                        + ("  [shaped_peak]" if s.shaped else "")
                        for s in chain
                    )
                )
        else:
            lines.extend(render_tree(p, indent))
        for a, b, axis in p.redundant:
            lines.append(
                f"{indent}! redundant edge {a}|{b} ({_arrow(axis)}): "
                "both ends already placed, so this offset is never used"
            )
        for group in p.over_anchored:
            lines.append(
                f"{indent}! {len(group)} anchors in one component: {', '.join(group)}"
            )
        for name in p.unplaced:
            lines.append(f"{indent}! {name} is alive but cannot be placed")
        lines.append("")
    return lines


def placements_for(layout, t0: int = 0, t1: int | None = None):
    t1 = layout.nt if t1 is None else t1
    return [plan_placement(layout, t) for t in range(t0, t1)]


def anchor_skeleton(layout, t0: int = 0, t1: int | None = None) -> dict[int, tuple]:
    """The smallest set of tiles per timepoint that still fixes every anchor.

    An anchor at t needs its own position at t-1 to drift from. If it was also
    the anchor at t-1 that costs nothing extra, so a long stretch with one
    steady anchor needs only that one tile drawn per timepoint. A handover is
    where it grows: the incoming anchor has to be placed at t-1 through the
    neighbour graph, which pulls in the chain of tiles between the two.

    Returns {t: tiles}. Everything else in the mosaic is decoration as far as
    the coordinate system is concerned.
    """
    t1 = layout.nt if t1 is None else t1
    places = {t: plan_placement(layout, t) for t in range(t0, t1)}
    needed: dict[int, set[str]] = {t: set(layout.anchors_at(t)) for t in range(t0, t1)}

    # An anchor at t must already have a position at t-1.
    for t in range(t0 + 1, t1):
        for name in layout.anchors_at(t):
            needed[t - 1].add(name)

    # Placing a tile means placing everything on its route back to a seed.
    for t in range(t0, t1):
        closed: set[str] = set()
        for name in needed[t]:
            for step in places[t].route_to(name):
                closed.add(step.tile)
        if not closed and layout.tiles_at(t):
            closed = {places[t].seeds[0].tile} if places[t].seeds else set()
        needed[t] = closed

    return {t: tuple(sorted(v)) for t, v in needed.items()}

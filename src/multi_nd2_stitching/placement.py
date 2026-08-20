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

import attrs

ORIGIN = "origin"
DRIFT = "drift"
PAIR = "pair"


@attrs.frozen
class Step:
    """One placement. `via` is None for a seed."""

    tile: str
    kind: str
    via: str | None = None
    axis: int | None = None

    @property
    def label(self) -> str:
        if self.kind == ORIGIN:
            return "origin"
        if self.kind == DRIFT:
            return "drift from t-1"
        return f"{'y' if self.axis == 1 else 'x'} from {self.via}"


@attrs.frozen
class Placement:
    """The full routing at one timepoint."""

    t: int
    steps: tuple[Step, ...]
    redundant: tuple[tuple[str, str, int], ...] = ()
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

    def children(self) -> dict[str | None, list[Step]]:
        out: dict[str | None, list[Step]] = {}
        for s in self.steps:
            out.setdefault(s.via, []).append(s)
        return out

    def signature(self):
        """Topology only -- used to collapse runs of identical timepoints."""
        return (
            tuple((s.tile, s.kind, s.via, s.axis) for s in self.steps),
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
                steps.append(Step(name, DRIFT))
                placed.add(name)
        if not placed:
            steps.append(Step(alive[0], ORIGIN))
            placed.add(alive[0])

    # Which seeds share a component? More than one means the component is
    # over-determined: two independent chains fix the same tiles.
    seeds = set(placed)

    pairs = list(layout.pairs_at(t))
    redundant: list[tuple[str, str, int]] = []
    queue = deque(sorted(pairs, key=lambda p: (p.a, p.b, p.axis)))
    stalled = 0
    while queue and stalled <= len(queue):
        p = queue.popleft()
        if p.a in placed and p.b in placed:
            # Both ends already fixed: this edge closes a cycle, or joins two
            # separately seeded chains. Either way the placement is not unique.
            redundant.append((p.a, p.b, p.axis))
            stalled += 1
        elif p.a in placed:
            steps.append(Step(p.b, PAIR, via=p.a, axis=p.axis))
            placed.add(p.b)
            stalled = 0
        elif p.b in placed:
            steps.append(Step(p.a, PAIR, via=p.b, axis=p.axis))
            placed.add(p.a)
            stalled = 0
        else:
            queue.append(p)
            stalled += 1

    over = []
    if len(seeds) > 1:
        for group in _components(alive, pairs):
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
        if root:
            lines.append(f"{indent}{step.tile}  [{step.label}]")
            new_prefix = indent
        else:
            arm = "└─" if last else "├─"
            axis = "y" if step.axis == 1 else "x"
            lines.append(f"{prefix}{arm}{axis}→ {step.tile}")
            new_prefix = prefix + ("   " if last else "│  ")
        kids = sorted(children.get(step.tile, []), key=lambda s: (s.axis, s.tile))
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
        flag = "  [AMBIGUOUS]" if p.ambiguous else ""
        lines.append(f"{span}{count}{flag}")
        if tile is not None:
            chain = p.route_to(tile)
            if not chain:
                lines.append(f"{indent}{tile}: not placed")
            else:
                lines.append(
                    indent
                    + " → ".join(
                        f"{s.tile}[{s.label}]"
                        if s.via is None
                        else f"{'y' if s.axis == 1 else 'x'}→ {s.tile}"
                        for s in chain
                    )
                )
        else:
            lines.extend(render_tree(p, indent))
        for a, b, axis in p.redundant:
            lines.append(
                f"{indent}! redundant edge {a}|{b} ({'y' if axis == 1 else 'x'}): "
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

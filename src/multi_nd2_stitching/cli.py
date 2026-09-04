"""Command line entry point.

stitch validate ch6.yaml [--deep]
stitch status   ch6.yaml
stitch offsets  ch6.yaml [--between 0 50] [--limit 20] [--concurrency 2]
stitch show     ch6.yaml --at 21
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import attrs

from .config import Overview, load_config
from .layout import build_layout
from .metadata import load_metadata
from .offsets import TimeTask, build_plan, file_keys
from .store import OffsetStore
from .validate import (
    check,
    check_corner,
    check_layout,
    check_overview,
    check_realign,
    check_shaped_peak,
)
from .workspace import Workspace


class Abort(Exception):
    pass


def _report(problems, header: str) -> None:
    if problems:
        print(f"{header}: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise Abort


def _prepare(args, *, need_metadata: bool):
    """config -> (workspace, config, metadata, layout). Validates as it goes."""
    ws = Workspace.of(args.config)
    cfg = load_config(ws.config_path)
    _report(check(cfg, check_files=need_metadata), str(args.config))
    if not need_metadata:
        return ws, cfg, None, None

    ws.create()
    meta = load_metadata(cfg.files, cache=ws.metadata)
    layout = build_layout(cfg, meta)
    _report(
        check(cfg, nts=meta.nts)
        + check_layout(layout)
        + check_shaped_peak(layout)
        + check_realign(layout)
        + check_corner(layout)
        + check_overview(cfg),
        str(args.config),
    )
    return ws, cfg, meta, layout


def _plan_for(args, layout, meta):
    plan = build_plan(layout, meta, precision=args.precision)
    if getattr(args, "between", None):
        t0, t1 = args.between
        plan = plan.between(t0, t1)
    return plan


# --- commands -----------------------------------------------------------------
def cmd_validate(args) -> int:
    _ws, cfg, _meta, layout = _prepare(args, need_metadata=args.deep)
    extra = ""
    if layout is not None:
        extra = f", {layout.nt} timepoints, {len(layout.pairs)} pairs"
    tier = "config+graph" if args.deep else "config"
    print(
        f"{args.config}: OK  ({cfg.n_files} files, {len(cfg.positions)} positions, "
        f"{len(cfg.overrides)} override(s){extra}) [{tier}]"
    )
    if layout is not None and layout.raw_nt > layout.nt:
        print(
            f"stopped    at t={layout.nt} ({layout.raw_nt - layout.nt} more "
            "timepoint(s) in the files, not processed)"
        )
    return 0


def cmd_timeline(args) -> int:
    _ws, cfg, _meta, layout = _prepare(args, need_metadata=True)

    if args.at is not None:
        if not 0 <= args.at < layout.nt:
            print(
                f"t={args.at} is outside the timeline 0..{layout.nt - 1}",
                file=sys.stderr,
            )
            return 1
        file_i, local_t = layout.locate(args.at)
        print(f"t={args.at}  ->  file {file_i}, timepoint {local_t}")
        print(f"           {Path(cfg.files[file_i]).name}")
        alive = layout.tiles_at(args.at)
        print(f"tiles      {len(alive)}: {', '.join(alive)}")
        print(f"anchors    {', '.join(layout.anchors_at(args.at)) or '(none)'}")
        return 0

    name_w = max(len(Path(f).name) for f in cfg.files)
    span_w = len(f"{layout.nt - 1}") * 2 + 2
    print(
        f"{'file':>4}  {'timepoints':>{span_w}}  {'n':>5}  "
        f"{'tiles':>5}  {'anchors':<18}  {'name':<{name_w}}"
    )
    for i in range(cfg.n_files):
        start = layout.file_start[i]
        n = layout.nts[i]
        if start >= layout.nt:
            # Entirely past stop_at: layout has no rows for this file at all,
            # so tiles_at(start) would index straight past the truncated mask.
            print(
                f"{i:>4}  {f'{start}..{start + n - 1}':>{span_w}}  {n:>5}  "
                f"{'-':>5}  {'(beyond stop_at)':<18}  "
                f"{Path(cfg.files[i]).name:<{name_w}}"
            )
            continue
        anchors = (
            layout.references_for_file(i)
            if hasattr(layout, "references_for_file")
            else cfg.references_for_file(i)
        )
        alive = len(layout.tiles_at(start))
        print(
            f"{i:>4}  {f'{start}..{start + n - 1}':>{span_w}}  {n:>5}  "
            f"{alive:>5}  {', '.join(anchors) or '-':<18}  "
            f"{Path(cfg.files[i]).name:<{name_w}}"
        )
    print(f"{'':>4}  {'':>{span_w}}  {layout.nt:>5}  total")
    if layout.raw_nt > layout.nt:
        print(
            f"stopped    at t={layout.nt} ({layout.raw_nt - layout.nt} more "
            "timepoint(s) in the files, not processed)"
        )
    return 0


def cmd_graph(args) -> int:
    from .placement import placements_for, render

    _ws, _cfg, _meta, layout = _prepare(args, need_metadata=True)
    t0 = 0 if args.between is None else args.between[0]
    t1 = layout.nt if args.between is None else args.between[1]
    if not 0 <= t0 < t1 <= layout.nt:
        print(f"t={t0}..{t1} is outside the timeline 0..{layout.nt}", file=sys.stderr)
        return 1

    if args.tile and args.tile not in layout.tiles:
        print(
            f"unknown tile '{args.tile}'; known: {', '.join(layout.tiles)}",
            file=sys.stderr,
        )
        return 1

    places = placements_for(layout, t0, t1)
    ambiguous = [p for p in places if p.ambiguous]
    stuck = [p for p in places if p.unplaced]

    print(
        f"tiles      {len(layout.tiles)}   pairs {len(layout.pairs)}"
        f"   corners {len(layout.corners)}"
    )
    print(f"timepoints {t1 - t0}  (t={t0}..{t1 - 1})")
    print(
        f"ambiguous  {len(ambiguous)} timepoint(s)"
        f"{'' if not ambiguous else f' -- first at t={ambiguous[0].t}'}"
    )
    if stuck:
        print(f"unplaced   {len(stuck)} timepoint(s) with tiles that cannot be placed")
    print()

    lines = render(places, tile=args.tile)
    if args.only_ambiguous:
        keep, block = [], []
        for line in lines:
            block.append(line)
            if line == "":
                if any("!" in b or "AMBIGUOUS" in b for b in block):
                    keep.extend(block)
                block = []
        lines = keep or ["(no ambiguous timepoints)"]

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 1 if (ambiguous and args.strict) else 0


def cmd_status(args) -> int:
    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = _plan_for(args, layout, meta)
    store = OffsetStore(ws.offsets)
    pending = plan.pending(store)
    done = len(plan.tasks) - len(pending)

    print(f"workspace  {ws.root}")
    print(f"timeline   {layout.nt} timepoints across {cfg.n_files} files")
    print(
        f"tiles      {len(layout.tiles)}   pairs {len(layout.pairs)}"
        f"   corners {len(layout.corners)}"
    )
    print(
        f"tasks      {len(plan.tasks)}  ({len(plan.time_tasks)} drift, "
        f"{len(plan.pair_tasks)} pair, {len(plan.corner_tasks)} corner)"
    )
    print(
        f"cached     {done}   pending {len(pending)}"
        f"   [{done / max(len(plan.tasks), 1):.0%} complete]"
    )
    if pending:
        ts = sorted({t.t_to if isinstance(t, TimeTask) else t.t for t in pending})
        print(f"next       t={ts[0]}  ({pending[0].describe()})")
        print(f"missing at t={ts[0]}..{ts[-1]}")
    return 0


def cmd_show(args) -> int:
    ws, _cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = build_plan(layout, meta, precision=args.precision)
    store = OffsetStore(ws.offsets)
    tasks = plan.at(args.at)
    if not tasks:
        print(f"no tasks at t={args.at}")
        return 0
    print(f"t={args.at}: {len(tasks)} task(s)")
    for task in tasks:
        off = store.get(task.key)
        value = (
            f"dz={off.dz:>5} dy={off.dy:>5} dx={off.dx:>5}"
            if off is not None
            else "(not computed)"
        )
        print(f"  {task.describe():<42} {value}")
    return 0


def cmd_offsets(args) -> int:
    from .compute import run_plan
    from .reader import SpectrumCache

    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = _plan_for(args, layout, meta)
    store = OffsetStore(ws.offsets)
    pending = plan.pending(store)
    todo = pending if args.limit is None else pending[: args.limit]

    print(f"workspace  {ws.root}")
    print(
        f"tasks      {len(plan.tasks)} total, {len(plan.tasks) - len(pending)} "
        f"cached, {len(pending)} pending"
    )
    if args.limit is not None:
        print(f"limit      running {len(todo)} of them")
    if not todo:
        print("nothing to do")
        return 0
    if args.dry_run:
        for task in todo[:20]:
            print(f"  would run  {task.describe()}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return 0

    from .reader import Nd2Reader

    max_bytes = None if args.max_mb is None else args.max_mb * 2**20
    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = lambda xs: tqdm(xs, unit="task")
        except ImportError:
            pass

    with Nd2Reader(
        cfg.files,
        file_keys(meta),
        nz=layout.nz,
        ny=layout.ny,
        nx=layout.nx,
        threads=args.read_threads,
    ) as reader:
        # Count only the tasks that will actually run: counting the whole plan
        # on a resumed run leaves every spectrum's use count above zero, so the
        # cache never releases anything.
        cache = SpectrumCache(
            reader, plan.spectrum_uses(todo), workers=args.workers, max_bytes=max_bytes
        )
        n = run_plan(
            plan,
            store,
            cache,
            workers=args.workers,
            concurrency=args.concurrency,
            limit=args.limit,
            progress=progress,
        )

    print(f"ran        {n} task(s)")
    print(f"cache      {cache.stats()}")
    remaining = len(plan.pending(OffsetStore(ws.offsets)))
    print(f"remaining  {remaining}")
    return 0


def cmd_blend(args) -> int:
    from .blend import (
        BlendLog,
        CanvasGeometry,
        CanvasMismatch,
        Timings,
        blend,
        resolve_geometry,
    )
    from .coordinates import MissingOffsets, build_coordinates
    from .placement import anchor_skeleton
    from .reader import Nd2Reader

    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = build_plan(layout, meta, precision=args.precision)
    store = OffsetStore(ws.offsets)

    if args.output:
        output = Path(args.output).expanduser()
    elif args.skeleton:
        # A skeleton canvas is sized to a handful of tiles. Sharing a default
        # path with the full blend would fix that small frame permanently.
        output = ws.root / "skeleton.zarr"
    else:
        output = ws.canvas
    tile_shape = (layout.nz, layout.ny, layout.nx)
    t0 = 0 if args.between is None else args.between[0]
    t1 = layout.nt if args.between is None else args.between[1]

    try:
        coords = build_coordinates(layout, plan, store, t0, t1)
    except MissingOffsets as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.skeleton:
        keep = anchor_skeleton(layout, t0, t1)
        full = sum(len(coords.at(t)) for t in range(t0, t1))
        coords = coords.restrict(keep)
        thin = sum(len(v) for v in keep.values())
        print(
            f"skeleton   {thin} of {full} tile placements "
            f"({thin / max(full, 1):.0%}) -- only what fixes the anchors"
        )

    if args.recreate and output.exists():
        import shutil

        shutil.rmtree(output, ignore_errors=True)
        for suffix in (".blended", ".geometry.json"):
            Path(str(output) + suffix).unlink(missing_ok=True)

    try:
        geometry, is_new = resolve_geometry(
            output,
            coords,
            tile_shape,
            layout.nt,
            args.dtype,
            t0,
            t1,
            recreate=args.recreate,
            pad=args.pad[0] if len(args.pad) == 1 else args.pad,
        )
    except CanvasMismatch as e:
        print(str(e), file=sys.stderr)
        return 1

    log = BlendLog(None if args.no_log else Path(str(output) + ".blended"))
    todo = [
        t
        for t in range(t0, t1)
        if args.force or not log.is_done(t, log.key(t, coords, geometry))
    ]

    print(f"output     {output}")
    print(
        f"canvas     {tuple(int(v) for v in geometry.shape)}  "
        f"origin={geometry.origin}  dtype={geometry.dtype}"
        f"{'  (new)' if is_new else '  (existing frame)'}"
    )
    slack = geometry.slack(coords, tile_shape, layout.nt, t0, t1)
    if any(v > 0 for v in slack):
        need = CanvasGeometry.required(
            coords, tile_shape, layout.nt, args.dtype, t0, t1
        )
        print(
            f"slack      {slack} larger than the current coordinates need "
            f"{need.spatial}; delete the canvas and re-run to shrink it"
        )
    print(f"timepoints {t1 - t0} in range, {len(todo)} to write")
    if not todo:
        print("nothing to do")
        return 0
    if args.dry_run:
        rewrites = [t for t in todo if log.written(t)]
        print(
            f"  would write t={todo[0]}..{todo[-1]}"
            f"{f' ({len(rewrites)} overwriting existing data)' if rewrites else ''}"
        )
        return 0

    refs = {}
    for t in range(t0, t1):
        for name in layout.tiles_at(t):
            if name in coords.at(t):
                refs[(t, name)] = _volume_ref(layout, meta, name, t)

    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = lambda total: tqdm(total=total, unit="tile")
        except ImportError:
            pass

    timings = Timings()
    with Nd2Reader(
        cfg.files,
        file_keys(meta),
        nz=layout.nz,
        ny=layout.ny,
        nx=layout.nx,
        threads=args.read_threads,
        max_open_files=args.open_files,
    ) as reader:
        try:
            n = blend(
                layout,
                coords,
                reader,
                refs,
                output,
                log,
                geometry,
                t0=t0,
                t1=t1,
                chunk=tuple(args.chunk),
                force=args.force,
                progress=progress,
                attempts=args.attempts,
                writers=args.writers,
                pipeline=not args.no_pipeline,
                timings=timings,
            )
        except CanvasMismatch as e:
            print(str(e), file=sys.stderr)
            return 1
    print(f"wrote      {n} timepoint(s)")
    print(f"timings    {timings.as_dict()}")
    return 0


def cmd_inspect(args) -> int:
    from .inspect import inspect_pair
    from .reader import Nd2Reader

    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = build_plan(layout, meta, precision=args.precision)
    store = OffsetStore(ws.offsets)

    tasks = [t for t in plan.pair_tasks if t.t == args.at]
    if args.pair:
        want = tuple(args.pair.split(","))
        tasks = [t for t in tasks if (t.a, t.b) == want or (t.b, t.a) == want]
        if not tasks:
            names = sorted(
                {n for t in plan.pair_tasks if t.t == args.at for n in (t.a, t.b)}
            )
            print(
                f"no pair {want} at t={args.at}; tiles here: {names}", file=sys.stderr
            )
            return 1
    if not tasks:
        print(f"no neighbour pairs at t={args.at}", file=sys.stderr)
        return 1

    missing = [t for t in tasks if store.get(t.key) is None]
    if missing:
        print(
            f"{len(missing)} pair offset(s) at t={args.at} not computed; "
            f"run `stitch offsets --between {args.at} {args.at + 1}`",
            file=sys.stderr,
        )
        return 1

    root = Path(args.out) if args.out else ws.root / "inspect"
    written = []
    with Nd2Reader(
        cfg.files,
        file_keys(meta),
        nz=layout.nz,
        ny=layout.ny,
        nx=layout.nx,
        threads=args.read_threads,
    ) as reader:
        for task in tasks:
            offset = store.get(task.key)
            out = root / f"t{task.t}" / f"{task.a}__{task.b}"
            inspect_pair(task, offset, reader, out, response=not args.no_response)
            nominal = [0, 0, 0]
            nominal[task.axis] = task.shift_px
            delta = [
                int(a - b)
                for a, b in zip(
                    (offset.dz, offset.dy, offset.dx), nominal, strict=False
                )
            ]
            print(
                f"{task.a} | {task.b}  axis={task.axis}  "
                f"offset=({offset.dz}, {offset.dy}, {offset.dx})  "
                f"vs nominal {tuple(nominal)}  delta={tuple(delta)}"
            )
            written.append(out)

    print(f"\nwrote {len(written)} pair(s) under {root}")
    print("napari " + " ".join(str(w / "measured.zarr") for w in written[:3]))
    if not args.no_response:
        print(f"candidates  {written[0] / 'candidates.csv'}")
        print(f"drop-off    {written[0] / 'profiles.csv'}")
    return 0


def cmd_times(args) -> int:
    from .times import WRITERS, build_time_table

    ws, _cfg, meta, layout = _prepare(args, need_metadata=True)
    rows = build_time_table(layout, meta)

    out = args.out or (ws.root / f"times.{args.format}")
    WRITERS[args.format](rows, out)

    skipped = [r.t for r in rows if r.skipped]
    print(f"wrote      {out}")
    print(f"timepoints {len(rows)}  ({len(skipped)} with no recorded real time)")
    if skipped:
        print(
            f"skipped    t={skipped[0]}..{skipped[-1]}"
            if len(skipped) > 1
            else f"skipped    t={skipped[0]}",
            file=sys.stderr,
        )
    return 0


def cmd_overview(args) -> int:
    from .overview import build_overview, read_overview_meta

    ws, cfg, meta, _layout = _prepare(args, need_metadata=True)

    base = cfg.overview or Overview(file="")
    ov = attrs.evolve(
        base,
        file=str(args.overview_file) if args.overview_file else base.file,
        channel=args.channel if args.channel is not None else base.channel,
        pixel_size_um=(
            args.pixel_size_um if args.pixel_size_um is not None else base.pixel_size_um
        ),
        max_output_px=args.max_output_px or base.max_output_px,
        reduction=args.reduction or base.reduction,
    )
    if not ov.file:
        print(
            "overview: need a file, from overview.file in the config "
            "or --overview-file",
            file=sys.stderr,
        )
        return 1

    n_positions = len(read_overview_meta(ov.file).stage_um)
    if ov.channel is not None and not 0 <= ov.channel < n_positions:
        print(
            f"channel {ov.channel} is out of range for {ov.file} "
            f"({n_positions} position(s))",
            file=sys.stderr,
        )
        return 1
    if ov.channel is None and n_positions > 1:
        print(
            f"overview: {ov.file} has {n_positions} positions; pass "
            f"--channel 0..{n_positions - 1}",
            file=sys.stderr,
        )
        return 1

    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = lambda xs: tqdm(xs, unit="block", desc="downsampling")
        except ImportError:
            pass

    out = args.out or (ws.root / "overview.png")
    markers = build_overview(cfg, meta, ov, out, progress=progress)
    print(f"wrote      {out}")
    print(f"markers    {len(markers)} tile(s)")
    return 0


def cmd_drift(args) -> int:
    import numpy as np

    from .inspect import inspect_drift
    from .reader import Nd2Reader

    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = build_plan(layout, meta, precision=args.precision)
    store = OffsetStore(ws.offsets)

    tasks = sorted(
        (t for t in plan.time_tasks if t.name == args.tile), key=lambda t: t.t_to
    )
    if not tasks:
        anchors = sorted({t.name for t in plan.time_tasks})
        print(
            f"'{args.tile}' has no drift steps; anchors are {anchors}", file=sys.stderr
        )
        return 1
    if args.between:
        t0, t1 = args.between
        tasks = [t for t in tasks if t0 <= t.t_to < t1]
        if not tasks:
            print(
                f"no drift steps for '{args.tile}' in t={t0}..{t1 - 1}", file=sys.stderr
            )
            return 1

    pending = [t for t in tasks if store.get(t.key) is None]
    if pending:
        print(
            f"{len(pending)} drift offset(s) not computed; "
            f"run `stitch offsets --between {tasks[0].t_to} {tasks[-1].t_to + 1}`",
            file=sys.stderr,
        )
        return 1

    # flag the steps worth looking at before writing anything
    mags = []
    for t in tasks:
        o = store.get(t.key)
        mags.append(
            (t.t_to, float(np.linalg.norm([o.dz, o.dy, o.dx])), (o.dz, o.dy, o.dx))
        )
    typical = float(np.median([m for _, m, _ in mags])) if mags else 0.0
    outliers = [m for m in mags if m[1] > max(3 * typical, typical + 5)]

    print(f"tile       {args.tile}")
    print(f"steps      {len(tasks)}  (t={tasks[0].t_to}..{tasks[-1].t_to})")
    print(f"median     {typical:.1f} px per step")
    if outliers:
        print(f"outliers   {len(outliers)}:")
        for t, mag, off in outliers[:10]:
            print(f"  t={t:<6} {mag:7.1f} px  (dz,dy,dx)={off}")
        if len(outliers) > 10:
            print(f"  ... and {len(outliers) - 10} more")
    else:
        print("outliers   none stand out from the median")

    out = Path(args.out) if args.out else ws.root / "drift" / args.tile
    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = lambda xs: tqdm(xs, unit="t")
        except ImportError:
            pass

    with Nd2Reader(
        cfg.files,
        file_keys(meta),
        nz=layout.nz,
        ny=layout.ny,
        nx=layout.nx,
        threads=args.read_threads,
        max_open_files=args.open_files,
    ) as reader:
        inspect_drift(
            args.tile,
            tasks,
            store,
            reader,
            out,
            size=None if args.size <= 0 else args.size,
            response=not args.no_response,
            full=args.full,
            progress=progress,
        )

    print(f"\nwrote {out}")
    print(
        f"napari {out}/aligned_xy.zarr {out}/response.zarr"
        if not args.full
        else f"napari {out}/aligned.zarr"
    )
    print(f"offsets table: {out}/offsets.csv")
    return 0


def _volume_ref(layout, meta, name: str, t: int):
    from .offsets import VolumeRef

    fkeys = file_keys(meta)
    file_i, pos = layout.frame_index(name, t)
    _, local_t = layout.locate(t)
    return VolumeRef(fkeys[file_i], pos, local_t, layout.nz)


# --- wiring -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="stitch")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("config", type=Path)
        p.add_argument(
            "--precision",
            choices=("float32", "float64"),
            default="float32",
            help="float32 is ~2x faster and halves memory; it is part "
            "of the cache key, so switching recomputes",
        )
        return p

    v = common(sub.add_parser("validate", help="check the config"))
    v.add_argument(
        "--deep",
        action="store_true",
        help="read ND2 headers and check the neighbour graph",
    )
    v.set_defaults(func=cmd_validate)

    tl = common(
        sub.add_parser("timeline", help="which global timepoints live in which file")
    )
    tl.add_argument("--at", type=int, help="look up one global timepoint")
    tl.set_defaults(func=cmd_timeline)

    g = common(
        sub.add_parser(
            "graph", help="how each tile is placed, and whether the route is unique"
        )
    )
    g.add_argument("--between", type=int, nargs=2, metavar=("T0", "T1"))
    g.add_argument("--tile", help="show only the route that places this tile")
    g.add_argument(
        "--only-ambiguous",
        action="store_true",
        help="print only the timepoints with a flagged route",
    )
    g.add_argument("--out", type=Path, help="write to a file instead of stdout")
    g.add_argument(
        "--strict", action="store_true", help="exit non-zero if any route is ambiguous"
    )
    g.set_defaults(func=cmd_graph)

    s = common(sub.add_parser("status", help="what is done and what is left"))
    s.add_argument("--between", type=int, nargs=2, metavar=("T0", "T1"))
    s.set_defaults(func=cmd_status)

    w = common(sub.add_parser("show", help="offsets at one timepoint"))
    w.add_argument("--at", type=int, required=True)
    w.set_defaults(func=cmd_show)

    o = common(sub.add_parser("offsets", help="compute missing offsets"))
    o.add_argument("--between", type=int, nargs=2, metavar=("T0", "T1"))
    o.add_argument("--limit", type=int, help="run at most N tasks, then stop")
    o.add_argument(
        "--concurrency", type=int, default=1, help="correlations in flight at once"
    )
    o.add_argument("--workers", type=int, default=-1, help="threads per FFT")
    o.add_argument("--read-threads", type=int, default=10)
    o.add_argument(
        "--open-files", type=int, default=2, help="ND2 files kept open at once"
    )
    o.add_argument("--max-mb", type=int, help="cap on the spectrum cache")
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--no-progress", action="store_true")
    o.set_defaults(func=cmd_offsets)

    b = common(sub.add_parser("blend", help="composite tiles onto a zarr canvas"))
    b.add_argument(
        "--output",
        type=Path,
        help="where to write the canvas; defaults to <workspace>/canvas.zarr. "
        "Point this at local disk if the share is unreliable.",
    )
    b.add_argument(
        "--skeleton",
        action="store_true",
        help="draw only the tiles that carry the anchor chain; much faster, and "
        "enough to see whether the drift is tracking. Writes to "
        "<workspace>/skeleton.zarr unless --output says otherwise",
    )
    b.add_argument("--between", type=int, nargs=2, metavar=("T0", "T1"))
    b.add_argument("--dtype", default="uint16")
    b.add_argument(
        "--pad",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="extra pixels on every side; one value pads y and x, "
        "three pad z y x. Use when blending a prefix: the frame "
        "is fixed once created and later timepoints usually "
        "drift outside a tight one",
    )
    b.add_argument(
        "--recreate",
        action="store_true",
        help="delete the canvas and start from a fresh, tight extent",
    )
    b.add_argument(
        "--force",
        action="store_true",
        help="rewrite timepoints already recorded as done",
    )
    b.add_argument(
        "--attempts",
        type=int,
        default=4,
        help="retries per timepoint before giving up on a flaky mount",
    )
    b.add_argument(
        "--no-log",
        action="store_true",
        help="do not record progress (every run rewrites everything)",
    )
    b.add_argument("--read-threads", type=int, default=10)
    b.add_argument(
        "--open-files", type=int, default=2, help="ND2 files kept open at once"
    )
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--no-progress", action="store_true")
    b.add_argument(
        "--writers", type=int, default=4, help="zarr chunks written concurrently"
    )
    b.add_argument(
        "--chunk",
        type=int,
        nargs=4,
        default=[1, 32, 512, 512],
        metavar=("T", "Z", "Y", "X"),
        help="canvas chunk shape; only applied when the canvas is created",
    )
    b.add_argument(
        "--no-pipeline",
        action="store_true",
        help="do not overlap read/compose/write (lower memory, slower)",
    )
    b.set_defaults(func=cmd_blend)

    i = common(sub.add_parser("inspect", help="export a neighbour pair for napari"))
    i.add_argument("--at", type=int, required=True, help="timepoint")
    i.add_argument("--pair", help="'a,b'; default is every pair at that timepoint")
    i.add_argument("--out", type=Path, help="defaults to <workspace>/inspect")
    i.add_argument(
        "--no-response",
        action="store_true",
        help="skip the correlation surface (saves one FFT)",
    )
    i.add_argument("--read-threads", type=int, default=10)
    i.set_defaults(func=cmd_inspect)

    dr = common(sub.add_parser("drift", help="export one tile's drift over time"))
    dr.add_argument("--tile", required=True, help="anchor tile name")
    dr.add_argument("--between", type=int, nargs=2, metavar=("T0", "T1"))
    dr.add_argument(
        "--size",
        type=int,
        default=256,
        help="centred lateral crop; 0 keeps the whole tile",
    )
    dr.add_argument(
        "--full",
        action="store_true",
        help="write a full (T, z, y, x) stack instead of projections (large)",
    )
    dr.add_argument("--out", type=Path, help="defaults to <workspace>/drift/<tile>")
    dr.add_argument("--no-response", action="store_true")
    dr.add_argument("--read-threads", type=int, default=10)
    dr.add_argument("--open-files", type=int, default=2)
    dr.add_argument("--no-progress", action="store_true")
    dr.set_defaults(func=cmd_drift)

    tm = common(
        sub.add_parser(
            "times", help="one real-world time per global timepoint of the canvas"
        )
    )
    tm.add_argument(
        "--format",
        choices=("csv", "npy", "parquet"),
        default="csv",
        help="csv/npy always work; parquet needs pyarrow installed",
    )
    tm.add_argument("--out", type=Path, help="defaults to <workspace>/times.<format>")
    tm.set_defaults(func=cmd_times)

    ov = common(
        sub.add_parser(
            "overview", help="PNG of an overview image with tile positions marked"
        )
    )
    ov.add_argument(
        "--channel",
        type=int,
        help="index into overview.nd2's P axis; overrides overview.channel",
    )
    ov.add_argument("--overview-file", type=Path, help="overrides overview.file")
    ov.add_argument(
        "--pixel-size-um", type=float, help="overrides overview.pixel_size_um"
    )
    ov.add_argument(
        "--max-output-px", type=int, help="overrides overview.max_output_px"
    )
    ov.add_argument(
        "--reduction",
        choices=("mean", "median"),
        help="overrides overview.reduction",
    )
    ov.add_argument("--out", type=Path, help="defaults to <workspace>/overview.png")
    ov.add_argument("--no-progress", action="store_true")
    ov.set_defaults(func=cmd_overview)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Abort:
        return 1
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

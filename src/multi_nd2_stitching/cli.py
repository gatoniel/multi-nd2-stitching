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

from .config import load_config
from .layout import build_layout
from .metadata import load_metadata
from .offsets import PairTask, build_plan, file_keys
from .store import OffsetStore
from .validate import check, check_layout
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
    _report(check(cfg, nts=meta.nts) + check_layout(layout), str(args.config))
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
    return 0


def cmd_status(args) -> int:
    ws, cfg, meta, layout = _prepare(args, need_metadata=True)
    plan = _plan_for(args, layout, meta)
    store = OffsetStore(ws.offsets)
    pending = plan.pending(store)
    done = len(plan.tasks) - len(pending)

    print(f"workspace  {ws.root}")
    print(f"timeline   {layout.nt} timepoints across {cfg.n_files} files")
    print(f"tiles      {len(layout.tiles)}   pairs {len(layout.pairs)}")
    print(
        f"tasks      {len(plan.tasks)}  ({len(plan.time_tasks)} drift, "
        f"{len(plan.pair_tasks)} pair)"
    )
    print(
        f"cached     {done}   pending {len(pending)}"
        f"   [{done / max(len(plan.tasks), 1):.0%} complete]"
    )
    if pending:
        ts = sorted({t.t_to if not isinstance(t, PairTask) else t.t for t in pending})
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
        cache = SpectrumCache(
            reader, plan.spectrum_uses(), workers=args.workers, max_bytes=max_bytes
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


# --- wiring -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="stitch")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("config", type=Path)
        p.add_argument(
            "--precision",
            choices=("float32", "float64"),
            default="float64",
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
    o.add_argument("--max-mb", type=int, help="cap on the spectrum cache")
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--no-progress", action="store_true")
    o.set_defaults(func=cmd_offsets)
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

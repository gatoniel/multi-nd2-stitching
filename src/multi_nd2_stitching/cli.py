import argparse
import sys
from pathlib import Path

from .config import load_config
from .layout import build_layout
from .metadata import load_metadata
from .validate import check, check_layout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="stitch-validate",
        description="Check a stitching config before spending an hour on it.",
    )
    ap.add_argument("config", type=Path, nargs="+")
    ap.add_argument(
        "--check-files",
        action="store_true",
        help="verify that every .nd2 path exists (no ND2 parsing)",
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="read ND2 headers and check the neighbour graph at every timepoint",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="directory for the ND2 header cache used by --deep",
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    failed = False
    for path in args.config:
        try:
            cfg = load_config(path)
        except Exception as e:
            print(f"{path}: NOT PARSEABLE", file=sys.stderr)
            print(f"  {type(e).__name__}: {e}", file=sys.stderr)
            failed = True
            continue

        problems = check(cfg, check_files=args.check_files or args.deep)
        tier = "config"
        layout = None
        if args.deep and not problems:
            cache = (args.cache / f"{path.stem}.meta.json") if args.cache else None
            try:
                meta = load_metadata(cfg.files, cache=cache)
                layout = build_layout(cfg, meta)
            except Exception as e:
                problems.append(f"{type(e).__name__}: {e}")
            else:
                problems += check(cfg, nts=meta.nts)
                problems += check_layout(layout)
                tier = "config+graph"

        if problems:
            print(f"{path}: {len(problems)} problem(s)", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            failed = True
        elif not args.quiet:
            extra = ""
            if layout is not None:
                extra = f", {layout.nt} timepoints, {len(layout.pairs)} pairs"
            print(
                f"{path}: OK  ({cfg.n_files} files, {len(cfg.positions)} positions, "
                f"{len(cfg.overrides)} override(s){extra}) [{tier}]"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

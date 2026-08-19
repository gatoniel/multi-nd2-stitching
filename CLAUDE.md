# multi-nd2-stitching

Stitches multi-position, multi-timepoint microscopy time-lapses from Nikon
`.nd2` files into one mosaic (a `zarr` canvas), for sessions that span
**several separate .nd2 files** (e.g. the microscope was restarted) and where
tiles can appear, disappear, or need manual correction partway through.

Core ideas:
- **Tiles** (named stage positions) are tracked across files by name/alias,
  with explicit `start`/`end` file ranges in a YAML config
  (`StitchingConfig` in `config.py`).
- **Neighbor pairs** are inferred automatically from stage coordinates (grid
  spacing ± tolerance), never configured manually.
- Two kinds of offsets are computed via FFT phase correlation: **pair
  offsets** (tile-to-tile overlap, per timepoint) and **time offsets** (drift
  of "anchor" tiles between consecutive timepoints). A flood-fill
  (`final_coordinates`) turns pairwise offsets into one global coordinate per
  tile per timepoint.
- **Overrides** in the YAML let a human drop a bad tile, designate a new
  anchor, or force realignment at specific timepoints; `validate.py` checks
  these stay internally consistent and that the anchor graph never
  disconnects.

## Two generations of code — do not conflate them

1. **Legacy** — `stitching.py`'s `PositionAlignment` class. Monolithic (I/O +
   FFT + blending in one class), and it reads config attributes
   (`config.names`, `config.start_names`, `config.ignore_timepoints`,
   `config.start_names_manual`, `config.manual_realignment_time`) that **no
   longer exist** on the current `StitchingConfig`. It is out of sync with
   the current config schema and will raise at runtime. No test imports it.
   Do not "fix" it by changing the new config schema to match it — the
   schema in `config.py` is the current one; `stitching.py` is the thing
   that's stale.
2. **New, layered rewrite** — clean separation, fully tested:
   - `metadata.py` — reads ND2 headers only (no pixel data), cached by file
     stamp (path/size/mtime)
   - `config.py` — YAML → `StitchingConfig` via `attrs`/`cattrs`
   - `layout.py` — pure function `(config, metadata) -> Layout`: tiles,
     pairs, aliveness masks, anchors. Nothing here is cached; it's cheap
     enough to rebuild from scratch every run so the result depends only on
     the YAML.
   - `validate.py` — three tiers of checks: config-only → +timeline (needs
     per-file timepoint counts) → +neighbor graph (needs a built `Layout`).
     Used by the `stitch-validate` CLI.
   - `offsets.py` — turns a `Layout` into a `Plan` of content-addressed
     `TimeTask`/`PairTask` units.
   - `store.py` — append-only JSONL offset cache: interrupt-safe, resumable,
     greppable (one JSON record per line, last line wins for a repeated key).
   - `compute.py` — runs a `Plan` against an `OffsetStore` and a
     `VolumeReader` protocol; pure correlation functions are array → array.
   - `cli.py` — currently only wires up `stitch-validate`.

**Gap**: the new pipeline stops at offset computation. There is no rebuilt
equivalent yet of `stitching.py`'s `blend()` (assembling the final `zarr`
canvas with weighted blending across overlaps). `stitch-validate` is the only
CLI entry point that exists today.

## The caching invariant (`offsets.py` / `store.py`)

A task's cache key is a hash of *exactly* the inputs that determine its
value: the file's (path, size, mtime) stamp, the stage position index, the
local timepoint, and the crop — **never** a tile's name or its index in a
sorted list. Renaming a tile or reordering/prepending config entries must not
invalidate cached FFT results. Preserve this when touching `offsets.py`.

## Dev commands

```
uv run pytest -q
uv run stitch-validate <config.yaml> [--deep] [--check-files] [--cache DIR]
```

## Testing conventions

- `tests/helpers.py` builds synthetic `Metadata` (`make_meta`) and a
  `Layout` (`build`) from a plain dict + YAML, so `layout.py`/`validate.py`
  tests need no real `.nd2` files.
- `FakeReader` (in `tests/helpers.py`) produces deterministic, known-shift
  volumes for `compute.py` correlation tests.
- Note: `tests/conftest.py` also defines its own copy of `FakeReader`, which
  currently looks unused — a candidate cleanup, not something to silently
  resolve without checking first.

## Style already in the repo — keep it consistent

- `attrs` (`@attrs.frozen`, `@attrs.define(kw_only=True)`) over plain classes
  or `dataclass`.
- `cattrs` for YAML/JSON (de)serialization, with custom structure hooks
  (see `config.py`'s handling of `Slices3D`/`Timepoints`).
- Comments explain *why*, not what — module docstrings state the rationale
  for a layer's existence (e.g. why `metadata.py` is the only layer that
  touches `nd2`, why `layout.py` is uncached).
- No defensive error handling for internal invariants that can't happen;
  validation lives in `validate.py` at the config/metadata boundary.

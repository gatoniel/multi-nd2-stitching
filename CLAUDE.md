# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Commands

```bash
uv sync                      # install, including the dev group
uv run pytest                # 400+ tests, runs in ~10s
uv run pytest tests/test_blend.py::test_name    # one test
uv run stitch --help         # the CLI

ruff format src/ tests/
ruff check --fix --unsafe-fixes src/ tests/
```

**Run `ruff format` and `ruff check --fix --unsafe-fixes` before running the
tests and before reporting work as done.** There is no `[tool.ruff]` section in
`pyproject.toml` — ruff's defaults are the style, deliberately.

There is no way to run the pipeline end to end without ND2 files. Everything
below the reader is tested against synthetic metadata and fake volumes; see
`tests/helpers.py`.

## What this does

Stitches multi-tile ND2 microscopy timelapses. Several `.nd2` files, each with
several stage positions (tiles), concatenated onto one global time axis, offsets
found by FFT phase correlation, blended into a zarr canvas. A single YAML file
configures a run.

## Architecture

Layers, in dependency order. Each one only knows about the ones above it.

| module | holds | cost | cached? |
| --- | --- | --- | --- |
| `config.py` | the YAML, as attrs classes | free | no |
| `metadata.py` | ND2 header facts | file opens | **yes**, JSON |
| `layout.py` | tiles, pairs, timeline, masks | microseconds | no |
| `placement.py` | how each tile gets placed | microseconds | no |
| `offsets.py` | the work list and its cache keys | free | no |
| `compute.py` | phase correlation, the runner | hours | — |
| `reader.py` | ND2 volumes, refcounted caches | I/O | in memory |
| `store.py` | computed offsets | — | **yes**, JSONL |
| `coordinates.py` | offsets → absolute positions | free | no |
| `blend.py` | compositing onto zarr | hours | canvas + log |
| `validate.py` | three tiers of checks | free–cheap | no |
| `cli.py` | subcommands | — | — |

The caching rule that everything follows: **cache only what is expensive *and*
derived from bytes on disk. Recompute everything else.** `Layout` is free, so it
is rebuilt every run; that is what keeps the result a pure function of the YAML.

## Invariants

Breaking any of these produces silently wrong output rather than an error.
Treat a change that touches one as needing a test that pins it.

**Cache keys name pixels, not tiles.** A `VolumeRef` is
`(file_hash, position, local_t, nz)` — never a tile name, never an index into a
sorted list, never a global timepoint. Renaming a tile or prepending a file must
leave every cached offset valid. Adding anything to a key invalidates the
store, so `precision` belongs there (it changes the result) and a comment does
not.

**Deleting a cache must change nothing but runtime.** True of the metadata
cache, the offset store, and the blend log. If a code path can behave
differently because a cache is present, that path is wrong.

**Canvas geometry is fixed when the canvas is created** and stored in
`<output>.geometry.json`. Never recompute it for an existing canvas: the
required extent is a min/max over every timepoint, so recomputing can move the
origin and silently remap everything already written. A corrupt sidecar raises;
it does not fall back to deriving a new one.

**`placement.py` owns the traversal.** `build_coordinates` walks
`plan_placement`'s steps, so `stitch graph` shows the placement that actually
happens. Do not reimplement the flood fill anywhere else.

**Every tile has exactly one placement route.** Two anchors in one connected
component, or a cycle in the neighbour graph, means the result depends on
traversal order. Those are flagged, never silently resolved.

**The blend crop must not touch the correlation axis.** `crop_for_alignment`
owns that axis; cropping it removes the overlap strip the correlation needs.
See `Crop.free_axis`. This holds for *either* crop a `PairTask` can use --
`build_plan` applies `.free_axis` after picking `realignment_slices` or
`slices`, not before, so a `realign` pair is never accidentally starved of
its own overlap strip.

**A diagonal neighbour never gets an edge `Pair`.** Two tiles can overlap
across a corner (one grid step away in both y and x) with no third tile
occupying the missing corner position to connect them via two edge `Pair`s
instead. `layout._discover_corners` finds this relationship separately
(`Corner`, no `axis` — it tapers both), and `blend_weights` folds its 2D
patch into the edge ramps via `min`, never multiply: each neighbour only
ever imposes an upper bound on the weight `blend_weights` may hand back, so
a well-covered corner (a real third tile *is* there) is a no-op, and an
under-covered one (there isn't) gets tapered for the first time.

**A `corner` override is the only way a `Corner` becomes a placement edge.**
Discovery alone never makes one — `plan_placement` only adds a `Corner` to
its edge pool when `t in o.at` *and* `layout.corner_alive[t, k]`, exactly
the same "opt-in and alive" gate `shaped_peak`/`realign` pairs already use.
`check_layout`'s own connectivity check has to build the identical pool
(pairs plus enabled, alive corners) or it will report a component the
override genuinely connects as unanchored — the two must not drift apart.

**A `CornerTask`'s two crops must be the same physical strip, not just
"roughly the corner".** `blend.py`'s corner taper can get away with an
oversized or off-centre band — a solo region is invisible to the
normalized blend result either way. A correlation crop has no such safety
net: if `crop_a` and `crop_b` don't correspond to the same real overlap,
the FFT correlates two unrelated regions and returns a number with no
error to signal it's wrong (confirmed by hand: an earlier version of
`_corner_crop` used `blend.py`'s band directly and silently returned
offsets nowhere near the true one). `_corner_crop` instead gives both
crops the same `n - shift_px` overlap-strip convention
`crop_for_alignment`/`trim_for` already use for a `Pair`'s one axis, just
independently on both lateral axes. That crop only lets the correlation
measure the *correction* to the nominal `corner_direction` guess it
encodes, not the tiles' true offset, so `run_task` has to add
`CornerTask.nominal` back afterward — the two-axis analogue of
`PairTask`'s `offset[axis] += shift_px`. A `CornerTask` built with no real
crop (e.g. `(None, None, None)` in a test) exercises none of this; it
proves only that the dispatch runs, not that the offset is right.

**Per-timepoint blend work is bounded by the bounding box, not the canvas.**
A whole-canvas `np.divide` costs in proportion to the canvas, which makes
padding ruinously expensive. Same for zarr writes: they must start on chunk
boundaries or zarr read-modify-writes every partial chunk.

**Two timepoint numberings exist, and nothing may mix them.** `Override.at`,
`stop_at`, and `exclude_at` are all *raw* global timepoints -- counted
straight off the concatenated files, unaffected by what `exclude_at` itself
removes, so editing one config field never silently renumbers another.
`layout.nt` and everything that loops over it (`placement.py`,
`build_plan`, `build_coordinates`, `blend`, `times`, every CLI `--at`/
`--between`) are in the *compacted* numbering `build_layout` produces by
deleting excluded rows outright -- not masking them -- which is what makes
a drift step spanning a gap just an ordinary `t-1 -> t` step against
whatever real timepoint is now adjacent, with no special-casing anywhere
downstream. Any code that reads `Override.at` (or calls
`shaped_peak_at`/`realigned_at`/`corner_at`/`near_hint`) against a
compacted array or compacted loop variable has to translate through
`layout.raw_t`/`layout.raw_to_t` first, or it reads the wrong row -- or, if
the excluded count differs, indexes out of range.

## Conventions

- attrs classes, `@attrs.frozen` where possible. cattrs for YAML and JSON.
- Config parsing is structural; **semantic checks live in `validate.py`** and
  return a *list* of problems rather than raising on the first.
- Optional config fields need `| None` in the annotation for
  `default_if_none` converters to run — cattrs structures before the converter
  sees the value.
- Narrow exception handlers. `_PARSE_ERRORS` is the tuple for malformed
  JSON/JSONL. A skipped line warns; it never passes silently.
- Threads, not processes: scipy.fft and the ND2 reads release the GIL, and the
  caches hold arrays far too large to pickle.
- No `__del__`. Resources go through context managers.

## Tests

- `tests/helpers.py` holds pure helpers (`make_meta`, `grid_meta`, `build`,
  `stub_files`, `FakeReader`) and is imported normally. `tests/conftest.py`
  holds only real fixtures (`cfg_dict`, `parse`) and is never imported.
- Config tests parametrize over *every optional field being absent* and over
  *every optional field being explicitly `null`* — two different code paths,
  and the second is the common hand-editing accident.
- Test the module you import from, not the package root.
- `grid_meta` when a test needs a 2D layout; `make_meta` only makes a line, so
  it cannot produce a cycle.

## Gotchas

- `end` in a position is an **exclusive file index**, while `start` is a
  `(file, timepoint)` pair. The asymmetry is a trap.
- `stop_at` (config-level) is an **exclusive global timepoint**, same
  convention as `end`. It truncates `layout.nt` at build time; everything
  downstream already loops over `layout.nt` rather than raw metadata, so
  truncation is free everywhere except code that loops over `cfg.n_files`
  directly (`stitch timeline`'s per-file table has to skip files whose start
  is past it, or `tiles_at()` indexes straight past the truncated mask).
- `exclude_at` removes whole raw timepoints from the *middle* of the
  timeline, not just the tail -- no tile, no canvas frame, no `times` row,
  and a drift step whose anchor is alive on both sides of the gap
  correlates directly across it (a jump, logged in `TimeTask.raw_gap` /
  `describe()`) rather than against the missing frames. `layout.nt` is the
  *compacted* count afterward; `layout.stop_nt` is what `nt` meant before
  `exclude_at` existed, kept around only so `stitch timeline`/`validate` can
  still report the stop_at truncation on its own. See the "two timepoint
  numberings" invariant above before touching anything that reads
  `Override.at` near this.
- Drift is absolute: placing timepoint `t` needs every drift step from 0.
  Pair offsets are local to their timepoint.
- `Override.near`'s `[dz, dy, dx]` is in *measured* (final-offset) space --
  the same units as `candidates.csv`'s `dz,dy,dx` columns. `run_task` has to
  subtract `PairTask.shift_px` back out before the value means anything to
  `compute._windowed_peak_index`, which searches the *raw* (pre-`shift_px`)
  space, same as `to_signed_shift` always has.
- `layout.corner_direction` is the one place that needs a `Corner`'s
  direction from *nominal* stage coordinates rather than `coords.at(t)` --
  `CornerTask`'s crop has to exist before either tile necessarily has a
  coordinate at all (that's the point of fitting one), so `blend.py`'s
  "read the real offset out of coords" trick isn't available yet.
- `Position.missing_in_files` is not the same problem `drop` solves. `drop`
  only hides an already-*resolved* tile at a timepoint; a file where the
  position plain doesn't exist fails during resolution itself, before any
  override runs. Use `missing_in_files` for a genuine gap in the middle of
  a tile's `[start, end)`, `drop` for hiding a tile that's still there.
- The `nd2` handle is not thread-safe. Never peek into the handle pool
  (`pool.queue[0]`); check a handle out, or use the reserved index handle.
- `np.zeros` hands back lazily-zeroed pages, so allocating a fresh buffer per
  timepoint measures *faster* than reusing one with `[:] = 0`. Do not "optimize"
  that.
- Measure before attributing a slowdown. Several plausible causes in this
  codebase turned out to be noise; the real ones were unaligned zarr writes and
  whole-canvas passes.

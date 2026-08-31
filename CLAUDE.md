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

**Per-timepoint blend work is bounded by the bounding box, not the canvas.**
A whole-canvas `np.divide` costs in proportion to the canvas, which makes
padding ruinously expensive. Same for zarr writes: they must start on chunk
boundaries or zarr read-modify-writes every partial chunk.

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
- Drift is absolute: placing timepoint `t` needs every drift step from 0.
  Pair offsets are local to their timepoint.
- `Override.near`'s `[dz, dy, dx]` is in *measured* (final-offset) space --
  the same units as `candidates.csv`'s `dz,dy,dx` columns. `run_task` has to
  subtract `PairTask.shift_px` back out before the value means anything to
  `compute._windowed_peak_index`, which searches the *raw* (pre-`shift_px`)
  space, same as `to_signed_shift` always has.
- The `nd2` handle is not thread-safe. Never peek into the handle pool
  (`pool.queue[0]`); check a handle out, or use the reserved index handle.
- `np.zeros` hands back lazily-zeroed pages, so allocating a fresh buffer per
  timepoint measures *faster* than reusing one with `[:] = 0`. Do not "optimize"
  that.
- Measure before attributing a slowdown. Several plausible causes in this
  codebase turned out to be noise; the real ones were unaligned zarr writes and
  whole-canvas passes.

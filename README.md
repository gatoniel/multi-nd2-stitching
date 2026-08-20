# multi-nd2-stitching

Stitch multi-tile ND2 microscopy timelapses into a single zarr canvas.

Several `.nd2` files, each holding several stage positions, are concatenated
onto one global time axis. Tile-to-tile and frame-to-frame offsets are found by
FFT phase correlation, and the tiles are blended onto a common canvas.

The pipeline is built around one idea: **the result depends only on the YAML.**
Everything else on disk is a cache. Delete it and you lose runtime, never
results.

That makes the slow part resumable. Offsets are computed once and keyed by the
pixels they were computed from, so you can interrupt a run, change one thing in
the config, and only the affected offsets recompute.

## Install

```bash
git clone <this repo>
cd multi-nd2-stitching
uv sync
```

Requires Python 3.14+.

## Quickstart

```bash
stitch validate ch5.yaml --deep      # check the config before spending an hour
stitch offsets  ch5.yaml             # the slow part; resumable
stitch blend    ch5.yaml --output /local/disk/ch5.zarr
```

Everything derived from `path/to/ch5.yaml` lives in `path/to/ch5/`.

## Configuration

```yaml
files:
  - /data/seq001.nd2
  - /data/seq002.nd2
  - /data/seq003.nd2

grid_spacing: 55            # nominal tile separation, µm
grid_spacing_error: 5       # tolerance when inferring which tiles are neighbours
flip_x: false
flip_y: false

positions:
  ch5_a:
    start: [0, 0]           # [file index, timepoint within that file]
    end: 2                  # file index where it stops, EXCLUSIVE
    aliases: [ch5_a_old]    # other names this position has in the ND2 files
    reference_in_files: [0, 1]
  ch5_b:
    start: [1, 24]
  ch5_c:
    start: [2, 0]
    reference_in_files: [2]

overrides:
  - at: 143
    reason: "bubble drifts through ch5_b"
    drop: [ch5_b]
    anchor: [ch5_c]

  - at: [4, 16, 26, 38]
    reason: "stage jump; phase correlation fails on the full volume"
    realign: [ch5_a]

slices:                     # restrict which part of the volume is correlated
  z: [5, 100]
realignment_slices:         # used instead, for `realign` timepoints
  z: [5, 100]
  y: [300, 724]
  x: [300, 724]
```

Neighbours are **not** declared. They are inferred from the stage coordinates
in the ND2 headers: two tiles are neighbours if they sit `grid_spacing ±
grid_spacing_error` apart along x or y.

### Anchors

One tile per connected component carries the drift from timepoint to timepoint;
everything else hangs off its neighbours. `reference_in_files` says which tile
does that, per file.

Every tile must have exactly **one** route to a position. Two anchors in one
component, or a cycle in the neighbour graph, means the placement depends on
traversal order rather than on the data. `stitch graph` finds these.

### Overrides

One block per incident, at one or more timepoints. `reason` is not decoration —
in six months it is the only record of why a timepoint was special.

| verb | tile placed? | in the neighbour graph? | drifts from t-1? |
| --- | --- | --- | --- |
| `drop` | no | no | no |
| `unanchor` | yes | yes | no |
| `anchor` | yes | yes | yes |
| `realign` | yes | yes | yes, using `realignment_slices` |

`unanchor` is how you hand the drift from one tile to another at a boundary
while keeping both in the mosaic:

```yaml
- at: 200
  reason: "ch5_c carries the drift across the file 2/3 boundary"
  unanchor: [ch5_a]
  anchor: [ch5_c]
```

## Commands

| command | what it does |
| --- | --- |
| `stitch validate` | check the config. `--deep` reads ND2 headers and checks the neighbour graph at every timepoint |
| `stitch timeline` | which global timepoints live in which file. `--at T` for the reverse lookup |
| `stitch graph` | how each tile is placed, and whether the route is unique |
| `stitch status` | how much is computed and what is next |
| `stitch offsets` | compute missing offsets |
| `stitch show --at T` | the offsets at one timepoint |
| `stitch blend` | composite onto a zarr canvas |
| `stitch inspect --at T` | export a neighbour pair for visual inspection |
| `stitch drift --tile X` | export one tile's drift over time |

### Validation, in three tiers

`validate` stops at the first tier that fails, so a long run never starts on a
config that cannot finish.

1. **Config only** — instant, no files touched. Ranges, references, aliases,
   contradictory overrides.
2. **+ timeline** — needs the number of timepoints per file. Catches overrides
   pointing at timepoints that do not exist or at tiles that are not alive.
3. **+ neighbour graph** (`--deep`) — the one that matters. Every connected
   component of alive tiles must contain an anchor at every timepoint:

```
- tiles ['ch5_c'] form a component with no anchor at t=143:
  final_coordinates cannot place them
```

### Computing offsets

```bash
stitch offsets ch5.yaml --precision float32 --concurrency 2 --max-mb 4000
stitch offsets ch5.yaml --between 20 30      # just a window
stitch offsets ch5.yaml --limit 20           # run a few, then stop
```

Results are appended to `<workspace>/offsets.jsonl` as they land, so an
interrupted run loses nothing. The file is greppable on purpose.

`--precision float32` is roughly twice as fast as float64 and halves memory.
It is part of the cache key, so switching recomputes rather than mixing.

### Blending

```bash
stitch blend ch5.yaml --output /local/disk/ch5.zarr
stitch blend ch5.yaml --between 0 40 --pad 200    # preview a prefix
stitch blend ch5.yaml --skeleton                  # only the anchor chain
```

`--output` matters if your storage is unreliable: point it at local disk. Each
timepoint is one unit of work, recorded when it completes, and writes are
retried, so a dropped mount costs one timepoint rather than the run.

**The canvas frame is fixed when the canvas is created** and stored beside it in
`<output>.geometry.json`. Later runs place tiles into that frame rather than
recomputing it — recomputing could move the origin and silently shift every
timepoint already written. An oversized canvas stays oversized until you
`--recreate`; the `slack` line tells you when that is worth doing.

`--skeleton` draws only the tiles that carry the anchor chain. Where one anchor
holds steady, that is a single tile per timepoint — fast, and enough to see
whether the drift is tracking. It writes to `<workspace>/skeleton.zarr` so it
cannot lock the full blend into a small frame.

## When a stitch looks wrong

Bad output almost always means one bad offset, and the tools are built around
finding it.

```bash
stitch drift ch5.yaml --tile ch5_a
```

Flags outlier steps before writing anything:

```
median     2.1 px per step
outliers   1:
  t=143      165.0 px  (dz,dy,dx)=(1, 137, -92)
```

and writes, for napari: `aligned_xy` (drift-corrected — the sample should sit
still as you scrub time), `aligned_zx` (the same for z), `raw_xy`, `response`
(each step's correlation surface, peak-centred), and `offsets.csv`.

Then look at that timepoint:

```bash
stitch inspect ch5.yaml --at 143
```

which writes `measured` (both tiles positioned by the computed offset — toggle
the second layer and the seam should vanish), `nominal` (the same pair at bare
grid spacing), `overlap` (the two strips the correlation actually compared), and
`response`.

`response` is usually the answer to *why*. It is the phase-correlation surface,
which is what the offset is actually read off: both strips are Fourier
transformed, multiplied with one conjugated to give the cross-power spectrum,
that product is normalised to unit magnitude so only the phase survives, and the
result is transformed back into real space. Each voxel is then the correlation
strength for one candidate shift, and `fftshift` puts *no shift* at the centre
of the array. The brightest voxel is the offset that was chosen, and its
displacement from the centre is that offset.

So a good match is one sharp peak. A broad smear, or two peaks of similar
height, means the strips did not contain enough distinct structure to pin a
single shift — the correlation had to pick between near-equal candidates and may
well have picked wrong. That is a `slices` problem (or a "there is nothing in
this region yet" problem), not a code problem.

To look at all four at once in napari:

```python
from pathlib import Path

import napari
import zarr

CONFIG = Path(r"Z:\data\ch5.yaml")
TIMEPOINT = 40
# The folder is "<a>__<b>", and that order is the pair's own, not alphabetical
# -- `stitch inspect` prints it ("ch5_b | ch5_a  axis=2 ..."), or just list the
# t<N> directory. Layer 0 of each array is <a>, layer 1 is <b>.
PAIR = ("ch5_b", "ch5_a")

# the workspace is the config path without its suffix
path = CONFIG.parent / CONFIG.stem / "inspect" / f"t{TIMEPOINT}" / "__".join(PAIR)

viewer = napari.Viewer()
viewer.layers.clear()

colormaps = ["magenta", "cyan"]
for name in ("response", "overlap", "nominal", "measured"):
    array = path / f"{name}.zarr"
    if not array.exists():  # response is absent if --no-response was used
        continue
    img = zarr.open(str(array), mode="r")
    if name == "response":
        viewer.add_image(img, name="response")
    else:
        for i, tile in enumerate(PAIR):
            viewer.add_image(
                img[i],
                name=f"{name}: {tile}",
                colormap=colormaps[i],
                blending="additive",
            )
```

Each pair goes in as two additive layers, magenta and cyan, so the colours
combine wherever the tiles overlap and agree. On `measured` the seam should
disappear; on `nominal` you see how far the correlation moved things from the
bare grid spacing. Toggling one layer of `measured` on and off is the fastest
check there is.

Once you have a fix, recompute only what changed:

```bash
stitch offsets ch5.yaml --between 143 144
stitch blend   ch5.yaml --output /local/disk/ch5.zarr
```

Changing `slices`, or adding an override, changes the cache keys only for the
offsets it actually affects. Everything else stays computed.

## Development

```bash
uv run pytest
ruff format src/ tests/
ruff check --fix --unsafe-fixes src/ tests/
```

There is no `[tool.ruff]` section — ruff's defaults are the style.

The test suite runs in about ten seconds and needs no microscopy data:
everything below the ND2 reader is exercised against synthetic metadata and
fake volumes.

## Acknowledgements

This repository was written with substantial help from Claude (Anthropic),
working interactively with the author: architecture, implementation, and the
test suite were developed in dialogue. The design decisions, the domain
knowledge about the microscopy data, and the review of every change are the
author's.

"""End-to-end CLI tests with a stubbed ND2 layer."""

import pytest
import yaml
from helpers import FakeReader, grid_meta, make_meta

from multi_nd2_stitching import cli
from multi_nd2_stitching import metadata as M
from multi_nd2_stitching import reader as R
from multi_nd2_stitching.workspace import Workspace

CFG = {
    "grid_spacing": 55,
    "grid_spacing_error": 5,
    "shift_px": 3,
    "positions": {
        "tile_a": {"start": [0, 0], "reference_in_files": [0, 1]},
        "tile_b": {"start": [0, 0]},
    },
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A yaml plus two stub .nd2 files, with ND2 access stubbed out."""
    files = []
    for i in range(2):
        p = tmp_path / f"f{i}.nd2"
        p.write_bytes(b"x" * (10 + i))
        files.append(str(p))

    cfg_path = tmp_path / "ch6.yaml"
    cfg_path.write_text(yaml.safe_dump({**CFG, "files": files}))

    monkeypatch.setattr(
        M,
        "read_metadata",
        lambda paths: make_meta(n_files=2, nt=4, nz=4, ny=8, nx=8, paths=list(paths)),
    )

    class StubReader(FakeReader):
        def __init__(self, *a, **kw):
            super().__init__(shape=(4, 8, 8))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(R, "Nd2Reader", StubReader)
    return cfg_path


def run(*argv):
    return cli.main([str(a) for a in argv])


# --- workspace layout ---------------------------------------------------------
def test_workspace_sits_next_to_the_yaml(tmp_path):
    ws = Workspace.of(tmp_path / "sub" / "ch6.yaml")
    assert ws.root == (tmp_path / "sub" / "ch6").resolve()
    assert ws.offsets.name == "offsets.jsonl"
    assert ws.metadata.name == "metadata.json"


def test_workspace_is_created_on_demand(project, capsys):
    run("validate", project, "--deep")
    assert (project.parent / "ch6").is_dir()


# --- validate -----------------------------------------------------------------
def test_validate_shallow(project, capsys):
    assert run("validate", project) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_deep_reports_the_graph(project, capsys):
    assert run("validate", project, "--deep") == 0
    out = capsys.readouterr().out
    assert "8 timepoints" in out and "1 pairs" in out


def test_validate_fails_on_a_broken_config(project, capsys):
    project.write_text(yaml.safe_dump({**CFG, "files": ["nope.nd2"]}))
    assert run("validate", project) == 1
    assert "out of range" in capsys.readouterr().err


def test_deep_validate_reports_a_missing_file(project, capsys):
    project.write_text(yaml.safe_dump({**CFG, "files": ["nope0.nd2", "nope1.nd2"]}))
    assert run("validate", project, "--deep") == 1
    assert "does not exist" in capsys.readouterr().err


def test_deep_validate_catches_an_unanchored_component(
    project, capsys, tmp_path, monkeypatch
):
    """A chain a-b-c anchored at a; dropping b at t=3 orphans c.

    Only the graph check can see this -- b is not a reference, so every
    config-only rule is satisfied.
    """
    tiles = ("a", "b", "c")
    files = [str(tmp_path / f"f{i}.nd2") for i in range(2)]
    monkeypatch.setattr(
        M,
        "read_metadata",
        lambda paths: make_meta(
            n_files=2, nt=4, nz=4, ny=8, nx=8, tiles=tiles, paths=list(paths)
        ),
    )
    project.write_text(
        yaml.safe_dump(
            {
                "grid_spacing": 55,
                "grid_spacing_error": 5,
                "shift_px": 3,
                "files": files,
                "positions": {
                    "a": {"start": [0, 0], "reference_in_files": [0, 1]},
                    "b": {"start": [0, 0]},
                    "c": {"start": [0, 0]},
                },
                "overrides": [{"at": 3, "drop": ["b"]}],
            }
        )
    )
    assert run("validate", project) == 0, "config-only rules see nothing wrong"
    assert run("validate", project, "--deep") == 1
    assert "no anchor at t=3" in capsys.readouterr().err


# --- status -------------------------------------------------------------------
def test_status_on_an_empty_workspace(project, capsys):
    assert run("status", project) == 0
    out = capsys.readouterr().out
    assert "0% complete" in out
    assert "pending 15" in out  # 8 pair + 7 drift


def test_status_respects_between(project, capsys):
    run("status", project, "--between", 0, 2)
    assert "pending 3" in capsys.readouterr().out


def test_status_reports_corner_count(project, capsys):
    """A line, not a grid -- no diagonal neighbours, so 0 is the right count,
    but the field itself must be there."""
    run("status", project)
    assert "corners 0" in capsys.readouterr().out


# --- offsets: the run, and the second run -------------------------------------
def test_dry_run_computes_nothing(project, capsys):
    assert run("offsets", project, "--dry-run") == 0
    assert "would run" in capsys.readouterr().out
    assert not Workspace.of(project).offsets.exists()


def test_offsets_runs_and_persists(project, capsys):
    assert run("offsets", project, "--no-progress") == 0
    out = capsys.readouterr().out
    assert "ran        15" in out
    assert "remaining  0" in out
    assert Workspace.of(project).offsets.exists()


def test_second_run_does_nothing(project, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    assert run("offsets", project, "--no-progress") == 0
    out = capsys.readouterr().out
    assert "15 total, 15 cached, 0 pending" in out
    assert "nothing to do" in out


def test_limit_runs_a_slice_then_resumes(project, capsys):
    run("offsets", project, "--limit", 4, "--no-progress")
    assert "ran        4" in capsys.readouterr().out
    run("offsets", project, "--no-progress")
    out = capsys.readouterr().out
    assert "4 cached, 11 pending" in out
    assert "ran        11" in out


def test_between_restricts_the_work(project, capsys):
    run("offsets", project, "--between", 0, 2, "--no-progress")
    assert "ran        3" in capsys.readouterr().out
    run("status", project)
    assert "cached     3" in capsys.readouterr().out


def test_concurrency_gives_the_same_count(project, capsys):
    assert run("offsets", project, "--concurrency", 4, "--no-progress") == 0
    assert "ran        15" in capsys.readouterr().out


def test_metadata_cache_is_written_and_reused(project, monkeypatch):
    ws = Workspace.of(project)
    run("status", project)
    assert ws.metadata.exists()
    monkeypatch.setattr(
        M,
        "read_metadata",
        lambda paths: (_ for _ in ()).throw(AssertionError("should not re-read")),
    )
    assert run("status", project) == 0


def test_changing_precision_invalidates(project, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("status", project, "--precision", "float64")
    assert "cached     0" in capsys.readouterr().out


# --- show ---------------------------------------------------------------------
def test_show_before_and_after(project, capsys):
    run("show", project, "--at", 1)
    assert "(not computed)" in capsys.readouterr().out
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("show", project, "--at", 1)
    out = capsys.readouterr().out
    assert "dz=" in out and "(not computed)" not in out


def test_show_names_both_kinds(project, capsys):
    run("show", project, "--at", 1)
    out = capsys.readouterr().out
    assert "time tile_a 0->1" in out
    assert "pair" in out


# --- blend --------------------------------------------------------------------
def test_blend_refuses_without_offsets(project, capsys):
    assert run("blend", project) == 1
    assert "stitch offsets --between" in capsys.readouterr().err


def test_blend_writes_to_a_chosen_output(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "elsewhere" / "canvas.zarr"
    assert run("blend", project, "--output", out, "--no-progress") == 0
    text = capsys.readouterr().out
    assert str(out) in text
    assert "wrote      8 timepoint(s)" in text
    assert out.exists()


def test_blend_defaults_into_the_workspace(project, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("blend", project, "--no-progress")
    assert Workspace.of(project).canvas.exists()


def test_second_blend_is_a_noop(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    capsys.readouterr()
    assert run("blend", project, "--output", out, "--no-progress") == 0
    assert "nothing to do" in capsys.readouterr().out


def test_blend_between_then_resume(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--between", 0, 3, "--no-progress")
    assert "wrote      3" in capsys.readouterr().out
    run("blend", project, "--output", out, "--no-progress")
    assert "wrote      5" in capsys.readouterr().out


def test_blend_dry_run_writes_nothing(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    assert run("blend", project, "--output", out, "--dry-run") == 0
    assert "would write" in capsys.readouterr().out
    assert not out.exists()


def test_blend_keeps_an_oversized_canvas_and_says_so(project, tmp_path, capsys):
    """The workflow: an over-wide canvas stays usable; you shrink it on purpose."""
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    # simulate the real situation: the canvas was created wider than the
    # coordinates now require (early offsets were wrong)
    import shutil

    geom = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())
    geom["shape"][3] += 40
    (tmp_path / "canvas.zarr.geometry.json").write_text(json.dumps(geom))
    shutil.rmtree(out)
    capsys.readouterr()
    assert run("blend", project, "--output", out, "--no-progress", "--force") == 0
    text = capsys.readouterr().out
    assert "existing frame" in text
    assert "slack" in text and "delete the canvas" in text


def test_blend_refuses_when_the_sidecar_and_the_zarr_disagree(
    project, tmp_path, capsys
):
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    geom = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())
    geom["shape"][3] += 40
    (tmp_path / "canvas.zarr.geometry.json").write_text(json.dumps(geom))
    capsys.readouterr()
    assert run("blend", project, "--output", out, "--no-progress", "--force") == 1
    assert "but its geometry says" in capsys.readouterr().err


def test_recreate_shrinks_back(project, tmp_path, capsys):
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    tight = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())["shape"]
    geom = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())
    geom["shape"][3] += 40
    (tmp_path / "canvas.zarr.geometry.json").write_text(json.dumps(geom))
    capsys.readouterr()
    run("blend", project, "--output", out, "--recreate", "--no-progress")
    text = capsys.readouterr().out
    assert "(new)" in text
    assert (
        json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())["shape"]
        == tight
    )


def test_blend_refuses_a_tile_that_no_longer_fits(project, tmp_path, capsys):
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    geom = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())
    geom["shape"][3] -= 4
    (tmp_path / "canvas.zarr.geometry.json").write_text(json.dumps(geom))
    capsys.readouterr()
    assert run("blend", project, "--output", out, "--no-progress", "--force") == 1
    assert "no longer fit" in capsys.readouterr().err


# --- blending a prefix --------------------------------------------------------
def test_blend_a_prefix_before_everything_is_computed(project, tmp_path, capsys):
    """The point of the exercise: look at the first stretch early."""
    run("offsets", project, "--between", 0, 4, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "canvas.zarr"
    assert (
        run(
            "blend",
            project,
            "--output",
            out,
            "--between",
            0,
            4,
            "--pad",
            16,
            "--no-progress",
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "wrote      4 timepoint(s)" in text
    assert out.exists()


def test_blending_past_the_computed_range_still_refuses(project, tmp_path, capsys):
    run("offsets", project, "--between", 0, 4, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "canvas.zarr"
    assert run("blend", project, "--output", out, "--no-progress") == 1
    assert "not computed yet" in capsys.readouterr().err


def test_pad_leaves_room_for_the_rest_of_the_run(project, tmp_path, capsys):
    """Without --pad the prefix's tight frame would reject later timepoints."""
    run("offsets", project, "--between", 0, 4, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run(
        "blend",
        project,
        "--output",
        out,
        "--between",
        0,
        4,
        "--pad",
        16,
        "--no-progress",
    )
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    assert run("blend", project, "--output", out, "--no-progress") == 0
    text = capsys.readouterr().out
    assert "existing frame" in text
    assert "wrote      4 timepoint(s)" in text, "the first four stay cached"


def test_pad_widens_the_new_canvas(project, tmp_path, capsys):
    import json

    run("offsets", project, "--no-progress")
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    run("blend", project, "--output", a, "--no-progress")
    run("blend", project, "--output", b, "--pad", 10, "--no-progress")
    sa = json.loads((tmp_path / "a.zarr.geometry.json").read_text())
    sb = json.loads((tmp_path / "b.zarr.geometry.json").read_text())
    assert sb["shape"][3] == sa["shape"][3] + 20
    assert sb["origin"][2] == sa["origin"][2] - 10


# --- inspect ------------------------------------------------------------------
def test_inspect_writes_arrays_napari_can_open(project, tmp_path, capsys):
    import json

    import zarr

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "look"
    assert run("inspect", project, "--at", 1, "--out", out) == 0
    text = capsys.readouterr().out
    assert "vs nominal" in text and "napari" in text
    d = next((out / "t1").iterdir())
    for name in ("measured", "nominal", "overlap", "response"):
        assert zarr.open(str(d / f"{name}.zarr"), mode="r").shape
    info = json.loads((d / "info.json").read_text())
    assert info["t"] == 1 and len(info["measured_offset"]) == 3


def test_inspect_pair_info_json_reports_realign(project, tmp_path, capsys):
    import json

    cfg = yaml.safe_load(project.read_text())
    cfg["realignment_slices"] = {"y": [1, 2]}
    cfg["overrides"] = [{"at": 1, "realign": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "look"
    assert (
        run("inspect", project, "--at", 1, "--pair", "tile_a,tile_b", "--out", out) == 0
    )
    d = next((out / "t1").iterdir())
    info = json.loads((d / "info.json").read_text())
    assert info["realign"] is True


def test_inspect_candidates_csv_has_exactly_one_taken_row(project, tmp_path, capsys):
    import csv
    import json

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "look"
    assert run("inspect", project, "--at", 1, "--out", out) == 0
    text = capsys.readouterr().out
    assert "candidates" in text and "drop-off" in text

    d = next((out / "t1").iterdir())
    rows = list(csv.DictReader((d / "candidates.csv").open()))
    assert rows and rows[0].keys() == {
        "rank",
        "dz",
        "dy",
        "dx",
        "px_z",
        "px_y",
        "px_x",
        "value",
        "decay",
        "taken",
    }
    taken = [r for r in rows if r["taken"] == "1"]
    assert len(taken) == 1

    info = json.loads((d / "info.json").read_text())
    measured = tuple(info["measured_offset"])
    assert (int(taken[0]["dz"]), int(taken[0]["dy"]), int(taken[0]["dx"])) == measured


def test_inspect_candidates_csv_pixel_positions_match_response_zarr(
    project, tmp_path, capsys
):
    import csv

    import numpy as np
    import zarr

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "look"
    assert run("inspect", project, "--at", 1, "--out", out) == 0
    d = next((out / "t1").iterdir())
    rows = list(csv.DictReader((d / "candidates.csv").open()))
    response = zarr.open(str(d / "response.zarr"), mode="r")[:]

    for row in rows:
        px = (int(row["px_z"]), int(row["px_y"]), int(row["px_x"]))
        assert response[px] == np.float32(float(row["value"]))


def test_inspect_profiles_csv_covers_every_candidate_and_axis(project, tmp_path):
    import csv

    run("offsets", project, "--no-progress")
    out = tmp_path / "look"
    run("inspect", project, "--at", 1, "--out", out)
    d = next((out / "t1").iterdir())
    rows = list(csv.DictReader((d / "profiles.csv").open()))
    assert rows[0].keys() == {"rank", "axis", "step", "value"}
    assert {r["axis"] for r in rows} == {"z", "y", "x"}
    ranks = {int(r["rank"]) for r in rows}
    assert ranks == set(range(1, len(ranks) + 1))  # every candidate covered


def test_inspect_measured_has_two_layers(project, tmp_path):
    import zarr

    run("offsets", project, "--no-progress")
    out = tmp_path / "look"
    run("inspect", project, "--at", 1, "--out", out)
    d = next((out / "t1").iterdir())
    arr = zarr.open(str(d / "measured.zarr"), mode="r")
    assert arr.shape[0] == 2, "layer 0 = a, layer 1 = b"


def test_inspect_reports_the_correction(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("inspect", project, "--at", 1, "--out", tmp_path / "look")
    assert "delta=" in capsys.readouterr().out


def test_inspect_needs_the_offsets_first(project, tmp_path, capsys):
    assert run("inspect", project, "--at", 1, "--out", tmp_path / "look") == 1
    assert "not computed" in capsys.readouterr().err


def test_inspect_unknown_pair_lists_the_tiles(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    assert (
        run(
            "inspect",
            project,
            "--at",
            1,
            "--pair",
            "nope,alsonope",
            "--out",
            tmp_path / "look",
        )
        == 1
    )
    assert "tiles here:" in capsys.readouterr().err


def test_no_response_skips_that_array(project, tmp_path):
    run("offsets", project, "--no-progress")
    out = tmp_path / "look"
    run("inspect", project, "--at", 1, "--out", out, "--no-response")
    d = next((out / "t1").iterdir())
    assert not (d / "response.zarr").exists()
    assert not (d / "candidates.csv").exists()
    assert not (d / "profiles.csv").exists()
    assert (d / "measured.zarr").exists()


def test_blend_default_canvas_is_tight(project, tmp_path, capsys):
    """No --pad means no slack line and no wasted canvas."""
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--no-progress")
    text = capsys.readouterr().out
    assert "slack" not in text


def test_a_padded_canvas_persists_until_recreated(project, tmp_path, capsys):
    """The frame is fixed at creation, so dropping --pad later changes nothing."""
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "canvas.zarr"
    run("blend", project, "--output", out, "--pad", 20, "--no-progress")
    padded = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())["shape"]

    capsys.readouterr()
    run("blend", project, "--output", out, "--no-progress", "--force")
    assert (
        json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())["shape"]
        == padded
    )
    assert "slack" in capsys.readouterr().out

    run("blend", project, "--output", out, "--recreate", "--no-progress")
    tight = json.loads((tmp_path / "canvas.zarr.geometry.json").read_text())["shape"]
    assert tight[2] == padded[2] - 40 and tight[3] == padded[3] - 40


# --- drift --------------------------------------------------------------------
def test_drift_writes_the_stacks_and_the_table(project, tmp_path, capsys):
    import zarr

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "drift"
    assert run("drift", project, "--tile", "tile_a", "--out", out, "--no-progress") == 0
    text = capsys.readouterr().out
    assert "median" in text and "napari" in text
    for name in ("aligned_xy", "aligned_zx", "raw_xy", "response"):
        assert zarr.open(str(out / f"{name}.zarr"), mode="r").shape
    rows = (out / "offsets.csv").read_text().strip().splitlines()
    assert rows[0].startswith("t,dz,dy,dx")
    assert len(rows) == 8, "one header plus one row per step"


def test_drift_stacks_are_one_frame_longer_than_the_steps(project, tmp_path):
    import zarr

    run("offsets", project, "--no-progress")
    out = tmp_path / "drift"
    run("drift", project, "--tile", "tile_a", "--out", out, "--no-progress")
    aligned = zarr.open(str(out / "aligned_xy.zarr"), mode="r")
    response = zarr.open(str(out / "response.zarr"), mode="r")
    assert aligned.shape[0] == response.shape[0] + 1


def test_drift_canvas_is_wider_than_the_tile_when_it_moves(project, tmp_path):
    import json

    run("offsets", project, "--no-progress")
    out = tmp_path / "drift"
    run("drift", project, "--tile", "tile_a", "--out", out, "--no-progress")
    info = json.loads((out / "info.json").read_text())
    assert info["tile"] == "tile_a"
    assert len(info["total_drift"]) == 3
    assert info["canvas"][1] >= info["tile_shape"][1]


def test_drift_full_writes_a_4d_stack(project, tmp_path):
    import zarr

    run("offsets", project, "--no-progress")
    out = tmp_path / "drift"
    run("drift", project, "--tile", "tile_a", "--out", out, "--full", "--no-progress")
    assert zarr.open(str(out / "aligned.zarr"), mode="r").ndim == 4
    assert not (out / "aligned_xy.zarr").exists()


def test_drift_between_limits_the_steps(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    out = tmp_path / "drift"
    run(
        "drift",
        project,
        "--tile",
        "tile_a",
        "--between",
        2,
        5,
        "--out",
        out,
        "--no-progress",
    )
    assert "steps      3" in capsys.readouterr().out


def test_drift_needs_the_offsets(project, tmp_path, capsys):
    assert (
        run(
            "drift",
            project,
            "--tile",
            "tile_a",
            "--out",
            tmp_path / "d",
            "--no-progress",
        )
        == 1
    )
    assert "not computed" in capsys.readouterr().err


def test_drift_on_a_non_anchor_lists_the_anchors(project, tmp_path, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    assert (
        run(
            "drift",
            project,
            "--tile",
            "tile_b",
            "--out",
            tmp_path / "d",
            "--no-progress",
        )
        == 1
    )
    assert "anchors are ['tile_a']" in capsys.readouterr().err


def test_drift_flags_an_outlier_step(project, tmp_path, capsys, monkeypatch):
    """A single large jump should be called out before anything is written."""
    from multi_nd2_stitching.offsets import build_plan as real_build

    run("offsets", project, "--no-progress")

    from multi_nd2_stitching.store import Offset
    from multi_nd2_stitching.store import OffsetStore as RealStore
    from multi_nd2_stitching.workspace import Workspace

    ws = Workspace.of(project)
    plan_store = RealStore(ws.offsets)
    # overwrite one drift step with a big jump
    from multi_nd2_stitching.config import load_config
    from multi_nd2_stitching.layout import build_layout
    from multi_nd2_stitching.metadata import load_metadata

    cfg = load_config(ws.config_path)
    meta = load_metadata(cfg.files, cache=ws.metadata)
    plan = real_build(build_layout(cfg, meta), meta)
    target = sorted(plan.time_tasks, key=lambda t: t.t_to)[3]
    plan_store.put(target, Offset(0, 400, 0))

    capsys.readouterr()
    run("drift", project, "--tile", "tile_a", "--out", tmp_path / "d", "--no-progress")
    text = capsys.readouterr().out
    assert "outliers   1" in text
    assert f"t={target.t_to}" in text


# --- timeline -----------------------------------------------------------------
def test_timeline_lists_every_file(project, capsys):
    assert run("timeline", project) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert "timepoints" in lines[0]
    assert "0..3" in out and "4..7" in out
    assert "total" in out


def test_timeline_shows_anchors_per_file(project, capsys):
    run("timeline", project)
    assert "tile_a" in capsys.readouterr().out


# --- stop_at --------------------------------------------------------------------
def test_validate_reports_stop_at_truncation(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["stop_at"] = 5
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 0
    out = capsys.readouterr().out
    assert "stopped" in out and "t=5" in out


def test_validate_says_nothing_extra_without_stop_at(project, capsys):
    assert run("validate", project, "--deep") == 0
    assert "stopped" not in capsys.readouterr().out


def test_timeline_reports_stop_at_and_marks_excluded_files(project, capsys):
    """nt=4 per file, 2 files -> file 1 starts at t=4; stop_at=4 excludes it
    entirely, which used to crash tiles_at() on the truncated mask."""
    cfg = yaml.safe_load(project.read_text())
    cfg["stop_at"] = 4
    project.write_text(yaml.safe_dump(cfg))
    assert run("timeline", project) == 0
    out = capsys.readouterr().out
    assert "beyond stop_at" in out
    assert "stopped" in out and "t=4" in out


def test_timeline_says_nothing_extra_without_stop_at(project, capsys):
    run("timeline", project)
    out = capsys.readouterr().out
    assert "stopped" not in out and "beyond stop_at" not in out


def test_stop_at_shrinks_the_task_count(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["stop_at"] = 4
    project.write_text(yaml.safe_dump(cfg))
    assert run("status", project) == 0
    out = capsys.readouterr().out
    assert "pending 7" in out  # 4 pair + 3 drift, vs. 15 for the full 8 timepoints


def test_timeline_at_resolves_a_global_timepoint(project, capsys):
    assert run("timeline", project, "--at", 5) == 0
    out = capsys.readouterr().out
    assert "t=5" in out and "file 1, timepoint 1" in out
    assert "tiles" in out and "anchors" in out


@pytest.mark.parametrize(
    "t,expected",
    [
        (0, "file 0, timepoint 0"),
        (3, "file 0, timepoint 3"),
        (4, "file 1, timepoint 0"),
        (7, "file 1, timepoint 3"),
    ],
)
def test_timeline_at_boundaries(project, capsys, t, expected):
    run("timeline", project, "--at", t)
    assert expected in capsys.readouterr().out


def test_timeline_at_out_of_range(project, capsys):
    assert run("timeline", project, "--at", 99) == 1
    assert "outside the timeline" in capsys.readouterr().err


# --- precision default --------------------------------------------------------
def test_float32_is_the_default(project, capsys):
    """Explicitly float32 must be a no-op against a default run."""
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("status", project, "--precision", "float32")
    out = capsys.readouterr().out
    assert "pending 0" in out


def test_library_and_cli_defaults_agree():
    """A mismatch here silently makes library-built plans miss the CLI's cache."""
    import inspect as _inspect

    from multi_nd2_stitching.cli import build_parser
    from multi_nd2_stitching.offsets import build_plan

    cli_default = build_parser().parse_args(["status", "x.yaml"]).precision
    lib_default = _inspect.signature(build_plan).parameters["precision"].default
    assert cli_default == lib_default == "float32"


# --- graph --------------------------------------------------------------------
def test_graph_prints_the_routing(project, capsys):
    assert run("graph", project) == 0
    out = capsys.readouterr().out
    assert "ambiguous  0 timepoint(s)" in out
    assert "[origin]" in out and "[drift from t-1]" in out
    assert "x→ tile_b" in out or "y→ tile_b" in out


def test_graph_collapses_identical_timepoints(project, capsys):
    run("graph", project)
    out = capsys.readouterr().out
    assert "t=1..7" in out, "constant topology should print as one run"


def test_graph_tile_mode(project, capsys):
    assert run("graph", project, "--tile", "tile_b") == 0
    out = capsys.readouterr().out
    assert "tile_b" in out
    assert "└─" not in out


def test_graph_unknown_tile(project, capsys):
    assert run("graph", project, "--tile", "nope") == 1
    assert "unknown tile" in capsys.readouterr().err


def test_graph_between(project, capsys):
    run("graph", project, "--between", 0, 3)
    assert "timepoints 3" in capsys.readouterr().out


def test_graph_writes_a_file(project, tmp_path, capsys):
    out = tmp_path / "graph.txt"
    assert run("graph", project, "--out", out) == 0
    assert "[origin]" in out.read_text()


def test_graph_flags_a_dropped_tile(project, capsys, tmp_path, monkeypatch):
    import yaml as _y

    cfg = _y.safe_load(project.read_text())
    cfg["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    project.write_text(_y.safe_dump(cfg))
    run("graph", project)
    out = capsys.readouterr().out
    assert "t=3" in out


def test_graph_flags_shaped_peak(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["overrides"] = [{"at": 3, "shaped_peak": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("graph", project) == 0
    out = capsys.readouterr().out
    assert "t=3  [SHAPED_PEAK]" in out
    assert "[shaped_peak]" in out


def test_graph_quiet_about_shaped_peak_when_unused(project, capsys):
    run("graph", project)
    out = capsys.readouterr().out
    assert "SHAPED_PEAK" not in out and "shaped_peak" not in out


def test_validate_deep_flags_a_shaped_peak_pair_not_alive(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["overrides"] = [{"at": 3, "drop": ["tile_b"], "shaped_peak": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 1
    assert "not alive at t=3" in capsys.readouterr().err


def test_validate_deep_is_fine_with_a_live_shaped_peak_pair(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["overrides"] = [{"at": 3, "shaped_peak": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 0


def test_validate_deep_flags_a_realign_pair_not_alive(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["realignment_slices"] = {"y": [1, 2]}
    cfg["overrides"] = [{"at": 3, "drop": ["tile_b"], "realign": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 1
    assert "not alive at t=3" in capsys.readouterr().err


def test_validate_deep_flags_a_realign_pair_that_only_frees_its_own_axis(
    project, capsys
):
    cfg = yaml.safe_load(project.read_text())
    cfg["realignment_slices"] = {"x": [5, 15]}  # x is the pair's own axis here
    cfg["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 1
    assert "always freed for a pair" in capsys.readouterr().err


def test_validate_deep_is_fine_with_a_live_realign_pair(project, capsys):
    cfg = yaml.safe_load(project.read_text())
    cfg["realignment_slices"] = {"y": [1, 2]}
    cfg["overrides"] = [{"at": 3, "realign": ["tile_a,tile_b"]}]
    project.write_text(yaml.safe_dump(cfg))
    assert run("validate", project, "--deep") == 0


def test_graph_strict_exits_nonzero_on_ambiguity(project, capsys, monkeypatch):
    """Two anchors on one chain: the placement depends on traversal order."""
    import yaml as _y
    from helpers import make_meta

    from multi_nd2_stitching import metadata as _M

    monkeypatch.setattr(
        _M,
        "read_metadata",
        lambda paths: make_meta(n_files=2, nt=4, nz=4, ny=8, nx=8, paths=list(paths)),
    )
    cfg = _y.safe_load(project.read_text())
    cfg["positions"]["tile_b"]["reference_in_files"] = [0, 1]
    project.write_text(_y.safe_dump(cfg))
    assert run("graph", project, "--strict") == 1
    out = capsys.readouterr().out
    assert "anchors in one component" in out


def test_graph_only_ambiguous_filters(project, capsys, monkeypatch):
    import yaml as _y

    cfg = _y.safe_load(project.read_text())
    cfg["positions"]["tile_b"]["reference_in_files"] = [0, 1]
    project.write_text(_y.safe_dump(cfg))
    run("graph", project, "--only-ambiguous")
    out = capsys.readouterr().out
    assert "t=0\n" not in out, "t=0 has a single seed, so it is not ambiguous"
    assert "anchors in one component" in out


# --- skeleton blend -----------------------------------------------------------
def test_skeleton_draws_fewer_tiles(project, capsys):
    import zarr

    run("offsets", project, "--no-progress")
    capsys.readouterr()
    assert run("blend", project, "--skeleton", "--no-progress") == 0
    out = capsys.readouterr().out
    assert "skeleton" in out and "tile placements" in out
    assert zarr.open(str(Workspace.of(project).root / "skeleton.zarr"), mode="r").shape


def test_skeleton_uses_its_own_default_path(project, capsys):
    """A skeleton canvas is small; it must not fix the frame for the full blend."""
    run("offsets", project, "--no-progress")
    run("blend", project, "--skeleton", "--no-progress")
    ws = Workspace.of(project)
    assert (ws.root / "skeleton.zarr").exists()
    assert not ws.canvas.exists()


def test_skeleton_canvas_is_smaller_than_the_full_one(project, capsys):
    import json

    run("offsets", project, "--no-progress")
    run("blend", project, "--skeleton", "--no-progress")
    run("blend", project, "--no-progress")
    ws = Workspace.of(project)
    thin = json.loads((ws.root / "skeleton.zarr.geometry.json").read_text())
    full = json.loads((ws.root / "canvas.zarr.geometry.json").read_text())
    assert thin["shape"][3] < full["shape"][3]


def test_skeleton_and_full_can_share_an_output_only_via_recreate(project, tmp_path):
    """Explicit --output lets you collide the two; the frame guard catches it."""
    run("offsets", project, "--no-progress")
    out = tmp_path / "shared.zarr"
    run("blend", project, "--skeleton", "--output", out, "--no-progress")
    assert run("blend", project, "--output", out, "--no-progress") == 1


def test_skeleton_reports_the_reduction(project, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("blend", project, "--skeleton", "--no-progress")
    out = capsys.readouterr().out
    assert "of 16 tile placements" in out, "2 tiles x 8 timepoints"
    assert "8 of" in out, "one steady anchor means one tile per timepoint"


def test_full_blend_is_unaffected(project, capsys):
    run("offsets", project, "--no-progress")
    capsys.readouterr()
    run("blend", project, "--no-progress")
    assert "skeleton" not in capsys.readouterr().out


# --- corner: an FFT-fitted diagonal edge, end to end ---------------------------
def test_corner_override_places_a_diagonal_only_component(
    tmp_path, capsys, monkeypatch
):
    """x and y share no edge Pair at all -- without `corner` this component
    has no anchor; with it, stitch validate/graph both go clean end to end."""
    files = []
    for i in range(2):
        p = tmp_path / f"f{i}.nd2"
        p.write_bytes(b"x" * (10 + i))
        files.append(str(p))

    monkeypatch.setattr(
        M,
        "read_metadata",
        lambda paths: grid_meta(
            {"x": (0.0, 0.0), "y": (55.0, 55.0)}, list(paths), nt=2, nz=4, ny=8, nx=8
        ),
    )
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {
            "x": {"start": [0, 0], "reference_in_files": [0, 1]},
            "y": {"start": [0, 0]},
        },
        "overrides": [{"at": "0-3", "corner": ["x,y"]}],
    }
    cfg_path = tmp_path / "diag.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    assert run("validate", cfg_path, "--deep") == 0
    assert run("graph", cfg_path) == 0
    assert "corner→ y" in capsys.readouterr().out


def test_without_corner_the_diagonal_component_is_flagged(tmp_path, monkeypatch):
    files = []
    for i in range(2):
        p = tmp_path / f"f{i}.nd2"
        p.write_bytes(b"x" * (10 + i))
        files.append(str(p))

    monkeypatch.setattr(
        M,
        "read_metadata",
        lambda paths: grid_meta(
            {"x": (0.0, 0.0), "y": (55.0, 55.0)}, list(paths), nt=2, nz=4, ny=8, nx=8
        ),
    )
    cfg = {
        "files": files,
        "grid_spacing": 55,
        "grid_spacing_error": 5,
        "positions": {
            "x": {"start": [0, 0], "reference_in_files": [0, 1]},
            "y": {"start": [0, 0]},
        },
    }
    cfg_path = tmp_path / "diag.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert run("validate", cfg_path, "--deep") == 1

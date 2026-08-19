"""End-to-end CLI tests with a stubbed ND2 layer."""

import pytest
import yaml
from helpers import FakeReader, make_meta

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
    run("status", project, "--precision", "float32")
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
    assert "run `stitch offsets`" in capsys.readouterr().err


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

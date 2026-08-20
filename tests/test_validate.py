import pytest

from multi_nd2_stitching.validate import ConfigError, check, validate


def test_minimal_is_valid(cfg_dict, parse):
    assert check(parse(cfg_dict)) == []


def test_realistic_config_is_valid(cfg_dict, parse):
    """The ch6 shape: a reference that dies, replaced by two others."""
    cfg_dict["files"] = [f"f{i}.nd2" for i in range(9)]
    cfg_dict["positions"] = {
        "a": {"start": [0, 0], "end": 8, "reference_in_files": list(range(8))},
        "a1": {"start": [3, 20], "end": 8},
        "a2": {"start": [4, 0], "end": 8},
        "c": {"start": [4, 0], "reference_in_files": [8]},
        "a3": {"start": [5, 0], "reference_in_files": [8]},
    }
    cfg_dict["overrides"] = [
        {"at": 143, "reason": "artefact", "drop": ["a1"], "anchor": ["a2"]},
        {"at": [4, 16, 26, 38], "realign": ["a"]},
    ]
    assert check(parse(cfg_dict)) == []


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        # positions
        (lambda d: d["positions"]["tile_b"].update(start=[9, 0]), "out of range"),
        (lambda d: d["positions"]["tile_b"].update(end=0), "out of range"),
        (lambda d: d["positions"]["tile_b"].update(start=[1, 0], end=1), "never alive"),
        (
            lambda d: d["positions"]["tile_b"].update(aliases=["tile_a"]),
            "also a position name",
        ),
        (lambda d: d["positions"]["tile_b"].update(aliases=["tile_b"]), "its own name"),
        # references
        (
            lambda d: d["positions"]["tile_b"].update(end=1, reference_in_files=[1]),
            "anchors file 1 but is only alive",
        ),
        (
            lambda d: d["positions"]["tile_a"].update(reference_in_files=[0, 0, 1]),
            "duplicates",
        ),
        (
            lambda d: d["positions"]["tile_a"].update(reference_in_files=[0]),
            "no reference position",
        ),
        (
            lambda d: d["positions"]["tile_a"].update(reference_in_files=[]),
            "no reference position",
        ),
        (
            lambda d: d["positions"]["tile_a"].update(reference_in_files=[5]),
            "out of range",
        ),
        # global
        (lambda d: d.update(grid_spacing_error=30), "windows overlap"),
        (lambda d: d.update(files=["a.nd2", "a.nd2"]), "duplicates"),
    ],
)
def test_detects_problem(cfg_dict, parse, mutate, fragment):
    mutate(cfg_dict)
    problems = check(parse(cfg_dict))
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"at": 5, "drop": ["nope"]}, "unknown position"),
        ({"at": 5}, "block does nothing"),
        ({"at": [], "drop": ["tile_b"]}, "at: empty"),
        ({"at": -1, "drop": ["tile_b"]}, "negative timepoint"),
        ({"at": 5, "drop": ["tile_b"], "anchor": ["tile_b"]}, "same block"),
        ({"at": 0, "realign": ["tile_a"]}, "no predecessor"),
        # the coupling this schema exists to surface
        ({"at": 143, "drop": ["tile_a"]}, "without naming a replacement"),
    ],
)
def test_detects_override_problem(cfg_dict, parse, override, fragment):
    cfg_dict["overrides"] = [override]
    problems = check(parse(cfg_dict))
    assert any(fragment in p for p in problems), problems


def test_dropping_a_reference_with_an_anchor_is_fine(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 143, "drop": ["tile_a"], "anchor": ["tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_duplicate_timepoint_across_overrides(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {"at": 143, "drop": ["tile_b"]},
        {"at": [143], "realign": ["tile_a"]},
    ]
    problems = check(parse(cfg_dict))
    assert any("merge them into one block" in p for p in problems), problems


# --- tier 2: needs timepoints-per-file ---------------------------------------
def test_timeline_checks_are_skipped_without_nts(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 99999, "drop": ["tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_timeline_catches_out_of_range_timepoint(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 99999, "drop": ["tile_b"]}]
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("beyond timeline (nt=20)" in p for p in problems), problems


def test_timeline_catches_start_beyond_file(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"] = {"start": [0, 50]}
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("beyond file 0" in p for p in problems), problems


def test_timeline_catches_override_on_dead_tile(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"] = {"start": [0, 0], "end": 1}
    cfg_dict["overrides"] = [{"at": 15, "drop": ["tile_b"]}]
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("is not alive" in p for p in problems), problems


def test_timeline_length_mismatch(cfg_dict, parse):
    problems = check(parse(cfg_dict), nts=[10])
    assert any("1 file lengths for 2 files" in p for p in problems), problems


def test_validate_collects_all_problems(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"].update(start=[9, 0], end=0)
    with pytest.raises(ConfigError) as e:
        validate(parse(cfg_dict))
    assert len(e.value.problems) >= 2


def test_check_files_reports_missing_paths(cfg_dict, parse):
    problems = check(parse(cfg_dict), check_files=True)
    assert sum("does not exist" in p for p in problems) == 2


# --- unanchor -----------------------------------------------------------------
@pytest.mark.parametrize(
    "override,fragment",
    [
        (
            {"at": 5, "unanchor": ["tile_a"], "anchor": ["tile_a"]},
            "both unanchored and anchored",
        ),
        (
            {"at": 5, "unanchor": ["tile_a"], "drop": ["tile_a"]},
            "unanchor does nothing",
        ),
        ({"at": 5, "unanchor": ["nope"]}, "unknown position"),
        (
            {"at": 5, "unanchor": ["tile_a"]},
            "without naming a replacement",
        ),
    ],
)
def test_detects_unanchor_problem(cfg_dict, parse, override, fragment):
    cfg_dict["overrides"] = [override]
    problems = check(parse(cfg_dict))
    assert any(fragment in p for p in problems), problems


def test_a_clean_handover_passes(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"]["reference_in_files"] = [0, 1]
    cfg_dict["overrides"] = [
        {
            "at": 5,
            "reason": "tile_b carries the drift across the file boundary",
            "unanchor": ["tile_a"],
            "anchor": ["tile_b"],
        }
    ]
    assert check(parse(cfg_dict)) == []

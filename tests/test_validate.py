import pytest

from multi_nd2_stitching.validate import ConfigError, check, check_overview, validate


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
    cfg_dict["realignment_slices"] = {"y": [300, 700], "x": [300, 700]}
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
        # position_in_files
        (
            lambda d: d["positions"]["tile_b"].update(position_in_files={5: 0}),
            "out of range",
        ),
        (
            lambda d: d["positions"]["tile_b"].update(end=1, position_in_files={1: 0}),
            "names file 1 but is only alive",
        ),
        # missing_in_files
        (
            lambda d: d["positions"]["tile_b"].update(missing_in_files=[5]),
            "out of range",
        ),
        (
            lambda d: d["positions"]["tile_b"].update(end=1, missing_in_files=[1]),
            "outside 0..0",
        ),
        (
            lambda d: d["positions"]["tile_a"].update(missing_in_files=[0]),
            "also in reference_in_files",
        ),
        (
            lambda d: d["positions"]["tile_b"].update(
                position_in_files={0: 1}, missing_in_files=[0]
            ),
            "also in position_in_files",
        ),
        (
            lambda d: d["positions"]["tile_b"].update(missing_in_files=[0, 0]),
            "missing_in_files: contains duplicates",
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
        # shaped_peak
        ({"at": 5, "shaped_peak": ["nope"]}, "unknown position"),
        ({"at": 5, "shaped_peak": ["tile_a,nope"]}, "unknown position"),
        ({"at": 5, "shaped_peak": ["a,b,c"]}, "tile name or an 'a,b' pair"),
        # realign, pair form
        ({"at": 5, "realign": ["nope"]}, "unknown position"),
        ({"at": 5, "realign": ["tile_a,nope"]}, "unknown position"),
        ({"at": 5, "realign": ["a,b,c"]}, "tile name or an 'a,b' pair"),
        # near
        ({"at": 5, "near": {"tile_a": [0, 1, 2]}}, "no matching shaped_peak entry"),
        (
            {"at": 5, "shaped_peak": ["tile_a"], "near": {"tile_a": [0, 1]}},
            "must be [dz, dy, dx] (3 integers)",
        ),
        # corner
        ({"at": 5, "corner": ["nope,tile_a"]}, "unknown position"),
        ({"at": 5, "corner": ["tile_a"]}, "must be an 'a,b' pair"),
        ({"at": 5, "corner": ["tile_a,tile_b,tile_c"]}, "must be an 'a,b' pair"),
    ],
)
def test_detects_override_problem(cfg_dict, parse, override, fragment):
    cfg_dict["overrides"] = [override]
    problems = check(parse(cfg_dict))
    assert any(fragment in p for p in problems), problems


def test_dropping_a_reference_with_an_anchor_is_fine(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 143, "drop": ["tile_a"], "anchor": ["tile_b"]}]
    assert check(parse(cfg_dict)) == []


# --- position_in_files ---------------------------------------------------------
def test_position_in_files_collision_is_flagged(cfg_dict, parse):
    cfg_dict["positions"]["tile_a"]["position_in_files"] = {0: 1}
    cfg_dict["positions"]["tile_b"]["position_in_files"] = {0: 1}
    problems = check(parse(cfg_dict))
    assert any("also claimed by" in p for p in problems), problems


def test_position_in_files_mixed_with_a_named_file_is_fine(cfg_dict, parse):
    """One file resolved by index, the other still by name -- the whole
    point is being able to mix both for the same tile."""
    cfg_dict["positions"]["tile_b"]["position_in_files"] = {0: 1}
    assert check(parse(cfg_dict)) == []


# --- missing_in_files -----------------------------------------------------------
def test_missing_in_files_gap_is_fine(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"]["missing_in_files"] = [1]
    assert check(parse(cfg_dict)) == []


# --- shaped_peak ----------------------------------------------------------------
def test_shaped_peak_only_block_is_not_a_no_op(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "shaped_peak": ["tile_a,tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_shaped_peak_tile_name_form_is_fine(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "shaped_peak": ["tile_a"]}]
    assert check(parse(cfg_dict)) == []


def test_shaped_peak_with_a_matching_near_hint_is_fine(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {
            "at": 5,
            "shaped_peak": ["tile_a,tile_b"],
            "near": {"tile_a,tile_b": [0, 4, -2]},
        }
    ]
    assert check(parse(cfg_dict)) == []


# --- realign, pair form ----------------------------------------------------
def test_realign_pair_only_block_is_not_a_no_op(cfg_dict, parse):
    cfg_dict["realignment_slices"] = {"y": [1, 2]}
    cfg_dict["overrides"] = [{"at": 5, "realign": ["tile_a,tile_b"]}]
    assert check(parse(cfg_dict)) == []


# --- corner -----------------------------------------------------------------
def test_corner_only_block_is_not_a_no_op(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "corner": ["tile_a,tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_realign_pair_at_t0_has_no_predecessor_requirement(cfg_dict, parse):
    """Unlike a drift step, a pair correlation at t=0 needs no t-1 -- the
    'no predecessor' check must not fire for the pair form."""
    cfg_dict["realignment_slices"] = {"y": [1, 2]}
    cfg_dict["overrides"] = [{"at": 0, "realign": ["tile_a,tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_realign_still_flags_a_bare_name_at_t0(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 0, "realign": ["tile_a"]}]
    problems = check(parse(cfg_dict))
    assert any("no predecessor" in p for p in problems), problems


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


# --- stop_at --------------------------------------------------------------------
def test_stop_at_flags_an_override_past_it(cfg_dict, parse):
    cfg_dict["stop_at"] = 5
    cfg_dict["overrides"] = [{"at": 5, "drop": ["tile_b"]}]
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("beyond timeline (nt=5)" in p for p in problems), problems


def test_stop_at_allows_an_override_within_it(cfg_dict, parse):
    cfg_dict["stop_at"] = 5
    cfg_dict["overrides"] = [{"at": 4, "drop": ["tile_b"]}]
    assert check(parse(cfg_dict), nts=[10, 10]) == []


def test_stop_at_flags_a_tile_that_never_appears(cfg_dict, parse):
    cfg_dict["stop_at"] = 3
    cfg_dict["positions"]["tile_b"] = {"start": [1, 0]}  # global t=10
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("never appears" in p for p in problems), problems


def test_stop_at_unset_never_flags_never_appears(cfg_dict, parse):
    cfg_dict["positions"]["tile_b"] = {"start": [1, 0]}
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert not any("never appears" in p for p in problems), problems


def test_stop_at_does_not_relieve_a_file_of_needing_a_reference(cfg_dict, parse):
    """Deliberate scope boundary: a file entirely past stop_at still needs a
    reference in cfg.files -- stop_at truncates the timeline, not the file
    list; trim `files:` itself if a trailing file is not wanted at all."""
    cfg_dict["files"] = ["a.nd2", "b.nd2", "c.nd2"]
    cfg_dict["stop_at"] = 5  # entirely within file 0 (nts=[10, 10, 10])
    problems = check(parse(cfg_dict), nts=[10, 10, 10])
    assert any("have no reference position" in p for p in problems), problems


# --- exclude_at ---------------------------------------------------------------
def test_exclude_at_negative_timepoint_is_flagged(cfg_dict, parse):
    cfg_dict["exclude_at"] = [-1]
    problems = check(parse(cfg_dict))
    assert any("negative timepoint -1" in p for p in problems), problems


def test_exclude_at_duplicates_are_flagged(cfg_dict, parse):
    cfg_dict["exclude_at"] = [3, 3]
    problems = check(parse(cfg_dict))
    assert any("contains duplicates" in p for p in problems), problems


def test_exclude_at_clean_is_fine_without_nts(cfg_dict, parse):
    cfg_dict["exclude_at"] = [3, 4]
    assert check(parse(cfg_dict)) == []


def test_exclude_at_beyond_timeline_is_flagged(cfg_dict, parse):
    cfg_dict["exclude_at"] = [99999]
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("beyond timeline (nt=20)" in p for p in problems), problems


def test_exclude_at_within_timeline_is_fine(cfg_dict, parse):
    cfg_dict["exclude_at"] = [3, 4]
    assert check(parse(cfg_dict), nts=[10, 10]) == []


def test_exclude_at_removing_everything_is_flagged(cfg_dict, parse):
    cfg_dict["exclude_at"] = list(range(20))
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any("removes the entire timeline" in p for p in problems), problems


def test_exclude_at_flags_an_override_it_swallows(cfg_dict, parse):
    cfg_dict["exclude_at"] = [3]
    cfg_dict["overrides"] = [{"at": 3, "drop": ["tile_b"]}]
    problems = check(parse(cfg_dict), nts=[10, 10])
    assert any(
        "overrides[0].at: timepoint(s) 3 excluded by exclude_at" in p for p in problems
    ), problems


def test_exclude_at_leaves_an_override_elsewhere_alone(cfg_dict, parse):
    cfg_dict["exclude_at"] = [3]
    cfg_dict["overrides"] = [{"at": 4, "drop": ["tile_b"]}]
    assert check(parse(cfg_dict), nts=[10, 10]) == []


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


# --- realign needs a different crop, or it silently does nothing ---------------
def test_realign_without_realignment_slices_is_refused(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "realign": ["tile_a"]}]
    problems = check(parse(cfg_dict))
    assert any("realignment_slices is not set" in p for p in problems), problems


def test_realign_with_identical_slices_is_refused(cfg_dict, parse):
    """Same crop means the same cache key and the same answer: a no-op."""
    cfg_dict["overrides"] = [{"at": 5, "realign": ["tile_a"]}]
    cfg_dict["slices"] = {"z": [5, 40]}
    cfg_dict["realignment_slices"] = {"z": [5, 40]}
    problems = check(parse(cfg_dict))
    assert any("identical to slices" in p for p in problems), problems


def test_realign_with_a_different_crop_is_fine(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "realign": ["tile_a"]}]
    cfg_dict["slices"] = {"z": [5, 40]}
    cfg_dict["realignment_slices"] = {"z": [5, 40], "y": [300, 700]}
    assert check(parse(cfg_dict)) == []


def test_realignment_slices_alone_is_not_required(cfg_dict, parse):
    """No realign override means the field is simply unused."""
    cfg_dict["overrides"] = [{"at": 5, "drop": ["tile_b"]}]
    assert check(parse(cfg_dict)) == []


def test_the_message_names_the_tiles(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {"at": 5, "realign": ["tile_a"]},
        {"at": 9, "realign": ["tile_b"]},
    ]
    problems = check(parse(cfg_dict))
    assert any("['tile_a', 'tile_b']" in p for p in problems), problems


# --- overview -------------------------------------------------------------------
def test_no_overview_is_fine(cfg_dict, parse):
    assert check(parse(cfg_dict)) == []


def test_overview_with_file_and_channel_is_fine(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "overview.nd2", "channel": 1}
    assert check(parse(cfg_dict)) == []


def test_overview_without_channel_is_fine_at_tier_one(cfg_dict, parse):
    # Whether an unset channel is actually OK depends on how many positions
    # the file has -- that needs a read, so it's `check_overview`'s job, not
    # this tier's.
    cfg_dict["overview"] = {"file": "overview.nd2"}
    assert check(parse(cfg_dict)) == []


def test_overview_negative_channel_is_flagged(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "overview.nd2", "channel": -1}
    problems = check(parse(cfg_dict))
    assert any("overview.channel" in p for p in problems), problems


def test_overview_bad_reduction_is_flagged(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "overview.nd2", "reduction": "max"}
    problems = check(parse(cfg_dict))
    assert any("overview.reduction" in p for p in problems), problems


def test_overview_non_positive_max_output_px_is_flagged(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "overview.nd2", "max_output_px": 0}
    problems = check(parse(cfg_dict))
    assert any("overview.max_output_px" in p for p in problems), problems


def test_overview_missing_file_is_flagged_only_with_check_files(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "does-not-exist.nd2", "channel": 1}
    assert check(parse(cfg_dict)) == []
    problems = check(parse(cfg_dict), check_files=True)
    assert any("overview.file" in p for p in problems), problems


# --- check_overview (deep tier: opens the overview file) -----------------------
def test_check_overview_skips_when_no_overview(cfg_dict, parse):
    assert check_overview(parse(cfg_dict)) == []


def test_check_overview_skips_when_file_does_not_exist(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "does-not-exist.nd2", "channel": 1}
    assert check_overview(parse(cfg_dict)) == []


def test_check_overview_requires_a_channel_for_multiple_positions(
    cfg_dict, parse, monkeypatch, tmp_path
):
    from helpers import make_meta

    f = tmp_path / "overview.nd2"
    f.write_bytes(b"x")
    cfg_dict["overview"] = {"file": str(f)}
    meta = make_meta(n_files=1, tiles=("p0", "p1"))[0]
    monkeypatch.setattr(
        "multi_nd2_stitching.overview.read_overview_meta", lambda path: meta
    )
    problems = check_overview(parse(cfg_dict))
    assert any("must choose one" in p for p in problems), problems


def test_check_overview_flags_out_of_range_channel(
    cfg_dict, parse, monkeypatch, tmp_path
):
    from helpers import make_meta

    f = tmp_path / "overview.nd2"
    f.write_bytes(b"x")
    cfg_dict["overview"] = {"file": str(f), "channel": 5}
    meta = make_meta(n_files=1, tiles=("p0", "p1"))[0]
    monkeypatch.setattr(
        "multi_nd2_stitching.overview.read_overview_meta", lambda path: meta
    )
    problems = check_overview(parse(cfg_dict))
    assert any("out of range" in p for p in problems), problems


def test_check_overview_fine_with_a_valid_channel(
    cfg_dict, parse, monkeypatch, tmp_path
):
    from helpers import make_meta

    f = tmp_path / "overview.nd2"
    f.write_bytes(b"x")
    cfg_dict["overview"] = {"file": str(f), "channel": 0}
    meta = make_meta(n_files=1, tiles=("p0", "p1"))[0]
    monkeypatch.setattr(
        "multi_nd2_stitching.overview.read_overview_meta", lambda path: meta
    )
    assert check_overview(parse(cfg_dict)) == []

import pytest
from cattrs.errors import BaseValidationError

from multi_nd2_stitching.config import clamp_z


# --- defaults: every optional field must survive being absent -----------------
@pytest.mark.parametrize(
    "field,expected",
    [
        ("end", None),
        ("aliases", []),
        ("reference_in_files", []),
    ],
)
def test_position_optional_absent(cfg_dict, parse, field, expected):
    cfg_dict["positions"]["tile_b"].pop(field, None)
    assert getattr(parse(cfg_dict).positions["tile_b"], field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("shift_px", None),
        ("flip_x", False),
        ("flip_y", False),
        ("overrides", []),
        ("overview", None),
        ("stop_at", None),
    ],
)
def test_toplevel_optional_absent(cfg_dict, parse, field, expected):
    cfg_dict.pop(field, None)
    assert getattr(parse(cfg_dict), field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("shift_px", None),
        ("overview", None),
        ("stop_at", None),
    ],
)
def test_toplevel_optional_explicit_null(cfg_dict, parse, field, expected):
    cfg_dict[field] = None
    assert getattr(parse(cfg_dict), field) == expected


def test_stop_at_present(cfg_dict, parse):
    cfg_dict["stop_at"] = 143
    assert parse(cfg_dict).stop_at == 143


# --- explicit YAML null must behave like absent -------------------------------
# `field:` with nothing after it is the most common hand-editing accident.
@pytest.mark.parametrize("field", ["aliases", "reference_in_files"])
def test_position_explicit_null(cfg_dict, parse, field):
    cfg_dict["positions"]["tile_b"][field] = None
    assert getattr(parse(cfg_dict).positions["tile_b"], field) == []


@pytest.mark.parametrize(
    "field", ["drop", "anchor", "realign", "shaped_peak", "reason"]
)
def test_override_explicit_null(cfg_dict, parse, field):
    cfg_dict["overrides"] = [{"at": 5, "drop": ["tile_b"], field: None}]
    o = parse(cfg_dict).overrides[0]
    assert getattr(o, field) in ([], None)


def test_overrides_null(cfg_dict, parse):
    cfg_dict["overrides"] = None
    assert parse(cfg_dict).overrides == []


# --- required fields must fail loudly ----------------------------------------
@pytest.mark.parametrize("field", ["files", "grid_spacing", "positions"])
def test_required_missing_raises(cfg_dict, parse, field):
    """cattrs wraps per-field failures, so the concrete type varies by version;
    what matters is that a missing required field never structures silently."""
    cfg_dict.pop(field)
    with pytest.raises((KeyError, ValueError, TypeError, BaseValidationError)):
        parse(cfg_dict)


# --- `at` accepts a scalar or a list ------------------------------------------
@pytest.mark.parametrize(
    "given,expected",
    [
        (143, (143,)),
        ([143], (143,)),
        ([4, 16, 26], (4, 16, 26)),
        ("20-23", (20, 21, 22, 23)),
        (["20-23"], (20, 21, 22, 23)),
        (["4-4"], (4,)),  # single-point range
        (["20-23", 45], (20, 21, 22, 23, 45)),
        ([4, "20-23", 45], (4, 20, 21, 22, 23, 45)),
    ],
)
def test_at_scalar_or_list(cfg_dict, parse, given, expected):
    cfg_dict["overrides"] = [{"at": given, "drop": ["tile_b"]}]
    assert parse(cfg_dict).overrides[0].at == expected


@pytest.mark.parametrize("given", ["20-4x", "20-", "-20", "abc"])
def test_at_range_malformed_is_loud(cfg_dict, parse, given):
    cfg_dict["overrides"] = [{"at": [given], "drop": ["tile_b"]}]
    with pytest.raises((ValueError, BaseValidationError)):
        parse(cfg_dict)


def test_at_range_reversed_is_loud(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": ["40-20"], "drop": ["tile_b"]}]
    with pytest.raises((ValueError, BaseValidationError)):
        parse(cfg_dict)


# --- alive_in_file: `end` is EXCLUSIVE ----------------------------------------
@pytest.mark.parametrize(
    "start,end,file_i,alive",
    [
        (0, 8, 7, True),
        (0, 8, 8, False),  # end=8 -> last alive file is 7
        (3, 8, 2, False),
        (3, 8, 3, True),
        (4, None, 8, True),  # open-ended -> alive to the last file
    ],
)
def test_alive_in_file(cfg_dict, parse, start, end, file_i, alive):
    cfg_dict["files"] = [f"f{i}.nd2" for i in range(9)]
    cfg_dict["positions"]["tile_b"] = {"start": [start, 0], "end": end}
    pos = parse(cfg_dict).positions["tile_b"]
    assert pos.alive_in_file(file_i, 9) is alive


# --- override lookups ---------------------------------------------------------
def test_override_lookups(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {"at": 143, "drop": ["tile_b"], "anchor": ["tile_a"]},
        {"at": [4, 16], "realign": ["tile_a"]},
    ]
    cfg = parse(cfg_dict)
    assert cfg.dropped_at(143) == {"tile_b"}
    assert cfg.anchored_at(143) == {"tile_a"}
    assert cfg.dropped_at(4) == set()
    assert cfg.realigned_at(16) == {"tile_a"}


# --- shaped_peak: tile-name and 'a,b' pair forms, side by side with the other
# four verbs, none of which it should interfere with -----------------------
def test_shaped_peak_at_holds_tile_names_and_pairs(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {"at": 21, "shaped_peak": ["tile_a", "tile_a,tile_b"]},
    ]
    cfg = parse(cfg_dict)
    assert cfg.shaped_peak_at(21) == {"tile_a", "tile_a,tile_b"}
    assert cfg.shaped_peak_at(22) == set()


def test_shaped_peak_is_not_part_of_names(cfg_dict, parse):
    """It never changes graph membership -- unlike the other four verbs, it
    must not show up in Override.names."""
    cfg_dict["overrides"] = [{"at": 5, "shaped_peak": ["tile_a,tile_b"]}]
    o = parse(cfg_dict).overrides[0]
    assert o.names == set()


def test_shaped_peak_coexists_with_other_verbs_in_one_block(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {
            "at": 5,
            "drop": ["tile_b"],
            "anchor": ["tile_a"],
            "shaped_peak": ["tile_a"],
        }
    ]
    cfg = parse(cfg_dict)
    assert cfg.dropped_at(5) == {"tile_b"}
    assert cfg.shaped_peak_at(5) == {"tile_a"}


# --- near: a rough manual estimate attached to a shaped_peak entry ------------
def test_near_absent_defaults_to_empty_dict(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "shaped_peak": ["tile_a"]}]
    assert parse(cfg_dict).overrides[0].near == {}


def test_near_explicit_null_defaults_to_empty_dict(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 5, "shaped_peak": ["tile_a"], "near": None}]
    assert parse(cfg_dict).overrides[0].near == {}


def test_near_hint_returns_the_configured_shift(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {
            "at": 21,
            "shaped_peak": ["tile_a,tile_b"],
            "near": {"tile_a,tile_b": [0, 12, -4]},
        }
    ]
    cfg = parse(cfg_dict)
    assert cfg.near_hint("tile_a,tile_b", 21) == (0, 12, -4)


def test_near_hint_absent_for_the_name_returns_none(cfg_dict, parse):
    cfg_dict["overrides"] = [{"at": 21, "shaped_peak": ["tile_a"]}]
    cfg = parse(cfg_dict)
    assert cfg.near_hint("tile_a", 21) is None


def test_near_hint_wrong_timepoint_returns_none(cfg_dict, parse):
    cfg_dict["overrides"] = [
        {"at": 21, "shaped_peak": ["tile_a"], "near": {"tile_a": [1, 2, 3]}}
    ]
    cfg = parse(cfg_dict)
    assert cfg.near_hint("tile_a", 22) is None


# --- slices -------------------------------------------------------------------
def test_slices_absent_is_full(cfg_dict, parse):
    assert parse(cfg_dict).slices == (slice(None),) * 3


def test_slices_partial_axis(cfg_dict, parse):
    cfg_dict["slices"] = {"z": [5, 100]}
    assert parse(cfg_dict).slices == (slice(5, 100), slice(None), slice(None))


def test_slices_null_is_full(cfg_dict, parse):
    cfg_dict["slices"] = None
    assert parse(cfg_dict).slices == (slice(None),) * 3


@pytest.mark.parametrize(
    "stop,min_nz,expected",
    [
        (100, 60, 60),
        (40, 60, 40),
        (None, 60, 60),
    ],
)
def test_clamp_z(stop, min_nz, expected):
    assert clamp_z((slice(5, stop), slice(None), slice(None)), min_nz)[0] == slice(
        5, expected
    )


def test_clamp_z_leaves_yx_alone():
    yx = (slice(300, 724), slice(1, 2))
    assert clamp_z((slice(5, 100), *yx), 60)[1:] == yx


# --- overview -------------------------------------------------------------------
def test_overview_present(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "overview.nd2", "channel": "pos1"}
    ov = parse(cfg_dict).overview
    assert ov.file == "overview.nd2"
    assert ov.channel == "pos1"
    assert ov.label is True


def test_overview_label_can_be_disabled(cfg_dict, parse):
    cfg_dict["overview"] = {"file": "o.nd2", "channel": "p1", "label": False}
    assert parse(cfg_dict).overview.label is False

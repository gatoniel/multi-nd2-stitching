import pytest

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
    ],
)
def test_toplevel_optional_absent(cfg_dict, parse, field, expected):
    cfg_dict.pop(field, None)
    assert getattr(parse(cfg_dict), field) == expected


# --- explicit YAML null must behave like absent -------------------------------
# `field:` with nothing after it is the most common hand-editing accident.
@pytest.mark.parametrize("field", ["aliases", "reference_in_files"])
def test_position_explicit_null(cfg_dict, parse, field):
    cfg_dict["positions"]["tile_b"][field] = None
    assert getattr(parse(cfg_dict).positions["tile_b"], field) == []


@pytest.mark.parametrize("field", ["drop", "anchor", "realign", "reason"])
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
    cfg_dict.pop(field)
    with pytest.raises(Exception):
        parse(cfg_dict)


# --- `at` accepts a scalar or a list ------------------------------------------
@pytest.mark.parametrize(
    "given,expected",
    [
        (143, (143,)),
        ([143], (143,)),
        ([4, 16, 26], (4, 16, 26)),
    ],
)
def test_at_scalar_or_list(cfg_dict, parse, given, expected):
    cfg_dict["overrides"] = [{"at": given, "drop": ["tile_b"]}]
    assert parse(cfg_dict).overrides[0].at == expected


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

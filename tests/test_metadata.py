import json

import attrs
import pytest
from helpers import make_meta

from multi_nd2_stitching import metadata as M


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "sub" / "meta.json"


@pytest.fixture
def fake_nd2(monkeypatch):
    """Stand in for read_metadata and count how often it is hit."""
    calls = []

    def fake(paths):
        calls.append(list(paths))
        return make_meta(n_files=len(paths))

    monkeypatch.setattr(M, "read_metadata", fake)
    return calls


@pytest.fixture
def files(tmp_path):
    out = []
    for i in range(2):
        p = tmp_path / f"f{i}.nd2"
        p.write_bytes(b"x" * (10 + i))
        out.append(p)
    return out


# --- roundtrip ----------------------------------------------------------------
def test_roundtrip_is_lossless():
    blob = M.MetadataCache(
        stamp=(M.FileStamp("f0.nd2", 10, 1),), metadata=make_meta(n_files=1)
    )
    assert M.converter.loads(M.converter.dumps(blob), M.MetadataCache) == blob


def test_tuples_survive_json():
    """JSON has no tuples; cattrs must put them back, or `stage_um` becomes lists."""
    meta = make_meta(n_files=1, tiles=("a", "b"))
    back = M.converter.loads(M.converter.dumps(meta), M.Metadata)
    assert isinstance(back[0].stage_um, tuple)
    assert isinstance(back[0].stage_um[0], tuple)
    assert isinstance(back[0].position_names, tuple)
    assert back == meta


# --- caching behaviour --------------------------------------------------------
def test_no_cache_path_always_reads(files, fake_nd2):
    M.load_metadata(files)
    M.load_metadata(files)
    assert len(fake_nd2) == 2


def test_second_call_hits_the_cache(files, fake_nd2, cache_file):
    a = M.load_metadata(files, cache=cache_file)
    b = M.load_metadata(files, cache=cache_file)
    assert len(fake_nd2) == 1
    assert a == b


def test_cache_parent_is_created(files, fake_nd2, cache_file):
    M.load_metadata(files, cache=cache_file)
    assert cache_file.exists()


def test_touching_a_file_invalidates(files, fake_nd2, cache_file):
    M.load_metadata(files, cache=cache_file)
    files[0].write_bytes(b"y" * 999)  # size changes -> stamp changes
    M.load_metadata(files, cache=cache_file)
    assert len(fake_nd2) == 2


def test_different_file_list_invalidates(files, fake_nd2, cache_file):
    M.load_metadata(files, cache=cache_file)
    M.load_metadata(files[:1], cache=cache_file)
    assert len(fake_nd2) == 2


def test_corrupt_cache_is_ignored_not_fatal(files, fake_nd2, cache_file):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{ not json")
    meta = M.load_metadata(files, cache=cache_file)
    assert len(meta) == 2
    assert len(fake_nd2) == 1


def test_cache_written_by_an_older_schema_is_ignored(files, fake_nd2, cache_file):
    """Add a field to FileMeta and every existing cache must fall back, not crash."""
    M.load_metadata(files, cache=cache_file)
    blob = json.loads(cache_file.read_text())
    for f in blob["metadata"]["files"]:
        del f["voxel_x_um"]
    cache_file.write_text(json.dumps(blob))
    M.load_metadata(files, cache=cache_file)
    assert len(fake_nd2) == 2


def test_cache_only_affects_runtime(files, fake_nd2, cache_file):
    """The invariant: deleting the cache changes nothing but speed."""
    cached = M.load_metadata(files, cache=cache_file)
    cache_file.unlink()
    assert M.load_metadata(files, cache=cache_file) == cached


# --- FileStamp ----------------------------------------------------------------
def test_stamp_reflects_size(files):
    before = M.FileStamp.of(files[0])
    files[0].write_bytes(b"z" * 500)
    assert M.FileStamp.of(files[0]) != before


def test_stamp_is_stable_for_an_untouched_file(files):
    assert M.FileStamp.of(files[0]) == M.FileStamp.of(files[0])


# --- position_of --------------------------------------------------------------
@pytest.mark.parametrize(
    "names,expected",
    [
        (("a",), 0),
        (("b",), 1),
        (("nope",), None),
        (("nope", "b"), 1),
    ],
)
def test_position_of(names, expected):
    fm = make_meta(n_files=1, tiles=("a", "b"))[0]
    assert fm.position_of(names) == expected


def test_position_of_is_loud_on_ambiguity():
    fm = attrs.evolve(make_meta(n_files=1)[0], position_names=("a", "a"))
    with pytest.raises(ValueError, match="match several positions"):
        fm.position_of(("a",))

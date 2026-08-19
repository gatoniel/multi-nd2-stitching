import json

from multi_nd2_stitching.offsets import Crop, TimeTask, VolumeRef
from multi_nd2_stitching.store import Offset, OffsetStore


def task(i=0):
    return TimeTask(
        name=f"t{i}",
        t_from=i,
        t_to=i + 1,
        src=VolumeRef("f0", 0, i, 16),
        dst=VolumeRef("f0", 0, i + 1, 16),
        crop=Crop((None, None), (None, None), (None, None)),
    )


def test_in_memory_store_needs_no_path():
    s = OffsetStore()
    s.put(task(), Offset(1, 2, 3))
    assert s[task().key] == Offset(1, 2, 3)


def test_survives_a_reopen(tmp_path):
    p = tmp_path / "off.jsonl"
    OffsetStore(p).put(task(), Offset(1, 2, 3))
    assert OffsetStore(p)[task().key] == Offset(1, 2, 3)


def test_parent_directory_is_created(tmp_path):
    p = tmp_path / "deep" / "off.jsonl"
    OffsetStore(p).put(task(), Offset(0, 0, 0))
    assert p.exists()


def test_last_write_wins(tmp_path):
    p = tmp_path / "off.jsonl"
    s = OffsetStore(p)
    s.put(task(), Offset(1, 1, 1))
    s.put(task(), Offset(9, 9, 9))
    assert OffsetStore(p)[task().key] == Offset(9, 9, 9)
    assert len(p.read_text().strip().splitlines()) == 2  # append-only


def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    """A killed process can leave half a line. That must not lose the rest."""
    p = tmp_path / "off.jsonl"
    s = OffsetStore(p)
    s.put(task(0), Offset(1, 1, 1))
    s.put(task(1), Offset(2, 2, 2))
    with p.open("a") as f:
        f.write('{"key": "abc", "off')
    reopened = OffsetStore(p)
    assert len(reopened) == 2
    assert reopened.skipped == 1


def test_record_is_human_readable(tmp_path):
    p = tmp_path / "off.jsonl"
    OffsetStore(p).put(task(7), Offset(0, -3, 5))
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["kind"] == "time"
    assert "t7" in rec["task"]["what"]
    assert rec["offset"] == {"dz": 0, "dy": -3, "dx": 5}


def test_describe_returns_the_readable_record(tmp_path):
    s = OffsetStore(tmp_path / "off.jsonl")
    s.put(task(3), Offset(0, 0, 0))
    assert "3->4" in s.describe(task(3).key)["what"]


def test_compact_drops_superseded_lines(tmp_path):
    p = tmp_path / "off.jsonl"
    s = OffsetStore(p)
    for _ in range(5):
        s.put(task(), Offset(1, 1, 1))
    assert len(p.read_text().strip().splitlines()) == 5
    assert s.compact() == 1
    assert len(p.read_text().strip().splitlines()) == 1
    assert OffsetStore(p)[task().key] == Offset(1, 1, 1)


def test_compact_is_never_required_for_correctness(tmp_path):
    p = tmp_path / "off.jsonl"
    s = OffsetStore(p)
    s.put(task(0), Offset(1, 1, 1))
    s.put(task(1), Offset(2, 2, 2))
    before = {k: s[k] for k in (task(0).key, task(1).key)}
    s.compact()
    after = OffsetStore(p)
    assert {k: after[k] for k in before} == before


def test_offset_roundtrips_to_array():
    import numpy as np

    assert np.array_equal(Offset(1, -2, 3).as_array(), np.array([1, -2, 3]))

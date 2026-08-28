import attrs
import numpy as np
import pytest
from helpers import build, make_meta, stub_files

from multi_nd2_stitching.metadata import Metadata
from multi_nd2_stitching.times import (
    TimeRow,
    build_time_table,
    write_csv,
    write_npy,
)


@pytest.fixture
def setup(cfg_dict, tmp_path):
    def _make(nt=3, n_files=2, tiles=("tile_a", "tile_b")):
        files = stub_files(tmp_path, n_files)
        cfg_dict["files"] = files
        meta = make_meta(n_files=n_files, nt=nt, tiles=tiles, paths=files)
        lay = build(cfg_dict, n_files=n_files, nt=nt, tiles=tiles, paths=files)
        return lay, meta

    return _make


def test_one_row_per_global_timepoint(setup):
    lay, meta = setup(nt=3, n_files=2)
    rows = build_time_table(lay, meta)
    assert [r.t for r in rows] == list(range(lay.nt))
    assert lay.nt == 6  # 2 files * 3 timepoints


def test_locates_file_and_local_t(setup):
    lay, meta = setup(nt=3, n_files=2)
    rows = build_time_table(lay, meta)
    assert (rows[0].file, rows[0].local_t) == (0, 0)
    assert (rows[3].file, rows[3].local_t) == (1, 0)


def test_real_time_matches_metadata(setup):
    lay, meta = setup(nt=3, n_files=2)
    rows = build_time_table(lay, meta)
    for r in rows:
        assert r.real_time_s == meta[r.file].real_time_s[r.local_t]
        assert not r.skipped


def test_missing_entry_is_a_gap_not_a_crash(setup):
    lay, meta = setup(nt=3, n_files=2)
    short = attrs.evolve(meta.files[0], real_time_s=(0.0,))  # only t=0 known
    meta = Metadata((short, meta.files[1]))
    rows = build_time_table(lay, meta)
    assert rows[0].real_time_s == 0.0
    assert rows[1].real_time_s is None
    assert rows[1].skipped


def test_real_time_iso_round_trips_utc():
    row = TimeRow(t=0, file=0, local_t=0, real_time_s=0.0, skipped=False)
    assert row.real_time_iso == "1970-01-01T00:00:00+00:00"


def test_real_time_iso_is_none_when_skipped():
    row = TimeRow(t=0, file=0, local_t=0, real_time_s=None, skipped=True)
    assert row.real_time_iso is None


def test_write_csv_round_trips(tmp_path):
    rows = [
        TimeRow(t=0, file=0, local_t=0, real_time_s=0.0, skipped=False),
        TimeRow(t=1, file=0, local_t=1, real_time_s=None, skipped=True),
    ]
    out = tmp_path / "sub" / "times.csv"
    write_csv(rows, out)
    text = out.read_text()
    assert "t,file,local_t,real_time_iso,real_time_s,skipped" in text
    assert "1970-01-01T00:00:00+00:00" in text
    assert ",True" in text  # the skipped row


def test_write_npy_round_trips(tmp_path):
    rows = [
        TimeRow(t=0, file=0, local_t=0, real_time_s=0.0, skipped=False),
        TimeRow(t=1, file=0, local_t=1, real_time_s=None, skipped=True),
    ]
    out = tmp_path / "times.npy"
    write_npy(rows, out)
    arr = np.load(out)
    assert arr["t"].tolist() == [0, 1]
    assert arr["real_time_s"][0] == 0.0
    assert np.isnan(arr["real_time_s"][1])
    assert arr["skipped"].tolist() == [False, True]

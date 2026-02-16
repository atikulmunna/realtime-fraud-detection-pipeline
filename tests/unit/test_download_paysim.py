from pathlib import Path

from src.data.download_paysim import copy_csv_to_raw, select_csv


def test_select_csv_uses_preferred_name(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x", encoding="utf-8")
    b.write_text("y", encoding="utf-8")

    chosen = select_csv(tmp_path, preferred_name="b.csv")
    assert chosen.name == "b.csv"


def test_select_csv_picks_largest_when_multiple(tmp_path: Path):
    small = tmp_path / "small.csv"
    large = tmp_path / "large.csv"
    small.write_text("1", encoding="utf-8")
    large.write_text("123456", encoding="utf-8")

    chosen = select_csv(tmp_path)
    assert chosen.name == "large.csv"


def test_copy_csv_to_raw_copies_file(tmp_path: Path):
    src = tmp_path / "source.csv"
    src.write_text("col\n1\n", encoding="utf-8")
    out_dir = tmp_path / "raw"

    copied = copy_csv_to_raw(src, out_dir)
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "col\n1\n"

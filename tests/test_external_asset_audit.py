from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.scripts.audit_safeconf_external_assets import split_summary


def test_split_summary_supports_headerless_official_format(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    path.write_text("cell_a,train\ncell_b,val\ncell_c,test\n")
    got = split_summary(path)
    assert got["rows"] == 3
    assert got["unique_ids"] == 3
    assert got["split_counts"] == {"test": 1, "train": 1, "val": 1}


def test_split_summary_supports_explicit_header(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    path.write_text("cell_id,split\ncell_a,train\ncell_b,train\n")
    got = split_summary(path)
    assert got["rows"] == 2
    assert got["unique_ids"] == 2
    assert got["split_counts"] == {"train": 2}

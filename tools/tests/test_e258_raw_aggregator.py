"""Small isolation test for the native E258 raw-count aggregator."""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/scripts/aggregate_e258_raw_dev_counts.cpp"


def test_aggregator_skips_unmapped_numeric_tokens(tmp_path: Path) -> None:
    executable = tmp_path / "aggregate"
    subprocess.run(["g++", "-O2", "-std=c++17", str(SOURCE), "-lz", "-o",
                    str(executable)], check=True)
    counts = tmp_path / "tiny.csv.gz"
    with gzip.open(counts, "wt") as stream:
        stream.write(",allowed1,forbidden,allowed2,control\n")
        stream.write("ENSG1:GENE1:Gene-Expression,2,SECRET,3,5\n")
        stream.write("ENSG2:OTHER:Gene-Expression,7,PRIVATE,1,0\n")
        stream.write("ENSG3:GENE1:Gene-Expression,1,NOPE,2,0\n")
    mapping = tmp_path / "map.i32"
    np.asarray([0, -1, 0, 1], dtype=np.int32).tofile(mapping)
    axis = tmp_path / "axis.txt"
    axis.write_text("GENE1\n")
    output = tmp_path / "result"
    run = subprocess.run([str(executable), str(counts), str(mapping), str(axis),
                          "2", str(output)], check=True, capture_output=True, text=True)
    assert '"test_target_numeric_tokens_parsed":0' in run.stdout
    assert np.fromfile(f"{output}.u32", dtype=np.uint32).reshape(2, 2).tolist() == [
        [5, 5], [3, 0]]
    assert np.fromfile(f"{output}.library.u64", dtype=np.uint64).tolist() == [16, 5]
    assert Path(f"{output}.genes.txt").read_text().splitlines() == [
        "ENSG1:GENE1:Gene-Expression", "ENSG3:GENE1:Gene-Expression"]

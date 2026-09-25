"""Integration guard: selected data export excludes an adjacent test treatment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/scripts'))
from run_e263_tahoe_pretruth_expression_extract import run  # noqa: E402


def test_pretruth_extract_writes_only_registered_rows(tmp_path: Path) -> None:
    raw = tmp_path / 'raw'
    raw.mkdir()
    pq.write_table(pa.table({
        'BARCODE_SUB_LIB_ID': ['train_treated', 'test_control', 'test_treated'],
        'sample': ['s1', 's2', 's3'],
        'cell_line_id': ['train_line', 'test_line', 'test_line'],
        'drug': ['DrugA', 'DMSO_TF', 'DrugA'],
        'plate': ['p1', 'p1', 'p1'],
        'genes': pa.array([[1, 2], [1], [1, 2]], type=pa.list_(pa.int64())),
        'expressions': pa.array([[2.0, 3.0], [1.0], [9.0, 9.0]],
                                type=pa.list_(pa.float32())),
    }), raw / 'part.parquet', row_group_size=3)
    selected = tmp_path / 'selected.parquet'
    pd.DataFrame([
        {'barcode': 'train_treated', 'sample': 's1', 'cell_line': 'train_line',
         'drug_dose': 'DrugA_1uM', 'plate': 'p1', 'split': 'train',
         'role': 'train_validation_treatment'},
        {'barcode': 'test_control', 'sample': 's2', 'cell_line': 'test_line',
         'drug_dose': '', 'plate': 'p1', 'split': 'test_sealed',
         'role': 'available_control'},
    ]).to_parquet(selected, index=False)
    exact = tmp_path / 'exact.csv'
    pd.DataFrame([{'shard': 'part.parquet', 'row_group': 0,
                   'selected_samples_present': 's1;s2'}]).to_csv(exact, index=False)
    output = tmp_path / 'output.parquet'
    status = run(argparse.Namespace(selected=selected, exact=exact,
                                    raw_dir=raw, output=output,
                                    status=tmp_path / 'status.json'))
    assert status['n_extracted_cells'] == 2
    assert status['n_test_treated_records_exported'] == 0
    assert set(pd.read_parquet(output).barcode) == {'train_treated', 'test_control'}

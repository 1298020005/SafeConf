#!/usr/bin/env python3
"""Extract only E261-selected Tahoe cells from E262b exact row groups.

Raw Parquet row groups must be decompressed to locate selected barcodes and
obtain sparse expression, so unrelated cells can be physically read inside
those row groups. Only the pre-registered train/validation treated cells and
available DMSO controls are retained or written. In particular, no test-line
treated record, effect, or error is exported to the development artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from run_e260_tahoe_metadata_panel import sha256
from run_e262_tahoe_pretruth_shard_map import DEFAULT_RAW


READ_COLUMNS = ['BARCODE_SUB_LIB_ID', 'sample', 'cell_line_id', 'drug',
                'plate', 'genes', 'expressions']
OUT_SCHEMA = pa.schema([
    ('barcode', pa.string()), ('sample', pa.string()),
    ('cell_line', pa.string()), ('drug', pa.string()),
    ('plate', pa.string()), ('drug_dose', pa.string()),
    ('split', pa.string()), ('role', pa.string()),
    ('genes', pa.list_(pa.int64())),
    ('expressions', pa.list_(pa.float32())),
])


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.status.exists() or \
       args.output.with_suffix(args.output.suffix + '.inprogress').exists():
        raise FileExistsError('E263 output or in-progress file already exists')
    selected = pd.read_parquet(args.selected)
    exact = pd.read_csv(args.exact)
    if selected.barcode.duplicated().any() or \
       ((selected.split == 'test_sealed') &
        (selected.role != 'available_control')).any():
        raise ValueError('E261 selected inventory invalid')
    if exact[['shard', 'row_group']].duplicated().any():
        raise ValueError('duplicate exact raw row-group mapping')
    lookup = selected.set_index('barcode').to_dict('index')
    selected_by_sample = {sample: set(group.barcode.astype(str))
                          for sample, group in selected.groupby('sample', sort=False)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.inprogress')
    writer = pq.ParquetWriter(temporary, OUT_SCHEMA, compression='zstd')
    found: set[str] = set()
    count_by_role: dict[str, int] = {}
    try:
        for i, (shard, frame) in enumerate(exact.groupby('shard', sort=True)):
            parquet = pq.ParquetFile(args.raw_dir / shard)
            for item in frame.itertuples(index=False):
                group = int(item.row_group)
                sample_names = str(item.selected_samples_present).split(';')
                possible = set().union(*(selected_by_sample[sample]
                                         for sample in sample_names))
                keys = pa.array(sorted(possible), type=pa.string())
                barcodes = parquet.read_row_group(group,
                                                  columns=['BARCODE_SUB_LIB_ID'])[
                                                      'BARCODE_SUB_LIB_ID']
                mask = pc.is_in(barcodes, value_set=keys)
                if pc.sum(mask).as_py() == 0:
                    continue
                data = parquet.read_row_group(group, columns=READ_COLUMNS).filter(mask)
                barcode_values = data['BARCODE_SUB_LIB_ID'].to_pylist()
                sample_values = data['sample'].to_pylist()
                line_values = data['cell_line_id'].to_pylist()
                drug_values = data['drug'].to_pylist()
                plate_values = data['plate'].to_pylist()
                roles, splits, conditions = [], [], []
                for barcode, sample, line, drug, plate in zip(
                        barcode_values, sample_values, line_values,
                        drug_values, plate_values):
                    if barcode in found:
                        raise ValueError(f'duplicate extracted barcode: {barcode}')
                    record = lookup[barcode]
                    if (str(sample) != str(record['sample']) or
                        str(line) != str(record['cell_line']) or
                        str(plate) != str(record['plate'])):
                        raise ValueError(f'raw/metadata mismatch: {barcode}')
                    role = str(record['role'])
                    split = str(record['split'])
                    if (role == 'available_control') != (str(drug) == 'DMSO_TF'):
                        raise ValueError(f'control/treatment mismatch: {barcode}')
                    if split == 'test_sealed' and role != 'available_control':
                        raise ValueError('test treated cell would be written')
                    found.add(barcode)
                    roles.append(role)
                    splits.append(split)
                    conditions.append(str(record['drug_dose'])
                                      if role != 'available_control' else '')
                    count_by_role[role] = count_by_role.get(role, 0) + 1
                out = pa.Table.from_arrays([
                    pa.array(barcode_values, type=pa.string()),
                    pa.array(sample_values, type=pa.string()),
                    pa.array(line_values, type=pa.string()),
                    pa.array(drug_values, type=pa.string()),
                    pa.array(plate_values, type=pa.string()),
                    pa.array(conditions, type=pa.string()),
                    pa.array(splits, type=pa.string()),
                    pa.array(roles, type=pa.string()),
                    data['genes'].combine_chunks().cast(pa.list_(pa.int64())),
                    data['expressions'].combine_chunks().cast(pa.list_(pa.float32())),
                ], schema=OUT_SCHEMA)
                writer.write_table(out)
            if (i + 1) % 100 == 0:
                print(f'expression shards {i+1}/{exact.shard.nunique()}, '
                      f'cells {len(found)}/{len(selected)}', flush=True)
    finally:
        writer.close()
    if found != set(lookup):
        missing = set(lookup) - found
        raise ValueError(f'{len(missing)} selected barcodes absent from raw expression')
    os.replace(temporary, args.output)
    status = {'status': 'PRETRUTH_EXPRESSION_EXTRACT_COMPLETE',
              'completed_at': datetime.now().astimezone().isoformat(),
              'selected_barcode_sha256': sha256(args.selected),
              'exact_map_sha256': sha256(args.exact),
              'n_selected_cells': len(selected),
              'n_extracted_cells': len(found),
              'n_shards_read': exact.shard.nunique(),
              'n_candidate_row_groups': len(exact),
              'count_by_role': count_by_role,
              'n_test_treated_records_exported': 0,
              'expression_file': str(args.output),
              'expression_file_size_bytes': args.output.stat().st_size,
              'expression_file_sha256': sha256(args.output),
              'scope_note': 'Some unrelated raw cells may have been decompressed within selected row groups; no test treated row is persisted, aggregated or evaluated.'}
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--exact', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Select Tahoe train/validation treated cells and available controls.

Reads only official metadata. Test-line treated barcodes are not selected or
exported; test-line DMSO controls are allowed as prediction inputs. A fixed
hash of the barcode determines the capped cell sample, independently of any
expression value, true perturbation effect, or prediction error.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from run_e260_tahoe_metadata_panel import DEFAULT_METADATA, MAX_SAMPLED_CELLS, sha256


SEED = 'SafeConf-E261-Tahoe-cell-sampling-20260925-v1'
FIELDS = ['cell_line', 'drugname_drugconc', 'drug', 'plate', 'sample',
          'BARCODE_SUB_LIB_ID', 'pass_filter']


def priority(barcode: str) -> int:
    digest = hashlib.blake2b(f'{SEED}:{barcode}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder='big')


def push_smallest(heap: list, item: tuple, limit: int) -> None:
    # heapq is a min-heap. Negative rank makes its root the *worst* retained
    # candidate. The barcode tie-break makes the result deterministic.
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def choose(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    panel = pd.read_csv(args.panel)
    if len(panel) != 2400 or panel.cell_line.nunique() != 24 or \
       set(panel.split) != {'train', 'validation', 'test_sealed'}:
        raise ValueError('E260 locked panel inventory changed')
    treated = {(str(row.cell_line), str(row.drug_dose), str(row.plate)):
               str(row.split) for row in panel.itertuples(index=False)
               if row.split in ('train', 'validation')}
    control = {(str(row.cell_line), str(row.plate)): str(row.split)
               for row in panel.itertuples(index=False)}
    lines = sorted(panel.cell_line.astype(str).unique())
    conditions = sorted(panel.drug_dose.astype(str).unique())
    parquet = pq.ParquetFile(args.metadata)
    if not set(FIELDS).issubset(parquet.schema_arrow.names):
        raise ValueError('Tahoe metadata schema changed')
    line_set = pa.array(lines)
    condition_set = pa.array(conditions)
    heaps: dict[tuple, list] = defaultdict(list)
    scanned, candidate_count = 0, 0
    for index in range(parquet.num_row_groups):
        data = parquet.read_row_group(index, columns=FIELDS)
        scanned += data.num_rows
        allowed = pc.and_(pc.equal(data['pass_filter'], 'full'),
                          pc.is_in(data['cell_line'], value_set=line_set))
        treatment_or_control = pc.or_(
            pc.is_in(data['drugname_drugconc'], value_set=condition_set),
            pc.equal(data['drug'], 'DMSO_TF'))
        found = data.filter(pc.and_(allowed, treatment_or_control)).to_pandas()
        candidate_count += len(found)
        for line, condition, drug, plate, sample, barcode, _ in found.itertuples(
                index=False, name=None):
            line, condition, drug, plate = map(str, (line, condition, drug, plate))
            sample, barcode = str(sample), str(barcode)
            if drug == 'DMSO_TF':
                split = control.get((line, plate))
                role = 'available_control'
                key = (role, line, plate)
            else:
                split = treated.get((line, condition, plate))
                role = 'train_validation_treatment'
                key = (role, line, condition, plate)
            if split is None or not barcode or barcode == 'nan':
                continue
            if role == 'train_validation_treatment' and split == 'test_sealed':
                raise AssertionError('test treated cell entered pretruth selection')
            rank = priority(barcode)
            push_smallest(heaps[key], (-rank, barcode, sample, split),
                          MAX_SAMPLED_CELLS)
        if (index + 1) % 12 == 0:
            print(f'barcode selection row groups {index+1}/{parquet.num_row_groups}',
                  flush=True)
    records = []
    for key in sorted(heaps):
        role = key[0]
        for negative_rank, barcode, sample, split in sorted(heaps[key], reverse=True):
            records.append({'role': role, 'split': split, 'cell_line': key[1],
                            'drug_dose': key[2] if role != 'available_control' else '',
                            'plate': key[3] if role != 'available_control' else key[2],
                            'sample': sample, 'barcode': barcode,
                            'hash_priority': -negative_rank})
    selected = pd.DataFrame(records)
    actual_treated = selected.loc[selected.role.eq('train_validation_treatment')]
    if actual_treated.groupby(['cell_line', 'drug_dose', 'plate']).ngroups != 1800:
        raise ValueError('missing locked train/validation treatment task')
    if actual_treated.groupby(['cell_line', 'drug_dose', 'plate']).size().min() != MAX_SAMPLED_CELLS:
        raise ValueError('a treatment group did not supply 128 fixed-hash cells')
    actual_control = selected.loc[selected.role.eq('available_control')]
    if actual_control.groupby(['cell_line', 'plate']).ngroups != len(control) or \
       actual_control.groupby(['cell_line', 'plate']).size().min() < 64:
        raise ValueError('a locked cell-line/plate lacks available DMSO controls')
    if selected.loc[selected.split.eq('test_sealed'), 'role'].ne('available_control').any():
        raise ValueError('test treated barcode exposed')
    if selected.barcode.duplicated().any():
        raise ValueError('duplicate selected barcode')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output_dir / 'PRETRUTH_BARCODES.parquet', index=False)
    audit = {'status': 'PRETRUTH_METADATA_SELECTION_COMPLETE',
             'generated_at': datetime.now().astimezone().isoformat(),
             'panel_sha256': sha256(args.panel),
             'metadata_sha256': sha256(args.metadata),
             'sampling_seed': SEED,
             'max_cells_per_group': MAX_SAMPLED_CELLS,
             'metadata_rows_scanned': scanned,
             'metadata_candidate_rows': candidate_count,
             'selected_train_validation_treated_cells': len(actual_treated),
             'selected_available_control_cells':
                 int(selected.role.eq('available_control').sum()),
             'selected_test_treated_cells': 0,
             'selected_test_control_cells':
                 int((selected.split.eq('test_sealed') &
                      selected.role.eq('available_control')).sum()),
             'selected_barcode_file_sha256':
                 sha256(args.output_dir / 'PRETRUTH_BARCODES.parquet'),
             'target_perturbation_expression_read': 0,
             'target_perturbation_error_read': 0}
    (args.output_dir / 'STATUS.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    choose(parser.parse_args())

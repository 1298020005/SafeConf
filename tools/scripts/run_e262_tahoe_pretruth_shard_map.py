#!/usr/bin/env python3
"""Map selected Tahoe sample IDs to possible raw Parquet row groups.

Uses only Parquet footer statistics; does not read any expression data.
The min/max filter is conservative and may include unrelated samples.
"""
from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from run_e260_tahoe_metadata_panel import sha256


DEFAULT_RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M/data')


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    cells = pd.read_parquet(args.selected)
    if cells.barcode.duplicated().any() or \
       ((cells.split == 'test_sealed') &
        (cells.role != 'available_control')).any():
        raise ValueError('selection exposes test treated cells or duplicate barcode')
    samples = sorted(cells['sample'].astype(str).unique())
    if not samples:
        raise ValueError('no selected samples')
    files = sorted(args.raw_dir.glob('train-*-of-03388.parquet'))
    if len(files) != 3388:
        raise ValueError(f'expected 3388 complete Tahoe shards, found {len(files)}')
    output = []
    total_row_groups = 0
    for i, path in enumerate(files):
        if Path(str(path) + '.aria2').exists():
            raise ValueError(f'incomplete Tahoe shard: {path}')
        parquet = pq.ParquetFile(path)
        index = parquet.schema_arrow.names.index('sample')
        for row_group in range(parquet.num_row_groups):
            total_row_groups += 1
            metadata = parquet.metadata.row_group(row_group)
            stat = metadata.column(index).statistics
            if stat is None or not stat.has_min_max:
                raise ValueError(f'missing sample footer stats in {path}:{row_group}')
            first = bisect.bisect_left(samples, str(stat.min))
            if first < len(samples) and samples[first] <= str(stat.max):
                output.append({'shard': path.name,
                               'row_group': row_group,
                               'min_sample': str(stat.min),
                               'max_sample': str(stat.max),
                               'n_rows': metadata.num_rows})
        if (i + 1) % 500 == 0:
            print(f'shard footer {i+1}/{len(files)}', flush=True)
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError('no row groups match selected samples')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'CANDIDATE_ROW_GROUPS.csv', index=False)
    status = {'status': 'FOOTER_ONLY_SHARD_MAP_COMPLETE',
              'generated_at': datetime.now().astimezone().isoformat(),
              'selected_barcode_sha256': sha256(args.selected),
              'n_selected_samples': len(samples),
              'n_raw_shards': len(files),
              'n_total_row_groups': total_row_groups,
              'n_candidate_shards': frame.shard.nunique(),
              'n_candidate_row_groups': len(frame),
              'n_candidate_rows_upper_bound': int(frame.n_rows.sum()),
              'candidate_map_sha256': sha256(args.output_dir / 'CANDIDATE_ROW_GROUPS.csv'),
              'test_treatment_expression_loaded': 0,
              'expression_columns_read': False}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--output-dir', type=Path, required=True)
    build(parser.parse_args())

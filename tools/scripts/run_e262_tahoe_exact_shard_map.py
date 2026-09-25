#!/usr/bin/env python3
"""Refine E262 footer candidate row groups by reading `sample` only."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from run_e260_tahoe_metadata_panel import sha256
from run_e262_tahoe_pretruth_shard_map import DEFAULT_RAW


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    candidates = pd.read_csv(args.candidates)
    cells = pd.read_parquet(args.selected, columns=['sample'])
    samples = set(cells['sample'].astype(str))
    rows = []
    for i, (shard, group) in enumerate(candidates.groupby('shard', sort=True)):
        file = pq.ParquetFile(args.raw_dir / shard)
        for item in group.itertuples(index=False):
            values = file.read_row_group(int(item.row_group), columns=['sample'])['sample']
            present = set(pc.unique(values).to_pylist()) & samples
            if present:
                rows.append({'shard': shard, 'row_group': int(item.row_group),
                             'n_rows': int(item.n_rows),
                             'selected_samples_present': ';'.join(sorted(present))})
        if (i + 1) % 500 == 0:
            print(f'exact sample audit shards {i+1}/{candidates.shard.nunique()}',
                  flush=True)
    exact = pd.DataFrame(rows)
    if exact.empty or exact.row_group.isna().any():
        raise ValueError('no exact row group')
    found = set(';'.join(exact.selected_samples_present).split(';'))
    if found != samples:
        raise ValueError(f'selected samples absent from raw shards: {sorted(samples-found)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(args.output_dir / 'EXACT_SAMPLE_ROW_GROUPS.csv', index=False)
    status = {'status': 'EXACT_SAMPLE_MAP_NO_EXPRESSION_READ',
              'generated_at': datetime.now().astimezone().isoformat(),
              'candidate_map_sha256': sha256(args.candidates),
              'selected_barcode_sha256': sha256(args.selected),
              'n_selected_samples': len(samples),
              'n_candidate_shards': candidates.shard.nunique(),
              'n_exact_shards': exact.shard.nunique(),
              'n_candidate_row_groups': len(candidates),
              'n_exact_row_groups': len(exact),
              'n_exact_rows_upper_bound': int(exact.n_rows.sum()),
              'exact_map_sha256': sha256(args.output_dir / 'EXACT_SAMPLE_ROW_GROUPS.csv'),
              'expression_columns_read': False}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

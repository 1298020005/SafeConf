#!/usr/bin/env python3
"""Choose a Tahoe drug-dose × cell-line panel using metadata only.

No expression vector, target perturbation effect, model error, or E258 test
truth is opened. Counts are for the official `pass_filter == full` cells.
The high-coverage panel is a new locked analysis of a previously studied
Tahoe dataset, NOT an independent external study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_METADATA = Path(
    '/home/yyf/data/singlecell_perturbation_atlas/mega_external/'
    'Tahoe-100M/metadata/obs_metadata.parquet'
)
FIELDS = ['cell_line', 'drugname_drugconc', 'drug', 'plate', 'pass_filter']
PANEL_TIERS = ((24, 100, (12, 6, 6)),
               (18, 100, (9, 4, 5)),
               (12, 100, (6, 3, 3)),
               (12, 60, (6, 3, 3)))
MIN_TREATMENT_CELLS = 64
MIN_MATCHED_CONTROL_CELLS = 64
MAX_SAMPLED_CELLS = 128
SPLIT_SEED = 'SafeConf-E260-Tahoe-panel-20260925-v1'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def count_metadata(path: Path) -> tuple[pd.DataFrame, dict]:
    parquet = pq.ParquetFile(path)
    if not set(FIELDS).issubset(parquet.schema_arrow.names):
        raise ValueError('Required official Tahoe metadata fields absent')
    combined: Counter[tuple[str, str, str, str]] = Counter()
    full_cells = 0
    for index in range(parquet.num_row_groups):
        data = parquet.read_row_group(index, columns=FIELDS)
        filtered = data.filter(pc.equal(data['pass_filter'], 'full'))
        full_cells += filtered.num_rows
        grouped = filtered.group_by(FIELDS[:-1]).aggregate([('cell_line', 'count')])
        frame = grouped.to_pandas()
        for row in frame.itertuples(index=False):
            line, condition, drug, plate, n = row
            combined[str(line), str(condition), str(drug), str(plate)] += int(n)
        if (index + 1) % 12 == 0:
            print(f'metadata row groups {index+1}/{parquet.num_row_groups}', flush=True)
    rows = [(*key, count) for key, count in combined.items()]
    frame = pd.DataFrame(rows, columns=['cell_line', 'drug_dose', 'drug',
                                        'plate', 'n_full_cells'])
    frame.sort_values(['cell_line', 'drug_dose', 'plate'], inplace=True)
    audit = {'raw_metadata_rows': parquet.metadata.num_rows,
             'full_filter_cells': full_cells,
             'row_groups': parquet.num_row_groups,
             'n_context_condition_plate_groups': len(frame)}
    return frame, audit


def eligible_pairs(counts: pd.DataFrame) -> pd.DataFrame:
    control = counts.loc[counts.drug.eq('DMSO_TF')]
    control = control.groupby(['cell_line', 'plate'], as_index=False)[
        'n_full_cells'].sum().rename(columns={'n_full_cells': 'matched_control_cells'})
    treated = counts.loc[~counts.drug.eq('DMSO_TF')].merge(
        control, on=['cell_line', 'plate'], how='left', validate='many_to_one')
    treated['matched_control_cells'] = treated.matched_control_cells.fillna(0).astype(int)
    treated = treated.loc[(treated.n_full_cells >= MIN_TREATMENT_CELLS) &
                          (treated.matched_control_cells >= MIN_MATCHED_CONTROL_CELLS)]
    # If a cell line/drug-dose appears on multiple plates, select only the
    # plate with most eligible treated cells. No measured effect enters this.
    treated = treated.sort_values(['cell_line', 'drug_dose', 'n_full_cells',
                                   'matched_control_cells', 'plate'],
                                  ascending=[True, True, False, False, True])
    return treated.drop_duplicates(['cell_line', 'drug_dose']).reset_index(drop=True)


def greedy_lines(pairs: pd.DataFrame, n_lines: int) -> tuple[list[str], set[str]]:
    coverage = {line: set(group.drug_dose)
                for line, group in pairs.groupby('cell_line', sort=True)}
    candidates = {line: conditions for line, conditions in coverage.items()
                  if len(conditions) >= 100}
    if len(candidates) < n_lines:
        return [], set()
    selected: list[str] = []
    common: set[str] = set()
    while len(selected) < n_lines:
        best = sorted((line for line in candidates if line not in selected),
                      key=lambda line: (-len(candidates[line] if not selected
                                             else common & candidates[line]), line))[0]
        common = candidates[best].copy() if not selected else common & candidates[best]
        selected.append(best)
    return selected, common


def choose_panel(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    attempt = []
    for n_lines, n_conditions, split_sizes in PANEL_TIERS:
        lines, common = greedy_lines(pairs, n_lines)
        attempt.append({'n_lines': n_lines, 'requested_conditions': n_conditions,
                        'common_conditions': len(common)})
        if len(common) < n_conditions:
            continue
        selected_pairs = pairs.loc[pairs.cell_line.isin(lines) &
                                   pairs.drug_dose.isin(common)].copy()
        pivot = selected_pairs.pivot(index='drug_dose', columns='cell_line',
                                     values='n_full_cells')
        ranked = sorted(common, key=lambda condition:
                        (-int(pivot.loc[condition, lines].min()), condition))
        conditions = ranked[:n_conditions]
        panel = selected_pairs.loc[selected_pairs.drug_dose.isin(conditions)].copy()
        if len(panel) != n_lines * n_conditions:
            raise ValueError('incomplete locked panel')
        hashed = sorted(lines, key=lambda line:
                        (hashlib.sha256(f'{SPLIT_SEED}:{line}'.encode()).hexdigest(), line))
        split = dict(zip(hashed, ['train'] * split_sizes[0] +
                         ['validation'] * split_sizes[1] +
                         ['test_sealed'] * split_sizes[2]))
        panel['split'] = panel.cell_line.map(split)
        panel['maximum_cells_to_extract'] = MAX_SAMPLED_CELLS
        panel = panel.sort_values(['split', 'cell_line', 'drug_dose']).reset_index(drop=True)
        return panel, {'tier': [n_lines, n_conditions, list(split_sizes)],
                       'attempts': attempt,
                       'split_seed': SPLIT_SEED,
                       'line_splits': split,
                       'selection_uses_expression_or_model_error': False,
                       'test_perturbation_truth_loaded': 0,
                       'test_label_status': 'SEALED',
                       'interpretation': 'locked new panel within previously studied Tahoe, not independent external study'}
    raise ValueError(f'No predefined metadata-only coverage tier qualifies: {attempt}')


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, audit = count_metadata(args.metadata)
    candidates = eligible_pairs(counts)
    panel, selected = choose_panel(candidates)
    counts.to_csv(args.output_dir / 'METADATA_COUNTS.csv', index=False)
    candidates.to_csv(args.output_dir / 'ELIGIBLE_PAIRS.csv', index=False)
    panel.to_csv(args.output_dir / 'LOCKED_PANEL.csv', index=False)
    status = {'status': 'METADATA_ONLY_PANEL_LOCKED_NO_EXPRESSION_READ',
              'generated_at': datetime.now().astimezone().isoformat(),
              'metadata': str(args.metadata),
              'metadata_sha256': sha256(args.metadata),
              'rules': {'pass_filter': 'full',
                        'minimum_treatment_cells': MIN_TREATMENT_CELLS,
                        'minimum_plate_matched_DMSO_cells': MIN_MATCHED_CONTROL_CELLS,
                        'max_cells_extracted_per_condition': MAX_SAMPLED_CELLS,
                        'tiers': PANEL_TIERS,
                        'plate_selection': 'highest eligible treated-cell count, lexical tie'},
              'audit': audit,
              'n_eligible_line_drugdose_pairs': len(candidates),
              'selected': selected}
    (args.output_dir / 'STATUS.json').write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': status['status'],
                      'n_eligible_pairs': len(candidates),
                      'tier': selected['tier'],
                      'n_panel_pairs': len(panel),
                      'splits': panel.groupby('split').cell_line.nunique().to_dict()},
                     indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

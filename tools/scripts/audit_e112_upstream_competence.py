#!/usr/bin/env python3
"""Retrospective E112 capability audit against the no-change predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/yyf/proj/docs/实验结果/E112_external_formal_dual_models_20260713')
OUT = ROOT / 'docs/实验结果/E251_e112_upstream_capability_20260924'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in ('Lara_exvivo', 'Santinha'):
        base = SOURCE / dataset
        table = pd.read_csv(base / 'PREDICTION_RECORDS.csv')
        with np.load(base / 'arrays/true_effects.npz', allow_pickle=False) as truths:
            for row in table.itertuples(index=False):
                truth = np.asarray(truths[str(row.true_effect_key)], dtype=np.float64)
                no_change = float(np.sqrt(np.mean(truth ** 2)))
                records.append({'dataset': dataset, 'fold_id': str(row.fold_id),
                                'task_id': str(row.task_id), 'perturbation': str(row.perturbation),
                                'predictor': str(row.predictor_name),
                                'model_rmse': float(row.true_error_rmse),
                                'no_change_rmse': no_change})
    data = pd.DataFrame(records)
    if data[['model_rmse', 'no_change_rmse']].isna().any().any():
        raise ValueError('missing errors')
    folds = data.groupby(['dataset', 'fold_id', 'predictor'], as_index=False).agg(
        n_tasks=('task_id', 'size'), model_mean_rmse=('model_rmse', 'mean'),
        no_change_mean_rmse=('no_change_rmse', 'mean'))
    folds['relative_gain'] = 1 - folds.model_mean_rmse / folds.no_change_mean_rmse
    folds['model_beats_no_change'] = folds.relative_gain > 0
    data.to_csv(OUT / 'E112_TASK_CAPABILITY.csv', index=False)
    folds.to_csv(OUT / 'E112_FOLD_CAPABILITY.csv', index=False)
    status = {'status': 'RETROSPECTIVE_ONLY', 'n_task_model_rows': len(data),
              'n_fold_model_pairs': len(folds),
              'competent_pairs': int(folds.model_beats_no_change.sum()),
              'warning': 'E112 test truth previously public; audit cannot be blind confirmation'}
    (OUT / 'AUDIT.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

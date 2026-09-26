#!/usr/bin/env python3
"""E271: label-free fixed Bsafe formula under perturbation-cold splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_e256_biological_history_feature_ablation import utility
from run_e266_source_discrimination import MATRIX, P_FEATURES, percentile_against

STUDIES = ('CuiHacohen2023', 'LaraAstiasoHuntly2023_exvivo', 'SantinhaPlatt2023')
PREDICTORS = ('ContextSimBaseline', 'PertMeanPredictor', 'V0StrongBaseline')
BSAFE = ('perturbation_support_count', 'context_similarity_max', 'context_similarity_mean')


def bucket(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(('E271:' + str(value)).encode(), digest_size=8).digest(), 'big') % 5


def rank_against(train: pd.DataFrame, test: pd.DataFrame, col: str, negate: bool = False):
    a = -pd.to_numeric(train[col], errors='coerce') if negate else pd.to_numeric(train[col], errors='coerce')
    b = -pd.to_numeric(test[col], errors='coerce') if negate else pd.to_numeric(test[col], errors='coerce')
    return percentile_against(a, a), percentile_against(a, b)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    cols = ['dataset_name', 'fold_id', 'split', 'task_key', 'perturbation',
            'predictor_name', 'true_error_rmse'] + list(P_FEATURES) + list(BSAFE)
    raw = pd.read_csv(MATRIX, usecols=cols)
    raw = raw.loc[raw.dataset_name.isin(STUDIES) & raw.predictor_name.isin(PREDICTORS) & raw.split.eq('test')].copy()
    raw['bucket'] = raw.perturbation.map(bucket)
    rows = []
    for (dataset, predictor), task in raw.groupby(['dataset_name', 'predictor_name'], sort=True):
        for held in range(5):
            fit = task.loc[task.bucket.ne(held)]
            test = task.loc[task.bucket.eq(held)]
            if set(fit.perturbation) & set(test.perturbation):
                raise AssertionError('cold perturbation overlap')
            _, m = rank_against(fit, test, 'prediction_l2_norm')
            _, d = rank_against(fit, test, 'model_disagreement_rmse')
            b = []
            for col in BSAFE:
                _, z = rank_against(fit, test, col, negate=True)
                b.append(z)
            bmean = np.mean(np.vstack(b), axis=0)
            score_m = m
            score_fixed = m + bmean
            score_fixed_p = (m + d + bmean) / 2.0
            y = test.true_error_rmse.to_numpy(float)
            for method, score in [('M', score_m), ('M_plus_Bsafe_fixed', score_fixed),
                                  ('P_plus_Bsafe_fixed', score_fixed_p)]:
                rows.append({'dataset': dataset, 'predictor': predictor, 'held_bucket': held,
                             'method': method, 'n_test': len(test),
                             'utility20': float(utility(score, y)),
                             'spearman': float(spearmanr(score, y).statistic)})
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / 'FOLD_RESULTS.csv', index=False)
    summary = []
    for (dataset, predictor), frame in result.groupby(['dataset', 'predictor'], sort=True):
        wide_u = frame.pivot(index='held_bucket', columns='method', values='utility20')
        wide_r = frame.pivot(index='held_bucket', columns='method', values='spearman')
        for method in ('M_plus_Bsafe_fixed', 'P_plus_Bsafe_fixed'):
            summary.append({'dataset': dataset, 'predictor': predictor, 'method': method,
                            'n_buckets': len(wide_u),
                            'utility20_mean': float(wide_u[method].mean()),
                            'delta_utility20_vs_M': float((wide_u[method]-wide_u.M).mean()),
                            'positive_utility_buckets_vs_M': int((wide_u[method]>wide_u.M).sum()),
                            'spearman_mean': float(wide_r[method].mean()),
                            'delta_spearman_vs_M': float((wide_r[method]-wide_r.M).mean()),
                            'positive_spearman_buckets_vs_M': int((wide_r[method]>wide_r.M).sum())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / 'SUMMARY.csv', index=False)
    status = {'status': 'PERTURBATION_COLD_LABEL_FREE_FIXED_FORMULA',
              'formula': 'M + mean(rank(-support), rank(-context_similarity_max), rank(-context_similarity_mean)); optional current disagreement',
              'n_buckets': 5, 'studies': list(STUDIES), 'predictors': list(PREDICTORS),
              'limits': ['Existing public task table and previously public truth; development audit only.',
                         'Bsafe fields are not rebuilt from raw expression in this run.',
                         'Weights are fixed equal weights; no labels or result-based selection used.']}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary_df.to_string(index=False), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

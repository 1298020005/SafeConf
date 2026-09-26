#!/usr/bin/env python3
"""E267: perturbation-cold source audit for E266 candidates.

All rows of a perturbation are assigned to one of five deterministic buckets;
the held bucket is never represented in the risk-label training set.  This
tests whether the apparent source gains survive when the target perturbation
has no past error label in the fitted risk model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_e256_biological_history_feature_ablation import utility
from run_e266_source_discrimination import (
    B_FEATURES, E_FEATURES, MATRIX, P_FEATURES, fit_score,
    percentile_against, rank_frame, safe_error_history,
)

STUDIES = ('CuiHacohen2023', 'LaraAstiasoHuntly2023_exvivo', 'SantinhaPlatt2023')
PREDICTORS = ('ContextSimBaseline', 'PertMeanPredictor', 'V0StrongBaseline')


def bucket(value: str) -> int:
    digest = hashlib.blake2b(('E267:' + str(value)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % 5


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    cols = ['dataset_name', 'fold_id', 'split', 'task_key', 'context',
            'perturbation', 'predictor_name', 'true_error_rmse']
    cols += list(dict.fromkeys(P_FEATURES + B_FEATURES))
    raw = pd.read_csv(MATRIX, usecols=cols)
    raw = raw.loc[raw.dataset_name.isin(STUDIES) &
                  raw.predictor_name.isin(PREDICTORS) &
                  raw.split.eq('test')].copy()
    raw['bucket'] = raw.perturbation.map(bucket)
    groups = {
        'M': ('prediction_l2_norm',),
        'P': P_FEATURES,
        'P_plus_B': P_FEATURES + B_FEATURES,
        'P_plus_E': P_FEATURES + E_FEATURES,
        'P_plus_B_plus_E': P_FEATURES + B_FEATURES + E_FEATURES,
    }
    rows = []
    for (dataset, predictor), task in raw.groupby(['dataset_name', 'predictor_name'], sort=True):
        for held in range(5):
            fit = task.loc[task.bucket.ne(held)].copy()
            test = task.loc[task.bucket.eq(held)].copy()
            if len(test) < 20 or len(fit) < 100:
                raise ValueError(f'{dataset}/{predictor}/bucket{held}: insufficient rows')
            if set(fit.perturbation) & set(test.perturbation):
                raise AssertionError('perturbation overlap in cold split')
            fit[list(E_FEATURES)] = safe_error_history(fit, fit, leave_one_out=True).to_numpy()
            test[list(E_FEATURES)] = safe_error_history(fit, test, leave_one_out=False).to_numpy()
            features = tuple(dict.fromkeys(f for fs in groups.values() for f in fs))
            fit_r, test_r = rank_frame(fit, test, features)
            y = test.true_error_rmse.to_numpy(float)
            for method, fs in groups.items():
                score = (test_r.r_prediction_l2_norm.to_numpy(float) if method == 'M'
                         else fit_score(fit_r, test_r, fs, dynamic=False))
                rows.append({'dataset': dataset, 'predictor': predictor,
                             'held_perturbation_bucket': held, 'method': method,
                             'n_fit': len(fit), 'n_test': len(test),
                             'n_fit_perturbations': fit.perturbation.nunique(),
                             'n_test_perturbations': test.perturbation.nunique(),
                             'utility20': float(utility(score, y)),
                             'spearman': float(spearmanr(score, y).statistic)})
            score = fit_score(fit_r, test_r, groups['P_plus_B_plus_E'], dynamic=True)
            rows.append({'dataset': dataset, 'predictor': predictor,
                         'held_perturbation_bucket': held,
                         'method': 'P_plus_B_plus_E_dynamic', 'n_fit': len(fit),
                         'n_test': len(test), 'n_fit_perturbations': fit.perturbation.nunique(),
                         'n_test_perturbations': test.perturbation.nunique(),
                         'utility20': float(utility(score, y)),
                         'spearman': float(spearmanr(score, y).statistic)})
            print(f'{dataset}/{predictor} cold bucket={held} complete', flush=True)
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / 'FOLD_RESULTS.csv', index=False)
    summary = []
    for (dataset, predictor), frame in result.groupby(['dataset', 'predictor'], sort=True):
        wide = frame.pivot(index='held_perturbation_bucket', columns='method', values='utility20')
        rho = frame.pivot(index='held_perturbation_bucket', columns='method', values='spearman')
        for method in wide.columns:
            if method == 'M':
                continue
            summary.append({'dataset': dataset, 'predictor': predictor,
                            'method': method, 'n_buckets': len(wide),
                            'utility20_mean': float(wide[method].mean()),
                            'delta_utility20_vs_M': float((wide[method]-wide.M).mean()),
                            'positive_utility_buckets_vs_M': int((wide[method]>wide.M).sum()),
                            'spearman_mean': float(rho[method].mean()),
                            'delta_spearman_vs_M': float((rho[method]-rho.M).mean()),
                            'positive_spearman_buckets_vs_M': int((rho[method]>rho.M).sum())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / 'SUMMARY.csv', index=False)
    status = {'status': 'PERTURBATION_COLD_RETROSPECTIVE_DEVELOPMENT_ONLY',
              'studies': list(STUDIES), 'predictors': list(PREDICTORS),
              'n_buckets': 5, 'hash': 'blake2b(E267:<perturbation>) mod 5',
              'methods': list(groups) + ['P_plus_B_plus_E_dynamic'],
              'limits': ['All labels are previously public development labels.',
                         'The public B fields were created by the older fold contract; this run does not rebuild raw expression.',
                         'This tests cold perturbation transfer, not unseen cell-background transfer.',
                         'No rows/studies were removed according to result sign.'],
              'summary_rows': int(len(summary_df))}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary_df.to_string(index=False), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

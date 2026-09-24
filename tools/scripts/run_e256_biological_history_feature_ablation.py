#!/usr/bin/env python3
"""E256 fold-safe biological-history input ablation on public E1–E4 data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
OUT = ROOT / 'docs/实验结果/E256_biological_history_feature_ablation_20260924'
A = ('prediction_l2_norm', 'prediction_abs_mean', 'prediction_norm_ratio',
     'prediction_magnitude_deviation', 'model_disagreement_rmse', 'model_disagreement_cosine',
     'context_similarity_max', 'context_similarity_mean', 'perturbation_support_count',
     'ood_nearest_distance', 'ood_mean_k_distance')
B = ('historical_residual_risk',)
C = ('perturbation_effect_stability', 'perturbation_effect_variance')
GROUPS = {'A': A, 'A+B': A+B, 'A+C': A+C, 'A+B+C': A+B+C}


def utility(score: np.ndarray, errors: np.ndarray) -> float:
    n = len(score)
    k = max(1, int(np.ceil(.2*n)))
    def selected(values: np.ndarray) -> np.ndarray:
        threshold = np.partition(values, n-k)[n-k]
        higher = values > threshold
        tied = values == threshold
        weights = higher.astype(float)
        weights[tied] = (k-int(higher.sum()))/int(tied.sum())
        return weights
    average = float(np.mean(errors))
    optimum = float(np.dot(selected(errors), errors)/k-average)
    return float((np.dot(selected(score), errors)/k-average)/optimum) if optimum > 1e-12 else float('nan')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    if len(raw) != 68775 or raw.dataset_name.nunique() != 7:
        raise ValueError('E1-E4 source matrix changed')
    missing = raw.groupby('dataset_name')[list(B+C)].agg(lambda col: float(col.isna().mean()))
    missing.reset_index().to_csv(OUT / 'E256_HISTORY_MISSINGNESS.csv', index=False)
    rows = []
    for (dataset, fold), frame in raw.groupby(['dataset_name', 'fold_id'], sort=True):
        source = frame.loc[frame.predictor_name.isin(('V0StrongBaseline', 'ContextSimBaseline')) &
                           frame.split.isin(('train', 'val'))].copy()
        target = frame.loc[frame.predictor_name.eq('PertMeanPredictor') & frame.split.eq('test')].copy()
        overlap = set(source.task_key) & set(target.task_key)
        if overlap or len(source) < 50 or len(target) < 20:
            raise ValueError(f'{dataset} fold {fold}: source/target overlap or insufficient rows')
        target_ids = set(target.task_key)
        if len(target_ids) != len(target):
            raise ValueError(f'{dataset} fold {fold}: target task duplicate')
        y_train = source.true_error_rmse.to_numpy(float)
        y_test = target.true_error_rmse.to_numpy(float)
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError('missing error label')
        for group, features in GROUPS.items():
            x_train = source[list(features)].replace([np.inf, -np.inf], np.nan)
            x_test = target[list(features)].replace([np.inf, -np.inf], np.nan)
            model = HistGradientBoostingRegressor(max_iter=60, max_depth=3,
                       learning_rate=.05, min_samples_leaf=20, l2_regularization=.1,
                       early_stopping=False, random_state=256)
            model.fit(x_train, y_train)
            risk = model.predict(x_test)
            rows.append({'dataset': dataset, 'fold': int(fold), 'group': group,
                         'n_source_rows': len(source), 'n_test_tasks': len(target),
                         'source_target_task_overlap': 0,
                         'spearman': float(spearmanr(risk, y_test).statistic),
                         'utility_20': utility(risk, y_test),
                         'magnitude_utility_20': utility(target.prediction_l2_norm.to_numpy(float), y_test),
                         'magnitude_spearman': float(spearmanr(target.prediction_l2_norm, y_test).statistic)})
        print(f'{dataset} fold={fold}: four inputs evaluated', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'E256_FOLD_RESULTS.csv', index=False)
    wide_rho = results.pivot(index=['dataset','fold'], columns='group', values='spearman')
    wide_u = results.pivot(index=['dataset','fold'], columns='group', values='utility_20')
    summary = []
    for dataset in sorted(raw.dataset_name.unique()):
        r = wide_rho.loc[dataset]
        u = wide_u.loc[dataset]
        row = {'dataset': dataset, 'n_folds': len(r)}
        for label in GROUPS:
            row[f'{label}_rho'] = float(r[label].mean())
            row[f'{label}_utility20'] = float(u[label].mean())
        for metric, table in (('rho', r), ('utility20', u)):
            for name, left, right in (('C_given_A', 'A+C', 'A'),
                                      ('C_given_AB', 'A+B+C', 'A+B'),
                                      ('B_given_A', 'A+B', 'A')):
                delta = table[left]-table[right]
                row[f'delta_{name}_{metric}'] = float(delta.mean())
                row[f'positive_folds_{name}_{metric}'] = int((delta > 0).sum())
        base = results.loc[(results.dataset == dataset) & results.group.eq('A')]
        row['M_only_utility20'] = float(base.magnitude_utility_20.mean())
        row['M_only_rho'] = float(base.magnitude_spearman.mean())
        summary.append(row)
    report = pd.DataFrame(summary)
    report.to_csv(OUT / 'E256_SUMMARY.csv', index=False)
    status = {'status': 'RETROSPECTIVE_DEVELOPMENT_ONLY', 'n_datasets': len(report),
              'n_folds': len(wide_rho), 'n_training_runs': len(results),
              'all_source_target_task_overlap_zero': bool((results.source_target_task_overlap == 0).all()),
              'same_source_error_labels_for_all_groups': True,
              'historical_residual_risk_is_train_true_effect_transfer_error_not_model_error': True,
              'source_model_errors_only_as_supervised_labels': True,
              'target_truth_previously_public': True}
    (OUT / 'E256_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(report.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""E266: prediction, public biology, and past-model-error source audit.

This is a retrospective development audit on the frozen E256 task table.  It
does not open any new test truth.  For every held outer fold, the risk model
is fitted only on the other outer-fold test rows.  ``E`` features are
leave-one-row-out summaries of *past errors of the same predictor*; they are
not biological historical effects.  ``B`` features are the public biological
history/context fields already present in the frozen matrix.

The purpose is source discrimination, not selection of a winning dataset.
Every aligned study, predictor and fold is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from run_e256_biological_history_feature_ablation import utility


MATRIX = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
STUDIES = ('CuiHacohen2023', 'LaraAstiasoHuntly2023_exvivo', 'SantinhaPlatt2023')
PREDICTORS = ('ContextSimBaseline', 'PertMeanPredictor', 'V0StrongBaseline')
P_FEATURES = ('prediction_l2_norm', 'prediction_abs_mean',
              'model_disagreement_rmse')
B_FEATURES = ('perturbation_support_count',
              'perturbation_effect_stability',
              'perturbation_effect_variance',
              'context_similarity_max', 'context_similarity_mean')
# These fields do not use observed perturbation-effect vectors.  They are the
# cleanest public-history/context proxy available in the frozen matrix.
B_SAFE_FEATURES = ('perturbation_support_count',
                   'context_similarity_max', 'context_similarity_mean')
E_FEATURES = ('error_hist_pert_mean', 'error_hist_pert_std',
              'error_hist_pert_n', 'error_hist_context_mean',
              'error_hist_context_n')


def percentile_against(reference: pd.Series, values: pd.Series) -> np.ndarray:
    ref = pd.to_numeric(reference, errors='coerce').replace([np.inf, -np.inf], np.nan)
    val = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    fill = float(ref.median()) if ref.notna().any() else 0.0
    ref_np = np.sort(ref.fillna(fill).to_numpy(float))
    val_np = val.fillna(fill).to_numpy(float)
    # Right-sided empirical CDF; no target labels enter this transform.
    return np.searchsorted(ref_np, val_np, side='right') / max(1, len(ref_np))


def safe_error_history(fit: pd.DataFrame, query: pd.DataFrame,
                       leave_one_out: bool) -> pd.DataFrame:
    """Build model-error history using only fit labels.

    For fitting rows, remove that row from its aggregate.  For held rows,
    use all fit rows.  Sparse groups fall back to the fit-wide mean/std and
    expose the support count, so the model cannot mistake absence for a
    measured low error.
    """
    y = pd.to_numeric(fit.true_error_rmse, errors='coerce').to_numpy(float)
    global_mean = float(np.nanmean(y))
    global_std = float(np.nanstd(y))
    if not np.isfinite(global_std):
        global_std = 0.0

    def aggregate(keys: pd.Series) -> dict[str, tuple[float, float, int]]:
        temp = pd.DataFrame({'key': keys.astype(str).to_numpy(), 'y': y})
        out = {}
        for key, group in temp.groupby('key', sort=False):
            vals = group.y.to_numpy(float)
            out[str(key)] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                             int(len(vals)))
        return out

    pert = aggregate(fit.perturbation)
    context = aggregate(fit.context)
    rows = []
    for idx, row in query.iterrows():
        p = pert.get(str(row.perturbation), (global_mean, global_std, 0))
        c = context.get(str(row.context), (global_mean, global_std, 0))
        if leave_one_out:
            # The query row is a member of fit.  Remove exactly its own label
            # from its perturbation/context group before exposing the feature.
            yy = float(row.true_error_rmse)
            for key, group in ((str(row.perturbation), pert), (str(row.context), context)):
                m, s, n = group.get(key, (global_mean, global_std, 0))
                if n > 1:
                    # Recover sum/sumsq from mean/std, then remove this row.
                    ss = float((n - 1) * (s ** 2) + n * (m ** 2))
                    new_n = n - 1
                    new_m = (n * m - yy) / new_n
                    new_ss = max(0.0, ss - yy * yy)
                    new_s = float(np.sqrt(max(0.0, new_ss / new_n - new_m * new_m)))
                    value = (new_m, new_s, new_n)
                else:
                    value = (global_mean, global_std, 0)
                if key == str(row.perturbation):
                    p = value
                else:
                    c = value
        rows.append({'error_hist_pert_mean': p[0], 'error_hist_pert_std': p[1],
                     'error_hist_pert_n': p[2], 'error_hist_context_mean': c[0],
                     'error_hist_context_n': c[2]})
    return pd.DataFrame(rows, index=query.index)


def rank_frame(train: pd.DataFrame, test: pd.DataFrame,
               features: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    for col in features:
        tr[f'r_{col}'] = percentile_against(tr[col], tr[col])
        te[f'r_{col}'] = percentile_against(tr[col], te[col])
    tr['r_y'] = percentile_against(tr.true_error_rmse, tr.true_error_rmse)
    return tr, te


def fit_score(train: pd.DataFrame, test: pd.DataFrame,
              features: tuple[str, ...], dynamic: bool = False) -> np.ndarray:
    xtr = train[[f'r_{c}' for c in features]].to_numpy(float)
    xte = test[[f'r_{c}' for c in features]].to_numpy(float)
    if dynamic:
        model = make_pipeline(SimpleImputer(strategy='median'),
                               HistGradientBoostingRegressor(max_iter=60,
                               max_depth=3, learning_rate=.05,
                               min_samples_leaf=20, l2_regularization=.1,
                               early_stopping=False, random_state=266))
    else:
        model = make_pipeline(SimpleImputer(strategy='median'), Ridge(alpha=10.0))
    model.fit(xtr, train.r_y.to_numpy(float))
    return model.predict(xte)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    use = ['dataset_name', 'fold_id', 'split', 'task_key', 'context',
           'perturbation', 'predictor_name', 'true_error_rmse']
    use += list(dict.fromkeys(P_FEATURES + B_FEATURES))
    raw = pd.read_csv(MATRIX, usecols=use)
    raw = raw.loc[raw.dataset_name.isin(STUDIES) &
                  raw.predictor_name.isin(PREDICTORS)].copy()
    rows, summary = [], []
    groups = {
        'M': ('prediction_l2_norm',),
        'P': P_FEATURES,
        'P_plus_B': P_FEATURES + B_FEATURES,
        'P_plus_Bsafe': P_FEATURES + B_SAFE_FEATURES,
        'P_plus_E': P_FEATURES + E_FEATURES,
        'P_plus_B_plus_E': P_FEATURES + B_FEATURES + E_FEATURES,
        'P_plus_Bsafe_plus_E': P_FEATURES + B_SAFE_FEATURES + E_FEATURES,
    }
    for (dataset, predictor), all_rows in raw.groupby(['dataset_name', 'predictor_name'], sort=True):
        tasks = all_rows.loc[all_rows.split.eq('test')].copy()
        if tasks.task_key.duplicated().any() or tasks.fold_id.nunique() != 5:
            raise ValueError(f'{dataset}/{predictor}: task/fold contract changed')
        for held in sorted(tasks.fold_id.unique()):
            fit = tasks.loc[~tasks.fold_id.eq(held)].copy()
            test = tasks.loc[tasks.fold_id.eq(held)].copy()
            fit_e = safe_error_history(fit, fit, leave_one_out=True)
            test_e = safe_error_history(fit, test, leave_one_out=False)
            fit[list(E_FEATURES)] = fit_e.to_numpy()
            test[list(E_FEATURES)] = test_e.to_numpy()
            all_features = tuple(dict.fromkeys(f for fs in groups.values() for f in fs))
            fit_r, test_r = rank_frame(fit, test, all_features)
            y = test.true_error_rmse.to_numpy(float)
            for method, features in groups.items():
                if method == 'M':
                    score = test_r.r_prediction_l2_norm.to_numpy(float)
                else:
                    score = fit_score(fit_r, test_r, features,
                                      dynamic=(method == 'P_plus_B_plus_E_dynamic'))
                rows.append({'dataset': dataset, 'predictor': predictor,
                             'held_fold': int(held), 'method': method,
                             'n_fit': len(fit), 'n_test': len(test),
                             'utility20': float(utility(score, y)),
                             'spearman': float(spearmanr(score, y).statistic)})
            # Full dynamic model is deliberately one fixed extra comparator,
            # not a hyperparameter search.
            score = fit_score(fit_r, test_r, groups['P_plus_B_plus_E'], dynamic=True)
            rows.append({'dataset': dataset, 'predictor': predictor,
                         'held_fold': int(held), 'method': 'P_plus_B_plus_E_dynamic',
                         'n_fit': len(fit), 'n_test': len(test),
                         'utility20': float(utility(score, y)),
                         'spearman': float(spearmanr(score, y).statistic)})
            score = fit_score(fit_r, test_r, groups['P_plus_Bsafe_plus_E'], dynamic=True)
            rows.append({'dataset': dataset, 'predictor': predictor,
                         'held_fold': int(held), 'method': 'P_plus_Bsafe_plus_E_dynamic',
                         'n_fit': len(fit), 'n_test': len(test),
                         'utility20': float(utility(score, y)),
                         'spearman': float(spearmanr(score, y).statistic)})
            print(f'{dataset}/{predictor} fold={held} complete', flush=True)
        print(f'{dataset}/{predictor} complete', flush=True)
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / 'FOLD_RESULTS.csv', index=False)
    wide_u = result.pivot_table(index=['dataset', 'predictor', 'held_fold'],
                                columns='method', values='utility20')
    wide_r = result.pivot_table(index=['dataset', 'predictor', 'held_fold'],
                                columns='method', values='spearman')
    for (dataset, predictor), _ in wide_u.groupby(level=[0, 1]):
        u = wide_u.loc[(dataset, predictor)]
        r = wide_r.loc[(dataset, predictor)]
        base = u['M']
        for method in u.columns:
            if method == 'M':
                continue
            summary.append({'dataset': dataset, 'predictor': predictor,
                            'method': method, 'n_folds': len(u),
                            'utility20_mean': float(u[method].mean()),
                            'delta_utility20_vs_M': float((u[method] - base).mean()),
                            'positive_utility_folds_vs_M': int((u[method] > base).sum()),
                            'spearman_mean': float(r[method].mean()),
                            'delta_spearman_vs_M': float((r[method] - r['M']).mean()),
                            'positive_spearman_folds_vs_M': int((r[method] > r['M']).sum())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / 'SUMMARY.csv', index=False)
    status = {
        'status': 'RETROSPECTIVE_SOURCE_DISCRIMINATION_DEVELOPMENT_ONLY',
        'studies': list(STUDIES), 'predictors': list(PREDICTORS),
        'outer_folds': 5, 'task_source': str(MATRIX),
        'methods': list(groups) + ['P_plus_B_plus_E_dynamic', 'P_plus_Bsafe_plus_E_dynamic'],
        'P_definition': list(P_FEATURES),
        'B_definition': list(B_FEATURES),
        'B_safe_definition': list(B_SAFE_FEATURES),
        'E_definition': list(E_FEATURES),
        'E_contract': 'same-predictor true errors from other outer folds; leave-one-row-out for fit rows',
        'limits': [
            'All labels were previously public development labels; this is not independent confirmation.',
            'B fields are existing biological/context history features and are not newly reconstructed here.',
            'E tests availability of past model-error labels; it is not available in a cold start with no labels.',
            'No study or predictor was removed according to the result sign.',
            'No upstream perturbation predictor or graph model was trained.'
        ], 'summary_rows': int(len(summary_df))
    }
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary_df.to_string(index=False), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Lara ex vivo E256: source-only split-half Q and historical increment.

Reconstructs the original first-5000-gene, min-six-cell, cap-2200-cell,
seed-5201 effect task contract. Original effect vectors and all source H
features must match archived E256 values before any risk comparison is made.
This is retrospective development on already-public errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from run_e259_cui_fold_label_budget import rank_within_folds
from run_e256_biological_history_feature_ablation import utility
from run_e264_seven_study_basic_quality import (
    MATRIX, PREDICTORS, RAW, build_test_table, raw_pair_cells,
)


DATASET = 'LaraAstiasoHuntly2023_exvivo'
INPUT = Path('/home/yyf/safeconf_runtime/outputs/safeconf_formal_main_v3_drop_blank_inputs_20260609/LaraAstiasoHuntly2023_exvivo/input')
GROUPS = {
    'Ridge_M': ('prediction_l2_norm',),
    'Ridge_M_Qbasic': ('prediction_l2_norm', 'perturbation_support_count',
                       'median_source_cells'),
    'Ridge_M_Qsplit': ('prediction_l2_norm', 'perturbation_support_count',
                       'median_source_cells_capped', 'source_split_noise_variance'),
    'Ridge_M_Qsplit_Hraw': ('prediction_l2_norm', 'perturbation_support_count',
                            'median_source_cells_capped', 'source_split_noise_variance',
                            'perturbation_effect_variance'),
    'Ridge_M_Qsplit_Hbio': ('prediction_l2_norm', 'perturbation_support_count',
                            'median_source_cells_capped', 'source_split_noise_variance',
                            'H_bio_approx_variance'),
}


def raw_effect_halves() -> tuple[dict, dict]:
    obj = ad.read_h5ad(RAW / 'LaraAstiasoHuntly2023_exvivo.h5ad', backed='r')
    try:
        if obj.n_vars < 5000 or obj.layers.keys() or \
           any(str(col).startswith('highly_variable') for col in obj.var.columns):
            raise ValueError('original fixed 5000-gene view cannot be reconstructed')
        obs = obj.obs[['celltype', 'perturbation']].copy()
        obs['context'] = obs.celltype.map(lambda value: '' if pd.isna(value)
                                         else str(value).strip())
        obs['pert'] = obs.perturbation.map(lambda value: '' if pd.isna(value)
                                         else str(value).strip())
        rng = np.random.default_rng(5201)
        groups = {}
        for (context, pert), sub in obs.groupby(['context', 'pert'], observed=False):
            if not context or pert.lower() in ('', 'nan', 'none', 'null'):
                continue
            positions = obs.index.get_indexer(sub.index)
            if len(positions) < 6:
                continue
            if len(positions) > 2200:
                positions = rng.choice(positions, size=2200, replace=False)
            groups[context, pert] = positions
        matrix = obj.X[:, :5000]
        matrix = sp.csr_matrix(matrix).astype(np.float32)
        means = {}
        for key, positions in groups.items():
            first = np.asarray(matrix[positions[::2]].mean(axis=0),
                               np.float32).ravel()
            second = np.asarray(matrix[positions[1::2]].mean(axis=0),
                                np.float32).ravel()
            full = np.asarray(matrix[positions].mean(axis=0), np.float32).ravel()
            means[key] = (full, first, second, len(positions))
        controls = {context: key for key in means
                    for context, pert in [key]
                    if pert.strip().lower() in ('control', 'ctrl', 'non-targeting',
                                                 'nt', 'mock', 'vehicle', 'dmso') or
                    pert.strip().lower().startswith(('ctrl', 'control'))}
        effects = {}
        for (context, pert), (full, first, second, n) in means.items():
            if context not in controls or (context, pert) == controls[context]:
                continue
            base = means[controls[context]]
            effects[context, pert] = (full - base[0], first - base[1],
                                      second - base[2], n)
        return effects, {'n_raw_cells': obj.n_obs,
                         'n_effect_pairs': len(effects),
                         'n_genes': 5000,
                         'preprocessing': 'original X, first 5000 genes, min6, cap2200, seed5201'}
    finally:
        obj.file.close()


def verify_archived(effects: dict) -> float:
    records = pd.read_csv(INPUT / 'PREDICTION_RECORDS.csv',
                          usecols=['context', 'perturbation', 'true_effect_key'])
    records = records.drop_duplicates(['context', 'perturbation']).sort_values(
        ['context', 'perturbation']).head(30)
    diffs = []
    with np.load(INPUT / 'true_effects.npz', allow_pickle=False) as original:
        for row in records.itertuples(index=False):
            key = (str(row.context), str(row.perturbation))
            if key not in effects:
                raise ValueError(f'archived task absent in source rebuild: {key}')
            diffs.append(float(np.max(np.abs(effects[key][0] -
                                             original[str(row.true_effect_key)]))))
    maximum = max(diffs)
    if maximum > 2e-4:
        raise ValueError(f'archived effect mismatch: {maximum}')
    return maximum


def source_features(effects: dict, test: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    one = test.loc[test.predictor_name.eq(PREDICTORS[0])]
    dataset_matrix = matrix.loc[matrix.dataset_name.eq(DATASET) &
                                matrix.predictor_name.eq(PREDICTORS[0])]
    output = []
    for fold, part in dataset_matrix.groupby('fold_id', sort=True):
        source = part.loc[part.split.eq('train')].groupby('perturbation').context.agg(
            lambda values: sorted(set(values.astype(str))))
        targets = one.loc[one.fold_id.eq(fold)]
        for row in targets.itertuples(index=False):
            contexts = source.loc[str(row.perturbation)]
            if row.context in contexts or len(contexts) < 2:
                raise ValueError('held context included in historical source')
            values = np.stack([effects[context, row.perturbation][0]
                               for context in contexts])
            first = np.stack([effects[context, row.perturbation][1]
                              for context in contexts])
            second = np.stack([effects[context, row.perturbation][2]
                               for context in contexts])
            observed = values.var(axis=0)
            sampling = ((first - second) ** 2).mean(axis=0) / 4
            biological = np.maximum(observed -
                                    (1 - 1 / len(contexts)) * sampling, 0)
            output.append({'fold_id': fold, 'task_key': row.task_key,
                           'source_split_noise_variance': float(sampling.mean()),
                           'H_bio_approx_variance': float(biological.mean()),
                           'H_raw_reconstructed': float(observed.mean()),
                           'median_source_cells_capped': float(np.median(
                               [effects[context, row.perturbation][3]
                                for context in contexts]))})
    frame = pd.DataFrame(output)
    if len(frame) != one.shape[0] or frame[['fold_id', 'task_key']].duplicated().any():
        raise ValueError('source feature task inventory changed')
    return frame


def evaluate(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = tuple(dict.fromkeys(feature for features in GROUPS.values()
                                  for feature in features))
    scored = []
    for predictor, part in test.groupby('predictor_name', sort=True):
        ranked = rank_within_folds(part, columns)
        for held in sorted(part.fold_id.unique()):
            train = ranked.loc[~ranked.fold_id.eq(held)]
            target = ranked.loc[ranked.fold_id.eq(held)]
            for method, features in GROUPS.items():
                cols = [f'rank_{feature}' for feature in features]
                model = Ridge(alpha=10)
                model.fit(train[cols], train.rank_true_error_rmse)
                prediction = model.predict(target[cols])
                y = target.true_error_rmse.to_numpy(float)
                scored.append({'predictor': predictor, 'fold_id': int(held),
                               'method': method, 'n_tasks': len(target),
                               'utility20': float(utility(prediction, y)),
                               'spearman': float(spearmanr(prediction, y).statistic)})
    result = pd.DataFrame(scored)
    pivot = result.pivot(index=['predictor', 'fold_id'], columns='method',
                          values='utility20')
    summary = {}
    for predictor, part in pivot.groupby(level=0):
        delta_q = part.Ridge_M_Qsplit - part.Ridge_M
        delta_h_raw = part.Ridge_M_Qsplit_Hraw - part.Ridge_M_Qsplit
        delta_h_bio = part.Ridge_M_Qsplit_Hbio - part.Ridge_M_Qsplit
        summary[predictor] = {
            'n_tasks': int(test.predictor_name.eq(predictor).sum()),
            'M_utility20': float(part.Ridge_M.mean()),
            'MQbasic_utility20': float(part.Ridge_M_Qbasic.mean()),
            'MQsplit_utility20': float(part.Ridge_M_Qsplit.mean()),
            'MQsplit_Hraw_utility20': float(part.Ridge_M_Qsplit_Hraw.mean()),
            'MQsplit_Hbio_utility20': float(part.Ridge_M_Qsplit_Hbio.mean()),
            'delta_Qsplit_vs_M': float(delta_q.mean()),
            'delta_Hraw_given_Qsplit': float(delta_h_raw.mean()),
            'delta_Hbio_given_Qsplit': float(delta_h_bio.mean()),
            'Q_positive_folds': int((delta_q > 0).sum()),
            'Hraw_positive_folds': int((delta_h_raw > 0).sum()),
            'Hbio_positive_folds': int((delta_h_bio > 0).sum()),
        }
    return result, summary


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    effects, raw_audit = raw_effect_halves()
    raw_audit['max_archived_effect_difference'] = verify_archived(effects)
    matrix = pd.read_csv(MATRIX, usecols=['dataset_name', 'fold_id', 'split',
                                         'task_key', 'context', 'perturbation',
                                         'predictor_name', 'prediction_l2_norm',
                                         'perturbation_support_count',
                                         'perturbation_effect_variance',
                                         'true_error_rmse'])
    cells = raw_pair_cells(RAW / 'LaraAstiasoHuntly2023_exvivo.h5ad', 'celltype')
    test = build_test_table(matrix, cells, DATASET)
    half = source_features(effects, test, matrix)
    test = test.merge(half, on=['fold_id', 'task_key'], how='left', validate='many_to_one')
    if test[['H_bio_approx_variance', 'source_split_noise_variance']].isna().any().any():
        raise ValueError('split-half source feature missing')
    difference = float(np.max(np.abs(test.perturbation_effect_variance -
                                     test.H_raw_reconstructed)))
    if difference > 1e-5:
        raise ValueError(f'historical variance not exactly E256-aligned: {difference}')
    result, summary = evaluate(test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    half.to_csv(args.output_dir / 'SOURCE_SPLIT_HALF_FEATURES.csv', index=False)
    result.to_csv(args.output_dir / 'FOLD_RISK_RESULTS.csv', index=False)
    status = {'status': 'RETROSPECTIVE_FULL_Q_SPLIT_HALF_AUDIT',
              'dataset': DATASET, 'n_tasks': half.shape[0],
              'n_predictors': len(PREDICTORS), 'n_folds': 5,
              'raw_audit': raw_audit,
              'max_abs_historical_variance_difference': difference,
              'same_error_label_budget': 'four public outer test folds fit, fifth evaluates',
              'results': summary,
              'limits': ['Split-half is a sampling/measurement proxy, not pure technical noise.',
                         'Both this study and its outcomes were already public.',
                         'Five folds from one study are not five independent external studies.']}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

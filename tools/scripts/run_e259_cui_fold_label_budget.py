#!/usr/bin/env python3
"""Retrospective Cui M+measurement quality versus +history, all predictors.

Risk-label training rotates over four already-public test folds and evaluates
the fifth. Source cell counts come from official h5ad metadata and only
outer-fold train pairs. Optional split-half noise features come from a
separately audited, source-only expression table. This is development, not
independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from run_e256_biological_history_feature_ablation import utility


ROOT = Path(__file__).resolve().parents[2]
MATRIX = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/CuiHacohen2023.h5ad')
PREDICTORS = ('ContextSimBaseline', 'PertMeanPredictor', 'V0StrongBaseline')
GROUPS = {
    'M': ('prediction_l2_norm',),
    'M_Q': ('prediction_l2_norm', 'perturbation_support_count',
            'median_source_cells'),
    'M_Q_Hraw': ('prediction_l2_norm', 'perturbation_support_count',
                 'median_source_cells', 'perturbation_effect_variance'),
}
SPLIT_GROUPS = {
    'M_Q_split': ('prediction_l2_norm', 'perturbation_support_count',
                   'median_source_cells', 'source_split_noise_variance'),
    'M_Q_split_Hraw': ('prediction_l2_norm', 'perturbation_support_count',
                        'median_source_cells', 'source_split_noise_variance',
                        'perturbation_effect_variance'),
    'M_Q_split_Hbio': ('prediction_l2_norm', 'perturbation_support_count',
                        'median_source_cells', 'source_split_noise_variance',
                        'H_bio_approx_variance'),
}


def source_counts() -> dict[tuple[str, str], int]:
    obj = ad.read_h5ad(RAW, backed='r')
    try:
        grouped = obj.obs.groupby(['celltype', 'perturbation'], observed=True).size()
        return {(str(context), str(pert)): int(n)
                for (context, pert), n in grouped.items()}
    finally:
        obj.file.close()


def rank_within_folds(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns + ('true_error_rmse',):
        out[f'rank_{col}'] = out.groupby('fold_id')[col].transform(
            lambda v: (rankdata(v.to_numpy(float), method='average') - .5) / len(v))
    return out


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = pd.read_csv(MATRIX, usecols=['dataset_name', 'fold_id', 'split',
                                      'task_key', 'context', 'perturbation',
                                      'predictor_name', 'prediction_l2_norm',
                                      'perturbation_support_count',
                                      'perturbation_effect_variance',
                                      'true_error_rmse', 'true_effect_l2_norm'])
    raw = raw.loc[raw.dataset_name.eq('CuiHacohen2023') &
                  raw.predictor_name.isin(PREDICTORS)].copy()
    if len(raw) != 5 * 1253 * 3:
        raise ValueError('Cui five-fold/three-predictor matrix changed')
    pair_cells = source_counts()
    rows = []
    for (predictor, fold), part in raw.groupby(['predictor_name', 'fold_id'], sort=True):
        train = part.loc[part.split.eq('train')]
        test = part.loc[part.split.eq('test')].copy()
        if test.task_key.duplicated().any():
            raise ValueError('duplicate fold target task')
        source = train.groupby('perturbation', observed=True).agg(
            source_contexts=('context', lambda x: sorted(set(x.astype(str))))
        )
        nsource, median_cells = [], []
        for row in test.itertuples(index=False):
            contexts = source.loc[row.perturbation, 'source_contexts']
            if row.context in contexts:
                raise ValueError('test context in train-source pair')
            counts = [pair_cells[context, row.perturbation] for context in contexts]
            nsource.append(len(contexts))
            median_cells.append(float(np.median(counts)))
        test['median_source_cells'] = median_cells
        if not np.array_equal(test.perturbation_support_count.to_numpy(int),
                              np.asarray(nsource)):
            raise ValueError('source count mismatches frozen E256 feature')
        rows.append(test)
    tasks = pd.concat(rows, ignore_index=True)
    groups = dict(GROUPS)
    if args.split_half is not None:
        half = pd.read_csv(args.split_half)
        tasks = tasks.merge(half, on=['fold_id', 'task_key'], how='left',
                            validate='many_to_one')
        if len(tasks) != 1253 * 3 or tasks[
                ['source_split_noise_variance', 'H_bio_approx_variance']].isna().any().any():
            raise ValueError('Cui split-half feature alignment failed')
        if np.max(np.abs(tasks.perturbation_effect_variance -
                         tasks.historical_variance_reconstructed)) > 1e-5:
            raise ValueError('Cui historical variance did not reconstruct E256')
        if np.max(np.abs(tasks.median_source_cells -
                         tasks.median_source_cells_reconstructed)) > 1:
            raise ValueError('Cui source-cell support did not reconstruct old input')
        groups.update(SPLIT_GROUPS)
    if tasks[['prediction_l2_norm', 'perturbation_effect_variance',
              'median_source_cells', 'true_error_rmse']].isna().any().any():
        raise ValueError('missing Cui feature/label')
    if any(tasks.loc[tasks.predictor_name.eq(p)].task_key.duplicated().any()
           for p in PREDICTORS):
        raise ValueError('same task appears in multiple outer test folds')
    columns = tuple(dict.fromkeys(x for group in groups.values() for x in group))
    scored = []
    competence = []
    for predictor, group in tasks.groupby('predictor_name', sort=True):
        ranked = rank_within_folds(group, columns)
        for held in sorted(group.fold_id.unique()):
            train = ranked.loc[~ranked.fold_id.eq(held)]
            test = ranked.loc[ranked.fold_id.eq(held)]
            if len(train) < 1000 or len(test) < 200:
                raise ValueError('Cui fold label-budget task count changed')
            y = test.true_error_rmse.to_numpy(float)
            nochange = test.true_effect_l2_norm.to_numpy(float) / np.sqrt(5000)
            competence.append({'predictor': predictor, 'fold': int(held),
                               'n_tasks': len(test), 'model_rmse': float(y.mean()),
                               'nochange_rmse': float(nochange.mean()),
                               'beats_nochange': bool(y.mean() < nochange.mean())})
            for method, features in groups.items():
                if method == 'M':
                    risk = test.rank_prediction_l2_norm.to_numpy(float)
                else:
                    cols = [f'rank_{feature}' for feature in features]
                    model = Ridge(alpha=10)
                    model.fit(train[cols].to_numpy(float),
                              train.rank_true_error_rmse.to_numpy(float))
                    risk = model.predict(test[cols].to_numpy(float))
                scored.append({'predictor': predictor, 'fold': int(held),
                               'method': method, 'n_tasks': len(test),
                               'utility20': utility(risk, y),
                               'spearman': float(spearmanr(risk, y).statistic)})
    result = pd.DataFrame(scored)
    ability = pd.DataFrame(competence)
    if len(result) != len(groups) * 15 or len(ability) != 15:
        raise ValueError('incomplete Cui methods/folds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    ability.to_csv(args.output.with_suffix('.competence.csv'), index=False)
    pivot = result.pivot(index=['predictor', 'fold'], columns='method', values='utility20')
    summary = []
    for predictor, part in pivot.groupby(level=0):
        delta = part.M_Q_Hraw - part.M_Q
        summary.append({'predictor': predictor, 'n_folds': len(part),
                        'n_tasks': int(ability.loc[ability.predictor.eq(predictor),
                                                    'n_tasks'].sum()),
                        'M_utility20': float(part.M.mean()),
                        'M_Q_utility20': float(part.M_Q.mean()),
                        'M_Q_Hraw_utility20': float(part.M_Q_Hraw.mean()),
                        'delta_H_given_MQ_utility20': float(delta.mean()),
                        'positive_folds_H_given_MQ': int((delta > 0).sum()),
                        'M_Q_split_utility20': float(part.M_Q_split.mean()) if args.split_half else None,
                        'M_Q_split_Hraw_utility20': float(part.M_Q_split_Hraw.mean()) if args.split_half else None,
                        'M_Q_split_Hbio_utility20': float(part.M_Q_split_Hbio.mean()) if args.split_half else None,
                        'delta_Hraw_given_MQsplit_utility20':
                            float((part.M_Q_split_Hraw - part.M_Q_split).mean()) if args.split_half else None,
                        'delta_Hbio_given_MQsplit_utility20':
                            float((part.M_Q_split_Hbio - part.M_Q_split).mean()) if args.split_half else None,
                        'positive_folds_Hbio_given_MQsplit':
                            int((part.M_Q_split_Hbio > part.M_Q_split).sum()) if args.split_half else None,
                        'upstream_winning_folds_vs_nochange': int(ability.loc[
                            ability.predictor.eq(predictor), 'beats_nochange'].sum())})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output.with_suffix('.summary.csv'), index=False)
    status = {'status': 'RETROSPECTIVE_DEVELOPMENT_ONLY',
              'dataset': 'CuiHacohen2023',
              'n_predictors': len(PREDICTORS), 'n_outer_folds': 5,
              'source_cell_counts_from_raw_metadata_only': True,
              'source_context_count_matches_E256': True,
              'split_half_measurement_variation_available': args.split_half is not None,
              'split_half_feature_file': str(args.split_half) if args.split_half else None,
              'all_predictors_reported': list(PREDICTORS),
              'same_risk_label_budget': 'four public outer test folds train Ridge, fifth evaluates',
              'same_error_object': 'task-level predictor RMSE on fixed 5000-gene axis',
              'limits': [
                  'Cui was chosen after E256 results were already known; not an independent confirmation.',
                  'Cell counts come from raw h5ad pair metadata; the optional split-half feature reconstruction audits the original fixed gene axis.',
                  ('Split-half Q is an approximate sampling/measurement proxy, not pure technical noise.'
                   if args.split_half else
                   'Q lacks split-half sampling variation, so this is not full M+measurement-quality comparison.'),
                  'Five folds are one study, not five independent datasets.'
              ], 'summary': summary}
    args.output.with_suffix('.status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary_frame.to_string(index=False), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-half', type=Path)
    run(parser.parse_args())

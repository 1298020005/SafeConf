#!/usr/bin/env python3
"""Retrospective seven-study basic-Q eligibility and same-budget risk audit.

Do not interpret historical H gain here as biological gain: this stage has
source counts but not split-half measurement noise. Studies with mismatched
raw/task contracts are reported as ineligible, not silently subsetted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from run_e259_cui_fold_label_budget import rank_within_folds
from run_e256_biological_history_feature_ablation import utility


MATRIX = Path('/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv')
RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb')
PREDICTORS = ('ContextSimBaseline', 'PertMeanPredictor', 'V0StrongBaseline')
STUDIES = (
    ('CuiHacohen2023', 'CuiHacohen2023.h5ad', 'celltype'),
    ('Frangieh', 'FrangiehIzar2021_RNA.h5ad', None),
    ('LaraAstiasoHuntly2023_exvivo', 'LaraAstiasoHuntly2023_exvivo.h5ad', 'celltype'),
    ('LaraAstiasoHuntly2023_invivo', 'LaraAstiasoHuntly2023_invivo.h5ad', 'celltype'),
    ('McFarlandTsherniak2020', 'McFarlandTsherniak2020.h5ad', None),
    ('SantinhaPlatt2023', 'SantinhaPlatt2023.h5ad', 'cell_types'),
    ('SrivatsanTrapnell2020_sciplex3', 'SrivatsanTrapnell2020_sciplex3.h5ad', None),
)
GROUPS = {
    'Ridge_M': ('prediction_l2_norm',),
    'Ridge_M_Qbasic': ('prediction_l2_norm', 'perturbation_support_count',
                       'median_source_cells'),
    'Ridge_M_Qbasic_Hraw': ('prediction_l2_norm', 'perturbation_support_count',
                            'median_source_cells', 'perturbation_effect_variance'),
}


def raw_pair_cells(path: Path, context_col: str) -> dict[tuple[str, str], int]:
    obj = ad.read_h5ad(path, backed='r')
    try:
        if context_col not in obj.obs or 'perturbation' not in obj.obs:
            raise ValueError(f'missing raw context or perturbation: {context_col}')
        frame = obj.obs[[context_col, 'perturbation']].astype(str)
        counts = frame.groupby([context_col, 'perturbation'], observed=True).size()
        return {(str(context), str(pert)): int(n)
                for (context, pert), n in counts.items()}
    finally:
        obj.file.close()


def build_test_table(matrix: pd.DataFrame, cells: dict,
                     dataset: str) -> pd.DataFrame:
    part = matrix.loc[matrix.dataset_name.eq(dataset) &
                      matrix.predictor_name.isin(PREDICTORS)]
    if part.empty or part.fold_id.nunique() != 5:
        raise ValueError('five-fold E256 task contract absent')
    output = []
    for (predictor, fold), section in part.groupby(['predictor_name', 'fold_id'],
                                                    sort=True):
        train = section.loc[section.split.eq('train')]
        test = section.loc[section.split.eq('test')].copy()
        # The frozen E256 contract excludes the target context from the
        # historical source pool.  Other folds can contain the same context
        # because this is an outer-fold retrospective audit; do not count it
        # as an independent historical source for the target row.
        source = train.groupby('perturbation').context.agg(
            lambda v: sorted(set(v.astype(str))))
        count, median = [], []
        for row in test.itertuples(index=False):
            contexts = [c for c in source.loc[str(row.perturbation)]
                        if c != str(row.context)]
            if len(contexts) < 2:
                raise ValueError('fewer than two independent source contexts after target-context exclusion')
            keys = [(context, str(row.perturbation)) for context in contexts]
            absent = [key for key in keys if key not in cells]
            if absent:
                raise ValueError(f'{len(absent)} source context/perturbation pairs absent from raw; example {absent[0]}')
            count.append(len(contexts))
            median.append(float(np.median([cells[key] for key in keys])))
        if not np.array_equal(test.perturbation_support_count.to_numpy(int),
                              np.asarray(count)):
            raise ValueError('historical source count mismatches E256 frozen feature')
        test['median_source_cells'] = median
        output.append(test)
    result = pd.concat(output, ignore_index=True)
    if result[['prediction_l2_norm', 'true_error_rmse', 'median_source_cells',
               'perturbation_effect_variance']].isna().any().any():
        raise ValueError('required E256 risk feature or label missing')
    for predictor, group in result.groupby('predictor_name'):
        if group.task_key.duplicated().any():
            raise ValueError(f'target task repeated across held folds: {predictor}')
    return result


def evaluate(tasks: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict]:
    cols = tuple(dict.fromkeys(feature for group in GROUPS.values()
                               for feature in group))
    scores = []
    for predictor, part in tasks.groupby('predictor_name', sort=True):
        ranked = rank_within_folds(part, cols)
        for held in sorted(part.fold_id.unique()):
            train = ranked.loc[~ranked.fold_id.eq(held)]
            test = ranked.loc[ranked.fold_id.eq(held)]
            if len(train) < 150 or len(test) < 25:
                raise ValueError('too few risk labels in one E256 fold')
            truth = test.true_error_rmse.to_numpy(float)
            for method, group in GROUPS.items():
                columns = [f'rank_{feature}' for feature in group]
                model = Ridge(alpha=10.0)
                model.fit(train[columns].to_numpy(float),
                          train.rank_true_error_rmse.to_numpy(float))
                score = model.predict(test[columns].to_numpy(float))
                scores.append({'dataset': dataset, 'predictor': predictor,
                               'fold_id': int(held), 'method': method,
                               'n_tasks': len(test),
                               'utility20': float(utility(score, truth)),
                               'spearman': float(spearmanr(score, truth).statistic)})
    results = pd.DataFrame(scores)
    if len(results) != 5 * len(PREDICTORS) * len(GROUPS):
        raise ValueError('incomplete fixed-budget method matrix')
    pivot = results.pivot(index=['predictor', 'fold_id'], columns='method',
                          values='utility20')
    summary = {}
    for predictor, frame in pivot.groupby(level=0):
        dq = frame.Ridge_M_Qbasic - frame.Ridge_M
        dh = frame.Ridge_M_Qbasic_Hraw - frame.Ridge_M_Qbasic
        summary[predictor] = {
            'n_tasks': int(tasks.loc[tasks.predictor_name.eq(predictor)].shape[0]),
            'delta_Qbasic_utility20_mean': float(dq.mean()),
            'delta_Hraw_given_Qbasic_utility20_mean': float(dh.mean()),
            'Q_positive_folds': int((dq > 0).sum()),
            'H_positive_folds': int((dh > 0).sum()),
            'M_utility20_mean': float(frame.Ridge_M.mean()),
            'MQ_utility20_mean': float(frame.Ridge_M_Qbasic.mean()),
            'MQH_utility20_mean': float(frame.Ridge_M_Qbasic_Hraw.mean()),
        }
    return results, summary


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    matrix = pd.read_csv(MATRIX, usecols=['dataset_name', 'fold_id', 'split',
                                         'task_key', 'context', 'perturbation',
                                         'predictor_name', 'prediction_l2_norm',
                                         'perturbation_support_count',
                                         'perturbation_effect_variance',
                                         'true_error_rmse'])
    inventory, result_frames = [], []
    for dataset, filename, field in STUDIES:
        if field is None:
            inventory.append({'dataset': dataset, 'status': 'NOT_ELIGIBLE_YET',
                              'reason': 'context/dose/time mapping must be audited before raw-cell Q'})
            continue
        try:
            cells = raw_pair_cells(RAW / filename, field)
            tasks = build_test_table(matrix, cells, dataset)
            result, summary = evaluate(tasks, dataset)
        except Exception as error:
            inventory.append({'dataset': dataset, 'status': 'CONTRACT_FAILED',
                              'reason': f'{type(error).__name__}: {error}'})
            print(f'{dataset}: CONTRACT_FAILED {type(error).__name__}: {error}', flush=True)
            continue
        result_frames.append(result)
        inventory.append({'dataset': dataset, 'status': 'RETROSPECTIVE_QBASIC_DONE',
                          'n_tasks_per_predictor': int(tasks.shape[0] / len(PREDICTORS)),
                          'n_folds': 5, 'predictor_results': summary})
        print(f'{dataset}: Qbasic complete, {len(tasks)} predictor-task rows', flush=True)
    if not result_frames:
        raise ValueError('no study had an aligned raw-cell Qbasic contract')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(result_frames, ignore_index=True).to_csv(
        args.output_dir / 'FOLD_RESULTS.csv', index=False)
    status = {'status': 'RETROSPECTIVE_BASIC_Q_ONLY_NOT_FINAL_BIOLOGICAL_HISTORY',
              'same_risk_error_label_budget':
                  'Ridge(alpha=10), train on four public outer test folds, evaluate fifth',
              'risk_scale': 'within-fold feature percentiles',
              'utility_definition': 'oracle-adjusted utility from E256',
              'n_prespecified_studies': len(STUDIES),
              'n_qbasic_aligned_studies': len(result_frames),
              'results': inventory,
              'limits': ['Qbasic lacks source split-half sampling variation.',
                         'All E256 target truths were previously public; these are development results.',
                         'Raw available cell counts may exceed the archived task-view cap and require a contract audit before definitive Q claims.',
                         'No study or predictor removed based on the sign of the risk result.']}
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'aligned_studies': status['n_qbasic_aligned_studies'],
                      'statuses': [(r['dataset'], r['status']) for r in inventory]},
                     ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

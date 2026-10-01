#!/usr/bin/env python3
"""Guide-block reproducibility diagnostic on already-seen common-axis truth.

This does not redefine the primary RMSE, train a predictor, select a method,
or manufacture a fresh confirmation. Guide disagreement is a measurement /
biological-response proxy, not an unbiased estimate of independent cell noise.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import metrics, SEEDS
from tools.scripts.prepare_safeconf_common_gene_predictions import h5column
from tools.scripts import run_safeconf_research_closure as closure

COMMON = closure.RUNTIME / 'common_gene_axis'
OUT = closure.OUT / 'common_gene_axis' / 'truth_reproducibility'
H5 = Path('/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad')
SPLIT = H5.parent / 'splits/mcfaline23_gxe_splits/full_covariate_split.csv'


def rho(a, b):
    return float(spearmanr(a, b).statistic) if np.ptp(a) and np.ptp(b) else np.nan


def cell_sampling_diagnostic():
    """Fix sampling size and inspect independent within-guide cell halves.

    Twenty cells is the largest round count below every query's observed size.
    Five hash orders are all reported. These retrospective diagnoses neither
    replace the primary truth nor permit choosing another risk algorithm.
    """
    out = OUT / 'cell_sampling'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'STATUS.json').exists():
        raise FileExistsError('completed cell-sampling diagnosis is immutable')
    tasks = pd.read_csv(COMMON / 'TEST_TASKS.csv')
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    lookup = {t.task_id: i for i, t in enumerate(tasks.itertuples())}
    n_task, width = len(tasks), len(genes)
    counts = np.zeros((2, n_task), int)
    halves = np.zeros((2, n_task, width), np.float64)
    fixed = np.zeros((5, n_task, width), np.float64)
    all_sums = np.zeros((n_task, width), np.float64)
    with h5py.File(H5, 'r') as f:
        ids = h5column(f['obs'], '_index').astype(str)
        roles = pd.read_csv(SPLIT, header=None, names=['cell', 'role']).set_index('cell').role.reindex(ids).to_numpy()
        selected = np.flatnonzero((roles == 'test') & (h5column(f['obs'], 'control').astype(int) == 0))
        context = h5column(f['obs'], 'cell_type').astype(str)
        treatment = h5column(f['obs'], 'treatment').astype(str)
        target = h5column(f['obs'], 'perturbation').astype(str)
        guide = h5column(f['obs'], 'gRNA_id').astype(str)
        task_rows = np.asarray([lookup[target[r] + '::' + context[r] + '::' + treatment[r]] for r in selected])
        meta = pd.DataFrame({'task': task_rows, 'guide': guide[selected]})
        half = np.zeros(len(selected), int)
        for _, positions in meta.groupby(['task', 'guide'], sort=True).groups.items():
            ordered = sorted(positions, key=lambda j: hashlib.sha256(
                f'SafeConf-cell-halves-v1\0{ids[selected[j]]}'.encode()).hexdigest())
            half[ordered[1::2]] = 1
        chosen = np.zeros((5, len(selected)), bool)
        for order in range(5):
            for _, positions in meta.groupby('task', sort=True).groups.items():
                if len(positions) < 20:
                    raise RuntimeError('predefined 20-cell diagnosis lacks full task coverage')
                ordered = sorted(positions, key=lambda j: hashlib.sha256(
                    f'SafeConf-cell-equal20-v1\0{order}\0{ids[selected[j]]}'.encode()).hexdigest())
                chosen[order, ordered[:20]] = True
        axis = {g: i for i, g in enumerate(h5column(f['var'], 'gene_name').astype(str))}
        projection = np.full(len(axis), -1, int)
        for j, g in enumerate(genes):
            projection[axis[g]] = j
        x = f['X']; pointer = x['indptr'][:]
        for begin in range(0, len(selected), 512):
            chunk = selected[begin:begin + 512]
            start, stop = int(pointer[chunk[0]]), int(pointer[chunk[-1] + 1])
            columns = x['indices'][start:stop]; values = x['data'][start:stop]
            for pos, row in enumerate(chunk, begin):
                a, b = int(pointer[row]) - start, int(pointer[row + 1]) - start
                dest = projection[columns[a:b]]; keep = dest >= 0
                dest, value = dest[keep], values[a:b][keep]
                t = task_rows[pos]; h = half[pos]
                all_sums[t, dest] += value
                halves[h, t, dest] += value; counts[h, t] += 1
                for order in np.flatnonzero(chosen[:, pos]):
                    fixed[order, t, dest] += value
            if (begin // 512) % 20 == 0:
                print(f'Cell-sampling diagnostic: {min(begin + 512, len(selected))}/{len(selected)}', flush=True)
    control = np.load(COMMON / 'TEST_CONTROLS.npy')
    full = all_sums / counts.sum(axis=0)[:, None] - control
    if not np.allclose(full, np.load(COMMON / 'TEST_TRUE_EFFECTS.npy'), atol=2e-6):
        raise RuntimeError('cell-sampling sums differ from primary registered truth')
    halves = halves / counts[:, :, None] - control[None, :, :]
    fixed = fixed / 20 - control[None, :, :]
    prediction = np.load(COMMON / 'TEST_CALIBRATED_EFFECTS.npy')
    cross = np.mean((prediction - halves[0]) * (prediction - halves[1]), axis=1)
    frame = pd.DataFrame({'task_id': tasks.task_id, 'gene': tasks.gene, 'target': tasks.context,
                          'true_error_rmse': np.sqrt(np.mean((prediction - full) ** 2, axis=1)),
                          'n_test_cells': counts.sum(axis=0),
                          'cell_half_a_error': np.sqrt(np.mean((prediction - halves[0]) ** 2, axis=1)),
                          'cell_half_b_error': np.sqrt(np.mean((prediction - halves[1]) ** 2, axis=1)),
                          'cross_half_squared_error': cross})
    for order in range(5):
        frame[f'equal20_error_{order}'] = np.sqrt(np.mean((prediction - fixed[order]) ** 2, axis=1))
    manual = pd.read_parquet(COMMON / 'risk_cache/external_Manual.parquet').set_index('task_id')
    frame['log_history_support'] = frame.task_id.map(manual.log_history_support)
    associations, risk_results = [], []
    predictions = pd.read_csv(closure.OUT / 'common_gene_axis/results/MATRIX_TASK_PREDICTIONS.csv.gz')
    support = pd.read_csv(closure.OUT / 'common_gene_axis/results/SUPPORT_CONTROL_TASK_PREDICTIONS.csv.gz')
    predictions = pd.concat([predictions, support], ignore_index=True)
    predictions = predictions[predictions.line.eq('TxPert_to_McFaline') & predictions.seed.eq(SEEDS[0])]
    truth_columns = ['true_error_rmse', 'cell_half_a_error', 'cell_half_b_error',
                     'cross_half_squared_error', *[f'equal20_error_{i}' for i in range(5)]]
    for context, part in frame.groupby('target', sort=True):
        for column in truth_columns:
            associations.append({'context': context, 'diagnostic_truth': column,
                'rho_with_history_support': rho(part.log_history_support, part[column]),
                'rho_with_test_cell_count': rho(part.n_test_cells, part[column]),
                'rho_with_primary_error': rho(part.true_error_rmse, part[column]), 'n_tasks': len(part)})
        associations.append({'context': context, 'diagnostic_truth': 'independent_cell_halves_agreement',
            'rho_with_history_support': np.nan, 'rho_with_test_cell_count': np.nan,
            'rho_with_primary_error': rho(part.cell_half_a_error, part.cell_half_b_error), 'n_tasks': len(part)})
    for method in ['Learned_WeightedHistoryDistance', 'Learned_hgb', 'Magnitude', 'NegativeHistorySupport']:
        risk = predictions[predictions.method.eq(method)].set_index('task_id').risk
        for column in truth_columns:
            for context, part in frame.groupby('target', sort=True):
                if column == 'cross_half_squared_error':
                    result = {'spearman': rho(part.task_id.map(risk), part[column])}
                else:
                    evaluation = part.copy(); evaluation['true_error_rmse'] = part[column]
                    result = metrics(evaluation, part.task_id.map(risk).to_numpy())
                risk_results.append({'method': method, 'context': context, 'diagnostic_truth': column, **result})
    closure.tx.atomic_csv(out / 'CELL_SAMPLING_TASK_DIAGNOSTICS.csv', frame)
    closure.tx.atomic_csv(out / 'CELL_SAMPLING_ASSOCIATIONS.csv', pd.DataFrame(associations))
    closure.tx.atomic_csv(out / 'CELL_SAMPLING_RISK_DIAGNOSTICS.csv', pd.DataFrame(risk_results))
    closure.tx.atomic_json(out / 'STATUS.json', {'status': 'COMPLETE', 'role': 'SEEN_DIAGNOSTIC',
        'primary_error_unchanged': True, 'no_model_refit_or_selection': True, 'n_tasks': n_task,
        'equal_cell_count': 20, 'all_five_hash_orders_reported': True,
        'n_negative_cross_half_squared_errors': int((cross < 0).sum()),
        'interpretation': 'sampling-size sensitivity; cross-half diagnostic assumes independent within-guide cell sampling and treats observed controls as fixed; not a new primary endpoint'})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'STATUS.json').exists():
        raise FileExistsError('completed diagnostics remain immutable')
    tasks = pd.read_csv(COMMON / 'TEST_TASKS.csv')
    genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
    prediction = np.load(COMMON / 'TEST_CALIBRATED_EFFECTS.npy')
    control = np.load(COMMON / 'TEST_CONTROLS.npy')
    full_truth = np.load(COMMON / 'TEST_TRUE_EFFECTS.npy')
    task_lookup = {t.task_id: i for i, t in enumerate(tasks.itertuples())}
    with h5py.File(H5, 'r') as f:
        ids = h5column(f['obs'], '_index')
        roles = pd.read_csv(SPLIT, header=None, names=['cell', 'role']).set_index('cell').role.reindex(ids).to_numpy()
        selected = np.flatnonzero((roles == 'test') & (h5column(f['obs'], 'control').astype(int) == 0))
        context = h5column(f['obs'], 'cell_type').astype(str)
        treatment = h5column(f['obs'], 'treatment').astype(str)
        target = h5column(f['obs'], 'perturbation').astype(str)
        guide = h5column(f['obs'], 'gRNA_id').astype(str)
        keys = np.asarray([target[r] + '::' + context[r] + '::' + treatment[r] for r in selected])
        row_task = np.asarray([task_lookup[k] for k in keys])
        grouping = pd.DataFrame({'task': row_task, 'guide': guide[selected]})
        pairs = list(grouping.groupby(['task', 'guide'], sort=True).groups)
        group_lookup = {k: j for j, k in enumerate(pairs)}
        group_idx = np.asarray([group_lookup[(int(t), g)] for t, g in zip(row_task, guide[selected])])
        counts = np.bincount(group_idx, minlength=len(pairs))
        sums = np.zeros((len(pairs), len(genes)), np.float64)
        axis = {g: i for i, g in enumerate(h5column(f['var'], 'gene_name').astype(str))}
        projection = np.full(len(axis), -1, int)
        for j, g in enumerate(genes):
            projection[axis[g]] = j
        x = f['X']; pointer = x['indptr'][:]
        # Bounded CSR spans can contain other rows; only selected TEST treated
        # rows contribute to the sums. No row outside selected is aggregated.
        for begin in range(0, len(selected), 512):
            chunk = selected[begin:begin + 512]
            data_begin, data_end = int(pointer[chunk[0]]), int(pointer[chunk[-1] + 1])
            columns = x['indices'][data_begin:data_end]
            values = x['data'][data_begin:data_end]
            for pos, row in enumerate(chunk, begin):
                a, b = int(pointer[row]) - data_begin, int(pointer[row + 1]) - data_begin
                dest = projection[columns[a:b]]; valid = dest >= 0
                sums[group_idx[pos], dest[valid]] += values[a:b][valid]
            if (begin // 512) % 20 == 0:
                print(f'Guide diagnostic: {min(begin + 512, len(selected))}/{len(selected)} cells', flush=True)
    means = sums / counts[:, None]
    rows, reconstructed, half_a, half_b = [], [], [], []
    for index, task in enumerate(tasks.itertuples()):
        units = [j for j, (t, _) in enumerate(pairs) if t == index]
        if len(units) < 2:
            raise RuntimeError('guide-block diagnostic needs two genuinely distinct guides')
        ordered = sorted(units, key=lambda j: hashlib.sha256(
            f'SafeConf-guide-diagnostic-v1\0{task.task_id}\0{pairs[j][1]}'.encode()).hexdigest())
        split = (len(ordered) + 1) // 2
        a, b = ordered[:split], ordered[split:]
        total = sums[units].sum(axis=0) / counts[units].sum() - control[index]
        ta = sums[a].sum(axis=0) / counts[a].sum() - control[index]
        tb = sums[b].sum(axis=0) / counts[b].sum() - control[index]
        weights = counts[units] / counts[units].sum()
        disagreement = np.sqrt(np.mean(np.sum(weights[:, None] *
            (means[units] - (total + control[index])) ** 2, axis=0)))
        reconstructed.append(total); half_a.append(ta); half_b.append(tb)
        rows.append({'task_id': task.task_id, 'gene': task.gene, 'target': task.context,
                     'treatment': task.treatment, 'n_test_cells': int(counts[units].sum()),
                     'n_distinct_guides': len(units), 'effective_guides': float(1 / np.sum(weights ** 2)),
                     'guide_disagreement': float(disagreement),
                     'guide_split_effect_rmse': float(np.sqrt(np.mean((ta - tb) ** 2))),
                     'true_error_rmse': float(np.sqrt(np.mean((prediction[index] - total) ** 2))),
                     'guide_half_a_error': float(np.sqrt(np.mean((prediction[index] - ta) ** 2))),
                     'guide_half_b_error': float(np.sqrt(np.mean((prediction[index] - tb) ** 2)))})
    if not np.allclose(reconstructed, full_truth, atol=2e-6):
        raise RuntimeError('guide sums do not reproduce the registered full-cell task truth')
    frame = pd.DataFrame(rows)
    manual = pd.read_parquet(COMMON / 'risk_cache/external_Manual.parquet').set_index('task_id')
    frame['log_history_support'] = frame.task_id.map(manual.log_history_support)
    associations = []
    for context, part in frame.groupby('target', sort=True):
        for x, y in [('guide_half_a_error', 'guide_half_b_error'),
                     ('log_history_support', 'true_error_rmse'),
                     ('log_history_support', 'guide_split_effect_rmse'),
                     ('guide_disagreement', 'true_error_rmse'),
                     ('n_test_cells', 'true_error_rmse')]:
            associations.append({'context': context, 'x': x, 'y': y,
                                 'spearman': rho(part[x], part[y]), 'n_tasks': len(part)})
    scores = pd.read_csv(closure.OUT / 'common_gene_axis/results/MATRIX_TASK_PREDICTIONS.csv.gz')
    scores = scores[scores.line.eq('TxPert_to_McFaline') & scores.seed.eq(SEEDS[0])]
    summaries = []
    for method in ['Learned_WeightedHistoryDistance', 'Manual_WeightedHistoryDistance', 'Learned_hgb', 'Magnitude']:
        lookup = scores[scores.method.eq(method)].set_index('task_id').risk
        for label in ['true_error_rmse', 'guide_half_a_error', 'guide_half_b_error']:
            for context, part in frame.groupby('target', sort=True):
                evaluation = part.copy(); evaluation['true_error_rmse'] = evaluation[label]
                result = metrics(evaluation, part.task_id.map(lookup).to_numpy())
                summaries.append({'method': method, 'diagnostic_truth': label, 'context': context, **result})
    closure.tx.atomic_csv(OUT / 'GUIDE_TASK_DIAGNOSTICS.csv', frame)
    closure.tx.atomic_csv(OUT / 'GUIDE_ASSOCIATIONS.csv', pd.DataFrame(associations))
    closure.tx.atomic_csv(OUT / 'GUIDE_SPLIT_RISK_METRICS.csv', pd.DataFrame(summaries))
    closure.tx.atomic_json(OUT / 'STATUS.json', {
        'status': 'COMPLETE', 'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC',
        'n_tasks': len(tasks), 'n_genes': len(genes), 'n_test_treated_cells': len(selected),
        'n_task_guide_groups': len(pairs), 'registered_full_truth_reproduced': True,
        'no_risk_model_fit': True, 'no_primary_error_redefinition': True,
        'guide_disagreement_is_unbiased_noise_estimate': False,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-sampling', action='store_true')
    args = parser.parse_args()
    cell_sampling_diagnostic() if args.cell_sampling else main()

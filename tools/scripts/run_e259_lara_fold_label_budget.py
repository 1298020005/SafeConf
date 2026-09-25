#!/usr/bin/env python3
"""Retrospective Lara ex-vivo fixed-budget Q/H check on existing E253 outputs.

No new graph model is trained. Existing GEARS predictions passed the E251
no-change gate in all five folds. Source cell counts come from the E99 train
manifest only. Optional audited source-only split halves add an approximate
sampling-noise feature. E112/E253 truth was already public, so this cannot
confirm a new paper claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
E253 = ROOT / 'docs/实验结果/E253_lara_history_transfer_20260924/E253_ALL_120_TASKS.csv'
E99 = Path('/home/yyf/proj/docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')
GROUPS = {'Ridge_M': ('M',),
          'Ridge_M_Q': ('M', 'n_source_contexts', 'median_source_cells'),
          'Ridge_M_Q_Hraw': ('M', 'n_source_contexts', 'median_source_cells', 'H')}
SPLIT_GROUPS = {
    'Ridge_M_Q_split': ('M', 'n_source_contexts', 'median_source_cells',
                        'source_split_noise'),
    'Ridge_M_Q_split_Hraw': ('M', 'n_source_contexts', 'median_source_cells',
                             'source_split_noise', 'H'),
    'Ridge_M_Q_split_Hbio': ('M', 'n_source_contexts', 'median_source_cells',
                             'source_split_noise', 'H_bio_approx'),
}


def utility(score: np.ndarray, error: np.ndarray) -> float:
    n = len(score)
    k = max(1, int(np.ceil(.2 * n)))
    def weights(values: np.ndarray) -> np.ndarray:
        cut = np.partition(values, n-k)[n-k]
        above = values > cut
        tied = values == cut
        result = above.astype(float)
        result[tied] = (k - int(above.sum())) / int(tied.sum())
        return result
    s, oracle = weights(score), weights(error)
    mean = float(error.mean())
    gain = float(np.dot(oracle, error) / k - mean)
    return float((np.dot(s, error) / k - mean) / gain) if gain > 1e-12 else float('nan')


def percentiles(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    ranked = frame.copy()
    for col in columns + ('absolute_error',):
        ranked[f'rank_{col}'] = ranked.groupby('fold_id')[col].transform(
            lambda x: (rankdata(x.to_numpy(float), method='average') - .5) / len(x))
    return ranked


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    task = pd.read_csv(E253)
    manifest = pd.read_csv(E99)
    manifest = manifest.loc[manifest.dataset.eq('Lara_exvivo') &
                            manifest.split.eq('train') &
                            manifest.in_train_fraction_100.astype(bool)]
    counts = manifest.groupby(['fold_id', 'perturbation'], observed=True).agg(
        n_source_contexts=('context', 'nunique'),
        median_source_cells=('n_cells', 'median'),
        min_source_cells=('n_cells', 'min')).reset_index()
    task = task.merge(counts, on=['fold_id', 'perturbation'],
                      how='left', validate='many_to_one', suffixes=('', '_manifest'))
    if len(task) != 116 or task.fold_id.nunique() != 5 or \
       task.isna().any().any() or \
       not (task.n_train_source_contexts == task.n_source_contexts).all():
        raise ValueError('E253/E99 train-source count contract changed')
    groups = dict(GROUPS)
    if args.split_half is not None:
        half = pd.read_csv(args.split_half)
        task = task.merge(half, on=['fold_id', 'task_id', 'n_train_source_contexts'],
                          how='left', validate='one_to_one')
        if len(task) != 116 or task[['source_split_noise', 'H_bio_approx']].isna().any().any():
            raise ValueError('source split-half features not aligned')
        maximum = float(np.max(np.abs(task.H_raw_split_reconstructed - task.H)))
        if maximum > 2e-3:
            raise ValueError(f'split-half history effect mismatch: {maximum}')
        groups.update(SPLIT_GROUPS)
    capability = json.loads((ROOT / 'docs/实验结果/E253_lara_history_transfer_20260924/E253_STATUS.json').read_text())
    if not capability['all_folds_GEARS_beats_no_change']:
        raise ValueError('upstream no-change gate failed')
    cols = tuple(dict.fromkeys(x for group in groups.values() for x in group))
    ranked = percentiles(task, cols)
    rows = []
    for held in sorted(task.fold_id.unique()):
        train = ranked.loc[~ranked.fold_id.eq(held)]
        test = ranked.loc[ranked.fold_id.eq(held)]
        if len(test) < 20 or len(train) < 80:
            raise ValueError('fold task count changed')
        label = train.rank_absolute_error.to_numpy(float)
        scores = {'M_unsupervised': test.rank_M.to_numpy(float)}
        for name, group in groups.items():
            xcols = [f'rank_{col}' for col in group]
            model = Ridge(alpha=10)
            model.fit(train[xcols].to_numpy(float), label)
            scores[name] = model.predict(test[xcols].to_numpy(float))
        for name, score in scores.items():
            error = test.absolute_error.to_numpy(float)
            rows.append({'fold_id': held, 'method': name, 'n_tasks': len(test),
                         'utility20': utility(score, error),
                         'spearman': float(spearmanr(score, error).statistic),
                         'upstream_beats_no_change':
                             bool(error.mean() < test.no_change_error.mean())})
    result = pd.DataFrame(rows)
    wide = result.pivot(index='fold_id', columns='method', values='utility20')
    comparisons = [('Ridge_M_Q_Hraw', 'Ridge_M_Q')]
    if args.split_half is not None:
        comparisons += [('Ridge_M_Q_split_Hraw', 'Ridge_M_Q_split'),
                        ('Ridge_M_Q_split_Hbio', 'Ridge_M_Q_split')]
    deltas = {f'{name}_vs_{base}': {
                  'mean_delta_utility20': float((wide[name] - wide[base]).mean()),
                  'positive_folds': int((wide[name] > wide[base]).sum()),
                  'per_fold_delta': (wide[name] - wide[base]).to_dict()}
              for name, base in comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {'status': 'RETROSPECTIVE_DEVELOPMENT_ONLY',
               'study': 'LaraAstiasoHuntly2023_exvivo',
               'upstream': 'existing E112 GEARS; NO NEW GRAPH TRAINING',
               'same_risk_label_budget': True,
               'risk_fit': 'leave-one-E112-background-fold-out Ridge(alpha=10)',
               'source_cell_count_from': 'E99 manifest train pairs only',
               'split_half_Q_available': args.split_half is not None,
               'full_measurement_Q_test': args.split_half is not None,
               'split_half_feature_file': str(args.split_half) if args.split_half else None,
               'n_tasks': len(task), 'n_folds': task.fold_id.nunique(),
               'metrics': result.groupby('method')[['utility20', 'spearman']].mean().to_dict('index'),
               'pairwise': deltas,
               'limits': [
                   'All E112/E253 target errors were previously public: this is not a blind validation.',
                   'Five folds share perturbation identities and are not five independent studies.',
                   ('Split-half Q is an approximate sampling/measurement proxy, not pure technical noise.'
                    if args.split_half else
                    'Q has source cell counts but no split-half sampling variation.'),
                   'Existing upstream is a graph predictor; this run does not train or create a graph model.',
               ]}
    args.output.with_suffix('.status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'metrics': summary['metrics'], 'pairwise': deltas}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-half', type=Path)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Retrospective same-budget error-label efficiency on E201 (not a blind test)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/yyf/proj/docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables/E201_TASK_METRICS.csv')
OUT = ROOT / 'docs/实验结果/E252_error_label_efficiency_20260924'
FEATURES = ('predicted_magnitude', 'family_disagreement', 'model_source_gap',
            'source_delta_dispersion', 'negative_log_source_cells', 'support_context_deficit')
BUDGETS = (.01, .05, .10, .20, .50, 1.0)
N_SEEDS = 16


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def membership(values: np.ndarray, fraction: float) -> np.ndarray:
    n = len(values)
    k = max(1, int(np.ceil(n * fraction)))
    threshold = np.partition(values, n-k)[n-k]
    above = values > threshold
    tied = values == threshold
    weights = above.astype(float)
    weights[tied] = (k - int(above.sum())) / int(tied.sum())
    return weights


def utility(values: np.ndarray, errors: np.ndarray) -> float:
    chosen = membership(values, .20)
    oracle = membership(errors, .20)
    average = float(np.mean(errors))
    chosen_error = float(np.dot(chosen, errors) / chosen.sum())
    oracle_error = float(np.dot(oracle, errors) / oracle.sum())
    return (chosen_error-average)/(oracle_error-average) if oracle_error-average > 1e-12 else float('nan')


def rho(values: np.ndarray, errors: np.ndarray) -> float:
    if len(np.unique(values)) < 2 or len(np.unique(errors)) < 2:
        return float('nan')
    return float(np.corrcoef(rankdata(values), rankdata(errors))[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ['task_id', 'target', 'gene', 'analysis_stratum', 'family_rms_error', *FEATURES]
    data = pd.read_csv(SOURCE, usecols=columns)
    data = data.loc[data.analysis_stratum.eq('primary_ge30')].copy()
    if len(data) != 1808 or data.target.nunique() != 4:
        raise ValueError('E201 primary task inventory changed')
    if data[columns].isna().any().any() or data.task_id.duplicated().any():
        raise ValueError('missing or duplicated task')
    output = []
    splits = []
    coefficients = []
    for target, part in data.groupby('target', sort=True):
        part = part.reset_index(drop=True)
        genes = sorted(part.gene.astype(str).unique(),
                       key=lambda gene: hashlib.sha256(f'E252:eval:{target}:{gene}'.encode()).hexdigest())
        n_eval_genes = int(np.ceil(.2 * len(genes)))
        eval_genes = set(genes[:n_eval_genes])
        calibration_pool = np.asarray(genes[n_eval_genes:], dtype=str)
        is_eval = part.gene.isin(eval_genes).to_numpy()
        if is_eval.sum() < 30 or not (set(part.loc[is_eval, 'gene']) & set(part.loc[~is_eval, 'gene']) == set()):
            raise ValueError(f'{target}: invalid gene split')
        for gene in genes:
            splits.append({'target': target, 'gene': gene, 'split': 'evaluation' if gene in eval_genes else 'calibration_pool'})
        x = np.stack([rankdata(part[name].to_numpy(float)) / len(part) for name in FEATURES], axis=1)
        y = part.family_rms_error.to_numpy(float)
        y_eval = y[is_eval]
        baseline_scores = {'M': x[:, 0], 'D': x[:, 1],
                           'M_plus_D_fixed': .8*x[:, 0]+.2*x[:, 1],
                           'M_plus_H_fixed': .8*x[:, 0]+.2*x[:, 3]}
        base_result = {name: (utility(score[is_eval], y_eval), rho(score[is_eval], y_eval))
                       for name, score in baseline_scores.items()}
        for seed_index in range(N_SEEDS):
            rng = np.random.default_rng(20260924 + seed_index)
            ordering = rng.permutation(calibration_pool)
            for budget in BUDGETS:
                n_cal = max(1, int(np.ceil(budget*len(calibration_pool))))
                selected_genes = set(ordering[:n_cal])
                train_mask = part.gene.isin(selected_genes).to_numpy()
                if np.any(train_mask & is_eval):
                    raise ValueError('calibration/evaluation gene overlap')
                for method, (value, corr) in base_result.items():
                    output.append({'target': target, 'seed_index': seed_index, 'budget': budget,
                                   'n_calibration_genes': n_cal, 'n_evaluation_tasks': int(is_eval.sum()),
                                   'method': method, 'utility_20': value, 'spearman': corr})
                for method, usecols in [('ridge_no_history', [0, 1]),
                                        ('ridge_with_history', [0, 1, 2, 3, 4, 5])]:
                    model = Ridge(alpha=10.0, positive=True, solver='lbfgs')
                    model.fit(x[train_mask][:, usecols], y[train_mask])
                    risk = model.predict(x[is_eval][:, usecols])
                    output.append({'target': target, 'seed_index': seed_index, 'budget': budget,
                                   'n_calibration_genes': n_cal, 'n_evaluation_tasks': int(is_eval.sum()),
                                   'method': method, 'utility_20': utility(risk, y_eval),
                                   'spearman': rho(risk, y_eval)})
                    coefficients.append({'target': target, 'seed_index': seed_index, 'budget': budget,
                                         'method': method, 'n_calibration_genes': n_cal,
                                         'intercept': float(model.intercept_),
                                         **{feature: float(model.coef_[index]) if index in usecols else 0.0
                                            for index, feature in enumerate(FEATURES)}})
    scores = pd.DataFrame(output)
    scores.to_csv(OUT / 'E252_ALL_RUNS.csv', index=False)
    pd.DataFrame(splits).to_csv(OUT / 'E252_GENE_SPLIT.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'E252_COEFFICIENTS.csv', index=False)
    piv = scores.pivot(index=['target', 'seed_index', 'budget'], columns='method', values='utility_20').reset_index()
    piv['history_minus_no_history'] = piv.ridge_with_history - piv.ridge_no_history
    piv['history_minus_M'] = piv.ridge_with_history - piv.M
    piv['history_minus_best_fixed'] = piv.ridge_with_history - piv[['M', 'D', 'M_plus_D_fixed', 'M_plus_H_fixed']].max(axis=1)
    piv.to_csv(OUT / 'E252_PAIRED.csv', index=False)
    summary = piv.groupby('budget', as_index=False).agg(
        n_target_seed_pairs=('history_minus_no_history', 'size'),
        mean_delta_history_vs_no_history=('history_minus_no_history', 'mean'),
        positive_pairs_history_vs_no_history=('history_minus_no_history', lambda x: int((x > 0).sum())),
        mean_delta_history_vs_M=('history_minus_M', 'mean'),
        mean_delta_history_vs_best_fixed=('history_minus_best_fixed', 'mean'),
        positive_pairs_vs_best_fixed=('history_minus_best_fixed', lambda x: int((x > 0).sum())),
        mean_utility_M=('M', 'mean'), mean_utility_no_history=('ridge_no_history', 'mean'),
        mean_utility_with_history=('ridge_with_history', 'mean'))
    summary.to_csv(OUT / 'E252_SUMMARY.csv', index=False)
    status = {'status': 'RETROSPECTIVE_DEVELOPMENT', 'created_at': datetime.now().astimezone().isoformat(),
              'n_primary_tasks': len(data), 'targets': sorted(data.target.unique().tolist()),
              'n_seeds': N_SEEDS, 'budgets': list(BUDGETS),
              'source_sha256': sha(SOURCE), 'runs_sha256': sha(OUT / 'E252_ALL_RUNS.csv'),
              'warning': 'E201 outcomes were public; this is not an independent confirmation or PertEMA reproduction'}
    (OUT / 'E252_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

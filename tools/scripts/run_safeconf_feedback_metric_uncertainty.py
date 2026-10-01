#!/usr/bin/env python3
"""Paired cluster uncertainty for every feedback metric and curve integral.

The same biological-cluster draw is applied to all methods and budgets.
Zero-feedback target learners are not invented: comparable learning-curve
integrals span the fitted 10%--100% budgets; zero-feedback is separate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import SEEDS, metrics
from tools.scripts import run_safeconf_research_closure as closure

METRICS = ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate',
           'error_at_10', 'error_at_20', 'error_at_50']


def vector_metrics(truth, ids, scores):
    n = len(truth); count = len(scores)
    if not n: return np.full((count, len(METRICS)), np.nan)
    k = int(np.ceil(.2 * n))
    row_ids = np.broadcast_to(ids, scores.shape)
    high = np.lexsort((row_ids, -scores), axis=1)[:, :k]
    low = np.lexsort((row_ids, scores), axis=1)
    oracle = np.lexsort((ids, -truth))[:k]
    den = truth[oracle].mean() - truth.mean()
    u = (truth[high].mean(axis=1) - truth.mean()) / den if n >= 20 and den > 1e-12 else np.full(count, np.nan)
    a = rankdata(scores, axis=1); a -= a.mean(axis=1, keepdims=True)
    b = rankdata(truth); b -= b.mean()
    norm = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b))
    rho = np.divide(a @ b, norm, out=np.full(count, np.nan), where=norm > 0) if n >= 3 else np.full(count, np.nan)
    accepted = truth[low]
    aurc = (np.cumsum(accepted, axis=1) / np.arange(1, n + 1)).mean(axis=1)
    miss = 1 - np.isin(high, oracle).mean(axis=1)
    coverages = [accepted[:, :int(np.ceil(c / 100 * n))].mean(axis=1) for c in (10, 20, 50)]
    return np.column_stack([u, rho, aurc, miss, *coverages])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=closure.OUT)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--include-fixed-baselines', action='store_true',
                        help='Compare feedback gains with zero-feedback history/support rules on the same holdout')
    parser.add_argument('--output-dir', type=Path,
                        help='Separate derived statistics version; preserve existing interval files')
    args = parser.parse_args(); out = args.results
    destination = args.output_dir or out
    destination.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(out / 'STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz')
    budgets = [.1, .25, .5, .75, 1.]
    parts = [base]
    for relative in ['official_pertema_adaptation/TASK_PREDICTIONS.csv.gz',
                     'native_control_pertema_adaptation/TASK_PREDICTIONS.csv.gz']:
        if (out / relative).exists(): parts.append(pd.read_csv(out / relative))
    if args.include_fixed_baselines:
        template = base[base.method.eq('Shared') & base.seed.eq(SEEDS[0]) & base.budget.gt(0)].copy()
        for filename, names in [('MATRIX_TASK_PREDICTIONS.csv.gz',
                                {'Learned_WeightedHistoryDistance': 'FixedLearnedHistoryDistance',
                                 'Manual_WeightedHistoryDistance': 'FixedManualHistoryDistance'}),
                               ('SUPPORT_CONTROL_TASK_PREDICTIONS.csv.gz',
                                {'NegativeHistorySupport': 'FixedHistorySupport'})]:
            fixed = pd.read_csv(out / filename)
            fixed = fixed[fixed.seed.eq(SEEDS[0]) & fixed.line.eq('TxPert_to_McFaline')]
            for original, name in names.items():
                reference = fixed[fixed.method.eq(original)].set_index('task_id').risk
                control = template.copy(); control['method'] = name
                control['risk'] = control.task_id.map(reference)
                if control.risk.isna().any():
                    raise RuntimeError('fixed no-feedback baseline lacks the exact holdout cohort')
                parts.append(control)
    all_data = pd.concat(parts, ignore_index=True)
    all_data = all_data[all_data.seed.eq(SEEDS[0]) & all_data.budget.gt(0)]
    methods = sorted(all_data.method.unique())
    meta = ['task_id', 'target', 'gene', 'true_error_rmse']
    frame = base[base.method.eq('Shared') & base.seed.eq(SEEDS[0]) & base.budget.eq(0)][meta].sort_values('task_id').reset_index(drop=True)
    matrices = []
    for budget in budgets:
        wide = all_data[all_data.budget.eq(budget)].pivot(index='task_id', columns='method', values='risk')
        values = wide.loc[frame.task_id, methods].to_numpy(float).T
        if not np.isfinite(values).all(): raise RuntimeError('different task coverage in feedback curves')
        matrices.append(values)
    scores = np.stack(matrices)
    contexts = frame.target.to_numpy(str); truth = frame.true_error_rmse.to_numpy(float); ids = frame.task_id.to_numpy(str)
    def evaluate(idx):
        values = []
        for context in sorted(set(contexts)):
            rows = idx[contexts[idx] == context]
            flat = scores[:, :, rows].reshape(len(budgets) * len(methods), len(rows))
            values.append(vector_metrics(truth[rows], ids[rows], flat).reshape(len(budgets), len(methods), len(METRICS)))
        return np.nanmean(values, axis=0)
    point = evaluate(np.arange(len(frame)))
    # Verify the optimized calculation against the existing scalar endpoint code.
    for bi, budget in enumerate(budgets):
        for mi, method in enumerate(methods):
            check = []
            for context in sorted(set(contexts)):
                rows = np.flatnonzero(contexts == context)
                result = metrics(frame.iloc[rows], scores[bi, mi, rows])
                check.append([result[k] for k in METRICS])
            expected = np.nanmean(check, axis=0)
            if not np.allclose(point[bi, mi], expected, equal_nan=True, atol=1e-12):
                raise RuntimeError('vectorized metric does not reproduce registered scalar endpoints')
    groups = [g.index.to_numpy() for _, g in frame.groupby('gene', sort=True)]
    rng = np.random.default_rng(SEEDS[0])
    draws = np.empty((args.bootstrap, *point.shape))
    for i in range(args.bootstrap):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        draws[i] = evaluate(rows)
        if (i + 1) % 500 == 0: print(f'Feedback all-metric cluster draws: {i + 1}/{args.bootstrap}', flush=True)
    rows = []
    for bi, budget in enumerate(budgets):
        for mi, method in enumerate(methods):
            for ki, metric in enumerate(METRICS):
                sample = draws[:, bi, mi, ki]
                finite = np.isfinite(sample)
                rows.append({'budget': budget, 'method': method, 'metric': metric, 'point_estimate': point[bi, mi, ki],
                    'ci95_lower': float(np.nanquantile(sample, .025)) if finite.any() else np.nan,
                    'ci95_upper': float(np.nanquantile(sample, .975)) if finite.any() else np.nan,
                    'n_valid_draws': int(finite.sum()), 'bootstrap_replicates': args.bootstrap,
                    'n_tasks': len(frame), 'n_clusters': len(groups), 'role': 'SEEN_POST_CONFIRMATION'})
    closure.tx.atomic_csv(destination / 'FEEDBACK_ALL_METRIC_INTERVALS.csv', pd.DataFrame(rows))
    comparisons = [('ResidualHGB', 'Shared'), ('PublicTarget_HGB', 'TargetOnly_HGB'),
                   ('ResidualHGB', 'PublicTarget_HGB'), ('SharedTarget_HGB', 'PublicTarget_HGB')]
    if 'PertEMA_Native_adapted' in methods:
        comparisons += [('PertEMA_Native_Public_adapted', 'PertEMA_Native_adapted'),
                        ('PertEMA_Native_Shared_adapted', 'PertEMA_Native_Public_adapted'),
                        ('ResidualHGB', 'PertEMA_Native_Public_adapted')]
    if args.include_fixed_baselines:
        comparisons += [('Shared', 'FixedHistorySupport'),
                        ('Shared', 'FixedLearnedHistoryDistance'),
                        ('PublicTarget_HGB', 'FixedHistorySupport'),
                        ('ResidualHGB', 'FixedHistorySupport')]
        if 'PertEMA_Native_Public_adapted' in methods:
            comparisons.append(('PertEMA_Native_Public_adapted', 'FixedHistorySupport'))
    differences = []
    for a, b in comparisons:
        ai, bj = methods.index(a), methods.index(b)
        for bi, budget in enumerate(budgets):
            for ki, metric in enumerate(METRICS):
                sample = draws[:, bi, ai, ki] - draws[:, bi, bj, ki]
                finite = np.isfinite(sample)
                differences.append({'budget': budget, 'method_a': a, 'method_b': b, 'metric': metric,
                    'point_difference': point[bi, ai, ki] - point[bi, bj, ki],
                    'ci95_lower': float(np.nanquantile(sample, .025)) if finite.any() else np.nan,
                    'ci95_upper': float(np.nanquantile(sample, .975)) if finite.any() else np.nan,
                    'bootstrap_replicates': args.bootstrap})
    closure.tx.atomic_csv(destination / 'FEEDBACK_ALL_METRIC_PAIRED_DIFFERENCES.csv', pd.DataFrame(differences))
    integral = np.trapezoid(point, budgets, axis=0) / .9
    integral_draws = np.trapezoid(draws, budgets, axis=1) / .9
    auc_rows = []
    for mi, method in enumerate(methods):
        for ki, metric in enumerate(METRICS):
            sample = integral_draws[:, mi, ki]; finite = np.isfinite(sample)
            auc_rows.append({'method': method, 'metric': metric, 'budget_range': '10%--100%',
                'normalized_curve_integral': integral[mi, ki],
                'ci95_lower': float(np.nanquantile(sample, .025)) if finite.any() else np.nan,
                'ci95_upper': float(np.nanquantile(sample, .975)) if finite.any() else np.nan,
                'n_valid_draws': int(finite.sum()), 'bootstrap_replicates': args.bootstrap,
                'zero_feedback_target_learner_imputed': False})
    closure.tx.atomic_csv(destination / 'FEEDBACK_LEARNING_CURVE_INTEGRALS.csv', pd.DataFrame(auc_rows))
    closure.tx.atomic_json(destination / 'FEEDBACK_METRIC_INTERVAL_STATUS.json', {'status': 'COMPLETE',
        'methods': methods, 'budgets': budgets, 'metric_names': METRICS, 'bootstrap_replicates': args.bootstrap,
        'same_cluster_draw_across_budgets': True, 'scalar_metric_reproduction_passed': True,
        'target_zero_feedback_imputation': False, 'role': 'SEEN_POST_CONFIRMATION',
        'fixed_zero_feedback_rules_included': args.include_fixed_baselines})


if __name__ == '__main__': main()

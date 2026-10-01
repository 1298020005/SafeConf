#!/usr/bin/env python3
"""Same-task support/content/source attribution, with paired cluster intervals.

All methods and all five physical-content nulls are retained. No best test
method is promoted. Every paired comparison must contain the complete cohort.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import SEEDS, metrics, paired_prediction_wide
from tools.scripts.run_safeconf_feedback_metric_uncertainty import METRICS, vector_metrics
from tools.scripts import run_safeconf_research_closure as closure

OUT = closure.OUT / 'common_gene_axis/results'


def main():
    matrix = pd.read_csv(OUT / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    support = pd.read_csv(OUT / 'SUPPORT_CONTROL_TASK_PREDICTIONS.csv.gz')
    content = pd.read_csv(OUT / 'public_mechanisms/CONTENT_SHUFFLE_TASK_PREDICTIONS.csv.gz')
    content['method'] = content.apply(lambda r: f'ContentNull_{r.method}_{int(r.order)}', axis=1)
    data = pd.concat([matrix, support, content], ignore_index=True)
    data = data[data.seed.eq(SEEDS[0])]
    primary = matrix[matrix.seed.eq(SEEDS[0]) & matrix.method.eq('Manual_hgb')]
    error_lookup = primary.set_index(['line', 'task_id']).true_error_rmse
    audit = []
    for line, part in data.groupby('line', sort=True):
        mapped = np.asarray([error_lookup.loc[(line, task)] for task in part.task_id])
        if not np.allclose(part.true_error_rmse, mapped, rtol=2 * np.finfo(np.float32).eps, atol=1e-10):
            raise RuntimeError('content control used a different registered truth contract')
        for (method, context), group in part.groupby(['method', 'target'], sort=True):
            expected = np.asarray([error_lookup.loc[(line, task)] for task in group.task_id])
            old_order = np.lexsort((group.task_id.to_numpy(str), group.true_error_rmse.to_numpy()))
            new_order = np.lexsort((group.task_id.to_numpy(str), expected))
            if not np.array_equal(old_order, new_order):
                raise RuntimeError('truth precision changed error ranks; refit affected label models first')
            audit.append({'line': line, 'method': method, 'context': context, 'n_tasks': len(group),
                          'max_absolute_difference': float(np.max(np.abs(group.true_error_rmse - expected))),
                          'entire_error_rank_order_unchanged': True,
                          'canonical_evaluation_truth': 'existing primary matrix; no primary label change'})
        data.loc[part.index, 'true_error_rmse'] = mapped
    closure.tx.atomic_csv(OUT / 'TRUTH_PRECISION_ALIGNMENT_AUDIT.csv', pd.DataFrame(audit))
    all_intervals, absolute = [], []
    for line, part in data.groupby('line', sort=True):
        comparisons = [('Learned_WeightedHistoryDistance', 'NegativeHistorySupport'),
                       ('Manual_WeightedHistoryDistance', 'NegativeHistorySupport'),
                       ('Learned_hgb', 'NegativeHistorySupport'),
                       ('Manual_hgb', 'NegativeHistorySupport'),
                       ('Learned_WeightedHistoryDistance', 'Manual_WeightedHistoryDistance')]
        comparisons += [('Manual_WeightedHistoryDistance', f'ContentNull_ManualHistoryDistance_{i}')
                        for i in range(5)]
        comparisons += [('Manual_hgb', f'ContentNull_ManualHGB_{i}') for i in range(5)]
        needed = sorted({x for pair in comparisons for x in pair})
        wide = paired_prediction_wide(part[part.method.isin(needed)])
        if wide[needed].isna().any().any():
            raise RuntimeError('support/content attribution changed the paired task cohort')
        values = wide[needed].to_numpy(float).T
        contexts, ids = wide.target.to_numpy(str), wide.task_id.to_numpy(str)
        truth = wide.true_error_rmse.to_numpy(float)
        def evaluate(indices):
            return np.nanmean([vector_metrics(truth[rows], ids[rows], values[:, rows])
                for c in sorted(set(contexts))
                for rows in [indices[contexts[indices] == c]]], axis=0)
        point = evaluate(np.arange(len(wide)))
        for mi, method in enumerate(needed):
            check = []
            for c in sorted(set(contexts)):
                rows = np.flatnonzero(contexts == c)
                result = metrics(wide.iloc[rows], values[mi, rows])
                check.append([result[k] for k in METRICS])
            if not np.allclose(point[mi], np.nanmean(check, axis=0), equal_nan=True, atol=1e-12):
                raise RuntimeError('optimized statistics changed the registered metric')
        groups = [g.index.to_numpy() for _, g in wide.groupby('gene', sort=True)]
        rng = np.random.default_rng(SEEDS[0])
        draws = np.empty((5000, *point.shape))
        for i in range(5000):
            rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            draws[i] = evaluate(rows)
        for mi, method in enumerate(needed):
            for ki, metric in enumerate(METRICS):
                sample = draws[:, mi, ki]
                absolute.append({'line': line, 'method': method, 'metric': metric,
                    'point_estimate': point[mi, ki], 'ci95_lower': np.nanquantile(sample, .025),
                    'ci95_upper': np.nanquantile(sample, .975), 'n_tasks': len(wide),
                    'n_clusters': len(groups), 'bootstrap_replicates': 5000})
        for a, b in comparisons:
            ai, bi = needed.index(a), needed.index(b)
            for ki, metric in enumerate(METRICS):
                sample = draws[:, ai, ki] - draws[:, bi, ki]
                all_intervals.append({'line': line, 'method_a': a, 'method_b': b, 'metric': metric,
                    'point_difference': point[ai, ki] - point[bi, ki],
                    'ci95_lower': np.nanquantile(sample, .025), 'ci95_upper': np.nanquantile(sample, .975),
                    'n_tasks': len(wide), 'n_clusters': len(groups), 'bootstrap_replicates': 5000,
                    'role': 'DEV_SEEN' if line != 'TxPert_to_McFaline' else 'SEEN_POST_CONFIRMATION'})
        closure.tx.atomic_csv(OUT / 'SUPPORT_CONTENT_ATTRIBUTION_INTERVALS.csv', pd.DataFrame(absolute))
        closure.tx.atomic_csv(OUT / 'SUPPORT_CONTENT_ATTRIBUTION_COMPARISONS.csv', pd.DataFrame(all_intervals))
        print(f'Attribution intervals complete: {line}; {len(wide)} tasks; {len(groups)} clusters', flush=True)
    closure.tx.atomic_json(OUT / 'ATTRIBUTION_STATISTICS_STATUS.json', {
        'status': 'COMPLETE', 'bootstrap_replicates': 5000,
        'task_pairing_on_biological_identity': True, 'full_cohort_required': True,
        'all_five_content_nulls_reported': True, 'primary_metric_unchanged': True})


if __name__ == '__main__':
    main()

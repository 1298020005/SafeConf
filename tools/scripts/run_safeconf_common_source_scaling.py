#!/usr/bin/env python3
"""Reproduce the registered source-budget/diversity matrix on the common axis.

Uses unchanged cached source/query numerical features, original five hash
orders, budgets, three seeds and learners. Only previously released primary
query errors are appended for evaluation in an isolated cache. No new method,
gene axis, upstream prediction or confirmation is introduced.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import SEEDS, ids_hash
from tools.scripts import run_safeconf_common_axis_closure as common


def main():
    closure = common.closure
    original_cache = common.COMMON / 'risk_cache'
    cache = common.COMMON / 'source_scaling_recheck/cache'
    out = common.DOC / 'results/source_scaling_recheck'
    if (out / 'STATUS.json').exists():
        raise RuntimeError('completed derived matrix cannot be overwritten')
    cache.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(common.DOC / 'results/MATRIX_TASK_PREDICTIONS.csv.gz')
    truth = truth[(truth.line == 'TxPert_to_McFaline') & (truth.method == 'Manual_hgb') &
                  (truth.seed == SEEDS[0])].set_index('task_id').true_error_rmse
    hashes = []
    for ref in ['Manual', 'Learned']:
        for domain in ['source', 'external']:
            path = original_cache / f'{domain}_{ref}.parquet'
            part = pd.read_parquet(path)
            hashes.append({'input': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'rows': len(part), 'cluster_hash': ids_hash(part.gene.unique())})
            if domain == 'external':
                part['true_error_rmse'] = part.task_id.map(truth)
                if part.true_error_rmse.isna().any():
                    raise RuntimeError('cached query lacks registered primary evaluation errors')
            part.to_parquet(cache / path.name, index=False)
    (out / 'REGISTERED_MATRIX.json').write_text(json.dumps({
        'role': 'SEEN_POST_CONFIRMATION', 'n_genes': 2840,
        'same_registered_methods_and_parameters': True, 'orders': list(range(5)),
        'budgets': [.1, .25, .5, .75, 1.], 'seeds': list(SEEDS),
        'only_allowed_budget_errors_enter_CDF': True,
        'history_bank_and_source_reference_features_fixed': True,
        'query_error_use': 'evaluation only, copied from canonical primary matrix',
        'new_upstream_calls': 0, 'new_large_upstream_training': 0, 'inputs': hashes,
    }, indent=2) + '\n')
    closure.RUNTIME = cache; closure.OUT = out
    original_fit = closure.fit_risk
    costs = []
    def timed_fit(frame, labels, columns, kind, seed=SEEDS[0], weighted=True):
        started = time.perf_counter()
        fitted = original_fit(frame, labels, columns, kind, seed, weighted)
        costs.append({'fit_seconds': time.perf_counter() - started,
                      'training_rows': len(frame), 'training_clusters': frame.gene.nunique(),
                      'source_predictors': frame.upstream.nunique(), 'seed': seed,
                      'learner': kind, 'new_upstream_calls': 0})
        return fitted
    closure.fit_risk = timed_fit
    started = time.perf_counter()
    closure.run_source_scaling()
    pd.DataFrame(costs).to_csv(out / 'FIT_COSTS.csv', index=False)
    # Query rows/scores are server-only. Keep reviewable summaries in the repo.
    for name in ['SOURCE_SCALING_TASK_PREDICTIONS.csv.gz', 'SOURCE_DIVERSITY_PREDICTIONS.csv.gz']:
        (out / name).rename(cache.parent / name)
    (out / 'STATUS.json').write_text(json.dumps({
        'status': 'COMPLETE', 'role': 'SEEN_POST_CONFIRMATION',
        'fitted_models': len(costs), 'cpu_fit_seconds_sum': sum(x['fit_seconds'] for x in costs),
        'elapsed_seconds': time.perf_counter() - started,
        'query_tasks': len(truth), 'new_upstream_calls': 0,
        'new_large_upstream_training': 0, 'original_inputs_preserved': True,
        'server_only_predictions': str(cache.parent),
    }, indent=2) + '\n')
    print((out / 'STATUS.json').read_text(), flush=True)


if __name__ == '__main__':
    main()

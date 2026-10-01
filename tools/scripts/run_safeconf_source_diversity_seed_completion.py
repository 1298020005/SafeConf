#!/usr/bin/env python3
"""Complete the registered three-seed source-diversity comparison.

This reuses the released common-axis DEV/SEEN feature cache and exact original
source/task selection. It does not alter the frozen Orion models or read Orion.
The first seed must reproduce every previous score before results are accepted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, fit_risk, ids_hash, rank_labels, summarize,
)

RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results/source_diversity_seed_completion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cache = RUNTIME / 'source_scaling_recheck/cache'
    old_path = RUNTIME / 'source_scaling_recheck/SOURCE_DIVERSITY_PREDICTIONS.csv.gz'
    output = RUNTIME / 'source_diversity_seed_completion'
    if output.exists() or DOC.exists():
        raise RuntimeError('Use new output version; prior results cannot be overwritten')
    old = pd.read_csv(old_path)
    DOC.mkdir(parents=True)
    output.mkdir(parents=True)
    inputs = [old_path] + [cache / f'{domain}_{ref}.parquet'
                           for ref in ('Manual', 'Learned') for domain in ('source', 'external')]
    registration = {'role': 'SEEN_POST_CONFIRMATION_SEED_COMPLETION',
                    'methods_features_labels_selection_and_weights_unchanged': True,
                    'seeds': list(SEEDS), 'source_feature_columns': P + PUBLIC,
                    'target_errors_used': 'released McFaline evaluation only; never in risk/CDF fit',
                    'new_upstream_calls': 0, 'Orion_access': False,
                    'input_bindings': [{'path': str(p), 'sha256': sha(p)} for p in inputs]}
    (DOC / 'REGISTRATION.json').write_text(json.dumps(registration, indent=2) + '\n')
    records, costs, audits = [], [], []
    began = time.monotonic()
    for ref in ('Manual', 'Learned'):
        source = pd.read_parquet(cache / f'source_{ref}.parquet')
        query = pd.read_parquet(cache / f'external_{ref}.parquet')
        if query.true_error_rmse.isna().any():
            raise RuntimeError('Released common-axis evaluation truth missing')
        selected = []
        for task, part in source.groupby('task_id', sort=True):
            part = part.sort_values('upstream')
            choice = int(hashlib.sha256(f'source-diversity-v1|{task}'.encode()).hexdigest(), 16) % len(part)
            selected.append(part.iloc[choice])
        equal = pd.DataFrame(selected).reset_index(drop=True)
        fits = {name: source[source.upstream.eq(name)].reset_index(drop=True)
                for name in sorted(source.upstream.unique())}
        fits.update(PooledEqualRecords=equal, PooledFullClusterWeight=source)
        labels = {}
        for name, fit in fits.items():
            labels[name], audit = rank_labels(fit, f'diversity-seed-completion/{ref}/{name}')
            audits.extend(audit)
        for seed in SEEDS:
            scores = {}
            for name, fit in fits.items():
                started = time.monotonic()
                scores[name] = fit_risk(fit, labels[name], P + PUBLIC, 'hgb', seed).predict(query)
                costs.append({'reference': ref, 'method': name, 'seed': seed,
                              'fit_seconds': time.monotonic() - started, 'n_source_rows': len(fit),
                              'n_source_clusters': fit.gene.nunique(),
                              'source_records_hash': ids_hash(fit.upstream + '::' + fit.task_id),
                              'new_upstream_calls': 0})
            scores['SeparateRiskAverage'] = np.mean(
                [scores[name] for name in sorted(source.upstream.unique())], axis=0)
            for name, values in scores.items():
                part = query[['task_id', 'target', 'gene', 'fold', 'upstream', 'true_error_rmse']].copy()
                part['line'] = 'TxPert_to_McFaline'
                part['method'] = f'{ref}/{name}'
                part['seed'] = seed
                part['risk'] = values
                records.append(part)
    predictions = pd.concat(records, ignore_index=True)
    first = predictions[predictions.seed.eq(SEEDS[0])]
    paired = first.merge(old[['task_id', 'method', 'risk']], on=['task_id', 'method'],
                         suffixes=('_new', '_old'), validate='one_to_one')
    difference = float(np.max(np.abs(paired.risk_new - paired.risk_old)))
    if len(paired) != len(first) or len(first) != len(old) or difference > 1e-14:
        raise RuntimeError('First seed did not exactly reproduce prior registered diversity')
    predictions.to_csv(output / 'TASK_PREDICTIONS.csv.gz', index=False, compression='gzip')
    strata, macro = summarize(predictions)
    strata.to_csv(DOC / 'STRATA.csv', index=False, lineterminator='\n')
    macro.to_csv(DOC / 'MACRO.csv', index=False, lineterminator='\n')
    pd.DataFrame(costs).to_csv(DOC / 'FIT_COSTS.csv', index=False, lineterminator='\n')
    pd.DataFrame(audits).to_csv(DOC / 'CDF_AUDIT.csv', index=False, lineterminator='\n')
    ranges = predictions.groupby(['task_id', 'method']).risk.agg(['min', 'max'])
    status = {'status': 'COMPLETE', 'role': registration['role'], 'fitted_models': len(costs),
              'macro_rows': len(macro), 'query_tasks': query.task_id.nunique(),
              'first_seed_score_max_abs_difference': difference,
              'maximum_score_range_across_seeds': float((ranges['max'] - ranges['min']).max()),
              'elapsed_seconds': time.monotonic() - began, 'new_upstream_calls': 0,
              'new_large_upstream_training': 0, 'Orion_access': False,
              'original_outputs_preserved': True, 'server_only_predictions': str(output)}
    (DOC / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()

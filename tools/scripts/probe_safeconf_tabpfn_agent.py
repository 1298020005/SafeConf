#!/usr/bin/env python3
"""Fixed TabPFN V2 insurance check using existing DEV risk-transfer caches.

No external truth is loaded. Model identity defines the frozen source/query
split and CDF strata only; the numeric learner sees P + PUBLIC features.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

# Local weights and offline inference prevent implicit account/license flows.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import numpy as np
import pandas as pd
import torch
import tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.learners import NumericPreprocessor
from tools.safeconf_continual.research import (
    P, PUBLIC, bootstrap_u20, cluster_weights, fit_risk, ids_hash, metrics,
    paired_prediction_wide, rank_labels, summarize,
)

CACHE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache')
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/tabpfn_v2_probe')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/tabpfn_feasibility'
SEED = 20260930
CHECKPOINT_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(source, target, ref, fold, output):
    paths = [CACHE / f'nested_{fold}_{name}_{ref}.parquet' for name in (source, target)]
    a, b = [pd.read_parquet(path) for path in paths]
    train = a[a.fold.ne(fold)].copy().reset_index(drop=True)
    query = b[b.fold.eq(fold)].copy().reset_index(drop=True)
    if set(train.gene) & set(query.gene):
        raise RuntimeError('outer biological clusters overlap')
    columns = P + PUBLIC
    forbidden = {'upstream', 'model_version', 'dataset_id', 'output_contract_id',
                 'target', 'fold', 'gene', 'task_id', 'true_error_rmse'}
    if set(columns) & forbidden:
        raise RuntimeError('identity/truth cannot be a risk feature')
    line = 'GAT_to_Exphormer' if source == 'TxPert_GAT' else 'Exphormer_to_GAT'
    labels, cdf = rank_labels(train, f'TabPFN_V2/{line}/{ref}/outer{fold}')
    valid = np.isfinite(labels)
    transform = NumericPreprocessor().fit(train.loc[valid, columns].to_numpy(float))
    x_train = transform.transform(train.loc[valid, columns].to_numpy(float))
    x_query = transform.transform(query[columns].to_numpy(float))
    weights = cluster_weights(train.loc[valid])
    checkpoint = RUNTIME / 'tabpfn-v2-regressor.ckpt'
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError('fixed official V2 checkpoint hash changed')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learner = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=str(checkpoint), random_state=SEED, device=device,
    )
    supports_weights = 'sample_weight' in inspect.signature(learner.fit).parameters
    parameters = learner.get_params(deep=False)
    audit = {
        'line': line, 'reference': ref, 'outer_fold': fold, 'seed': SEED,
        'role': 'DEV_SEEN_NESTED_ALGORITHM_ROBUSTNESS_DIAGNOSTIC',
        'n_train_rows': len(train), 'n_train_finite_label_rows': int(valid.sum()),
        'n_train_clusters': train.gene.nunique(), 'n_query_rows': len(query),
        'n_query_clusters': query.gene.nunique(), 'source_predictors': train.upstream.nunique(),
        'training_records_hash': ids_hash(train.upstream+'::'+train.task_id),
        'training_clusters_hash': ids_hash(train.gene),
        'query_records_hash': ids_hash(query.upstream+'::'+query.task_id),
        'query_clusters_hash': ids_hash(query.gene), 'clusters_disjoint': True,
        'input_files': [{'path': str(p), 'sha256': sha(p)} for p in paths],
        'input_columns': columns, 'preprocessed_columns': x_train.shape[1],
        'preprocessor': 'same NumericPreprocessor as fixed HGB, training-only fit',
        'rank_labels': 'same rank_labels helper; CDF fit on outer training rows only',
        'training_rank_label_sha256': hashlib.sha256(labels.astype('<f8').tobytes()).hexdigest(),
        'sample_weight_supported': supports_weights,
        'tabpfn_training_weights': 'cluster_equal' if supports_weights else 'row_equal',
        'weighted_hgb_training_weights': 'cluster_equal',
        'unweighted_hgb_training_weights': 'row_equal',
        'cluster_weight_min': float(weights.min()), 'cluster_weight_max': float(weights.max()),
        'weighted_hgb_vs_tabpfn_weight_mismatch': not supports_weights and not np.allclose(weights, 1),
        'row_resampling': False, 'parameter_search': False,
        'model_identity_as_feature': False, 'new_test_truth_opened': False,
        'new_upstream_calls': 0, 'new_large_upstream_training': 0,
        'package_version': tabpfn.__version__, 'torch_version': torch.__version__,
        'torch_path': torch.__file__, 'device': device,
        'parameters': parameters, 'checkpoint_sha256': CHECKPOINT_SHA,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'REGISTERED_CHECK.json', audit)
    pd.DataFrame(cdf).to_csv(output / 'CDF_AUDIT.csv', index=False)
    records, costs = [], []
    for name, weighted in [('HGB_cluster_equal', True), ('HGB_row_equal', False)]:
        start = time.perf_counter()
        model = fit_risk(train, labels, columns, 'hgb', SEED, weighted=weighted)
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        risk = model.predict(query)
        predict_seconds = time.perf_counter()-start
        part = query[['task_id','target','gene','fold','upstream','true_error_rmse']].copy()
        part['risk'], part['line'], part['method'], part['seed'] = risk, line, name, SEED
        records.append(part)
        costs.append({'method': name, 'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                      'gpu_hours': 0, 'peak_gpu_allocated_bytes': 0, 'peak_gpu_reserved_bytes': 0})
    print(json.dumps({'phase': 'hgb_complete', 'line': line, 'reference': ref, 'fold': fold}), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gpu_sync()
    start = time.perf_counter()
    kwargs = {'sample_weight': weights} if supports_weights else {}
    learner.fit(x_train, labels[valid], **kwargs)
    gpu_sync()
    fit_seconds = time.perf_counter()-start
    print(json.dumps({'phase': 'tabpfn_fit_complete', 'fit_seconds': fit_seconds}), flush=True)
    start = time.perf_counter()
    risk = np.clip(np.asarray(learner.predict(x_query), float), 0, 1)
    gpu_sync()
    predict_seconds = time.perf_counter()-start
    if not np.isfinite(risk).all():
        raise RuntimeError('nonfinite TabPFN risks')
    part = query[['task_id','target','gene','fold','upstream','true_error_rmse']].copy()
    part['risk'], part['line'], part['method'], part['seed'] = risk, line, 'TabPFN_V2_fixed', SEED
    records.append(part)
    costs.append({'method': 'TabPFN_V2_fixed', 'fit_seconds': fit_seconds,
                  'predict_seconds': predict_seconds,
                  'gpu_hours': (fit_seconds+predict_seconds)/3600 if device.startswith('cuda') else 0,
                  'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                  'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0})
    predictions = pd.concat(records, ignore_index=True)
    runtime_out = RUNTIME / output.name
    runtime_out.mkdir(exist_ok=True)
    predictions.to_csv(runtime_out / 'DEV_TASK_PREDICTIONS.csv.gz', index=False)
    strata, macro = summarize(predictions)
    strata.to_csv(output / 'DEV_CONTEXT_RESULTS.csv', index=False)
    macro.to_csv(output / 'DEV_MACRO_RESULTS.csv', index=False)
    pd.DataFrame(costs).to_csv(output / 'RESOURCE_COSTS.csv', index=False)
    resolved = {'n_estimators_fitted': getattr(learner, 'n_estimators_', learner.n_estimators),
                'inference_precision': str(getattr(learner, 'inference_precision_', learner.inference_precision)),
                'inference_config': str(getattr(learner, 'inference_config_', None))}
    write_json(output / 'STATUS.json', {'status': 'COMPLETE', 'role': audit['role'],
        'external_matrix_run': False, 'new_test_truth_opened': False,
        'server_predictions': str(runtime_out / 'DEV_TASK_PREDICTIONS.csv.gz'),
        'gpu_hours': sum(c['gpu_hours'] for c in costs), 'resolved_tabpfn': resolved})
    print(macro[['method','utility20','spearman','aurc']].to_json(orient='records'), flush=True)
    return sum(c['gpu_hours'] for c in costs)


def aggregate_dev():
    """Aggregate only the complete fixed DEV cohort and pair every query."""
    prediction_parts, cost_parts, audits, fold_parts = [], [], [], []
    rank_hashes = {}
    for source, target in [('TxPert_GAT', 'TxPert_Exphormer'),
                           ('TxPert_Exphormer', 'TxPert_GAT')]:
        for ref in ['Manual', 'Learned']:
            for fold in range(5):
                output = DOC / f'{source}_to_{target}_{ref}_fold{fold}'
                status = json.loads((output / 'STATUS.json').read_text())
                if status['status'] != 'COMPLETE':
                    raise RuntimeError('all twenty fixed DEV checks must complete')
                audit = json.loads((output / 'REGISTERED_CHECK.json').read_text())
                training = pd.read_parquet(audit['input_files'][0]['path'])
                training = training[training.fold.ne(fold)].reset_index(drop=True)
                labels, _ = rank_labels(training, 'TabPFN_DEV_reproduction_audit')
                label_hash = hashlib.sha256(labels.astype('<f8').tobytes()).hexdigest()
                if audit.get('training_rank_label_sha256', label_hash) != label_hash:
                    raise RuntimeError('source CDF labels changed')
                audit['reproduced_training_rank_label_sha256'] = label_hash
                key = (source, fold)
                if key in rank_hashes and rank_hashes[key] != label_hash:
                    raise RuntimeError('Manual/Learned source rank label mismatch')
                rank_hashes[key] = label_hash
                audit['parameters'] = json.dumps(audit['parameters'], sort_keys=True)
                audits.append(audit)
                part = pd.read_csv(status['server_predictions'])
                if part.groupby('method').size().nunique() != 1:
                    raise RuntimeError('methods do not cover the same query cohort')
                part['reference'] = ref
                prediction_parts.append(part)
                cost = pd.read_csv(output / 'RESOURCE_COSTS.csv')
                cost['line'], cost['reference'], cost['outer_fold'] = audit['line'], ref, fold
                cost_parts.append(cost)
                fold_result = pd.read_csv(output / 'DEV_CONTEXT_RESULTS.csv')
                fold_result['reference'], fold_result['outer_fold'] = ref, fold
                fold_parts.append(fold_result)
                for p in audit['input_files']:
                    if sha(Path(p['path'])) != p['sha256']:
                        raise RuntimeError('frozen input changed during the check')
    predictions = pd.concat(prediction_parts, ignore_index=True)
    costs = pd.concat(cost_parts, ignore_index=True)
    strata_parts, macro_parts, bootstrap = [], [], []
    for ref, part in predictions.groupby('reference', sort=True):
        strata, macro = summarize(part)
        strata['reference'], macro['reference'] = ref, ref
        strata_parts.append(strata); macro_parts.append(macro)
        for line, cohort in part.groupby('line', sort=True):
            wide = paired_prediction_wide(cohort)
            if len(wide) != 1808 or wide[['TabPFN_V2_fixed', 'HGB_cluster_equal', 'HGB_row_equal']].isna().any().any():
                raise RuntimeError('fixed complete DEV cohort or complete pairing changed')
            for comparator in ['HGB_cluster_equal', 'HGB_row_equal']:
                result = bootstrap_u20(wide, wide['TabPFN_V2_fixed'].to_numpy(),
                                       wide[comparator].to_numpy(), 5000, SEED)
                bootstrap.append({'line': line, 'reference': ref,
                                  'method_a': 'TabPFN_V2_fixed', 'method_b': comparator,
                                  'complete_paired_cohort': True, **result})
    pd.concat(strata_parts).to_csv(DOC / 'DEV_FULL_CONTEXT_RESULTS.csv', index=False)
    macro = pd.concat(macro_parts)
    macro.to_csv(DOC / 'DEV_FULL_MACRO_RESULTS.csv', index=False)
    pd.concat(fold_parts).to_csv(DOC / 'DEV_FOLD_CONTEXT_RESULTS.csv', index=False)
    fold_macro = pd.concat(fold_parts).groupby(
        ['line','reference','outer_fold','method','seed'], as_index=False,
    )[['utility20','spearman','aurc','high_risk_miss_rate']].mean()
    fold_macro.to_csv(DOC / 'DEV_FOLD_MACRO_RESULTS.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(DOC / 'DEV_PAIRED_CLUSTER_BOOTSTRAP.csv', index=False)
    costs.to_csv(DOC / 'DEV_FULL_RESOURCE_COSTS.csv', index=False)
    write_json(DOC / 'DEV_FULL_INPUT_AUDIT.json', audits)
    predictions.to_csv(RUNTIME / 'DEV_FULL_TASK_PREDICTIONS.csv.gz', index=False)
    write_json(DOC / 'DEV_FULL_STATUS.json', {
        'status': 'COMPLETE', 'n_checks': len(audits), 'fitted_models': len(costs),
        'n_tabpfn_fits': 20, 'n_weighted_hgb_fits': 20, 'n_row_equal_hgb_fits': 20,
        'role': 'DEV_SEEN_NESTED_ALGORITHM_ROBUSTNESS_DIAGNOSTIC',
        'parameters_searched': 0, 'seeds': [SEED], 'checkpoint_sha256': CHECKPOINT_SHA,
        'gpu_hours': float(costs.gpu_hours.sum()),
        'peak_gpu_allocated_bytes': int(costs.peak_gpu_allocated_bytes.max()),
        'peak_gpu_reserved_bytes': int(costs.peak_gpu_reserved_bytes.max()),
        'full_paired_query_tasks_per_line_reference': 1808,
        'bootstrap_replicates': 5000, 'bootstrap_cluster': 'gene',
        'external_matrix_run': False, 'new_test_truth_opened': False,
        'source_inputs_unchanged_verified_sha256': True,
        'manual_learned_source_rank_labels_identical_verified': True,
        'model_identity_as_feature': False, 'new_upstream_calls': 0,
        'server_predictions': str(RUNTIME / 'DEV_FULL_TASK_PREDICTIONS.csv.gz'),
    })
    print(macro[['line','reference','method','utility20','spearman','aurc']].to_json(orient='records'), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['TxPert_GAT','TxPert_Exphormer'], default='TxPert_GAT')
    parser.add_argument('--reference', choices=['Manual','Learned'], default='Manual')
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--aggregate-dev', action='store_true')
    args = parser.parse_args()
    if args.aggregate_dev:
        aggregate_dev()
        return
    if any(f not in range(5) for f in args.folds):
        raise ValueError('only existing frozen outer folds 0..4 are allowed')
    target = 'TxPert_Exphormer' if args.source == 'TxPert_GAT' else 'TxPert_GAT'
    gpu_hours = 0.0
    for fold in args.folds:
        output = DOC / f'{args.source}_to_{target}_{args.reference}_fold{fold}'
        if (output / 'STATUS.json').exists():
            raise RuntimeError('completed check cannot be overwritten')
        gpu_hours += run_one(args.source, target, args.reference, fold, output)
        if gpu_hours >= 2:
            raise RuntimeError('GPU budget reached; no additional fit is allowed')


if __name__ == '__main__':
    main()

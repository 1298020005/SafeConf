#!/usr/bin/env python3
"""Prepare Source-only numerical core on the frozen 3285-gene metadata axis.

Only historical Source DEV vectors and the frozen gene manifest are read.
The two public/risk model targets are deliberately distinct. No target
expression, target realised errors, parameter search or upstream fit occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.contracts import FrozenErrorCDF
from tools.safeconf_continual.learners import NumericPreprocessor
from tools.safeconf_continual.research import CDF_KEYS, FittedRisk, P, PUBLIC, SEEDS, fit_risk, ids_hash, rank_labels
from tools.scripts import run_safeconf_research_closure as closure

DEFAULT_RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1')
DEFAULT_DOC = closure.OUT / 'orion_preparation/source_core'
DEFAULT_GENES = Path('/home/yyf/data/safeconf_orion_frozen40_20261002/metadata_preparation_20261002_v1/GENE_MANIFEST.csv')
CONTRACT = 'Source3285_CP4000_log1p_matched_batch_delta_v1'
SCHEMA = 'safeconf_orion_source_core_20261002_v1'
INPUTS: set[Path] = set()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    INPUTS.add(path)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_array(runtime, name, value):
    path = runtime / (name + '.npy')
    np.save(path, np.asarray(value), allow_pickle=False)
    return path


def source_inputs(runtime, gene_manifest):
    tx = closure.tx
    genes = pd.read_csv(record(gene_manifest))
    native_genes = json.loads(record(tx.STORE_ROOT / 'gene_ids.json').read_text())['gene_ids']
    columns = genes.source_index.to_numpy(int)
    if len(genes) != 3285 or len(native_genes) != 3352:
        raise RuntimeError('Frozen Source gene dimensions changed')
    if not np.array_equal(genes.axis_index, np.arange(3285)) or len(set(columns)) != 3285:
        raise RuntimeError('Gene axis order/uniqueness changed')
    if list(np.asarray(native_genes)[columns]) != genes.gene_name.astype(str).tolist():
        raise RuntimeError('Source gene index/name mapping differs')
    tasks = pd.read_csv(record(tx.TASK_PATH))
    tasks = tasks[tasks.analysis_stratum.eq('primary_ge30')].reset_index(drop=True)
    tasks['fold'] = tx.fold_assignment(tasks)
    if len(tasks) != 1808 or not tasks.groupby('gene').fold.nunique().eq(1).all():
        raise RuntimeError('Source task/fold contract changed')
    rows = tasks.source_mean_delta_row.to_numpy(int)

    def load_vector(path):
        values = np.load(record(path), mmap_mode='r')
        if values.shape[1] != 3352:
            raise RuntimeError(f'Native vector axis differs: {path}')
        return np.asarray(values[rows][:, columns], dtype=float)

    controls = load_vector(tx.VECTOR_ROOT / 'e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy')
    truth = load_vector(tx.VECTOR_ROOT / 'e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy') - controls
    for name in ('manifest.json', 'public_memory.parquet', 'effect_vectors.npy', 'control_vectors.npy', 'gene_ids.json'):
        record(tx.STORE_ROOT / name)
    memory, native_effects, native_memory_controls, manifest = PublicMemoryStore(tx.STORE_ROOT).load()
    effects = np.asarray(native_effects[:, columns], dtype=float)
    memory_controls = np.asarray(native_memory_controls[:, columns], dtype=float)
    eligibility = pd.read_parquet(record(tx.STORE_ROOT / manifest['eligibility']['path']))
    pairs = tx.build_pairs(tasks, truth, controls, memory, effects, memory_controls, eligibility)
    predictions, frames = {}, {}
    for name, short, series in [('TxPert_GAT', 'txpert_gat', 'e201'), ('TxPert_Exphormer', 'txpert_exphormer', 'e205')]:
        prefix = 'E201' if series == 'e201' else 'E205'
        prediction = load_vector(tx.VECTOR_ROOT / f'{series}/pretruth_vectors/{prefix}_FAMILY_CENTROIDS.npy')
        model_controls = load_vector(tx.VECTOR_ROOT / f'{series}/pretruth_vectors/{prefix}_CONTROL_CENTROIDS.npy')
        prediction -= model_controls
        frame = pd.read_csv(record(tx.STAGE / f'safeconf_v4_development/{short}/FEATURE_TABLE.csv.gz'))
        frame = frame.set_index('task_id').loc[tasks.task_id].reset_index()
        frame['fold'] = tasks.fold
        frame['true_error_rmse'] = tx.rmse_rows(prediction, truth)
        frame = closure.mc.prediction_features(frame, prediction)
        frame = closure.annotate(frame, 'TxPert_E201', name, f'{series}_official_frozen', CONTRACT)
        predictions[name], frames[name] = prediction, frame
        save_array(runtime, f'SOURCE_{name}_PREDICTED_EFFECTS', prediction)
        save_array(runtime, f'SOURCE_{name}_CONTROLS', model_controls)
        frame.to_parquet(runtime / f'SOURCE_{name}_P_BASE.parquet', index=False)
    for name, values in [('SOURCE_TRUE_EFFECTS', truth), ('SOURCE_CONTROLS', controls),
                         ('SOURCE_PUBLIC_EFFECTS', effects), ('SOURCE_PUBLIC_CONTROLS', memory_controls)]:
        save_array(runtime, name, values)
    genes.to_csv(runtime / 'GENE_MANIFEST.csv', index=False)
    write_json(runtime / 'GENE_IDS.json', genes.gene_name.astype(str).tolist())
    tasks.to_parquet(runtime / 'SOURCE_TASKS.parquet', index=False)
    tasks[['task_id', 'gene', 'target', 'fold']].to_csv(runtime / 'SPLIT_MANIFEST.csv', index=False)
    pairs.to_parquet(runtime / 'SOURCE_PUBLIC_PAIR_FEATURES.parquet', index=False)
    memory.to_parquet(runtime / 'SOURCE_PUBLIC_MEMORY_METADATA.parquet', index=False)
    eligibility.to_parquet(runtime / 'SOURCE_PUBLIC_ELIGIBILITY.parquet', index=False)
    sidecar = tasks[['task_id', 'gene', 'target', 'fold']].copy()
    studies = {'K562': 'Replogle2022', 'RPE1': 'Replogle2022', 'hepg2': 'Nadig2025', 'jurkat': 'Nadig2025'}
    sidecar['OriginalStudy'] = sidecar.target.map(studies)
    if sidecar.OriginalStudy.isna().any():
        raise RuntimeError('Unregistered Source original study')
    sidecar.to_csv(runtime / 'SOURCE_ORIGINAL_STUDY_SIDECAR.csv', index=False)
    return tasks, pairs, effects, frames, predictions, truth


def build_nested(runtime, inputs):
    tasks, pairs, effects, bases, predictions, truth = inputs
    audits, biology = [], []
    oof_scores = np.full(len(pairs), np.nan)
    nested_scores = []
    nested_masks = []
    for outer in range(5):
        started = time.monotonic()
        train_pairs, test_pairs = pairs.fold.ne(outer), pairs.fold.eq(outer)
        scores = np.full(len(pairs), np.nan)
        scores[test_pairs] = closure.fit_public_scores(pairs, train_pairs, test_pairs, f'outer{outer}/test', audits)
        oof_scores[test_pairs] = scores[test_pairs]
        nested_masks.append(train_pairs.to_numpy())
        train_tasks = tasks[tasks.fold.ne(outer)].copy()
        for inner, (fit_rows, query_rows) in enumerate(GroupKFold(4).split(train_tasks, groups=train_tasks.gene)):
            fit_ids, query_ids = set(train_tasks.iloc[fit_rows].task_id), set(train_tasks.iloc[query_rows].task_id)
            f, q = pairs.task_id.isin(fit_ids), pairs.task_id.isin(query_ids)
            if (pairs.loc[f | q, 'fold'] == outer).any():
                raise RuntimeError('Outer query entered inner Public OOF')
            scores[q] = closure.fit_public_scores(pairs, f, q, f'outer{outer}/inner{inner}', audits)
        if not np.isfinite(scores).all():
            raise RuntimeError('Incomplete nested Source public predictions')
        nested_scores.append(scores)
        priors, summary = closure.prior_arrays(tasks, pairs, effects, scores)
        summary.to_parquet(runtime / f'nested_{outer}_PRIOR_SUMMARY.parquet', index=False)
        for ref in closure.REFS:
            save_array(runtime, f'nested_{outer}_{ref}_PRIOR_EFFECTS', priors[ref])
        for name, base in bases.items():
            for ref in closure.REFS:
                frame = closure.reference_frame(base, predictions[name], priors[ref], summary, ref)
                frame.to_parquet(runtime / f'nested_{outer}_{name}_{ref}.parquet', index=False)
        for ref in closure.REFS:
            query = tasks.fold.eq(outer).to_numpy()
            for i in np.flatnonzero(query):
                biology.append({'task_id': tasks.iloc[i].task_id, 'gene': tasks.iloc[i].gene,
                                'target': tasks.iloc[i].target, 'fold': outer, 'reference': ref,
                                'effect_rmse': float(closure.tx.rmse_rows(priors[ref][i:i+1], truth[i:i+1])[0]),
                                'effect_cosine': float(closure.tx.cosine_rows(priors[ref][i:i+1], truth[i:i+1])[0])})
        print(json.dumps({'phase': 'nested_public', 'outer_fold': outer,
                          'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)
    if not np.isfinite(oof_scores).all() or len(audits) != 25:
        raise RuntimeError('Source OOF/audit incomplete')
    save_array(runtime, 'SOURCE_PUBLIC_OOF_TRANSFER_SCORES', oof_scores)
    save_array(runtime, 'SOURCE_PUBLIC_NESTED_TRANSFER_SCORES', nested_scores)
    save_array(runtime, 'SOURCE_PUBLIC_OUTER_TRAIN_PAIR_MASKS', nested_masks)
    priors, summary = closure.prior_arrays(tasks, pairs, effects, oof_scores)
    summary.to_parquet(runtime / 'SOURCE_OOF_PRIOR_SUMMARY.parquet', index=False)
    source = {}
    for ref in closure.REFS:
        save_array(runtime, f'SOURCE_OOF_{ref}_PRIOR_EFFECTS', priors[ref])
        full = pd.concat([closure.reference_frame(base, predictions[name], priors[ref], summary, ref)
                          for name, base in bases.items()], ignore_index=True)
        full.to_parquet(runtime / f'source_{ref}.parquet', index=False)
        source[ref] = full
    pd.DataFrame(audits).to_csv(runtime / 'NESTED_ISOLATION_AUDIT.csv', index=False)
    pd.DataFrame(biology).to_csv(runtime / 'PUBLIC_BIOLOGY_SOURCE_OOF.csv.gz', index=False)
    return source, audits


def fit_final(runtime, pairs, source):
    labels, audit = rank_labels(source['Manual'], 'OrionSource3285/full_Source_OOF')
    if len(labels) != 3616 or not np.isfinite(labels).all():
        raise RuntimeError('Source CDF labels incomplete')
    cdfs, cdf_arrays = [], {}
    for number, (key, group) in enumerate(source['Manual'].groupby(CDF_KEYS, dropna=False, sort=True)):
        cdf = FrozenErrorCDF.fit(group.true_error_rmse.to_numpy(float))
        transformed = cdf.transform(group.true_error_rmse.to_numpy(float))
        if not np.array_equal(transformed, labels[group.index]):
            raise RuntimeError('Source midrank CDF alignment differs')
        array_key = f'cdf_{number}'
        cdf_arrays[array_key] = np.asarray(cdf.sorted_training_errors)
        cdfs.append(dict(zip(CDF_KEYS, key)) | {'array_key': array_key, 'n_rows': len(group),
                     'training_records_hash': ids_hash(group.upstream.astype(str) + '::' + group.task_id.astype(str)),
                     'sorted_training_errors': list(cdf.sorted_training_errors)})
    np.savez_compressed(runtime / 'SOURCE_ERROR_CDF_ARRAYS.npz', **cdf_arrays)
    write_json(runtime / 'SOURCE_ERROR_CDF.json', {'schema': 'Source_only_per_upstream_per_context_midrank_CDF',
               'formula': '(left_count + right_count) / (2 * n_source_group)',
               'group_keys': CDF_KEYS, 'groups': cdfs, 'fit_role': 'Source_DEV_only'})
    pd.DataFrame(audit).to_csv(runtime / 'SOURCE_ERROR_CDF_AUDIT.csv', index=False)
    save_array(runtime, 'SOURCE_RISK_MIDRANK_LABELS', labels)
    entries = []
    final_predictions = source['Manual'][['task_id', 'gene', 'target', 'upstream', 'fold', 'true_error_rmse']].copy()
    for ref, training_ref, columns in [('P_only', 'Manual', P), ('Manual', 'Manual', P + PUBLIC), ('Learned', 'Learned', P + PUBLIC)]:
        frame = source[training_ref]
        save_array(runtime, f'SOURCE_{ref}_RISK_FEATURES', frame[columns].to_numpy(float))
        for kind in ('ridge', 'hgb'):
            fitted = fit_risk(frame, labels, columns, kind, SEEDS[0])
            path = runtime / f'SOURCE_RISK_{ref}_{kind}.joblib'
            joblib.dump(fitted, path, compress=3)
            restored = joblib.load(path)
            raw = fitted.predict(frame, clip=False)
            if not np.array_equal(raw, restored.predict(frame, clip=False)):
                raise RuntimeError('Serialized risk model predictions differ')
            final_predictions[f'{ref}_{kind}'] = restored.predict(frame)
            entries.append({'role': 'Source_risk', 'reference': ref, 'learner': kind,
                            'path': path.name, 'sha256': sha(path), 'columns': columns,
                            'n_rows': len(frame), 'target': 'Source_per_upstream_per_context_midrank_CDF',
                            'seed': SEEDS[0], 'preprocessor_persisted': True,
                            'numeric_model_width_with_missing_flags': 2 * len(columns),
                            'model_params': fitted.model.get_params()})
            print(json.dumps({'phase': 'final_risk', 'reference': ref, 'learner': kind, 'sha256': entries[-1]['sha256']}), flush=True)
    final_predictions.to_parquet(runtime / 'SOURCE_FINAL_RISK_FIT_PREDICTIONS.parquet', index=False)
    # Public biology supervision is continuous transfer RMSE, unweighted,
    # with the original 20260929 seed and no [0,1] output clipping.
    columns = closure.tx.PAIR_FEATURES
    preprocessor = NumericPreprocessor().fit(pairs[columns].to_numpy(float))
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_depth=3,
              min_samples_leaf=20, l2_regularization=10., random_state=closure.tx.SEED)
    model.fit(preprocessor.transform(pairs[columns].to_numpy(float)), pairs.transfer_rmse.to_numpy(float))
    retrieval = FittedRisk(preprocessor, model, columns)
    path = runtime / 'SOURCE_PUBLIC_BIOLOGY_FULL_HGB.joblib'
    joblib.dump(retrieval, path, compress=3)
    raw = retrieval.predict(pairs, clip=False)
    restored = joblib.load(path)
    if not np.array_equal(raw, restored.predict(pairs, clip=False)):
        raise RuntimeError('Serialized public model predictions differ')
    save_array(runtime, 'SOURCE_PUBLIC_FULLFIT_TRANSFER_PREDICTIONS', raw)
    save_array(runtime, 'SOURCE_PUBLIC_NUMERIC_FEATURES', pairs[columns].to_numpy(float))
    save_array(runtime, 'SOURCE_PUBLIC_TRANSFER_RMSE_LABELS', pairs.transfer_rmse.to_numpy(float))
    entries.append({'role': 'Public_biology_retrieval', 'reference': 'Learned_regularized_fixed_blend_0.5',
                    'learner': 'hgb', 'path': path.name, 'sha256': sha(path), 'columns': columns,
                    'n_rows': len(pairs), 'target': 'biological_transfer_rmse', 'predict_clip': False,
                    'seed': closure.tx.SEED, 'sample_weight': None, 'preprocessor_persisted': True,
                    'numeric_model_width_with_missing_flags': 2 * len(columns), 'model_params': model.get_params()})
    write_json(runtime / 'MODEL_MANIFEST.json', {'schema': SCHEMA, 'models': entries})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    parser.add_argument('--gene-manifest', type=Path, default=DEFAULT_GENES)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.doc.mkdir(parents=True, exist_ok=True)
    if (args.runtime / 'STATUS.json').exists():
        raise RuntimeError('Existing preparation status: use a new runtime version rather than overwrite')
    started = time.monotonic()
    status = {'schema': SCHEMA, 'status': 'RUNNING', 'started_utc': utc(), 'pid': os.getpid(),
              'python': sys.executable, 'cpu_threads': args.cpu_threads,
              'runtime': str(args.runtime), 'target_expression_read': False,
              'Orion_validation_errors_used': False, 'Orion_test_truth_opened': False,
              'new_upstream_fits': 0, 'parameter_searches': 0}
    write_json(args.runtime / 'STATUS.json', status)
    write_json(args.doc / 'STATUS.json', status)
    print(json.dumps(status), flush=True)
    try:
        with threadpool_limits(limits=args.cpu_threads):
            inputs = source_inputs(args.runtime, args.gene_manifest)
            print(json.dumps({'phase': 'source_reprojection_complete', 'n_tasks': len(inputs[0]),
                              'n_genes': inputs[2].shape[1], 'n_pairs': len(inputs[1])}), flush=True)
            source, isolation = build_nested(args.runtime, inputs)
            entries = fit_final(args.runtime, inputs[1], source)
        input_manifest = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sorted(INPUTS)]
        write_json(args.runtime / 'SOURCE_INPUT_HASHES.json', input_manifest)
        feature_manifest = {'schema': SCHEMA, 'output_contract_id': CONTRACT, 'source_native_genes': 3352,
            'prepared_genes': 3285, 'source_gene_fraction': 3285 / 3352, 'n_biological_tasks': 1808,
            'n_upstream_task_records': 3616, 'upstreams': sorted(inputs[3]),
            'risk_P_columns': P, 'risk_public_columns': PUBLIC, 'public_pair_columns': closure.tx.PAIR_FEATURES,
            'metadata_never_used_as_numeric_features': ['OriginalStudy', 'gene', 'target', 'upstream', 'task_id', 'fold'],
            'OriginalStudy_sidecar': {'K562': 'Replogle2022', 'RPE1': 'Replogle2022', 'hepg2': 'Nadig2025', 'jurkat': 'Nadig2025'},
            'normalization': {'native': 'official CP4000 then log1p; TxPert Methods 4.2.2',
                              'native_arrays_already_normalized': True, 'delta_rescaled': False,
                              'axis_operation': 'column projection only; no library renormalization'},
            'public_history_policy': 'frozen upstream Source TRAIN eligibility; actual known history for each query',
            'public_history_is_restricted_to_risk_outer_train': False,
            'split_policy': 'five same-gene outer folds; all contexts and upstream models share gene folds; four inner Public folds',
            'public_supervision_nested_fits': len(isolation), 'public_fullfit_target': 'transfer_rmse',
            'public_fullfit_inference': 'joblib.load(...).predict(pair_frame, clip=False)',
            'risk_training': 'all Source Public OOF feature rows; Source only per upstream/context midrank labels',
            'risk_cluster_weighting': 'inverse records per gene, normalized to mean 1',
            'models': entries, 'sklearn_version': sklearn.__version__,
            'arrays': [{'path': path.name, 'shape': list(np.load(path, mmap_mode='r').shape),
                        'dtype': str(np.load(path, mmap_mode='r').dtype)} for path in sorted(args.runtime.glob('*.npy'))]}
        write_json(args.runtime / 'SOURCE_FEATURE_MANIFEST.json', feature_manifest)
        for name in ('GENE_MANIFEST.csv', 'SPLIT_MANIFEST.csv', 'NESTED_ISOLATION_AUDIT.csv',
                     'SOURCE_ERROR_CDF_AUDIT.csv', 'SOURCE_ORIGINAL_STUDY_SIDECAR.csv',
                     'SOURCE_FEATURE_MANIFEST.json', 'MODEL_MANIFEST.json', 'SOURCE_INPUT_HASHES.json'):
            shutil.copyfile(args.runtime / name, args.doc / name)
        artifacts = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                     for path in sorted(args.runtime.iterdir()) if path.is_file() and path.name != 'STATUS.json']
        write_json(args.runtime / 'ARTIFACT_HASHES.json', artifacts)
        shutil.copyfile(args.runtime / 'ARTIFACT_HASHES.json', args.doc / 'ARTIFACT_HASHES.json')
        elapsed = time.monotonic() - started
        status.update(status='COMPLETE', finished_utc=utc(), elapsed_seconds=round(elapsed, 3),
                      within_20_minute_budget=elapsed <= 1200, n_genes=3285, n_biological_tasks=1808,
                      n_upstream_task_records=3616, public_nested_fits=25, public_fullfit_models=1,
                      final_risk_models=6, all_serialized_models_prediction_roundtrip_exact=True,
                      source_feature_manifest_sha256=sha(args.runtime / 'SOURCE_FEATURE_MANIFEST.json'),
                      artifact_hash_manifest_sha256=sha(args.runtime / 'ARTIFACT_HASHES.json'))
        write_json(args.runtime / 'STATUS.json', status)
        write_json(args.doc / 'STATUS.json', status)
        print(json.dumps(status), flush=True)
    except Exception as exc:
        status.update(status='FAILED', failed_utc=utc(), elapsed_seconds=round(time.monotonic() - started, 3),
                      error_type=type(exc).__name__, error=str(exc))
        write_json(args.runtime / 'STATUS.json', status)
        write_json(args.doc / 'STATUS.json', status)
        raise


if __name__ == '__main__':
    main()

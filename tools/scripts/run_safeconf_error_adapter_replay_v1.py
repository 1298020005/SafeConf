#!/usr/bin/env python3
"""Actual isolated ErrorResidualAdapter lifecycle on Source DEV/SEEN errors.

This is a CPU-only operational replay. It uses one frozen upstream version and
existing OOF/held-out source errors, never permanent test truth. It appends two
legal error batches, fits an exact-version residual adapter after each batch,
registers immutable adapter artifacts, applies the training-only release gate,
and verifies serving reload from the current pointer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.contracts import (
    ErrorMemoryItem, FrozenErrorCDF, biological_task_key, perturbation_cluster_key,
)
from tools.safeconf_continual.learners import ErrorResidualAdapter, SharedRiskCore
from tools.safeconf_continual.versioned_runtime import VersionedErrorRuntime, ServingModelRegistry
from tools.safeconf_continual.research import P, PUBLIC, metrics, ids_hash

SOURCE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/source_scaling_recheck/cache/source_Manual.parquet')
RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/error_adapter_replay_v1')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/error_adapter_replay_v1'
UPSTREAM = 'TxPert_GAT'
MODEL_VERSION = 'e201_official_frozen'
CONTRACT = 'E201_common2840gene_log1p_delta_v1'
SEED = 20261003
FEATURES = P + PUBLIC
COMPONENT = 'TxPert_GAT_ErrorResidualAdapter'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path: Path) -> dict:
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator='\n')


def task_key(row) -> str:
    return biological_task_key('TxPert_E201', str(row.target), 'perturbation', str(row.gene),
                               str(row.condition), CONTRACT)


def cluster_key(row) -> str:
    return perturbation_cluster_key('TxPert_E201', 'perturbation', str(row.gene))


def adapter_items(frame: pd.DataFrame, shared: np.ndarray, ranks: np.ndarray, batch: str) -> list[ErrorMemoryItem]:
    values = []
    for i, row in frame.reset_index(drop=True).iterrows():
        shared_value = float(shared[i]); rank = float(ranks[i]); realised = float(row.true_error_rmse)
        values.append(ErrorMemoryItem(
            upstream_model_id=UPSTREAM,
            model_version=MODEL_VERSION,
            prediction_id=f'{UPSTREAM}::{row.task_id}',
            biological_task_key=task_key(row),
            perturbation_cluster_key=cluster_key(row),
            shared_risk=shared_value,
            realised_error=realised,
            error_rank=rank,
            shared_risk_residual=rank - shared_value,
            feedback_timestamp=f'2026-10-03T00:00:00+00:00::{batch}',
            provenance=f'source_Manual.parquet; {batch}; legal OOF/heldout Source DEV/SEEN error; no permanent TEST truth',
            is_oof_or_heldout=True,
        ))
    return values


def fit_bundle(adapter: ErrorResidualAdapter, version: int, cdf: FrozenErrorCDF,
               train_frame: pd.DataFrame, feature_frame: pd.DataFrame, shared_model: SharedRiskCore,
               out: Path) -> tuple[Path, dict]:
    bundle = {
        'schema': 'safeconf_error_residual_adapter_bundle_v1',
        'component': COMPONENT,
        'adapter_version': f'adapter-v{version}',
        'upstream_model_id': UPSTREAM,
        'model_version': MODEL_VERSION,
        'feature_columns': FEATURES,
        'output_contract_id': CONTRACT,
        'shared_core_scale': 'frozen CDF rank from fold2 only',
        'feedback_clusters': int(train_frame.gene.nunique()),
        'feedback_rows': int(len(train_frame)),
        'training_prediction_ids_hash': ids_hash(UPSTREAM + '::' + train_frame.task_id.astype(str)),
        'training_cluster_hash': ids_hash(train_frame.gene.astype(str).unique()),
        'cdf_training_rows': int(len(cdf.sorted_training_errors)),
        'cdf_training_error_hash': hashlib.sha256(np.asarray(cdf.sorted_training_errors, dtype=np.float64).tobytes()).hexdigest(),
        'adapter': adapter,
        'shared_model': shared_model,
    }
    path = out / f'ADAPTER_{version}.joblib'
    joblib.dump(bundle, path)
    path.chmod(0o444)
    loaded = joblib.load(path)
    if loaded['adapter'].upstream_model_id != UPSTREAM or loaded['adapter'].model_version != MODEL_VERSION:
        raise RuntimeError('adapter identity changed on reload')
    return path, bundle


def score_bundle(path: Path, frame: pd.DataFrame, shared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bundle = joblib.load(path)
    correction, final = bundle['adapter'].predict(frame[FEATURES].to_numpy(float), shared)
    if not np.isfinite(final).all() or not ((final >= 0) & (final <= 1)).all():
        raise RuntimeError('adapter produced invalid risk scores')
    return correction, final


def role_metrics(frame: pd.DataFrame, scores: np.ndarray, role: str) -> list[dict]:
    rows = []
    frame = frame.copy(); frame['risk'] = scores; frame['role'] = role
    for context, part in frame.groupby('target', sort=True):
        result = metrics(part, part.risk.to_numpy(float))
        rows.append({'role': role, 'target': str(context), **result})
    return rows


def release_gate(anchor: pd.DataFrame, gate: pd.DataFrame, old_anchor_scores: np.ndarray,
                 new_anchor_scores: np.ndarray, old_gate_scores: np.ndarray,
                 new_gate_scores: np.ndarray) -> dict:
    old_anchor = []; new_anchor = []; old_gate = []; new_gate = []
    for context, part in anchor.groupby('target', sort=True):
        idx = part.index.to_numpy(); old_anchor.append(metrics(part, old_anchor_scores[idx]))
        new_anchor.append(metrics(part, new_anchor_scores[idx]))
    for context, part in gate.groupby('target', sort=True):
        idx = part.index.to_numpy(); old_gate.append(metrics(part, old_gate_scores[idx]))
        new_gate.append(metrics(part, new_gate_scores[idx]))
    def mean(rows, key): return float(np.nanmean([row[key] for row in rows]))
    old_aurc = mean(old_anchor, 'aurc'); new_aurc = mean(new_anchor, 'aurc')
    delta_u20 = mean(new_gate, 'utility20') - mean(old_gate, 'utility20')
    aurc_relative = (new_aurc - old_aurc) / old_aurc if old_aurc > 0 else float('inf')
    miss_delta = max(mean(new_anchor, 'high_risk_miss_rate') - mean(old_anchor, 'high_risk_miss_rate'),
                     mean(new_gate, 'high_risk_miss_rate') - mean(old_gate, 'high_risk_miss_rate'))
    strata = [float(n['utility20'] - o['utility20']) for o, n in zip(old_gate, new_gate)]
    finite = bool(np.isfinite([delta_u20, aurc_relative, miss_delta, *strata]).all())
    passes = bool(finite and delta_u20 >= -0.005 and aurc_relative <= 0.05 and miss_delta <= 0.02
                  and np.mean(np.asarray(strata) >= 0) >= 0.60)
    return {
        'passes': passes, 'all_planned_strata_finite': finite,
        'release_metrics': {'delta_u20_new_feedback_gate': delta_u20,
                            'relative_aurc_degradation_anchor': aurc_relative,
                            'max_high_risk_miss_rate_degradation': miss_delta,
                            'nonnegative_feedback_context_fraction': float(np.mean(np.asarray(strata) >= 0))},
        'old_anchor_strata': len(old_anchor), 'new_feedback_strata': len(new_gate),
        'gate_used_only_OOF_heldout_DEV': True,
        'thresholds': {'delta_u20_min': -0.005, 'relative_aurc_max': 0.05,
                       'miss_rate_max': 0.02, 'nonnegative_fraction_min': 0.60},
        'stratum_deltas': strata,
    }


def main():
    started = time.monotonic()
    if RUNTIME.exists() or DOC.exists():
        raise FileExistsError('isolated adapter replay output already exists')
    RUNTIME.mkdir(parents=True); DOC.mkdir(parents=True)
    source = pd.read_parquet(SOURCE)
    required = set(['upstream','model_version','fold','gene','target','task_id','condition','true_error_rmse',*FEATURES])
    if not required.issubset(source.columns):
        raise RuntimeError('Source cache missing required adapter columns')
    source = source[source.upstream.eq(UPSTREAM)].copy().reset_index(drop=True)
    if len(source) != 1808 or source.model_version.nunique() != 1 or str(source.model_version.iloc[0]) != MODEL_VERSION:
        raise RuntimeError('source model/version scope changed')
    if source.task_id.duplicated().any() or source.gene.isna().any() or not np.isfinite(source[['true_error_rmse',*FEATURES]].to_numpy(float)).all():
        raise RuntimeError('invalid or nonfinite source adapter inputs')
    # Fixed biological-cluster roles. No permanent test truth is read.
    roles = {2:'SHARED_TRAIN', 3:'FEEDBACK_BATCH_1', 4:'FEEDBACK_BATCH_2', 1:'NEW_FEEDBACK_GATE', 0:'OLD_ANCHOR'}
    source['role'] = source.fold.map(roles)
    if source.role.isna().any() or source.groupby('gene').fold.nunique().max() != 1:
        raise RuntimeError('source fold/cluster identity is not indivisible')
    train = source[source.role.eq('SHARED_TRAIN')].reset_index(drop=True)
    feedback1 = source[source.role.eq('FEEDBACK_BATCH_1')].reset_index(drop=True)
    feedback2 = source[source.role.eq('FEEDBACK_BATCH_2')].reset_index(drop=True)
    gate = source[source.role.eq('NEW_FEEDBACK_GATE')].reset_index(drop=True)
    anchor = source[source.role.eq('OLD_ANCHOR')].reset_index(drop=True)
    if min(map(len, [train, feedback1, feedback2, gate, anchor])) < 20:
        raise RuntimeError('role is too small for registered utility metric')
    write_json(DOC/'REGISTRATION.json', {
        'schema':'safeconf_error_adapter_replay_v1', 'status':'REGISTERED_FIXED_DEV_REPLAY',
        'upstream_model_id':UPSTREAM, 'model_version':MODEL_VERSION, 'output_contract_id':CONTRACT,
        'source_path':bind(SOURCE), 'source_role':'DEV_SEEN_OOF_HELDOUT_ONLY', 'permanent_test_truth_read':False,
        'feature_columns':FEATURES, 'shared_train_fold':2, 'feedback_batches':[3,4],
        'new_feedback_gate_fold':1, 'old_anchor_fold':0, 'adapter_kind':'ridge', 'kappa':50.0,
        'release_gate':'train-only fixed thresholds; no permanent evaluation selection',
        'seed':SEED,
    })
    # Shared risk and CDF are frozen before any adapter feedback is ingested.
    shared_core = SharedRiskCore(kind='hgb', seed=SEED).fit(train[FEATURES].to_numpy(float), train.true_error_rmse.to_numpy(float))
    cdf = FrozenErrorCDF.fit(train.true_error_rmse.to_numpy(float))
    all_shared = shared_core.predict(source[FEATURES].to_numpy(float))
    all_rank = cdf.transform(source.true_error_rmse.to_numpy(float))
    source_index = {k:i for i,k in enumerate(source.task_id)}
    def arrays(frame):
        idx=np.asarray([source_index[k] for k in frame.task_id],dtype=int)
        return all_shared[idx], all_rank[idx]
    # Initial adapter consumes only batch 1.
    shared1, rank1 = arrays(feedback1)
    adapter1 = ErrorResidualAdapter(UPSTREAM, MODEL_VERSION, kind='ridge', kappa=50.0, seed=SEED).fit(
        feedback1[FEATURES].to_numpy(float), shared1, rank1, int(feedback1.gene.nunique()))
    adapter2_train = pd.concat([feedback1, feedback2], ignore_index=True)
    shared12, rank12 = arrays(adapter2_train)
    adapter2 = ErrorResidualAdapter(UPSTREAM, MODEL_VERSION, kind='ridge', kappa=50.0, seed=SEED).fit(
        adapter2_train[FEATURES].to_numpy(float), shared12, rank12, int(adapter2_train.gene.nunique()))
    # Error Memory is append-only and exact-version isolated.
    errors = VersionedErrorRuntime(RUNTIME/'error_memory', CONTRACT)
    errors.append(adapter_items(feedback1, shared1, rank1, 'feedback_batch_1'), 'feedback-batch-1')
    errors.append(adapter_items(feedback2, arrays(feedback2)[0], arrays(feedback2)[1], 'feedback_batch_2'), 'feedback-batch-2')
    memory = errors.load(UPSTREAM, MODEL_VERSION)
    if len(memory) != len(feedback1)+len(feedback2) or memory.prediction_id.duplicated().any():
        raise RuntimeError('error memory append/reload identity failed')
    # Persist candidate artifacts and register versioned serving objects.
    out_models=RUNTIME/'models'; out_models.mkdir(parents=True)
    path1, bundle1 = fit_bundle(adapter1, 1, cdf, feedback1, source, shared_core, out_models)
    path2, bundle2 = fit_bundle(adapter2, 2, cdf, adapter2_train, source, shared_core, out_models)
    serving = ServingModelRegistry(RUNTIME/'serving_models')
    split_hash = hashlib.sha256('fixed gene folds adapter replay v1'.encode()).hexdigest()
    schema_hash = hashlib.sha256('\n'.join(FEATURES).encode()).hexdigest()
    serving.register(COMPONENT, 'adapter-v1', path1, split_hash, schema_hash, status='RELEASED')
    serving.publish(COMPONENT, 'adapter-v1', 'initial exact-version OOF feedback adapter release')
    # Predictions before/after update, then apply gate using predeclared DEV roles.
    def score(path, frame):
        shared,_=arrays(frame); return score_bundle(path, frame, shared)
    old_gate_corr, old_gate = score(path1, gate); new_gate_corr, new_gate = score(path2, gate)
    old_anchor_corr, old_anchor = score(path1, anchor); new_anchor_corr, new_anchor = score(path2, anchor)
    gate_result = release_gate(anchor, gate, old_anchor, new_anchor, old_gate, new_gate)
    status = 'RELEASED' if gate_result['passes'] else 'REJECTED'
    serving.register(COMPONENT, 'adapter-v2', path2, split_hash, schema_hash, parent_version='adapter-v1', status=status)
    if status == 'RELEASED':
        serving.publish(COMPONENT, 'adapter-v2', 'new exact-version feedback batch passed fixed DEV release gate')
        selected='adapter-v2'; selected_path=path2
    else:
        selected='adapter-v1'; selected_path=path1
    served = serving.current(COMPONENT)
    loaded=joblib.load(served)
    sample = gate.copy()
    sample_shared,_=arrays(sample)
    _, served_score = loaded['adapter'].predict(sample[FEATURES].to_numpy(float), sample_shared)
    _, expected_score = score(selected_path, sample)
    if not np.array_equal(served_score, expected_score):
        raise RuntimeError('serving reload predictions differ from selected adapter')
    # Save auditable predictions and metrics; all evaluation rows are DEV/SEEN only.
    rows=[]
    for label, frame, s1, s2 in [('OLD_ANCHOR',anchor,old_anchor,new_anchor),('NEW_FEEDBACK_GATE',gate,old_gate,new_gate)]:
        for i,row in frame.reset_index(drop=True).iterrows():
            rows.append({'role':label,'task_id':row.task_id,'gene':row.gene,'target':row.target,
                         'fold':int(row.fold),'shared_risk':float(arrays(frame)[0][i]),
                         'risk_adapter_v1':float(s1[i]),'risk_adapter_v2':float(s2[i]),
                         'realised_error':float(row.true_error_rmse)})
    write_csv(DOC/'PREDICTIONS.csv',pd.DataFrame(rows))
    metric_rows=[]
    for label, frame, s1, s2 in [('OLD_ANCHOR',anchor,old_anchor,new_anchor),('NEW_FEEDBACK_GATE',gate,old_gate,new_gate)]:
        metric_rows.extend([{'role':label,'adapter':'v1',**x} for x in role_metrics(frame,s1,label)])
        metric_rows.extend([{'role':label,'adapter':'v2',**x} for x in role_metrics(frame,s2,label)])
    write_csv(DOC/'METRICS.csv',pd.DataFrame(metric_rows))
    write_json(DOC/'CDF_AUDIT.json', {
        'fit_role':'SHARED_TRAIN fold2 only', 'training_rows':len(train), 'training_clusters':int(train.gene.nunique()),
        'unique_errors':int(train.true_error_rmse.nunique()), 'nominal_resolution':1/len(train),
        'feedback_batch_1_rows':len(feedback1), 'feedback_batch_2_rows':len(feedback2),
        'cdf_sha256':hashlib.sha256(np.asarray(cdf.sorted_training_errors,dtype=np.float64).tobytes()).hexdigest(),
        'permanent_test_truth_read':False,
    })
    write_json(DOC/'INFORMATION_BUDGET_LEDGER.json', {
        'public_truth_used':0, 'source_error_records_batch_1':len(feedback1), 'source_error_records_batch_2':len(feedback2),
        'source_error_records_total':len(feedback1)+len(feedback2), 'cdf_training_error_records':len(train),
        'target_model_error_records':0, 'permanent_test_truth_read':False,
        'adapter_training_prediction_id_hash_v1':ids_hash(UPSTREAM+'::'+feedback1.task_id.astype(str)),
        'adapter_training_prediction_id_hash_v2':ids_hash(UPSTREAM+'::'+adapter2_train.task_id.astype(str)),
    })
    receipt={
        'status':'COMPLETE_EXACT_VERSION_ERROR_ADAPTER_LIFECYCLE', 'completed_utc':now(),
        'runtime_root':str(RUNTIME), 'report_root':str(DOC), 'upstream_model_id':UPSTREAM,
        'model_version':MODEL_VERSION, 'error_memory_revision_count':2,
        'feedback_batch_1_clusters':int(feedback1.gene.nunique()), 'feedback_batch_2_clusters':int(feedback2.gene.nunique()),
        'adapter_v1_training_rows':len(feedback1), 'adapter_v2_training_rows':len(adapter2_train),
        'adapter_v1_artifact':bind(path1), 'adapter_v2_artifact':bind(path2),
        'release_gate':gate_result, 'candidate_status':status, 'selected_serving_version':selected,
        'serving_reload_prediction_byte_identical':True, 'error_memory_exact_model_version_isolation':True,
        'new_external_feedback_benefit_claim':False, 'permanent_test_truth_read':False,
        'elapsed_seconds':time.monotonic()-started,
    }
    write_json(DOC/'COMPLETION.json',receipt)
    print(json.dumps(receipt,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()

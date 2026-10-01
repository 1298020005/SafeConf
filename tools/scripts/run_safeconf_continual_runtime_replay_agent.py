#!/usr/bin/env python3
"""Actual isolated Source DEV ingestion/update/gate/rollback replay.

No Orion or McFaline input is opened. Manual Public priors are reconstructed
from each real bank version. This does not fit a Public biological learner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.contracts import (
    ErrorMemoryItem, FrozenErrorCDF, PublicMemoryItem, biological_task_key,
    perturbation_cluster_key,
)
from tools.safeconf_continual.research import (
    CDF_KEYS, P, PUBLIC, cluster_weights, fit_risk, ids_hash, metrics, rank_labels,
)
from tools.safeconf_continual.update import ContinualUpdateManager, ReleaseMetrics
from tools.safeconf_continual.versioned_runtime import (
    ServingModelRegistry, VersionedErrorRuntime, VersionedPublicRuntime,
)
from tools.scripts.run_dual_memory_txpert_public_biology import prior_from_scores, rmse_rows, cosine_rows

COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
SOURCE = COMMON / 'source_scaling_recheck/cache/source_Manual.parquet'
ORIGINAL_PUBLIC = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201')
DOC_BASE = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
PROVENANCE = DOC_BASE / 'history_provenance_check/TXPERT_CONTEXT_STUDY_PROVENANCE.csv'
DEFAULT_REPORT = DOC_BASE / 'continual_runtime_replay/actual_v1'
DEFAULT_RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/continual_runtime_replay_20261002_v1')
SEED = 20260930
CONTRACT = 'E201_common2840gene_log1p_delta_v1'
ROLE = {0: 'OLD_ANCHOR', 1: 'NEW_TASK_GATE', 2: 'TRAIN_INITIAL', 3: 'TRAIN_INITIAL', 4: 'TRAIN_ADDED'}
STUDY = {'K562': 'Replogle_2022', 'RPE1': 'Replogle_2022',
         'hepg2': 'Nadig_2025', 'jurkat': 'Nadig_2025'}
MODELS = ('TxPert_GAT', 'TxPert_Exphormer')
CONTEXTS = ('K562', 'RPE1', 'hepg2', 'jurkat')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write_json(path, value, immutable=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if immutable:
        path.chmod(0o444)


def write_csv(path, frame, immutable=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, lineterminator='\n')
    if immutable:
        path.chmod(0o444)


def frame_records(frame):
    return frame.upstream.astype(str) + '::' + frame.task_id.astype(str)


def cdf_objects(frame):
    return {tuple(map(str, key)): FrozenErrorCDF.fit(part.true_error_rmse.to_numpy(float))
            for key, part in frame.groupby(CDF_KEYS, dropna=False, sort=True)}


def transform_errors(frame, cdfs):
    result = np.full(len(frame), np.nan)
    frame = frame.reset_index(drop=True)
    for key, group in frame.groupby(CDF_KEYS, dropna=False, sort=True):
        cdf = cdfs[tuple(map(str, key))]
        result[group.index] = cdf.transform(group.true_error_rmse.to_numpy(float))
    if not np.isfinite(result).all():
        raise RuntimeError('Training-frozen CDF does not cover these records')
    return result


def rebuild_features(base, tasks, task_rows, predictions, public_runtime, eligibility, version, out):
    """Only source history and predictions; no current task biological truth."""
    out.mkdir(parents=True, exist_ok=True)
    memory, effects, controls, manifest = public_runtime.load()
    if int(manifest.get('version', 1)) != version:
        raise RuntimeError('Public runtime is not the requested immutable version')
    if effects.shape != controls.shape or effects.shape[1] != 2840:
        raise RuntimeError('Public aligned full common-axis vectors differ')
    current_ids = set(memory.experiment_id.astype(str))
    available = eligibility[eligibility.public_experiment_id.isin(current_ids)]
    lookup = available.groupby(['target_context', 'condition']).public_experiment_id.apply(list).to_dict()
    by_id = memory.set_index('experiment_id')
    priors = np.empty((len(tasks), 2840), dtype=np.float64)
    summaries, edges = [], []
    for row, task in enumerate(tasks.itertuples(index=False)):
        candidates = lookup.get((str(task.target), str(task.condition)), [])
        if not candidates or len(candidates) != len(set(candidates)):
            raise RuntimeError('Frozen known-history task has missing/duplicate source records')
        items = by_id.loc[candidates]
        if (items.context.astype(str) == str(task.target)).any():
            raise RuntimeError('Exact current task/context/replicates entered Public history')
        if not items.condition.astype(str).eq(str(task.condition)).all():
            raise RuntimeError('History condition differs from current perturbation')
        if not items.perturbation_target.astype(str).eq(str(task.gene)).all():
            raise RuntimeError('History perturbation identity differs')
        indices = items.effect_vector_row.to_numpy(int)
        vectors = np.asarray(effects[indices], dtype=float)
        pair = pd.DataFrame({'memory_row': indices,
                             'log_source_cells': np.log1p(items.n_cells.to_numpy(float))})
        prior, weights, uncertainty, effective = prior_from_scores(pair, effects, 'unused', 'cells')
        priors[row] = prior
        conflict = float(rmse_rows(vectors, vectors.mean(axis=0, keepdims=True)).mean())
        summaries.append({'task_id': task.task_id, 'prior_uncertainty': uncertainty,
                          'effective_sources': effective, 'history_conflict': conflict,
                          'log_history_support': float(np.log1p(items.n_cells.sum()))})
        for item, weight in zip(items.itertuples(), weights):
            edges.append({'public_version': version, 'query_task_id': task.task_id,
                          'query_gene': task.gene, 'query_context': task.target,
                          'query_fold': int(task.fold), 'query_role': ROLE[int(task.fold)],
                          'history_experiment_id': item.Index, 'history_context': item.context,
                          'history_condition': item.condition, 'history_original_study': item.study_id,
                          'history_weight': float(weight),
                          'eligible_query_upstream_source_train': True,
                          'current_task_or_current_context_replicate': False,
                          'use': 'known-other-context source-history inference; not Public supervised target',
                          'source_bank_provenance': item.provenance})
    summary = pd.DataFrame(summaries).set_index('task_id')
    task_lookup = {task: row for row, task in enumerate(tasks.task_id)}
    # Preserve the original float32 full-array reduction/layout. Selecting rows
    # first copies F-order predictions into C-order and changes their norm sum.
    full_priors = np.zeros((len(task_rows), 2840), dtype=np.float64)
    full_priors[[task_rows[task] for task in tasks.task_id]] = priors
    frames = []
    for upstream in MODELS:
        frame = base[base.upstream.eq(upstream)].copy().reset_index(drop=True)
        rows = np.asarray([task_lookup[task] for task in frame.task_id], dtype=int)
        source_rows = np.asarray([task_rows[task] for task in frame.task_id], dtype=int)
        predicted = np.asarray(predictions[upstream][source_rows])
        prior = priors[rows]
        frame['prior_magnitude'] = np.sqrt(np.mean(prior**2, axis=1))
        frame['prediction_prior_rmse'] = rmse_rows(predicted, prior)
        frame['prediction_prior_cosine'] = cosine_rows(predictions[upstream], full_priors)[source_rows]
        for column in ['prior_uncertainty', 'effective_sources', 'history_conflict', 'log_history_support']:
            frame[column] = frame.task_id.map(summary[column])
        frames.append(frame)
    feature = pd.concat(frames, ignore_index=True)
    if not np.isfinite(feature[P + PUBLIC].to_numpy(float)).all():
        raise RuntimeError('Manual Public feature reconstruction must be finite')
    feature_path = out / f'FEATURES_PUBLIC_v{version}.parquet'
    feature.to_parquet(feature_path, index=False)
    feature_path.chmod(0o444)
    prior_path = out / f'PUBLIC_PRIORS_v{version}.npy'
    np.save(prior_path, priors, allow_pickle=False)
    prior_path.chmod(0o444)
    write_csv(out / f'PUBLIC_HISTORY_EDGES_v{version}.csv', pd.DataFrame(edges), immutable=True)
    return feature, priors, manifest, feature_path


def read_source_errors(feature, allowed_roles):
    """Actual DEV outcome column; only allowed records enter returned fit frame."""
    raw = pd.read_parquet(SOURCE, columns=['upstream', 'task_id', 'true_error_rmse'])
    raw['prediction_id'] = frame_records(raw)
    if raw.prediction_id.duplicated().any():
        raise RuntimeError('Source predictor/task outcomes are not unique')
    lookup = raw.set_index('prediction_id').true_error_rmse
    fit = feature[feature.replay_role.isin(allowed_roles)].copy().reset_index(drop=True)
    fit['true_error_rmse'] = frame_records(fit).map(lookup)
    if not np.isfinite(fit.true_error_rmse).all():
        raise RuntimeError('Allowed real Source outcomes must be finite')
    if not set(fit.replay_role).issubset(allowed_roles):
        raise RuntimeError('Anchor/new-gate outcomes entered a fit scope')
    return fit


def initial_shared_oof(initial):
    risk = np.full(len(initial), np.nan)
    audits = []
    for heldout in (2, 3):
        use = initial.fold.eq(heldout).to_numpy()
        fit = initial.loc[~use].reset_index(drop=True)
        query = initial.loc[use]
        if set(fit.gene) & set(query.gene):
            raise RuntimeError('Initial Error shared-risk OOF has overlapping genes')
        labels, audit = rank_labels(fit, f'initial-error-shared-oof/fold{heldout}')
        audits.extend(audit)
        risk[use] = fit_risk(fit, labels, P + PUBLIC, 'hgb', SEED).predict(query)
    if not np.isfinite(risk).all():
        raise RuntimeError('Initial Error shared-risk OOF predictions incomplete')
    return risk, audits


def error_items(frame, shared, cdfs, cdf_binding, stage):
    ranks = transform_errors(frame, cdfs)
    items = []
    source_sha = sha(SOURCE)
    for row, score, rank in zip(frame.itertuples(index=False), shared, ranks):
        items.append(ErrorMemoryItem(
            upstream_model_id=str(row.upstream), model_version=str(row.model_version),
            prediction_id=str(row.upstream) + '::' + str(row.task_id),
            biological_task_key=biological_task_key(str(row.dataset_id), str(row.target),
                'genetic_single_gene', str(row.gene), str(row.condition), CONTRACT),
            perturbation_cluster_key=perturbation_cluster_key(str(row.dataset_id),
                'genetic_single_gene', str(row.gene)),
            shared_risk=float(score), realised_error=float(row.true_error_rmse),
            error_rank=float(rank), shared_risk_residual=float(rank - score),
            feedback_timestamp=utc(), is_oof_or_heldout=True,
            provenance=(f'Retrospective real Source DEV replay; existing {row.model_version} composite OOF family, '
                        f'not an individual checkpoint; common2840 contract; {stage}; '
                        f'Source cache SHA {source_sha}; error_rank annotation uses frozen initial CDF SHA {cdf_binding}')))
    return items


def append_errors(registry, items, stage):
    rows = []
    for upstream in MODELS:
        selected = [item for item in items if item.upstream_model_id == upstream]
        versions = {item.model_version for item in selected}
        if len(versions) != 1:
            raise RuntimeError('Actual upstream-version isolation failed')
        rows.append(registry.append(selected, f'{stage}/{upstream}'))
    return rows


def fit_from_error_memory(features, registry, allowed_roles):
    expected = features[features.replay_role.isin(allowed_roles)].copy().reset_index(drop=True)
    pieces = []
    for upstream in MODELS:
        version = expected[expected.upstream.eq(upstream)].model_version.unique()
        if len(version) != 1:
            raise RuntimeError('One exact existing model version required')
        pieces.append(registry.load(upstream, str(version[0])))
    saved = pd.concat(pieces, ignore_index=True)
    expected['prediction_id'] = frame_records(expected)
    if set(expected.prediction_id) != set(saved.prediction_id) or saved.prediction_id.duplicated().any():
        raise RuntimeError('Persistent Error records differ from frozen fit partition')
    fit = expected.merge(saved[['prediction_id', 'realised_error']], on='prediction_id', validate='one_to_one')
    fit['true_error_rmse'] = fit.realised_error
    if set(fit.fold) & {0, 1} or not np.isfinite(fit.true_error_rmse).all():
        raise RuntimeError('Nonfinite or anchor/new-gate Error labels entered fit')
    if not set(fit.replay_role).issubset(allowed_roles):
        raise RuntimeError('Forbidden Source label role')
    labels, audits = rank_labels(fit, 'actual-replay/' + '+'.join(sorted(allowed_roles)))
    if not np.isfinite(labels).all():
        raise RuntimeError('Allowed Source CDF is incomplete')
    model = fit_risk(fit, labels, P + PUBLIC, 'hgb', SEED)
    return model, fit, labels, audits


def save_model(out, version, model, features, feature_path, public_manifest, fit, cdfs):
    bundle = {'schema': 'safeconf_source_dev_continual_model_v1', 'risk_version': version,
              'model': model, 'public_version': version, 'public_manifest': public_manifest,
              'feature_binding': bind(feature_path), 'output_contract_id': CONTRACT,
              'feature_columns': P + PUBLIC, 'training_record_ids_hash': ids_hash(frame_records(fit)),
              'training_gene_clusters_hash': ids_hash(fit.gene.unique()),
              'training_folds': sorted(map(int, fit.fold.unique())), 'cdf_objects': cdfs,
              'seed': SEED, 'no_anchor_or_new_gate_labels_in_fit': True,
              'Public_biological_learner_fitted': False}
    path = out / f'MODEL_v{version}.joblib'
    joblib.dump(bundle, path)
    path.chmod(0o444)
    predictions = model.predict(features)
    if predictions.shape != (len(features),) or not np.isfinite(predictions).all():
        raise RuntimeError('All frozen prediction records must be finite and aligned; no survivor filtering')
    loaded = joblib.load(path)
    again = loaded['model'].predict(features)
    if predictions.tobytes() != again.tobytes():
        raise RuntimeError('Saved model predictions changed on reload')
    return path, predictions


def serving_prediction(serving, component):
    artifact = serving.current(component)
    bundle = joblib.load(artifact)
    feature = Path(bundle['feature_binding']['path'])
    if sha(feature) != bundle['feature_binding']['sha256']:
        raise RuntimeError('Serving model feature-version binding changed')
    frame = pd.read_parquet(feature)
    predictions = bundle['model'].predict(frame)
    if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
        raise RuntimeError('All serving prediction records must be finite and aligned')
    return bundle, predictions


def finite_release_gate(old_metrics, new_metrics):
    """Fail closed before the original manager's NaN-sensitive comparisons."""
    required = ['utility20', 'aurc', 'high_risk_miss_rate']
    expected = {(model, context) for model in MODELS for context in CONTEXTS}
    metric_columns = [f'{column}_v{v}' for column in required for v in (1, 2)]
    for table, role in [(old_metrics, 'OLD_ANCHOR'), (new_metrics, 'NEW_TASK_GATE')]:
        if (not set(['upstream', 'context', 'role', 'n_tasks'] + metric_columns).issubset(table.columns)
            or len(table) != len(expected) or table.duplicated(['upstream', 'context']).any()
            or set(zip(table.upstream, table.context)) != expected or not table.role.eq(role).all()):
            return {'passes': False, 'reason': 'MISSING_OR_DUPLICATE_PLANNED_STRATA_FAIL_CLOSED',
                    'all_planned_strata_finite': False}
    values = []
    for table in (old_metrics, new_metrics):
        values.extend(table[[f'{column}_v{v}' for column in required for v in (1, 2)]].to_numpy(float).ravel())
    finite = bool(np.isfinite(values).all())
    valid = bool(finite and len(old_metrics) == 8 and len(new_metrics) == 8
                 and (old_metrics.n_tasks >= 20).all() and (new_metrics.n_tasks >= 20).all())
    if not valid:
        return {'passes': False, 'reason': 'NONFINITE_OR_INCOMPLETE_DEV_GATE_FAIL_CLOSED',
                'all_planned_strata_finite': finite, 'new_task_strata': len(new_metrics),
                'old_anchor_strata': len(old_metrics)}
    new_u20 = float((new_metrics.utility20_v2 - new_metrics.utility20_v1).mean())
    old_base = float(old_metrics.aurc_v1.mean())
    if old_base <= 0:
        return {'passes': False, 'reason': 'INVALID_OLD_ANCHOR_AURC_DENOMINATOR',
                'all_planned_strata_finite': True}
    degradation = float((old_metrics.aurc_v2.mean() - old_base) / old_base)
    miss_old = float((old_metrics.high_risk_miss_rate_v2 - old_metrics.high_risk_miss_rate_v1).mean())
    miss_new = float((new_metrics.high_risk_miss_rate_v2 - new_metrics.high_risk_miss_rate_v1).mean())
    fraction = float((new_metrics.utility20_v2 >= new_metrics.utility20_v1).mean())
    measured = ReleaseMetrics(new_u20, degradation, max(miss_old, miss_new), fraction)
    if not np.isfinite(list(asdict(measured).values())).all():
        raise RuntimeError('Nonfinite metrics reached the release manager')
    decision = ContinualUpdateManager.release(measured)
    return {'passes': bool(decision.trigger), 'reasons': list(decision.reasons),
            'all_planned_strata_finite': True, 'release_metrics': asdict(measured),
            'old_anchor_miss_rate_delta': miss_old, 'new_task_miss_rate_delta': miss_new,
            'old_anchor_strata': 8, 'new_task_strata': 8,
            'macro_unit': 'equal upstream-by-context strata',
            'gate_used_only_DEV_anchors': True, 'nonfinite_fail_closed': True}


def measure_roles(features, scores1, scores2):
    rows = []
    for role in ('OLD_ANCHOR', 'NEW_TASK_GATE'):
        truth = read_source_errors(features, {role})
        truth_lookup = truth.set_index('prediction_id').true_error_rmse if 'prediction_id' in truth else pd.Series(
            truth.true_error_rmse.to_numpy(), index=frame_records(truth))
        for upstream in MODELS:
            for context in CONTEXTS:
                use = features.replay_role.eq(role) & features.upstream.eq(upstream) & features.target.eq(context)
                frame = features.loc[use].copy()
                frame['true_error_rmse'] = frame_records(frame).map(truth_lookup)
                if not np.isfinite(frame.true_error_rmse).all():
                    raise RuntimeError('Frozen DEV gate has missing actual truth')
                a, b = metrics(frame, scores1[use.to_numpy()]), metrics(frame, scores2[use.to_numpy()])
                row = {'role': role, 'upstream': upstream, 'context': context,
                       'n_tasks': len(frame), 'n_gene_clusters': frame.gene.nunique(),
                       'task_ids_hash': ids_hash(frame.task_id), 'gene_clusters_hash': ids_hash(frame.gene.unique())}
                row.update({f'{key}_v1': value for key, value in a.items() if key not in ['n_tasks', 'n_planned']})
                row.update({f'{key}_v2': value for key, value in b.items() if key not in ['n_tasks', 'n_planned']})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--report-root', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    out, report = args.runtime_root.resolve(), args.report_root.resolve()
    if out.exists() or report.exists():
        raise FileExistsError('Completed/partial replay cannot be overwritten; use a new version')
    if not out.is_relative_to(Path('/home/yyf/runtime_artifacts')) or not report.is_relative_to(DOC_BASE / 'continual_runtime_replay'):
        raise RuntimeError('Only isolated authorized replay output roots are allowed')
    out.mkdir(parents=True)
    report.mkdir(parents=True)
    began = time.monotonic()
    state = {'status': 'STARTING', 'pid': os.getpid(), 'started_utc': utc(),
             'role': 'RETROSPECTIVE_REAL_SOURCE_DEV_OPERATIONAL_REPLAY',
             'Orion_access': False, 'McFaline_access': False, 'new_upstream_calls': 0}
    def progress(phase, **extra):
        state.update(status=phase, updated_utc=utc(), **extra)
        temporary = out / '.status.tmp'
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'STATUS.json')
        print(json.dumps({'phase': phase, **extra}), flush=True)
    try:
        dependencies = [Path(__file__), ROOT / 'tools/safeconf_continual/versioned_runtime.py',
                        ROOT / 'tools/safeconf_continual/memory.py', ROOT / 'tools/safeconf_continual/update.py',
                        ROOT / 'tools/safeconf_continual/research.py', ROOT / 'tools/safeconf_continual/learners.py',
                        ROOT / 'tools/safeconf_continual/contracts.py',
                        ROOT / 'tools/scripts/run_dual_memory_txpert_public_biology.py',
                        ROOT / 'tools/safeconf_continual/orion_linear.R']
        inputs = [SOURCE, COMMON / 'GENE_IDS.json', COMMON / 'risk_cache/TX_TASK_SPLIT.csv', PROVENANCE,
                  ORIGINAL_PUBLIC / 'public_memory.parquet', ORIGINAL_PUBLIC / 'eligibility.parquet',
                  ORIGINAL_PUBLIC / 'manifest.json', ORIGINAL_PUBLIC / 'gene_ids.json',
                  ORIGINAL_PUBLIC / 'effect_vectors.npy', ORIGINAL_PUBLIC / 'control_vectors.npy',
                  COMMON / 'SOURCE_PUBLIC_EFFECTS.npy', COMMON / 'SOURCE_PUBLIC_CONTROLS.npy',
                  COMMON / 'SOURCE_TRUE_EFFECTS.npy'] + [COMMON / f'SOURCE_{name}_PREDICTED_EFFECTS.npy' for name in MODELS]
        registration = {'schema': 'safeconf_real_dev_continual_replay_registration_v1',
                        'registered_utc': utc(), 'role': state['role'], 'output_contract_id': CONTRACT,
                        'gene_axis': 2840, 'seed': SEED, 'features': P + PUBLIC,
                        'hgb': {'max_iter': 200, 'learning_rate': .05, 'max_depth': 3,
                                'min_samples_leaf': 20, 'l2_regularization': 10},
                        'risk_initial_train_folds': [2, 3], 'risk_added_train_fold': 4,
                        'old_anchor_fold': 0, 'new_task_gate_fold': 1,
                        'Public_initial_study': 'Replogle_2022', 'Public_added_study': 'Nadig_2025',
                        'Public_inference': 'existing Manual cell-count-weighted prior; version-specific recomputation',
                        'Public_biological_learner_fitted': False,
                        'error_update_component': 'shared Source risk core; no target C adapter or new external result',
                        'history_contract': 'query-upstream source-train same-perturbation other-context history, not risk-label outer-train-only history',
                        'current_task_and_all_current_context_replicates_excluded': True,
                        'finer_guide_or_plate_disjoint_claim': False,
                        'cohort_rule': 'identity-only common known-history under initial Public version; freeze before fits',
                        'gate': {'new_task_macro_delta_U20_min': -.005, 'old_anchor_AURC_relative_max': .05,
                                 'max_anchor_newtask_macro_miss_delta_max': .02,
                                 'newtask_nonnegative_strata_fraction_min': .60,
                                 'strata': 'four contexts by two existing upstream families',
                                 'all_strata_min_tasks': 20, 'nonfinite': 'REJECT_FAIL_CLOSED'},
                        'rollback_drill': 'deterministic administrative restoration; not an observed quality failure',
                        'anchor_truth_used_for_fit_or_CDF': False, 'new_gate_truth_used_for_fit_or_CDF': False,
                        'Orion_access': False, 'McFaline_access': False, 'parameter_search': False,
                        'input_bindings': [bind(path) for path in inputs],
                        'code_bindings': [bind(path) for path in dependencies]}
        write_json(report / 'REGISTRATION.json', registration, immutable=True)
        progress('REGISTERED_FIXED_DEV_REPLAY')

        genes = json.loads((COMMON / 'GENE_IDS.json').read_text())
        if len(genes) != 2840 or len(set(genes)) != 2840:
            raise RuntimeError('Frozen common gene axis differs')
        original_manifest = json.loads((ORIGINAL_PUBLIC / 'manifest.json').read_text())
        for name, key in [('public_memory.parquet', 'table_sha256'), ('gene_ids.json', 'gene_ids_sha256'),
                          ('effect_vectors.npy', 'effect_vectors_sha256'), ('control_vectors.npy', 'control_vectors_sha256')]:
            if sha(ORIGINAL_PUBLIC / name) != original_manifest[key]:
                raise RuntimeError('Existing real Public metadata/gene integrity failed')
        if sha(ORIGINAL_PUBLIC / original_manifest['eligibility']['path']) != original_manifest['eligibility']['sha256']:
            raise RuntimeError('Existing query-upstream eligibility SHA differs')
        memory = pd.read_parquet(ORIGINAL_PUBLIC / 'public_memory.parquet')
        eligibility = pd.read_parquet(ORIGINAL_PUBLIC / 'eligibility.parquet')
        tasks_all = pd.read_csv(COMMON / 'risk_cache/TX_TASK_SPLIT.csv')
        tasks_all['condition'] = tasks_all.task_id.str.split('::').str[1]
        tasks_all['replay_role'] = tasks_all.fold.map(ROLE)
        if tasks_all.gene.groupby(tasks_all.gene).size().empty or tasks_all.groupby('gene').fold.nunique().max() != 1:
            raise RuntimeError('Source gene folds are not indivisible')
        if set(tasks_all.fold) != set(range(5)) or tasks_all.task_id.duplicated().any():
            raise RuntimeError('Frozen five-fold Source task identities differ')
        study = pd.read_csv(PROVENANCE).set_index('context').original_study.to_dict()
        if study != STUDY:
            raise RuntimeError('Existing actual study provenance differs')
        initial_ids = set(memory.loc[memory.context.isin(['K562', 'RPE1']), 'experiment_id'])
        initial_eligible = eligibility[eligibility.public_experiment_id.isin(initial_ids)].groupby(
            ['target_context', 'condition']).size().to_dict()
        tasks_all['initial_known_history'] = [initial_eligible.get((str(row.target), str(row.condition)), 0) > 0
                                               for row in tasks_all.itertuples()]
        write_csv(report / 'FROZEN_TASK_ROLE_SCOPE.csv', tasks_all, immutable=True)
        tasks = tasks_all[tasks_all.initial_known_history].reset_index(drop=True)
        task_rows = {task: row for row, task in enumerate(tasks_all.task_id)}
        source_columns = ['upstream', 'model_version', 'dataset_id', 'output_contract_id',
                          'task_id', 'condition', 'gene', 'target', 'fold'] + P
        base_all = pd.read_parquet(SOURCE, columns=source_columns)
        if len(base_all) != 3616 or base_all.duplicated(['upstream', 'task_id']).any():
            raise RuntimeError('Actual Source predictor/task identity differs')
        if set(base_all.output_contract_id) != {CONTRACT} or set(base_all.upstream) != set(MODELS):
            raise RuntimeError('Source model or common-axis contract differs')
        if set(base_all.dataset_id) != {'TxPert_E201'}:
            raise RuntimeError('Actual Source dataset identity differs')
        base = base_all[base_all.task_id.isin(tasks.task_id)].copy().reset_index(drop=True)
        base['replay_role'] = base.fold.map(ROLE)
        if len(base) != 2 * len(tasks) or not np.isfinite(base[P].to_numpy(float)).all():
            raise RuntimeError('Fixed known-history Source features are incomplete')
        for name in MODELS:
            part = base[base.upstream.eq(name)]
            aligned = tasks_all.set_index('task_id').loc[part.task_id]
            if not np.array_equal(part.fold.to_numpy(int), aligned.fold.to_numpy(int)):
                raise RuntimeError('Source cache and prediction-array identity folds differ')
            if (part.model_version.nunique() != 1
                or not np.array_equal(part[['gene', 'target', 'condition']].to_numpy(str),
                                      aligned[['gene', 'target', 'condition']].to_numpy(str))):
                raise RuntimeError('Source cache gene/context/condition/model-version identity differs')
        write_json(report / 'INPUT_IDENTITY_LEDGER.json', {
            'source_original_predictor_task_records': len(base_all), 'source_original_tasks': len(tasks_all),
            'known_history_tasks': len(tasks), 'known_history_predictor_task_records': len(base),
            'known_history_gene_clusters': tasks.gene.nunique(),
            'metadata_only_excluded_no_initial_history_tasks': int((~tasks_all.initial_known_history).sum()),
            'source_models': [{'upstream': name, 'cache_model_version': str(base[base.upstream.eq(name)].model_version.iloc[0]),
                               'scope': 'existing composite official-family OOF outputs; not individual checkpoints'} for name in MODELS],
            'biological_contexts': list(CONTEXTS), 'biological_studies': sorted(set(STUDY.values())),
            'role_record_counts': base.groupby('replay_role').size().to_dict(),
            'role_gene_counts': base.groupby('replay_role').gene.nunique().to_dict(),
            'new_upstream_models_or_data': 0}, immutable=True)
        effects = np.load(COMMON / 'SOURCE_PUBLIC_EFFECTS.npy', mmap_mode='r')
        controls = np.load(COMMON / 'SOURCE_PUBLIC_CONTROLS.npy', mmap_mode='r')
        if effects.shape != (len(memory), 2840) or controls.shape != effects.shape:
            raise RuntimeError('Registered Public projection shape differs')
        native_genes = json.loads((ORIGINAL_PUBLIC / 'gene_ids.json').read_text())['gene_ids']
        native_columns = np.asarray([{gene: row for row, gene in enumerate(native_genes)}[gene] for gene in genes], int)
        for name, projected in [('effect_vectors.npy', effects), ('control_vectors.npy', controls)]:
            native = np.load(ORIGINAL_PUBLIC / name, mmap_mode='r')
            for start in range(0, len(memory), 128):
                if not np.array_equal(native[start:start + 128, native_columns], projected[start:start + 128]):
                    raise RuntimeError('Common2840 Public vectors differ from exact real native-bank projection')
        predictions = {name: np.load(COMMON / f'SOURCE_{name}_PREDICTED_EFFECTS.npy', mmap_mode='r') for name in MODELS}
        if any(array.shape != (len(tasks_all), 2840) for array in predictions.values()):
            raise RuntimeError('Existing upstream prediction-array shape differs')
        public = VersionedPublicRuntime(out / 'public_memory', genes)
        initial_mask = memory.context.isin(['K562', 'RPE1']).to_numpy()
        def public_items(mask):
            items = []
            for row_number, record in enumerate(memory.loc[mask].to_dict('records')):
                original_study = record['study_id']
                record.update(study_id=STUDY[str(record['context'])], effect_vector_row=row_number,
                              gene_space_id='Source_common2840_fixed_v1', effect_contract_id=CONTRACT,
                              provenance=(str(record['provenance']) + '; original aggregate study alias=' + str(original_study)
                                          + '; original study crosswalk SHA=' + sha(PROVENANCE)
                                          + '; retrospective isolated DEV replay; no new biological observation'))
                items.append(PublicMemoryItem(**record))
            return items
        v1_eligibility = eligibility[eligibility.public_experiment_id.isin(initial_ids)].copy()
        public.create(public_items(initial_mask), np.asarray(effects[initial_mask]), np.asarray(controls[initial_mask]),
                      {'source_bank': bind(ORIGINAL_PUBLIC / 'manifest.json'), 'projection_gene_ids': bind(COMMON / 'GENE_IDS.json'),
                       'study_provenance': bind(PROVENANCE), 'phase': 'Replogle_2022_initial'}, v1_eligibility)
        features1, priors1, pub1, feature_path1 = rebuild_features(base, tasks, task_rows, predictions, public,
                                                               eligibility, 1, out)
        progress('REAL_PUBLIC_v1_CREATED_FEATURES_REBUILT', public_items_v1=int(initial_mask.sum()))
        initial = read_source_errors(features1, {'TRAIN_INITIAL'})
        oof_risk, oof_audits = initial_shared_oof(initial)
        initial_cdfs = cdf_objects(initial)
        cdf1_path = out / 'CDF_INITIAL.joblib'
        joblib.dump(initial_cdfs, cdf1_path)
        cdf1_path.chmod(0o444)
        error = VersionedErrorRuntime(out / 'error_memory', CONTRACT)
        error_v1 = append_errors(error, error_items(initial, oof_risk, initial_cdfs, sha(cdf1_path),
                                'initial shared-risk predictions are gene-disjoint two-fold OOF within folds2/3'), 'initial')
        model1, train1, labels1, audit1 = fit_from_error_memory(features1, error, {'TRAIN_INITIAL'})
        model1_path, score1 = save_model(out, 1, model1, features1, feature_path1, pub1, train1, initial_cdfs)
        np.save(out / 'PRE_GATE_RISK_v1.npy', score1, allow_pickle=False)
        (out / 'PRE_GATE_RISK_v1.npy').chmod(0o444)
        split_hash = sha(report / 'FROZEN_TASK_ROLE_SCOPE.csv')
        feature_schema_hash = hashlib.sha256('\n'.join(P + PUBLIC).encode()).hexdigest()
        serving = ServingModelRegistry(out / 'serving_models')
        component = 'Source_Manual_HGB_shared_risk'
        serving.register(component, 'risk-v1', model1_path, split_hash, feature_schema_hash, status='RELEASED')
        serving.publish(component, 'risk-v1', 'initial isolated real DEV baseline, no external deployment')
        _, resolved1 = serving_prediction(serving, component)
        if resolved1.tobytes() != score1.tobytes():
            raise RuntimeError('Actual initial serving artifact differs')
        progress('REAL_ERROR_v1_AND_INITIAL_MODEL_PUBLISHED', error_rows_initial=len(train1))

        new_mask = ~initial_mask
        old_clusters = set(memory.loc[initial_mask, 'perturbation_target'].astype(str))
        new_clusters = set(memory.loc[new_mask, 'perturbation_target'].astype(str)) - old_clusters
        public_trigger = ContinualUpdateManager.public_update_due(len(old_clusters), len(new_clusters), 1)
        if not public_trigger.trigger:
            raise RuntimeError('Actual new-study Public trigger did not fire')
        public.append(public_items(new_mask), np.asarray(effects[new_mask]), np.asarray(controls[new_mask]),
                      {'source_bank': bind(ORIGINAL_PUBLIC / 'manifest.json'), 'study_provenance': bind(PROVENANCE),
                       'phase': 'append_Nadig_2025_actual_existing_records', 'new_studies': 1}, eligibility)
        features2, priors2, pub2, feature_path2 = rebuild_features(base, tasks, task_rows, predictions, public,
                                                               eligibility, 2, out)
        if not frame_records(features1).equals(frame_records(features2)):
            raise RuntimeError('Public update changed the frozen comparison record order')
        previous = pd.read_parquet(SOURCE, columns=['upstream', 'task_id'] + PUBLIC)
        previous['prediction_id'] = frame_records(previous)
        rebuilt = features2.assign(prediction_id=frame_records(features2)).merge(previous, on='prediction_id',
                                suffixes=('_reconstructed', '_original'), validate='one_to_one')
        reconstruction = {column: float(np.max(np.abs(rebuilt[column + '_reconstructed'] - rebuilt[column + '_original'])))
                          for column in PUBLIC}
        if max(reconstruction.values()) > 1e-8:
            raise RuntimeError('Full-bank Manual features do not reproduce original fixed formula')
        added = read_source_errors(features1, {'TRAIN_ADDED'})
        prior_query = model1.predict(added)
        error_v2 = append_errors(error, error_items(added, prior_query, initial_cdfs, sha(cdf1_path),
                              'new fold4 shared-risk scored by frozen initial risk-v1, never trained on fold4'), 'added')
        previous_clusters = train1.gene.nunique()
        current_clusters = pd.concat([train1, added]).gene.nunique()
        error_trigger = ContinualUpdateManager.error_update_due(previous_clusters, current_clusters)
        if not error_trigger.trigger:
            raise RuntimeError('Actual added Error cluster trigger did not fire')
        model2, train2, labels2, audit2 = fit_from_error_memory(features2, error, {'TRAIN_INITIAL', 'TRAIN_ADDED'})
        cdf2 = cdf_objects(train2)
        model2_path, score2 = save_model(out, 2, model2, features2, feature_path2, pub2, train2, cdf2)
        np.save(out / 'PRE_GATE_RISK_v2.npy', score2, allow_pickle=False)
        (out / 'PRE_GATE_RISK_v2.npy').chmod(0o444)
        frozen = features1[['upstream', 'model_version', 'task_id', 'gene', 'target', 'fold', 'replay_role']].copy()
        frozen['risk_v1'], frozen['risk_v2'] = score1, score2
        write_csv(out / 'ALL_PRE_GATE_RISK_PREDICTIONS.csv', frozen, immutable=True)
        write_json(report / 'PRE_GATE_MODEL_PREDICTION_FREEZE.json', {
            'frozen_utc': utc(), 'models': [bind(model1_path), bind(model2_path)],
            'predictions': bind(out / 'ALL_PRE_GATE_RISK_PREDICTIONS.csv'),
            'all_model_params_and_predictions_fixed_before_DEV_gate_evaluation': True,
            'old_anchor_and_new_gate_labels_in_CDF_or_fit': 0,
            'public_version_specific_features': [bind(feature_path1), bind(feature_path2)],
            'full_bank_original_Manual_feature_max_abs_difference': reconstruction,
            'Public_feature_changes_due_to_real_append': int(np.any(features1[PUBLIC].to_numpy() != features2[PUBLIC].to_numpy(), axis=1).sum()),
            'risk_train_initial_records': len(train1), 'risk_train_updated_records': len(train2),
            'risk_train_initial_gene_clusters': previous_clusters, 'risk_train_updated_gene_clusters': current_clusters}, immutable=True)
        write_json(report / 'REAL_INGESTION_TRIGGER_RECEIPT.json', {
            'actual_Public_create_then_append': True, 'Public_current_version': 2,
            'actual_Source_Error_append_batches_per_upstream': 2, 'immutable_Error_revision_snapshots': True,
            'Public_trigger': {'trigger': bool(public_trigger.trigger), 'reasons': list(public_trigger.reasons),
                               'previous_clusters': len(old_clusters), 'new_clusters': len(new_clusters), 'new_studies': 1},
            'Error_trigger': {'trigger': bool(error_trigger.trigger), 'reasons': list(error_trigger.reasons),
                              'previous_clusters': previous_clusters, 'current_clusters': current_clusters},
            'Error_initial_manifests': error_v1, 'Error_updated_manifests': error_v2,
            'Public_v1_manifest': pub1, 'Public_v2_manifest': pub2}, immutable=True)
        progress('ACTUAL_ADDED_RECORDS_TRIGGERED_REFIT_BOTH_MODELS_FROZEN', error_rows_updated=len(train2))

        measured = measure_roles(features1, score1, score2)
        write_csv(report / 'REAL_DEV_ANCHOR_NEW_TASK_METRICS.csv', measured, immutable=True)
        old_metrics = measured[measured.role.eq('OLD_ANCHOR')].copy()
        new_metrics = measured[measured.role.eq('NEW_TASK_GATE')].copy()
        gate = finite_release_gate(old_metrics, new_metrics)
        # Prove the known NaN comparison weakness is closed in the actual wrapper.
        nan_fixture = new_metrics.copy()
        nan_fixture.iloc[0, nan_fixture.columns.get_loc('utility20_v2')] = np.nan
        nan_gate = finite_release_gate(old_metrics, nan_fixture)
        if nan_gate['passes']:
            raise RuntimeError('NaN release guard unexpectedly accepted')
        write_json(report / 'FINITE_GATE_PROOF.json', {'generated_nonfinite_gate_rejected': True,
                   'nonfinite_guard_reason': nan_gate['reason'], 'actual_gate_all_planned_strata_finite': gate['all_planned_strata_finite']}, immutable=True)
        release_status = 'RELEASED' if gate['passes'] else 'REJECTED'
        serving.register(component, 'risk-v2', model2_path, split_hash, feature_schema_hash,
                         parent_version='risk-v1', status=release_status)
        if gate['passes']:
            serving.publish(component, 'risk-v2', 'actual predeclared finite DEV gate passed; isolated replay only')
            expected_score = score2
            selected_version = 'risk-v2'
        else:
            expected_score = score1
            selected_version = 'risk-v1'
        selected_bundle, selected_score = serving_prediction(serving, component)
        if selected_score.tobytes() != expected_score.tobytes():
            raise RuntimeError('Publish/retain did not serve the gate-selected fixed model')
        write_json(report / 'ACTUAL_RELEASE_OR_RETAIN_DECISION.json', {
            **gate, 'candidate_status': release_status, 'serving_version': selected_version,
            'serving_Public_version': int(selected_bundle['public_version']), 'ingestion_Public_CURRENT_version': 2,
            'failed_candidate_published': False, 'no_second_candidate_or_parameter_search': True,
            'added_data_preserved_if_rejected': True, 'new_fits_after_gate': 0}, immutable=True)
        progress('DEV_QUALITY_GATE_APPLIED_PUBLISH_OR_RETAIN', candidate_status=release_status)

        # Separate deterministic administrative drill. Never publish rejected v2.
        if not gate['passes']:
            serving.register(component, 'risk-v1-administrative-marker', model1_path, split_hash,
                             feature_schema_hash, parent_version='risk-v1', status='RELEASED')
            serving.publish(component, 'risk-v1-administrative-marker',
                            'administrative drill marker; SAME previously approved real model bytes, no new fit or quality claim')
        before_drill_artifact = bind(serving.current(component))
        serving.rollback(component, 'risk-v1', 'deterministic administrative rollback drill; not a performance failure')
        restored_bundle, restored_score = serving_prediction(serving, component)
        if restored_score.tobytes() != score1.tobytes() or sha(serving.current(component)) != sha(model1_path):
            raise RuntimeError('Actual artifact rollback did not restore initial predictions/bytes')
        public.rollback_to(1, 'deterministic administrative Public pointer rollback drill; retained added version2 data')
        rebuilt1, reconstructed_prior1, _, _ = rebuild_features(base, tasks, task_rows, predictions, public,
                                                               eligibility, 1, out / 'rollback_reconstruction')
        if reconstructed_prior1.tobytes() != priors1.tobytes() or model1.predict(rebuilt1).tobytes() != score1.tobytes():
            raise RuntimeError('Actual Public pointer rollback failed feature/prediction restoration')
        public.activate(2, 'restore accepted ingestion version2 after administrative drill; all added data retained')
        if gate['passes']:
            serving.publish(component, 'risk-v2', 'restore approved gate-selected serving version after administrative drill')
        # If the candidate was rejected, serving remains initial artifact after rollback.
        final_bundle, final_score = serving_prediction(serving, component)
        if final_score.tobytes() != expected_score.tobytes():
            raise RuntimeError('Administrative drill changed the actual quality-gate outcome')
        write_json(report / 'ACTUAL_ROLLBACK_RELOAD_PROOF.json', {
            'kind': 'DETERMINISTIC_ADMINISTRATIVE_DRILL_NOT_OBSERVED_QUALITY_FAILURE',
            'before_drill_serving_artifact': before_drill_artifact,
            'rolled_back_artifact': bind(model1_path), 'rollback_uses_actual_registry_and_CURRENT_pointers': True,
            'restored_model_bytes_sha_match': True, 'restored_predictions_byte_identical': True,
            'Public_v1_reconstructed_priors_byte_identical': True,
            'Public_v1_reconstructed_risk_predictions_byte_identical': True,
            'final_ingestion_Public_version': 2,
            'final_serving_Public_version': int(final_bundle['public_version']),
            'final_gate_selected_predictions_byte_identical': True,
            'rejected_candidate_ever_published': False, 'added_Public_Error_data_still_persisted': True}, immutable=True)

        # BIOtarget current-task means are used only for post-freeze DEV evaluation.
        truth = np.load(COMMON / 'SOURCE_TRUE_EFFECTS.npy', mmap_mode='r')
        if truth.shape != (len(tasks_all), 2840):
            raise RuntimeError('Source evaluation biology axis differs')
        source_truth_alignment = {}
        raw_errors = pd.read_parquet(SOURCE, columns=['upstream', 'task_id', 'true_error_rmse'])
        for name in MODELS:
            actual_errors = rmse_rows(predictions[name], truth)
            error_by_task = dict(zip(tasks_all.task_id, actual_errors))
            part = raw_errors[raw_errors.upstream.eq(name)]
            difference = float(np.max(np.abs(part.task_id.map(error_by_task) - part.true_error_rmse)))
            if difference > 1e-6:
                raise RuntimeError('Actual Source prediction/truth cache alignment differs')
            source_truth_alignment[name] = difference
        biology = []
        for role in ('OLD_ANCHOR', 'NEW_TASK_GATE'):
            for context in CONTEXTS:
                use = tasks.replay_role.eq(role) & tasks.target.eq(context)
                rows = np.flatnonzero(use.to_numpy())
                true_rows = np.asarray([task_rows[task] for task in tasks.loc[use, 'task_id']], int)
                observed = np.asarray(truth[true_rows], float)
                first = rmse_rows(priors1[rows], observed)
                second = rmse_rows(priors2[rows], observed)
                biology.append({'role': role, 'context': context, 'n_tasks': len(rows),
                                'Public_prior_RMSE_v1': float(first.mean()),
                                'Public_prior_RMSE_v2': float(second.mean()),
                                'delta_RMSE_v2_minus_v1': float((second - first).mean()),
                                'target_biology_used_for_Public_learner_fit': False})
        write_csv(report / 'REAL_DEV_PUBLIC_PRIOR_GROWTH_METRICS.csv', pd.DataFrame(biology), immutable=True)
        write_csv(report / 'SOURCE_CDF_TRAINING_AUDIT.csv', pd.DataFrame(oof_audits + audit1 + audit2), immutable=True)
        ledger = []
        for version, fit in [(1, train1), (2, train2)]:
            ledger.append({'risk_version': version, 'allowed_folds': '|'.join(map(str, sorted(fit.fold.unique()))),
                           'source_error_predictor_task_records': len(fit), 'source_biological_task_ids': fit.task_id.nunique(),
                           'source_gene_clusters': fit.gene.nunique(), 'upstream_families': fit.upstream.nunique(),
                           'Source_error_record_ids_hash': ids_hash(frame_records(fit)),
                           'Source_gene_clusters_hash': ids_hash(fit.gene.unique()),
                           'sum_training_weights': float(cluster_weights(fit).sum()),
                           'anchor_error_labels_in_fit_CDF': 0, 'new_gate_error_labels_in_fit_CDF': 0,
                           'anchor_BIOtarget_supervision_labels': 0, 'new_gate_BIOtarget_supervision_labels': 0,
                           'target_C_error_labels': 0, 'Public_biological_learner_fits': 0})
        write_csv(report / 'SOURCE_INFORMATION_BUDGET_LEDGER.csv', pd.DataFrame(ledger), immutable=True)
        write_json(report / 'ACCESS_AND_SCOPE_RECEIPT.json', {
            'actual_expression_cell_arrays_opened': False, 'only_existing_Source_DEV_aggregates_and_outcomes_used': True,
            'Source_outcome_parquet_column_read_scope': 'existing full DEV/SEEN column; only fixed train-role rows passed to fits/CDFs',
            'current_task_Source_BIOtarget_vector_access': 'post-freeze DEV prior-performance evaluation only',
            'historical_source_biology': 'query-relative legal other-context source-train; may share risk gene fold, disclosed',
            'exact_current_task_and_all_current_context_replicates_in_history': 0,
            'finer_replicate_identity_unavailable_no_disjoint_claim': True,
            'Public_learner_update_claim': False, 'actual_Manual_prior_feature_update': True,
            'actual_shared_Source_risk_update_fit': True, 'actual_external_C_adapter_update_claim': False,
            'Source_prediction_truth_alignment_max_abs_error_difference': source_truth_alignment,
            'Orion_paths_opened': [], 'McFaline_paths_opened': [], 'permanent_test_used_for_fit_gate_or_publication': False,
            'parameter_search': False, 'new_upper_training': 0}, immutable=True)
        for binding in registration['input_bindings'] + registration['code_bindings']:
            if sha(binding['path']) != binding['sha256']:
                raise RuntimeError('Original bound data or frozen implementation changed during isolated replay')
        completed = {'status': 'COMPLETE_ACTUAL_REAL_DEV_OPERATIONAL_REPLAY', 'completed_utc': utc(),
                     'role': state['role'], 'runtime_root': str(out),
                     'elapsed_seconds': time.monotonic() - began,
                     'table_fits': 4, 'initial_error_shared_risk_OOF_fits': 2, 'actual_versioned_risk_fits': 2,
                     'actual_Public_append_current_v2': True, 'actual_Error_sequential_appends': True,
                     'actual_serving_resolver_publish_or_retain': True, 'actual_rollback_reload_proof': True,
                     'candidate_release_gate_passes': gate['passes'], 'candidate_status': release_status,
                     'old_task_degradation_reported': True, 'Public_growth_performance_reported': True,
                     'Public_biological_learner_fitted': False, 'Public_self_learning_claim': False,
                     'Source_shared_risk_version_update_fit': True, 'new_external_C_feedback_benefit_claim': False,
                     'new_upstream_calls': 0, 'Orion_access': False, 'McFaline_access': False,
                     'original_input_and_frozen_code_hashes_preserved': True,
                     'reports': [bind(path) for path in sorted(report.iterdir()) if path.is_file()]}
        write_json(report / 'STATUS.json', completed, immutable=True)
        progress('COMPLETE_ACTUAL_REAL_DEV_OPERATIONAL_REPLAY', candidate_status=release_status,
                 elapsed_seconds=completed['elapsed_seconds'])
        print(json.dumps(completed, indent=2), flush=True)
    except Exception as failure:
        progress('FAILED_PRESERVED_ISOLATED_REPLAY', error_type=type(failure).__name__, error=str(failure))
        raise


if __name__ == '__main__':
    main()

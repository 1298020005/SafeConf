#!/usr/bin/env python3
"""Fixed two-fit 2x2 Source DEV diagnosis; never publishes or changes replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.memory import ErrorMemoryRegistry
from tools.safeconf_continual.research import P, PUBLIC, ids_hash, metrics
from tools.scripts import run_safeconf_continual_runtime_replay_agent as replay
from tools.scripts.run_safeconf_source_scaling_uncertainty_agent import counter_metrics, METRICS

FORMAL = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/continual_runtime_replay_20261002_v2')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/continual_runtime_replay'
DEFAULT_RUNTIME = FORMAL.parent / 'continual_component_diagnostic_20261002_v1'
DEFAULT_REPORT = DOC / 'component_diagnostic_v1'
CONDITIONS = ['P1E1', 'P1E2', 'P2E1', 'P2E2', 'V1_ON_P2_ZERO_FIT_DRIFT']
ROLES = ['OLD_ANCHOR', 'NEW_TASK_GATE']
CONTRASTS = {
    'Error_change_at_P1': {'P1E2': 1, 'P1E1': -1},
    'Public_change_at_E1': {'P2E1': 1, 'P1E1': -1},
    'Public_change_at_E2': {'P2E2': 1, 'P1E2': -1},
    'Error_change_at_P2': {'P2E2': 1, 'P2E1': -1},
    'Coupled_total_change': {'P2E2': 1, 'P1E1': -1},
    'Interaction': {'P2E2': 1, 'P1E2': -1, 'P2E1': -1, 'P1E1': 1},
    'Zero_fit_interface_drift': {'V1_ON_P2_ZERO_FIT_DRIFT': 1, 'P1E1': -1},
    'E1_refit_on_P2_vs_zero_fit_drift': {'P2E1': 1, 'V1_ON_P2_ZERO_FIT_DRIFT': -1},
}
SEED = 20260930
REPLICATES = 5000


def sha(path):
    return replay.sha(path)


def bind(path):
    return replay.bind(path)


class ImmutableErrorView:
    """Read existing fixed revision; no live registry or pointer mutation."""
    def __init__(self, manifests):
        self.entries = {(entry['upstream_model_id'], entry['model_version']): entry for entry in manifests}

    def load(self, upstream, version):
        entry = self.entries[(upstream, version)]
        root = Path(entry['runtime_revision_path'])
        if not root.is_relative_to(FORMAL / 'error_memory'):
            raise RuntimeError('Only immutable original replay Error snapshots allowed')
        revision = root / 'runtime_revision.json'
        if sha(revision) != entry['runtime_revision_sha256']:
            raise RuntimeError('Original Error revision binding changed')
        metadata = json.loads(revision.read_text())
        if metadata['output_contract_id'] != replay.CONTRACT:
            raise RuntimeError('Error snapshot output contract differs')
        for record in metadata['files']:
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root) or sha(path) != record['sha256']:
                raise RuntimeError('Error immutable revision artifact changed')
        frame = ErrorMemoryRegistry(root).load(upstream, version)
        if (not frame.is_oof_or_heldout.eq(True).all() or frame.prediction_id.duplicated().any()
            or not np.isfinite(frame.realised_error).all()):
            raise RuntimeError('Original legal Source Error snapshot differs')
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--report-root', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    out, report = args.runtime_root.resolve(), args.report_root.resolve()
    formal_report = DOC / 'actual_v2'
    if out.exists() or report.exists():
        raise FileExistsError('Refuse existing diagnostic roots')
    if (out.parent != FORMAL.parent or out.is_relative_to(FORMAL)
        or not out.name.startswith('continual_component_diagnostic_')
        or report.parent != DOC or report.is_relative_to(formal_report)
        or not report.name.startswith('component_diagnostic_')):
        raise RuntimeError('Only new isolated diagnostic roots allowed')
    out.mkdir(parents=True); report.mkdir(parents=True)
    began = time.monotonic()
    files = [FORMAL / 'MODEL_v1.joblib', FORMAL / 'MODEL_v2.joblib',
             FORMAL / 'FEATURES_PUBLIC_v1.parquet', FORMAL / 'FEATURES_PUBLIC_v2.parquet',
             FORMAL / 'PRE_GATE_RISK_v1.npy', FORMAL / 'PRE_GATE_RISK_v2.npy',
             FORMAL / 'public_memory/CURRENT.json', FORMAL / 'serving_models/SERVING_CURRENT.json',
             FORMAL / 'serving_models/model_registry.jsonl', formal_report / 'REGISTRATION.json',
             formal_report / 'REAL_INGESTION_TRIGGER_RECEIPT.json', formal_report / 'PRE_GATE_MODEL_PREDICTION_FREEZE.json',
             formal_report / 'FROZEN_TASK_ROLE_SCOPE.csv', formal_report / 'STATUS.json', replay.SOURCE]
    code = [Path(__file__), Path(replay.__file__), ROOT / 'tools/safeconf_continual/research.py',
            ROOT / 'tools/scripts/run_safeconf_source_scaling_uncertainty_agent.py'] + [
            ROOT / f'tools/safeconf_continual/{name}.py' for name in
            ['memory', 'learners', 'contracts', 'update', 'versioned_runtime']]
    original_registration = json.loads((formal_report / 'REGISTRATION.json').read_text())
    original_pins = original_registration['input_bindings'] + original_registration['code_bindings']
    for binding in original_pins:
        if sha(binding['path']) != binding['sha256']:
            raise RuntimeError('Original formal replay input or mathematical implementation changed')
    plan = {'schema': 'safeconf_fixed_DEV_2x2_component_diagnostic_v1', 'registered_utc': replay.utc(),
            'evidence_role': 'RETROSPECTIVE_REAL_SOURCE_DEV_DIAGNOSIS_ONLY',
            'conditions': CONDITIONS, 'additional_fits': ['P1E2', 'P2E1'], 'max_additional_fits': 2,
            'roles': ROLES, 'initial_error_folds': [2, 3], 'updated_error_folds': [2, 3, 4],
            'features': P + PUBLIC, 'hgb_seed': SEED, 'hgb_parameters_unchanged': True,
            'CDFs': 'same training-budget-only per upstream/version/context grouping',
            'weights': 'original mean-one gene-cluster weights',
            'bootstrap_replicates': REPLICATES, 'bootstrap_seed': SEED,
            'bootstrap_unit': 'complete gene blocks jointly across both DEV roles, contexts, architectures and conditions',
            'fixed_contrasts': CONTRASTS, 'metrics': list(METRICS),
            'existing_upstream_predictors': 2, 'existing_upstream_families': 1,
            'Public_learner_or_residual_adapter_retrained': False, 'publication_or_winner_selection': False,
            'Orion_access': False, 'McFaline_access': False,
            'input_bindings': [bind(path) for path in files], 'code_bindings': [bind(path) for path in code],
            'original_formal_replay_pins_verified': original_pins}
    replay.write_json(report / 'REGISTRATION.json', plan, immutable=True)
    print(json.dumps({'phase': 'registered', 'pid': os.getpid(), 'additional_fits': 2}), flush=True)
    status = json.loads((formal_report / 'STATUS.json').read_text())
    if status['status'] != 'COMPLETE_ACTUAL_REAL_DEV_OPERATIONAL_REPLAY' or status['candidate_status'] != 'REJECTED':
        raise RuntimeError('Original fixed completed rejected candidate required')
    frozen = json.loads((formal_report / 'PRE_GATE_MODEL_PREDICTION_FREEZE.json').read_text())
    for binding in frozen['models'] + frozen['public_version_specific_features'] + [frozen['predictions']]:
        if sha(binding['path']) != binding['sha256']:
            raise RuntimeError('Original frozen model/feature/prediction bytes differ')
    f1 = pd.read_parquet(FORMAL / 'FEATURES_PUBLIC_v1.parquet')
    f2 = pd.read_parquet(FORMAL / 'FEATURES_PUBLIC_v2.parquet')
    identity = ['upstream', 'model_version', 'task_id', 'gene', 'target', 'fold', 'replay_role', 'dataset_id', 'output_contract_id']
    if (len(f1) != 3398 or not f1[identity].equals(f2[identity])
        or f1.task_id.nunique() != 1699 or f1.groupby('gene').fold.nunique().max() != 1
        or f1.duplicated(['upstream', 'task_id']).any()
        or not np.isfinite(f1[P + PUBLIC].to_numpy(float)).all()
        or not np.isfinite(f2[P + PUBLIC].to_numpy(float)).all()
        or not np.array_equal(f1[P].to_numpy(), f2[P].to_numpy())):
        raise RuntimeError('Fixed full identity cohort or prediction features differ')
    scope = pd.read_csv(formal_report / 'FROZEN_TASK_ROLE_SCOPE.csv')
    scope = scope[scope.initial_known_history].set_index('task_id')
    for upstream in replay.MODELS:
        part = f1[f1.upstream.eq(upstream)]
        expected = scope.loc[part.task_id]
        if (set(part.task_id) != set(scope.index) or part.model_version.nunique() != 1
            or not np.array_equal(part[['gene', 'target', 'condition', 'fold', 'replay_role']].to_numpy(str),
                                  expected[['gene', 'target', 'condition', 'fold', 'replay_role']].to_numpy(str))):
            raise RuntimeError('Original frozen Source identities/fold roles differ')
    b1, b2 = joblib.load(FORMAL / 'MODEL_v1.joblib'), joblib.load(FORMAL / 'MODEL_v2.joblib')
    scores = {'P1E1': b1['model'].predict(f1), 'P2E2': b2['model'].predict(f2),
              'V1_ON_P2_ZERO_FIT_DRIFT': b1['model'].predict(f2)}
    for condition, version in [('P1E1', 1), ('P2E2', 2)]:
        if scores[condition].tobytes() != np.load(FORMAL / f'PRE_GATE_RISK_v{version}.npy').tobytes():
            raise RuntimeError('Existing diagonal score reproduction failed')
    receipt = json.loads((formal_report / 'REAL_INGESTION_TRIGGER_RECEIPT.json').read_text())
    views = {1: ImmutableErrorView(receipt['Error_initial_manifests']),
             2: ImmutableErrorView(receipt['Error_updated_manifests'])}
    for error_version, allowed, bundle in [(1, {'TRAIN_INITIAL'}, b1),
                                           (2, {'TRAIN_INITIAL', 'TRAIN_ADDED'}, b2)]:
        expected = f1[f1.replay_role.isin(allowed)]
        saved = pd.concat([views[error_version].load(upstream, str(expected[expected.upstream.eq(upstream)].model_version.iloc[0]))
                           for upstream in replay.MODELS], ignore_index=True)
        if (set(saved.prediction_id) != set(replay.frame_records(expected))
            or ids_hash(saved.prediction_id) != bundle['training_record_ids_hash']
            or ids_hash(expected.gene.unique()) != bundle['training_gene_clusters_hash']):
            raise RuntimeError('Error-budget record/gene identity differs from original frozen fit')
    fit_costs, audits, budgets = [], [], []
    for condition, feature, error_version, roles in [
            ('P1E2', f1, 2, {'TRAIN_INITIAL', 'TRAIN_ADDED'}),
            ('P2E1', f2, 1, {'TRAIN_INITIAL'})]:
        started = time.monotonic()
        model, fit, labels, audit = replay.fit_from_error_memory(feature, views[error_version], roles)
        model_path = out / f'{condition}.joblib'
        joblib.dump({'model': model, 'condition': condition, 'CDFs': replay.cdf_objects(fit),
                     'training_record_ids_hash': ids_hash(replay.frame_records(fit)),
                     'features_binding': bind(FORMAL / f'FEATURES_PUBLIC_v{1 if condition == "P1E2" else 2}.parquet')}, model_path)
        model_path.chmod(0o444)
        scores[condition] = model.predict(feature)
        if scores[condition].tobytes() != joblib.load(model_path)['model'].predict(feature).tobytes():
            raise RuntimeError('Cross-condition saved-model reload differs')
        fit_costs.append({'condition': condition, 'fit_predict_save_reload_seconds': time.monotonic() - started,
                          'table_fits': 1, 'GPU_hours': 0, 'numerical_threads': 1, 'model_sha256': sha(model_path)})
        audits.extend([{'condition': condition, **row} for row in audit])
        budgets.append({'condition': condition, 'Public_version': 1 if condition == 'P1E2' else 2,
                        'Error_version': error_version, 'Source_error_records': len(fit),
                        'Source_biological_tasks': fit.task_id.nunique(), 'Source_gene_clusters': fit.gene.nunique(),
                        'allowed_fit_folds': '|'.join(map(str, sorted(fit.fold.unique()))),
                        'anchor_newtask_labels_in_fit_or_CDF': 0})
    matrix = np.column_stack([scores[name] for name in CONDITIONS])
    if matrix.shape != (3398, 5) or not np.isfinite(matrix).all():
        raise RuntimeError('Every diagnostic condition must predict all finite frozen records')
    score_frame = f1[identity].copy()
    for name in CONDITIONS: score_frame[name] = scores[name]
    score_frame.to_parquet(out / 'FROZEN_DIAGNOSTIC_SCORES.parquet', index=False)
    (out / 'FROZEN_DIAGNOSTIC_SCORES.parquet').chmod(0o444)
    replay.write_json(report / 'PREDICTION_FREEZE.json', {'scores': bind(out / 'FROZEN_DIAGNOSTIC_SCORES.parquet'),
                     'all_conditions_frozen_before_DEV_scoring': True, 'actual_additional_fits': len(fit_costs),
                     'existing_diagonal_predictions_byte_identical': True, 'no_gate_or_publication': True}, immutable=True)
    print(json.dumps({'phase': 'two_fits_complete_all_predictions_frozen'}), flush=True)

    evaluation = replay.read_source_errors(f1, set(ROLES))
    by_id = evaluation.set_index(replay.frame_records(evaluation)).true_error_rmse
    evaluation = f1[f1.replay_role.isin(ROLES)].copy().reset_index(drop=True)
    evaluation['true_error_rmse'] = replay.frame_records(evaluation).map(by_id)
    eval_positions = np.flatnonzero(f1.replay_role.isin(ROLES).to_numpy())
    score_matrix = matrix[eval_positions]
    genes = sorted(evaluation.gene.unique())
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    gene_index = evaluation.gene.map(gene_lookup).to_numpy(int)
    rng = np.random.default_rng(SEED)
    counts = np.asarray([np.bincount(rng.integers(0, len(genes), len(genes)), minlength=len(genes))
                         for _ in range(REPLICATES)], dtype=np.uint16)
    np.savez_compressed(out / 'JOINT_GENE_BOOTSTRAP_COUNTS.npz', genes=np.asarray(genes), counts=counts, seed=np.asarray(SEED))
    strata, macro, contrasts = [], [], []
    all_draws = np.empty((2, REPLICATES, 5, len(METRICS)))
    all_points = np.empty((2, 5, len(METRICS)))
    for role_index, role in enumerate(ROLES):
        points, draws = [], []
        for upstream in replay.MODELS:
            for context in replay.CONTEXTS:
                use = np.flatnonzero((evaluation.replay_role.eq(role) & evaluation.upstream.eq(upstream)
                                     & evaluation.target.eq(context)).to_numpy())
                if len(use) < 20: raise RuntimeError('Original planned stratum is invalid')
                part = evaluation.iloc[use]
                multiplicity = counts[:, gene_index[use]].astype(np.int32)
                measured, boot = [], []
                for index, condition in enumerate(CONDITIONS):
                    result = metrics(part, score_matrix[use, index])
                    point = np.asarray([result[name] for name in METRICS])
                    compact = counter_metrics(part.true_error_rmse.to_numpy(), part.task_id.to_numpy(str),
                                              score_matrix[use, index], np.ones((1, len(use)), int))[0]
                    if not np.allclose(point, compact, atol=1e-12, rtol=0, equal_nan=True):
                        raise RuntimeError('Count metric does not match original metric')
                    measured.append(point)
                    boot.append(counter_metrics(part.true_error_rmse.to_numpy(), part.task_id.to_numpy(str),
                                                score_matrix[use, index], multiplicity))
                    strata.append({'role': role, 'upstream': upstream, 'context': context,
                                   'condition': condition, 'n_tasks': len(use), 'n_gene_clusters': part.gene.nunique(),
                                   'task_ids_hash': ids_hash(part.task_id), **result})
                points.append(measured); draws.append(np.stack(boot, axis=1))
        all_points[role_index] = np.mean(points, axis=0)
        all_draws[role_index] = np.mean(draws, axis=0)
        for index, condition in enumerate(CONDITIONS):
            macro.append({'role': role, 'condition': condition, 'n_contexts': 4,
                          'n_predictors': 2, 'n_upstream_families': 1, 'n_strata': 8,
                          **dict(zip(METRICS, all_points[role_index, index]))})
        for contrast, coefficients in CONTRASTS.items():
            vector = np.asarray([coefficients.get(condition, 0) for condition in CONDITIONS])
            point = vector @ all_points[role_index]
            delta = np.einsum('rcm,c->rm', all_draws[role_index], vector)
            for metric_index, metric in enumerate(METRICS):
                finite = delta[np.isfinite(delta[:, metric_index]), metric_index]
                if len(finite) != REPLICATES: raise RuntimeError('Planned paired bootstrap draw invalid')
                lower, upper = np.quantile(finite, [.025, .975], method='linear')
                contrasts.append({'role': role, 'contrast': contrast, 'metric': metric,
                                  'point_difference': float(point[metric_index]),
                                  'bootstrap_mean_difference': float(finite.mean()), 'ci95_lower': float(lower),
                                  'ci95_upper': float(upper), 'valid_draws': len(finite),
                                  'replicates': REPLICATES, 'seed': SEED,
                                  'direction': 'positive favors first condition for U20/Spearman; negative for AURC/miss/errors'})
        print(json.dumps({'phase': 'paired_bootstrap_done', 'role': role, 'replicates': REPLICATES}), flush=True)
    np.savez_compressed(out / 'JOINT_GENE_METRIC_DRAWS.npz', macro=all_draws,
                        condition=np.asarray(CONDITIONS), role=np.asarray(ROLES), metrics=np.asarray(METRICS))
    for filename, records in [('STRATA.csv', strata), ('MACRO.csv', macro), ('PAIRED_CONTRASTS.csv', contrasts),
                              ('FIT_COSTS.csv', fit_costs), ('CDF_ALLOWED_BUDGET_AUDIT.csv', audits),
                              ('INFORMATION_BUDGET.csv', budgets)]:
        replay.write_csv(report / filename, pd.DataFrame(records), immutable=True)
    for binding in plan['input_bindings'] + plan['code_bindings'] + original_pins:
        if sha(binding['path']) != binding['sha256']:
            raise RuntimeError('Formal replay/input/code changed during diagnostic')
    summary = {'status': 'COMPLETE_FIXED_TWO_FIT_DEV_COMPONENT_DIAGNOSTIC',
               'elapsed_seconds': time.monotonic() - began, 'actual_additional_fits': len(fit_costs),
               'bootstrap_replicates': REPLICATES, 'joint_evaluation_gene_clusters': len(genes),
               'all_conditions_all_planned_strata_retained': True, 'strata_rows': len(strata),
               'macro_rows': len(macro), 'contrast_rows': len(contrasts),
               'all_formal_artifacts_and_code_unchanged': True, 'all_bootstrap_draws_valid': True,
               'Public_Biology_Learner_or_residual_adapter_fits': 0, 'published_candidates': 0,
               'Orion_access': False, 'McFaline_access': False, 'runtime_root': str(out),
               'server_artifacts': [bind(path) for path in sorted(out.iterdir()) if path.is_file()]}
    replay.write_json(report / 'STATUS.json', summary, immutable=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

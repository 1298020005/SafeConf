#!/usr/bin/env python3
"""Registered Source DEV in-sample scale stress; no training or target access."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, metrics

BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
SOURCE = BASE / 'orion_source_core_20261002_v1'
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/source_counterfactual_scale_stress_v1'
OUTPUT = BASE / 'source_counterfactual_scale_stress_20261002_v1'
SCALES = [1.0, 0.1, 0.05]
UPSTREAMS = ['TxPert_GAT', 'TxPert_Exphormer']
CONTEXTS = ['K562', 'RPE1', 'hepg2', 'jurkat']
METHODS = ['P_only_hgb', 'Manual_hgb', 'Manual_DirectRMSE', 'Manual_WeightedHistoryDistance', 'Magnitude']
IDENTITY = ['task_id', 'gene', 'target', 'condition', 'fold', 'upstream', 'model_version', 'output_contract_id']
SCHEMA = 'safeconf_source_counterfactual_scale_stress_v1'
PROPOSAL_SCHEMA = SCHEMA + '_proposal'
APPROVAL_SCHEMA = SCHEMA + '_root_approval'
PINS = {
    'SOURCE_FEATURE_MANIFEST.json': 'b74b9dfbc64f5acc82b7b0d34023aa6a523bd11d1904ced7fd638c5bec4ef2d1',
    'MODEL_MANIFEST.json': 'c8d9d5003860d8a5e117398a7d44b5c67cfb700c6fb56f37bcf04a5d549909a6',
}
MATH = ROOT / 'tools/scripts/seal_safeconf_orion_source_risk_agent.py'
MATH_SHA = '8b69e6ce29dc92aa34150b3b08aac62ce8333a10f84ec432a39d0e81ffc64482'
MAX_SECONDS, MAX_RSS_BYTES = 300, 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(item, readonly=False):
    path = Path(item['path']).resolve()
    if not path.is_file() or binding(path) != item:
        raise RuntimeError(f'Exact file binding changed: {path}')
    if readonly and path.stat().st_mode & 0o222:
        raise RuntimeError(f'Readonly registration required: {path}')
    return path


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, obj, readonly=False):
    path = Path(path)
    with path.open('x') as handle:
        handle.write(json.dumps(json_clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if readonly:
        path.chmod(0o444)


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


def finite(values, scope):
    if not np.isfinite(np.asarray(values, float)).all():
        raise RuntimeError(f'Nonfinite values in {scope}; no survivor selection')


def paths(out, report):
    out, report = Path(out).resolve(), Path(report).resolve()
    if out.parent != BASE or not out.name.startswith('source_counterfactual_scale_stress_'):
        raise RuntimeError('Fresh direct-child Source-only runtime namespace required')
    if report.parent != DOC.parent or not report.name.startswith('source_counterfactual_scale_stress_'):
        raise RuntimeError('Fresh direct-child Source-only report namespace required')
    return out, report


def dependency_paths():
    return [Path(__file__).resolve(), MATH,
            ROOT / 'tools/scripts/prepare_safeconf_orion_source_core_agent.py',
            ROOT / 'tools/scripts/run_safeconf_research_closure.py',
            ROOT / 'tools/scripts/run_dual_memory_txpert_public_biology.py',
            ROOT / 'tools/scripts/seal_mcfaline_dual_memory_risk.py',
            ROOT / 'tools/safeconf_continual/research.py',
            ROOT / 'tools/safeconf_continual/learners.py',
            ROOT / 'tools/safeconf_continual/contracts.py']


def registration_inputs():
    for name, expected in PINS.items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError(f'Original Source registry pin changed: {name}')
    if sha(MATH) != MATH_SHA:
        raise RuntimeError('Original 8b69 math changed')
    artifacts = json.loads((SOURCE / 'ARTIFACT_HASHES.json').read_text())
    if len({item['path'] for item in artifacts}) != len(artifacts):
        raise RuntimeError('Duplicate Source registry entries')
    result = []
    for item in artifacts:
        path = (SOURCE / item['path']).resolve()
        if path.parent != SOURCE or path.name != item['path']:
            raise RuntimeError('Noncanonical Source artifact path')
        actual = binding(path)
        if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
            raise RuntimeError(f'Original Source artifact differs: {path}')
        result.append(actual)
    result.append(binding(SOURCE / 'ARTIFACT_HASHES.json'))
    return result


def prepare(report, out):
    out, report = paths(out, report)
    if out.exists() or report.exists():
        raise RuntimeError('New output/report version required')
    inputs = registration_inputs()  # Byte integrity only, no array values or predictions.
    source_feature = json.loads((SOURCE / 'SOURCE_FEATURE_MANIFEST.json').read_text())
    if source_feature['risk_P_columns'] != P or source_feature['risk_public_columns'] != PUBLIC:
        raise RuntimeError('Original exact13 feature contract changed')
    proposal = {
        'schema': PROPOSAL_SCHEMA, 'status': 'PROPOSED_NOT_AUTHORIZED',
        'diagnostic_type': 'Source_DEV_SEEN_in_sample_counterfactual_input_scale_stress',
        'source_root': str(SOURCE), 'runtime_output': str(out), 'report_output': str(report),
        'scales': SCALES, 'upstreams': UPSTREAMS, 'contexts': CONTEXTS,
        'n_tasks': 1808, 'n_upstream_task_records_per_scale': 3616,
        'n_model_scale_upstream_conditions': 12, 'n_upstream_predictor_architectures': 2,
        'n_upstream_families': 1, 'in_sample_risk_training_overlap': True,
        'public_reference': 'cached_original_Manual_gene_outer_OOF_prior_full_vectors',
        'feature_columns': P + PUBLIC, 'model_ids': METHODS[:2], 'comparators': METHODS[2:],
        'scale_operation': 's * original Source predicted DELTA; truth and prior unchanged',
        'changed_features': P + ['prediction_prior_rmse', 'prediction_prior_cosine'],
        'frozen_public_features': [c for c in PUBLIC if c not in ('prediction_prior_rmse', 'prediction_prior_cosine')],
        'numeric_rules': {'sparsity_abs_le': 1e-8, 'cosine_norm_product_gt': 1e-12,
                          'std_ddof': 0, 'abs_quantile': 0.95,
                          'weighted_distance': 'sqrt(prediction_prior_rmse**2 + prior_uncertainty**2)'},
        'prediction_stage': 'All 3 scales x 2 architectures x 2 HGB feature/score records readonly and hashsealed before Source truth numeric access',
        's1_baseline': 'Full3616 original order/batch; exact cached feature and fitted prediction bit equality required',
        'source_truth_after_seal': 'RMSE(s*original Source predicted DELTA, original Source true effects)',
        'reporting': ['all8 upstream/context strata for every scale/method', 'equal4context macro per upstream',
                      'original research.metrics U20/Spearman/AURC/miss/error_at10/20/50',
                      'risk ties and risk/error ordering versus s1'],
        'truth_bytes_preseal': 'Streaming SHA integrity checks only; no numeric truth load before prediction seal',
        'prohibitions': {'new_fits': 0, 'CDF_fits': 0, 'public_retrieval_refits': 0, 'bootstrap_draws': 0,
                         'Target_access': 0, 'Orion_access': 0, 'new_upstream_candidates': 0,
                         'winner_or_scale_selection': False, 'source_mean_variant': False,
                         'new_scientific_confirmation': False, 'registry_or_serving_writes': False},
        'resources': {'max_wall_seconds': MAX_SECONDS, 'max_peak_rss_bytes': MAX_RSS_BYTES,
                      'max_CPU_threads': 4, 'GPU_hours': 0, 'new_download_bytes': 0},
        'source_input_bindings': inputs, 'code_bindings': [binding(p) for p in dependency_paths()],
        'source_full_registry': binding(SOURCE / 'MODEL_MANIFEST.json'),
        'original_inference_math': binding(MATH),
        'interpretation_limit': 'Final risk cores trained all original3616Source rows; descriptive stress is not heldout transfer or proof of Orion failure cause',
    }
    report.mkdir()
    write_json(report / 'PROPOSED_SCOPE.json', proposal, readonly=True)
    write_json(report / 'ROOT_APPROVAL_TEMPLATE.json', {
        'schema': APPROVAL_SCHEMA, 'status': 'DRAFT_NOT_AUTHORIZED',
        'authorized_to_run_source_only_counterfactual_stress': False,
        'proposal': binding(report / 'PROPOSED_SCOPE.json'),
        'independent_review': {'path': 'ROOT_TO_BIND_ACTUAL_PASS_RECEIPT', 'bytes': None, 'sha256': None},
        'conditions': 'Exact proposed code/input/scales/resource/scope only; no Target access or new models',
    })
    print(json.dumps({'proposal': binding(report / 'PROPOSED_SCOPE.json'), 'script': binding(__file__),
                      'runtime_output': str(out), 'actual_run_started': False}, indent=2), flush=True)


def authorize(approval_path):
    approval_path = Path(approval_path).resolve()
    if approval_path.stat().st_mode & 0o222:
        raise RuntimeError('Readonly Root approval required')
    approval = json.loads(approval_path.read_text())
    if (approval.get('schema') != APPROVAL_SCHEMA or approval.get('status') != 'APPROVED'
            or approval.get('authorized_to_run_source_only_counterfactual_stress') is not True):
        raise RuntimeError('Actual Root Source-only diagnostic approval missing')
    proposal_path = checked(approval['proposal'], readonly=True)
    proposal = json.loads(proposal_path.read_text())
    review = json.loads(checked(approval['independent_review'], readonly=True).read_text())
    if review.get('status') != 'PASS' or review.get('proposal') != approval['proposal']:
        raise RuntimeError('Independent PASS must bind this exact proposal')
    if proposal.get('schema') != PROPOSAL_SCHEMA or proposal.get('scales') != SCALES or proposal.get('upstreams') != UPSTREAMS or proposal.get('contexts') != CONTEXTS:
        raise RuntimeError('Fixed diagnostic proposal differs')
    if proposal.get('model_ids') != METHODS[:2] or proposal.get('comparators') != METHODS[2:] or proposal.get('source_root') != str(SOURCE):
        raise RuntimeError('Original Source-only model/reference scope changed')
    if proposal.get('resources') != {'max_wall_seconds': MAX_SECONDS, 'max_peak_rss_bytes': MAX_RSS_BYTES, 'max_CPU_threads': 4, 'GPU_hours': 0, 'new_download_bytes': 0}:
        raise RuntimeError('Registered resource limits changed')
    if proposal.get('code_bindings') != [binding(p) for p in dependency_paths()]:
        raise RuntimeError('Actual executing/dependency code differs')
    if proposal.get('source_input_bindings') != registration_inputs():
        raise RuntimeError('Original Source artifact set changed')
    out, report = paths(proposal['runtime_output'], proposal['report_output'])
    if out.exists() or (report / 'RESULT_MANIFEST.json').exists():
        raise RuntimeError('Actual diagnostic output must be fresh; no rerun overwrite')
    return proposal, out, report


def load_math():
    spec = importlib.util.spec_from_file_location('_frozen_source_stress_math', MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine_rows(a, b):
    # Exact original tx.cosine_rows vector formula, including its cutoff.
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 1e-12)


def feature_frame(scale, frame, tasks, prior, math_module):
    result = frame.copy()
    for upstream in UPSTREAMS:
        rows = np.flatnonzero(frame.upstream.eq(upstream).to_numpy())
        if frame.iloc[rows].task_id.tolist() != tasks.task_id.tolist():
            raise RuntimeError('Full original task/Source frame row order differs')
        original = np.load(SOURCE / f'SOURCE_{upstream}_PREDICTED_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
        if original.shape != (1808, 3285) or original.dtype != np.dtype('float64'):
            raise RuntimeError('Original Source DELTA shape/dtype changed')
        delta = np.asarray(original) if scale == 1.0 else np.multiply(original, scale)
        computed = math_module.prediction_features(delta)
        result.loc[rows, P] = computed.to_numpy(float)
        result.loc[rows, 'prediction_prior_rmse'] = np.sqrt(np.mean(np.square(delta - prior), axis=-1))
        result.loc[rows, 'prediction_prior_cosine'] = cosine_rows(delta, prior)
        if not bits(np.sqrt(np.mean(prior**2, axis=1)), frame.iloc[rows].prior_magnitude.to_numpy(float)):
            raise RuntimeError('Cached legal Manual prior magnitude differs from its exact OOF vector')
    finite(result[P + PUBLIC], f'scale={scale} features')
    constant = [c for c in PUBLIC if c not in ('prediction_prior_rmse', 'prediction_prior_cosine')]
    if not bits(result[constant].to_numpy(float), frame[constant].to_numpy(float)):
        raise RuntimeError('Fixed legal Public features changed')
    if scale == 1.0 and not bits(result[P + PUBLIC].to_numpy(float), frame[P + PUBLIC].to_numpy(float)):
        raise RuntimeError('s1 original complete feature bytes failed reproduction')
    return result


def association(a, b, kind='spearman'):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float((spearmanr if kind == 'spearman' else kendalltau)(a, b).statistic)


def rank_diagnostics(ids, current, baseline):
    ids, current, baseline = np.asarray(ids, str), np.asarray(current, float), np.asarray(baseline, float)
    order = np.lexsort((ids, -current))
    base_order = np.lexsort((ids, -baseline))
    rank, base_rank = np.empty(len(ids), int), np.empty(len(ids), int)
    rank[order], base_rank[base_order] = np.arange(len(ids)), np.arange(len(ids))
    k = math.ceil(0.2 * len(ids))
    _, counts = np.unique(current, return_counts=True)
    boundary = current[order[k - 1]]
    return {'spearman_vs_s1': association(current, baseline), 'kendall_vs_s1': association(current, baseline, 'kendall'),
            'sort_order_exact_vs_s1': bool(np.array_equal(order, base_order)),
            'mean_abs_rank_shift_vs_s1': float(np.mean(np.abs(rank - base_rank))),
            'top20_overlap_fraction_vs_s1': len(set(order[:k]) & set(base_order[:k])) / k,
            'n_unique': len(counts), 'largest_tie_count': int(counts.max()),
            'top20_boundary_total_ties': int(np.sum(current == boundary)),
            'top20_boundary_selected_ties': int(np.sum(current[order[:k]] == boundary))}


def runtime_limits():
    def abort(_signal, _frame):
        raise RuntimeError('Registered wall/RSS resource limit exceeded')
    signal.signal(signal.SIGALRM, abort)
    signal.signal(signal.SIGUSR1, abort)
    signal.alarm(MAX_SECONDS)
    stop = threading.Event()
    def monitor():
        while not stop.wait(0.05):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS_BYTES:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
    threading.Thread(target=monitor, daemon=True).start()
    return stop


def run(approval_path):
    started, cpu_started = time.monotonic(), time.process_time()
    stop = runtime_limits()
    proposal, out, report = authorize(approval_path)
    out.mkdir()
    write_json(out / 'EXECUTION_REGISTRATION.json', {'schema': SCHEMA, 'pid': os.getpid(),
               'approval': binding(approval_path), 'proposal': json.loads(Path(approval_path).read_text())['proposal'],
               'script': binding(__file__), 'start_time_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'Source_truth_numeric_access_started': False})
    try:
        feature_manifest = json.loads((SOURCE / 'SOURCE_FEATURE_MANIFEST.json').read_text())
        model_manifest = json.loads((SOURCE / 'MODEL_MANIFEST.json').read_text())
        if feature_manifest['n_biological_tasks'] != 1808 or feature_manifest['n_upstream_task_records'] != 3616:
            raise RuntimeError('Fixed original Source budget differs')
        task_columns = ['task_id', 'gene', 'target', 'condition', 'fold']
        tasks = pd.read_parquet(SOURCE / 'SOURCE_TASKS.parquet', columns=task_columns, use_threads=False)
        frame = pd.read_parquet(SOURCE / 'source_Manual.parquet', columns=IDENTITY + P + PUBLIC, use_threads=False)
        if len(tasks) != 1808 or not tasks.task_id.is_unique or tasks.gene.nunique() != 575 or len(frame) != 3616:
            raise RuntimeError('Frozen Source tasks/genes/records changed')
        if (set(frame.upstream) != set(UPSTREAMS) or set(frame.target) != set(CONTEXTS)
                or frame.duplicated(['upstream', 'task_id']).any() or not tasks.groupby('gene').fold.nunique().eq(1).all()):
            raise RuntimeError('Frozen Source predictor/context/identity/folds changed')
        for upstream in UPSTREAMS:
            part = frame[frame.upstream.eq(upstream)]
            if not part[task_columns].reset_index(drop=True).equals(tasks) or part.model_version.nunique() != 1:
                raise RuntimeError('Exact Source task metadata/model version identity differs')
        cached = np.load(SOURCE / 'SOURCE_Manual_RISK_FEATURES.npy', allow_pickle=False)
        if not bits(cached, frame[P + PUBLIC].to_numpy(float)):
            raise RuntimeError('Original cached Source Manual feature layout/bytes differs')
        if not bits(np.load(SOURCE / 'SOURCE_P_only_RISK_FEATURES.npy', allow_pickle=False), frame[P].to_numpy(float)):
            raise RuntimeError('Original cached Source P-only feature bytes differ')
        finite(cached, 'original Source features')
        prior = np.load(SOURCE / 'SOURCE_OOF_Manual_PRIOR_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
        if prior.shape != (1808, 3285) or prior.dtype != np.dtype('float64'):
            raise RuntimeError('Original cached OOF Manual prior shape/dtype differs')
        finite(prior, 'original legal OOF prior')
        models = {}
        for name, reference, columns in [('P_only_hgb', 'P_only', P), ('Manual_hgb', 'Manual', P + PUBLIC)]:
            entry = [e for e in model_manifest['models'] if e['reference'] == reference and e['learner'] == 'hgb' and e['role'] == 'Source_risk']
            if len(entry) != 1 or entry[0]['n_rows'] != 3616 or entry[0]['columns'] != columns:
                raise RuntimeError('Exact frozen HGB registry entry changed')
            model = joblib.load(SOURCE / entry[0]['path'])
            if model.columns != columns or model.model.get_params() != entry[0]['model_params']:
                raise RuntimeError('Frozen model/preprocessor contract differs')
            models[name] = model
        baseline = pd.read_parquet(SOURCE / 'SOURCE_FINAL_RISK_FIT_PREDICTIONS.parquet', columns=['upstream', 'task_id'] + METHODS[:2], use_threads=False)
        if not baseline[['upstream', 'task_id']].equals(frame[['upstream', 'task_id']]):
            raise RuntimeError('Cached full-fit baseline task order differs')
        math_module = load_math()
        all_features, all_scores, conditions, records = [], [], [], []
        with threadpool_limits(limits=4):
            for scale in SCALES:
                features = feature_frame(scale, frame, tasks, prior, math_module)
                score = pd.DataFrame(index=features.index)
                for name in METHODS[:2]:
                    score[name] = models[name].predict(features)
                score['Manual_DirectRMSE'] = features.prediction_prior_rmse
                score['Manual_WeightedHistoryDistance'] = np.sqrt(features.prediction_prior_rmse**2 + features.prior_uncertainty**2)
                score['Magnitude'] = features.predicted_magnitude
                finite(score, f'scale={scale} risk/comparator scores')
                if scale == 1.0:
                    for name in METHODS[:2]:
                        if not bits(score[name].to_numpy(float), baseline[name].to_numpy(float)):
                            raise RuntimeError(f's1 full-batch cached fitted risk bytes differ: {name}')
                all_features.append(features[P + PUBLIC].to_numpy(float))
                all_scores.append(score[METHODS].to_numpy(float))
                records.append(pd.concat([features[IDENTITY + P + PUBLIC], score], axis=1).assign(scale=scale))
                for upstream in UPSTREAMS:
                    index = np.flatnonzero(frame.upstream.eq(upstream).to_numpy())
                    for method in METHODS[:2]:
                        width = len(P) if method == 'P_only_hgb' else len(P + PUBLIC)
                        conditions.append({'scale': scale, 'upstream': upstream, 'model': method, 'n_tasks': len(index),
                                           'n_model_features': width,
                                           'feature_sha256': hashlib.sha256(all_features[-1][index, :width].tobytes()).hexdigest(),
                                           'risk_sha256': hashlib.sha256(score[method].to_numpy(float)[index].tobytes()).hexdigest(),
                                           'in_sample_training_overlap': True})
        feature_array, score_array = np.stack(all_features), np.stack(all_scores)
        np.save(out / 'SCALED_MANUAL_FEATURES.npy', feature_array, allow_pickle=False)
        np.save(out / 'SCALED_RISK_COMPARATOR_SCORES.npy', score_array, allow_pickle=False)
        pd.concat(records, ignore_index=True).to_parquet(out / 'SCALED_FEATURE_RISK_RECORDS.parquet', index=False)
        frame[IDENTITY].to_csv(out / 'RECORD_IDENTITIES.csv', index=False)
        pd.DataFrame(conditions).to_csv(report / 'PREDICTION_CONDITIONS.csv', index=False)
        prediction_files = ['SCALED_MANUAL_FEATURES.npy', 'SCALED_RISK_COMPARATOR_SCORES.npy', 'SCALED_FEATURE_RISK_RECORDS.parquet', 'RECORD_IDENTITIES.csv']
        for name in prediction_files:
            (out / name).chmod(0o444)
        (report / 'PREDICTION_CONDITIONS.csv').chmod(0o444)
        seal = {'schema': SCHEMA + '_pretruth_seal', 'status': 'SEALED_BEFORE_SOURCE_TRUTH_NUMERIC_ACCESS',
                'proposal': json.loads(Path(approval_path).read_text())['proposal'], 'approval': binding(approval_path),
                'prediction_bindings': [binding(out / name) for name in prediction_files],
                'conditions': binding(report / 'PREDICTION_CONDITIONS.csv'),
                'n_model_scale_upstream_conditions': 12, 'n_scaled_record_rows': 10848,
                's1_feature_bits_equal_cached': True, 's1_two_HGB_score_bits_equal_fullfit_cache': True,
                'five_frozen_Public_features_bits_equal_all_scales': True,
                'Source_truth_numeric_values_read': 0, 'truth_hash_bytes_only_for_integrity': True,
                'new_fits': 0, 'CDF_fits': 0, 'new_upstream_candidates': 0, 'Target_access': 0,
                'in_sample_training_overlap': True, 'elapsed_seconds': time.monotonic() - started}
        write_json(out / 'PRETRUTH_STRESS_SEAL.json', seal, readonly=True)
        seal_binding = binding(out / 'PRETRUTH_STRESS_SEAL.json')
        for item in seal['prediction_bindings']:
            checked(item, readonly=True)
        print(json.dumps({'phase': 'Source_predictions_hashsealed', 'seal': seal_binding, 'Source_truth_numeric_values_read': 0}), flush=True)

        # First numeric Source truth access is strictly after all12 conditions are sealed.
        write_json(out / 'SOURCE_TRUTH_ACCESS_STARTED.json', {'pretruth_seal': seal_binding,
                   'source_truth': next(x for x in proposal['source_input_bindings'] if Path(x['path']).name == 'SOURCE_TRUE_EFFECTS.npy'),
                   'elapsed_seconds': time.monotonic() - started}, readonly=True)
        truth = np.load(SOURCE / 'SOURCE_TRUE_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
        if truth.shape != (1808, 3285) or truth.dtype != np.dtype('float64'):
            raise RuntimeError('Original Source truth shape/dtype differs')
        finite(truth, 'Source truth after prediction seal')
        errors = np.empty((len(SCALES), len(frame)), dtype=float)
        for upstream in UPSTREAMS:
            index = np.flatnonzero(frame.upstream.eq(upstream).to_numpy())
            original = np.load(SOURCE / f'SOURCE_{upstream}_PREDICTED_EFFECTS.npy', mmap_mode='r', allow_pickle=False)
            for number, scale in enumerate(SCALES):
                delta = np.asarray(original) if scale == 1.0 else np.multiply(original, scale)
                errors[number, index] = np.sqrt(np.mean(np.square(delta - truth), axis=1))
        finite(errors, 'Source errors after seal')
        cached_errors = pd.read_parquet(SOURCE / 'SOURCE_FINAL_RISK_FIT_PREDICTIONS.parquet', columns=['true_error_rmse'], use_threads=False).true_error_rmse.to_numpy(float)
        if not bits(errors[0], cached_errors):
            raise RuntimeError('s1 Source truth RMSE bytes differ from original full-fit cache')
        np.save(out / 'POSTSEAL_SOURCE_ERRORS.npy', errors, allow_pickle=False)
        strata, error_order, feature_summary = [], [], []
        for number, scale in enumerate(SCALES):
            for (upstream, context), group in frame.groupby(['upstream', 'target'], sort=True):
                index = group.index.to_numpy(int)
                ids = group.task_id.to_numpy(str)
                evaluated = group[IDENTITY].copy()
                evaluated['true_error_rmse'] = errors[number, index]
                common = {'scale': scale, 'upstream': upstream, 'context': context, 'in_sample_training_overlap': True}
                error_order.append(common | {'n_tasks': len(index), 'mean_error_rmse': float(np.mean(errors[number, index])),
                    'median_error_rmse': float(np.median(errors[number, index])),
                    'n_errors_changed_vs_s1': int(np.sum(errors[number, index] != errors[0, index]))}
                    | rank_diagnostics(ids, errors[number, index], errors[0, index]))
                for j, method in enumerate(METHODS):
                    current, old = score_array[number, index, j], score_array[0, index, j]
                    values = metrics(evaluated, current)
                    if values['n_tasks'] != len(index) or values['n_planned'] != len(index):
                        raise RuntimeError('No nonfinite survivor filtering permitted')
                    strata.append(common | {'method': method} | values | rank_diagnostics(ids, current, old))
                for j, column in enumerate(P + PUBLIC):
                    values = feature_array[number, index, j]
                    feature_summary.append(common | {'feature': column, 'n_tasks': len(index),
                        'min': float(np.min(values)), 'median': float(np.median(values)), 'max': float(np.max(values)),
                        'mean': float(np.mean(values))})
        stratum_frame = pd.DataFrame(strata)
        macro_columns = ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate', 'error_at_10', 'error_at_20', 'error_at_50']
        macro = stratum_frame.groupby(['scale', 'upstream', 'method'], as_index=False)[macro_columns].mean()
        macro['actual_contexts'] = 4
        macro['aggregation'] = 'equal_context_macro'
        macro['in_sample_training_overlap'] = True
        pd.DataFrame(error_order).to_csv(report / 'SOURCE_ERROR_ORDERING.csv', index=False)
        stratum_frame.to_csv(report / 'ALL_STRATUM_METRICS_RANKS_TIES.csv', index=False)
        macro.to_csv(report / 'ALL_CONTEXT_MACRO_METRICS.csv', index=False)
        pd.DataFrame(feature_summary).to_csv(report / 'ALL_FEATURE_SUMMARIES.csv', index=False)
        if len(stratum_frame) != 120 or len(macro) != 30 or len(error_order) != 24:
            raise RuntimeError('All fixed scale/predictor/context/method conditions must remain visible')
        for item in proposal['source_input_bindings'] + proposal['code_bindings']:
            checked(item)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if peak > MAX_RSS_BYTES or time.monotonic() - started > MAX_SECONDS:
            raise RuntimeError('Registered measured resource budget exceeded')
        result = {'schema': SCHEMA, 'status': 'COMPLETE', 'approval': binding(approval_path),
            'proposal': json.loads(Path(approval_path).read_text())['proposal'], 'pretruth_seal': seal_binding,
            'n_tasks': 1808, 'n_Source_genes': 575, 'n_records_per_scale': 3616,
            'actual_contexts': CONTEXTS, 'upstreams': UPSTREAMS, 'scales': SCALES,
            'n_model_scale_upstream_conditions': 12, 'stratum_metric_rows': len(stratum_frame), 'macro_metric_rows': len(macro),
            'all_original_Source_artifacts_and_code_before_after_SHA_equal': True,
            's1_feature_and_fitted_risk_prediction_and_Source_error_bytes_exact': True,
            'truth_numeric_access_only_after_pretruth_seal': True,
            'new_fits': 0, 'CDF_fits': 0, 'bootstrap_draws': 0, 'new_upstream_candidates': 0,
            'Target_access': 0, 'Orion_access': 0, 'source_mean_variant': False, 'winner_or_scale_selection': False,
            'in_sample_training_overlap': True, 'not_heldout_transfer': True,
            'causal_Orion_failure_explanation_established': False,
            'elapsed_seconds': time.monotonic() - started, 'CPU_seconds': time.process_time() - cpu_started,
            'peak_RSS_bytes': peak, 'max_CPU_threads': 4, 'GPU_hours': 0, 'new_download_bytes': 0,
            'output_bindings': [binding(p) for p in sorted(out.iterdir()) if p.is_file()],
            'report_bindings': [binding(p) for p in sorted(report.iterdir()) if p.is_file()],
            'limitations': ['Final risk cores trained these same Source records; OOF priors do not make final risk inference held out',
                            'Input scale is an artificial counterfactual; no Source mean variant or new upstream model',
                            'No confidence intervals, scale selection, serving publication, or new scientific confirmation'],}
        write_json(report / 'RESULT_MANIFEST.json', result, readonly=True)
        for p in out.iterdir():
            if p.is_file(): p.chmod(0o444)
        for p in report.iterdir():
            if p.is_file(): p.chmod(0o444)
        print(json.dumps({'phase': 'COMPLETE', 'result_manifest': binding(report / 'RESULT_MANIFEST.json'),
                          'elapsed_seconds': result['elapsed_seconds'], 'CPU_seconds': result['CPU_seconds'], 'peak_RSS_bytes': peak}), flush=True)
    except BaseException as error:
        if not (out / 'ABORT.json').exists():
            write_json(out / 'ABORT.json', {'schema': SCHEMA, 'status': 'ABORT', 'error_type': type(error).__name__,
                       'message': str(error), 'elapsed_seconds': time.monotonic() - started,
                       'new_fits': 0, 'Target_access': 0}, readonly=True)
        raise
    finally:
        signal.alarm(0)
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--report', default=str(DOC))
    prep.add_argument('--output', default=str(OUTPUT))
    actual = sub.add_parser('run')
    actual.add_argument('--root-approval', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.report, args.output)
    else:
        run(args.root_approval)


if __name__ == '__main__':
    main()

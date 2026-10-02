#!/usr/bin/env python3
"""Evaluate a sealed single-direction control proxy, no fits or new sampling."""
from pathlib import Path
import importlib.util
import json
import os
import resource
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import prepare_safeconf_orion_target_control_expression_agent as preparation
from tools.scripts.prepare_safeconf_orion_physical_content_null_scorecodec_agent import bind

BASE = preparation.BASE
PREP = preparation.REPORT
CONTROL = BASE / 'orion_existing_validation_risk_control_20261002_v1'
REFERENCES = BASE / 'orion_validation_strong_reference_completion_20261002_v1'
COMPARISON = BASE / 'orion_public_bank_extension_20261002_v4_partitioned/expanded_comparison'
TRUTH = BASE / 'orion_registered_test_extendedbank_20261002_v1/registered_test_truth/TEST_TASK_ERRORS.parquet'
METRIC = ROOT / 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'
DOC = PREP.parent / 'target_control_expression_actual_v1'
OUT = BASE / 'orion_target_control_expression_actual_20261002_v1'
NEW = preparation.METHOD
QUERY = preparation.QUERY
MISSING = ['Magnitude', 'P_only_ridge', 'P_only_hgb', 'Manual_ridge', 'Manual_hgb', 'Learned_ridge']
REFERENCE_NAMES = ['Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Learned_DirectRMSE', 'Manual_WeightedHistoryDistance']
ALIASES = {'Learned_hgb': 'FrozenLearnedSourceHGB',
           'Learned_WeightedHistoryDistance': 'LearnedWeightedDistance',
           'NegativeSourceHistorySupport': 'NegativeSourceSupport'}


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    began, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200 seconds exceeded')))
    signal.alarm(1200)
    if OUT.exists() or DOC.exists():
        raise RuntimeError('Fresh isolated evaluation roots required')
    receipt = json.loads((PREP / 'PREPARATION_RESULT.json').read_text())
    if (receipt['status'] != 'COMPLETE_FIXED232_TRAIN_CONTROL_PROXY_PREPARATION'
            or not receipt['higher_score_is_higher_risk'] or receipt['CP10000_executed']):
        raise RuntimeError('Exactly the sealed CP4000 single-direction preparation is required')
    for item in receipt['original_input_bindings'] + receipt['outputs']:
        if bind(item['path']) != item:
            raise RuntimeError('Preparation binding changed')
    if bind(METRIC)['sha256'] != 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c':
        raise RuntimeError('Original metric code differs')
    score_path = preparation.OUTPUT / 'FIXED_PRIMARY_TARGET_CONTROL_RISKS.parquet'
    inputs = [Path(__file__), PREP / 'PREPARATION_RESULT.json', score_path, METRIC, TRUTH,
              CONTROL / 'ALL_PREDICTIONS_SEAL.json', CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet',
              CONTROL / 'BOOTSTRAP_GENE_INDICES.npy', CONTROL / 'SHARED_BOOTSTRAP_METRIC_DRAWS.npy',
              REFERENCES / 'FOUR_REFERENCE_SAVED_DRAW_METRICS.npy',
              COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet']
    before = [bind(path) for path in inputs]
    DOC.mkdir(); OUT.mkdir()
    write_json(DOC / 'EVALUATION_REGISTRATION.json', {
        'role': 'SEEN_fixed_control_only_strong_comparator_not_new_method_or_confirmation',
        'method': NEW, 'direction': 'higher_expression_higher_risk; no sign or parameter selection',
        'input_domain': 'Existing38606gene owncontext TRAIN NTC, beyond original3285 P/PUBLIC feature domain',
        'cohort': 'All232original primary tasks/144genes, no87gene-readout subset selection',
        'comparison_scope': 'All13original rules andall30existing fixed target learners; no winner selection',
        'statistics': 'Same saved5000sorted144gene indices; no new bootstrap RNG',
        'new_fits': 0, 'new_upstream_calls': 0, 'new_expression_reads': 0,
        'new_C_error_labels_for_training': 0, 'new_primary': False, 'inputs': before})
    spec = importlib.util.spec_from_file_location('fixed_control_proxy_metrics', METRIC)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = pd.read_parquet(COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet')
    original_names = list(module.BASE_METHODS) + list(module.OPTIONAL) + [module.SUPPORT]
    control = pd.read_parquet(CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet')
    target_names = [n for n in control if n.startswith('B')]
    old_names = json.loads((CONTROL / 'ALL_PREDICTIONS_SEAL.json').read_text())['score_columns']
    frame = pd.read_parquet(score_path).merge(original[QUERY + original_names], on=QUERY, validate='one_to_one')
    frame = frame.merge(control[QUERY + target_names], on=QUERY, validate='one_to_one')
    truth = pd.read_parquet(TRUTH, columns=QUERY + ['true_error_rmse'], filters=[('query_id', 'in', frame.query_id.tolist())])
    frame = frame.merge(truth, on=QUERY, validate='one_to_one')
    if len(frame) != 232 or not frame.query_id.is_unique or len(target_names) != 30 or len(original_names) != 13:
        raise RuntimeError('Exact complete44method scope required')
    names = original_names + target_names + [NEW]
    if not np.isfinite(frame[names + ['true_error_rmse']].to_numpy(float)).all():
        raise RuntimeError('No task filtering, imputed zeros or unsupported scores allowed')
    old = np.load(CONTROL / 'SHARED_BOOTSTRAP_METRIC_DRAWS.npy', mmap_mode='r')
    refs = np.load(REFERENCES / 'FOUR_REFERENCE_SAVED_DRAW_METRICS.npy', mmap_mode='r')
    indices = np.load(CONTROL / 'BOOTSTRAP_GENE_INDICES.npy', mmap_mode='r')
    if old.shape != (5000, 3, 33, 7) or refs.shape != (5000, 3, 4, 7) or indices.shape != (5000, 144):
        raise RuntimeError('Original saved draw shapes differ')
    genes = sorted(frame.target_gene_id.astype(str).unique())
    if genes != sorted(control.target_gene_id.astype(str).unique()):
        raise RuntimeError('Exact sorted gene mapping required')
    if set(map(tuple, frame[QUERY].to_numpy())) != set(map(tuple, control[QUERY].to_numpy())):
        raise RuntimeError('Exact task set for paired saved draws required')
    blocks = [np.flatnonzero(frame.target_gene_id.astype(str).eq(gene)) for gene in genes]
    computed = MISSING + [NEW]
    points = module.metric_values(frame, names)
    new_draws = np.full((5000, 3, len(computed), 7), np.nan)
    contexts = ['HCT116', 'HEK293T', 'macro']
    for number, selected in enumerate(indices):
        rows = frame.iloc[np.concatenate([blocks[int(i)] for i in selected])].reset_index(drop=True)
        values = module.metric_values(rows, computed + ['Learned_hgb'])
        for ci, context in enumerate(contexts):
            for mi, name in enumerate(computed):
                new_draws[number, ci, mi] = [values[context][name][m] for m in module.METRICS]
            check = np.array([values[context]['Learned_hgb'][m] for m in module.METRICS])
            if not np.allclose(check, old[number, ci, old_names.index('FrozenLearnedSourceHGB')], rtol=0, atol=1e-15, equal_nan=True):
                raise RuntimeError('Saved-draw pairing must reproduce the original Source anchor')
        if number % 1000 == 0:
            print(json.dumps({'saved_draws_completed': number, 'total': 5000}), flush=True)
    np.save(OUT / 'SIX_ORIGINAL_PLUS_NEW_CONTROL_METRIC_DRAWS.npy', new_draws)
    def get_draw(name):
        if name in computed: return new_draws[:, :, computed.index(name)]
        if name in REFERENCE_NAMES: return refs[:, :, REFERENCE_NAMES.index(name)]
        return old[:, :, old_names.index(ALIASES.get(name, name))]
    intervals, comparisons = [], []
    for ci, context in enumerate(contexts):
        for name in names:
            draw = get_draw(name)[:, ci]
            for ki, metric in enumerate(module.METRICS):
                finite = draw[:, ki][np.isfinite(draw[:, ki])]
                lower, upper = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                intervals.append({'context': context, 'method': name, 'metric': metric, 'point': points[context][name][metric],
                    'ci95_lower': lower, 'ci95_upper': upper, 'valid_draws': len(finite), 'saved_draws': 5000})
        for name in names[:-1]:
            differences = get_draw(name)[:, ci] - get_draw(NEW)[:, ci]
            for ki, metric in enumerate(module.METRICS):
                finite = differences[:, ki][np.isfinite(differences[:, ki])]
                lower, upper = np.quantile(finite, [.025, .975]) if len(finite) >= 2 else (np.nan, np.nan)
                comparisons.append({'context': context, 'method_a': name, 'method_b': NEW, 'metric': metric,
                    'point_a': points[context][name][metric], 'point_b': points[context][NEW][metric],
                    'difference_a_minus_control': points[context][name][metric] - points[context][NEW][metric],
                    'ci95_lower': lower, 'ci95_upper': upper, 'valid_draws': len(finite), 'saved_draws': 5000,
                    'scope': 'SEEN_same_task_comparison; extra fullgene control information disclosed'})
    pd.DataFrame(intervals).to_csv(DOC / 'ALL_FIXED44_METHOD_METRIC_INTERVALS.csv', index=False)
    pd.DataFrame(comparisons).to_csv(DOC / 'ALL_FIXED43_METHODS_MINUS_TARGET_CONTROL.csv', index=False)
    if len(intervals) != 924 or len(comparisons) != 903:
        raise RuntimeError('All registered methods/contexts/metrics must remain visible')
    if before != [bind(path) for path in inputs]: raise RuntimeError('Original inputs changed during analysis')
    write_json(DOC / 'RESULT_MANIFEST.json', {
        'status': 'COMPLETE_FIXED_SEEN_TARGET_CONTROL_COMPARISON', 'fixed_tasks': 232, 'gene_clusters': 144,
        'methods': 44, 'paired_comparators': 43, 'metric_rows': 924, 'paired_rows': 903,
        'single_control_direction': True, 'new_fits': 0, 'new_model_calls': 0, 'new_bootstrap_draws': 0,
        'saved_draws': 5000, 'all5000Source_anchor_metric_arrays_reproduced': True,
        'extra_input_domain_full38606_control_not_same_P13': True, 'no_primary_change': True, 'no_new_confirmation': True,
        'elapsed_seconds': time.monotonic() - began, 'CPU_seconds': time.process_time() - cpu,
        'reported_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'original_input_bindings': before, 'outputs': [bind(p) for p in sorted(DOC.glob('*.csv'))]})
    print(json.dumps({'status': 'COMPLETE_FIXED_SEEN_TARGET_CONTROL_COMPARISON'}), flush=True)


if __name__ == '__main__':
    main()

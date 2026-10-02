#!/usr/bin/env python3
"""SEEN fixed-prediction comparison; reuse saved draws, zero fitting/sampling."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import resource
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001')
CONTROL = BASE / 'orion_existing_validation_risk_control_20261002_v1'
FROZEN = BASE / 'orion_registered_test_extendedbank_20261002_v1/fixed_risk_evaluation'
COMPARISON = BASE / 'orion_public_bank_extension_20261002_v4_partitioned/expanded_comparison'
TRUTH = BASE / 'orion_registered_test_extendedbank_20261002_v1/registered_test_truth/TEST_TASK_ERRORS.parquet'
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/orion_preparation/validation_strong_reference_completion_v1'
OUT = BASE / 'orion_validation_strong_reference_completion_20261002_v1'
CODE = ROOT / 'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py'
METRIC_SHA = 'dec8f056918ac5626eb12732eec3cc07041eb3680cc6addd6d9f18b32d1f486c'
NEW_REFERENCES = ['Uniform_DirectRMSE', 'Manual_DirectRMSE', 'Learned_DirectRMSE', 'Manual_WeightedHistoryDistance']
EXISTING_REFERENCE = 'LearnedWeightedDistance'
QUERY = ['query_id', 'target_gene_id', 'target_gene_symbol', 'context_id', 'role']


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    Path(path).chmod(0o444)


def main():
    started, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError('600-second bound exceeded')))
    signal.alarm(600)
    if OUT.exists() or DOC.exists():
        raise RuntimeError('Fresh isolated output required; no rerun overwrite')
    if binding(CODE)['sha256'] != METRIC_SHA:
        raise RuntimeError('Frozen metric implementation differs')
    files = [Path(__file__), CODE, CONTROL / 'ALL_PREDICTIONS_SEAL.json',
             CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet', CONTROL / 'SHARED_BOOTSTRAP_METRIC_DRAWS.npy',
             CONTROL / 'BOOTSTRAP_GENE_INDICES.npy', TRUTH,
             FROZEN / 'METHOD_METRIC_INTERVALS.csv', COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet']
    before = [binding(path) for path in files]
    OUT.mkdir(); DOC.mkdir()
    write_json(DOC / 'ANALYSIS_REGISTRATION.json', {
        'role': 'SEEN_POSTCONFIRMATION_STRONG_REFERENCE_COMPLETENESS_NOT_NEW_CONFIRMATION',
        'fixed_missing_references': NEW_REFERENCES, 'existing_reference': EXISTING_REFERENCE,
        'input_bindings': before, 'new_fits': 0, 'new_random_draws': 0, 'raw_expression_reads': 0,
        'same_saved5000_gene_indices_not_same_seed_assumption': True,
        'preserves_original13_primary_and30_target_models': True,
        'all30_methods_vs_all5rules_no_bestbaseline_selection': True,
        'resources': {'max_seconds': 600, 'CPU_threads': 4, 'GPU_hours': 0, 'downloads': 0}})
    seal = json.loads((CONTROL / 'ALL_PREDICTIONS_SEAL.json').read_text())
    names = seal['score_columns']
    targets = [name for name in names if name.startswith('B')]
    if len(targets) != 30 or len(names) != 33:
        raise RuntimeError('Fixed existing33-score scope differs')
    frame = pd.read_parquet(CONTROL / 'ALL_PRIMARY_PREDICTIONS.parquet')
    if len(frame) != 232 or frame.target_gene_id.nunique() != 144 or not frame.query_id.is_unique:
        raise RuntimeError('Same232task/144gene population required')
    frozen_scores = pd.read_parquet(COMPARISON / 'PRETRUTH_COMPARISON_SCORES.parquet',
        columns=QUERY + NEW_REFERENCES)
    frame = frame.merge(frozen_scores, on=QUERY, validate='one_to_one')
    truth = pd.read_parquet(TRUTH, columns=QUERY + ['true_error_rmse'],
        filters=[('query_id', 'in', frame.query_id.tolist())])
    frame = frame.merge(truth, on=QUERY, validate='one_to_one')
    if len(frame) != 232 or not np.isfinite(frame[NEW_REFERENCES + ['true_error_rmse']].to_numpy(float)).all():
        raise RuntimeError('No reference/truth survivor filtering permitted')
    old = np.load(CONTROL / 'SHARED_BOOTSTRAP_METRIC_DRAWS.npy', mmap_mode='r')
    indices = np.load(CONTROL / 'BOOTSTRAP_GENE_INDICES.npy', mmap_mode='r')
    if old.shape != (5000, 3, 33, 7) or indices.shape != (5000, 144):
        raise RuntimeError('Exact existing draw shapes required')
    genes = sorted(frame.target_gene_id.astype(str).unique())
    blocks = [np.flatnonzero(frame.target_gene_id.astype(str).eq(gene)) for gene in genes]
    spec = importlib.util.spec_from_file_location('fixed_reference_metrics', CODE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    contexts = ['HCT116', 'HEK293T', 'macro']
    original_valid = [int(frame.context_id.eq(c).sum()) >= 20 for c in contexts[:2]]
    original_valid.append(all(original_valid))
    point = module.metric_values(frame, names + NEW_REFERENCES)
    missing = np.full((5000, 3, 4, 7), np.nan)
    for number, chosen in enumerate(indices):
        sampled = frame.iloc[np.concatenate([blocks[int(i)] for i in chosen])].reset_index(drop=True)
        values = module.metric_values(sampled, NEW_REFERENCES + [EXISTING_REFERENCE])
        for ci, context in enumerate(contexts):
            if original_valid[ci]:
                for mi, method in enumerate(NEW_REFERENCES):
                    missing[number, ci, mi] = [values[context][method][m] for m in module.METRICS]
                reused = np.array([values[context][EXISTING_REFERENCE][m] for m in module.METRICS])
                if not np.allclose(reused, old[number, ci, names.index(EXISTING_REFERENCE)], rtol=0, atol=1e-15, equal_nan=True):
                    raise RuntimeError('Saved draw pairing/row order differs')
        if number % 1000 == 0:
            print(json.dumps({'completed_saved_draw_evaluations': number, 'total': 5000}), flush=True)
    np.save(OUT / 'FOUR_REFERENCE_SAVED_DRAW_METRICS.npy', missing)
    rows = []
    for ci, context in enumerate(contexts):
        for method in targets:
            for reference in NEW_REFERENCES + [EXISTING_REFERENCE]:
                ref_draw = missing[:, ci, NEW_REFERENCES.index(reference)] if reference in NEW_REFERENCES else old[:, ci, names.index(reference)]
                differences = old[:, ci, names.index(method)] - ref_draw
                for ki, metric in enumerate(module.METRICS):
                    finite = differences[:, ki][np.isfinite(differences[:, ki])]
                    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) >= 2 else (np.nan, np.nan)
                    rows.append({'context': context, 'target_method': method, 'reference': reference,
                        'metric': metric, 'target_point': point[context][method][metric],
                        'reference_point': point[context][reference][metric],
                        'difference_a_minus_b': point[context][method][metric] - point[context][reference][metric],
                        'ci95_lower': lo, 'ci95_upper': hi, 'valid_draws': len(finite),
                        'saved_draws': 5000, 'new_draws': 0, 'original_context_valid': original_valid[ci],
                        'postconfirmation_completeness_analysis': True})
    if len(rows) != 30 * 5 * 3 * 7:
        raise RuntimeError('All reference comparisons must remain visible')
    pd.DataFrame(rows).to_csv(DOC / 'ALL_TARGET_VS_ALL_FIVE_HISTORY_RULES.csv', index=False)
    original = pd.read_csv(FROZEN / 'METHOD_METRIC_INTERVALS.csv')
    for context in contexts:
        for reference in NEW_REFERENCES:
            for metric in module.METRICS:
                match = original[(original.cohort.eq('primary_common')) & original.context.eq(context)
                                 & original.method.eq(reference) & original.metric.eq(metric)]
                if len(match) != 1 or not np.isclose(point[context][reference][metric], match.point.iloc[0], rtol=0, atol=1e-15, equal_nan=True):
                    raise RuntimeError('Frozen baseline points not reproduced')
    if before != [binding(path) for path in files]:
        raise RuntimeError('Existing artifacts changed during analysis')
    result = {'status': 'COMPLETE_SEEN_SUPPLEMENT_NO_PROMOTION', 'input_bindings': before,
        'target_models': 30, 'all_history_rules': 5, 'metric_comparison_rows': len(rows),
        'existing_draws': 5000, 'new_draws': 0, 'new_fits': 0, 'upstream_calls': 0,
        'Source_or_Public_changes': 0, 'new_primary_or_confirmation': False,
        'all5000_existing_weighted_reference_draw_metrics_reproduced': True,
        'all84_original_missing_reference_context_metric_points_reproduced': True,
        'elapsed_seconds': time.monotonic() - started, 'CPU_seconds': time.process_time() - cpu,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'result': binding(DOC / 'ALL_TARGET_VS_ALL_FIVE_HISTORY_RULES.csv')}
    write_json(DOC / 'RESULT_MANIFEST.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()

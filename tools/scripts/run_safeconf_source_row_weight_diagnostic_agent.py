#!/usr/bin/env python3
"""Fixed, information-free training-copy diagnostic on released Source/queries.

This is a SEEN diagnostic, never an Orion method change or target selection.
Unique-source/pool scores must reproduce the archived original seed before any
duplicate or weight-scale controls are fitted. CDFs see original records only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.learners import NumericPreprocessor
from tools.safeconf_continual.research import (
    FittedRisk, P, PUBLIC, cluster_weights, fit_risk, ids_hash, metrics, rank_labels,
)

RUNTIME = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
DOC = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results/source_row_weight_diagnostic'
SEED = 20260930
REPLICATES = 5000
PARAMS = dict(max_iter=200, learning_rate=.05, max_depth=3,
              min_samples_leaf=20, l2_regularization=10., random_state=SEED)
PAIRS = [
    ('GAT_duplicate2', 'GAT_unique'),
    ('Exphormer_duplicate2', 'Exphormer_unique'),
    ('GAT_duplicate2', 'Full_pool_standard'),
    ('Exphormer_duplicate2', 'Full_pool_standard'),
    ('Full_pool_single_source_weight', 'Full_pool_standard'),
    ('Full_pool_standard', 'GAT_unique'),
    ('Full_pool_standard', 'Exphormer_unique'),
]
OLD_NAMES = {'GAT_unique': 'TxPert_GAT', 'Exphormer_unique': 'TxPert_Exphormer',
             'Full_pool_standard': 'PooledFullClusterWeight'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scaled_fit(frame, labels, total_weight):
    valid = np.isfinite(labels)
    fit = frame.loc[valid].copy()
    prep = NumericPreprocessor().fit(fit[P + PUBLIC].to_numpy(float))
    weights = cluster_weights(fit)
    weights *= total_weight / weights.sum()
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(prep.transform(fit[P + PUBLIC].to_numpy(float)), labels[valid],
              sample_weight=weights)
    return FittedRisk(prep, model, P + PUBLIC), weights


def bootstrap_all(query, scores):
    """One gene-block draw jointly across all three contexts and all controls."""
    query = query.reset_index(drop=True)
    truth = query.true_error_rmse.to_numpy(float)
    ids = query.task_id.to_numpy(str)
    contexts = query.target.to_numpy(str)
    genes = query.gene.to_numpy(str)
    names = list(scores)
    matrix = np.column_stack([scores[name] for name in names])
    if not np.isfinite(truth).all() or not np.isfinite(matrix).all():
        raise RuntimeError('No diagnostic cohort filtering is permitted')
    clusters = sorted(set(genes))
    blocks = [np.flatnonzero(genes == gene) for gene in clusters]
    context_names = sorted(set(contexts))
    for context in context_names:
        if (contexts == context).sum() < 20:
            raise RuntimeError('Original context has fewer than 20 tasks')

    def utilities(use):
        result = []
        for context in context_names:
            rows = use[contexts[use] == context]
            if len(rows) < 20:
                return np.full(len(names), np.nan)
            error = truth[rows]
            count = math.ceil(.2 * len(rows))
            oracle = np.lexsort((ids[rows], -error))[:count]
            mean = error.mean()
            denominator = error[oracle].mean() - mean
            if denominator <= 1e-12:
                return np.full(len(names), np.nan)
            result.append([(error[np.lexsort((ids[rows], -matrix[rows, j]))[:count]].mean()
                            - mean) / denominator for j in range(len(names))])
        return np.mean(result, axis=0)

    observed = utilities(np.arange(len(query)))
    # Check the compact helper against the original research metric, all scenarios.
    original = np.array([np.mean([metrics(query.loc[contexts == c], scores[name][contexts == c])['utility20']
                                  for c in context_names]) for name in names])
    if not np.allclose(observed, original, atol=1e-14, rtol=0):
        raise RuntimeError('Compact U20 helper differs from original research.metrics')
    rng = np.random.default_rng(SEED)
    draws = np.asarray([utilities(np.concatenate([blocks[j] for j in
                        rng.integers(0, len(blocks), len(blocks))])) for _ in range(REPLICATES)])
    records = []
    for a, b in PAIRS:
        delta = draws[:, names.index(a)] - draws[:, names.index(b)]
        finite = np.isfinite(delta)
        if finite.sum() != REPLICATES:
            raise RuntimeError('A planned joint bootstrap draw is invalid')
        low, high = np.quantile(delta, [.025, .975], method='linear')
        records.append({'method_a': a, 'method_b': b,
                        'delta_utility20_a_minus_b': float(observed[names.index(a)] - observed[names.index(b)]),
                        'bootstrap_mean_delta': float(delta.mean()),
                        'ci95_lower': float(low), 'ci95_upper': float(high),
                        'bootstrap_replicates': REPLICATES, 'valid_draws': int(finite.sum()),
                        'bootstrap_seed': SEED, 'n_tasks': len(query),
                        'n_gene_clusters': len(clusters), 'n_contexts': len(context_names),
                        'macro': 'equal context mean', 'direction': 'positive favors method_a'})
    return records, draws, names


def main():
    began = time.monotonic()
    cache = RUNTIME / 'source_scaling_recheck/cache'
    old_path = RUNTIME / 'source_scaling_recheck/SOURCE_DIVERSITY_PREDICTIONS.csv.gz'
    output = RUNTIME / 'source_row_weight_diagnostic'
    if output.exists() or DOC.exists():
        raise RuntimeError('Existing results cannot be overwritten; choose a new version')
    inputs = [old_path] + [cache / f'{domain}_{ref}.parquet'
                           for ref in ('Manual', 'Learned') for domain in ('source', 'external')]
    dependencies = [Path(__file__), ROOT / 'tools/safeconf_continual/research.py',
                    ROOT / 'tools/safeconf_continual/learners.py',
                    ROOT / 'tools/safeconf_continual/contracts.py']
    registration = {
        'role': 'SEEN_POST_CONFIRMATION_ALGORITHMIC_TRAINING_COPY_DIAGNOSTIC',
        'purpose': 'Separate additional predictor information from training-row and absolute-weight effects',
        'seed': SEED, 'bootstrap_seed': SEED, 'bootstrap_replicates': REPLICATES,
        'common_gene_axis': 2840, 'features': P + PUBLIC, 'hgb_parameters': PARAMS,
        'scenarios': list(OLD_NAMES) + ['GAT_duplicate2', 'Exphormer_duplicate2', 'Full_pool_single_source_weight'],
        'paired_comparisons': [list(pair) for pair in PAIRS],
        'CDF_policy': 'Fit original unique allowed Source records only; copy original labels for repeated training rows',
        'duplicate_policy': 'Two identical training copies add zero independent records or predictor information',
        'matched_weight_policy': 'Full pool gene-cluster weights scaled to original single-source total weight',
        'query_truth_use': 'Already released McFaline evaluation only, never training/CDF/model selection',
        'new_upstream_calls': 0, 'new_upstream_predictors': 0, 'parameter_search': False,
        'Orion_access': False, 'frozen_method_or_existing_model_change': False,
        'input_bindings': [{'path': str(path), 'sha256': sha(path)} for path in inputs],
        'code_bindings': [{'path': str(path), 'sha256': sha(path)} for path in dependencies],
        'software': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'leaf_count_caveat': 'min_samples_leaf=20 counts training rows. Exact duplication can admit leaves with fewer original records; matching absolute sample-weight sum does not match row counts or histogram construction.',
    }
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    DOC.mkdir(parents=True)
    write_json(DOC / 'REGISTRATION.json', registration)
    old = pd.read_csv(old_path)
    states = {}
    audits, costs, budgets, reproduced = [], [], [], []

    def fit_one(ref, name, frame, labels, original_frame, weight_target=None):
        started = time.monotonic()
        if weight_target is None:
            model = fit_risk(frame, labels, P + PUBLIC, 'hgb', SEED)
            weights = cluster_weights(frame.loc[np.isfinite(labels)])
        else:
            model, weights = scaled_fit(frame, labels, weight_target)
        score = model.predict(states[ref]['query'])
        path = output / 'models' / f'{ref}_{name}.joblib'
        joblib.dump(model, path)
        # Same saved model must preserve frozen diagnostic predictions on reload.
        reloaded = joblib.load(path).predict(states[ref]['query'])
        if not np.array_equal(reloaded, score):
            raise RuntimeError('Diagnostic saved-model predictions changed on reload')
        leaves = np.concatenate([stage[0].nodes['count'][stage[0].nodes['is_leaf'].astype(bool)]
                                 for stage in model.model._predictors])
        original_ids = original_frame.upstream.astype(str) + '::' + original_frame.task_id.astype(str)
        training_ids = frame.upstream.astype(str) + '::' + frame.task_id.astype(str)
        budgets.append({'reference': ref, 'scenario': name,
                        'unique_gene_clusters': int(original_frame.gene.nunique()),
                        'original_unique_predictor_task_records': len(set(original_ids)),
                        'original_biological_task_ids': int(original_frame.task_id.nunique()),
                        'training_rows': len(frame), 'finite_training_labels': int(np.isfinite(labels).sum()),
                        'unique_upstream_predictors': int(original_frame.upstream.nunique()),
                        'independent_new_records': 0,
                        'sum_training_weights': float(weights.sum()),
                        'weight_scale_from_mean_one': float(weights.sum() / len(weights)),
                        'min_samples_leaf': 20, 'minimum_observed_leaf_training_count': int(leaves.min()),
                        'total_tree_leaves': int(len(leaves)),
                        'original_record_ids_hash': ids_hash(set(original_ids)),
                        'training_record_ids_with_copies_hash': ids_hash(training_ids),
                        'gene_clusters_hash': ids_hash(original_frame.gene.unique()),
                        'original_CDF_labels_sha256': hashlib.sha256(np.asarray(labels[:len(original_frame)], '<f8').tobytes()).hexdigest(),
                        'model_path': str(path), 'model_sha256': sha(path)})
        costs.append({'reference': ref, 'scenario': name, 'fit_and_reload_seconds': time.monotonic() - started})
        states[ref]['scores'][name] = score
        states[ref]['models'][name] = model
        print(json.dumps({'phase': 'fit', 'reference': ref, 'scenario': name,
                          'training_rows': len(frame), 'weight_sum': float(weights.sum())}), flush=True)

    # Reproduce every unique/pool original case before fitting new controls.
    for ref in ('Manual', 'Learned'):
        source = pd.read_parquet(cache / f'source_{ref}.parquet')
        query = pd.read_parquet(cache / f'external_{ref}.parquet').reset_index(drop=True)
        if len(source) != 3616 or len(query) != 543 or query.task_id.duplicated().any():
            raise RuntimeError('Registered Source or released query cohort differs')
        if source.duplicated(['upstream', 'task_id']).any() or not np.isfinite(query.true_error_rmse).all():
            raise RuntimeError('Unique Source identity or released query truth fails')
        if set(source.upstream) != {'TxPert_GAT', 'TxPert_Exphormer'}:
            raise RuntimeError('Source predictor identities differ')
        frames = {'GAT_unique': source[source.upstream.eq('TxPert_GAT')].reset_index(drop=True),
                  'Exphormer_unique': source[source.upstream.eq('TxPert_Exphormer')].reset_index(drop=True),
                  'Full_pool_standard': source}
        states[ref] = {'source': source, 'query': query, 'frames': frames, 'labels': {},
                       'scores': {}, 'models': {}}
        for name, frame in frames.items():
            labels, audit = rank_labels(frame, f'row-weight-diagnostic/{ref}/{name}')
            if not np.isfinite(labels).all():
                raise RuntimeError('Original allowed Source CDF has missing labels')
            states[ref]['labels'][name] = labels
            audits.extend(audit)
            fit_one(ref, name, frame, labels, frame)
            archive = old[old.method.eq(f'{ref}/{OLD_NAMES[name]}') & old.seed.eq(SEED)]
            fresh = pd.DataFrame({'task_id': query.task_id, 'risk_new': states[ref]['scores'][name]})
            paired = fresh.merge(archive[['task_id', 'risk']], on='task_id', validate='one_to_one')
            difference = float(np.max(np.abs(paired.risk_new - paired.risk)))
            if len(paired) != 543 or difference > 1e-14:
                raise RuntimeError('Original unique/pool scores failed reproduction')
            reproduced.append({'reference': ref, 'scenario': name, 'query_tasks': len(paired),
                               'max_abs_score_difference': difference, 'tolerance': 1e-14})
    pd.DataFrame(reproduced).to_csv(DOC / 'ORIGINAL_REPRODUCTION.csv', index=False)
    print(json.dumps({'phase': 'all_original_cases_reproduced', 'cases': len(reproduced),
                      'maximum_difference': max(row['max_abs_score_difference'] for row in reproduced)}), flush=True)

    prediction_frames, strata, macro, comparisons, preprocess = [], [], [], [], []
    for ref, state in states.items():
        single_budget = len(state['frames']['GAT_unique'])
        if single_budget != len(state['frames']['Exphormer_unique']) or single_budget != 1808:
            raise RuntimeError('Single-source budgets differ')
        for original, repeated in [('GAT_unique', 'GAT_duplicate2'),
                                   ('Exphormer_unique', 'Exphormer_duplicate2')]:
            frame = state['frames'][original]
            copied = pd.concat([frame, frame], ignore_index=True)
            copied_labels = np.tile(state['labels'][original], 2)
            fit_one(ref, repeated, copied, copied_labels, frame)
            a, b = state['models'][original].preprocessor, state['models'][repeated].preprocessor
            preprocess.append({'reference': ref, 'original': original, 'duplicate': repeated,
                               'median_max_abs_difference': float(np.max(np.abs(a.medians_ - b.medians_))),
                               'center_max_abs_difference': float(np.max(np.abs(a.center_ - b.center_))),
                               'scale_max_abs_difference': float(np.max(np.abs(a.scale_ - b.scale_)))})
        pool = state['frames']['Full_pool_standard']
        fit_one(ref, 'Full_pool_single_source_weight', pool, state['labels']['Full_pool_standard'],
                pool, single_budget)
        query = state['query']
        for name, score in state['scores'].items():
            part = query[['task_id', 'target', 'gene', 'fold', 'upstream', 'true_error_rmse']].copy()
            part['reference'] = ref
            part['scenario'] = name
            part['seed'] = SEED
            part['risk'] = score
            prediction_frames.append(part)
            context_metrics = []
            for context in sorted(query.target.unique()):
                use = query.target.eq(context).to_numpy()
                measured = metrics(query.loc[use], score[use])
                context_metrics.append(measured)
                strata.append({'reference': ref, 'scenario': name, 'target': context, **measured})
            macro.append({'reference': ref, 'scenario': name, 'n_contexts': len(context_metrics),
                          'query_tasks': len(query), 'query_gene_clusters': int(query.gene.nunique()),
                          **{key: float(np.mean([row[key] for row in context_metrics]))
                             for key in ['utility20', 'spearman', 'aurc', 'high_risk_miss_rate',
                                         'error_at_10', 'error_at_20', 'error_at_50']}})
        compared, draws, names = bootstrap_all(query, state['scores'])
        comparisons.extend([{'reference': ref, **row} for row in compared])
        np.savez_compressed(output / f'{ref}_JOINT_GENE_BOOTSTRAP.npz',
                            macro_utility20=draws, scenario=np.asarray(names), seed=np.asarray(SEED))
        print(json.dumps({'phase': 'paired_gene_bootstrap_complete', 'reference': ref,
                          'replicates': REPLICATES, 'comparisons': len(compared)}), flush=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / 'TASK_PREDICTIONS.csv.gz', index=False)
    tables = {'INFORMATION_WEIGHT_LEDGER.csv': budgets, 'FIT_COSTS.csv': costs,
              'CDF_ORIGINAL_RECORDS_AUDIT.csv': audits, 'STRATA.csv': strata, 'MACRO.csv': macro,
              'PAIRED_GENE_U20.csv': comparisons, 'DUPLICATE_PREPROCESSOR_DIFFERENCES.csv': preprocess}
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(DOC / name, index=False, lineterminator='\n')
    status = {'status': 'COMPLETE', 'role': registration['role'],
              'fitted_table_models': len(costs), 'original_reproduction_cases': len(reproduced),
              'maximum_original_score_difference': max(row['max_abs_score_difference'] for row in reproduced),
              'paired_comparisons': len(comparisons), 'bootstrap_replicates_per_reference': REPLICATES,
              'all_saved_models_reload_predictions_byte_identical': True,
              'elapsed_seconds': time.monotonic() - began, 'new_upstream_calls': 0,
              'Orion_access': False, 'parameter_search': False, 'target_winner_selected': False,
              'original_models_and_outputs_preserved': True, 'server_output': str(output),
              'script_sha256': sha(__file__),
              'output_bindings': [{'path': str(path), 'sha256': sha(path)}
                                  for path in sorted(output.rglob('*')) if path.is_file()]}
    write_json(DOC / 'STATUS.json', status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Replay the frozen source risk learner on a cell-aligned public reference.

Source training rows, labels, features, parameters and seed are unchanged.
First require reproduction of the frozen guide-reference scores. This is an
already-seen contract diagnostic, not model selection or new confirmation.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, rank_labels, fit_risk, summarize, bootstrap_u20
from tools.scripts import run_safeconf_common_axis_closure as common


def main():
    out = common.DOC / 'reference_estimand_risk_replay'
    runtime = common.COMMON / 'reference_estimand_risk_replay'
    if (out / 'STATUS.json').exists():
        raise RuntimeError('completed diagnostic cannot be overwritten')
    out.mkdir(parents=True, exist_ok=True); runtime.mkdir(parents=True, exist_ok=True)
    cache = common.COMMON / 'risk_cache'
    train = pd.read_parquet(cache / 'source_Manual.parquet')
    query = pd.read_parquet(cache / 'external_Manual.parquet')
    alternate_path = common.COMMON / 'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy'
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'REGISTERED_DIAGNOSTIC.json').write_text(json.dumps({
        'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC',
        'hypothesis': 'within-history estimand mismatch changes frozen source-risk inputs',
        'source_train_sha256': digest(cache / 'source_Manual.parquet'),
        'frozen_query_features_sha256': digest(cache / 'external_Manual.parquet'),
        'alternate_public_effects_sha256': digest(alternate_path),
        'fixed_learner': 'original HGB, original P+PUBLIC features, original seed',
        'new_target_error_training_labels': 0, 'new_upstream_calls': 0,
        'parameter_search': False, 'replay_must_reproduce_saved_scores': True,
    }, indent=2) + '\n')
    labels, cdf = rank_labels(train, 'estimand_replay/full_source_only')
    start = time.perf_counter()
    model = fit_risk(train, labels, P + PUBLIC, 'hgb', SEEDS[0])
    fit_seconds = time.perf_counter() - start
    original_score = model.predict(query)
    saved = pd.read_csv(common.DOC / 'results/MATRIX_TASK_PREDICTIONS.csv.gz')
    saved = saved[(saved.line == 'TxPert_to_McFaline') & (saved.method == 'Manual_hgb') &
                  (saved.seed == SEEDS[0])].set_index('task_id').loc[query.task_id]
    if not np.allclose(original_score, saved.risk, rtol=1e-10, atol=1e-10):
        raise RuntimeError('source learner replay did not reproduce original risk')
    query['true_error_rmse'] = saved.true_error_rmse.to_numpy(float)
    memory, _, controls, _ = PublicMemoryStore(common.COMMON / 'public_mcfaline_trainval').load()
    effects = np.load(alternate_path)
    prediction = np.load(common.COMMON / 'TEST_CALIBRATED_EFFECTS.npy')
    pairs = common.closure.mc.build_pairs(query, np.load(common.COMMON / 'TEST_CONTROLS.npy'),
                                        memory, effects, controls, None)
    priors = np.empty_like(prediction, dtype=float); summaries = []
    for q, group in pairs.groupby('task_row', sort=True):
        vectors = effects[group.memory_row.to_numpy(int)]
        weights = np.expm1(group.log_source_cells.to_numpy(float)); weights /= weights.sum()
        mean = weights @ vectors
        priors[q] = mean
        summaries.append({'task_id': query.iloc[q].task_id, 'reference': 'Manual',
            'prior_uncertainty': np.sqrt(weights @ np.mean((vectors - mean) ** 2, axis=1)),
            'effective_sources': 1 / (weights @ weights),
            'history_conflict': float(group.source_conflict.mean()),
            'log_history_support': float(np.log1p(np.expm1(group.log_source_cells).sum()))})
    alternate = common.closure.reference_frame(query, prediction, priors, pd.DataFrame(summaries), 'Manual')
    if not np.allclose(query.log_history_support, alternate.log_history_support, atol=1e-12):
        raise RuntimeError('history support changed in estimand replay')
    alternate_score = model.predict(alternate)
    distance = np.sqrt(alternate.prediction_prior_rmse ** 2 + alternate.prior_uncertainty ** 2).to_numpy()
    scores = {'FrozenManualHGBReplay': original_score, 'CellAlignedManualHGB': alternate_score,
              'CellWeightedHistoryDistance': distance, 'NegativeHistorySupport': -query.log_history_support.to_numpy()}
    records = []
    for name, score in scores.items():
        part = query[['task_id', 'target', 'gene', 'true_error_rmse']].copy()
        part['risk'] = score; part['method'] = name; part['seed'] = SEEDS[0]
        part['line'] = 'TxPert_to_McFaline'; records.append(part)
    predictions = pd.concat(records, ignore_index=True)
    predictions.to_csv(runtime / 'TASK_PREDICTIONS.csv.gz', index=False)
    strata, macro = summarize(predictions)
    strata.to_csv(out / 'STRATA.csv', index=False); macro.to_csv(out / 'MACRO.csv', index=False)
    comparisons = [{'method_a': 'CellAlignedManualHGB', 'method_b': name,
        **bootstrap_u20(query, alternate_score, score)} for name, score in scores.items()
        if name != 'CellAlignedManualHGB']
    pd.DataFrame(comparisons).to_csv(out / 'PAIRED_COMPARISONS.csv', index=False)
    pd.DataFrame(cdf).to_csv(out / 'SOURCE_CDF_AUDIT.csv', index=False)
    status = {'status': 'COMPLETE', 'n_tasks': len(query), 'source_training_rows': len(train),
        'source_training_clusters': train.gene.nunique(), 'source_training_predictors': train.upstream.nunique(),
        'replayed_fits': 1, 'new_target_risk_training_labels': 0, 'new_upstream_calls': 0,
        'original_scores_reproduced': True, 'fitting_seconds': fit_seconds,
        'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC', 'promoted_as_final_method': False}
    (out / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)
    print(macro[['method', 'utility20', 'spearman', 'aurc']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

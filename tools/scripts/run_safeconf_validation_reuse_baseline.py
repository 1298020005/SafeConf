#!/usr/bin/env python3
"""Existing target-validation feedback as a budget/cost-identified strong control.

Validation perturbation clusters overlapping any test cluster are excluded
from risk fitting/CDFs. Their already-authorized biology/upper-model selection
uses are disclosed separately. This is an additional SEEN control, not a zero
target-label method or a replacement of the frozen confirmation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, bootstrap_u20, fit_risk, ids_hash, rank_labels, summarize,
    paired_prediction_wide,
)
from tools.scripts import run_safeconf_research_closure as closure
from tools.scripts import run_safeconf_common_axis_closure as canonical

COMMON = canonical.COMMON
RESULTS = canonical.DOC / 'results'
OUT = RESULTS / 'validation_reuse_baseline'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'STATUS.json').exists():
        raise FileExistsError('completed validation-reuse comparison is immutable')
    memory, effects, controls, _ = closure.PublicMemoryStore(COMMON / 'public_mcfaline_trainval').load()
    validation = pd.read_csv(COMMON / 'VALIDATION_BIOLOGY_TASKS.csv')
    validation['gene'] = validation.perturbation
    validation['target'] = validation.context
    validation['fold'] = validation.gene.map(lambda g: int.from_bytes(hashlib.sha256(
        f'SafeConf-McFaline-Decoder-repair-v1\0{g}'.encode()).digest()[:8], 'big') % 5)
    test = pd.read_csv(COMMON / 'TEST_TASKS.csv')
    query_tasks = test.rename(columns={'gene': 'perturbation'}).copy()
    query_tasks['gene'] = query_tasks.perturbation; query_tasks['target'] = query_tasks.context
    query_tasks['fold'] = -1
    val_truth = np.load(COMMON / 'VALIDATION_TRUE_EFFECTS.npy')
    val_control = np.load(COMMON / 'VALIDATION_CONTROLS.npy')
    val_pred = np.load(COMMON / 'VALIDATION_CALIBRATED_EFFECTS.npy')
    priors, _, audit, _ = closure.mc.public_priors(
        validation, query_tasks, val_truth, val_control, np.load(COMMON / 'TEST_CONTROLS.npy'),
        memory, effects, controls)
    frame = closure.mc.add_prior_features(validation, val_pred, priors, audit)
    frame['true_error_rmse'] = closure.tx.rmse_rows(val_pred.astype(float), val_truth.astype(float))
    frame = closure.annotate(frame, 'McFaline23', 'DecoderOnly',
        'decoder_frozen_alpha025_consistent_trainmean_common2840_v1', 'McFaline_common2840gene_log1p_delta_v1')
    for column in PUBLIC:
        frame[column] = frame[f'LearnedHGBRegularized_{column}']
    blocked = set(test.gene.astype(str))
    fit = frame[~frame.gene.astype(str).isin(blocked)].reset_index(drop=True)
    if not len(fit) or set(fit.gene) & blocked:
        raise RuntimeError('no legal validation risk rows or test cluster overlap')
    query = pd.read_parquet(COMMON / 'risk_cache/external_Learned.parquet')
    source = pd.read_parquet(COMMON / 'risk_cache/source_Learned.parquet')
    source_labels, source_audit = rank_labels(source, 'validation-reuse/shared-core')
    fixed_shared = fit_risk(source, source_labels, P + PUBLIC, 'hgb', SEEDS[0])
    fit['shared_risk'] = fixed_shared.predict(fit)
    original = pd.read_csv(RESULTS / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    original = original[original.line.eq('TxPert_to_McFaline') & original.seed.eq(SEEDS[0])]
    shared = original[original.method.eq('Learned_hgb')].set_index('task_id').risk
    query['shared_risk'] = query.task_id.map(shared)
    labels, cdf = rank_labels(fit, 'validation-reuse/legal-target-errors')
    cdf.extend(source_audit)
    records, costs, ledger = [], [], []
    choices = [('ValidationOnly_Ridge', P, 'ridge'), ('ValidationOnly_HGB', P, 'hgb'),
               ('PublicValidation_HGB', P + PUBLIC, 'hgb'),
               ('SharedPublicValidation_HGB', P + PUBLIC + ['shared_risk'], 'hgb')]
    for method, columns, kind in choices:
        for seed in SEEDS:
            started = time.monotonic()
            learner = fit_risk(fit, labels, columns, kind, seed)
            scores = learner.predict(query)
            closure.add_predictions(records, query, scores, 'TxPert_to_McFaline', method, seed)
            costs.append({'method': method, 'seed': seed, 'fit_and_predict_seconds': time.monotonic() - started,
                          'new_upstream_calls': 0, 'n_target_validation_errors': len(fit)})
        ledger.append({'method': method, 'c_validation_risk_error_rows': len(fit),
                       'c_validation_risk_error_clusters': fit.gene.nunique(),
                       'c_validation_upstream_selection_and_calibration_rows': len(validation),
                       'c_validation_biology_rows': len(validation) if set(columns) & set(PUBLIC) else 0,
                       'target_feedback_error_rows': 0, 'new_upstream_calls': 0,
                       'risk_training_records_hash': ids_hash(fit.task_id),
                       'heldout_cluster_overlap': 0,
                       'source_error_score_used': 'shared_risk' in columns,
                       'role': 'SEEN_VALIDATION_REUSE_ADDITIONAL_INFORMATION_CONTROL'})
    predictions = pd.concat(records, ignore_index=True)
    # Drop query truth for the freeze artifact; labels are already SEEN but may
    # not influence these fixed fits or the set of comparison methods.
    score_file = OUT / 'RISK_PREDICTIONS_BEFORE_EVALUATION.csv.gz'
    closure.tx.atomic_csv(score_file, predictions.drop(columns='true_error_rmse'))
    closure.tx.atomic_json(OUT / 'PREDICTION_FREEZE.json', {
        'risk_prediction_sha256': hashlib.sha256(score_file.read_bytes()).hexdigest(),
        'new_pristine_confirmation_claim': False,
        'n_legal_validation_risk_rows': len(fit), 'n_legal_validation_risk_clusters': fit.gene.nunique(),
        'risk_parameters': 'original registered Ridge/HGB; no hyperparameter selection',
        'upstream_validation_selection_bias_disclosed': True})
    closure.tx.atomic_csv(OUT / 'TASK_PREDICTIONS.csv.gz', predictions)
    strata, macro = summarize(predictions)
    for filename, value in [('MACRO.csv', macro), ('STRATA.csv', strata), ('CDF_AUDIT.csv', pd.DataFrame(cdf)),
                            ('INFORMATION_LEDGER.csv', pd.DataFrame(ledger)), ('COSTS.csv', pd.DataFrame(costs))]:
        closure.tx.atomic_csv(OUT / filename, value)
    # Equivalent serialized truths validated before joining; no floats in keys.
    joint = pd.concat([original, predictions[predictions.seed.eq(SEEDS[0])]], ignore_index=True)
    wide = paired_prediction_wide(joint)
    support = pd.read_csv(RESULTS / 'SUPPORT_CONTROL_TASK_PREDICTIONS.csv.gz')
    support = support[support.line.eq('TxPert_to_McFaline') & support.method.eq('NegativeHistorySupport') & support.seed.eq(SEEDS[0])]
    wide['NegativeHistorySupport'] = wide.task_id.map(support.set_index('task_id').risk)
    comparisons = [('PublicValidation_HGB', 'Learned_hgb'),
                   ('PublicValidation_HGB', 'Learned_WeightedHistoryDistance'),
                   ('PublicValidation_HGB', 'NegativeHistorySupport'),
                   ('SharedPublicValidation_HGB', 'PublicValidation_HGB'),
                   ('ValidationOnly_HGB', 'Learned_hgb')]
    statistics = []
    for a, b in comparisons:
        if wide[[a, b]].isna().any().any():
            raise RuntimeError('validation-reuse paired task coverage changed')
        statistics.append({'method_a': a, 'method_b': b,
                           **bootstrap_u20(wide, wide[a].to_numpy(), wide[b].to_numpy(), 5000)})
    closure.tx.atomic_csv(OUT / 'PAIRED_BOOTSTRAP.csv', pd.DataFrame(statistics))
    closure.tx.atomic_json(OUT / 'STATUS.json', {'status': 'COMPLETE', 'role': 'SEEN_POST_CONFIRMATION',
        'n_test_tasks': len(query), 'n_test_clusters': query.gene.nunique(),
        'n_validation_tasks_before_filter': len(frame), 'n_legal_validation_risk_rows': len(fit),
        'n_legal_validation_risk_clusters': fit.gene.nunique(), 'new_upstream_calls': 0,
        'no_target_validation_cdf_budget_hidden': True, 'no_gene_cluster_overlap': True,
        'validation_errors_not_strict_oof_upstream': 'frozen checkpoint and alpha used validation for selection; declared held-out-gradient errors, not pristine OOF errors'})
    print(macro[macro.seed.eq(SEEDS[0])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

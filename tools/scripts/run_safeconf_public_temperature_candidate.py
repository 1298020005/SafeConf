#!/usr/bin/env python3
"""The single registered Public weighting change; development evidence only.

For each public fit, tau is the median absolute biological-transfer residual
from grouped OOF predictions contained inside that fit. Thus an inner risk
training query does not enter its own temperature. No temperature sweep and
no external test selection are allowed. The existing frozen method is kept.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, fit_risk, ids_hash, metrics, rank_labels, summarize
from tools.scripts import run_safeconf_research_closure as closure

OUT = closure.OUT / 'public_temperature_candidate'


def public_fit(fit, query):
    if set(fit.gene) & set(query.gene):
        raise RuntimeError('public temperature public fit crosses query clusters')
    return closure.tx.fit_predict_regressor(fit, query, closure.tx.PAIR_FEATURES,
                                           fit.transfer_rmse.to_numpy(float), 'hgb')


def training_tau(fit, scope, audit):
    fit = fit.reset_index(drop=True)
    oof = np.full(len(fit), np.nan)
    for i, j in GroupKFold(3).split(fit, groups=fit.gene):
        oof[j] = public_fit(fit.iloc[i], fit.iloc[j])
    tau = max(float(np.median(np.abs(oof - fit.transfer_rmse.to_numpy()))), 1e-8)
    audit.append({'scope': scope, 'tau': tau, 'n_fit_pairs': len(fit),
                  'n_fit_clusters': fit.gene.nunique(), 'fit_clusters_hash': ids_hash(fit.gene.unique()),
                  'tau_definition': 'median absolute group-OOF biological transfer residual',
                  'tau_oof_folds': 3, 'uses_query_truth': False})
    return tau


def priors(tasks, pairs, effects, score, tau):
    array = np.full((len(tasks), effects.shape[1]), np.nan)
    rows = []
    for q, group in pairs.groupby('task_row', sort=True):
        idx = group.index.to_numpy()
        if len(np.unique(tau[idx])) != 1:
            raise RuntimeError('one biological query got inconsistent fit temperatures')
        values = score[idx]
        learned = np.exp(np.clip(-(values - values.min()) / tau[idx[0]], -20, 20))
        learned /= learned.sum()
        cells = np.expm1(group.log_source_cells.to_numpy(float)); cells /= cells.sum()
        weights = .5 * learned + .5 * cells
        vectors = np.asarray(effects[group.memory_row.to_numpy(int)], float)
        array[q] = weights @ vectors
        uncertainty = np.sqrt(np.sum(weights * np.mean((vectors - array[q]) ** 2, axis=1)))
        rows.append({'task_id': tasks.iloc[q].task_id, 'reference': 'Temperature',
            'prior_uncertainty': uncertainty, 'effective_sources': 1 / np.sum(weights ** 2),
            'history_conflict': float(group.source_conflict.mean()),
            'log_history_support': float(np.log1p(np.expm1(group.log_source_cells).sum()))})
    return array, pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'METHOD_DECISION.json').exists():
        raise FileExistsError('single registered temperature attempt already completed')
    started = time.monotonic()
    closure.tx.atomic_json(OUT / 'EXECUTION_CONFIG.json', {
        'role': 'DEV_SEEN_ONLY', 'temperature_variants': 1, 'max_runtime_seconds': 21600,
        'formula': 'max(median(abs(public group-OOF prediction - true transfer error)),1e-8)',
        'fixed_support_mix': .5, 'inner_tau_cv': 3, 'risk_features': P + PUBLIC,
        'seed': SEEDS[0], 'external_test_reads': 0,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    tasks, pairs, effects, bases, predictions, truth = closure.read_tx()
    records, biology, audit, calibration = [], [], [], []
    for outer in range(5):
        fit = pairs[pairs.fold.ne(outer)]
        query = pairs[pairs.fold.eq(outer)]
        scores, tau = np.full(len(pairs), np.nan), np.full(len(pairs), np.nan)
        scores[query.index] = public_fit(fit, query)
        tau[query.index] = training_tau(fit, f'outer{outer}/query', audit)
        for inner, (i, j) in enumerate(GroupKFold(4).split(fit, groups=fit.gene)):
            inner_fit, inner_query = fit.iloc[i], fit.iloc[j]
            scores[inner_query.index] = public_fit(inner_fit, inner_query)
            tau[inner_query.index] = training_tau(inner_fit, f'outer{outer}/risktrain_inner{inner}', audit)
        if not np.isfinite(scores).all() or not np.isfinite(tau).all():
            raise RuntimeError('temperature nested public features incomplete')
        array, summary = priors(tasks, pairs, effects, scores, tau)
        frames = {name: closure.reference_frame(base, predictions[name], array, summary, 'Temperature')
                  for name, base in bases.items()}
        for line, source, target in [('GAT_to_Exphormer', 'TxPert_GAT', 'TxPert_Exphormer'),
                                     ('Exphormer_to_GAT', 'TxPert_Exphormer', 'TxPert_GAT')]:
            train = frames[source][frames[source].fold.ne(outer)].reset_index(drop=True)
            test = frames[target][frames[target].fold.eq(outer)].reset_index(drop=True)
            labels, _ = rank_labels(train, f'temperature/{line}/outer{outer}')
            model = fit_risk(train, labels, P + PUBLIC, 'hgb')
            closure.add_predictions(records, test, model.predict(test), line, 'Temperature_hgb')
        for row in np.flatnonzero(tasks.fold.eq(outer).to_numpy()):
            biology.append({'task_id': tasks.iloc[row].task_id, 'target': tasks.iloc[row].target,
                'gene': tasks.iloc[row].gene, 'fold': outer,
                'effect_rmse': float(closure.tx.rmse_rows(array[row:row+1], truth[row:row+1])[0])})
        for row in query.itertuples():
            i = row.Index
            calibration.append({'task_id': row.task_id, 'gene': row.gene, 'fold': outer,
                'predicted_transfer_error': scores[i], 'true_transfer_error': row.transfer_rmse,
                'training_tau': tau[i], 'memory_row': row.memory_row})
        closure.tx.atomic_csv(OUT / 'TEMPERATURE_AUDIT.csv', pd.DataFrame(audit))
        print(json.dumps({'phase': 'temperature_candidate', 'outer_fold': outer,
                          'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)
        if time.monotonic() - started > 21600:
            raise TimeoutError('temperature six-hour limit reached; no second attempt')
    predictions = pd.concat(records, ignore_index=True)
    strata, macro = summarize(predictions)
    closure.tx.atomic_csv(OUT / 'TASK_PREDICTIONS.csv.gz', predictions)
    closure.tx.atomic_csv(OUT / 'STRATA.csv', strata); closure.tx.atomic_csv(OUT / 'MACRO.csv', macro)
    closure.tx.atomic_csv(OUT / 'BIOLOGY.csv', pd.DataFrame(biology))
    closure.tx.atomic_csv(OUT / 'TRANSFER_ERROR_DIAGNOSTIC.csv.gz', pd.DataFrame(calibration))
    baseline = pd.read_csv(closure.OUT / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    baseline = baseline[baseline.method.eq('Learned_hgb') & baseline.seed.eq(SEEDS[0])]
    baseline = baseline[baseline.line.isin(['GAT_to_Exphormer', 'Exphormer_to_GAT'])]
    _, baseline_macro = summarize(baseline)
    gate_rows = []
    for key, part in predictions.groupby(['line', 'target', 'fold']):
        original = baseline[(baseline.line.eq(key[0])) & (baseline.target.eq(key[1]))
                            & (baseline.fold.eq(key[2]))].set_index('task_id').loc[part.task_id].reset_index()
        new, old = metrics(part, part.risk), metrics(original, original.risk)
        gate_rows.append(dict(zip(['line', 'target', 'fold'], key)) | {
            'delta_utility20': new['utility20'] - old['utility20'], 'valid': np.isfinite(new['utility20'])})
    gate = pd.DataFrame(gate_rows)
    delta = float(macro.utility20.mean() - baseline_macro.utility20.mean())
    aurc_ratio = float(macro.aurc.mean() / baseline_macro.aurc.mean() - 1)
    old_biology = pd.read_csv(closure.OUT / 'PUBLIC_BIOLOGY_OOF.csv.gz')
    old_biology = old_biology[old_biology.reference.eq('Learned')]
    old_error = float(old_biology.groupby('target').effect_rmse.mean().mean())
    new_error = float(pd.DataFrame(biology).groupby('target').effect_rmse.mean().mean())
    fraction = float(gate[gate.valid].delta_utility20.ge(0).mean())
    passed = delta >= .005 and fraction >= .6 and aurc_ratio <= .05 and new_error <= old_error
    decision = {'candidate': 'OOF_residual_temperature_fixed_50_50', 'passed_development_gate': passed,
        'delta_utility20_macro': delta, 'fraction_nonnegative_valid_strata': fraction,
        'valid_strata': int(gate.valid.sum()), 'planned_strata': len(gate),
        'aurc_relative_change': aurc_ratio, 'new_biology_rmse': new_error, 'old_biology_rmse': old_error,
        'elapsed_seconds': time.monotonic() - started, 'external_selection_used': False,
        'action': 'retain as development candidate only' if passed else 'STOP; keep original Public method',
        'original_confirmation_certifies_candidate': False, 'temperature_attempts': 1}
    closure.tx.atomic_csv(OUT / 'GATE_STRATA.csv', gate)
    closure.tx.atomic_json(OUT / 'METHOD_DECISION.json', decision)
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()

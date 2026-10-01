#!/usr/bin/env python3
"""Fixed public-bank coverage and matched-support content interventions.

Growth uses the non-learned support-weighted reference so excluded memory
effects cannot enter a retrieval learner. Source errors remain at full budget.
Content controls exchange real effects, preserve study/context/support strata,
and reconstruct priors, dispersion and discrepancy before refitting risk.
All analysis uses previously opened DEV/SEEN assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, fit_risk, ids_hash, rank_labels, summarize
from tools.scripts import run_safeconf_research_closure as closure

OUT = closure.OUT / 'public_mechanisms'


def manual_frames(bases, predictions, pairs, effects):
    n = len(next(iter(bases.values())))
    priors = np.full((n, effects.shape[1]), np.nan)
    summaries, cache = [], {}
    for q, group in pairs.groupby('task_row', sort=True):
        memory_rows = group.memory_row.to_numpy(int)
        support = np.expm1(group.log_source_cells.to_numpy(float))
        weights = support / support.sum()
        key = (tuple(memory_rows), tuple(support))
        if key not in cache:
            vectors = np.asarray(effects[memory_rows], float)
            mean = weights @ vectors
            dispersion = np.sum(weights * np.mean((vectors - mean) ** 2, axis=1))
            conflict = np.sqrt(np.mean((vectors - vectors.mean(axis=0)) ** 2, axis=1)).mean()
            cache[key] = mean, dispersion, conflict
        priors[q], dispersion, conflict = cache[key]
        summaries.append({'task_id': group.task_id.iloc[0], 'reference': 'Manual',
            'prior_uncertainty': np.sqrt(dispersion), 'effective_sources': 1 / np.sum(weights ** 2),
            'history_conflict': conflict, 'log_history_support': np.log1p(support.sum())})
    summary = pd.DataFrame(summaries, columns=['task_id', 'reference', 'prior_uncertainty',
        'effective_sources', 'history_conflict', 'log_history_support'])
    frames = {}
    for upstream, base in bases.items():
        frame = closure.reference_frame(base, predictions[upstream], priors, summary, 'Manual')
        frame['history_available'] = np.isfinite(priors).all(axis=1)
        missing = ~frame.history_available
        frame.loc[missing, ['log_history_support', 'effective_sources']] = 0
        frame['weighted_history_distance'] = np.sqrt(
            frame.prediction_prior_rmse ** 2 + frame.prior_uncertainty ** 2)
        frames[upstream] = frame
    return frames


def external_inputs():
    root = Path('/home/yyf/data/safeconf_dual_memory_20260929/public_mcfaline_trainval')
    memory = pd.read_parquet(root / 'public_memory.parquet')
    effects = np.load(root / 'effect_vectors.npy', mmap_mode='r')
    controls = np.load(root / 'control_vectors.npy', mmap_mode='r')
    base = pd.read_parquet(closure.RUNTIME / 'external_Manual.parquet')
    aggregate = closure.MC / 'test_pretruth_20260930/aggregated_predictions'
    prediction = np.load(aggregate / 'TEST_CALIBRATED_EFFECTS.npy')
    pairs = closure.mc.build_pairs(base, np.load(aggregate / 'TEST_CONTROL_STATES.npy'),
                                   memory, effects, controls, None)
    return memory, effects, pairs, base, prediction


def evaluate(frames, external, label, **metadata):
    records, coverage = [], []
    for line, source, target in [('GAT_to_Exphormer', 'TxPert_GAT', 'TxPert_Exphormer'),
                                  ('Exphormer_to_GAT', 'TxPert_Exphormer', 'TxPert_GAT')]:
        for fold in range(5):
            fit = frames[source][frames[source].fold.ne(fold)].reset_index(drop=True)
            query = frames[target][frames[target].fold.eq(fold)].reset_index(drop=True)
            if set(fit.gene) & set(query.gene):
                raise RuntimeError('public mechanism outer clusters overlap')
            labels, _ = rank_labels(fit, f'{label}/{line}/{fold}')
            model = fit_risk(fit, labels, P + PUBLIC, 'hgb')
            for method, score in [('ManualHGB', model.predict(query)),
                                   ('ManualHistoryDistance', query.weighted_history_distance)]:
                closure.add_predictions(records, query, score, line, method, **metadata)
    pooled = pd.concat(list(frames.values()), ignore_index=True)
    labels, _ = rank_labels(pooled, f'{label}/TxPert_to_McFaline')
    model = fit_risk(pooled, labels, P + PUBLIC, 'hgb')
    for method, score in [('ManualHGB', model.predict(external)),
                           ('ManualHistoryDistance', external.weighted_history_distance)]:
        closure.add_predictions(records, external, score, 'TxPert_to_McFaline', method, **metadata)
    for domain, frame in [('TxPert', frames['TxPert_GAT']), ('McFaline', external)]:
        for context, part in frame.groupby('target'):
            coverage.append({'domain': domain, 'context': context, **metadata,
                'n_planned_tasks': len(part), 'n_tasks_with_history': int(part.history_available.sum()),
                'eligible_task_coverage': part.history_available.mean()})
    return pd.concat(records, ignore_index=True), coverage


def bank_subset(memory, budget, order):
    targets = sorted(memory.perturbation_target.astype(str).unique(), key=lambda gene: hashlib.sha256(
        f'Public-memory-coverage-v1|{order}|{gene}'.encode()).hexdigest())
    selected = set(targets[:max(1, int(np.ceil(budget * len(targets))))])
    rows = memory[memory.perturbation_target.astype(str).isin(selected)].effect_vector_row.to_numpy(int)
    return rows, {'n_bank_clusters': len(selected), 'n_bank_experiments': len(rows),
                  'bank_row_hash': ids_hash(rows), 'bank_target_hash': ids_hash(selected)}


def permutation(memory, seed, domain):
    m = memory.copy()
    # TxPert's condition contains the gene name; retain genetic-single-gene type.
    m['condition_group'] = m.condition.astype(str) if domain == 'McFaline' else m.perturbation_type.astype(str)
    grouping = ['study_id', 'effect_contract_id', 'gene_space_id', 'context', 'condition_group']
    donor = np.arange(len(m))
    rng = np.random.default_rng(seed)
    audits = []
    for key, group in m.groupby(grouping, sort=True):
        bins = pd.qcut(np.log1p(group.n_cells), 4, labels=False, duplicates='drop')
        bins = bins.fillna(0).astype(int)
        for support_bin in sorted(bins.unique()):
            ix = group.index[bins.eq(support_bin)].to_numpy()
            shuffled = ix[rng.permutation(len(ix))]
            if len(ix) > 1:
                donor[shuffled] = np.roll(shuffled, 1)
            audits.append({'domain': domain, 'seed': seed, 'group': json.dumps(key),
                'support_bin': int(support_bin), 'n_records': len(ix),
                'n_moved': int(np.sum(donor[ix] != ix))})
    # Donors preserve context/condition; all existing query exclusions still hold.
    for col in grouping:
        if not np.array_equal(m[col].to_numpy(), m.iloc[donor][col].to_numpy()):
            raise RuntimeError('content permutation crossed an alignment or context contract')
    return donor, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['growth', 'content', 'amplitude', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    closure.tx.atomic_json(OUT / f'EXECUTION_CONFIG_{args.phase}.json', {
        'role': 'SEEN_POST_CONFIRMATION', 'orders': list(range(5)), 'budgets': [.1, .25, .5, .75, 1.],
        'source_error_budget': 'fixed 100%; no source labels added with public coverage',
        'growth_reference': 'support weighted; no full-bank learned scorer reused',
        'growth_scope': 'offline bank snapshots; not prospective or monotonic improvement guarantee',
        'content_null': 'quartile-matched support; preserve study/axis/context/condition; rebuild all content features',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    tasks, pairs, effects, bases, predictions, truth = closure.read_tx()
    tx_memory, _, _, _ = closure.PublicMemoryStore(closure.tx.STORE_ROOT).load()
    mc_memory, mc_effects, mc_pairs, mc_base, mc_prediction = external_inputs()
    if args.phase in ('growth', 'all'):
        records, coverage, banks = [], [], []
        for order in range(5):
            for budget in (.1, .25, .5, .75, 1.):
                started = time.monotonic()
                rows, audit = bank_subset(tx_memory, budget, order)
                mc_rows, mc_audit = bank_subset(mc_memory, budget, order)
                restricted = pairs[pairs.memory_row.isin(rows)]
                restricted_mc = mc_pairs[mc_pairs.memory_row.isin(mc_rows)]
                frames = manual_frames(bases, predictions, restricted, effects)
                external = manual_frames({'DecoderOnly': mc_base}, {'DecoderOnly': mc_prediction},
                                         restricted_mc, mc_effects)['DecoderOnly']
                result, cov = evaluate(frames, external, 'public-growth', budget=budget, order=order)
                records.append(result); coverage.extend(cov)
                banks.extend([{'domain': 'TxPert', 'budget': budget, 'order': order, **audit},
                              {'domain': 'McFaline', 'budget': budget, 'order': order, **mc_audit}])
                whole = pd.concat(records, ignore_index=True)
                strata, macro = summarize(whole)
                closure.tx.atomic_csv(OUT / 'GROWTH_MACRO.csv', macro)
                closure.tx.atomic_csv(OUT / 'GROWTH_STRATA.csv', strata)
                closure.tx.atomic_csv(OUT / 'GROWTH_COVERAGE.csv', pd.DataFrame(coverage))
                closure.tx.atomic_csv(OUT / 'GROWTH_BANK_AUDIT.csv', pd.DataFrame(banks))
                print(json.dumps({'phase': 'public_growth', 'order': order, 'budget': budget,
                                  'seconds': round(time.monotonic() - started, 2)}), flush=True)
        closure.tx.atomic_csv(OUT / 'GROWTH_TASK_PREDICTIONS.csv.gz', whole)
    if args.phase in ('content', 'all'):
        records, audits = [], []
        for order in range(5):
            seed = SEEDS[0] + order
            donor, audit = permutation(tx_memory, seed, 'TxPert')
            mc_donor, mc_audit = permutation(mc_memory, seed, 'McFaline')
            frames = manual_frames(bases, predictions, pairs, effects[donor])
            external = manual_frames({'DecoderOnly': mc_base}, {'DecoderOnly': mc_prediction},
                                     mc_pairs, mc_effects[mc_donor])['DecoderOnly']
            result, _ = evaluate(frames, external, 'content-null', order=order)
            records.append(result); audits.extend(audit + mc_audit)
            whole = pd.concat(records, ignore_index=True)
            strata, macro = summarize(whole)
            closure.tx.atomic_csv(OUT / 'CONTENT_SHUFFLE_MACRO.csv', macro)
            closure.tx.atomic_csv(OUT / 'CONTENT_SHUFFLE_STRATA.csv', strata)
            closure.tx.atomic_csv(OUT / 'CONTENT_SHUFFLE_AUDIT.csv', pd.DataFrame(audits))
            print(json.dumps({'phase': 'public_content_shuffle', 'order': order}), flush=True)
        closure.tx.atomic_csv(OUT / 'CONTENT_SHUFFLE_TASK_PREDICTIONS.csv.gz', whole)
    if args.phase in ('amplitude', 'all'):
        records = []
        for ref in ('Manual', 'Learned'):
            fit = pd.read_parquet(closure.RUNTIME / f'source_{ref}.parquet')
            query = pd.read_parquet(closure.RUNTIME / f'external_{ref}.parquet')
            labels, _ = rank_labels(fit, f'amplitude-diagnostic/{ref}')
            variants = {'FullShared': P + PUBLIC,
                        'NoAbsolutePredictionAmplitude': ['prediction_sparsity'] + PUBLIC,
                        'PublicDiscrepancyOnly': [c for c in PUBLIC if c != 'prior_magnitude']}
            for name, columns in variants.items():
                model = fit_risk(fit, labels, columns, 'hgb')
                closure.add_predictions(records, query, model.predict(query), 'TxPert_to_McFaline',
                                        f'{ref}_{name}')
        whole = pd.concat(records, ignore_index=True)
        strata, macro = summarize(whole)
        closure.tx.atomic_csv(OUT / 'AMPLITUDE_ABLATION_MACRO.csv', macro)
        closure.tx.atomic_csv(OUT / 'AMPLITUDE_ABLATION_STRATA.csv', strata)
        closure.tx.atomic_csv(OUT / 'AMPLITUDE_ABLATION_TASK_PREDICTIONS.csv.gz', whole)
        print(json.dumps({'phase': 'amplitude_diagnostic', 'model_selection_allowed': False}), flush=True)
    closure.tx.atomic_json(OUT / f'RUN_STATUS_{args.phase}.json', {'status': 'COMPLETE', 'phase': args.phase,
        'role': 'SEEN_POST_CONFIRMATION', 'new_large_upstream_training': 0})


if __name__ == '__main__':
    main()

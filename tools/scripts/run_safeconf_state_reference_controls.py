#!/usr/bin/env python3
"""Strong reference controls for the matched-support shuffle interpretation.

These controls use other perturbations in the same biological state. TxPert
reference statistics are recomputed inside outer/inner training clusters;
McFaline references use its already registered train/validation Public bank.
They are diagnostic comparators, never selected as a confirmed new method.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, fit_risk, ids_hash, rank_labels, summarize
from tools.scripts import run_safeconf_research_closure as closure
from tools.scripts import run_safeconf_public_mechanisms as public

OUT = closure.OUT / 'state_reference_controls'


def state_pairs(tasks, memory, allowed_genes, original_pairs, domain, matched_k, query_indices=None):
    rows, audit = [], []
    by_context = memory.groupby('context').indices
    original = {q: g for q, g in original_pairs.groupby('task_row')}
    genes = memory.perturbation_target.astype(str).to_numpy()
    conditions = memory.condition.astype(str).to_numpy()
    cell_log = np.log1p(memory.n_cells.to_numpy(float))
    vector_rows = memory.effect_vector_row.to_numpy(int)
    experiment_ids = memory.experiment_id.to_numpy(str)
    tie_hash = np.asarray([hashlib.sha256(f'state-reference-v1|{s}'.encode()).hexdigest()
                           for s in experiment_ids])
    allowed_mask = np.ones(len(memory), bool) if allowed_genes is None else np.isin(genes, sorted(allowed_genes))
    tuples = list(tasks.itertuples(index=False))
    for q in (range(len(tasks)) if query_indices is None else query_indices):
        task = tuples[q]
        context = str(task.target) if domain == 'TxPert' else str(task.context)
        gene = str(task.gene)
        indices = np.asarray(by_context.get(context, []), dtype=int)
        indices = indices[allowed_mask[indices] & (genes[indices] != gene)]
        if domain == 'McFaline':
            indices = indices[conditions[indices] == str(task.treatment)]
        available = len(indices)
        if not available:
            raise RuntimeError(f'no training-only same-state history for {task.task_id}')
        if matched_k:
            targets = np.log1p(np.expm1(original[q].log_source_cells.to_numpy(float)))
            chosen = []
            remaining = indices.copy()
            for cells in targets:
                index = remaining[np.lexsort((tie_hash[remaining], np.abs(cell_log[remaining] - cells)))[0]]
                chosen.append(index); remaining = remaining[remaining != index]
            indices = np.asarray(chosen)
        for i in indices:
            rows.append({'task_row': q, 'task_id': task.task_id, 'memory_row': int(vector_rows[i]),
                         'log_source_cells': float(cell_log[i])})
        audit.append({'task_id': task.task_id, 'domain': domain, 'matched_k': matched_k,
                      'n_available_training_state_history': available, 'n_selected': len(indices),
                      'query_gene_excluded': True, 'selected_history_hash': ids_hash(experiment_ids[indices])})
    return pd.DataFrame(rows), audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, original_pairs, effects, bases, predictions, truth = closure.read_tx()
    memory, _, _, _ = closure.PublicMemoryStore(closure.tx.STORE_ROOT).load()
    mc_memory, mc_effects, mc_original, mc_base, mc_prediction = public.external_inputs()
    records, audit = [], []
    for matched in (False, True):
        ref = 'MatchedKStateMean' if matched else 'StateMean'
        for outer in range(5):
            train = tasks[tasks.fold.ne(outer)].copy()
            test = tasks[tasks.fold.eq(outer)].copy()
            allowed = set(train.gene.astype(str))
            query_pairs, rows = state_pairs(tasks, memory, allowed, original_pairs, 'TxPert', matched, test.index)
            # Retain only outer-query pairs from the outer-trained state reference.
            assembled = [query_pairs[query_pairs.task_row.isin(test.index)]]
            audit.extend([{'scope': f'{ref}/outer{outer}/query', **r} for r in rows if r['task_id'] in set(test.task_id)])
            for inner, (i, j) in enumerate(GroupKFold(4).split(train, groups=train.gene)):
                genes = set(train.iloc[i].gene.astype(str))
                inner_pairs, rows = state_pairs(tasks, memory, genes, original_pairs, 'TxPert', matched, train.iloc[j].index)
                assembled.append(inner_pairs[inner_pairs.task_row.isin(train.iloc[j].index)])
                query_ids = set(train.iloc[j].task_id)
                audit.extend([{'scope': f'{ref}/outer{outer}/inner{inner}', **r} for r in rows if r['task_id'] in query_ids])
            pairs = pd.concat(assembled, ignore_index=True)
            if pairs.task_id.nunique() != len(tasks):
                raise RuntimeError('nested same-state reference misses a query')
            frames = public.manual_frames(bases, predictions, pairs, effects)
            for line, source, target in [('GAT_to_Exphormer', 'TxPert_GAT', 'TxPert_Exphormer'),
                                         ('Exphormer_to_GAT', 'TxPert_Exphormer', 'TxPert_GAT')]:
                fit = frames[source][frames[source].fold.ne(outer)].reset_index(drop=True)
                query = frames[target][frames[target].fold.eq(outer)].reset_index(drop=True)
                labels, _ = rank_labels(fit, f'{ref}/{line}/outer{outer}')
                for seed in SEEDS:
                    model = fit_risk(fit, labels, P + PUBLIC, 'hgb', seed)
                    for name, score in [('HGB', model.predict(query)), ('DirectRMSE', query.prediction_prior_rmse),
                                         ('WeightedDistance', query.weighted_history_distance)]:
                        closure.add_predictions(records, query, score, line, f'{ref}_{name}', seed)
            print(f'{ref}: outer fold {outer} complete', flush=True)
        # Fit independent-study risk supervision using source reference OOF.
        assembled = []
        for fold in range(5):
            allowed = set(tasks[tasks.fold.ne(fold)].gene.astype(str))
            pairs, _ = state_pairs(tasks, memory, allowed, original_pairs, 'TxPert', matched,
                                   tasks[tasks.fold.eq(fold)].index)
            assembled.append(pairs[pairs.task_row.isin(tasks[tasks.fold.eq(fold)].index)])
        source_frames = public.manual_frames(bases, predictions, pd.concat(assembled, ignore_index=True), effects)
        fit = pd.concat(list(source_frames.values()), ignore_index=True)
        labels, _ = rank_labels(fit, f'{ref}/source-OOF')
        pairs, rows = state_pairs(mc_base, mc_memory, None, mc_original, 'McFaline', matched)
        audit.extend([{'scope': f'{ref}/external', **r} for r in rows])
        query = public.manual_frames({'DecoderOnly': mc_base}, {'DecoderOnly': mc_prediction}, pairs,
                                     mc_effects)['DecoderOnly']
        for seed in SEEDS:
            model = fit_risk(fit, labels, P + PUBLIC, 'hgb', seed)
            for name, score in [('HGB', model.predict(query)), ('DirectRMSE', query.prediction_prior_rmse),
                                 ('WeightedDistance', query.weighted_history_distance)]:
                closure.add_predictions(records, query, score, 'TxPert_to_McFaline', f'{ref}_{name}', seed)
        result = pd.concat(records, ignore_index=True)
        strata, macro = summarize(result)
        closure.tx.atomic_csv(OUT / 'TASK_PREDICTIONS.csv.gz', result)
        closure.tx.atomic_csv(OUT / 'MACRO.csv', macro); closure.tx.atomic_csv(OUT / 'STRATA.csv', strata)
        closure.tx.atomic_csv(OUT / 'REFERENCE_AUDIT.csv.gz', pd.DataFrame(audit))
    closure.tx.atomic_json(OUT / 'RUN_STATUS.json', {'status': 'COMPLETE', 'role': 'SEEN_DIAGNOSTIC_COMPARATORS',
        'no_target_risk_errors': True, 'no_new_confirmed_winner_selected': True,
        'same_state_reference': 'other perturbations; query gene excluded; nested training-only state statistics',
        'matched_k': 'same history record count; nearest source-cell support; no target truth used for matching',
        'full_state_mean': 'uses additional historical experiments; extra information cost disclosed'})


if __name__ == '__main__':
    main()

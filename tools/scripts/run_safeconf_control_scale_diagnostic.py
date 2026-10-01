#!/usr/bin/env python3
"""One fixed dimensionless-control-scale diagnosis, not a new final method.

Keep native error targets, public retrieval, CDFs, learner parameters and task
cohorts unchanged. Amplitude inputs are divided by a mean-control RMS derived
from allowed training-side controls. Outer test genes never fit source scales.
This cannot certify or replace the frozen September/October method.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, fit_risk, rank_labels, summarize, bootstrap_u20, ids_hash,
    paired_prediction_wide,
)
from tools.scripts import run_safeconf_research_closure as closure
from tools.scripts import run_safeconf_common_axis_closure as common

RUNTIME = common.COMMON / 'risk_cache'
BASE = common.DOC / 'results'
OUT = common.DOC / 'control_scale_diagnostic'
AMPLITUDE = [c for c in P if c != 'prediction_sparsity'] + [
    'prior_magnitude', 'prediction_prior_rmse', 'prior_uncertainty', 'history_conflict']


def scale_map(memory, controls, allowed_genes=None, drug=False):
    eligible = memory if allowed_genes is None else memory[
        memory.perturbation_target.astype(str).isin(allowed_genes)]
    if eligible.empty:
        raise RuntimeError('no train-side controls for the registered source genes')
    keys = ['context', 'condition'] if drug else ['context']
    mapping, audit = {}, []
    for key, rows in eligible.groupby(keys, sort=True):
        key = tuple(key) if isinstance(key, tuple) else (key,)
        vector_rows = rows.effect_vector_row.to_numpy(int)
        values = controls[vector_rows].astype(float)
        value = float(np.sqrt(np.mean(values.mean(axis=0) ** 2)))
        if value <= 1e-12 or not np.isfinite(value):
            raise RuntimeError('nonfinite or zero train control RMS')
        mapping[key] = value
        audit.append({'context': key[0], 'condition': key[1] if drug else '__all__',
                      'scale': value, 'n_public_reference_rows': len(rows),
                      'n_reference_genes': rows.perturbation_target.nunique(),
                      'reference_row_hash': ids_hash(vector_rows), 'uses_perturbed_truth': False})
    return mapping, audit


def transform(frame, mapping, drug=False):
    out = frame.copy()
    keys = list(zip(out.target.astype(str), out.treatment.astype(str))) if drug else [
        (x,) for x in out.target.astype(str)]
    scales = np.asarray([mapping[key] for key in keys])
    out[AMPLITUDE] = out[AMPLITUDE].to_numpy(float) / scales[:, None]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'STATUS.json').exists():
        raise FileExistsError('completed scale diagnosis is immutable')
    closure.tx.atomic_json(OUT / 'REGISTERED_DIAGNOSTIC.json', {
        'role': 'DEV_SEEN_DIAGNOSTIC', 'rule': 'amplitude / RMS(mean allowed control vector)',
        'amplitude_columns': AMPLITUDE, 'learner_hyperparameters_unchanged': True,
        'native_error_target_unchanged': True, 'public_retrieval_unchanged': True,
        'select_final_method_from_this_diagnostic': False, 'one_fixed_rule_no_scale_search': True,
        'own_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    tx_memory, _, _, _ = closure.PublicMemoryStore(closure.tx.STORE_ROOT).load()
    tx_controls = np.load(common.COMMON / 'SOURCE_PUBLIC_CONTROLS.npy')
    mc_memory, _, mc_controls, _ = closure.PublicMemoryStore(common.COMMON / 'public_mcfaline_trainval').load()
    target_map, target_audit = scale_map(mc_memory, mc_controls, drug=True)
    original = pd.read_csv(BASE / 'MATRIX_TASK_PREDICTIONS.csv.gz')
    canonical_errors = original[original.method.eq('Manual_hgb') & original.seed.eq(SEEDS[0])].set_index(
        ['line', 'task_id']).true_error_rmse
    records, audits, cdf = [], [], []
    lines = [('GAT_to_Exphormer', 'TxPert_GAT', 'TxPert_Exphormer'),
             ('Exphormer_to_GAT', 'TxPert_Exphormer', 'TxPert_GAT'),
             ('TxPert_to_McFaline', None, 'DecoderOnly')]
    for line, source, target in lines:
        for fold in (range(5) if source else [-1]):
            for ref in ['Manual', 'Learned']:
                if source:
                    fit = pd.read_parquet(RUNTIME / f'nested_{fold}_{source}_{ref}.parquet')
                    fit = fit[fit.fold.ne(fold)].reset_index(drop=True)
                    query = pd.read_parquet(RUNTIME / f'nested_{fold}_{target}_{ref}.parquet')
                    query = query[query.fold.eq(fold)].reset_index(drop=True)
                    if set(fit.gene) & set(query.gene):
                        raise RuntimeError('scale diagnosis crossed held-out source clusters')
                else:
                    fit = pd.read_parquet(RUNTIME / f'source_{ref}.parquet')
                    query = pd.read_parquet(RUNTIME / f'external_{ref}.parquet')
                exact = np.asarray([canonical_errors.loc[(line, task)] for task in query.task_id])
                if not np.allclose(query.true_error_rmse, exact, rtol=2*np.finfo(np.float32).eps, atol=1e-10):
                    raise RuntimeError('normalization diagnosis changed the native query truth')
                query['true_error_rmse'] = exact
                mapping, audit = scale_map(tx_memory, tx_controls, set(fit.gene.astype(str)))
                fit_scaled = transform(fit, mapping)
                query_scaled = transform(query, mapping if source else target_map, drug=source is None)
                labels, label_audit = rank_labels(fit, f'control-scale/{line}/{fold}/{ref}')
                cdf.extend(label_audit)
                audits.extend([dict(x, line=line, outer_fold=fold, reference=ref, domain='source') for x in audit])
                if source is None:
                    audits.extend([dict(x, line=line, outer_fold=fold, reference=ref, domain='target_trainval')
                                   for x in target_audit])
                for seed in SEEDS:
                    model = fit_risk(fit_scaled, labels, P + PUBLIC, 'hgb', seed)
                    closure.add_predictions(records, query, model.predict(query_scaled), line,
                                            f'{ref}_ControlScaledHGB', seed)
            print(json.dumps({'line': line, 'outer_fold': fold, 'scale_diagnosis': 'fit_complete'}), flush=True)
    predictions = pd.concat(records, ignore_index=True)
    closure.tx.atomic_csv(OUT / 'TASK_PREDICTIONS.csv.gz', predictions)
    strata, macro = summarize(predictions)
    for name, frame in [('MACRO.csv', macro), ('STRATA.csv', strata), ('SCALE_AUDIT.csv', pd.DataFrame(audits)),
                        ('CDF_AUDIT.csv', pd.DataFrame(cdf))]:
        closure.tx.atomic_csv(OUT / name, frame)
    differences = []
    for line in sorted(predictions.line.unique()):
        added = predictions[predictions.line.eq(line) & predictions.seed.eq(SEEDS[0])]
        base = original[original.line.eq(line) & original.seed.eq(SEEDS[0])]
        wide = paired_prediction_wide(pd.concat([base, added], ignore_index=True))
        for ref in ['Manual', 'Learned']:
            a, b = f'{ref}_ControlScaledHGB', f'{ref}_hgb'
            if wide[[a, b]].isna().any().any():
                raise RuntimeError('scaled and original tasks differ')
            differences.append({'line': line, 'method_a': a, 'method_b': b,
                **bootstrap_u20(wide, wide[a].to_numpy(), wide[b].to_numpy(), 5000)})
        closure.tx.atomic_csv(OUT / 'PAIRED_BOOTSTRAP.csv', pd.DataFrame(differences))
    closure.tx.atomic_json(OUT / 'STATUS.json', {'status': 'COMPLETE', 'role': 'SEEN_DIAGNOSTIC',
        'no_new_upstream_training_or_calls': True, 'native_truth_and_scores_preserved': True,
        'one_rule_no_search': True, 'bootstrap_replicates': 5000})
    print(macro[macro.seed.eq(SEEDS[0])][['line', 'method', 'utility20', 'spearman']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

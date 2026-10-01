#!/usr/bin/env python3
"""Fixed, post-confirmation guide-vs-cell reference audit; no model fitting.

Only train/validation cells build the alternate reference. Frozen query
predictions, eligibility, history support, primary truth and metrics remain
unchanged. This diagnostic is not a selected replacement method.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual import PublicMemoryStore
from tools.safeconf_continual.research import SEEDS, bootstrap_u20, summarize
from tools.scripts import run_safeconf_common_axis_closure as common
from tools.scripts.build_safeconf_common_gene_biology import categorical, accumulate

OUT = common.DOC / 'reference_estimand_diagnostic'
RUNTIME = common.COMMON / 'reference_estimand_diagnostic'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.perf_counter()
    if (OUT / 'STATUS.json').exists():
        raise RuntimeError('completed diagnostic cannot be overwritten')
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    memory, original, controls, manifest = PublicMemoryStore(
        common.COMMON / 'public_mcfaline_trainval').load()
    bank_keys = (memory.perturbation_target.astype(str) + '::' +
                 memory.context.astype(str) + '::' + memory.condition.astype(str))
    if bank_keys.duplicated().any():
        raise RuntimeError('ambiguous biological task in public memory')
    lookup = {key: i for i, key in enumerate(bank_keys)}
    genes = json.loads((common.COMMON / 'GENE_IDS.json').read_text())
    registration = {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC',
        'fixed_change': 'within-experiment cell-weighted instead of guide-equal mean',
        'allowed_expression_roles': ['train', 'val'],
        'query_prediction': 'frozen TEST_CALIBRATED_EFFECTS.npy',
        'query_truth': 'unchanged canonical primary error',
        'history_weights': 'unchanged n_cells / sum(n_cells)',
        'eligibility': 'unchanged same-gene other-state historical records',
        'risk_or_retrieval_fits': 0, 'hyperparameter_search': False,
        'memory_manifest': manifest,
        'input_h5_path': str(common.H5),
        'input_h5_size': common.H5.stat().st_size,
        'input_h5_mtime_ns': common.H5.stat().st_mtime_ns,
        'split_sha256': digest(common.SPLIT),
        'bootstrap_replicates': 5000, 'bootstrap_seed': SEEDS[0],
    }
    (OUT / 'REGISTERED_DIAGNOSTIC.json').write_text(json.dumps(registration, indent=2) + '\n')
    split = pd.read_csv(common.SPLIT, header=None, names=['cell_id', 'role'])
    role_map = dict(zip(split.cell_id.astype(str), split.role.astype(str)))
    sums = np.zeros_like(original, dtype=np.float64)
    counts = np.zeros(len(memory), dtype=np.int64)
    with h5py.File(common.H5, 'r') as handle:
        obs = handle['obs']
        ids = categorical(obs, '_index')
        roles = np.asarray([role_map.get(x, 'missing') for x in ids])
        if np.any(roles == 'missing'):
            raise RuntimeError('official split metadata failed to align')
        keys = (pd.Series(categorical(obs, 'perturbation')) + '::' +
                pd.Series(categorical(obs, 'cell_type')) + '::' +
                pd.Series(categorical(obs, 'treatment'))).to_numpy(str)
        treated = np.isin(roles, ['train', 'val']) & (categorical(obs, 'control') != '1')
        codes = np.full(len(ids), -1, dtype=np.int64)
        codes[treated] = [lookup.get(k, -1) for k in keys[treated]]
        if np.any(codes[treated] < 0):
            raise RuntimeError('allowed public task absent from memory manifest')
        native_genes = categorical(handle['var'], 'gene_name')
        gene_lookup = {x: i for i, x in enumerate(native_genes)}
        columns = np.asarray([gene_lookup[x] for x in genes])
        x = handle['X']; indptr = x['indptr']
        for start in range(0, len(ids), 4096):
            end = min(start + 4096, len(ids))
            if np.all(codes[start:end] < 0):
                continue
            lo, hi = int(indptr[start]), int(indptr[end])
            block = sparse.csr_matrix((np.asarray(x['data'][lo:hi]),
                np.asarray(x['indices'][lo:hi]),
                np.asarray(indptr[start:end + 1], dtype=np.int64) - lo),
                shape=(end - start, len(native_genes)))[:, columns].toarray()
            accumulate(sums, counts, codes[start:end], block)
            if start % (4096 * 20) == 0:
                print(f'Public cell-mean aggregation {end}/{len(ids)}', flush=True)
    if not np.array_equal(counts, memory.n_cells.to_numpy(int)):
        raise RuntimeError('cell support differs from frozen public memory')
    alternate = sums / counts[:, None] - np.asarray(controls, float)
    np.save(RUNTIME / 'CELL_WEIGHTED_PUBLIC_EFFECTS.npy', alternate)
    base = pd.read_parquet(common.COMMON / 'risk_cache/external_Manual.parquet')
    prediction = np.load(common.COMMON / 'TEST_CALIBRATED_EFFECTS.npy')
    pairs = common.closure.mc.build_pairs(base,
        np.load(common.COMMON / 'TEST_CONTROLS.npy'), memory, original, controls, None)
    risks = {k: np.full(len(base), np.nan) for k in ['FrozenGuideEqualReference', 'CellWeightedReference']}
    for q, group in pairs.groupby('task_row', sort=True):
        rows = group.memory_row.to_numpy(int)
        support = np.expm1(group.log_source_cells.to_numpy(float))
        weights = support / support.sum()
        for label, effects in [('FrozenGuideEqualReference', original), ('CellWeightedReference', alternate)]:
            risks[label][q] = np.sqrt(weights @ np.mean(
                (np.asarray(effects[rows], float) - prediction[q]) ** 2, axis=1))
    if any(not np.isfinite(x).all() for x in risks.values()):
        raise RuntimeError('diagnostic changed eligible query cohort')
    primary = pd.read_csv(common.DOC / 'results/MATRIX_TASK_PREDICTIONS.csv.gz')
    old = primary[(primary.line == 'TxPert_to_McFaline') &
        (primary.seed == SEEDS[0]) & (primary.method == 'Manual_WeightedHistoryDistance')]
    old = old.set_index('task_id').loc[base.task_id]
    if not np.allclose(risks['FrozenGuideEqualReference'], old.risk, rtol=1e-6, atol=1e-8):
        raise RuntimeError('frozen guide-equal distance did not reproduce')
    base['true_error_rmse'] = old.true_error_rmse.to_numpy(float)
    support_scores = -base.log_history_support.to_numpy(float)
    frames = []
    for method, score in risks.items():
        part = base[['task_id', 'target', 'gene', 'true_error_rmse']].copy()
        part['method'] = method; part['risk'] = score
        part['line'] = 'TxPert_to_McFaline'; part['seed'] = SEEDS[0]
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(RUNTIME / 'TASK_PREDICTIONS.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    strata, macro = summarize(result)
    strata.to_csv(OUT / 'STRATA.csv', index=False)
    macro.to_csv(OUT / 'MACRO.csv', index=False)
    changes = []
    for b, score in [('FrozenGuideEqualReference', risks['FrozenGuideEqualReference']),
                     ('NegativeHistorySupport', support_scores)]:
        changes.append({'method_a': 'CellWeightedReference', 'method_b': b,
            **bootstrap_u20(base, risks['CellWeightedReference'], score)})
    pd.DataFrame(changes).to_csv(OUT / 'PAIRED_COMPARISONS.csv', index=False)
    delta = np.sqrt(np.mean((alternate - original) ** 2, axis=1))
    bank = memory[['experiment_id', 'context', 'condition', 'n_cells', 'n_guides']].copy()
    bank['guide_vs_cell_effect_rmse'] = delta
    bank.groupby(['context', 'condition']).agg(n_experiments=('experiment_id', 'size'),
        mean_reference_change=('guide_vs_cell_effect_rmse', 'mean'),
        max_reference_change=('guide_vs_cell_effect_rmse', 'max')).reset_index().to_csv(
            OUT / 'REFERENCE_CHANGE_BY_STATE.csv', index=False)
    bank.to_csv(RUNTIME / 'PER_EXPERIMENT_REFERENCE_CHANGE.csv.gz', index=False)
    status = {'status': 'COMPLETE', 'n_queries': len(base),
        'n_public_experiments': len(memory), 'n_public_cells': int(counts.sum()),
        'test_cells_aggregated': 0, 'query_truth_unchanged': True,
        'frozen_guide_distance_reproduced': True,
        'mean_guide_vs_cell_effect_rmse': float(delta.mean()),
        'new_model_fits': 0, 'new_upstream_calls': 0,
        'elapsed_seconds': time.perf_counter() - started,
        'alternate_effect_sha256': digest(RUNTIME / 'CELL_WEIGHTED_PUBLIC_EFFECTS.npy'),
        'role': 'SEEN_POST_CONFIRMATION_DIAGNOSTIC'}
    (OUT / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()

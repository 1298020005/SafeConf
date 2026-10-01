#!/usr/bin/env python3
"""Build a feature-faithful control-reference PertEMA adaptation from real NTCs.

Use only non-targeting controls that are in TRAIN in every official split.
Co-expression uses real control guide/technical-plate pseudobulks. PCR plate
variance is disclosed as a technical-instability proxy, never donor variance.
Only NTC rows enter feature statistics; no perturbed effect or error is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import ids_hash
from tools.scripts import run_safeconf_research_closure as closure
from tools.scripts.run_safeconf_official_pertema_feedback import factory

DATA = Path('/home/yyf/data/perturbench_mcfaline23_official')
OUT = closure.RUNTIME / 'native_control_reference'


def column(group, name):
    item = group[name]
    if isinstance(item, h5py.Group):
        categories = item['categories'][:]
        categories = np.asarray([x.decode() if isinstance(x, bytes) else x for x in categories])
        codes = item['codes'][:]
        if (codes < 0).any():
            raise RuntimeError(f'{name} has missing category codes')
        return categories[codes]
    data = item[:]
    return np.asarray([x.decode() if isinstance(x, bytes) else x for x in data]) if data.dtype.kind in 'OS' else data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'AUDIT.json').exists():
        raise FileExistsError('native control reference already built; preserve it')
    public = pd.read_parquet(closure.RUNTIME / 'external_Learned.parquet')
    bank = pd.read_parquet('/home/yyf/data/safeconf_dual_memory_20260929/public_mcfaline_trainval/public_memory.parquet')
    panel = sorted(bank.perturbation_target.astype(str).unique())
    path = DATA / 'mcfaline23_gxe_processed.h5ad'
    with h5py.File(path, 'r') as file:
        obs = file['obs']
        cell_ids = column(obs, '_index')
        control = column(obs, 'control').astype(int) == 1
        genes_observed = column(obs, 'gene_id')
        if set(genes_observed[control]) != {'control'}:
            raise RuntimeError('NTC mask contains perturbed cells')
        context, treatment = column(obs, 'cell_type'), column(obs, 'treatment')
        plate, guide = column(obs, 'PCR_plate'), column(obs, 'gRNA_id')
        allowed = control.copy()
        split_audit = []
        for name in ['small', 'medium', 'full']:
            split_path = DATA / f'splits/mcfaline23_gxe_splits/{name}_covariate_split.csv'
            split = pd.read_csv(split_path, header=None, names=['cell_id', 'role']).set_index('cell_id').role
            roles = split.reindex(cell_ids).to_numpy()
            if not split.index.is_unique or not pd.Index(cell_ids).isin(split.index).all():
                raise RuntimeError('official split does not cover unique raw cell IDs')
            # Small/medium releases retain all IDs but leave unsampled roles NA.
            # They are excluded, rather than silently treated as training cells.
            allowed &= roles == 'train'
            split_audit.append({'split': str(split_path), 'sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                                'n_ntc_train': int(np.sum(control & (roles == 'train'))),
                                'n_unassigned_roles': int(pd.isna(roles).sum()),
                                'n_ntc_unassigned_roles': int(np.sum(control & pd.isna(roles)))})
        rows = np.flatnonzero(allowed)
        if len(rows) < 100:
            raise RuntimeError('not enough unperturbed TRAIN controls')
        genes = column(file['var'], 'gene_name')
        gene_map = {str(g): i for i, g in enumerate(genes)}
        embeddable = [g for g in panel if g in gene_map]
        cols = np.asarray([gene_map[g] for g in embeddable])
        selected_obs = pd.DataFrame({'context': context[rows], 'treatment': treatment[rows],
                                     'plate': plate[rows], 'guide': guide[rows]})
        selected_obs['state'] = selected_obs.context.astype(str) + '::' + selected_obs.treatment.astype(str)
        group_keys = selected_obs[['state', 'plate', 'guide']].astype(str).agg('|'.join, axis=1)
        group_codes, groups = pd.factorize(group_keys, sort=True)
        n_groups = len(groups)
        group_meta = selected_obs.assign(group=group_codes).groupby('group', sort=True).first()
        cell_counts = np.bincount(group_codes, minlength=n_groups)
        sums = np.zeros((n_groups, len(cols)), dtype=np.float64)
        totals = np.zeros(n_groups, dtype=np.float64)
        matrix = file['layers/counts']
        pointer = matrix['indptr'][:]
        n_columns = int(matrix.attrs['shape'][1])
        # Block reads avoid hundreds of thousands of tiny compressed HDF5 reads.
        for start in range(0, len(rows), 4000):
            chosen = rows[start:start + 4000]
            first, last = int(chosen[0]), int(chosen[-1]) + 1
            begin, end = int(pointer[first]), int(pointer[last])
            block = sparse.csr_matrix((matrix['data'][begin:end], matrix['indices'][begin:end],
                                       pointer[first:last + 1] - begin), shape=(last - first, n_columns))
            controls = block[chosen - first]
            mapping = sparse.csr_matrix((np.ones(len(chosen)),
                (group_codes[start:start + len(chosen)], np.arange(len(chosen)))), shape=(n_groups, len(chosen)))
            sums += (mapping @ controls[:, cols]).toarray()
            totals += np.asarray(mapping @ np.asarray(controls.sum(axis=1)).ravel()).ravel()
            print(json.dumps({'phase': 'native_ntc_aggregation', 'control_cells_read': start + len(chosen),
                              'total_training_control_cells': len(rows)}), flush=True)
    keep = (cell_counts >= 3) & (totals > 0)
    normalized = np.log1p(sums[keep] * (1e6 / totals[keep])[:, None])
    meta = group_meta.loc[keep].reset_index(drop=True)
    states = sorted(meta.state.unique())
    baseline, dropout, plate_var = [], [], []
    for state in states:
        use = meta.state.eq(state).to_numpy()
        x = normalized[use]
        baseline.append(x.mean(axis=0)); dropout.append((x == 0).mean(axis=0))
        local_plates = meta.loc[use, 'plate'].to_numpy()
        plate_means = [x[local_plates == p].mean(axis=0) for p in sorted(set(local_plates))]
        plate_var.append(np.var(plate_means, axis=0))
    matrix = normalized.T
    matrix = (matrix - matrix.mean(axis=1, keepdims=True)) / (matrix.std(axis=1, keepdims=True) + 1e-8)
    svd = TruncatedSVD(n_components=50, random_state=0)
    embedding = svd.fit_transform(matrix).astype(np.float32)
    # Call the real official feature constructor after replacing CD4 state names.
    import importlib.util
    source = Path('/home/yyf/runtime_artifacts/official_pertema_43c09a/src/pertema/run_estimator.py')
    spec = importlib.util.spec_from_file_location('native_pertema_features_fixed', source)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    official.CONDS = states
    query = public.copy(); query['condition'] = query.context.astype(str) + '::' + query.treatment.astype(str)
    query['pred_magnitude'] = query.prediction_abs_mean
    features, _ = official.build_base_features(query, np.asarray(baseline), np.asarray(dropout),
        np.asarray(plate_var), {g: i for i, g in enumerate(embeddable)},
        {g: e for g, e in zip(embeddable, embedding)}, 50)
    names = ['native_prediction_abs_mean', 'native_control_baseline', 'native_control_dropout',
             'native_control_plate_variance_proxy'] + [f'native_state_{i}' for i in range(len(states))] + [f'native_embedding_{i}' for i in range(50)]
    if features.shape[1] != len(names):
        raise RuntimeError('official feature constructor width changed')
    for i, name in enumerate(names):
        public[name] = features[:, i]
    # Filled inside each feedback/inner training partition, never globally.
    public['native_training_similarity'] = np.nan
    public.to_parquet(OUT / 'TASK_FEATURES.parquet', index=False)
    np.savez(OUT / 'CONTROL_REFERENCE.npz', genes=np.asarray(embeddable), embedding=embedding,
             states=np.asarray(states), baseline=baseline, dropout=dropout, plate_variance=plate_var)
    audit = {'status': 'COMPLETE', 'raw_h5ad': str(path), 'raw_size_bytes': path.stat().st_size,
        'raw_shape': [len(cell_ids), len(genes)], 'n_available_ntc_cells': int(control.sum()),
        'n_train_control_cells_used': len(rows), 'train_control_cell_ids_hash': ids_hash(cell_ids[rows]),
        'official_splits': split_audit, 'n_guide_plate_pseudobulks': int(keep.sum()),
        'control_pseudobulk_min_cells': 3, 'n_panel_genes': len(panel), 'n_embedded_panel_genes': len(embeddable),
        'query_gene_mapping_coverage': float(public.gene.isin(embeddable).mean()),
        'feature_names': names + ['native_training_similarity'], 'biological_states': states,
        'normalization': 'raw-count control pseudobulk log1p CPM; train NTCs only',
        'embedding': '50-d control-only gene co-expression SVD; no perturbed cell or error label',
        'svd_explained_variance': float(svd.explained_variance_ratio_.sum()),
        'donor_information_available': False, 'donor_variance_replacement': 'explicit PCR technical-plate proxy',
        'perturbed_h5ad_expression_rows_used': 0, 'test_effects_used_for_feature_fitting': 0,
        'io_implementation': 'contiguous CSR blocks can contain interleaved cells; only TRAIN NTC rows enter aggregates',
        'target_error_labels_used_for_feature_fitting': 0, 'role': 'SEEN_CONTROL_FEATURE_ADAPTATION',
        'native_feature_constructor_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    closure.tx.atomic_json(OUT / 'AUDIT.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()

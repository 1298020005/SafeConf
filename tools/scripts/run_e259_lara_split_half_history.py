#!/usr/bin/env python3
"""E259 Lara train-source-only split-half history on E112's fixed gene panel.

Read the public raw single-cell data to estimate within-source sampling
variation. The full E112 effect cache is already public and is used only to
verify normalization and the fixed 512-gene axis. For each E99 fold, features
index only train source pairs. No E112 error labels are read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


RAW = Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/LaraAstiasoHuntly2023_exvivo.h5ad')
CACHE = Path('/home/yyf/data/safeconf_e112_external/Lara_exvivo_CONTROL_ONLY_512.npz')
MANIFEST = Path('/home/yyf/proj/docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')
TASKS = Path(__file__).resolve().parents[2] / 'docs/实验结果/E253_lara_history_transfer_20260924/E253_ALL_120_TASKS.csv'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_log1p(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, total, out=np.zeros_like(total, dtype=np.float32),
                      where=total > 0)
    matrix = (sp.diags(scale) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def split_mean_vectors() -> tuple[dict, dict]:
    with np.load(CACHE, allow_pickle=False) as cached:
        contexts = cached['contexts'].astype(str).tolist()
        perts = cached['perturbations'].astype(str).tolist()
        genes = cached['raw_genes'].astype(str).tolist()
        expected_control = cached['controls'].astype(np.float32)
        expected_effect = cached['effects'].reshape(len(contexts), len(perts),
                                                     len(genes)).astype(np.float32)
    obj = ad.read_h5ad(RAW, backed='r')
    try:
        raw_genes = obj.var_names.astype(str).tolist()
        index = {gene: i for i, gene in enumerate(raw_genes)}
        if any(gene not in index for gene in genes):
            raise ValueError('E112 gene panel not in raw source')
        c = obj.obs['celltype'].astype(str).to_numpy()
        p = obj.obs['perturbation'].astype(str).to_numpy()
        keep = np.isin(c, contexts) & (np.isin(p, perts) | (p == 'control'))
        labels = np.asarray([f'{context}\x1f{pert}' for context, pert
                             in zip(c[keep], p[keep])], dtype=str)
        groups, codes = np.unique(labels, return_inverse=True)
        if len(groups) != len(contexts) * (len(perts) + 1):
            raise ValueError('E112 context/perturbation groups changed')
        # E112 normalized before selecting the 512 fixed genes.
        expression = normalized_log1p(sp.csr_matrix(obj.X[keep]))[:,
                          [index[gene] for gene in genes]]
        order_in_group = pd.Series(codes).groupby(codes).cumcount().to_numpy()
        half_code = 2 * codes + order_in_group % 2
        counts = np.bincount(half_code, minlength=2 * len(groups))
        if (counts < 15).any():
            raise ValueError('source/control group has <15 cells in a half')
        membership = sp.csr_matrix((np.ones(len(codes), np.float32),
                                    (half_code, np.arange(len(codes)))),
                                   shape=(2 * len(groups), len(codes)))
        sums = membership @ expression
        means = np.asarray(sums.multiply((1 / counts)[:, None]).toarray(),
                           dtype=np.float32)
        half = {key: means[2*i:2*i+2] for i, key in enumerate(groups)}
        full = {key: (means[2*i] * counts[2*i] +
                      means[2*i+1] * counts[2*i+1]) /
                     (counts[2*i] + counts[2*i+1])
                for i, key in enumerate(groups)}
        max_control = max(float(np.max(np.abs(
            full[f'{context}\x1fcontrol'] - expected_control[i])))
            for i, context in enumerate(contexts))
        max_effect = max(float(np.max(np.abs(
            full[f'{context}\x1f{pert}'] - full[f'{context}\x1fcontrol'] -
            expected_effect[i, j])))
            for i, context in enumerate(contexts) for j, pert in enumerate(perts))
        if max(max_control, max_effect) > 2e-5:
            raise ValueError(f'E112 cache mismatch: control={max_control}, effect={max_effect}')
        effects = {}
        for context in contexts:
            control = half[f'{context}\x1fcontrol']
            for pert in perts:
                effects[context, pert] = half[f'{context}\x1f{pert}'] - control
        return effects, {
            'n_raw_cells': obj.n_obs,
            'n_selected_cells': int(keep.sum()),
            'n_groups': len(groups),
            'n_fixed_genes': len(genes),
            'minimum_half_cells': int(counts.min()),
            'maximum_full_control_difference': max_control,
            'maximum_full_effect_difference': max_effect,
            'raw_size_bytes': RAW.stat().st_size,
            'cache_sha256': sha256(CACHE),
            'manifest_sha256': sha256(MANIFEST),
        }
    finally:
        obj.file.close()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    halves, raw_audit = split_mean_vectors()
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest.loc[manifest.dataset.eq('Lara_exvivo') &
                            manifest.split.eq('train') &
                            manifest.in_train_fraction_100.astype(bool)]
    allowed = set(map(tuple, manifest[['fold_id', 'context', 'perturbation']]
                      .astype(str).itertuples(index=False, name=None)))
    task = pd.read_csv(TASKS)
    with np.load(CACHE, allow_pickle=False) as cached:
        contexts = cached['contexts'].astype(str).tolist()
    rows = []
    for item in task.itertuples(index=False):
        source = [context for context in contexts
                  if (item.fold_id, context, item.perturbation) in allowed]
        if len(source) != item.n_train_source_contexts or len(source) < 2 or \
           item.context in source:
            raise ValueError('E99/E253 source-only history mismatch')
        values = np.stack([halves[context, item.perturbation]
                           for context in source])
        means = values.mean(axis=1)
        observed_var = means.var(axis=0)
        noise_var = ((values[:, 0] - values[:, 1]) ** 2).mean(axis=0) / 4
        biological_var = np.maximum(observed_var -
                                    (1 - 1 / len(source)) * noise_var, 0)
        rows.append({'fold_id': item.fold_id, 'task_id': item.task_id,
                     'n_train_source_contexts': len(source),
                     'H_raw_split_reconstructed': float(np.sqrt(observed_var.mean())),
                     'H_bio_approx': float(np.sqrt(biological_var.mean())),
                     'source_split_noise': float(np.sqrt(noise_var.mean()))})
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    audit = {'status': 'SOURCE_ONLY_SPLIT_HALF_FEATURES_NO_ERROR_LABELS_READ',
             'n_tasks': len(frame), 'n_folds': frame.fold_id.nunique(),
             'raw_audit': raw_audit,
             'task_table_sha256': sha256(TASKS),
             'limits': ['Split halves approximate within-source sampling variation, not pure technical noise.',
                        'E112 cache physically contains all contexts; per-fold feature construction indexes only permitted train-source pairs.',
                        'Previously public study, not an independent test.']}
    args.output.with_suffix('.status.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Prepare Kaden task means without accessing held-out test expression.

The panel is selected solely from the physically isolated training view.  The
validation stage indexes only validation rows and the frozen train panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
SPLIT = Path('/home/yyf/data/perturbench_e247/asset_label_audit_20260924/E247_LABEL_ONLY_SPLIT.csv')
TRAIN_VIEW = Path('/home/yyf/data/perturbench_e247/train_only_view_20260924/kaden_train_only.h5ad')
SOURCE = Path('/home/yyf/data/scperteval_official_20260802/kaden25rpe1_processed_complete.h5ad')
OUT = Path('/home/yyf/data/perturbench_e247/task_mean_adapter_20260924')
PANEL_N = 1024
SPLIT_SHA256 = '3b6992bad649e4c7a12932ba016eb99fb61830209d29a3ffc7389b55c2ea09af'
TRAIN_VIEW_SHA256 = 'e39d9949a13e7a80d101058069bafb4dda101d20b1c93d7d9c6db0d4a89bb70f'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def task_means(matrix: sp.spmatrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names, codes = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    counts = np.bincount(codes, minlength=len(names)).astype(np.int64)
    group = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(names), len(codes)),
    )
    means = (group @ matrix).multiply((1.0 / counts)[:, None]).toarray().astype(np.float32)
    return names.astype(str), counts, means


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'TRAIN_TASK_MEANS.npz'
    if target.exists():
        raise FileExistsError(target)
    if sha(SPLIT) != SPLIT_SHA256:
        raise ValueError('frozen split changed')
    if sha(TRAIN_VIEW) != TRAIN_VIEW_SHA256:
        raise ValueError('physical train view changed')
    split = pd.read_csv(SPLIT)
    train_labels = set(split.loc[split.split.eq('train'), 'perturbation'].astype(str))
    adata = sc.read_h5ad(TRAIN_VIEW)
    try:
        labels = adata.obs['perturbation'].astype(str).to_numpy()
        if set(labels) != train_labels | {'control'}:
            raise ValueError('train view includes forbidden/missing label')
        if not adata.var_names.is_unique:
            raise ValueError('gene symbols are not unique')
        matrix = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        names, counts, means = task_means(matrix, labels)
        control = means[np.flatnonzero(names == 'control')[0]]
        effects = means[names != 'control'] - control
        # Fixed selection rule: largest train-task effect variance, symbol tie break.
        variance = np.var(effects, axis=0, dtype=np.float64)
        genes = np.asarray(adata.var_names.astype(str).tolist(), dtype=str)
        order = np.lexsort((genes, -variance))[:PANEL_N]
        if len(order) != PANEL_N or not np.all(np.isfinite(variance[order])):
            raise ValueError('invalid train-only panel')
        np.savez_compressed(target, names=names, counts=counts,
                            gene_indices=order.astype(np.int32), genes=genes[order],
                            means=means[:, order], control=control[order])
        panel = pd.DataFrame({'gene': genes[order], 'source_gene_index': order,
                              'train_effect_variance': variance[order]})
        panel.to_csv(OUT / 'TRAIN_ONLY_PANEL.csv', index=False)
        status = {'stage': 'TRAIN_ONLY_AGGREGATION', 'created_at': datetime.now().astimezone().isoformat(),
                  'n_train_labels': len(names) - 1, 'n_controls': int(counts[names == 'control'][0]),
                  'n_train_cells': int(len(labels)), 'n_panel_genes': PANEL_N,
                  'train_view_sha256': TRAIN_VIEW_SHA256, 'split_sha256': SPLIT_SHA256,
                  'task_means_sha256': sha(target), 'panel_sha256': sha(OUT / 'TRAIN_ONLY_PANEL.csv'),
                  'validation_expression_rows_read': 0, 'test_expression_rows_read': 0}
        (OUT / 'TRAIN_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(status, ensure_ascii=False), flush=True)
    finally:
        del adata


def validate() -> None:
    target = OUT / 'VALIDATION_TASK_MEANS.npz'
    if target.exists():
        raise FileExistsError(target)
    if sha(SPLIT) != SPLIT_SHA256:
        raise ValueError('frozen split changed')
    with np.load(OUT / 'TRAIN_TASK_MEANS.npz') as saved:
        panel = saved['gene_indices'].astype(int)
        panel_genes = saved['genes'].astype(str)
    split = pd.read_csv(SPLIT)
    val_labels = set(split.loc[split.split.eq('validation'), 'perturbation'].astype(str))
    test_labels = set(split.loc[split.split.eq('test'), 'perturbation'].astype(str))
    adata = sc.read_h5ad(SOURCE, backed='r')
    try:
        labels = adata.obs['perturbation'].astype(str).to_numpy()
        mask = np.isin(labels, list(val_labels))
        if np.any(np.isin(labels[mask], list(test_labels))) or len(panel_genes) != PANEL_N:
            raise ValueError('forbidden label or panel length')
        if not np.array_equal(adata.var_names.to_numpy()[panel], panel_genes):
            raise ValueError('gene axis changed')
        selected = adata[mask, panel].to_memory()
        matrix = selected.X.tocsr() if sp.issparse(selected.X) else sp.csr_matrix(selected.X)
        names, counts, means = task_means(matrix, labels[mask])
        if set(names) != val_labels or len(names) != 376:
            raise ValueError('validation labels incomplete')
        np.savez_compressed(target, names=names, counts=counts, means=means)
        status = {'stage': 'VALIDATION_AGGREGATION', 'created_at': datetime.now().astimezone().isoformat(),
                  'n_validation_labels': len(names), 'validation_expression_rows_read': int(mask.sum()),
                  'test_expression_rows_read': 0, 'validation_means_sha256': sha(target),
                  'training_panel_sha256': sha(OUT / 'TRAIN_ONLY_PANEL.csv')}
        (OUT / 'VALIDATION_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(status, ensure_ascii=False), flush=True)
    finally:
        adata.file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'validate'))
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else validate()

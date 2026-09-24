#!/usr/bin/env python3
"""Build GO and train-control-only gene graphs for the Kaden GEARS adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/yyf/data/perturbench_e247/task_mean_adapter_20260924')
TRAIN_VIEW = Path('/home/yyf/data/perturbench_e247/train_only_view_20260924/kaden_train_only.h5ad')
SPLIT = Path('/home/yyf/data/perturbench_e247/asset_label_audit_20260924/E247_LABEL_ONLY_SPLIT.csv')
GO = Path('/home/yyf/archive/external/TxPert/data/graphs/go/go_top_50.csv')
OUT = ROOT / 'docs/实验结果/E247_kaden_crispra_unseen_tf_20260924'
GRAPH = DATA / 'E247_GEARS_GRAPHS.npz'
TRAIN_VIEW_SHA256 = 'e39d9949a13e7a80d101058069bafb4dda101d20b1c93d7d9c6db0d4a89bb70f'
GO_SHA256 = 'ad469c852ba9b8b5489749c7987467687aa566598fc0706fd7232aa27edce27b'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def edges_with_self(edges: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    node = {name: i for i, name in enumerate(labels)}
    clean = edges.loc[edges.source.isin(node) & edges.target.isin(node)].copy()
    clean = clean.drop_duplicates(['source', 'target'])
    clean = clean.loc[~clean.source.eq(clean.target)]
    source = clean.source.map(node).to_numpy(np.int64)
    target = clean.target.map(node).to_numpy(np.int64)
    weights = clean.importance.to_numpy(np.float32)
    diagonal = np.arange(len(labels), dtype=np.int64)
    index = np.stack((np.r_[source, diagonal], np.r_[target, diagonal]))
    weight = np.r_[weights, np.ones(len(labels), dtype=np.float32)]
    if np.any(weight < 0) or not np.isfinite(weight).all():
        raise ValueError('invalid graph weights')
    return index, weight


def main() -> None:
    if GRAPH.exists():
        raise FileExistsError(GRAPH)
    if sha(TRAIN_VIEW) != TRAIN_VIEW_SHA256 or sha(GO) != GO_SHA256:
        raise ValueError('frozen input hash changed')
    split = pd.read_csv(SPLIT)
    perturbations = sorted(split.loc[~split.split.eq('excluded'), 'perturbation'].astype(str))
    if len(perturbations) != 1836:
        raise ValueError('perturbation axis changed')
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as saved:
        panel = saved['gene_indices'].astype(int)
        genes = saved['genes'].astype(str).tolist()
    go = pd.read_csv(GO, usecols=['source', 'target', 'importance'])
    go_index, go_weight = edges_with_self(go, perturbations)
    adata = sc.read_h5ad(TRAIN_VIEW, backed='r')
    try:
        mask = adata.obs['perturbation'].astype(str).eq('control').to_numpy()
        if int(mask.sum()) != 42233 or not np.array_equal(adata.var_names.to_numpy()[panel], genes):
            raise ValueError('train controls/panel changed')
        selected = adata[mask, panel].to_memory()
        matrix = selected.X.toarray() if sp.issparse(selected.X) else np.asarray(selected.X)
        corr = np.corrcoef(np.asarray(matrix, dtype=np.float32), rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0, posinf=0, neginf=0)
        rows = []
        for target_idx, target in enumerate(genes):
            candidates = np.argsort(corr[:, target_idx], kind='stable')[-21:][::-1]
            for source_idx in candidates:
                value = float(corr[source_idx, target_idx])
                if source_idx != target_idx and value >= .4:
                    rows.append({'source': genes[int(source_idx)], 'target': target, 'importance': value})
        co = pd.DataFrame(rows, columns=['source', 'target', 'importance'])
        co_index, co_weight = edges_with_self(co, genes)
    finally:
        adata.file.close()
    np.savez_compressed(GRAPH, perturbations=np.asarray(perturbations, dtype=str),
                        genes=np.asarray(genes, dtype=str),
                        go_index=go_index, go_weight=go_weight,
                        co_index=co_index, co_weight=co_weight)
    status = {'stage': 'E247_GEARS_TRAIN_ONLY_GRAPH_PREP',
              'created_at': datetime.now().astimezone().isoformat(),
              'n_control_cells_read': 42233, 'n_validation_expression_rows_read': 0,
              'n_test_expression_rows_read': 0,
              'n_perturbation_nodes': len(perturbations), 'n_gene_nodes': len(genes),
              'n_go_edges_including_self': int(go_index.shape[1]),
              'n_gene_edges_including_self': int(co_index.shape[1]),
              'graph_sha256': sha(GRAPH), 'train_view_sha256': TRAIN_VIEW_SHA256,
              'GO_sha256': GO_SHA256}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'E247_GEARS_GRAPH_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

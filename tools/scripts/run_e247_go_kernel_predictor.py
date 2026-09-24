#!/usr/bin/env python3
"""Frozen GO-neighborhood kernel predictor for E247 validation only.

No test expression is opened. The model is an upstream perturbation predictor,
not the SafeConf risk estimator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/yyf/data/perturbench_e247/task_mean_adapter_20260924')
SPLIT = Path('/home/yyf/data/perturbench_e247/asset_label_audit_20260924/E247_LABEL_ONLY_SPLIT.csv')
GO = Path('/home/yyf/archive/external/TxPert/data/graphs/go/go_top_50.csv')
OUT = ROOT / 'docs/实验结果/E247_kaden_crispra_unseen_tf_20260924'
LAMBDA_FACTORS = (0.001, 0.01, 0.1, 1.0, 10.0)
GRAPH_SCALES = (0.25, 0.5, 1.0)
INTERCEPT_SCALES = (0.0, 0.5, 1.0)
GO_SHA256 = 'ad469c852ba9b8b5489749c7987467687aa566598fc0706fd7232aa27edce27b'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def go_features(labels: list[str]) -> np.ndarray:
    if sha(GO) != GO_SHA256:
        raise ValueError('frozen GO graph changed')
    nodes = {name: i for i, name in enumerate(labels)}
    graph = pd.read_csv(GO, usecols=['source', 'target', 'importance'])
    graph = graph.loc[graph.source.isin(nodes) & graph.target.isin(nodes)]
    graph = graph.loc[~graph.source.eq(graph.target)].drop_duplicates(['target', 'source'])
    rows = graph.target.map(nodes).to_numpy(np.int32)
    cols = graph.source.map(nodes).to_numpy(np.int32)
    weights = graph.importance.to_numpy(np.float32)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('GO weights invalid')
    n = len(labels)
    adjacency = sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32)
    adjacency = adjacency + sp.eye(n, format='csr', dtype=np.float32)
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    adjacency = sp.diags(1.0 / row_sums, format='csr') @ adjacency
    two_hop = adjacency @ adjacency
    features = np.concatenate([adjacency.toarray(), 0.5 * two_hop.toarray()], axis=1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-12)).astype(np.float32)


def mean_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2, axis=1)).mean())


def main() -> None:
    split = pd.read_csv(SPLIT)
    if len(split) != 1837 or set(split.split) != {'train', 'validation', 'test', 'excluded'}:
        raise ValueError('frozen label split changed')
    labels = sorted(split.loc[~split.split.eq('excluded'), 'perturbation'].astype(str))
    label_to_index = {name: i for i, name in enumerate(labels)}
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as tr:
        tr_names = tr['names'].astype(str)
        tr_means = tr['means'].astype(np.float32)
        control = tr['control'].astype(np.float32)
        genes = tr['genes'].astype(str)
    with np.load(DATA / 'VALIDATION_TASK_MEANS.npz', allow_pickle=False) as va:
        val_names = va['names'].astype(str)
        val_means = va['means'].astype(np.float32)
    train_labels = set(split.loc[split.split.eq('train'), 'perturbation'].astype(str))
    val_labels = set(split.loc[split.split.eq('validation'), 'perturbation'].astype(str))
    test_labels = set(split.loc[split.split.eq('test'), 'perturbation'].astype(str))
    if set(tr_names) != train_labels | {'control'} or set(val_names) != val_labels:
        raise ValueError('task means labels do not match frozen split')
    if train_labels & val_labels or train_labels & test_labels or val_labels & test_labels:
        raise ValueError('split overlap')
    order = tr_names != 'control'
    train_names = tr_names[order]
    train_effect = tr_means[order] - control
    true_val_effect = val_means - control
    if train_effect.shape != (1093, 1024) or true_val_effect.shape != (376, 1024):
        raise ValueError('task mean axes mismatch')

    val_hash = {name: hashlib.sha256(('E247:tune:' + name).encode()).hexdigest() for name in val_names}
    tuning_set = set(sorted(val_names, key=lambda name: val_hash[name])[:188])
    tune = np.asarray([name in tuning_set for name in val_names], dtype=bool)
    gate = ~tune
    if int(tune.sum()) != 188 or int(gate.sum()) != 188:
        raise ValueError('validation split error')
    features = go_features(labels)
    f_train = features[[label_to_index[name] for name in train_names]]
    f_val = features[[label_to_index[name] for name in val_names]]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    t_train = torch.as_tensor(f_train, device=device)
    t_val = torch.as_tensor(f_val, device=device)
    target_mean = train_effect.mean(axis=0)
    target_centered = torch.as_tensor(train_effect - target_mean, device=device)
    train_kernel = t_train @ t_train.T
    val_kernel = t_val @ t_train.T
    diagonal_scale = float(torch.median(torch.diag(train_kernel)).cpu())
    identity = torch.eye(len(train_names), dtype=torch.float32, device=device)
    grids: list[dict] = []
    best: tuple[float, float, float, float] | None = None
    best_pred: np.ndarray | None = None
    best_alpha: np.ndarray | None = None
    baseline_zero_tune = mean_rmse(np.zeros_like(true_val_effect[tune]), true_val_effect[tune])
    baseline_zero_gate = mean_rmse(np.zeros_like(true_val_effect[gate]), true_val_effect[gate])
    baseline_mean_tune = mean_rmse(np.tile(target_mean, (int(tune.sum()), 1)), true_val_effect[tune])
    baseline_mean_gate = mean_rmse(np.tile(target_mean, (int(gate.sum()), 1)), true_val_effect[gate])

    for factor in LAMBDA_FACTORS:
        penalty = factor * max(diagonal_scale, 1e-8)
        alpha = torch.linalg.solve(train_kernel + penalty * identity, target_centered)
        graph_val = (val_kernel @ alpha).detach().cpu().numpy()
        for graph_scale in GRAPH_SCALES:
            for intercept_scale in INTERCEPT_SCALES:
                pred = graph_scale * graph_val + intercept_scale * target_mean
                tune_rmse = mean_rmse(pred[tune], true_val_effect[tune])
                row = {'lambda_factor': factor, 'lambda_absolute': penalty,
                       'graph_scale': graph_scale, 'intercept_scale': intercept_scale,
                       'tuning_mean_rmse': tune_rmse,
                       'tuning_no_change_mean_rmse': baseline_zero_tune}
                grids.append(row)
                key = (tune_rmse, -factor, graph_scale, intercept_scale)
                if best is None or key < best:
                    best, best_pred = key, pred.copy()
                    best_alpha = alpha.detach().cpu().numpy().copy()
    if best_pred is None or best_alpha is None:
        raise RuntimeError('no model candidate')
    selected = min(grids, key=lambda row: (row['tuning_mean_rmse'], -row['lambda_factor'],
                                           row['graph_scale'], row['intercept_scale']))
    rmse_model = np.sqrt(np.mean((best_pred[gate] - true_val_effect[gate]) ** 2, axis=1))
    rmse_zero = np.sqrt(np.mean(true_val_effect[gate] ** 2, axis=1))
    relative_gain = 1.0 - float(rmse_model.mean() / rmse_zero.mean())
    rng = np.random.default_rng(20260924)
    boots = np.empty(3000, dtype=np.float64)
    for i in range(len(boots)):
        sampled = rng.integers(0, len(rmse_zero), size=len(rmse_zero))
        boots[i] = 1.0 - float(rmse_model[sampled].mean() / rmse_zero[sampled].mean())
    ci = np.quantile(boots, [.025, .975]).tolist()
    passed = bool(relative_gain >= .02 and ci[0] > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grids).to_csv(OUT / 'E247_GO_KERNEL_TUNING_GRID.csv', index=False)
    detail = pd.DataFrame({'perturbation': val_names,
                           'validation_part': np.where(tune, 'tuning', 'capability_gate'),
                           'model_rmse': np.sqrt(np.mean((best_pred - true_val_effect) ** 2, axis=1)),
                           'no_change_rmse': np.sqrt(np.mean(true_val_effect ** 2, axis=1)),
                           'mean_effect_rmse': np.sqrt(np.mean((target_mean - true_val_effect) ** 2, axis=1)),
                           'predicted_magnitude': np.sqrt(np.mean(best_pred ** 2, axis=1))})
    detail.to_csv(OUT / 'E247_GO_KERNEL_VALIDATION_TASKS.csv', index=False)
    checkpoint = DATA / 'E247_GO_KERNEL_SELECTED_MODEL.npz'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    np.savez_compressed(checkpoint, alpha=best_alpha, train_names=train_names.astype(str),
                        train_mean_effect=target_mean, genes=genes,
                        control=control, selected_penalty=np.float32(selected['lambda_absolute']),
                        graph_scale=np.float32(selected['graph_scale']),
                        intercept_scale=np.float32(selected['intercept_scale']))
    status = {'stage': 'E247_GO_KERNEL_VALIDATION_GATE', 'created_at': datetime.now().astimezone().isoformat(),
              'device': str(device), 'n_train_labels': len(train_names), 'n_tuning_labels': int(tune.sum()),
              'n_gate_labels': int(gate.sum()), 'n_test_expression_rows_read': 0,
              'selected': selected, 'tuning_baselines': {'no_change': baseline_zero_tune,
                                                         'train_mean_effect': baseline_mean_tune},
              'gate_mean_rmse': {'model': float(rmse_model.mean()), 'no_change': baseline_zero_gate,
                                 'train_mean_effect': baseline_mean_gate},
              'gate_relative_gain_vs_no_change': relative_gain,
              'gate_bootstrap_95_ci': ci, 'capability_gate_passed': passed,
              'train_means_sha256': sha(DATA / 'TRAIN_TASK_MEANS.npz'),
              'validation_means_sha256': sha(DATA / 'VALIDATION_TASK_MEANS.npz'),
              'checkpoint_sha256': sha(checkpoint)}
    (OUT / 'E247_GO_KERNEL_VALIDATION_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

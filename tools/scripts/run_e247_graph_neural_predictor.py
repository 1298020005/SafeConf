#!/usr/bin/env python3
"""Two-seed full-graph upstream perturbation predictor for Kaden E247."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from run_e247_go_kernel_predictor import go_features, GO, GO_SHA256


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/yyf/data/perturbench_e247/task_mean_adapter_20260924')
SPLIT = Path('/home/yyf/data/perturbench_e247/asset_label_audit_20260924/E247_LABEL_ONLY_SPLIT.csv')
OUT = ROOT / 'docs/实验结果/E247_kaden_crispra_unseen_tf_20260924'
SEEDS = (31, 47)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assets():
    split = pd.read_csv(SPLIT)
    labels = sorted(split.loc[~split.split.eq('excluded'), 'perturbation'].astype(str))
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as tr:
        tr_names = tr['names'].astype(str)
        tr_means = tr['means'].astype(np.float32)
        control = tr['control'].astype(np.float32)
    with np.load(DATA / 'VALIDATION_TASK_MEANS.npz', allow_pickle=False) as va:
        va_names = va['names'].astype(str)
        va_means = va['means'].astype(np.float32)
    train_names = tr_names[tr_names != 'control']
    train_effect = tr_means[tr_names != 'control'] - control
    val_effect = va_means - control
    if set(train_names) != set(split.loc[split.split.eq('train'), 'perturbation']) or \
       set(va_names) != set(split.loc[split.split.eq('validation'), 'perturbation']):
        raise ValueError('frozen train/validation task alignment changed')
    if len(labels) != 1836 or train_effect.shape != (1093, 1024) or val_effect.shape != (376, 1024):
        raise ValueError('unexpected axis sizes')
    nodes = {name: i for i, name in enumerate(labels)}
    if sha(GO) != GO_SHA256:
        raise ValueError('GO graph hash changed')
    graph = pd.read_csv(GO, usecols=['source', 'target', 'importance'])
    graph = graph.loc[graph.source.isin(nodes) & graph.target.isin(nodes)]
    graph = graph.loc[~graph.source.eq(graph.target)].drop_duplicates(['target', 'source'])
    row = graph.target.map(nodes).to_numpy(np.int32)
    col = graph.source.map(nodes).to_numpy(np.int32)
    weight = graph.importance.to_numpy(np.float32)
    adjacency = sp.csr_matrix((weight, (row, col)), shape=(len(labels), len(labels)))
    adjacency = adjacency + sp.eye(len(labels), format='csr', dtype=np.float32)
    adjacency = sp.diags(1.0 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    features = go_features(labels)
    tune_names = set(sorted(va_names,
                            key=lambda name: hashlib.sha256(('E247:tune:' + name).encode()).hexdigest())[:188])
    tune_mask = np.asarray([name in tune_names for name in va_names], dtype=bool)
    return labels, train_names, train_effect, va_names, val_effect, features, adjacency.toarray().astype(np.float32), tune_mask


class GraphPredictor(nn.Module):
    def __init__(self, feature_dim: int, n_genes: int):
        super().__init__()
        self.encoder = nn.Linear(feature_dim, 128)
        self.norm0 = nn.LayerNorm(128)
        self.mix1 = nn.Linear(256, 128)
        self.norm1 = nn.LayerNorm(128)
        self.mix2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.decoder = nn.Linear(128, n_genes)
        self.dropout = nn.Dropout(.10)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor,
                train_mean: torch.Tensor) -> torch.Tensor:
        h = self.dropout(torch.relu(self.norm0(self.encoder(features))))
        h = self.dropout(torch.relu(self.norm1(self.mix1(torch.cat((h, adjacency @ h), dim=1)))))
        h = self.dropout(torch.relu(self.norm2(self.mix2(torch.cat((h, adjacency @ h), dim=1)))))
        return self.decoder(h) + train_mean


def mean_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred-truth)**2, axis=1)).mean())


def train(seed: int, device_name: str) -> None:
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    labels, tr_names, tr_effect, va_names, va_effect, features, adjacency, tune_mask = assets()
    nodes = {name: i for i, name in enumerate(labels)}
    tr_idx = torch.tensor([nodes[name] for name in tr_names], device=device)
    va_idx = torch.tensor([nodes[name] for name in va_names], device=device)
    ft = torch.tensor(features, device=device)
    adj = torch.tensor(adjacency, device=device)
    target = torch.tensor(tr_effect, device=device)
    mean = torch.tensor(tr_effect.mean(axis=0), device=device)
    model = GraphPredictor(features.shape[1], tr_effect.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    history = []
    for epoch in range(1, 251):
        model.train()
        pred = model(ft, adj, mean)
        loss = torch.mean((pred[tr_idx] - target)**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(ft, adj, mean)[va_idx].cpu().numpy()
        tune_error = mean_rmse(val_pred[tune_mask], va_effect[tune_mask])
        history.append({'seed': seed, 'epoch': epoch, 'train_mse': float(loss.detach().cpu()),
                        'tuning_mean_rmse': tune_error})
        if tune_error < best - 1e-7:
            best, best_epoch, stale = tune_error, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'seed={seed} epoch={epoch} train_mse={history[-1]["train_mse"]:.7f} tune_rmse={tune_error:.6f}', flush=True)
        if stale >= 30:
            break
    if best_state is None:
        raise RuntimeError('no checkpoint')
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        pred = model(ft, adj, mean)[va_idx].cpu().numpy()
    gate = ~tune_mask
    no_change = mean_rmse(np.zeros_like(va_effect[gate]), va_effect[gate])
    train_mean = mean_rmse(np.tile(tr_effect.mean(axis=0), (int(gate.sum()), 1)), va_effect[gate])
    out_data = DATA / f'E247_GNN_SEED_{seed}_VAL_PRED.npz'
    checkpoint = DATA / f'E247_GNN_SEED_{seed}_CHECKPOINT.pt'
    if out_data.exists() or checkpoint.exists():
        raise FileExistsError(f'seed {seed} output already exists')
    np.savez_compressed(out_data, names=va_names.astype(str), prediction=pred.astype(np.float32))
    torch.save({'state_dict': best_state, 'seed': seed, 'best_epoch': best_epoch,
                'labels_sha256': hashlib.sha256('\n'.join(labels).encode()).hexdigest()}, checkpoint)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUT / f'E247_GNN_SEED_{seed}_TRAINING_HISTORY.csv', index=False)
    status = {'seed': seed, 'stage': 'E247_GNN_VALIDATION',
              'created_at': datetime.now().astimezone().isoformat(), 'device': str(device),
              'epochs_completed': len(history), 'best_epoch': best_epoch,
              'tuning_best_mean_rmse': best, 'gate_mean_rmse': mean_rmse(pred[gate], va_effect[gate]),
              'gate_no_change_mean_rmse': no_change, 'gate_train_mean_effect_rmse': train_mean,
              'n_test_expression_rows_read': 0,
              'prediction_sha256': sha(out_data), 'checkpoint_sha256': sha(checkpoint)}
    (OUT / f'E247_GNN_SEED_{seed}_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


def aggregate() -> None:
    statuses = [json.loads((OUT / f'E247_GNN_SEED_{seed}_STATUS.json').read_text()) for seed in SEEDS]
    with np.load(DATA / 'VALIDATION_TASK_MEANS.npz', allow_pickle=False) as va:
        names = va['names'].astype(str)
        val_means = va['means'].astype(np.float32)
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as tr:
        control = tr['control'].astype(np.float32)
        train_mean_effect = tr['means'][tr['names'] != 'control'].mean(axis=0).astype(np.float32) - control
    truth = val_means - control
    tune_names = set(sorted(names, key=lambda name: hashlib.sha256(('E247:tune:' + name).encode()).hexdigest())[:188])
    tune = np.asarray([name in tune_names for name in names], dtype=bool)
    pred = []
    for seed in SEEDS:
        path = DATA / f'E247_GNN_SEED_{seed}_VAL_PRED.npz'
        if sha(path) != statuses[SEEDS.index(seed)]['prediction_sha256']:
            raise ValueError('prediction artifact changed')
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved['names'].astype(str), names):
                raise ValueError('seed prediction label order mismatch')
            pred.append(saved['prediction'].astype(np.float32))
    ensemble = np.mean(pred, axis=0)
    gate = ~tune
    baselines = {'no_change': np.zeros_like(truth),
                 'train_mean_effect': np.tile(train_mean_effect, (len(names), 1)),
                 'GO_kernel': None,
                 'GNN_seed_31': pred[0], 'GNN_seed_47': pred[1], 'GNN_two_seed_ensemble': ensemble}
    with np.load(DATA / 'E247_GO_KERNEL_SELECTED_MODEL.npz', allow_pickle=False) as saved:
        # The KRR validation task predictions are reproducible from its script;
        # only the previously published gate mean is repeated in the report.
        kernel_checkpoint_sha = sha(DATA / 'E247_GO_KERNEL_SELECTED_MODEL.npz')
    del baselines['GO_kernel']
    rows = []
    for method, values in baselines.items():
        for split_name, mask in (('tuning', tune), ('capability_gate', gate)):
            rows.append({'method': method, 'validation_part': split_name,
                         'n_tasks': int(mask.sum()), 'mean_rmse': mean_rmse(values[mask], truth[mask])})
    pd.DataFrame(rows).to_csv(OUT / 'E247_GNN_VALIDATION_COMPARISON.csv', index=False)
    err_ensemble = np.sqrt(np.mean((ensemble[gate]-truth[gate])**2, axis=1))
    err_mean = np.sqrt(np.mean((train_mean_effect-truth[gate])**2, axis=1))
    relative_gain = 1.0 - float(err_ensemble.mean()/err_mean.mean())
    rng = np.random.default_rng(20260924)
    boots = np.empty(3000, dtype=np.float64)
    for i in range(len(boots)):
        selected = rng.integers(0, len(err_mean), size=len(err_mean))
        boots[i] = 1.0 - float(err_ensemble[selected].mean()/err_mean[selected].mean())
    ci = np.quantile(boots, [.025, .975]).tolist()
    zero = mean_rmse(np.zeros_like(truth[gate]), truth[gate])
    status = {'stage': 'E247_GNN_TWO_SEED_ENSEMBLE_VALIDATION',
              'created_at': datetime.now().astimezone().isoformat(),
              'gate_mean_rmse': {'ensemble': float(err_ensemble.mean()),
                                 'train_mean_effect': float(err_mean.mean()), 'no_change': zero},
              'relative_gain_vs_train_mean_effect': relative_gain,
              'bootstrap_95_ci': ci,
              'strict_capability_gate_passed': bool(float(err_ensemble.mean()) < zero
                                                    and relative_gain > 0 and ci[0] > 0),
              'n_test_expression_rows_read': 0,
              'seed_checkpoints': {str(s): statuses[i]['checkpoint_sha256'] for i, s in enumerate(SEEDS)},
              'GO_kernel_checkpoint_sha256': kernel_checkpoint_sha}
    (OUT / 'E247_GNN_ENSEMBLE_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'aggregate'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.stage == 'train':
        if args.seed is None:
            parser.error('--seed required for train')
        train(args.seed, args.device)
    else:
        aggregate()

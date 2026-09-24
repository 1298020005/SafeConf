#!/usr/bin/env python3
"""GEARS/PRESCRIBE architecture on isolated Kaden task means (E247)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/yyf/data/perturbench_e247/task_mean_adapter_20260924')
OUT = ROOT / 'docs/实验结果/E247_kaden_crispra_unseen_tf_20260924'
GEARS_SOURCE = Path('/home/yyf/archive/external/PRESCRIBE/gears/model.py')
SEEDS = (31, 47)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_class():
    spec = importlib.util.spec_from_file_location('e247_prescribe_gears_model', GEARS_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import fixed GEARS source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GEARS_Model


def load_assets():
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as tr:
        tr_names = tr['names'].astype(str)
        tr_means = tr['means'].astype(np.float32)
        control = tr['control'].astype(np.float32)
        genes = tr['genes'].astype(str)
    with np.load(DATA / 'VALIDATION_TASK_MEANS.npz', allow_pickle=False) as va:
        va_names = va['names'].astype(str)
        va_means = va['means'].astype(np.float32)
    with np.load(DATA / 'E247_GEARS_GRAPHS.npz', allow_pickle=False) as graph:
        labels = graph['perturbations'].astype(str)
        if not np.array_equal(genes, graph['genes'].astype(str)):
            raise ValueError('graph gene axis mismatch')
        graphs = {key: torch.from_numpy(graph[key]) for key in
                  ('go_index', 'go_weight', 'co_index', 'co_weight')}
    train = tr_names != 'control'
    if int(train.sum()) != 1093 or len(va_names) != 376 or len(labels) != 1836:
        raise ValueError('task inventory changed')
    tune_set = set(sorted(va_names,
                          key=lambda name: hashlib.sha256(('E247:tune:' + name).encode()).hexdigest())[:188])
    tune = np.asarray([name in tune_set for name in va_names], dtype=bool)
    return tr_names[train], tr_means[train], va_names, va_means, control, labels, graphs, tune


def graphs_for_tasks(names: np.ndarray, targets: np.ndarray, control: np.ndarray,
                     label_to_index: dict[str, int]) -> list[Data]:
    x = torch.from_numpy(control.reshape(-1, 1))
    return [Data(x=x, y=torch.from_numpy(targets[i]).unsqueeze(0),
                 pert_idx=[label_to_index[str(name)]], task_name=str(name))
            for i, name in enumerate(names)]


def evaluate(model, graphs: list[Data], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False):
            batch = batch.to(device)
            output = model(batch)
            predictions.append(output.detach().cpu().numpy())
            truths.append(batch.y.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(truths)


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
    tr_names, tr_means, va_names, va_means, control, labels, graphs, tune = load_assets()
    node = {name: i for i, name in enumerate(labels)}
    train_graphs = graphs_for_tasks(tr_names, tr_means, control, node)
    tune_graphs = graphs_for_tasks(va_names[tune], va_means[tune], control, node)
    validation_graphs = graphs_for_tasks(va_names, va_means, control, node)
    config = {'num_genes': 1024, 'num_perts': 1836, 'hidden_size': 64,
              'num_go_gnn_layers': 1, 'num_gene_gnn_layers': 1,
              'decoder_hidden_size': 16, 'uncertainty': False,
              'no_perturb': False, 'device': str(device),
              'G_go': graphs['go_index'], 'G_go_weight': graphs['go_weight'],
              'G_coexpress': graphs['co_index'], 'G_coexpress_weight': graphs['co_weight']}
    model = model_class()(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(train_graphs, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    history = []
    for epoch in range(1, 101):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.mean((pred-batch.y)**2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred, val_truth = evaluate(model, tune_graphs, device)
        tune_error = mean_rmse(val_pred, val_truth)
        history.append({'seed': seed, 'epoch': epoch,
                        'train_mean_batch_mse': float(np.mean(losses)),
                        'tuning_mean_rmse': tune_error})
        if tune_error < best-1e-7:
            best, best_epoch, stale = tune_error, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'seed={seed} epoch={epoch} train_mse={history[-1]["train_mean_batch_mse"]:.7f} tune_rmse={tune_error:.6f}', flush=True)
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError('no checkpoint')
    model.load_state_dict(best_state)
    model.to(device)
    val_pred, val_truth = evaluate(model, validation_graphs, device)
    if np.max(np.abs(val_truth-va_means)) > 1e-6:
        raise ValueError('validation target axis mismatch')
    out_pred = DATA / f'E247_GEARS_SEED_{seed}_VAL_PRED.npz'
    checkpoint = DATA / f'E247_GEARS_SEED_{seed}_CHECKPOINT.pt'
    if out_pred.exists() or checkpoint.exists():
        raise FileExistsError(f'seed {seed} output exists')
    np.savez_compressed(out_pred, names=va_names.astype(str), prediction=val_pred.astype(np.float32))
    torch.save({'state_dict': best_state, 'seed': seed, 'best_epoch': best_epoch,
                'source_sha256': sha(GEARS_SOURCE), 'graph_sha256': sha(DATA / 'E247_GEARS_GRAPHS.npz')}, checkpoint)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUT / f'E247_GEARS_SEED_{seed}_TRAINING_HISTORY.csv', index=False)
    status = {'stage': 'E247_GEARS_MEAN_ADAPTER_VALIDATION',
              'created_at': datetime.now().astimezone().isoformat(),
              'seed': seed, 'device': str(device), 'epochs_completed': len(history),
              'best_epoch': best_epoch, 'tuning_best_mean_rmse': best,
              'gate_mean_rmse': mean_rmse(val_pred[~tune], va_means[~tune]),
              'gate_train_mean_effect_rmse': mean_rmse(
                  np.tile(tr_means.mean(axis=0), (int((~tune).sum()), 1)), va_means[~tune]),
              'gate_no_change_mean_rmse': mean_rmse(
                  np.tile(control, (int((~tune).sum()), 1)), va_means[~tune]),
              'test_expression_rows_read': 0, 'graph_sha256': sha(DATA / 'E247_GEARS_GRAPHS.npz'),
              'source_sha256': sha(GEARS_SOURCE), 'prediction_sha256': sha(out_pred),
              'checkpoint_sha256': sha(checkpoint)}
    (OUT / f'E247_GEARS_SEED_{seed}_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


def aggregate() -> None:
    statuses = [json.loads((OUT / f'E247_GEARS_SEED_{seed}_STATUS.json').read_text()) for seed in SEEDS]
    with np.load(DATA / 'TRAIN_TASK_MEANS.npz', allow_pickle=False) as tr:
        means = tr['means'].astype(np.float32)
        names_train = tr['names'].astype(str)
        control = tr['control'].astype(np.float32)
    with np.load(DATA / 'VALIDATION_TASK_MEANS.npz', allow_pickle=False) as va:
        names = va['names'].astype(str)
        truth = va['means'].astype(np.float32)
    tune_set = set(sorted(names,
                          key=lambda name: hashlib.sha256(('E247:tune:' + name).encode()).hexdigest())[:188])
    gate = np.asarray([name not in tune_set for name in names], dtype=bool)
    preds = []
    for seed, status in zip(SEEDS, statuses):
        path = DATA / f'E247_GEARS_SEED_{seed}_VAL_PRED.npz'
        if sha(path) != status['prediction_sha256']:
            raise ValueError('prediction hash changed')
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(names, saved['names'].astype(str)):
                raise ValueError('validation order mismatch')
            preds.append(saved['prediction'].astype(np.float32))
    ensemble = (preds[0]+preds[1])/2
    train_mean = means[names_train != 'control'].mean(axis=0)
    candidates = {'no_change': np.tile(control, (len(names), 1)),
                  'train_mean_effect': np.tile(train_mean, (len(names), 1)),
                  'GEARS_seed_31': preds[0], 'GEARS_seed_47': preds[1],
                  'GEARS_two_seed_ensemble': ensemble}
    rows = [{'method': method, 'validation_part': 'capability_gate',
             'n_tasks': int(gate.sum()), 'mean_rmse': mean_rmse(values[gate], truth[gate])}
            for method, values in candidates.items()]
    pd.DataFrame(rows).to_csv(OUT / 'E247_GEARS_VALIDATION_COMPARISON.csv', index=False)
    e = np.sqrt(np.mean((ensemble[gate]-truth[gate])**2, axis=1))
    b = np.sqrt(np.mean((train_mean-truth[gate])**2, axis=1))
    improvement = 1.0-float(e.mean()/b.mean())
    rng = np.random.default_rng(20260924)
    boot = np.empty(3000)
    for i in range(len(boot)):
        selected = rng.integers(0, len(e), size=len(e))
        boot[i] = 1.0-float(e[selected].mean()/b[selected].mean())
    ci = np.quantile(boot, [.025, .975]).tolist()
    zero = mean_rmse(candidates['no_change'][gate], truth[gate])
    status = {'stage': 'E247_GEARS_TWO_SEED_ENSEMBLE_VALIDATION',
              'created_at': datetime.now().astimezone().isoformat(),
              'gate_mean_rmse': {'ensemble': float(e.mean()),
                                 'train_mean_effect': float(b.mean()), 'no_change': zero},
              'relative_gain_vs_train_mean_effect': improvement,
              'bootstrap_95_ci': ci,
              'strict_capability_gate_passed': bool(float(e.mean()) < zero and improvement > 0 and ci[0] > 0),
              'test_expression_rows_read': 0,
              'seed_checkpoints': {str(seed): statuses[i]['checkpoint_sha256'] for i, seed in enumerate(SEEDS)}}
    (OUT / 'E247_GEARS_ENSEMBLE_STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
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

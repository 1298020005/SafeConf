#!/usr/bin/env python3
"""Two-seed-only scGPT (Transformer, not GNN) competence robustness audit.

Reuses the fixed E99 Lara ex-vivo 512-gene task contract and mature E112
scGPT training implementation. No GEARS model, graph edges or new architecture
is trained. Existing E112 test truth is public: this is retrospective.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
E112_FILE = ROOT / 'tools/scripts/run_e112_external_formal_dual_models.py'
spec = importlib.util.spec_from_file_location('e112_for_e259_scgpt', E112_FILE)
e112 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e112
spec.loader.exec_module(e112)
# Some large historical manifests are intentionally not present in a slim
# runtime worktree; use the exact original project artifact, never regenerate
# a split from observed outcomes.
if not e112.CONTRACT.is_file():
    e112.CONTRACT = Path('/home/yyf/proj/docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv')

DATASET = 'Lara_exvivo'
ALLOWED_SEEDS = (202609259, 202609260)


def train_scgpt(graphs: dict, assets: object, device: torch.device) -> tuple[object, object, pd.DataFrame]:
    model, _, meta = e112.E65.load_model(device)
    gene_ids = e112.E65.make_gene_ids(assets.genes, meta['vocab'])
    loader = e112.DataLoader(graphs['train'], batch_size=16, shuffle=True,
                             generator=torch.Generator().manual_seed(torch.initial_seed()))
    val = e112.DataLoader(graphs['val'], batch_size=16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    best_loss, best_state, stale = float('inf'), None, 0
    history = []
    for epoch in range(1, 11):
        train_loss = e112.E65.train_one_epoch(model, loader, gene_ids,
                                               optimizer, scaler, device,
                                               device.type == 'cuda')
        val_loss = e112.E65.evaluate_mse(model, val, gene_ids, device,
                                          device.type == 'cuda')
        history.append({'epoch': epoch, 'train_mse': float(train_loss),
                        'val_mse': float(val_loss)})
        print(f'epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}', flush=True)
        if not np.isfinite(val_loss):
            raise ValueError('nonfinite scGPT validation loss')
        if val_loss < best_loss - 1e-7:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError('scGPT produced no validation checkpoint')
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, gene_ids, pd.DataFrame(history)


def predict_test(model: object, gene_ids: object, graphs: list,
                 assets: object, source_tasks: pd.DataFrame,
                 device: torch.device) -> tuple[list[dict], dict[str, np.ndarray]]:
    allowed = set(map(tuple, source_tasks[['context', 'perturbation']].astype(str)
                      .itertuples(index=False, name=None)))
    rows, predictions = [], {}
    loader = e112.DataLoader(graphs, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            output, truth = e112.E65.model_forward(model, batch, gene_ids,
                                                    device, device.type == 'cuda', False)
            for task, context, perturbation, estimate, observed in zip(
                    batch.pert, batch.context, batch.perturbation,
                    output.detach().cpu().numpy(), truth.detach().cpu().numpy()):
                context = str(context)
                perturbation = str(perturbation)
                basal = assets.control(context)
                pred_effect = np.asarray(estimate, np.float32) - basal
                true_effect = np.asarray(observed, np.float32) - basal
                source_contexts = [c for c in assets.contexts
                                   if (c, perturbation) in allowed and c != context]
                source_mean = (np.stack([assets.effect(c, perturbation)
                                         for c in source_contexts]).mean(axis=0)
                               if source_contexts else None)
                key = str(task)
                predictions[key] = pred_effect
                rows.append({'task_id': key, 'context': context,
                             'perturbation': perturbation,
                             'n_source_contexts': len(source_contexts),
                             'scgpt_mse': float(np.mean((pred_effect - true_effect) ** 2)),
                             'nochange_mse': float(np.mean(true_effect ** 2)),
                             'train_source_mean_mse':
                                 (float(np.mean((source_mean - true_effect) ** 2))
                                  if source_mean is not None else float('nan')),
                             'scgpt_predicted_magnitude':
                                 float(np.sqrt(np.mean(pred_effect ** 2)))})
    return rows, predictions


def run(args: argparse.Namespace) -> dict:
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f'only two registered seeds allowed: {ALLOWED_SEEDS}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not e112.CONTRACT.is_file():
        raise FileNotFoundError(e112.CONTRACT)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    manifest = pd.read_csv(e112.CONTRACT, keep_default_na=False)
    manifest = manifest.loc[manifest.dataset.eq(DATASET)]
    assets = e112.Assets(DATASET)
    rows, histories, predictions = [], [], {}
    for fold_idx, (fold, part) in enumerate(manifest.groupby('fold_id', sort=True)):
        e112.seed(args.seed + fold_idx)
        part = part.loc[~part.split.eq('train') |
                        part.in_train_fraction_100.astype(bool)].copy()
        graphs = e112.graphs_for(part, assets)
        model, gene_ids, history = train_scgpt(graphs, assets, device)
        history.insert(0, 'fold_id', fold)
        histories.append(history)
        pred_rows, pred_vectors = predict_test(
            model, gene_ids, graphs['test'], assets,
            part.loc[part.split.eq('train')], device)
        for row in pred_rows:
            row['fold_id'] = fold
            rows.append(row)
        predictions.update({f'{fold}::{key}': value
                            for key, value in pred_vectors.items()})
        del model
        torch.cuda.empty_cache()
        print(f'{fold}: {len(pred_rows)} test tasks complete', flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 345 or frame.fold_id.nunique() != 5:
        raise ValueError('E99 five-fold test inventory changed')
    by_fold = frame.groupby('fold_id')[['scgpt_mse', 'nochange_mse']].mean()
    supported = frame.loc[frame.n_source_contexts >= 2]
    supported_by_fold = supported.groupby('fold_id')[['scgpt_mse', 'nochange_mse',
                                                      'train_source_mean_mse']].mean()
    if len(supported_by_fold) != 5 or supported.train_source_mean_mse.isna().any():
        raise ValueError('history-supported task contract failed')
    best_simple = supported_by_fold[['nochange_mse',
                                     'train_source_mean_mse']].mean().idxmin()
    gain = (supported_by_fold[best_simple].mean() -
            supported_by_fold.scgpt_mse.mean()) / supported_by_fold[best_simple].mean()
    win_folds = int((supported_by_fold.scgpt_mse < supported_by_fold[best_simple]).sum())
    all_gain = (by_fold.nochange_mse.mean() - by_fold.scgpt_mse.mean()) / by_fold.nochange_mse.mean()
    status = {
        'status': 'RETROSPECTIVE_UPSTREAM_COMPETENCE_AUDIT',
        'generated_at': datetime.now().astimezone().isoformat(),
        'seed': args.seed,
        'model': 'scGPT pretrained Transformer with E112 fixed training recipe',
        'new_graph_model_trained': False,
        'test_truth_used_for_model_selection': False,
        'test_truth_previously_public': True,
        'n_tasks': len(frame), 'n_folds': 5,
        'n_history_supported_tasks': len(supported),
        'all_task_relative_gain_vs_nochange': float(all_gain),
        'all_task_winning_folds_vs_nochange':
            int((by_fold.scgpt_mse < by_fold.nochange_mse).sum()),
        'best_simple_baseline_by_macro_mse': best_simple,
        'macro_scgpt_mse': float(supported_by_fold.scgpt_mse.mean()),
        'macro_best_simple_mse': float(supported_by_fold[best_simple].mean()),
        'relative_gain_vs_best_simple': float(gain),
        'winning_folds_vs_best_simple': win_folds,
        'qualified_for_risk_audit': bool(all_gain >= .02 and
                                         (by_fold.scgpt_mse < by_fold.nochange_mse).sum() >= 4 and
                                         gain >= .02 and win_folds >= 4),
        'qualification_rule': 'all tasks beat no-change and train-history-supported tasks beat same-task best simple by >=2% macro MSE and >=4/5 folds',
        'note': 'Two fixed seeds only; no tuning or test-based model selection. A passing seed is retrospective development evidence, not independent confirmation.'
    }
    frame.to_csv(args.output_dir / 'TASKS.csv', index=False)
    by_fold.reset_index().to_csv(args.output_dir / 'ALL_FOLD_CAPABILITY.csv', index=False)
    supported_by_fold.reset_index().to_csv(args.output_dir / 'SUPPORTED_FOLD_CAPABILITY.csv', index=False)
    pd.concat(histories).to_csv(args.output_dir / 'TRAINING_HISTORY.csv', index=False)
    np.savez_compressed(args.output_dir / 'PREDICTED_EFFECTS.npz', **predictions)
    (args.output_dir / 'STATUS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from encoders import stable_hash_features
from evaluators import effect_metrics, summarize_results
from program_bank import ProgramBank
from transport_models import V0StrongBaseline, V2GraphPriorTransport


MAIN_STUDIES = ["KaggleCrossCell", "Haber", "Parekh", "KaggleCrossPatient", "McFarland", "TianKampmann2019"]


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pick_datasets(scan: pd.DataFrame, names: list[str], max_datasets: int) -> pd.DataFrame:
    df = scan[scan["study_family"].astype(str).isin(names)].copy()
    df = df[df["local_path"].map(lambda p: Path(str(p)).exists())]
    if "scan_status" in df:
        df = df[df["scan_status"].astype(str) == "ok"]
    if "has_control_like" in df:
        df = df[df["has_control_like"].astype(str).str.lower() == "true"]
    df["priority"] = df["study_family"].map({k: i for i, k in enumerate(names)}).fillna(999)
    return df.sort_values(["priority", "n_obs"]).drop_duplicates("study_family").head(max_datasets)


def selected_splits(tasks: list[dict], per_type: int) -> list[dict]:
    raw = feasible_splits(tasks)
    if not raw:
        return []
    df = pd.DataFrame(raw)
    rows: list[dict] = []
    for split_type in ["leave_context", "heldout_perturbation"]:
        sub = df[df["split_type"] == split_type].sort_values(["n_test", "n_train"], ascending=False)
        rows.extend(sub.head(per_type).to_dict("records"))
    return rows


def source_effect(task: dict, tasks: list[dict], train_mask: np.ndarray) -> np.ndarray:
    vals = [
        t["effect"]
        for t, keep in zip(tasks, train_mask)
        if keep and t["perturbation"] == task["perturbation"] and t["context"] != task["context"]
    ]
    if vals:
        return np.mean(vals, axis=0)
    vals = [t["effect"] for t, keep in zip(tasks, train_mask) if keep and t["perturbation"] == task["perturbation"]]
    if vals:
        return np.mean(vals, axis=0)
    return np.mean([t["effect"] for t, keep in zip(tasks, train_mask) if keep], axis=0)


def build_features(
    tasks: list[dict],
    indices: np.ndarray,
    train_mask: np.ndarray,
    baseline_pred: np.ndarray,
    hash_dim: int,
) -> np.ndarray:
    selected = [tasks[int(i)] for i in indices]
    src = np.stack([source_effect(t, tasks, train_mask) for t in selected], axis=0)
    ctrl = np.stack([t["control_mean"] for t in selected], axis=0)
    pert_h = stable_hash_features([t["perturbation"] for t in selected], hash_dim)
    ctx_h = stable_hash_features([t["context"] for t in selected], hash_dim)
    return np.concatenate([src, ctrl, baseline_pred, src - baseline_pred, pert_h, ctx_h], axis=1).astype(np.float32)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(weight * (pred - target) ** 2)


def deep_effect_loss(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    base_effect: torch.Tensor,
    target_effect: torch.Tensor,
    weight: torch.Tensor,
    args,
) -> torch.Tensor:
    pred_effect = base_effect + pred_residual
    mse_loss = weighted_mse(pred_effect, target_effect, weight)
    residual_loss = torch.mean((pred_residual - target_residual) ** 2)
    cosine_loss = 1.0 - torch.nn.functional.cosine_similarity(pred_effect, target_effect, dim=1).mean()
    pred_rank = torch.softmax(torch.abs(pred_effect) / args.rank_temperature, dim=1)
    target_rank = torch.softmax(torch.abs(target_effect) / args.rank_temperature, dim=1)
    rank_loss = torch.mean((pred_rank - target_rank) ** 2) * pred_effect.shape[1]
    sign_target = torch.sign(target_effect)
    sign_mask = (torch.abs(target_effect) >= torch.quantile(torch.abs(target_effect), 0.90, dim=1, keepdim=True)).float()
    sign_loss = torch.mean(sign_mask * torch.relu(args.sign_margin - sign_target * pred_effect))
    return (
        mse_loss
        + args.residual_loss_weight * residual_loss
        + args.cosine_loss_weight * cosine_loss
        + args.rank_loss_weight * rank_loss
        + args.sign_loss_weight * sign_loss
    )


def row_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def mean_effect_objective(y_true: np.ndarray, y_pred: np.ndarray, eval_bank: ProgramBank) -> float:
    true_z = eval_bank.transform(y_true)
    pred_z = eval_bank.transform(y_pred)
    rows = [effect_metrics(y_true[i], y_pred[i], true_z[i], pred_z[i]) for i in range(len(y_true))]
    vals = []
    rmse_vals = []
    for row in rows:
        rmse_vals.append(row["rmse"])
        vals.append(
            0.16 * np.nan_to_num(row["pearson"], nan=0.0)
            + 0.12 * np.nan_to_num(row["spearman"], nan=0.0)
            + 0.24 * np.nan_to_num(row["top20_overlap"], nan=0.0)
            + 0.24 * np.nan_to_num(row["deg_precision_top50"], nan=0.0)
            + 0.24 * np.nan_to_num(row["program_shift_consistency"], nan=0.0)
        )
    rmse_scale = float(np.median(rmse_vals) + 1e-6)
    penalty = 0.10 * float(np.mean(rmse_vals)) / rmse_scale
    return float(np.mean(vals) - penalty)


def confidence_from_mc(mc: np.ndarray) -> np.ndarray:
    unc = np.sqrt(np.mean(np.var(mc, axis=0), axis=1))
    scale = float(np.median(unc) + 1e-6)
    return np.exp(-unc / scale)


def safe_score(confidence: np.ndarray, residual: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    residual_norm = np.linalg.norm(residual, axis=1)
    baseline_norm = np.linalg.norm(baseline, axis=1) + 1e-6
    return confidence / (1.0 + residual_norm / baseline_norm)


def choose_calibrated_gate(
    base_val: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    score_val: np.ndarray,
    eval_bank: ProgramBank,
    min_gain: float,
) -> tuple[float, float, bool, float]:
    base_obj = mean_effect_objective(y_val, base_val, eval_bank)
    best = (-float("inf"), float(np.max(score_val) + 1e-6), False)
    thresholds = sorted(set([float(np.min(score_val) - 1e-6), float(np.max(score_val) + 1e-6), *np.quantile(score_val, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()]))
    for threshold in thresholds:
        use_transport = score_val >= threshold
        candidate = base_val.copy()
        candidate[use_transport] = pred_val[use_transport]
        obj = mean_effect_objective(y_val, candidate, eval_bank)
        if obj > best[0]:
            best = (obj, threshold, bool(np.any(use_transport)))
    # If validation says residual transport is not better, be deliberately boring:
    # fall back to the strong baseline rather than forcing a flashy bad transport.
    if best[0] < base_obj + min_gain:
        return float(np.max(score_val) + 1e-6), base_obj, False, 0.0
    return best[1], best[0], best[2], float(best[0] - base_obj)


def choose_blend_by_effect_objective(
    base_val: np.ndarray,
    residual_val: np.ndarray,
    y_val: np.ndarray,
    blends: list[float],
    default_blend: float,
    eval_bank: ProgramBank,
) -> tuple[float, float]:
    if len(y_val) < 2:
        return float(default_blend), float("nan")
    best_blend = float(default_blend)
    best_obj = -float("inf")
    for blend in blends:
        pred = base_val + float(blend) * residual_val
        obj = mean_effect_objective(y_val, pred, eval_bank)
        if obj > best_obj:
            best_obj = obj
            best_blend = float(blend)
    return best_blend, best_obj


def split_inner_validation(train_idx: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 104729)
    order = rng.permutation(np.asarray(train_idx, dtype=int))
    n_val = max(2, int(round(len(order) * val_fraction))) if len(order) >= 8 else 0
    if n_val <= 0 or len(order) - n_val < 4:
        return order, np.array([], dtype=int)
    return order[n_val:], order[:n_val]


def choose_v2_effect_blend(
    tasks: list[dict],
    train_idx: np.ndarray,
    seed: int,
    args,
    eval_bank: ProgramBank,
) -> tuple[float, float, bool]:
    fit_idx, val_idx = split_inner_validation(train_idx, seed, args.val_fraction)
    if len(val_idx) < 2:
        return 0.0, 0.0, False
    inner_mask = np.zeros(len(tasks), dtype=bool)
    inner_mask[fit_idx] = True
    y_val = np.stack([tasks[int(i)]["effect"] for i in val_idx], axis=0)
    v0_inner = V0StrongBaseline().fit(tasks, inner_mask)
    v2_inner = V2GraphPriorTransport(
        ProgramBank(args.n_programs, seed, mode=args.inner_bank_mode),
        alpha=5.0,
        blend=args.v2_blend,
    ).fit(tasks, inner_mask)
    v0_val = v0_inner.predict(tasks, val_idx)
    v2_val = v2_inner.predict(tasks, val_idx)
    base_obj = mean_effect_objective(y_val, v0_val, eval_bank)
    best_blend = 0.0
    best_obj = base_obj
    for blend in [float(x) for x in args.v2_expert_blends.split(",") if x.strip()]:
        pred = (1.0 - blend) * v0_val + blend * v2_val
        obj = mean_effect_objective(y_val, pred, eval_bank)
        if obj > best_obj:
            best_obj = obj
            best_blend = float(blend)
    has_gain = best_obj >= base_obj + args.expert_min_gain
    if not has_gain and args.expert_safe_fallback:
        best_blend = 0.0
    return best_blend, float(best_obj - base_obj), bool(has_gain)


def scale_to_reference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cand_norm = np.linalg.norm(candidate, axis=1, keepdims=True) + 1e-6
    ref_norm = np.linalg.norm(reference, axis=1, keepdims=True) + 1e-6
    scale = np.clip(ref_norm / cand_norm, 0.15, 6.0)
    return (candidate * scale).astype(np.float32)


def top_rank_graft(
    base_pred: np.ndarray,
    expert_pred: np.ndarray,
    blend: float,
    top_k: int,
    background_blend: float,
) -> np.ndarray:
    expert_scaled = scale_to_reference(expert_pred, base_pred)
    pred = ((1.0 - background_blend) * base_pred + background_blend * expert_scaled).astype(np.float32)
    k = min(int(top_k), pred.shape[1])
    if k <= 0:
        return pred
    for i in range(pred.shape[0]):
        idx = np.argsort(-np.abs(expert_scaled[i]))[:k]
        pred[i, idx] = (1.0 - blend) * base_pred[i, idx] + blend * expert_scaled[i, idx]
    return pred.astype(np.float32)


def choose_top_rank_graft(
    tasks: list[dict],
    train_idx: np.ndarray,
    seed: int,
    args,
    eval_bank: ProgramBank,
) -> tuple[float, int, float, bool]:
    fit_idx, val_idx = split_inner_validation(train_idx, seed + 17, args.val_fraction)
    if len(val_idx) < 2:
        return 0.0, 20, 0.0, False
    inner_mask = np.zeros(len(tasks), dtype=bool)
    inner_mask[fit_idx] = True
    y_val = np.stack([tasks[int(i)]["effect"] for i in val_idx], axis=0)
    v0_inner = V0StrongBaseline().fit(tasks, inner_mask)
    v2_inner = V2GraphPriorTransport(
        ProgramBank(args.n_programs, seed, mode=args.inner_bank_mode),
        alpha=5.0,
        blend=args.v2_blend,
    ).fit(tasks, inner_mask)
    v0_val = v0_inner.predict(tasks, val_idx)
    v2_val = v2_inner.predict(tasks, val_idx)
    base_obj = mean_effect_objective(y_val, v0_val, eval_bank)
    best_obj = base_obj
    best_blend = 0.0
    best_k = 20
    for top_k in [int(x) for x in args.graft_topk_grid.split(",") if x.strip()]:
        for blend in [float(x) for x in args.graft_blend_grid.split(",") if x.strip()]:
            pred = top_rank_graft(v0_val, v2_val, blend, top_k, args.graft_background_blend)
            obj = mean_effect_objective(y_val, pred, eval_bank)
            if obj > best_obj:
                best_obj = obj
                best_blend = float(blend)
                best_k = int(top_k)
    has_gain = best_obj >= base_obj + args.graft_min_gain
    if not has_gain and args.graft_safe_fallback:
        best_blend = 0.0
    return best_blend, best_k, float(best_obj - base_obj), bool(has_gain)


def train_deep_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_train: np.ndarray,
    x_test: np.ndarray,
    base_test: np.ndarray,
    seed: int,
    args,
    device: torch.device,
    eval_bank: ProgramBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n = len(x_train)
    order = rng.permutation(n)
    n_val = max(2, int(round(n * args.val_fraction))) if n >= 8 else 0
    val_idx = order[:n_val]
    fit_idx = order[n_val:] if n_val else order
    if len(fit_idx) < 2:
        fit_idx = order
        val_idx = np.array([], dtype=int)

    x_mean = x_train[fit_idx].mean(axis=0, keepdims=True)
    x_std = x_train[fit_idx].std(axis=0, keepdims=True) + 1e-5
    x_fit = (x_train[fit_idx] - x_mean) / x_std
    x_all = (x_train - x_mean) / x_std
    x_eval = (x_test - x_mean) / x_std
    residual_train = y_train - base_train
    y_fit = residual_train[fit_idx]
    target_fit = y_train[fit_idx]
    base_fit = base_train[fit_idx]

    abs_y = np.abs(y_train[fit_idx])
    cutoff = np.sort(abs_y, axis=1)[:, -min(args.deg_weight_k, abs_y.shape[1])][:, None]
    w_fit = 1.0 + args.deg_weight * (abs_y >= cutoff).astype(np.float32)

    model = ResidualMLP(x_fit.shape[1], y_fit.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.tensor(x_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
        torch.tensor(base_fit, dtype=torch.float32),
        torch.tensor(target_fit, dtype=torch.float32),
        torch.tensor(w_fit, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, drop_last=False)

    best_state = None
    best_val = float("inf")
    stale = 0
    val_x = torch.tensor(x_all[val_idx], dtype=torch.float32, device=device) if len(val_idx) else None
    val_y = torch.tensor(residual_train[val_idx], dtype=torch.float32, device=device) if len(val_idx) else None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb, bb, tb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bb = bb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            pred_residual = model(xb)
            loss = deep_effect_loss(pred_residual, yb, bb, tb, wb, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        if val_x is not None:
            model.eval()
            with torch.no_grad():
                val_loss = torch.mean((model(val_x) - val_y) ** 2).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_residual = model(torch.tensor(x_eval, dtype=torch.float32, device=device)).cpu().numpy()
        train_residual = model(torch.tensor(x_all, dtype=torch.float32, device=device)).cpu().numpy()

    blends = [float(x) for x in args.blends.split(",") if x.strip()]
    if len(val_idx):
        if args.blend_objective == "effect":
            best_blend, best_blend_objective = choose_blend_by_effect_objective(
                base_train[val_idx],
                train_residual[val_idx],
                y_train[val_idx],
                blends,
                args.default_blend,
                eval_bank,
            )
        else:
            best_blend = min(
                blends,
                key=lambda b: float(np.mean((base_train[val_idx] + b * train_residual[val_idx] - y_train[val_idx]) ** 2)),
            )
            best_blend_objective = float("nan")
    else:
        best_blend = args.default_blend
        best_blend_objective = float("nan")
    pred = base_test + best_blend * test_residual

    model.train()
    mc_preds = []
    val_mc_preds = []
    with torch.no_grad():
        xt = torch.tensor(x_eval, dtype=torch.float32, device=device)
        xv = torch.tensor(x_all[val_idx], dtype=torch.float32, device=device) if len(val_idx) else None
        for _ in range(max(2, args.mc_samples)):
            mc_preds.append((base_test + best_blend * model(xt).cpu().numpy()).astype(np.float32))
            if xv is not None:
                val_mc_preds.append((base_train[val_idx] + best_blend * model(xv).cpu().numpy()).astype(np.float32))
    mc = np.stack(mc_preds, axis=0)
    confidence = confidence_from_mc(mc)
    safe_pred = pred.copy()
    unsafe = confidence < args.safe_confidence_threshold
    safe_pred[unsafe] = base_test[unsafe]

    calibrated_pred = base_test.copy()
    calibrated_used = np.zeros(len(base_test), dtype=bool)
    gate_threshold = float("inf")
    gate_val_loss = float("nan")
    if len(val_idx) >= 2 and val_mc_preds:
        val_pred = base_train[val_idx] + best_blend * train_residual[val_idx]
        val_confidence = confidence_from_mc(np.stack(val_mc_preds, axis=0))
        val_score = safe_score(val_confidence, best_blend * train_residual[val_idx], base_train[val_idx])
        test_score = safe_score(confidence, best_blend * test_residual, base_test)
        gate_threshold, gate_val_loss, gate_has_transport, gate_gain = choose_calibrated_gate(
            base_train[val_idx],
            val_pred,
            y_train[val_idx],
            val_score,
            eval_bank,
            args.calibrated_min_gain,
        )
        if gate_has_transport:
            calibrated_used = test_score >= gate_threshold
            calibrated_pred[calibrated_used] = pred[calibrated_used]
    else:
        gate_gain = 0.0
    info = {
        "epochs_ran": int(epoch + 1),
        "best_val": float(best_val),
        "best_blend": float(best_blend),
        "best_blend_objective": float(best_blend_objective),
        "unsafe_rate": float(np.mean(unsafe)) if len(unsafe) else 0.0,
        "calibrated_transport_rate": float(np.mean(calibrated_used)) if len(calibrated_used) else 0.0,
        "calibrated_gate_threshold": float(gate_threshold),
        "calibrated_gate_val_loss": float(gate_val_loss),
        "calibrated_gate_gain": float(gate_gain),
    }
    return pred.astype(np.float32), safe_pred.astype(np.float32), calibrated_pred.astype(np.float32), {"confidence": confidence, "unsafe": unsafe, **info}


def metric_rows(tasks: list[dict], test_idx: np.ndarray, pred: np.ndarray, eval_bank: ProgramBank, meta: dict) -> list[dict]:
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    tz = eval_bank.transform(y)
    pz = eval_bank.transform(pred)
    rows = []
    for i in range(len(test_idx)):
        row = effect_metrics(y[i], pred[i], tz[i], pz[i])
        row.update(meta)
        row["task_id"] = int(i)
        rows.append(row)
    return rows


def run_dataset(row: pd.Series, phase: str, seeds: list[int], args, out: Path, device: torch.device) -> tuple[list[dict], list[dict]]:
    dataset = str(row["study_family"])
    result_rows: list[dict] = []
    audit_rows: list[dict] = []
    for seed in seeds:
        try:
            print(f"[gpu_deep] dataset={dataset} seed={seed} building tasks", flush=True)
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=args.n_genes, seed=seed)
            splits = selected_splits(tasks, args.split_per_type)
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, **meta, "n_splits": len(splits), "status": "ok"})
            print(f"[gpu_deep] dataset={dataset} seed={seed} n_tasks={len(tasks)} splits={len(splits)}", flush=True)
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(train_idx) < 6 or len(test_idx) < 2:
                    print(
                        f"[gpu_deep] skip dataset={dataset} seed={seed} split={split['split_type']} heldout={split['heldout']} train={len(train_idx)} test={len(test_idx)}",
                        flush=True,
                    )
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[train_idx] = True
                y_train = np.stack([tasks[int(i)]["effect"] for i in train_idx], axis=0)
                y_test = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
                eval_bank = ProgramBank(args.n_programs, seed, mode=args.eval_bank).fit(y_train)

                v0 = V0StrongBaseline().fit(tasks, train_mask)
                v2 = V2GraphPriorTransport(ProgramBank(args.n_programs, seed, mode=args.eval_bank), alpha=5.0, blend=args.v2_blend).fit(tasks, train_mask)
                base_train = v0.predict(tasks, train_idx)
                base_test = v0.predict(tasks, test_idx)
                v2_test = v2.predict(tasks, test_idx)
                v2_effect_blend, v2_effect_gain, v2_effect_has_gain = choose_v2_effect_blend(
                    tasks,
                    train_idx,
                    seed,
                    args,
                    eval_bank,
                )
                effect_blend_v2 = ((1.0 - v2_effect_blend) * base_test + v2_effect_blend * v2_test).astype(np.float32)
                graft_blend, graft_top_k, graft_gain, graft_has_gain = choose_top_rank_graft(
                    tasks,
                    train_idx,
                    seed,
                    args,
                    eval_bank,
                )
                top_rank_graft_v2 = top_rank_graft(
                    base_test,
                    v2_test,
                    graft_blend,
                    graft_top_k,
                    args.graft_background_blend,
                )
                x_train = build_features(tasks, train_idx, train_mask, base_train, args.hash_dim)
                x_test = build_features(tasks, test_idx, train_mask, base_test, args.hash_dim)
                print(
                    f"[gpu_deep] train dataset={dataset} seed={seed} split={split['split_type']} heldout={split['heldout']} "
                    f"train={len(train_idx)} test={len(test_idx)} device={device}",
                    flush=True,
                )

                deep_pred, safe_pred, calibrated_pred, safe_info = train_deep_model(
                    x_train, y_train, base_train, x_test, base_test, seed, args, device, eval_bank
                )
                models = {
                    "V0": base_test,
                    "V2": v2_test,
                    "EffectBlendV2": effect_blend_v2,
                    "TopRankGraftV2": top_rank_graft_v2,
                    "DeepResidualTransport": deep_pred,
                    "DeepSafeTransport": safe_pred,
                    "DeepCalibratedSafeTransport": calibrated_pred,
                }
                base_meta = {
                    "phase": phase,
                    "dataset": dataset,
                    "split_type": split["split_type"],
                    "heldout": split["heldout"],
                    "seed": seed,
                    "n_train": int(len(train_idx)),
                    "n_tasks": int(len(test_idx)),
                    "device": str(device),
                    "v2_effect_blend": float(v2_effect_blend),
                    "v2_effect_gain": float(v2_effect_gain),
                    "v2_effect_has_gain": int(v2_effect_has_gain),
                    "graft_blend": float(graft_blend),
                    "graft_top_k": int(graft_top_k),
                    "graft_gain": float(graft_gain),
                    "graft_has_gain": int(graft_has_gain),
                    "deep_best_blend": safe_info["best_blend"],
                    "deep_best_blend_objective": safe_info["best_blend_objective"],
                    "deep_unsafe_rate": safe_info["unsafe_rate"],
                    "deep_calibrated_transport_rate": safe_info["calibrated_transport_rate"],
                    "deep_calibrated_gate_threshold": safe_info["calibrated_gate_threshold"],
                    "deep_calibrated_gate_val_loss": safe_info["calibrated_gate_val_loss"],
                    "deep_calibrated_gate_gain": safe_info["calibrated_gate_gain"],
                    "deep_epochs": safe_info["epochs_ran"],
                }
                for model_name, pred in models.items():
                    result_rows.extend(metric_rows(tasks, test_idx, pred, eval_bank, {**base_meta, "model": model_name}))
                pd.DataFrame(result_rows).to_csv(out / "GPU_DEEP_TASK_METRICS_INCREMENTAL.csv", index=False)
                print(
                    f"[gpu_deep] done dataset={dataset} seed={seed} split={split['split_type']} heldout={split['heldout']} rows={len(result_rows)}",
                    flush=True,
                )
        except Exception as exc:
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
            pd.DataFrame(audit_rows).to_csv(out / "GPU_DEEP_AUDIT_INCREMENTAL.csv", index=False)
            print(f"[gpu_deep] failed dataset={dataset} seed={seed}: {exc!r}", flush=True)
    return result_rows, audit_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--studies", default=",".join(MAIN_STUDIES))
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--max-datasets", type=int, default=3)
    p.add_argument("--n-genes", type=int, default=2000)
    p.add_argument("--n-programs", type=int, default=96)
    p.add_argument("--split-per-type", type=int, default=2)
    p.add_argument("--eval-bank", default="pca_nmf_hvg")
    p.add_argument("--inner-bank-mode", default="pca")
    p.add_argument("--hash-dim", type=int, default=96)
    p.add_argument("--v2-blend", type=float, default=0.18)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=3.0)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--deg-weight-k", type=int, default=50)
    p.add_argument("--deg-weight", type=float, default=3.0)
    p.add_argument("--residual-loss-weight", type=float, default=0.20)
    p.add_argument("--cosine-loss-weight", type=float, default=0.35)
    p.add_argument("--rank-loss-weight", type=float, default=0.45)
    p.add_argument("--sign-loss-weight", type=float, default=0.05)
    p.add_argument("--rank-temperature", type=float, default=0.08)
    p.add_argument("--sign-margin", type=float, default=0.002)
    p.add_argument("--blend-objective", choices=["mse", "effect"], default="effect")
    p.add_argument("--calibrated-min-gain", type=float, default=0.002)
    p.add_argument("--v2-expert-blends", default="0,0.05,0.1,0.18,0.25,0.35,0.5,0.75,1.0")
    p.add_argument("--expert-min-gain", type=float, default=0.001)
    p.add_argument("--expert-safe-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--graft-blend-grid", default="0,0.1,0.18,0.25,0.35,0.5,0.75")
    p.add_argument("--graft-topk-grid", default="20,35,50,80,120")
    p.add_argument("--graft-background-blend", type=float, default=0.02)
    p.add_argument("--graft-min-gain", type=float, default=0.001)
    p.add_argument("--graft-safe-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--blends", default="0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--default-blend", type=float, default=0.35)
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--safe-confidence-threshold", type=float, default=0.25)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment. Use scgpt_env.")
    device = torch.device("cuda:0")
    torch.manual_seed(123)
    root = Path(args.root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    studies = [x.strip() for x in args.studies.split(",") if x.strip()]
    selected = pick_datasets(scan, studies, args.max_datasets)
    selected.to_csv(out / "GPU_DEEP_SELECTED_DATASETS.csv", index=False)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    print(
        f"[gpu_deep] device={torch.cuda.get_device_name(0)} selected={selected['study_family'].astype(str).tolist()} seeds={seeds}",
        flush=True,
    )

    rows: list[dict] = []
    audits: list[dict] = []
    for _, ds in selected.iterrows():
        print(f"[gpu_deep] starting dataset={ds['study_family']} path={ds['local_path']}", flush=True)
        r, a = run_dataset(ds, "gpu_deep", seeds, args, out, device)
        rows.extend(r)
        audits.extend(a)
        pd.DataFrame(rows).to_csv(out / "GPU_DEEP_TASK_METRICS.csv", index=False)
        pd.DataFrame(audits).to_csv(out / "GPU_DEEP_AUDIT.csv", index=False)
        if rows:
            summarize_results(pd.DataFrame(rows)).to_csv(out / "GPU_DEEP_SUMMARY.csv", index=False)
    status = {
        "device": torch.cuda.get_device_name(0),
        "n_rows": len(rows),
        "datasets": selected["study_family"].astype(str).tolist(),
        "seeds": seeds,
    }
    (out / "GPU_DEEP_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

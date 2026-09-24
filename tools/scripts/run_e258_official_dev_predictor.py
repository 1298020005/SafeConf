#!/usr/bin/env python3
"""E258 official-LFC engineering predictor; NOT the formal raw-count experiment.

Two fixed residual MLP variants use separate GPUs. Only TRAIN/VALIDATION rows
from E258_OFFICIAL_DEV_VIEW.npz are loaded. Test donor effects are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_e258_official_dev_view import TRAIN, VALIDATION


VARIANTS = {
    "small": (64,),
    "medium": (256, 128),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_view(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        view = {key: data[key].copy() for key in data.files}
    if len(view["effect"]) != len(view["task_target"]) or \
       set(view["task_donor"].tolist()) != set(TRAIN + VALIDATION):
        raise ValueError("train/validation view contract failed")
    return view


def baselines(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = view["effect"].astype(np.float32)
    donors = view["task_donor"].astype(str)
    targets = view["task_target"].astype(str)
    lines = view["task_line"].astype(str)
    control = view["control"].astype(np.float32)
    line_ids = view["line"].astype(str)
    line_index = {line: i for i, line in enumerate(line_ids)}
    donor_control = {donor: np.mean(control[[i for i, line in enumerate(line_ids)
                                              if line.split("_")[0] == donor]], axis=0)
                     for donor in TRAIN}
    donor_effect: dict[tuple[str, str], np.ndarray] = {}
    for target in np.unique(targets[ np.isin(donors, TRAIN) ]):
        for donor in TRAIN:
            mask = (targets == target) & (donors == donor)
            if mask.any():
                donor_effect[(target, donor)] = y[mask].mean(axis=0)
    mean = np.empty_like(y)
    similar = np.empty_like(y)
    context_delta = np.empty_like(y)
    for i, (target, donor, line) in enumerate(zip(targets, donors, lines)):
        sources = [s for s in TRAIN if (target, s) in donor_effect and s != donor]
        if not sources:
            raise ValueError(f"no independent source for {line}:{target}")
        source_effect = np.stack([donor_effect[(target, s)] for s in sources])
        mean[i] = source_effect.mean(axis=0)
        source_controls = np.stack([donor_control[s] for s in sources])
        query = control[line_index[line]]
        source_centered = source_controls - source_controls.mean(axis=1, keepdims=True)
        query_centered = query - query.mean()
        den = np.linalg.norm(source_centered, axis=1) * np.linalg.norm(query_centered)
        corr = (source_centered @ query_centered) / np.maximum(den, 1e-12)
        weights = np.exp(20.0 * (corr - corr.max()))
        weights /= weights.sum()
        similar[i] = weights @ source_effect
        context_delta[i] = query - source_controls.mean(axis=0)
    return mean, similar, context_delta


def gene_mask(view: dict) -> np.ndarray:
    genes = view["gene"].astype(str)
    targets = view["task_target"].astype(str)
    return genes[None, :] != targets[:, None]


def line_mse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray,
             lines: np.ndarray) -> dict[str, float]:
    task_mse = (((pred - truth) ** 2) * mask).sum(axis=1) / mask.sum(axis=1)
    return {line: float(task_mse[lines == line].mean()) for line in sorted(set(lines.tolist()))}


class ResidualMLP(nn.Module):
    def __init__(self, genes: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        dim = 2 * genes
        for width in hidden:
            layers.extend((nn.Linear(dim, width), nn.GELU()))
            dim = width
        final = nn.Linear(dim, genes)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.network(x)


def train(view_path: Path, output_dir: Path, variant: str, device_name: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unregistered model variant")
    seed = 25801 if variant == "small" else 25802
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; do not silently substitute CPU")
    view = load_view(view_path)
    y = view["effect"].astype(np.float32)
    base, similar, delta = baselines(view)
    mask = gene_mask(view)
    donor = view["task_donor"].astype(str)
    lines = view["task_line"].astype(str)
    train_idx = np.flatnonzero(np.isin(donor, TRAIN))
    val_idx = np.flatnonzero(np.isin(donor, VALIDATION))
    if not 1527 <= len(val_idx) <= 1530:
        raise ValueError("more than three validation tasks missing from official engineering summary")
    scale = np.maximum(np.std(np.concatenate((base[train_idx], delta[train_idx]), axis=1), axis=0), 1e-3)
    x = np.concatenate((base, delta), axis=1) / scale
    x_train = torch.as_tensor(x[train_idx], device=device)
    y_train = torch.as_tensor(y[train_idx], device=device)
    b_train = torch.as_tensor(base[train_idx], device=device)
    m_train = torch.as_tensor(mask[train_idx], device=device)
    x_val = torch.as_tensor(x[val_idx], device=device)
    b_val = torch.as_tensor(base[val_idx], device=device)
    model = ResidualMLP(y.shape[1], VARIANTS[variant]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    best_error = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 121):
        model.train()
        order = torch.randperm(len(train_idx), device=device)
        for ids in order.split(128):
            pred = model(x_train[ids], b_train[ids])
            loss = (((pred - y_train[ids]) ** 2) * m_train[ids]).sum() / m_train[ids].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = torch.cat([model(xx, bb) for xx, bb in zip(x_val.split(128),
                                                                      b_val.split(128))]).cpu().numpy()
        by_line = line_mse(prediction, y[val_idx], mask[val_idx], lines[val_idx])
        macro = float(np.mean(list(by_line.values())))
        history.append({"epoch": epoch, "validation_macro_mse": macro})
        if macro < best_error - 1e-9:
            best_error, best_epoch, stale = macro, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{variant} epoch={epoch} val_macro_mse={macro:.9g}", flush=True)
        if stale >= 15:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        prediction = torch.cat([model(xx, bb) for xx, bb in zip(x_val.split(128),
                                                                  b_val.split(128))]).cpu().numpy()
    truth = y[val_idx]
    val_mask = mask[val_idx]
    val_lines = lines[val_idx]
    comparisons = {
        "no_change": line_mse(np.zeros_like(truth), truth, val_mask, val_lines),
        "train_source_mean": line_mse(base[val_idx], truth, val_mask, val_lines),
        "control_similarity": line_mse(similar[val_idx], truth, val_mask, val_lines),
        f"residual_mlp_{variant}": line_mse(prediction, truth, val_mask, val_lines),
    }
    macro = {name: float(np.mean(list(by_line.values()))) for name, by_line in comparisons.items()}
    simple = min(("no_change", "train_source_mean", "control_similarity"), key=macro.get)
    model_name = f"residual_mlp_{variant}"
    passed = (macro[model_name] < 0.98 * macro[simple]
              and sum(comparisons[model_name][line] < comparisons[simple][line]
                      for line in comparisons[simple]) >= 3
              and macro[model_name] < macro["no_change"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / f"E258_OFFICIAL_DEV_{variant.upper()}_VAL_PRED.npz"
    checkpoint_file = output_dir / f"E258_OFFICIAL_DEV_{variant.upper()}_CHECKPOINT.pt"
    if prediction_file.exists() or checkpoint_file.exists():
        raise FileExistsError("existing model artifact")
    np.savez_compressed(prediction_file, line=val_lines, target=view["task_target"][val_idx],
                        prediction=prediction)
    torch.save({"model": best_state, "variant": variant, "epoch": best_epoch,
                "view_sha256": sha256(view_path)}, checkpoint_file)
    result = {
        "stage": "E258_OFFICIAL_LFC_ENGINEERING_ONLY",
        "variant": variant,
        "device": str(device),
        "best_epoch": best_epoch,
        "n_validation_tasks": len(val_idx),
        "comparison_macro_mse": macro,
        "comparison_by_line_mse": comparisons,
        "strong_simple_baseline": simple,
        "strict_upstream_gate_passed": bool(passed),
        "test_donor_effect_values_loaded": 0,
        "formal_raw_count_result": False,
        "view_sha256": sha256(view_path),
        "prediction_sha256": sha256(prediction_file),
        "checkpoint_sha256": sha256(checkpoint_file),
        "training_history": history,
    }
    status_file = output_dir / f"E258_OFFICIAL_DEV_{variant.upper()}_STATUS.json"
    status_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "training_history"},
                     ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/official_dev_models"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    train(args.view, args.output_dir, args.variant, args.device)

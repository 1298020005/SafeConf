from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from encoders import stable_hash_features
from evaluators import effect_metrics, summarize_results, compare_model_vs_v0
from program_bank import ProgramBank
from transport_models import V0StrongBaseline, V2GraphPriorTransport


MAIN_STUDIES = ["Haber", "Parekh", "KaggleCrossCell", "kangCrossCell", "kangCrossPatient", "Wessels"]
EXTERNAL_STUDIES = ["KaggleCrossPatient", "McFarland", "crossPatient", "TCDD"]


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
    df = scan[scan["study_family"].isin(names)].copy()
    df = df[df["local_path"].map(lambda p: Path(str(p)).exists())]
    df = df[df["has_control_like"].astype(str).str.lower() == "true"]
    df["priority"] = df["study_family"].map({k: i for i, k in enumerate(names)}).fillna(999)
    return df.sort_values(["priority", "n_obs"]).drop_duplicates("study_family").head(max_datasets)


def selected_splits(tasks: list[dict], per_type: int) -> list[dict]:
    raw = feasible_splits(tasks)
    if not raw:
        return []
    df = pd.DataFrame(raw)
    rows = []
    for split_type in ["leave_context", "heldout_perturbation"]:
        rows.extend(df[df["split_type"] == split_type].sort_values(["n_test", "n_train"], ascending=False).head(per_type).to_dict("records"))
    return rows


def source_effect(task: dict, tasks: list[dict], train_mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
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
    return fallback


def make_features(tasks: list[dict], indices: np.ndarray, train_mask: np.ndarray, baseline: V0StrongBaseline, v2: V2GraphPriorTransport, hash_dim: int) -> tuple[np.ndarray, np.ndarray]:
    selected = [tasks[int(i)] for i in indices]
    fallback = np.mean([t["effect"] for t, keep in zip(tasks, train_mask) if keep], axis=0)
    src = np.stack([source_effect(t, tasks, train_mask, fallback) for t in selected], axis=0)
    ctrl = np.stack([t["control_mean"] for t in selected], axis=0)
    base = baseline.predict(tasks, indices)
    trans = v2.predict(tasks, indices)
    pert_h = stable_hash_features([t["perturbation"] for t in selected], hash_dim)
    ctx_h = stable_hash_features([t["context"] for t in selected], hash_dim)
    y = np.stack([t["effect"] for t in selected], axis=0)
    x = np.concatenate([base, trans, src, ctrl, pert_h, ctx_h], axis=1).astype(np.float32)
    return x, y.astype(np.float32)


def train_predict(
    tasks: list[dict],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    device: str,
    hidden: int,
    epochs: int,
    lr: float,
    dropout: float,
    hash_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask = np.zeros(len(tasks), dtype=bool)
    train_mask[train_idx] = True
    baseline = V0StrongBaseline().fit(tasks, train_mask)
    v2 = V2GraphPriorTransport(ProgramBank(128, seed, mode="pca_nmf_hvg"), alpha=5.0, blend=0.12).fit(tasks, train_mask)
    x_train, y_train = make_features(tasks, train_idx, train_mask, baseline, v2, hash_dim)
    x_test, y_test = make_features(tasks, test_idx, train_mask, baseline, v2, hash_dim)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-4] = 1.0
    xtr = torch.tensor((x_train - x_mean) / x_std, device=device)
    ytr = torch.tensor((y_train - y_mean) / y_std, device=device)
    xte = torch.tensor((x_test - x_mean) / x_std, device=device)
    model = ResidualMLP(xtr.shape[1], ytr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        pred = model(xtr)
        mse = F.mse_loss(pred, ytr)
        # DEG/top-k proxy: upweight large true effects.
        weight = 1.0 + 2.0 * torch.sigmoid(3.0 * (torch.abs(ytr) - torch.abs(ytr).median()))
        weighted = torch.mean(weight * (pred - ytr) ** 2)
        loss = 0.6 * mse + 0.4 * weighted
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        val = float(loss.detach().cpu())
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xte).detach().cpu().numpy() * y_std + y_mean
    base = baseline.predict(tasks, test_idx)
    v2_pred = v2.predict(tasks, test_idx)
    # Conservative residual blend over the strong baseline.
    final = (0.75 * base + 0.25 * pred).astype(np.float32)
    return final, base, v2_pred.astype(np.float32), y_test


def run_dataset(row: pd.Series, phase: str, seeds: list[int], args, out: Path) -> list[dict]:
    dataset = str(row["study_family"])
    rows: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=args.n_genes, seed=seed)
            splits = selected_splits(tasks, args.split_per_type)
            pd.DataFrame(splits).to_csv(out / f"{phase}_{dataset}_seed{seed}_splits.csv", index=False)
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(train_idx) < 4 or len(test_idx) < 2:
                    continue
                pred, base, v2_pred, y = train_predict(tasks, train_idx, test_idx, seed, args.device, args.hidden, args.epochs, args.lr, args.dropout, args.hash_dim)
                bank = ProgramBank(args.n_programs, seed, mode="pca_nmf_hvg").fit(np.stack([tasks[int(i)]["effect"] for i in train_idx]))
                for model_name, arr in [("V0", base), ("V2", v2_pred), ("DeepSafeTransGPU", pred)]:
                    tz = bank.transform(y)
                    pz = bank.transform(arr)
                    metric = pd.DataFrame([effect_metrics(y[i], arr[i], tz[i], pz[i]) for i in range(len(y))]).mean().to_dict()
                    metric.update({
                        "phase": phase,
                        "dataset": dataset,
                        "split_type": split["split_type"],
                        "heldout": split["heldout"],
                        "seed": seed,
                        "model": model_name,
                        "n_train": int(len(train_idx)),
                        "n_tasks": int(len(test_idx)),
                        "device": args.device,
                        "hidden": args.hidden,
                        "epochs": args.epochs,
                    })
                    rows.append(metric)
                    pd.DataFrame(rows).to_csv(out / f"{phase.upper()}_GPU_RESULTS_INCREMENTAL.csv", index=False)
        except Exception as exc:
            rows.append({"phase": phase, "dataset": dataset, "seed": seed, "model": "DeepSafeTransGPU", "status": "failed", "error": repr(exc)})
    return rows


def compare_model_vs(summary: pd.DataFrame, model_name: str, baseline_name: str) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for keys, sub in summary.groupby(["phase", "dataset", "split_type"], dropna=False):
        base = sub[sub["model"] == baseline_name]
        cur = sub[sub["model"] == model_name]
        if base.empty or cur.empty:
            continue
        b = base.iloc[0]
        v = cur.iloc[0]
        row = dict(zip(["phase", "dataset", "split_type"], keys))
        row["model"] = model_name
        row["baseline"] = baseline_name
        row.update(
            {
                "pearson_delta": v["pearson_mean"] - b["pearson_mean"],
                "spearman_delta": v["spearman_mean"] - b["spearman_mean"],
                "rmse_delta": v["rmse_mean"] - b["rmse_mean"],
                "top20_delta": v["top20_overlap_mean"] - b["top20_overlap_mean"],
                "deg_precision_delta": v["deg_precision_top50_mean"] - b["deg_precision_top50_mean"],
                "program_consistency_delta": v["program_shift_consistency_mean"] - b["program_shift_consistency_mean"],
            }
        )
        row["effect_positive_dims"] = int(
            (row["top20_delta"] > 0)
            + (row["deg_precision_delta"] > 0)
            + (row["program_consistency_delta"] > 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--external-seed-count", type=int, default=3)
    p.add_argument("--max-datasets", type=int, default=3)
    p.add_argument("--max-external-datasets", type=int, default=1)
    p.add_argument("--n-genes", type=int, default=2500)
    p.add_argument("--n-programs", type=int, default=128)
    p.add_argument("--split-per-type", type=int, default=2)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--hash-dim", type=int, default=128)
    p.add_argument("--main-studies", default=",".join(MAIN_STUDIES))
    p.add_argument("--external-studies", default=",".join(EXTERNAL_STUDIES))
    args = p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    out = Path(args.root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    main_names = [x.strip() for x in args.main_studies.split(",") if x.strip()]
    external_names = [x.strip() for x in args.external_studies.split(",") if x.strip()]
    main_ds = pick_datasets(scan, main_names, args.max_datasets)
    ext_ds = pick_datasets(scan, [x for x in external_names if x not in set(main_ds["study_family"].astype(str))], args.max_external_datasets)
    main_ds.to_csv(out / f"GPU_MAIN_SELECTED_{args.device.replace(':','')}.csv", index=False)
    ext_ds.to_csv(out / f"GPU_EXTERNAL_SELECTED_{args.device.replace(':','')}.csv", index=False)
    rows: list[dict] = []
    for _, row in main_ds.iterrows():
        rows.extend(run_dataset(row, "main", seeds, args, out))
        pd.DataFrame(rows).to_csv(out / f"GPU_ALL_RESULTS_{args.device.replace(':','')}.csv", index=False)
    for _, row in ext_ds.iterrows():
        rows.extend(run_dataset(row, "external", seeds[:args.external_seed_count], args, out))
        pd.DataFrame(rows).to_csv(out / f"GPU_ALL_RESULTS_{args.device.replace(':','')}.csv", index=False)
    df = pd.DataFrame(rows)
    summary = summarize_results(df)
    summary.to_csv(out / f"GPU_SUMMARY_{args.device.replace(':','')}.csv", index=False)
    delta = compare_model_vs_v0(summary, "DeepSafeTransGPU")
    delta_v2 = compare_model_vs(summary, "DeepSafeTransGPU", "V2")
    delta.to_csv(out / f"GPU_DEEPSAFE_VS_V0_{args.device.replace(':','')}.csv", index=False)
    delta_v2.to_csv(out / f"GPU_DEEPSAFE_VS_V2_{args.device.replace(':','')}.csv", index=False)
    status = {
        "device": args.device,
        "n_rows": int(len(df)),
        "main_datasets": main_ds["study_family"].astype(str).tolist(),
        "external_datasets": ext_ds["study_family"].astype(str).tolist(),
        "mean_top20_delta": float(delta["top20_delta"].mean()) if not delta.empty else None,
        "mean_deg_delta": float(delta["deg_precision_delta"].mean()) if not delta.empty else None,
        "mean_program_delta": float(delta["program_consistency_delta"].mean()) if not delta.empty else None,
        "mean_top20_delta_vs_v2": float(delta_v2["top20_delta"].mean()) if not delta_v2.empty else None,
        "mean_deg_delta_vs_v2": float(delta_v2["deg_precision_delta"].mean()) if not delta_v2.empty else None,
        "mean_program_delta_vs_v2": float(delta_v2["program_consistency_delta"].mean()) if not delta_v2.empty else None,
    }
    (out / f"GPU_STATUS_{args.device.replace(':','')}.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

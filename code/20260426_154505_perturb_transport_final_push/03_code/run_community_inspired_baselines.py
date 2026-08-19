from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from encoders import nearest_prior_smoothing, pathway_prior_features, stable_hash_features
from evaluators import effect_metrics, summarize_results
from network_modules import NetworkModuleBank
from program_bank import ProgramBank
from transport_models import V0StrongBaseline, V2GraphPriorTransport


MAIN_STUDIES = ["Haber", "Parekh", "KaggleCrossCell", "Wessels", "NormanWeissman2019", "DixitRegev2016"]
EXTERNAL_STUDIES = ["KaggleCrossPatient", "McFarland", "crossPatient", "TCDD", "TianKampmann2019", "PapalexiSatija2021"]


def _cosine(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    b = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
    return q @ b.T


class ScGenStyleLatentDelta:
    """scGen-inspired latent delta baseline.

    This is a lightweight comparator, not a full scGen reproduction.  It learns
    perturbation deltas in a low-dimensional program space and transports the
    nearest available delta to the target context.
    """

    name = "scGen_style_latent_delta"

    def __init__(self, n_programs: int, seed: int):
        self.bank = ProgramBank(n_programs, seed, mode="pca_nmf_hvg")

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "ScGenStyleLatentDelta":
        self.train_tasks = [t for t, keep in zip(tasks, train_mask) if keep]
        effects = np.stack([t["effect"] for t in self.train_tasks])
        self.bank.fit(effects)
        z = self.bank.transform(effects)
        self.global_z = z.mean(axis=0)
        self.by_pert = {}
        for task, row in zip(self.train_tasks, z):
            self.by_pert.setdefault(str(task["perturbation"]), []).append(row)
        self.by_pert = {k: np.mean(v, axis=0) for k, v in self.by_pert.items()}
        self.train_prior = pathway_prior_features(
            [t["perturbation"] for t in self.train_tasks],
            [t["context"] for t in self.train_tasks],
            dim=96,
        )
        self.train_z = z
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        pri = pathway_prior_features([t["perturbation"] for t in selected], [t["context"] for t in selected], dim=96)
        smooth = nearest_prior_smoothing(pri, self.train_prior, self.bank.inverse_transform(self.train_z), k=min(5, len(self.train_tasks)))
        smooth_z = self.bank.transform(smooth)
        rows = []
        for task, sz in zip(selected, smooth_z):
            rows.append(0.65 * self.by_pert.get(str(task["perturbation"]), self.global_z) + 0.35 * sz)
        return self.bank.inverse_transform(np.stack(rows))


class OTBarycentricProxy:
    """CellOT-inspired barycentric baseline over control-state neighborhoods."""

    name = "OT_barycentric_proxy"

    def __init__(self, k: int = 7):
        self.k = int(k)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "OTBarycentricProxy":
        self.train_tasks = [t for t, keep in zip(tasks, train_mask) if keep]
        self.controls = np.stack([t["control_mean"] for t in self.train_tasks])
        self.effects = np.stack([t["effect"] for t in self.train_tasks])
        self.prior = pathway_prior_features(
            [t["perturbation"] for t in self.train_tasks],
            [t["context"] for t in self.train_tasks],
            dim=96,
        )
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        controls = np.stack([t["control_mean"] for t in selected])
        prior = pathway_prior_features([t["perturbation"] for t in selected], [t["context"] for t in selected], dim=96)
        control_sim = _cosine(controls, self.controls)
        prior_sim = _cosine(prior, self.prior)
        sim = 0.55 * control_sim + 0.45 * prior_sim
        kk = min(self.k, sim.shape[1])
        idx = np.argsort(-sim, axis=1)[:, :kk]
        rows = []
        for r, ids in enumerate(idx):
            w = np.maximum(sim[r, ids] - np.min(sim[r, ids]), 0.0) + 1e-3
            w = w / w.sum()
            rows.append((self.effects[ids] * w[:, None]).sum(axis=0))
        return np.asarray(rows, dtype=np.float32)


class GraphKernelProxy:
    """GEARS-style graph/prior smoothing baseline."""

    name = "graph_kernel_proxy"

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "GraphKernelProxy":
        self.train_tasks = [t for t, keep in zip(tasks, train_mask) if keep]
        self.effects = np.stack([t["effect"] for t in self.train_tasks])
        self.prior = pathway_prior_features(
            [t["perturbation"] for t in self.train_tasks],
            [t["context"] for t in self.train_tasks],
            dim=128,
        )
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        prior = pathway_prior_features([t["perturbation"] for t in selected], [t["context"] for t in selected], dim=128)
        return nearest_prior_smoothing(prior, self.prior, self.effects, k=min(7, len(self.train_tasks)))


class ContextPerturbRidgeProxy:
    """CPA-style conditional ridge baseline using perturbation/context tokens."""

    name = "CPA_style_conditional_ridge"

    def __init__(self, alpha: float = 5.0, hash_dim: int = 128):
        self.alpha = float(alpha)
        self.hash_dim = int(hash_dim)

    def _features(self, selected: list[dict]) -> np.ndarray:
        pert = stable_hash_features([t["perturbation"] for t in selected], self.hash_dim)
        ctx = stable_hash_features([t["context"] for t in selected], self.hash_dim)
        ctrl = np.stack([t["control_mean"] for t in selected])
        ctrl_low = ctrl[:, : min(256, ctrl.shape[1])]
        return np.concatenate([pert, ctx, pert * ctx, ctrl_low], axis=1)

    def fit(self, tasks: list[dict], train_mask: np.ndarray) -> "ContextPerturbRidgeProxy":
        self.train_tasks = [t for t, keep in zip(tasks, train_mask) if keep]
        x = self._features(self.train_tasks)
        y = np.stack([t["effect"] for t in self.train_tasks])
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True)
        self.x_std[self.x_std < 1e-6] = 1.0
        xs = (x - self.x_mean) / self.x_std
        xtx = xs.T @ xs
        self.coef = np.linalg.solve(xtx + self.alpha * np.eye(xtx.shape[0]), xs.T @ y)
        return self

    def predict(self, tasks: list[dict], indices: np.ndarray) -> np.ndarray:
        selected = [tasks[int(i)] for i in indices]
        x = (self._features(selected) - self.x_mean) / self.x_std
        return (x @ self.coef).astype(np.float32)


def pick_datasets(scan: pd.DataFrame, names: list[str], max_datasets: int) -> pd.DataFrame:
    df = scan[scan["study_family"].isin(names)].copy()
    df = df[df["local_path"].map(lambda p: Path(str(p)).exists())]
    if "has_control_like" in df:
        df = df[df["has_control_like"].astype(str).str.lower() == "true"]
    if "scan_status" in df:
        df = df[df["scan_status"].astype(str) == "ok"]
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
                "network_module_consistency_delta": v.get("network_module_consistency_mean", np.nan)
                - b.get("network_module_consistency_mean", np.nan),
            }
        )
        row["effect_positive_dims"] = int(
            (row["top20_delta"] > 0) + (row["deg_precision_delta"] > 0) + (row["program_consistency_delta"] > 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_with_network(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_results(df)
    if df.empty or "network_module_consistency" not in df:
        return summary
    rows = []
    for keys, sub in df.groupby(["phase", "dataset", "split_type", "model"], dropna=False):
        row = dict(zip(["phase", "dataset", "split_type", "model"], keys))
        row["network_module_consistency_mean"] = float(sub["network_module_consistency"].mean())
        row["network_module_consistency_std"] = float(sub["network_module_consistency"].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return summary.merge(pd.DataFrame(rows), on=["phase", "dataset", "split_type", "model"], how="left")


def avg_metrics(tasks: list[dict], test_idx: np.ndarray, pred: np.ndarray, eval_bank: ProgramBank, network_bank: NetworkModuleBank) -> dict:
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    tz = eval_bank.transform(y)
    pz = eval_bank.transform(pred)
    nz_true = network_bank.transform(y)
    nz_pred = network_bank.transform(pred)
    rows = [effect_metrics(y[i], pred[i], tz[i], pz[i]) for i in range(len(test_idx))]
    metric = pd.DataFrame(rows).mean().to_dict()
    net = [effect_metrics(nz_true[i], nz_pred[i])["pearson"] for i in range(len(test_idx))]
    metric["network_module_consistency"] = float(pd.Series(net).mean())
    return metric


def build_models(n_programs: int, seed: int, tasks: list[dict], train_mask: np.ndarray):
    return [
        V0StrongBaseline().fit(tasks, train_mask),
        V2GraphPriorTransport(ProgramBank(n_programs, seed, mode="pca_nmf_hvg"), alpha=5.0, blend=0.12).fit(tasks, train_mask),
        ScGenStyleLatentDelta(n_programs, seed).fit(tasks, train_mask),
        OTBarycentricProxy(k=7).fit(tasks, train_mask),
        GraphKernelProxy().fit(tasks, train_mask),
        ContextPerturbRidgeProxy().fit(tasks, train_mask),
    ]


def run_dataset(row: pd.Series, phase: str, seeds: list[int], args, out: Path) -> tuple[list[dict], list[dict]]:
    dataset = str(row["study_family"])
    results: list[dict] = []
    audits: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=args.n_genes, seed=seed)
            splits = selected_splits(tasks, args.split_per_type)
            audits.append({"phase": phase, "dataset": dataset, "seed": seed, **meta, "n_splits": len(splits), "status": "ok"})
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(train_idx) < 4 or len(test_idx) < 2:
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[train_idx] = True
                train_effects = np.stack([tasks[int(i)]["effect"] for i in train_idx])
                eval_bank = ProgramBank(args.n_programs, seed, mode="pca_nmf_hvg").fit(train_effects)
                network_bank = NetworkModuleBank(args.n_programs, seed, max_network_genes=args.max_network_genes).fit(train_effects)
                for model in build_models(args.n_programs, seed, tasks, train_mask):
                    pred = model.predict(tasks, test_idx)
                    metric = avg_metrics(tasks, test_idx, pred, eval_bank, network_bank)
                    metric.update(
                        {
                            "phase": phase,
                            "dataset": dataset,
                            "split_type": split["split_type"],
                            "heldout": split["heldout"],
                            "seed": seed,
                            "model": model.name,
                            "n_train": int(len(train_idx)),
                            "n_tasks": int(len(test_idx)),
                        }
                    )
                    results.append(metric)
                    pd.DataFrame(results).to_csv(out / "COMMUNITY_BASELINE_RESULTS_INCREMENTAL.csv", index=False)
        except Exception as exc:
            audits.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
    return results, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    parser.add_argument("--seeds", default="13101,13111,13121")
    parser.add_argument("--external-seed-count", type=int, default=3)
    parser.add_argument("--max-datasets", type=int, default=4)
    parser.add_argument("--max-external-datasets", type=int, default=3)
    parser.add_argument("--n-genes", type=int, default=1400)
    parser.add_argument("--n-programs", type=int, default=96)
    parser.add_argument("--max-network-genes", type=int, default=1200)
    parser.add_argument("--split-per-type", type=int, default=2)
    parser.add_argument("--main-studies", default=",".join(MAIN_STUDIES))
    parser.add_argument("--external-studies", default=",".join(EXTERNAL_STUDIES))
    args = parser.parse_args()

    root = Path(args.root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ext_seeds = seeds[: max(1, min(args.external_seed_count, len(seeds)))]
    main_names = [x.strip() for x in args.main_studies.split(",") if x.strip()]
    external_names = [x.strip() for x in args.external_studies.split(",") if x.strip()]
    main_ds = pick_datasets(scan, main_names, args.max_datasets)
    ext_ds = pick_datasets(scan, [x for x in external_names if x not in set(main_ds["study_family"].astype(str))], args.max_external_datasets)
    main_ds.to_csv(out / "COMMUNITY_MAIN_SELECTED.csv", index=False)
    ext_ds.to_csv(out / "COMMUNITY_EXTERNAL_SELECTED.csv", index=False)

    all_rows: list[dict] = []
    audit_rows: list[dict] = []
    for _, ds in main_ds.iterrows():
        rows, audits = run_dataset(ds, "main", seeds, args, out)
        all_rows.extend(rows)
        audit_rows.extend(audits)
        pd.DataFrame(all_rows).to_csv(out / "COMMUNITY_BASELINE_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "COMMUNITY_BASELINE_AUDIT.csv", index=False)
    for _, ds in ext_ds.iterrows():
        rows, audits = run_dataset(ds, "external", ext_seeds, args, out)
        all_rows.extend(rows)
        audit_rows.extend(audits)
        pd.DataFrame(all_rows).to_csv(out / "COMMUNITY_BASELINE_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "COMMUNITY_BASELINE_AUDIT.csv", index=False)

    df = pd.DataFrame(all_rows)
    summary = summarize_with_network(df)
    summary.to_csv(out / "COMMUNITY_BASELINE_SUMMARY.csv", index=False)
    models = sorted([m for m in summary["model"].dropna().unique() if m not in {"V0", "V2"}]) if not summary.empty else []
    for model in models + ["V2"]:
        compare_model_vs(summary, model, "V0").to_csv(out / f"{model}_VS_V0.csv", index=False)
        compare_model_vs(summary, model, "V2").to_csv(out / f"{model}_VS_V2.csv", index=False)
    status = {
        "n_rows": int(len(df)),
        "models": sorted(df["model"].dropna().unique().tolist()) if not df.empty else [],
        "main_datasets": main_ds["study_family"].astype(str).tolist(),
        "external_datasets": ext_ds["study_family"].astype(str).tolist(),
    }
    (out / "COMMUNITY_BASELINE_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

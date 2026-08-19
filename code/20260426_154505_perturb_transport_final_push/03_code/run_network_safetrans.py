from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from evaluators import compare_model_vs_v0, effect_metrics, summarize_results
from network_modules import NetworkModuleBank
from program_bank import ProgramBank
from risk_coverage import risk_coverage_curve, task_errors
from safetrans_models import NetworkSafeTransPT, NetworkSafeTransPTNoAbstain, SafeTransPT
from transport_models import V0StrongBaseline, V2GraphPriorTransport


MAIN_STUDIES = [
    "Haber",
    "Parekh",
    "KaggleCrossCell",
    "Wessels",
    "Frangieh",
    "NormanWeissman2019",
    "DixitRegev2016",
]
EXTERNAL_STUDIES = [
    "KaggleCrossPatient",
    "McFarland",
    "crossPatient",
    "TCDD",
    "PapalexiSatija2021",
    "TianKampmann2019",
    "SrivatsanTrapnell2020",
]


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
    metric_pairs = {
        "pearson_delta": "pearson_mean",
        "spearman_delta": "spearman_mean",
        "rmse_delta": "rmse_mean",
        "top20_delta": "top20_overlap_mean",
        "deg_precision_delta": "deg_precision_top50_mean",
        "program_consistency_delta": "program_shift_consistency_mean",
        "network_module_consistency_delta": "network_module_consistency_mean",
    }
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
        for out_col, metric_col in metric_pairs.items():
            if metric_col in b and metric_col in v:
                row[out_col] = v[metric_col] - b[metric_col]
        row["effect_positive_dims"] = int(
            (row.get("top20_delta", 0) > 0)
            + (row.get("deg_precision_delta", 0) > 0)
            + (row.get("program_consistency_delta", 0) > 0)
        )
        row["network_positive_dims"] = int(row.get("network_module_consistency_delta", 0) > 0)
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
    net = pd.DataFrame(rows)
    if summary.empty:
        return net
    return summary.merge(net, on=["phase", "dataset", "split_type", "model"], how="left")


def avg_metrics(tasks: list[dict], test_idx: np.ndarray, pred: np.ndarray, eval_bank: ProgramBank, network_bank: NetworkModuleBank) -> dict:
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    tz = eval_bank.transform(y)
    pz = eval_bank.transform(pred)
    nz_true = network_bank.transform(y)
    nz_pred = network_bank.transform(pred)
    metric = pd.DataFrame([effect_metrics(y[i], pred[i], tz[i], pz[i]) for i in range(len(test_idx))]).mean().to_dict()
    metric["network_module_consistency"] = float(
        pd.Series([effect_metrics(nz_true[i], nz_pred[i])["pearson"] for i in range(len(test_idx))]).mean()
    )
    return metric


def build_models(n_programs: int, seed: int, train_mask: np.ndarray, tasks: list[dict], max_network_genes: int):
    return [
        V0StrongBaseline().fit(tasks, train_mask),
        V2GraphPriorTransport(ProgramBank(n_programs, seed, mode="pca_nmf_hvg"), alpha=5.0, blend=0.12).fit(tasks, train_mask),
        SafeTransPT(n_programs=n_programs, seed=seed, bank_mode="pca_nmf_hvg").fit(tasks, train_mask),
        NetworkSafeTransPT(n_programs=n_programs, seed=seed, max_network_genes=max_network_genes).fit(tasks, train_mask),
        NetworkSafeTransPTNoAbstain(n_programs=n_programs, seed=seed, max_network_genes=max_network_genes).fit(tasks, train_mask),
    ]


def run_dataset(row: pd.Series, phase: str, seeds: list[int], args, out: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dataset = str(row["study_family"])
    result_rows: list[dict] = []
    audit_rows: list[dict] = []
    detail_rows: list[dict] = []
    risk_rows: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=args.n_genes, seed=seed)
            splits = selected_splits(tasks, args.split_per_type)
            pd.DataFrame(splits).to_csv(out / f"{phase}_{dataset}_seed{seed}_network_splits.csv", index=False)
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, **meta, "n_splits": len(splits), "status": "ok"})
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(train_idx) < 4 or len(test_idx) < 2:
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[train_idx] = True
                train_effects = np.stack([tasks[int(i)]["effect"] for i in train_idx], axis=0)
                eval_bank = ProgramBank(args.n_programs, seed, mode="pca_nmf_hvg").fit(train_effects)
                network_bank = NetworkModuleBank(args.n_programs, seed, max_network_genes=args.max_network_genes).fit(train_effects)
                module_table = network_bank.module_table()
                if not module_table.empty:
                    module_table["phase"] = phase
                    module_table["dataset"] = dataset
                    module_table["split_type"] = split["split_type"]
                    module_table["heldout"] = split["heldout"]
                    module_table["seed"] = seed
                    module_table.to_csv(out / f"{phase}_{dataset}_{split['split_type']}_seed{seed}_NETWORK_MODULES.csv", index=False)
                y_true = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
                for model in build_models(args.n_programs, seed, train_mask, tasks, args.max_network_genes):
                    if hasattr(model, "predict_details"):
                        details = model.predict_details(tasks, test_idx)
                        pred = details["prediction"]
                        ttab = details["transportability"].copy()
                    else:
                        pred = model.predict(tasks, test_idx)
                        ttab = pd.DataFrame()
                    metrics = avg_metrics(tasks, test_idx, pred, eval_bank, network_bank)
                    metrics.update(
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
                    if not ttab.empty:
                        metrics["transportability_mean"] = float(ttab["transportability_score"].mean())
                        metrics["adaptive_blend_mean"] = float(ttab["adaptive_blend"].mean())
                        metrics["unsafe_rate"] = float(ttab["unsafe_flag"].mean())
                        if "network_preservation_score" in ttab:
                            metrics["network_preservation_mean"] = float(ttab["network_preservation_score"].mean())
                        terr = task_errors(y_true, pred)
                        for i, task_id in enumerate(test_idx):
                            detail = {
                                "phase": phase,
                                "dataset": dataset,
                                "split_type": split["split_type"],
                                "heldout": split["heldout"],
                                "seed": seed,
                                "model": model.name,
                                "task_index": int(task_id),
                                "context": tasks[int(task_id)]["context"],
                                "perturbation": tasks[int(task_id)]["perturbation"],
                            }
                            detail.update(ttab.iloc[i].to_dict())
                            detail.update(terr.iloc[i].drop(labels=["task_id"]).to_dict())
                            detail_rows.append(detail)
                        rc = risk_coverage_curve(y_true, pred, ttab["transportability_score"].to_numpy())
                        for item in rc.to_dict("records"):
                            item.update(
                                {
                                    "phase": phase,
                                    "dataset": dataset,
                                    "split_type": split["split_type"],
                                    "heldout": split["heldout"],
                                    "seed": seed,
                                    "model": model.name,
                                }
                            )
                            risk_rows.append(item)
                    result_rows.append(metrics)
                    pd.DataFrame(result_rows).to_csv(out / f"{phase.upper()}_NETWORK_RESULTS_INCREMENTAL.csv", index=False)
        except Exception as exc:
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
    return result_rows, audit_rows, detail_rows, risk_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    parser.add_argument("--seeds", default="7101,7201,7301")
    parser.add_argument("--external-seed-count", type=int, default=3)
    parser.add_argument("--max-datasets", type=int, default=4)
    parser.add_argument("--max-external-datasets", type=int, default=3)
    parser.add_argument("--n-genes", type=int, default=1600)
    parser.add_argument("--n-programs", type=int, default=96)
    parser.add_argument("--max-network-genes", type=int, default=1400)
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
    main_ds.to_csv(out / "NETWORK_MAIN_SELECTED_DATASETS.csv", index=False)
    ext_ds.to_csv(out / "NETWORK_EXTERNAL_SELECTED_DATASETS.csv", index=False)

    all_rows: list[dict] = []
    audit_rows: list[dict] = []
    detail_rows: list[dict] = []
    risk_rows: list[dict] = []
    for _, ds in main_ds.iterrows():
        rows, audits, details, risks = run_dataset(ds, "main", seeds, args, out)
        all_rows.extend(rows); audit_rows.extend(audits); detail_rows.extend(details); risk_rows.extend(risks)
        pd.DataFrame(all_rows).to_csv(out / "NETWORK_SAFETRANS_ALL_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "NETWORK_DATASET_AUDIT.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "NETWORK_TASK_DETAILS.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / "NETWORK_RISK_COVERAGE.csv", index=False)
    for _, ds in ext_ds.iterrows():
        rows, audits, details, risks = run_dataset(ds, "external", ext_seeds, args, out)
        all_rows.extend(rows); audit_rows.extend(audits); detail_rows.extend(details); risk_rows.extend(risks)
        pd.DataFrame(all_rows).to_csv(out / "NETWORK_SAFETRANS_ALL_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "NETWORK_DATASET_AUDIT.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "NETWORK_TASK_DETAILS.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / "NETWORK_RISK_COVERAGE.csv", index=False)

    results = pd.DataFrame(all_rows)
    summary = summarize_with_network(results)
    summary.to_csv(out / "NETWORK_SAFETRANS_SUMMARY.csv", index=False)
    for model in ["NetworkSafeTransPT", "NetworkSafeTransPT_no_abstain", "SafeTransPT"]:
        compare_model_vs_v0(summary, model).to_csv(out / f"{model}_VS_V0.csv", index=False)
        compare_model_vs(summary, model, "V2").to_csv(out / f"{model}_VS_V2.csv", index=False)
    status = {
        "n_rows": int(len(results)),
        "main_datasets": main_ds["study_family"].astype(str).tolist(),
        "external_datasets": ext_ds["study_family"].astype(str).tolist(),
        "network_rows": int((results["model"] == "NetworkSafeTransPT").sum()) if not results.empty else 0,
    }
    delta = compare_model_vs(summary, "NetworkSafeTransPT", "V2")
    if not delta.empty:
        status.update(
            {
                "mean_top20_delta_vs_v2": float(delta["top20_delta"].mean()),
                "mean_deg_delta_vs_v2": float(delta["deg_precision_delta"].mean()),
                "mean_program_delta_vs_v2": float(delta["program_consistency_delta"].mean()),
                "mean_network_module_delta_vs_v2": float(delta["network_module_consistency_delta"].mean()),
            }
        )
    (out / "NETWORK_SAFETRANS_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

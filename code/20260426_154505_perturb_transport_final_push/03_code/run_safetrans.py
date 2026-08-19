from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from evaluators import compare_model_vs_v0, effect_metrics, summarize_results
from program_bank import ProgramBank
from risk_coverage import risk_coverage_curve, task_errors
from safetrans_models import SafeTransPT, SafeTransPTNoAbstain, SafeTransPTNoPathway
from transport_models import V0StrongBaseline, V1ProgramTransport, V2GraphPriorTransport, V3UncertaintyTransport


MAIN_STUDIES = ["Haber", "Parekh", "KaggleCrossCell", "Wessels", "Frangieh", "kangCrossCell", "kangCrossPatient"]
EXTERNAL_STUDIES = ["KaggleCrossPatient", "McFarland", "crossPatient", "TCDD", "Afriat", "sciplex3"]


def pick_datasets(scan: pd.DataFrame, names: list[str], max_datasets: int, require_control: bool = True) -> pd.DataFrame:
    df = scan[scan["study_family"].isin(names)].copy()
    df = df[df["local_path"].map(lambda p: Path(str(p)).exists())]
    if require_control and "has_control_like" in df:
        df = df[df["has_control_like"].astype(str).str.lower() == "true"]
    if "scan_status" in df:
        df = df[df["scan_status"].astype(str) == "ok"]
    df["priority"] = df["study_family"].map({k: i for i, k in enumerate(names)}).fillna(999)
    return df.sort_values(["priority", "n_obs"]).drop_duplicates("study_family").head(max_datasets)


def selected_splits(tasks: list[dict], per_type: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    raw = feasible_splits(tasks)
    if not raw:
        return pd.DataFrame(), []
    df = pd.DataFrame(raw)
    rows = []
    for split_type in ["leave_context", "heldout_perturbation"]:
        sub = df[df["split_type"] == split_type].sort_values(["n_test", "n_train"], ascending=False).head(per_type)
        rows.extend(sub.to_dict("records"))
    return pd.DataFrame(rows), rows


def avg_metrics(tasks: list[dict], test_idx: np.ndarray, pred: np.ndarray, bank: ProgramBank) -> dict:
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    tz = bank.transform(y)
    pz = bank.transform(pred)
    return pd.DataFrame([effect_metrics(y[i], pred[i], tz[i], pz[i]) for i in range(len(test_idx))]).mean().to_dict()


def build_models(n_programs: int, seed: int, train_mask: np.ndarray, tasks: list[dict]):
    return [
        V0StrongBaseline().fit(tasks, train_mask),
        V1ProgramTransport(ProgramBank(n_programs, seed, mode="pca"), alpha=3.0).fit(tasks, train_mask),
        V2GraphPriorTransport(ProgramBank(n_programs, seed, mode="pca_nmf_hvg"), alpha=5.0, blend=0.12).fit(tasks, train_mask),
        V3UncertaintyTransport(ProgramBank(n_programs, seed, mode="pca"), alpha=3.0).fit(tasks, train_mask),
        SafeTransPT(n_programs=n_programs, seed=seed, bank_mode="pca_nmf_hvg").fit(tasks, train_mask),
        SafeTransPTNoAbstain(n_programs=n_programs, seed=seed, bank_mode="pca_nmf_hvg").fit(tasks, train_mask),
        SafeTransPTNoPathway(n_programs=n_programs, seed=seed, bank_mode="pca_nmf_hvg").fit(tasks, train_mask),
    ]


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
            (row["top20_delta"] > 0) + (row["deg_precision_delta"] > 0) + (row["program_consistency_delta"] > 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(
    row: pd.Series,
    phase: str,
    seeds: list[int],
    n_genes: int,
    n_programs: int,
    split_per_type: int,
    out: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dataset = str(row["study_family"])
    result_rows: list[dict] = []
    audit_rows: list[dict] = []
    detail_rows: list[dict] = []
    risk_rows: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=n_genes, seed=seed)
            split_df, splits = selected_splits(tasks, per_type=split_per_type)
            split_df.to_csv(out / f"{phase}_{dataset}_seed{seed}_splits.csv", index=False)
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, **meta, "n_splits": len(splits), "status": "ok"})
            if not splits:
                continue
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(test_idx) < 2 or len(train_idx) < 4:
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[train_idx] = True
                bank = ProgramBank(n_programs, seed, mode="pca_nmf_hvg").fit(
                    np.stack([tasks[int(i)]["effect"] for i in train_idx])
                )
                y_true = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
                for model in build_models(n_programs, seed, train_mask, tasks):
                    if hasattr(model, "train_mask_for_predict"):
                        model.train_mask_for_predict = train_mask
                    if hasattr(model, "predict_details"):
                        details = model.predict_details(tasks, test_idx)
                        pred = details["prediction"]
                        ttab = details["transportability"].copy()
                    else:
                        pred = model.predict(tasks, test_idx)
                        ttab = pd.DataFrame()
                    metrics = avg_metrics(tasks, test_idx, pred, bank)
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
                    if hasattr(model, "uncertainty"):
                        unc = model.uncertainty(tasks, test_idx)
                        errors = np.sqrt(np.mean((y_true - pred) ** 2, axis=1))
                        metrics["uncertainty_error_spearman"] = effect_metrics(unc, errors)["spearman"]
                    if not ttab.empty:
                        metrics["transportability_mean"] = float(ttab["transportability_score"].mean())
                        metrics["adaptive_blend_mean"] = float(ttab["adaptive_blend"].mean())
                        metrics["unsafe_rate"] = float(ttab["unsafe_flag"].mean())
                        terr = task_errors(y_true, pred)
                        for i, task_id in enumerate(test_idx):
                            row_detail = {
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
                            row_detail.update(ttab.iloc[i].to_dict())
                            row_detail.update(terr.iloc[i].drop(labels=["task_id"]).to_dict())
                            detail_rows.append(row_detail)
                        rc = risk_coverage_curve(y_true, pred, ttab["transportability_score"].to_numpy())
                        for rr in rc.to_dict("records"):
                            rr.update(
                                {
                                    "phase": phase,
                                    "dataset": dataset,
                                    "split_type": split["split_type"],
                                    "heldout": split["heldout"],
                                    "seed": seed,
                                    "model": model.name,
                                }
                            )
                            risk_rows.append(rr)
                    result_rows.append(metrics)
                    pd.DataFrame(result_rows).to_csv(out / f"{phase.upper()}_RESULTS_INCREMENTAL.csv", index=False)
        except Exception as exc:
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
    return result_rows, audit_rows, detail_rows, risk_rows


def strict_pass_settings(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return deltas
    return deltas[
        (deltas["effect_positive_dims"] >= 2)
        & (
            (deltas["top20_delta"] >= 0.002)
            | (deltas["deg_precision_delta"] >= 0.002)
            | (deltas["program_consistency_delta"] >= 0.005)
        )
    ].copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--external-seed-count", type=int, default=5)
    p.add_argument("--max-datasets", type=int, default=5)
    p.add_argument("--max-external-datasets", type=int, default=2)
    p.add_argument("--n-genes", type=int, default=2500)
    p.add_argument("--n-programs", type=int, default=128)
    p.add_argument("--split-per-type", type=int, default=2)
    args = p.parse_args()

    root = Path(args.root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ext_seeds = seeds[: max(1, min(args.external_seed_count, len(seeds)))]

    main_selected = pick_datasets(scan, MAIN_STUDIES, args.max_datasets, require_control=True)
    external_selected = pick_datasets(scan, [x for x in EXTERNAL_STUDIES if x not in set(main_selected["study_family"].astype(str))], args.max_external_datasets, require_control=True)
    main_selected.to_csv(out / "SAFETRANS_MAIN_SELECTED_DATASETS.csv", index=False)
    external_selected.to_csv(out / "SAFETRANS_EXTERNAL_SELECTED_DATASETS.csv", index=False)

    all_rows: list[dict] = []
    audit_rows: list[dict] = []
    detail_rows: list[dict] = []
    risk_rows: list[dict] = []
    for _, ds in main_selected.iterrows():
        rows, audits, details, risks = run_dataset(ds, "main", seeds, args.n_genes, args.n_programs, args.split_per_type, out)
        all_rows.extend(rows); audit_rows.extend(audits); detail_rows.extend(details); risk_rows.extend(risks)
        pd.DataFrame(all_rows).to_csv(out / "SAFETRANS_ALL_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "SAFETRANS_DATASET_AUDIT.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "SAFETRANS_TASK_DETAILS.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / "SAFETRANS_RISK_COVERAGE.csv", index=False)

    for _, ds in external_selected.iterrows():
        rows, audits, details, risks = run_dataset(ds, "external", ext_seeds, args.n_genes, args.n_programs, args.split_per_type, out)
        all_rows.extend(rows); audit_rows.extend(audits); detail_rows.extend(details); risk_rows.extend(risks)
        pd.DataFrame(all_rows).to_csv(out / "SAFETRANS_ALL_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "SAFETRANS_DATASET_AUDIT.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "SAFETRANS_TASK_DETAILS.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out / "SAFETRANS_RISK_COVERAGE.csv", index=False)

    results = pd.DataFrame(all_rows)
    summary = summarize_results(results)
    summary.to_csv(out / "SAFETRANS_SUMMARY_TABLE.csv", index=False)
    delta_v0 = compare_model_vs_v0(summary, "SafeTransPT")
    delta_v2 = compare_model_vs(summary, "SafeTransPT", "V2")
    delta_v0.to_csv(out / "SAFETRANS_VS_V0_DELTAS.csv", index=False)
    delta_v2.to_csv(out / "SAFETRANS_VS_FIXED_V2_DELTAS.csv", index=False)
    ablation_rows = []
    for model in ["SafeTransPT_no_abstain", "SafeTransPT_no_pathway"]:
        d = compare_model_vs(summary, model, "SafeTransPT")
        d["ablation"] = model
        ablation_rows.extend(d.to_dict("records"))
    pd.DataFrame(ablation_rows).to_csv(out / "SAFETRANS_ABLATION_DELTAS.csv", index=False)

    main_pass = strict_pass_settings(delta_v2[delta_v2["phase"] == "main"])
    ext_pass = strict_pass_settings(delta_v2[delta_v2["phase"] == "external"])
    risk = pd.DataFrame(risk_rows)
    coverage_ok = False
    if not risk.empty:
        target = risk[(risk["model"] == "SafeTransPT") & (risk["coverage"].between(0.69, 0.81))]
        coverage_ok = bool(len(target) and target["abstention_rate"].mean() <= 0.30)
    label = "SAFETRANS_PROMISING_FOR_PAPER" if len(main_pass) >= 2 and len(ext_pass) >= 1 and coverage_ok else "SAFETRANS_NEEDS_MORE_WORK"
    status = {
        "label": label,
        "n_rows": int(len(results)),
        "main_pass_vs_fixed_v2": int(len(main_pass)),
        "external_pass_vs_fixed_v2": int(len(ext_pass)),
        "coverage_ok": bool(coverage_ok),
        "selected_main": main_selected["study_family"].astype(str).tolist(),
        "selected_external": external_selected["study_family"].astype(str).tolist(),
        "strict_note": "Pass requires SafeTransPT to improve fixed V2 in at least two effect-based dimensions.",
    }
    (out / "SAFETRANS_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

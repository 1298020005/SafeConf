from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from evaluators import effect_metrics, summarize_results
from program_bank import ProgramBank
from risk_coverage import risk_coverage_curve
from safetrans_models import (
    NetworkSafeTransPT,
    PolicySafeTransPT,
    SafeTransPT,
    SafeTransPTNoAbstain,
    SafeTransPTNoPathway,
)
from transport_models import ContextSimilarityBaseline, V0StrongBaseline, V2GraphPriorTransport


MAIN_STUDIES = [
    "KaggleCrossCell",
    "kangCrossCell",
    "Haber",
    "Parekh",
    "Wessels",
    "NormanWeissman2019",
    "DixitRegev2016",
    "AdamsonWeissman2016",
]

EXTERNAL_STUDIES = [
    "KaggleCrossPatient",
    "McFarland",
    "crossPatient",
    "TCDD",
    "PapalexiSatija2021",
    "TianKampmann2019",
    "SrivatsanTrapnell2020",
    "Frangieh",
]


def pick_datasets(scan: pd.DataFrame, names: list[str], max_datasets: int) -> pd.DataFrame:
    df = scan[scan["study_family"].isin(names)].copy()
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
    for split_type in ["heldout_perturbation", "leave_context"]:
        sub = df[df["split_type"] == split_type].sort_values(["n_test", "n_train"], ascending=False)
        rows.extend(sub.head(per_type).to_dict("records"))
    return rows


def _task_metric_rows(
    y: np.ndarray,
    pred: np.ndarray,
    eval_bank: ProgramBank,
    meta: dict,
    confidence: np.ndarray | None = None,
    unsafe_flag: np.ndarray | None = None,
) -> list[dict]:
    true_program = eval_bank.transform(y)
    pred_program = eval_bank.transform(pred)
    rows = []
    for i in range(y.shape[0]):
        row = effect_metrics(y[i], pred[i], true_program[i], pred_program[i])
        row.update(meta)
        row["task_id"] = int(i)
        if confidence is not None:
            row["confidence"] = float(confidence[i])
        if unsafe_flag is not None:
            row["unsafe_flag"] = int(unsafe_flag[i])
        rows.append(row)
    return rows


def _risk_rows(y: np.ndarray, pred: np.ndarray, confidence: np.ndarray, meta: dict) -> list[dict]:
    curve = risk_coverage_curve(y, pred, confidence, coverages=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    for key, value in meta.items():
        curve[key] = value
    return curve.to_dict("records")


def _unsafe_contrast(task_df: pd.DataFrame, meta: dict) -> dict:
    out = dict(meta)
    if task_df.empty or "unsafe_flag" not in task_df:
        out.update({"status": "missing"})
        return out
    safe = task_df[task_df["unsafe_flag"] == 0]
    unsafe = task_df[task_df["unsafe_flag"] == 1]
    out["n_safe"] = int(len(safe))
    out["n_unsafe"] = int(len(unsafe))
    if safe.empty or unsafe.empty:
        out["status"] = "single_group"
        return out
    out["unsafe_minus_safe_rmse"] = float(unsafe["rmse"].mean() - safe["rmse"].mean())
    out["safe_minus_unsafe_top20"] = float(safe["top20_overlap"].mean() - unsafe["top20_overlap"].mean())
    out["safe_minus_unsafe_deg"] = float(safe["deg_precision_top50"].mean() - unsafe["deg_precision_top50"].mean())
    out["safe_minus_unsafe_program"] = float(safe["program_shift_consistency"].mean() - unsafe["program_shift_consistency"].mean())
    out["status"] = "ok"
    return out


def run_dataset(row: pd.Series, phase: str, seeds: list[int], args, out: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dataset = str(row["study_family"])
    metric_rows: list[dict] = []
    risk_rows: list[dict] = []
    contrast_rows: list[dict] = []
    audit_rows: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=args.n_genes, seed=seed)
            splits = selected_splits(tasks, args.split_per_type)
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, **meta, "n_splits": len(splits), "status": "ok"})
            for split in splits:
                train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
                if len(train_idx) < 4 or len(test_idx) < 2:
                    continue
                train_mask = np.zeros(len(tasks), dtype=bool)
                train_mask[train_idx] = True
                train_effects = np.stack([tasks[int(i)]["effect"] for i in train_idx])
                y = np.stack([tasks[int(i)]["effect"] for i in test_idx])
                eval_bank = ProgramBank(args.n_programs, seed, mode=args.eval_bank).fit(train_effects)
                base_meta = {
                    "phase": phase,
                    "dataset": dataset,
                    "split_type": split["split_type"],
                    "heldout": split["heldout"],
                    "seed": seed,
                    "n_train": int(len(train_idx)),
                    "n_tasks": int(len(test_idx)),
                }
                models = [
                    V0StrongBaseline().fit(tasks, train_mask),
                    ContextSimilarityBaseline().fit(tasks, train_mask),
                    V2GraphPriorTransport(ProgramBank(args.n_programs, seed, mode=args.eval_bank), alpha=5.0, blend=args.v2_blend).fit(tasks, train_mask),
                    SafeTransPT(n_programs=args.n_programs, seed=seed, max_blend=args.max_blend, unsafe_threshold=args.unsafe_threshold).fit(tasks, train_mask),
                    SafeTransPTNoAbstain(n_programs=args.n_programs, seed=seed, max_blend=args.max_blend, unsafe_threshold=args.unsafe_threshold).fit(tasks, train_mask),
                    SafeTransPTNoPathway(n_programs=args.n_programs, seed=seed, max_blend=args.max_blend, unsafe_threshold=args.unsafe_threshold).fit(tasks, train_mask),
                    NetworkSafeTransPT(n_programs=max(64, args.n_programs // 2), seed=seed, max_blend=args.max_blend, unsafe_threshold=args.unsafe_threshold).fit(tasks, train_mask),
                    PolicySafeTransPT(
                        n_programs=max(64, args.n_programs // 2),
                        seed=seed,
                        bank_mode=args.eval_bank,
                        routing_mode=args.policy_routing_mode,
                    ).fit(tasks, train_mask),
                ]
                for model in models:
                    meta_row = {**base_meta, "model": model.name}
                    confidence = None
                    unsafe_flag = None
                    if hasattr(model, "predict_details"):
                        details = model.predict_details(tasks, test_idx)
                        pred = details["prediction"]
                        table = details["transportability"]
                        confidence = table["transportability_score"].to_numpy(dtype=np.float64)
                        unsafe_flag = table["unsafe_flag"].to_numpy(dtype=int)
                    else:
                        pred = model.predict(tasks, test_idx)
                    metric_rows.extend(_task_metric_rows(y, pred, eval_bank, meta_row, confidence, unsafe_flag))
                    if confidence is not None:
                        risk_rows.extend(_risk_rows(y, pred, confidence, meta_row))
                        task_sub = pd.DataFrame([r for r in metric_rows if all(r.get(k) == v for k, v in meta_row.items())])
                        contrast_rows.append(_unsafe_contrast(task_sub, meta_row))
                    pd.DataFrame(metric_rows).to_csv(out / "SAFETY_TASK_METRICS_INCREMENTAL.csv", index=False)
                    pd.DataFrame(risk_rows).to_csv(out / "RISK_COVERAGE_INCREMENTAL.csv", index=False)
                    pd.DataFrame(contrast_rows).to_csv(out / "SAFE_UNSAFE_CONTRAST_INCREMENTAL.csv", index=False)
        except Exception as exc:
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
    return metric_rows, risk_rows, contrast_rows, audit_rows


def compare_model_vs(summary: pd.DataFrame, model_name: str, baseline_name: str) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for keys, sub in summary.groupby(["phase", "dataset", "split_type"], dropna=False):
        cur = sub[sub["model"] == model_name]
        base = sub[sub["model"] == baseline_name]
        if cur.empty or base.empty:
            continue
        c = cur.iloc[0]
        b = base.iloc[0]
        row = dict(zip(["phase", "dataset", "split_type"], keys))
        row["model"] = model_name
        row["baseline"] = baseline_name
        row["pearson_delta"] = c["pearson_mean"] - b["pearson_mean"]
        row["spearman_delta"] = c["spearman_mean"] - b["spearman_mean"]
        row["rmse_delta"] = c["rmse_mean"] - b["rmse_mean"]
        row["top20_delta"] = c["top20_overlap_mean"] - b["top20_overlap_mean"]
        row["deg_precision_delta"] = c["deg_precision_top50_mean"] - b["deg_precision_top50_mean"]
        row["program_consistency_delta"] = c["program_shift_consistency_mean"] - b["program_shift_consistency_mean"]
        row["effect_positive_dims"] = int(
            (row["top20_delta"] > 0) + (row["deg_precision_delta"] > 0) + (row["program_consistency_delta"] > 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    parser.add_argument("--seeds", default="4101,4111,4121")
    parser.add_argument("--external-seed-count", type=int, default=3)
    parser.add_argument("--max-datasets", type=int, default=5)
    parser.add_argument("--max-external-datasets", type=int, default=4)
    parser.add_argument("--n-genes", type=int, default=1800)
    parser.add_argument("--n-programs", type=int, default=128)
    parser.add_argument("--split-per-type", type=int, default=2)
    parser.add_argument("--eval-bank", default="pca_nmf_hvg")
    parser.add_argument("--v2-blend", type=float, default=0.12)
    parser.add_argument("--max-blend", type=float, default=0.24)
    parser.add_argument("--unsafe-threshold", type=float, default=0.42)
    parser.add_argument("--policy-routing-mode", default="hard", choices=["hard", "soft", "hybrid"])
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
    main_ds.to_csv(out / "SAFETY_MAIN_SELECTED.csv", index=False)
    ext_ds.to_csv(out / "SAFETY_EXTERNAL_SELECTED.csv", index=False)

    metrics: list[dict] = []
    risks: list[dict] = []
    contrasts: list[dict] = []
    audits: list[dict] = []
    for _, ds in main_ds.iterrows():
        a, b, c, d = run_dataset(ds, "main", seeds, args, out)
        metrics.extend(a)
        risks.extend(b)
        contrasts.extend(c)
        audits.extend(d)
    for _, ds in ext_ds.iterrows():
        a, b, c, d = run_dataset(ds, "external", ext_seeds, args, out)
        metrics.extend(a)
        risks.extend(b)
        contrasts.extend(c)
        audits.extend(d)

    metric_df = pd.DataFrame(metrics)
    risk_df = pd.DataFrame(risks)
    contrast_df = pd.DataFrame(contrasts)
    audit_df = pd.DataFrame(audits)
    metric_df.to_csv(out / "SAFETY_TASK_METRICS.csv", index=False)
    risk_df.to_csv(out / "RISK_COVERAGE.csv", index=False)
    contrast_df.to_csv(out / "SAFE_UNSAFE_CONTRAST.csv", index=False)
    audit_df.to_csv(out / "SAFETY_AUDIT.csv", index=False)
    summary = summarize_results(metric_df)
    summary.to_csv(out / "SAFETY_SUMMARY.csv", index=False)
    for model in [
        "SafeTransPT",
        "SafeTransPT_no_abstain",
        "SafeTransPT_no_pathway",
        "NetworkSafeTransPT",
        "PolicySafeTransPT",
        "ContextSimBaseline",
        "V2",
    ]:
        compare_model_vs(summary, model, "V0").to_csv(out / f"{model}_VS_V0.csv", index=False)
        compare_model_vs(summary, model, "V2").to_csv(out / f"{model}_VS_V2.csv", index=False)
    compare_model_vs(summary, "PolicySafeTransPT", "ContextSimBaseline").to_csv(
        out / "PolicySafeTransPT_VS_ContextSimBaseline.csv", index=False
    )
    status = {
        "n_task_metric_rows": int(len(metric_df)),
        "n_risk_rows": int(len(risk_df)),
        "n_contrast_rows": int(len(contrast_df)),
        "main_datasets": main_ds["study_family"].astype(str).tolist(),
        "external_datasets": ext_ds["study_family"].astype(str).tolist(),
        "models": sorted(metric_df["model"].dropna().unique().tolist()) if not metric_df.empty else [],
    }
    (out / "SAFETY_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))

    try:
        from evaluate_q1_readiness import evaluate

        report = evaluate(out)
        (out / "Q1_READINESS_REPORT.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Q1 readiness:", report["label"])
    except Exception as exc:
        print("Q1 readiness evaluation skipped:", repr(exc))


if __name__ == "__main__":
    main()

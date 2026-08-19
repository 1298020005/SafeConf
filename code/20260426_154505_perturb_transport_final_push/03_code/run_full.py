from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table
from evaluators import compare_model_vs_v0, compare_v1_vs_v0, effect_metrics, summarize_results
from program_bank import ProgramBank
from safetrans_models import NetworkSafeTransPT, PolicySafeTransPT, SafeTransPT, SafeTransPTNoAbstain, SafeTransPTNoPathway
from transport_models import V0StrongBaseline, V1ProgramTransport, V2GraphPriorTransport, V3UncertaintyTransport

MAIN_STUDIES = ["Haber", "Parekh", "KaggleCrossCell", "kangCrossCell", "kangCrossPatient", "Norman", "Wessels"]
EXTERNAL_STUDIES = ["KaggleCrossPatient", "McFarland", "crossPatient", "Afriat", "TCDD", "Schmidt", "TianActivation", "TianInhibition"]


def pick_datasets(scan: pd.DataFrame, names: list[str], max_datasets: int) -> pd.DataFrame:
    df = scan[scan["study_family"].isin(names)].copy()
    df = df[df["local_path"].map(lambda p: Path(str(p)).exists())]
    df["priority"] = df["study_family"].map({k: i for i, k in enumerate(names)}).fillna(999)
    return df.sort_values(["priority", "n_obs"]).drop_duplicates("study_family").head(max_datasets)


def avg_metrics(tasks, test_idx, pred, bank):
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    tz = bank.transform(y)
    pz = bank.transform(pred)
    return pd.DataFrame([effect_metrics(y[i], pred[i], tz[i], pz[i]) for i in range(len(test_idx))]).mean().to_dict()


def build_models(n_programs: int, seed: int, tasks: list[dict], train_mask: np.ndarray):
    v2_mode = os.environ.get("PAIRDELTA_V2_BANK_MODE", "pca_nmf_hvg")
    routing_mode = os.environ.get("PAIRDELTA_ROUTING_MODE", "hard")
    return [
        V0StrongBaseline().fit(tasks, train_mask),
        V1ProgramTransport(ProgramBank(n_programs, seed, mode="pca"), alpha=3.0).fit(tasks, train_mask),
        V2GraphPriorTransport(ProgramBank(n_programs, seed, mode=v2_mode), alpha=5.0).fit(tasks, train_mask),
        V3UncertaintyTransport(ProgramBank(n_programs, seed), alpha=3.0).fit(tasks, train_mask),
        SafeTransPT(n_programs=n_programs, seed=seed, bank_mode=v2_mode, max_blend=0.18, unsafe_threshold=0.42).fit(tasks, train_mask),
        SafeTransPTNoAbstain(n_programs=n_programs, seed=seed, bank_mode=v2_mode, max_blend=0.18, unsafe_threshold=0.42).fit(tasks, train_mask),
        SafeTransPTNoPathway(n_programs=n_programs, seed=seed, bank_mode=v2_mode, max_blend=0.18, unsafe_threshold=0.42).fit(tasks, train_mask),
        NetworkSafeTransPT(n_programs=max(64, n_programs // 2), seed=seed, max_blend=0.22, unsafe_threshold=0.40).fit(tasks, train_mask),
        PolicySafeTransPT(n_programs=max(64, n_programs // 2), seed=seed, bank_mode=v2_mode, routing_mode=routing_mode).fit(tasks, train_mask),
    ]


def selected_splits(tasks: list[dict], per_type: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    raw = feasible_splits(tasks)
    if not raw:
        return pd.DataFrame(), []
    df = pd.DataFrame(raw)
    if "split_type" not in df.columns:
        return pd.DataFrame(), []
    rows = []
    for split_type in ["leave_context", "heldout_perturbation"]:
        sub = df[df["split_type"] == split_type].sort_values(["n_test", "n_train"], ascending=False).head(per_type)
        rows.extend(sub.to_dict("records"))
    return pd.DataFrame(rows), rows


def run_dataset(row: pd.Series, phase: str, seeds: list[int], n_genes: int, n_programs: int, out: Path) -> tuple[list[dict], list[dict]]:
    dataset = str(row["study_family"])
    result_rows: list[dict] = []
    audit_rows: list[dict] = []
    for seed in seeds:
        try:
            tasks, genes, meta = build_effect_tasks(Path(row["local_path"]), dataset, n_genes=n_genes, seed=seed)
            split_df, splits = selected_splits(tasks, per_type=3)
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
                bank = ProgramBank(n_programs, seed).fit(np.stack([tasks[int(i)]["effect"] for i in train_idx]))
                for model in build_models(n_programs, seed, tasks, train_mask):
                    if hasattr(model, "train_mask_for_predict"):
                        model.train_mask_for_predict = train_mask
                    pred = model.predict(tasks, test_idx)
                    m = avg_metrics(tasks, test_idx, pred, bank)
                    m.update(
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
                    if model.name == "V3":
                        unc = model.uncertainty(tasks, test_idx)
                        errors = np.sqrt(np.mean((np.stack([tasks[int(i)]["effect"] for i in test_idx]) - pred) ** 2, axis=1))
                        m["uncertainty_error_spearman"] = effect_metrics(unc, errors)["spearman"]
                    result_rows.append(m)
                    pd.DataFrame(result_rows).to_csv(out / ("FULL_RESULTS.csv" if phase == "main" else "EXTERNAL_VALIDATION.csv"), index=False)
        except Exception as exc:
            audit_rows.append({"phase": phase, "dataset": dataset, "seed": seed, "path": row.get("local_path"), "status": "failed", "error": repr(exc)})
    return result_rows, audit_rows


def pass_settings(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in ["PolicySafeTransPT", "NetworkSafeTransPT"]:
        deltas = compare_model_vs_v0(summary, model_name)
        if deltas.empty:
            continue
        keep = deltas[
            (deltas["top20_delta"] >= 0.01)
            | (deltas["deg_precision_delta"] >= 0.01)
            | ((deltas["program_consistency_delta"] > 0) & (deltas["rmse_delta"] <= 0.002))
        ].copy()
        rows.append(keep)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(["phase", "dataset", "split_type"], keep="first")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--seeds", default="11,22,33,44,55,66,77,88,99,101")
    p.add_argument("--max-datasets", type=int, default=6)
    p.add_argument("--n-genes", type=int, default=2000)
    p.add_argument("--n-programs", type=int, default=96)
    p.add_argument("--external-seed-count", type=int, default=5)
    args = p.parse_args()

    root = Path(args.root)
    out = root / "06_full_runs"
    out.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    main_selected = pick_datasets(scan, MAIN_STUDIES, args.max_datasets)
    external_selected = pick_datasets(scan, [x for x in EXTERNAL_STUDIES if x not in set(main_selected["study_family"].astype(str))], 2)
    main_selected.to_csv(out / "MAIN_SELECTED_DATASETS.csv", index=False)
    external_selected.to_csv(out / "EXTERNAL_SELECTED_DATASETS.csv", index=False)

    main_rows, audit_rows = [], []
    for _, ds in main_selected.iterrows():
        rows, audits = run_dataset(ds, "main", seeds, args.n_genes, args.n_programs, out)
        main_rows.extend(rows); audit_rows.extend(audits)
        pd.DataFrame(main_rows).to_csv(out / "FULL_RESULTS.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "FULL_DATASET_AUDIT.csv", index=False)

    ext_rows = []
    ext_seeds = seeds[: max(1, min(args.external_seed_count, len(seeds)))]
    for _, ds in external_selected.iterrows():
        rows, audits = run_dataset(ds, "external", ext_seeds, args.n_genes, args.n_programs, out)
        ext_rows.extend(rows); audit_rows.extend(audits)
        pd.DataFrame(ext_rows).to_csv(out / "EXTERNAL_VALIDATION.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out / "FULL_DATASET_AUDIT.csv", index=False)

    # Explicit leave-study-out feasibility record: honest because current H5ADs have non-identical HVG panels.
    leave_study = pd.DataFrame([
        {
            "setting": "leave_study_out",
            "status": "attempted_not_used_for_claim",
            "reason": "Selected H5ADs use study-specific HVG panels; gene-level effects are not safely aligned enough for cross-study effect decoding without a new harmonization pass.",
            "n_candidate_studies": int(len(main_selected)),
        }
    ])
    leave_study.to_csv(out / "LEAVE_STUDY_OUT_ATTEMPT.csv", index=False)

    results = pd.DataFrame(main_rows)
    external = pd.DataFrame(ext_rows)
    summary = summarize_results(results)
    ext_summary = summarize_results(external)
    summary.to_csv(out / "FULL_SUMMARY_TABLE.csv", index=False)
    ext_summary.to_csv(out / "EXTERNAL_SUMMARY_TABLE.csv", index=False)
    deltas = compare_v1_vs_v0(summary)
    ext_deltas = compare_v1_vs_v0(ext_summary)
    v2_deltas = compare_model_vs_v0(summary, "V2")
    ext_v2_deltas = compare_model_vs_v0(ext_summary, "V2")
    safe_deltas = compare_model_vs_v0(summary, "SafeTransPT")
    ext_safe_deltas = compare_model_vs_v0(ext_summary, "SafeTransPT")
    net_deltas = compare_model_vs_v0(summary, "NetworkSafeTransPT")
    ext_net_deltas = compare_model_vs_v0(ext_summary, "NetworkSafeTransPT")
    deltas.to_csv(out / "FULL_V1_VS_V0_DELTAS.csv", index=False)
    ext_deltas.to_csv(out / "EXTERNAL_V1_VS_V0_DELTAS.csv", index=False)
    v2_deltas.to_csv(out / "FULL_V2_VS_V0_DELTAS.csv", index=False)
    ext_v2_deltas.to_csv(out / "EXTERNAL_V2_VS_V0_DELTAS.csv", index=False)
    safe_deltas.to_csv(out / "FULL_SAFETRANS_VS_V0_DELTAS.csv", index=False)
    ext_safe_deltas.to_csv(out / "EXTERNAL_SAFETRANS_VS_V0_DELTAS.csv", index=False)
    net_deltas.to_csv(out / "FULL_NETWORK_VS_V0_DELTAS.csv", index=False)
    ext_net_deltas.to_csv(out / "EXTERNAL_NETWORK_VS_V0_DELTAS.csv", index=False)

    pass_rows = pass_settings(summary)
    ext_pass = pass_settings(ext_summary)
    # V3 uncertainty check: positive association between support uncertainty and error on at least one result row group.
    unc_ok = False
    if not results.empty and "uncertainty_error_spearman" in results:
        vals = results.loc[results["model"] == "V3", "uncertainty_error_spearman"].dropna()
        unc_ok = bool(len(vals) and vals.mean() > 0)

    safe_pass_rows = pass_settings(summary)
    ext_safe_pass = pass_settings(ext_summary)
    full_label = "Q2_READY_WITH_FOCUSED_CLAIMS" if (len(safe_pass_rows) >= 2 and len(ext_safe_pass) >= 1 and unc_ok) else "NOT_Q2_READY_STOP"
    if full_label.startswith("Q2"):
        reason = "Full run found at least two main hard OOD settings plus external directional support and uncertainty-error association."
    else:
        reason = f"Full criteria not met: main_pass_settings={len(safe_pass_rows)}, external_pass_settings={len(ext_safe_pass)}, uncertainty_ok={unc_ok}."
    (out / "full_status.json").write_text(json.dumps({"full_label": full_label, "reason": reason, "n_rows": len(results), "main_pass_settings": len(safe_pass_rows), "uncertainty_ok": unc_ok, "target_model": "PolicySafeTransPT + NetworkSafeTransPT"}, indent=2), encoding="utf-8")
    (out / "external_status.json").write_text(json.dumps({"n_rows": len(external), "external_pass_settings": len(ext_safe_pass), "attempted_external_datasets": external_selected["study_family"].astype(str).tolist()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

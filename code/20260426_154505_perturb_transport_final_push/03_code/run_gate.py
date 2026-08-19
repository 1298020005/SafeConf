from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table, select_gate_datasets
from evaluators import compare_v1_vs_v0, effect_metrics, summarize_results
from program_bank import ProgramBank
from transport_models import V0StrongBaseline, V1ProgramTransport


def choose_gate_splits(tasks: list[dict], max_per_type: int = 3) -> list[dict]:
    rows = feasible_splits(tasks)
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    chosen = []
    for split_type in ["leave_context", "heldout_perturbation"]:
        sub = df[df["split_type"] == split_type].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["n_test", "n_train"], ascending=False).head(max_per_type)
        chosen.extend(sub.to_dict("records"))
    return chosen


def average_task_metrics(tasks: list[dict], test_idx: np.ndarray, pred: np.ndarray, bank: ProgramBank) -> dict:
    y = np.stack([tasks[int(i)]["effect"] for i in test_idx], axis=0)
    true_z = bank.transform(y)
    pred_z = bank.transform(pred)
    rows = []
    for j in range(len(test_idx)):
        rows.append(effect_metrics(y[j], pred[j], true_z[j], pred_z[j]))
    out = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    out["n_tasks"] = int(len(test_idx))
    return out


def run_one_dataset(row: pd.Series, out_dir: Path, seeds: list[int], n_genes: int, n_programs: int) -> list[dict]:
    dataset = str(row["study_family"])
    path = Path(row["local_path"])
    all_rows = []
    for seed in seeds:
        tasks, genes, meta = build_effect_tasks(path, dataset, n_genes=n_genes, seed=seed)
        dataset_dir = out_dir / "artifacts" / dataset / f"seed_{seed}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "task_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (dataset_dir / "tasks.pkl").open("wb") as f:
            pickle.dump({"tasks": tasks, "genes": genes, "meta": meta}, f)
        splits = choose_gate_splits(tasks)
        pd.DataFrame(splits).to_csv(dataset_dir / "selected_splits.csv", index=False)
        for split in splits:
            train_idx, test_idx = materialize_split(tasks, split["split_type"], split["heldout"])
            train_mask = np.zeros(len(tasks), dtype=bool)
            train_mask[train_idx] = True
            train_effects = np.stack([tasks[int(i)]["effect"] for i in train_idx], axis=0)
            bank = ProgramBank(n_programs=n_programs, seed=seed).fit(train_effects)

            models = [
                V0StrongBaseline().fit(tasks, train_mask),
                V1ProgramTransport(ProgramBank(n_programs=n_programs, seed=seed), alpha=3.0).fit(tasks, train_mask),
            ]
            for model in models:
                if hasattr(model, "train_mask_for_predict"):
                    model.train_mask_for_predict = train_mask
                pred = model.predict(tasks, test_idx)
                m = average_task_metrics(tasks, test_idx, pred, bank)
                m.update(
                    {
                        "phase": "gate",
                        "dataset": dataset,
                        "source_path": str(path),
                        "split_type": split["split_type"],
                        "heldout": split["heldout"],
                        "seed": seed,
                        "model": model.name,
                        "n_train": int(len(train_idx)),
                    }
                )
                all_rows.append(m)
    return all_rows


def gate_decision(results: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    summary = summarize_results(results)
    deltas = compare_v1_vs_v0(summary)
    if deltas.empty:
        return "NOT_Q2_READY_STOP", "No comparable V1 vs V0 gate setting was produced.", deltas

    pass_rows = []
    for _, row in deltas.iterrows():
        effect_positive = (
            (row["rmse_delta"] < 0)
            and (
                row["top20_delta"] > 0
                or row["deg_precision_delta"] > 0
                or row["program_consistency_delta"] > 0
            )
        )
        corr_only = (row["pearson_delta"] > 0 or row["spearman_delta"] > 0) and not effect_positive
        if effect_positive and not corr_only:
            pass_rows.append(row)
    if pass_rows:
        return (
            "GATE_PASS",
            "V1 shows non-correlation-only effect-metric improvement over V0 on at least one hard split.",
            pd.DataFrame(pass_rows),
        )
    return "NOT_Q2_READY_STOP", "V1 did not beat the strong baseline on effect-based hard-split metrics.", deltas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--max-datasets", type=int, default=3)
    p.add_argument("--n-genes", type=int, default=1500)
    p.add_argument("--n-programs", type=int, default=64)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root / "05_gate_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    scan = read_scan_table(Path(args.atlas_root))
    selected = select_gate_datasets(scan, args.max_datasets)
    selected.to_csv(root / "02_data" / "SELECTED_GATE_DATASETS.csv", index=False)

    rows = []
    failures = []
    for _, row in selected.iterrows():
        try:
            rows.extend(run_one_dataset(row, out_dir, seeds, args.n_genes, args.n_programs))
        except Exception as exc:
            failures.append({"dataset": row.get("study_family"), "path": row.get("local_path"), "error": repr(exc)})
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "GATE_RESULTS.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "GATE_FAILURES_RAW.csv", index=False)
    summary = summarize_results(results)
    summary.to_csv(out_dir / "GATE_RESULT_SUMMARY_TABLE.csv", index=False)
    label, reason, evidence = gate_decision(results)
    evidence.to_csv(out_dir / "GATE_V1_VS_V0_DELTAS.csv", index=False)
    (out_dir / "GATE_PASS_FAIL.md").write_text(f"# Gate decision\n\nVerdict: `{label}`\n\nReason: {reason}\n", encoding="utf-8")
    (out_dir / "GATE_SUMMARY.md").write_text(
        "# Gate summary\n\n"
        + (summary.to_string(index=False) if not summary.empty else "No gate results were produced.")
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "GATE_FAILURE_ANALYSIS.md").write_text(
        "# Gate failure analysis\n\n"
        + ("Raw dataset failures are in `GATE_FAILURES_RAW.csv`.\n\n" if failures else "No raw dataset build failures.\n\n")
        + f"Decision reason: {reason}\n",
        encoding="utf-8",
    )
    status = {"gate_label": label, "reason": reason, "n_rows": int(len(results)), "n_failures": int(len(failures))}
    (out_dir / "gate_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

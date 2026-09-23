#!/usr/bin/env python3
"""Seal then evaluate E232 history-similarity robustness experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_e201_pretruth_task_base as original
import run_e225_raw_evidence_residual_ranker as scoring
import run_e231_history_quality as e231

SEEDS = (101, 211, 307)
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
CANDIDATE = "similarity_variance_shrink"
BASELINE = "equal_count_fixed"


def sha256(path: Path) -> str:
    return original.sha256_file(path)


def corrupt_indices(source: pd.DataFrame, fraction: float, seed: int) -> tuple[np.ndarray, list[dict]]:
    donor = np.arange(len(source))
    mapping: list[dict] = []
    target = str(source.target.iloc[0])
    for context, idx in source.groupby("context", sort=True).indices.items():
        idx = np.asarray(idx)
        rng = e231.rng_for(f"E232|{target}|{context}|{fraction}|{seed}")
        count = max(2, int(np.ceil(len(idx) * fraction)))
        selected = rng.choice(idx, size=min(count, len(idx)), replace=False)
        ordered = selected[rng.permutation(len(selected))]
        shifted = np.roll(ordered, 1)
        donor[ordered] = shifted
        for destination, source_row in zip(ordered, shifted):
            mapping.append({
                "target": target,
                "context": context,
                "fraction": fraction,
                "seed": seed,
                "condition": source.iloc[destination].condition,
                "donor_condition": source.iloc[source_row].condition,
            })
    return donor, mapping


def score_target(job: tuple[str, str, Path, Path, Path, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target, device, e201_docs, e231_docs, work, vectors = job
    import torch

    tasks = pd.read_csv(e201_docs / "tables/E201_PRETRUTH_RISK_FEATURES.csv")
    tasks = tasks[tasks.target.eq(target)].reset_index(drop=True)
    controls = np.load(vectors / "E201_CONTROL_CENTROIDS.npy").astype(float)
    predictions = np.load(vectors / "E201_FAMILY_CENTROIDS.npy").astype(float)
    all_tasks = pd.read_csv(e201_docs / "tables/E201_PRETRUTH_RISK_FEATURES.csv")
    selection = all_tasks.target.eq(target).to_numpy()
    pred_delta = predictions[selection] - controls[selection]

    source = pd.read_csv(e231_docs / "SOURCE_QUALITY.csv.gz")
    source = source[source.target.eq(target)].reset_index(drop=True)
    packed = np.load(work / target / "source_arrays.npz", allow_pickle=False)
    effects = packed["effects"]
    raw_meta = pd.read_csv(work / target / "source_metadata.csv")
    if not raw_meta[["target", "context", "condition", "task_id"]].reset_index(drop=True).equals(
        source[["target", "context", "condition", "task_id"]].reset_index(drop=True)
    ):
        raise RuntimeError(f"E231 source order changed for {target}")

    gpu = torch.as_tensor(effects, device=device)
    scenarios: list[tuple[str, pd.DataFrame, np.ndarray]] = [("clean", source.copy(), effects)]
    maps: list[dict] = []
    for fraction, seed in itertools.product(FRACTIONS, SEEDS):
        donor, records = corrupt_indices(source, fraction, seed)
        moved = gpu.index_select(0, torch.as_tensor(donor, device=device)).cpu().numpy()
        modified = source.copy()
        modified["estimated_mean_variance"] = source.iloc[donor].estimated_mean_variance.to_numpy()
        scenarios.append((f"corrupt_{int(fraction * 100):03d}_seed{seed}", modified, moved))
        maps.extend(records)

    contexts = sorted(source.context.unique())
    for drop in contexts:
        keep = source.context.ne(drop).to_numpy()
        scenarios.append((f"two_sources_drop_{drop}", source[keep].reset_index(drop=True), effects[keep]))
    for keep_context in contexts:
        keep = source.context.eq(keep_context).to_numpy()
        scenarios.append((f"one_source_keep_{keep_context}", source[keep].reset_index(drop=True), effects[keep]))

    frames = []
    for scenario, current_source, current_effects in scenarios:
        scores, _, _ = e231.build_scores(tasks, pred_delta, current_source, current_effects, "clean")
        frame = scores[["task_id", "target", "condition", "analysis_stratum"]].copy()
        frame["scenario"] = scenario
        frame["candidate"] = scores[CANDIDATE]
        frame["e230"] = scores[BASELINE]
        frame["magnitude"] = scores["magnitude"]
        frames.append(frame)
    audit = {
        "target": target,
        "device": device,
        "n_source_rows": int(len(source)),
        "contexts": contexts,
        "effects_sha256": sha256(work / target / "source_arrays.npz"),
        "e208_test_rows_read": 0,
    }
    del gpu
    torch.cuda.synchronize(torch.device(device))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(maps), audit


def score(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "PRE_EVALUATION_SEAL.json").exists():
        raise RuntimeError("Refuse to overwrite a sealed E232 run")
    jobs = [
        (target, f"cuda:{i % 2}", args.e201_docs, args.e231_docs, args.work, args.vectors)
        for i, target in enumerate(original.TARGETS)
    ]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(score_target, jobs))
    scores = pd.concat([x[0] for x in results], ignore_index=True)
    mappings = pd.concat([x[1] for x in results], ignore_index=True)
    score_file = args.output / "SCORES.csv.gz"
    map_file = args.output / "CORRUPTION_MAP.csv.gz"
    scores.to_csv(score_file, index=False, compression={"method": "gzip", "mtime": 0})
    mappings.to_csv(map_file, index=False, compression={"method": "gzip", "mtime": 0})
    seal = {
        "status": "SEALED_NOT_EVALUATED",
        "candidate": CANDIDATE,
        "coefficient_search": False,
        "score_sha256": sha256(score_file),
        "corruption_map_sha256": sha256(map_file),
        "score_rows": int(len(scores)),
        "unique_tasks": int(scores.task_id.nunique()),
        "scenarios": sorted(scores.scenario.unique().tolist()),
        "target_truth_fields": sorted(set(scores.columns) & {"family_rms_error", "mse", "error", "truth"}),
        "target_audits": [x[2] for x in results],
        "e208_test_rows_read": 0,
    }
    original.atomic_json(args.output / "PRE_EVALUATION_SEAL.json", seal)
    if seal["target_truth_fields"]:
        raise RuntimeError("Target truth leaked into sealed scores")
    print(json.dumps(seal, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    seal = json.loads((args.output / "PRE_EVALUATION_SEAL.json").read_text())
    score_file = args.output / "SCORES.csv.gz"
    if sha256(score_file) != seal["score_sha256"]:
        raise RuntimeError("Sealed E232 scores changed")
    scores = pd.read_csv(score_file)
    truth = pd.read_csv(args.truth).set_index("task_id").family_rms_error
    rows = []
    for (target, scenario), frame in scores.groupby(["target", "scenario"], sort=True):
        for scope in ("primary", "all"):
            part = frame[frame.analysis_stratum.eq("primary_ge30")] if scope == "primary" else frame
            y = truth.loc[part.task_id].to_numpy(float)
            for method in ("candidate", "e230", "magnitude"):
                s = part[method].to_numpy(float)
                rho = float(np.corrcoef(e231.rank(s), e231.rank(y))[0, 1])
                for budget in (0.1, 0.2, 0.3):
                    rows.append({
                        "target": target, "scenario": scenario, "scope": scope,
                        "method": method, "budget": budget, "n_tasks": len(part),
                        "spearman": rho, **scoring.metrics(s, y, budget),
                    })
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "RESULTS.csv", index=False)
    primary = results[(results.scope.eq("primary")) & (results.budget.eq(0.2))]
    pivot = primary.pivot(index=["target", "scenario"], columns="method", values="utility").reset_index()
    pivot.to_csv(args.output / "PRIMARY_UTILITY.csv", index=False)

    clean = pivot[pivot.scenario.eq("clean")].set_index("target")
    full = pivot[pivot.scenario.str.startswith("corrupt_100")].groupby("target").candidate.mean()
    delta_clean = clean.candidate - clean.magnitude
    delta_specific = clean.candidate - full
    fractions = []
    for fraction in (0, 10, 25, 50, 100):
        if fraction == 0:
            value = float(clean.candidate.mean())
        else:
            value = float(pivot[pivot.scenario.str.startswith(f"corrupt_{fraction:03d}")].candidate.mean())
        fractions.append((fraction, value))
    slope = float(np.polyfit([x[0] for x in fractions], [x[1] for x in fractions], 1)[0])
    two = float(pivot[pivot.scenario.str.startswith("two_sources")].candidate.mean())
    one = float(pivot[pivot.scenario.str.startswith("one_source")].candidate.mean())
    summary = {
        "status": "COMPLETE",
        "clean_candidate_minus_magnitude": float(delta_clean.mean()),
        "clean_positive_targets": int((delta_clean > 0).sum()),
        "clean_minus_full_corruption": float(delta_specific.mean()),
        "specificity_positive_targets": int((delta_specific > 0).sum()),
        "corruption_trend_utility_per_percent": slope,
        "two_source_minus_one_source": two - one,
        "registered_mechanism_gate": "PASS" if (
            delta_clean.mean() >= 0.01 and (delta_clean > 0).sum() >= 3
            and delta_specific.mean() > 0 and (delta_specific > 0).sum() >= 3
            and slope < 0 and two > one
        ) else "NOT_SUPPORTED",
        "e208_test_rows_read": 0,
    }
    original.atomic_json(args.output / "RUN_STATUS.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "evaluate"))
    parser.add_argument("--e201-docs", type=Path, required=True)
    parser.add_argument("--e231-docs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()
    if args.stage == "score":
        score(args)
    else:
        if args.truth is None:
            parser.error("evaluate requires --truth")
        evaluate(args)


if __name__ == "__main__":
    main()

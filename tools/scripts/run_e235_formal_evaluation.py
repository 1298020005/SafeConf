#!/usr/bin/env python3
"""Evaluate the frozen E235 M+H hypothesis once, after dual-remote authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from run_e208_formal_evaluation import extract_truth


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_H5_SHA256 = "5d876c0fa5770dc632ad8ed8b211ad7aef00ccc93481f6ac029d439a0d7cd4d9"
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
METHODS = ("rank_M", "score_M_plus_H", "score_M_plus_D", "score_M_plus_G",
           "score_M_plus_N", "score_M_plus_C", "score_original_five_80_20",
           "score_random_fixed")
BUDGETS = (0.10, 0.20, 0.30)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260923


class EvaluationFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=not binary)


def checked_repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise EvaluationFailure(f"invalid repository path: {value}")
    resolved = (repo / path).resolve()
    if not resolved.is_relative_to(repo.resolve()):
        raise EvaluationFailure(f"repository path escaped root: {value}")
    return resolved


def verify_authorization(repo: Path, path: Path, score_path: Path, status_path: Path) -> dict:
    """Perform every remote/seal check before any test-expression file access."""
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise EvaluationFailure("worktree is not clean at truth authorization gate")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    score_status = json.loads(status_path.read_text(encoding="utf-8"))
    if (authorization.get("experiment") != "E235_jiang24_fixed_history_dispersion"
            or authorization.get("stage") != "TEST_TRUTH_AUTHORIZATION"
            or authorization.get("status") != "AUTHORIZED_AWAITING_REMOTE_PERSISTENCE"
            or authorization.get("test_perturbed_expression_rows_read_before_authorization") != 0
            or score_status.get("status") != "SCORES_READY_AWAITING_REMOTE_SEAL"
            or score_status.get("test_perturbed_expression_rows_read") != 0):
        raise EvaluationFailure("authorization or score seal status failed")
    pretruth = str(authorization.get("pretruth_git_commit", ""))
    if len(pretruth) != 40:
        raise EvaluationFailure("invalid pretruth commit")
    head = git(repo, "rev-parse", "HEAD").strip()
    auth_relative = path.resolve().relative_to(repo.resolve()).as_posix()
    committed_auth = git(repo, "log", "-1", "--format=%H", "--", auth_relative).strip()
    if head != committed_auth:
        raise EvaluationFailure("authorization must be the current committed HEAD")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", pretruth, head],
                      check=False).returncode:
        raise EvaluationFailure("authorization does not descend from pretruth seal")
    if hashlib.sha256(git(repo, "show", f"{head}:{auth_relative}", binary=True)).hexdigest() != sha256(path):
        raise EvaluationFailure("authorization changed after commit")
    ref = str(authorization.get("remote_ref", ""))
    if not ref.startswith("refs/heads/"):
        raise EvaluationFailure("invalid authorization branch ref")
    for remote in ("origin", "github"):
        values = git(repo, "ls-remote", remote, ref).strip().split()
        if not values or values[0] != head:
            raise EvaluationFailure(f"{remote} has not received authorization HEAD")
    for label, actual_path in (("score_table", score_path), ("score_status", status_path)):
        repo_path = str(authorization.get(f"{label}_repo_path", ""))
        if checked_repo_path(repo, repo_path) != actual_path.resolve():
            raise EvaluationFailure(f"{label} path differs from authorized path")
    for label in ("score_table", "score_status", "protocol", "scorer", "evaluator"):
        repo_path = str(authorization.get(f"{label}_repo_path", ""))
        local_path = checked_repo_path(repo, repo_path)
        expected = str(authorization.get(f"{label}_sha256", ""))
        if len(expected) != 64 or sha256(local_path) != expected:
            raise EvaluationFailure(f"{label} local hash differs from seal")
        sealed = git(repo, "show", f"{pretruth}:{repo_path}", binary=True)
        if hashlib.sha256(sealed).hexdigest() != expected:
            raise EvaluationFailure(f"{label} differs from pretruth commit")
    if score_status.get("score_sha256") != sha256(score_path):
        raise EvaluationFailure("score table differs from its no-truth status")
    return authorization


def stable_ties(task_ids: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(hashlib.sha256(f"E235_TIE_V1\0{task}\0{int(occ)}".encode()).hexdigest()[:16], 16)
        for task, occ in zip(task_ids, occurrences, strict=True)
    ], dtype=np.uint64)


def state_metrics(block: pd.DataFrame, method: str, budget: float,
                  occurrences: np.ndarray | None = None) -> dict[str, float]:
    n = len(block)
    if n < 5:
        return {"spearman": float("nan"), "utility": float("nan"), "capture": float("nan")}
    values = block.full_gene_rmse.to_numpy(float)
    scores = block[method].to_numpy(float)
    if occurrences is None:
        occurrences = np.zeros(n, dtype=np.int64)
    ties = stable_ties(block.task_id.to_numpy(str), occurrences)
    selected_count = math.ceil(budget * n)
    selected = np.lexsort((ties, -scores))[:selected_count]
    oracle = np.lexsort((ties, -values))[:selected_count]
    baseline = float(values.mean())
    denominator = float(values[oracle].mean()) - baseline
    xr, yr = rankdata(scores, method="average"), rankdata(values, method="average")
    rho = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 0 and np.std(yr) > 0 else float("nan")
    return {
        "spearman": rho,
        "utility": ((float(values[selected].mean()) - baseline) / denominator
                    if denominator > 1e-15 else float("nan")),
        "capture": float(values[selected].sum() / values.sum()),
    }


def macro_metrics(frame: pd.DataFrame, method: str, budget: float,
                  occurrences: np.ndarray | None = None) -> dict[str, float]:
    blocks = []
    for _, positions in frame.groupby(["cell_type", "treatment"], sort=True).indices.items():
        block = frame.iloc[positions]
        block_occurrences = None if occurrences is None else occurrences[positions]
        blocks.append(state_metrics(block, method, budget, block_occurrences))
    if len(blocks) != 12:
        raise EvaluationFailure("E235 lost one or more of its 12 states")
    return {key: float(np.mean([row[key] for row in blocks])) for key in ("spearman", "utility", "capture")}


def load_predictions(root: Path, score_status: dict, scores: pd.DataFrame) -> np.ndarray:
    records = score_status.get("prediction_inputs", [])
    if len(records) != 4 or [item.get("seed") for item in records] != [1, 2, 3, 4]:
        raise EvaluationFailure("score seal has no complete four-seed prediction manifest")
    members = []
    for record in records:
        seed = int(record["seed"])
        directory = root / f"seed_{seed}"
        prediction_path = directory / "E208_PREDICTION_CENTROIDS.npy"
        tasks_path = directory / "E208_PREDICTION_TASKS.csv"
        status_path = directory / "E208_PREDICTION_STATUS.json"
        if (sha256(prediction_path) != record["prediction_sha256"]
                or sha256(status_path) != record["status_sha256"]):
            raise EvaluationFailure(f"seed {seed} prediction changed after score seal")
        predictor_status = json.loads(status_path.read_text(encoding="utf-8"))
        if (predictor_status.get("status") != "PASS"
                or predictor_status.get("test_perturbed_expression_rows_read") != 0
                or predictor_status.get("checkpoint_sha256") != record["checkpoint_sha256"]):
            raise EvaluationFailure(f"seed {seed} predictor status changed")
        tasks = pd.read_csv(tasks_path)
        if not tasks.task_id.equals(scores.task_id):
            raise EvaluationFailure(f"seed {seed} tasks differ from sealed scores")
        prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
        if prediction.shape != (224, 15473) or not np.isfinite(prediction).all():
            raise EvaluationFailure(f"seed {seed} prediction shape/finite gate failed")
        members.append(prediction)
    return np.mean(members, axis=0)


def atomic_csv(path: Path, frame: pd.DataFrame, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if gzip else None)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "h5ad", "split", "prediction-root", "score-table", "score-status",
                 "authorization", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise EvaluationFailure("formal output directory is not empty; refusing a second run")
    authorization = verify_authorization(repo, args.authorization.resolve(),
                                         args.score_table.resolve(), args.score_status.resolve())
    score_status = json.loads(args.score_status.read_text(encoding="utf-8"))
    scores = pd.read_csv(args.score_table)
    if (len(scores) != 224 or scores.task_id.nunique() != 224
            or scores.condition.nunique() != 53
            or len(scores[["cell_type", "treatment"]].drop_duplicates()) != 12
            or not set(METHODS).issubset(scores.columns)
            or not np.isfinite(scores[list(METHODS)].to_numpy(float)).all()
            or scores.test_perturbed_expression_rows_read.ne(0).any()):
        raise EvaluationFailure("sealed score table is incomplete or contains invalid values")
    predictions = load_predictions(args.prediction_root, score_status, scores)
    if (args.h5ad.stat().st_size != EXPECTED_H5_BYTES
            or sha256(args.h5ad) != EXPECTED_H5_SHA256
            or sha256(args.split) != EXPECTED_SPLIT_SHA256):
        raise EvaluationFailure("frozen Jiang24 H5/split integrity gate failed")
    truth, counts = extract_truth(args.h5ad, args.split, scores)
    if int(counts.sum()) != 214_901:
        raise EvaluationFailure("Jiang24 test truth count differs from fixed manifest")
    result = scores.copy()
    result["n_truth_cells"] = counts
    result["full_gene_rmse"] = np.sqrt(np.mean((predictions - truth) ** 2, axis=1))
    if not np.isfinite(result.full_gene_rmse).all():
        raise EvaluationFailure("nonfinite primary endpoint")

    state_rows = []
    for (cell, treatment), block in result.groupby(["cell_type", "treatment"], sort=True):
        for method in METHODS:
            for budget in BUDGETS:
                stats = state_metrics(block, method, budget)
                state_rows.append({"cell_type": cell, "treatment": treatment, "n_tasks": len(block),
                                   "method": method, "budget": budget, **stats})
    states = pd.DataFrame(state_rows)
    summary_rows = []
    for method in METHODS:
        for budget in BUDGETS:
            block = states.loc[states.method.eq(method) & states.budget.eq(budget)]
            summary_rows.append({"method": method, "budget": budget, "n_states": len(block),
                                 **{key: float(block[key].mean()) for key in ("spearman", "utility", "capture")}})
    summary = pd.DataFrame(summary_rows)
    cell_lines = (states.groupby(["cell_type", "method", "budget"], sort=True)
                  [["spearman", "utility", "capture"]].mean().reset_index())

    clusters = sorted(result.condition.astype(str).unique())
    task_genes = result.condition.astype(str).to_numpy()
    indexes_by_gene = [np.flatnonzero(task_genes == gene) for gene in clusters]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for draw in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(clusters), len(clusters))
        positions = np.concatenate([indexes_by_gene[int(index)] for index in chosen])
        occurrences = np.concatenate([
            np.full(len(indexes_by_gene[int(index)]), occurrence, dtype=np.int64)
            for occurrence, index in enumerate(chosen)
        ])
        frame = result.iloc[positions].reset_index(drop=True)
        base = macro_metrics(frame, "rank_M", 0.20, occurrences)
        candidate = macro_metrics(frame, "score_M_plus_H", 0.20, occurrences)
        bootstrap_rows.append({"draw": draw, "delta_utility_20": candidate["utility"] - base["utility"],
                               "delta_spearman": candidate["spearman"] - base["spearman"],
                               "delta_capture_20": candidate["capture"] - base["capture"]})
    bootstrap = pd.DataFrame(bootstrap_rows)
    valid = bootstrap.dropna()
    if len(valid) < 1900:
        raise EvaluationFailure(f"too few valid gene-cluster bootstrap draws: {len(valid)}")
    ci = {key: [float(valid[key].quantile(0.025)), float(valid[key].quantile(0.975))]
          for key in ("delta_utility_20", "delta_spearman", "delta_capture_20")}
    main_summary = summary.loc[summary.budget.eq(0.20)].set_index("method")
    delta = float(main_summary.loc["score_M_plus_H", "utility"] - main_summary.loc["rank_M", "utility"])
    utility_states = states.loc[states.budget.eq(0.20) & states.method.isin(("rank_M", "score_M_plus_H"))]
    paired = utility_states.pivot(index=["cell_type", "treatment"], columns="method", values="utility")
    positive_states = int((paired.score_M_plus_H - paired.rank_M > 0).sum())
    gates = {"macro_utility_delta_at_least_0_02": delta >= 0.02,
             "gene_cluster_ci_lower_gt_zero": ci["delta_utility_20"][0] > 0,
             "at_least_9_of_12_states_positive": positive_states >= 9}
    decision = "SUPPORTED_IN_FIXED_SCOPE" if all(gates.values()) else "NOT_SUPPORTED"

    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "E235_FORMAL_TASK_RESULTS.csv"
    state_path = output / "E235_FORMAL_STATE_RESULTS.csv"
    summary_path = output / "E235_FORMAL_SUMMARY.csv"
    cells_path = output / "E235_FORMAL_CELL_LINE_RESULTS.csv"
    bootstrap_path = output / "E235_GENE_CLUSTER_BOOTSTRAP.csv.gz"
    atomic_csv(task_path, result)
    atomic_csv(state_path, states)
    atomic_csv(summary_path, summary)
    atomic_csv(cells_path, cell_lines)
    atomic_csv(bootstrap_path, bootstrap, gzip=True)
    formal = {
        "experiment": "E235_jiang24_fixed_history_dispersion",
        "status": "COMPLETE", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": decision, "gates": gates, "n_tasks": len(result), "n_target_genes": len(clusters),
        "n_states": len(paired), "positive_utility_states": positive_states,
        "n_truth_cells_read": int(counts.sum()), "primary_endpoint": "full_gene_rmse",
        "primary_budget": 0.20, "primary_delta_utility_20": delta,
        "gene_cluster_bootstrap_ci_95": ci, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": N_BOOTSTRAP, "bootstrap_valid_draws": len(valid),
        "authorization_sha256": sha256(args.authorization),
        "pretruth_git_commit": authorization["pretruth_git_commit"],
        "score_table_sha256": sha256(args.score_table),
        "output_sha256": {path.name: sha256(path) for path in
                          (task_path, state_path, summary_path, cells_path, bootstrap_path)},
        "scope_warning": "All 224 predeclared gene-perturbation tasks retained; no chemical/general-model claim.",
    }
    status_path = output / "E235_FORMAL_EVALUATION_STATUS.json"
    temporary = status_path.with_name(f".{status_path.name}.tmp")
    temporary.write_text(json.dumps(formal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, status_path)
    print(json.dumps(formal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

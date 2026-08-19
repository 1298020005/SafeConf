#!/usr/bin/env python3
"""Build the sealed E199 risk features without reading target expression.

The program opens only model predictions, matched controls, the official
general baseline, the official train split and the public STRING/GO graphs.
It refuses to run unless the sealed inputs and both Git remotes match the
frozen release.  Target-expression files are intentionally absent from this
source file and from every input constant below.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E199_txpert_public_k562_20260802"
FREEZE = OUT / "PRETRUTH_FEATURE_FREEZE.md"
RELEASE = OUT / "pretruth_release"
TABLES = RELEASE / "tables"
REPORTS = RELEASE / "reports"

TXPERT = Path("/home/yyf/archive/external/TxPert")
DATA = Path("/home/yyf/data/txpert_official_20260802")
PRED = DATA / "e199/predictions"
SPLIT = DATA / "cache/K562_single_cell_line/splits/train_test_split.pkl"
PERT_SET = TXPERT / "data/gears_gene_set.csv"
STRING_GRAPH = TXPERT / "data/graphs/string/v11.5.parquet"
GO_GRAPH = TXPERT / "data/graphs/go/go_top_50.csv"

EXPECTED = {
    PRED / "gat/test_predictions.h5ad": (
        1_084_480_564,
        "5e2d2bdfc67a368c3d2fd0f987c30f4c179e32158aea3f8eca9286557bd379a5",
    ),
    PRED / "exphormer/test_predictions.h5ad": (
        1_084_480_564,
        "ff475ed22837d0bb5fda92744e49019614e30e78a2465d8978b2601602f493db",
    ),
    PRED / "exphormer_mg/test_predictions.h5ad": (
        1_084_480_564,
        "764d6a864a99127904bd29f8d9e185ba2797681d19a45d8d334980feb5c62df0",
    ),
    PRED / "gat/test_controls.h5ad": (
        1_084_480_564,
        "3ae24c04804b33eb495c27ae75f4a3fa3930afdcdecf61c3f9cd3b272e9837a3",
    ),
    PRED / "general_baseline/test_predictions.h5ad": (
        771_973_828,
        "49e81f1743d29b332859fcb36c4b60dccb11d00ff8b141fffdb5e30682cc16a8",
    ),
    SPLIT: (
        14_254,
        "d05a0fe13bf24a4892a54ed8840d67c0a8322456e8a7ba50625b3ab0dd6f464c",
    ),
    PERT_SET: (
        110_387,
        "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d",
    ),
    STRING_GRAPH: (
        38_828_749,
        "55d312f8d6186078eb00a7caf108ef731cd27aa08a5b48de59cd0333f0206404",
    ),
    GO_GRAPH: (
        12_814_602,
        "ad469c852ba9b8b5489749c7987467687aa566598fc0706fd7232aa27edce27b",
    ),
}

MODEL_KEYS = ("gat", "exphormer", "exphormer_mg")
N_CELLS = 38_475
N_GENES = 5_000
N_TASKS = 272
OBS_HASH = "d79ed6c6fb7b0a8897926d8b50a01a5411ff4ed38870f2f2b2c1e6e26d31d22a"
VAR_HASH = "9dab8faec41a298cc2abedf9597385645f32803cba3862a100359543f0e9f9c6"
PERT_HASH = "024d3d3c85d9a0769a70f8c72ea14273ea43e698417054e3c683e13e50ba9920"
BATCH_HASH = "a7b0c73f09b10310748dacc0f863a9adf157b8c9ceb7f3ac2a690c9568f1b209"


class PretruthFailure(RuntimeError):
    """Fail-closed E199 feature-release error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_hash(values) -> str:
    payload = "\n".join(map(str, values)).encode()
    return hashlib.sha256(payload).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def remote_tip(remote: str, branch: str) -> str:
    line = git_output("ls-remote", remote, f"refs/heads/{branch}")
    if not line:
        raise PretruthFailure(f"missing remote branch: {remote}/{branch}")
    return line.split()[0]


def tracked_clean(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{rel}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    clean = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel],
        check=False,
    ).returncode == 0
    staged_clean = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "HEAD", "--", rel],
        check=False,
    ).returncode == 0
    return tracked and clean and staged_clean


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    for base, prefix in (
        (ROOT.resolve(), ""),
        (Path("/home/yyf/data").resolve(), "DATA/"),
        (Path("/home/yyf/archive/external").resolve(), "EXTERNAL/"),
    ):
        try:
            return prefix + resolved.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.name


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def verify_inputs() -> pd.DataFrame:
    rows = []
    for path, (expected_bytes, expected_sha) in EXPECTED.items():
        if not path.is_file():
            raise PretruthFailure(f"missing sealed input: {path}")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise PretruthFailure(f"sealed input mismatch: {path}")
        rows.append(
            {
                "path": logical_path(path),
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def verify_git_release() -> str:
    if not tracked_clean(SCRIPT) or not tracked_clean(FREEZE):
        raise PretruthFailure("feature runner/freeze is not tracked and clean")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if not branch:
        raise PretruthFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise PretruthFailure(f"{remote}/{branch} does not match local HEAD")
    return head


def verify_adata(handle: ad.AnnData, label: str) -> None:
    if handle.shape != (N_CELLS, N_GENES):
        raise PretruthFailure(f"{label} shape changed: {handle.shape}")
    checks = {
        "obs": (order_hash(handle.obs_names), OBS_HASH),
        "var": (order_hash(handle.var_names), VAR_HASH),
        "perturbation": (
            order_hash(handle.obs["pert_cond_names"].astype(str)),
            PERT_HASH,
        ),
        "batch": (
            order_hash(handle.obs["experimental_batches"].astype(str)),
            BATCH_HASH,
        ),
    }
    failed = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if failed:
        raise PretruthFailure(f"{label} alignment changed: {failed}")


def condition_from_label(label: str) -> str:
    prefix = "K562_"
    suffix = "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise PretruthFailure(f"unexpected condition label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    parts = condition.split("+")
    if len(parts) != 2 or parts[1] != "ctrl":
        raise PretruthFailure(f"not a single-gene condition: {label}")
    return condition


def load_processed_graph(path: Path, universe: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "source" in frame.columns:
        frame = frame.rename(columns={"source": "regulator"})
    if "importance" in frame.columns:
        frame = frame.rename(columns={"importance": "weight"})
    needed = {"regulator", "target", "weight"}
    if not needed.issubset(frame.columns):
        raise PretruthFailure(f"graph columns changed: {path}")
    frame = frame.loc[
        frame.regulator.astype(str).isin(universe)
        & frame.target.astype(str).isin(universe),
        ["regulator", "target", "weight"],
    ].copy()
    # This mirrors TxPert's frozen ``mode: top_20`` after reduce2perts.
    # Keep the grouping column explicitly.  Pandas 2.2's ``include_groups=False``
    # removes ``target`` from each returned block; a plain group loop preserves
    # TxPert's group-wise ``nlargest`` semantics without relying on that changing
    # ``GroupBy.apply`` default.
    top_blocks = [
        block.nlargest(20, ["weight"])
        for _, block in frame.groupby("target", sort=True, observed=True)
    ]
    frame = pd.concat(top_blocks, ignore_index=True)
    return frame


def training_neighbor_counts(
    graph: pd.DataFrame, train_genes: set[str], test_genes: list[str]
) -> dict[str, int]:
    incoming = graph.groupby("target")["regulator"].agg(lambda x: set(map(str, x)))
    outgoing = graph.groupby("regulator")["target"].agg(lambda x: set(map(str, x)))
    counts = {}
    for gene in test_genes:
        neighbors = set(incoming.get(gene, set())) | set(outgoing.get(gene, set()))
        neighbors.discard(gene)
        counts[gene] = len(neighbors & train_genes)
    return counts


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def centroid(handle: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(handle.X[indices], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise PretruthFailure("non-finite or malformed prediction block")
    return values.mean(axis=0)


def build_features() -> pd.DataFrame:
    split = joblib.load(SPLIT)
    if set(split) != {"train", "val", "test"} or len(split["test"]) != N_TASKS:
        raise PretruthFailure("official split structure changed")
    train_conditions = list(map(str, split["train"]))
    test_conditions = list(map(str, split["test"]))
    train_genes = {condition.split("+")[0] for condition in train_conditions}
    test_genes = [condition.split("+")[0] for condition in test_conditions]
    if len(train_genes) != len(train_conditions) or len(set(test_genes)) != N_TASKS:
        raise PretruthFailure("E199 expects unique single-gene train/test perturbations")
    if train_genes & set(test_genes):
        raise PretruthFailure("historical-support contract failed: test gene seen in train")

    universe = set(pd.read_csv(PERT_SET, index_col=0)["0"].astype(str))
    string_graph = load_processed_graph(STRING_GRAPH, universe)
    go_graph = load_processed_graph(GO_GRAPH, universe)
    string_counts = training_neighbor_counts(string_graph, train_genes, test_genes)
    go_counts = training_neighbor_counts(go_graph, train_genes, test_genes)

    paths = {
        key: PRED / f"{key}/test_predictions.h5ad" for key in MODEL_KEYS
    }
    paths["control"] = PRED / "gat/test_controls.h5ad"
    paths["general_baseline"] = PRED / "general_baseline/test_predictions.h5ad"
    handles = {key: ad.read_h5ad(path, backed="r") for key, path in paths.items()}
    try:
        for key, handle in handles.items():
            verify_adata(handle, key)
        labels = handles["gat"].obs["pert_cond_names"].astype(str).to_numpy()
        condition_to_label = {}
        for label in pd.unique(labels):
            condition = condition_from_label(str(label))
            if condition in condition_to_label:
                raise PretruthFailure(f"duplicate condition label: {condition}")
            condition_to_label[condition] = str(label)
        if set(condition_to_label) != set(test_conditions):
            raise PretruthFailure("prediction conditions do not equal official test split")

        rows = []
        for condition, gene in zip(test_conditions, test_genes):
            label = condition_to_label[condition]
            indices = np.flatnonzero(labels == label)
            members = np.stack(
                [centroid(handles[key], indices) for key in MODEL_KEYS], axis=0
            )
            family = members.mean(axis=0)
            control = centroid(handles["control"], indices)
            baseline = centroid(handles["general_baseline"], indices)
            diversity_mse = float(np.mean((members - family[None, :]) ** 2))
            pairwise = [
                rmse(members[left], members[right])
                for left in range(len(MODEL_KEYS))
                for right in range(left + 1, len(MODEL_KEYS))
            ]
            row = {
                "task_id": condition,
                "condition_label": label,
                "gene": gene,
                "cell_line": "K562",
                "n_prediction_cells": len(indices),
                "family_diversity": diversity_mse,
                "diversity_lower_bound": math.sqrt(max(diversity_mse, 0.0)),
                "predicted_magnitude": rmse(family, control),
                "model_baseline_gap": rmse(family, baseline),
                "family_pairwise_rmse_mean": float(np.mean(pairwise)),
                "family_pairwise_rmse_max": float(np.max(pairwise)),
                "string_train_neighbor_count": string_counts[gene],
                "go_train_neighbor_count": go_counts[gene],
                "graph_isolated": (
                    string_counts[gene] == 0 and go_counts[gene] == 0
                ),
                "historical_support": 0,
                "context_similarity": "NOT_APPLICABLE_SINGLE_CONTEXT",
            }
            numeric = [
                diversity_mse,
                row["diversity_lower_bound"],
                row["predicted_magnitude"],
                row["model_baseline_gap"],
                row["family_pairwise_rmse_mean"],
                row["family_pairwise_rmse_max"],
            ]
            if not np.isfinite(numeric).all():
                raise PretruthFailure(f"non-finite risk feature: {condition}")
            rows.append(row)
    finally:
        for handle in handles.values():
            handle.file.close()
    frame = pd.DataFrame(rows)
    if len(frame) != N_TASKS or frame.task_id.nunique() != N_TASKS:
        raise PretruthFailure("feature table cardinality changed")
    if frame.historical_support.ne(0).any():
        raise PretruthFailure("test perturbation leaked into historical support")
    return frame


def markdown_summary(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# E199 未读取目标表达的风险特征",
            "",
            f"- 任务数：{len(frame)}；训练中见过的目标扰动：0。",
            f"- 两张图均孤立的测试基因：{int(frame.graph_isolated.sum())}。",
            f"- 每个任务细胞数：{int(frame.n_prediction_cells.min())}–"
            f"{int(frame.n_prediction_cells.max())}；主分析门槛下可用 "
            f"{int(frame.n_prediction_cells.ge(30).sum())} 个。",
            "- 特征只使用三个模型输出、匹配对照、官方基线、训练划分和公开图。",
            "- K562 只有一个背景，本实验不产生 context similarity 证据。",
            "",
        ]
    )


def main() -> None:
    if RELEASE.exists():
        raise PretruthFailure("pretruth release is append-only and already exists")
    head = verify_git_release()
    input_hashes = verify_inputs()
    frame = build_features()
    write_csv(TABLES / "E199_PRETRUTH_INPUT_HASHES.csv", input_hashes)
    write_csv(TABLES / "E199_PRETRUTH_FEATURES.csv", frame)
    write_text(REPORTS / "E199_PRETRUTH_REPORT.md", markdown_summary(frame))
    status = {
        "experiment": "E199_txpert_public_k562",
        "stage": "PRETRUTH_FEATURE_RELEASE",
        "generated_at": now(),
        "status": "PASS",
        "git_head": head,
        "n_tasks": len(frame),
        "n_primary_tasks_ge30": int(frame.n_prediction_cells.ge(30).sum()),
        "n_sensitivity_tasks_10_29": int(
            frame.n_prediction_cells.between(10, 29).sum()
        ),
        "historical_support_nonzero": int(frame.historical_support.ne(0).sum()),
        "graph_isolated_tasks": int(frame.graph_isolated.sum()),
        "target_expression_files_opened": 0,
    }
    write_json(RELEASE / "E199_PRETRUTH_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

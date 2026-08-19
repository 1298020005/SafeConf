#!/usr/bin/env python3
"""E195: native GEARS-UQ on two frozen Norman task panels.

The runner has three explicit phases:

1. ``prepare`` verifies the frozen inputs and creates panel-isolated GEARS
   cache roots;
2. ``train`` runs six real GEARS-UQ fits with a prediction-only score lock
   before the test truth is read;
3. ``analyze`` evaluates native log-variance, seed disagreement and magnitude,
   then places GEARS-UQ beside the already frozen GEARS-scGPT and PRESCRIBE
   predictor-uncertainty systems.

Raw model states and arrays stay local under ``panels/*/raw_gears``.  The
tracked release contains the contract, code, task-level tables, figures,
hashes and run record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
OUT = (
    ROOT
    / "docs"
    / "实验结果"
    / "E195_native_gears_uq_norman_p1p2_20260730"
)
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
PYTHON = Path("/home/yyf/.conda/envs/scgpt_env/bin/python")
BASE_DATA_ROOT = Path("/home/yyf/data/gears_formal_baselines_v2")
BASE_DATASET = BASE_DATA_ROOT / "norman_local_atlas"
ISOLATED_ROOT = Path("/home/yyf/data/safeconf_e195_gears_strict_v1")
SEEDS = (11, 22, 33)
N_BOOT = 5000
BOOTSTRAP_SEED = 202607195

PINNED_LARGE_ASSETS = {
    BASE_DATASET / "perturb_processed.h5ad": (
        "039c3bc41e30a4575b7857a570c5db89d3a11e7503c1c51c36c8f17d80da1761"
    ),
    BASE_DATASET / "data_pyg" / "cell_graphs.pkl": (
        "c3f7c75960a2fa5151e4f27fdeb89dd554162f1e1704af1743c3194108f8e6c1"
    ),
    BASE_DATASET / "go.csv": (
        "1f7740c0db2971b5c079e54f61c3224368fd115606a235539da57a157363d41d"
    ),
    BASE_DATA_ROOT / "gene2go.pkl": (
        "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771"
    ),
}

TRAINING_CONTRACT = {
    "epochs": 10,
    "hidden_size": 48,
    "decoder_hidden_size": 16,
    "num_similar_genes": 10,
    "batch_size": 32,
    "requested_test_batch_size": 64,
    "max_cells_per_condition": 32,
    "lr": 0.001,
    "weight_decay": 0.0005,
    "coexpress_threshold": 0.4,
    "direction_lambda": 0.1,
    "uncertainty": True,
    "fixed_test_deterministic_val": True,
    "train_gene_set_size": 0.75,
    "max_genes": 6000,
}

PANELS: dict[str, dict[str, Any]] = {
    "P1": {
        "label": "Norman_P1",
        "manifest": ROOT
        / "docs"
        / "实验结果"
        / "E66_norman_gears_fixed_panel_formal_20260711"
        / "tables"
        / "E60_FIXED_TEST_PERTURBATIONS.csv",
        "manifest_sha256": "f1162e8378fa186153b393b9e3e2a7d5a99189f44e7e0afc6f079d76677e565a",
        "sampling_seed": 20260766,
        "pair": ROOT
        / "docs"
        / "实验结果"
        / "E67_norman_scgpt_formal_fixed_panel_20260711"
        / "tables"
        / "E67_TASK_RISK_TABLE.csv",
        "pair_sha256": "d11c88c53d799948b9ebf6d229fd24ea52bfb3ab51e0bda5ca1e3e4ed8b2f74b",
    },
    "P2": {
        "label": "Norman_P2",
        "manifest": ROOT
        / "docs"
        / "实验结果"
        / "E75b_norman_gears_panel2_20260711"
        / "tables"
        / "E60_FIXED_TEST_PERTURBATIONS.csv",
        "manifest_sha256": "36597e0cf025948598bc2195e34e4dd87517be38e7dc3c35bcc9fc05c42df8db",
        "sampling_seed": 202607752,
        "pair": ROOT
        / "docs"
        / "实验结果"
        / "E76b_norman_scgpt_panel2_20260711"
        / "tables"
        / "E76b_TASK_RISK_TABLE.csv",
        "pair_sha256": "5d1d28b39c5f617eceaa5c30fc93ae9e98585466b17b3f5a2c9004607f1bce71",
    },
}

PRESCRIBE_PAPER = (
    ROOT
    / "docs"
    / "实验结果"
    / "E145_prescribe_paper_endpoint_20260714"
    / "tables"
    / "E145_TASK_METRICS.csv"
)
PRESCRIBE_PAPER_SHA256 = (
    "dfc7bcb138aff82e0921158b34dc3ffe4b23b7b02d70ec0afc811fa9cd9a7eb6"
)
PRESCRIBE_RMSE = (
    ROOT
    / "docs"
    / "实验结果"
    / "E96_prescribe_native_comparison_20260713"
    / "tables"
    / "E96_PRESCRIBE_TASKS.csv"
)
PRESCRIBE_RMSE_SHA256 = (
    "423de938a4aaaf445187900f958310075322afdf42dd2649018f37100a2c4170"
)
GEARS_CLI = (
    CODE_ROOT / "safetrans_confidence" / "cli" / "run_gears_prediction_records.py"
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def ensure_dirs() -> None:
    for path in (OUT, TABLES, FIGURES, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        (OUT / "panels" / panel / "raw_gears").mkdir(parents=True, exist_ok=True)


def verify_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"input hash mismatch for {path}: expected {expected}, got {observed}"
        )
    return {
        "path": display_path(path),
        "bytes": int(path.stat().st_size),
        "sha256": observed,
        "expected_sha256": expected,
        "hash_match": True,
        "pin_scope": "frozen_external_input",
    }


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve_display_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
    )
    relevant = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--",
            str(GEARS_CLI.relative_to(ROOT)),
            str(Path(__file__).resolve().relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
    )
    return {
        "git_head": head,
        "working_tree_status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "working_tree_dirty_entry_count": len(
            [line for line in status.splitlines() if line.strip()]
        ),
        "e195_code_paths_clean": not bool(relevant.strip()),
        "e195_code_path_status": relevant.splitlines(),
    }


def verify_recorded_inputs() -> pd.DataFrame:
    path = TABLES / "E195_INPUT_HASHES.csv"
    if not path.exists():
        raise FileNotFoundError(
            "E195 input lock is missing; run --mode prepare and commit it first"
        )
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "path",
        "bytes",
        "sha256",
        "expected_sha256",
        "hash_match",
        "pin_scope",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("E195 input lock schema is incomplete")
    for row in frame.itertuples(index=False):
        source = resolve_display_path(str(row.path))
        if not source.exists():
            raise FileNotFoundError(source)
        observed = sha256_file(source)
        if observed != str(row.sha256) or (
            str(row.expected_sha256) and observed != str(row.expected_sha256)
        ):
            raise RuntimeError(f"E195 recorded input changed: {source}")
        if int(source.stat().st_size) != int(row.bytes):
            raise RuntimeError(f"E195 input size changed: {source}")
    return frame


def panel_data_root(panel: str, seed: int) -> Path:
    return ISOLATED_ROOT / panel / f"seed_{seed}"


def raw_root(panel: str, seed: int) -> Path:
    return OUT / "panels" / panel / "raw_gears" / f"seed_{seed}"


def child_root(panel: str, seed: int) -> Path:
    return raw_root(panel, seed) / "norman" / f"seed_{seed}"


def wrapper_status_path(panel: str, seed: int) -> Path:
    return raw_root(panel, seed) / "E195_SEED_STATUS.json"


def child_status_path(panel: str, seed: int) -> Path:
    return raw_root(panel, seed) / "GEARS_PREDICTION_RECORD_STATUS.json"


def frozen_manifest_path(panel: str) -> Path:
    return OUT / "panels" / panel / "E195_FROZEN_MANIFEST.csv"


def create_isolated_data_root(panel: str, seed: int) -> None:
    root = panel_data_root(panel, seed)
    dataset = root / "norman_local_atlas"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "splits").mkdir(exist_ok=True)
    links = {
        dataset / "perturb_processed.h5ad": BASE_DATASET / "perturb_processed.h5ad",
        dataset / "data_pyg": BASE_DATASET / "data_pyg",
        dataset / "go.csv": BASE_DATASET / "go.csv",
        root / "gene2go.pkl": BASE_DATA_ROOT / "gene2go.pkl",
    }
    for link, target in links.items():
        if not target.exists():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"unexpected existing symlink: {link}")
            continue
        if link.exists():
            raise RuntimeError(
                f"E195 isolated cache path must not contain a copied asset: {link}"
            )
        link.symlink_to(target, target_is_directory=target.is_dir())


def prepare() -> dict[str, Any]:
    ensure_dirs()
    input_rows: list[dict[str, Any]] = []
    manifests: dict[str, pd.DataFrame] = {}
    for panel, config in PANELS.items():
        input_rows.append(verify_hash(config["manifest"], config["manifest_sha256"]))
        input_rows.append(verify_hash(config["pair"], config["pair_sha256"]))
        frame = pd.read_csv(config["manifest"])
        if len(frame) != 24 or frame["condition"].astype(str).nunique() != 24:
            raise RuntimeError(f"{panel} manifest is not 24 unique tasks")
        manifests[panel] = frame.copy()
        shutil.copy2(config["manifest"], frozen_manifest_path(panel))
        for seed in SEEDS:
            create_isolated_data_root(panel, seed)
    input_rows.append(verify_hash(PRESCRIBE_PAPER, PRESCRIBE_PAPER_SHA256))
    input_rows.append(verify_hash(PRESCRIBE_RMSE, PRESCRIBE_RMSE_SHA256))
    if set(manifests["P1"]["condition"].astype(str)) & set(
        manifests["P2"]["condition"].astype(str)
    ):
        raise RuntimeError("P1 and P2 manifests overlap")

    # These are the large immutable GEARS assets actually consumed at runtime.
    for path, expected in PINNED_LARGE_ASSETS.items():
        input_rows.append(verify_hash(path, expected))
    for path in (GEARS_CLI, Path(__file__).resolve()):
        observed = sha256_file(path)
        input_rows.append(
            {
                "path": display_path(path),
                "bytes": int(path.stat().st_size),
                "sha256": observed,
                "expected_sha256": observed,
                "hash_match": True,
                "pin_scope": "implementation_hash_locked_before_formal_run",
            }
        )
    input_hashes = pd.DataFrame(input_rows).drop_duplicates("path")
    input_hashes.to_csv(TABLES / "E195_INPUT_HASHES.csv", index=False)

    # This is a pre-run context audit only.  The implementation gate later
    # reads each child run's actual train/val/test sets instead of inferring
    # them as the complement of the frozen test manifest.
    metadata = ad.read_h5ad(BASE_DATASET / "perturb_processed.h5ad", backed="r")
    all_conditions = set(metadata.obs["condition"].astype(str).unique())
    support_rows: list[dict[str, Any]] = []
    for panel, manifest in manifests.items():
        test = set(manifest["condition"].astype(str))
        non_test_context = sorted(
            condition for condition in all_conditions if condition not in test
        )
        for condition in sorted(test):
            gene = condition.replace("+ctrl", "")
            double_hits = [
                candidate
                for candidate in non_test_context
                if candidate != condition
                and gene in {
                    token for token in candidate.split("+") if token != "ctrl"
                }
            ]
            support_rows.append(
                {
                    "panel": panel,
                    "condition": condition,
                    "task_gene": gene,
                    "audit_stage": "pre_run_dataset_context_only",
                    "actual_split_verified": False,
                    "n_non_test_double_conditions_containing_gene": len(double_hits),
                    "non_test_double_condition_examples": ";".join(double_hits[:10]),
                }
            )
    metadata.file.close()
    pd.DataFrame(support_rows).to_csv(
        TABLES / "E195_SUPPORT_EXPOSURE_AUDIT.csv", index=False
    )
    provenance = git_provenance()
    status = {
        "experiment": "E195",
        "stage": "POSTTRUTH_DIRECT_COMPETITOR_REPLICATION",
        "status": "PREPARED_NOT_RUN",
        "prepared_at": now(),
        "panels": {panel: len(frame) for panel, frame in manifests.items()},
        "panel_overlap": 0,
        "seeds": list(SEEDS),
        "uncertainty": True,
        "strict_score_lock_before_truth": True,
        "input_hash_rows": int(len(input_hashes)),
        "isolated_cache_root": str(ISOLATED_ROOT),
        "cache_isolation": "panel_and_seed",
        **provenance,
    }
    (OUT / "E195_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def command_for(panel: str, seed: int, device: str) -> list[str]:
    config = PANELS[panel]
    return [
        str(PYTHON),
        "-m",
        "safetrans_confidence.cli.run_gears_prediction_records",
        "--dataset",
        "norman",
        "--seed",
        str(seed),
        "--split",
        "single",
        "--run-type",
        "formal",
        "--epochs",
        "10",
        "--hidden-size",
        "48",
        "--decoder-hidden-size",
        "16",
        "--num-similar-genes",
        "10",
        "--batch-size",
        "32",
        "--test-batch-size",
        "64",
        "--max-cells-per-condition",
        "32",
        "--condition-sampling-seed",
        str(config["sampling_seed"]),
        "--device",
        device,
        "--out-dir",
        str(raw_root(panel, seed)),
        "--data-path",
        str(panel_data_root(panel, seed)),
        "--test-perturbations-file",
        str(frozen_manifest_path(panel)),
        "--fixed-test-deterministic-val",
        "--train-gene-set-size",
        "0.75",
        "--max-genes",
        "6000",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0005",
        "--coexpress-threshold",
        "0.4",
        "--direction-lambda",
        "0.1",
        "--uncertainty",
        "--strict-score-lock-before-truth",
        "--require-cuda",
    ]


def completed(panel: str, seed: int, require_wrapper: bool = True) -> bool:
    child = child_status_path(panel, seed)
    if not child.exists():
        return False
    try:
        status = json.loads(child.read_text(encoding="utf-8"))
    except Exception:
        return False
    config = PANELS[panel]
    expected_rng = {
        "python": seed,
        "numpy": seed,
        "torch_cpu": seed,
        "torch_cuda_all": seed,
    }
    scalar_checks = [
        status.get("status") == "ok",
        bool(status.get("strict_score_lock_before_truth")),
        status.get("dataset") == "norman",
        status.get("split") == "single",
        status.get("run_type") == "formal",
        bool(status.get("require_cuda")),
        int(status.get("seed", -1)) == seed,
        int(status.get("epochs", -1)) == TRAINING_CONTRACT["epochs"],
        int(status.get("n_prediction_records", -1)) == 24,
        int(status.get("actual_test_batch_size", -1))
        == TRAINING_CONTRACT["requested_test_batch_size"],
        str(status.get("actual_device", "")).startswith("cuda"),
        status.get("rng_control") == expected_rng,
        status.get("training_contract")
        == {
            **TRAINING_CONTRACT,
            "condition_sampling_seed": config["sampling_seed"],
        },
        status.get("test_manifest_sha256") == config["manifest_sha256"],
        status.get("runner_sha256") == sha256_file(GEARS_CLI),
        str(status.get("data_path", ""))
        == str(panel_data_root(panel, seed).resolve()),
    ]
    if not all(scalar_checks):
        return False
    condition_sets = status.get("actual_condition_sets", {})
    expected_test = set(
        pd.read_csv(frozen_manifest_path(panel))["condition"].astype(str)
    )
    train = set(map(str, condition_sets.get("train", [])))
    val = set(map(str, condition_sets.get("val", [])))
    test = set(map(str, condition_sets.get("test", [])))
    if test != expected_test or train & val or train & test or val & test:
        return False
    sampling = status.get("condition_graph_sampling", {})
    if (
        not sampling.get("enabled")
        or int(sampling.get("sampling_seed", -1)) != config["sampling_seed"]
        or int(sampling.get("max_cells_per_condition", -1))
        != TRAINING_CONTRACT["max_cells_per_condition"]
    ):
        return False
    for condition in expected_test:
        item = sampling.get("conditions", {}).get(condition, {})
        if (
            item.get("split") != "test"
            or int(item.get("before", -1)) != int(item.get("after", -2))
        ):
            return False
    initial = str(status.get("initial_model_state_sha256", ""))
    trained = str(status.get("trained_model_state_sha256", ""))
    saved = str(status.get("saved_model_state_sha256", ""))
    if (
        len(initial) != 64
        or len(trained) != 64
        or initial == trained
        or saved != trained
    ):
        return False
    model_hashes = status.get("model_artifact_sha256", {})
    cache_hashes = status.get("cache_artifact_sha256", {})
    if set(model_hashes) != {"model/config.pkl", "model/model.pt"}:
        return False
    if (
        not any(name.startswith("norman_local_atlas/splits/") for name in cache_hashes)
        or not any(name.endswith("_co_expression_network.csv") for name in cache_hashes)
    ):
        return False
    artifact_roots = (
        (child_root(panel, seed), status.get("critical_output_sha256", {})),
        (child_root(panel, seed), model_hashes),
        (panel_data_root(panel, seed), cache_hashes),
    )
    for root, artifacts in artifact_roots:
        if not artifacts:
            return False
        for relative, expected_hash in artifacts.items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                return False
    if require_wrapper:
        wrapper_path = wrapper_status_path(panel, seed)
        if not wrapper_path.exists():
            return False
        try:
            wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if (
            int(wrapper.get("returncode", -1)) != 0
            or wrapper.get("command") != command_for(
                panel, seed, str(status["actual_device"])
            )
        ):
            return False
    return True


def launch_wave(jobs: list[tuple[str, int, str]], rerun: bool) -> None:
    processes: list[tuple[str, int, subprocess.Popen[str], Any, float, list[str]]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        for panel, seed, device in jobs:
            if completed(panel, seed) and not rerun:
                print(f"[E195] reuse {panel} seed={seed}", flush=True)
                continue
            root = raw_root(panel, seed)
            root.mkdir(parents=True, exist_ok=True)
            command = command_for(panel, seed, device)
            log_path = root / "E195_TRAIN.log"
            handle = log_path.open("w", encoding="utf-8")
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except BaseException:
                handle.close()
                raise
            processes.append((panel, seed, process, handle, started, command))
            print(f"[E195] start {panel} seed={seed} device={device}", flush=True)
        failures: list[str] = []
        for panel, seed, process, handle, started, command in processes:
            try:
                returncode = process.wait()
            finally:
                handle.close()
            child_status: dict[str, Any] = {}
            if child_status_path(panel, seed).exists():
                try:
                    child_status = json.loads(
                        child_status_path(panel, seed).read_text(encoding="utf-8")
                    )
                except Exception as exc:
                    child_status = {"status": "unreadable", "error": repr(exc)}
            wrapper = {
                "panel": panel,
                "seed": seed,
                "returncode": returncode,
                "elapsed_seconds": time.monotonic() - started,
                "command": command,
                "child_status": child_status,
                "finished_at": now(),
            }
            wrapper_status_path(panel, seed).write_text(
                json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[E195] finish {panel} seed={seed} returncode={returncode} "
                f"status={child_status.get('status')}",
                flush=True,
            )
            if (
                returncode != 0
                or child_status.get("status") != "ok"
                or not completed(panel, seed)
            ):
                failures.append(f"{panel}/seed_{seed}")
        if failures:
            raise RuntimeError(
                "E195 training failed or violated its contract: "
                + ", ".join(failures)
            )
    finally:
        # A spawn error, interruption, or wait error must not leave the peer
        # GPU job running.  Normal completions pass through without signals.
        for _, _, process, _, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process, handle, _, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not handle.closed:
                handle.close()


def train(rerun: bool = False) -> None:
    verify_recorded_inputs()
    provenance = git_provenance()
    if not provenance["e195_code_paths_clean"]:
        raise RuntimeError(
            "E195 code paths are dirty; commit the locked implementation before training"
        )
    waves = [
        [("P1", 11, "cuda:0"), ("P1", 22, "cuda:1")],
        [("P1", 33, "cuda:0"), ("P2", 11, "cuda:1")],
        [("P2", 22, "cuda:0"), ("P2", 33, "cuda:1")],
    ]
    for wave in waves:
        launch_wave(wave, rerun=rerun)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if (
        keep.sum() < 4
        or np.unique(a[keep]).size < 2
        or np.unique(b[keep]).size < 2
    ):
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(a[keep], method="average"),
            rankdata(b[keep], method="average"),
        )[0, 1]
    )


def stable_order(
    values: np.ndarray, task_ids: np.ndarray, descending: bool
) -> np.ndarray:
    primary = -np.asarray(values, float) if descending else np.asarray(values, float)
    return np.lexsort((np.asarray(task_ids, str), primary))


def routing_values(
    risk: np.ndarray, error: np.ndarray, task_ids: np.ndarray, budget: float
) -> dict[str, float]:
    n = len(error)
    k = int(math.ceil(n * budget))
    risk_order = stable_order(risk, task_ids, descending=True)
    oracle_order = stable_order(error, task_ids, descending=True)
    selected = set(risk_order[:k].tolist())
    oracle = set(oracle_order[:k].tolist())
    selected_mean = float(np.mean(error[risk_order[:k]]))
    oracle_mean = float(np.mean(error[oracle_order[:k]]))
    overall = float(np.mean(error))
    remaining = float(np.mean(error[risk_order[k:]])) if k < n else float("nan")
    denominator = oracle_mean - overall
    return {
        "budget": budget,
        "n_selected": k,
        "high_error_recall": len(selected & oracle) / k,
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall,
        "error_lift": selected_mean / overall if abs(overall) > 1e-15 else np.nan,
        "remaining_mean_error": remaining,
        "remaining_error_reduction": (
            (overall - remaining) / overall
            if math.isfinite(remaining) and abs(overall) > 1e-15
            else np.nan
        ),
        "oracle_mean_error": oracle_mean,
        "oracle_normalized_utility": (
            (selected_mean - overall) / denominator
            if denominator > 1e-15
            else np.nan
        ),
    }


def coverage_values(
    risk: np.ndarray, error: np.ndarray, task_ids: np.ndarray
) -> tuple[list[dict[str, float]], float]:
    order = stable_order(risk, task_ids, descending=False)
    overall = float(np.mean(error))
    rows: list[dict[str, float]] = []
    for coverage in np.arange(0.50, 1.001, 0.05):
        n_keep = int(math.ceil(len(error) * float(coverage)))
        retained = float(np.mean(error[order[:n_keep]]))
        rows.append(
            {
                "coverage": float(round(coverage, 2)),
                "n_retained": n_keep,
                "selective_error": retained,
                "normalized_selective_error": (
                    retained / overall if abs(overall) > 1e-15 else np.nan
                ),
            }
        )
    x = np.asarray([row["coverage"] for row in rows], float)
    y = np.asarray([row["normalized_selective_error"] for row in rows], float)
    normalized_aurc = float(np.trapz(y, x) / (x[-1] - x[0]))
    return rows, normalized_aurc


def bootstrap_association(
    risk: np.ndarray, error: np.ndarray, seed: int
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(N_BOOT):
        indexes = rng.integers(0, len(risk), len(risk))
        value = safe_spearman(risk[indexes], error[indexes])
        if math.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def bootstrap_utility(
    risk: np.ndarray, error: np.ndarray, task_ids: np.ndarray, seed: int
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for iteration in range(N_BOOT):
        indexes = rng.integers(0, len(risk), len(risk))
        sampled_ids = np.asarray(
            [f"{iteration}:{j}:{task_ids[index]}" for j, index in enumerate(indexes)]
        )
        value = routing_values(
            risk[indexes], error[indexes], sampled_ids, 0.20
        )["oracle_normalized_utility"]
        if math.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def bootstrap_paired_utility_delta(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    error: np.ndarray,
    task_ids: np.ndarray,
    seed: int,
) -> tuple[float, float, float, float, float, int]:
    """Compare two 20% routing scores with identical task resamples."""
    point_a = routing_values(risk_a, error, task_ids, 0.20)[
        "oracle_normalized_utility"
    ]
    point_b = routing_values(risk_b, error, task_ids, 0.20)[
        "oracle_normalized_utility"
    ]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for iteration in range(N_BOOT):
        indexes = rng.integers(0, len(error), len(error))
        sampled_ids = np.asarray(
            [f"{iteration}:{j}:{task_ids[index]}" for j, index in enumerate(indexes)]
        )
        value_a = routing_values(
            risk_a[indexes], error[indexes], sampled_ids, 0.20
        )["oracle_normalized_utility"]
        value_b = routing_values(
            risk_b[indexes], error[indexes], sampled_ids, 0.20
        )["oracle_normalized_utility"]
        if math.isfinite(value_a) and math.isfinite(value_b):
            deltas.append(value_a - value_b)
    if not deltas:
        return point_a, point_b, point_a - point_b, np.nan, np.nan, 0
    return (
        point_a,
        point_b,
        point_a - point_b,
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
        len(deltas),
    )


def validate_locked_run(
    panel: str, seed: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    if not completed(panel, seed):
        raise RuntimeError(f"incomplete or contract-invalid run {panel} seed={seed}")
    root = child_root(panel, seed)
    table_root = root / "tables"
    array_root = root / "arrays"
    lock_path = table_root / "PRETRUTH_SCORE_LOCK.json"
    unlock_path = table_root / "TRUTH_UNLOCK.json"
    score_path = table_root / "PRETRUTH_SCORE_LOCK.csv"
    record_path = table_root / "PREDICTION_RECORDS.csv"
    prediction_path = array_root / "gears_predicted_effects.npz"
    truth_path = array_root / "gears_true_effects.npz"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    score = pd.read_csv(score_path)
    record = pd.read_csv(record_path)
    expected_tasks = set(
        pd.read_csv(frozen_manifest_path(panel))["condition"].astype(str)
    )
    if (
        len(score) != 24
        or len(record) != 24
        or set(score.task_id.astype(str)) != expected_tasks
        or set(record.task_id.astype(str)) != expected_tasks
    ):
        raise RuntimeError(f"locked task set mismatch {panel} seed={seed}")
    file_checks = {
        "score_csv": (
            sha256_file(score_path),
            lock.get("score_csv_sha256"),
            unlock.get("pretruth_score_csv_sha256"),
        ),
        "prediction_npz": (
            sha256_file(prediction_path),
            lock.get("prediction_npz_sha256"),
            unlock.get("pretruth_prediction_npz_sha256"),
        ),
        "lock_json": (
            sha256_file(lock_path),
            unlock.get("pretruth_lock_json_sha256"),
            unlock.get("pretruth_lock_json_sha256"),
        ),
        "records_csv": (
            sha256_file(record_path),
            unlock.get("records_csv_sha256"),
            unlock.get("records_csv_sha256"),
        ),
        "truth_npz": (
            sha256_file(truth_path),
            unlock.get("truth_npz_sha256"),
            unlock.get("truth_npz_sha256"),
        ),
    }
    for name, values in file_checks.items():
        if len(set(map(str, values))) != 1:
            raise RuntimeError(f"{name} hash-chain mismatch {panel} seed={seed}")
    if (
        lock.get("truth_fields_accessed") is not False
        or int(lock.get("n_tasks", -1)) != 24
        or int(unlock.get("n_tasks", -1)) != 24
        or lock.get("prediction_cell_task_order_sha256")
        != unlock.get("prediction_cell_task_order_sha256")
        or unlock.get("prediction_cell_task_order_sha256")
        != unlock.get("truth_cell_task_order_sha256")
        or datetime.fromisoformat(str(unlock["truth_unlock_started_at"]))
        <= datetime.fromisoformat(str(lock["locked_at"]))
    ):
        raise RuntimeError(f"truth-unlock ordering contract failed {panel} seed={seed}")
    joined = score.merge(
        record,
        on="task_id",
        how="outer",
        suffixes=("_lock", "_record"),
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise RuntimeError(f"score/record join failed {panel} seed={seed}")
    for column in (
        "record_id",
        "task_key",
        "fold_id",
        "split",
        "gene_order_hash",
        "predicted_effect_key",
        "n_cells",
    ):
        left = joined[f"{column}_lock"].astype(str).to_numpy()
        right = joined[f"{column}_record"].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"locked column changed ({column}) {panel} seed={seed}"
            )
    for column in (
        "predicted_effect_rms_magnitude",
        "gears_uncertainty_logvar_mean",
    ):
        if not np.allclose(
            joined[f"{column}_lock"].to_numpy(float),
            joined[f"{column}_record"].to_numpy(float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"locked numeric score changed ({column}) {panel} seed={seed}"
            )
    if (
        record.pretruth_score_lock_sha256.astype(str)
        .ne(lock["score_csv_sha256"])
        .any()
        or record.pretruth_prediction_npz_sha256.astype(str)
        .ne(lock["prediction_npz_sha256"])
        .any()
    ):
        raise RuntimeError(f"record hash binding failed {panel} seed={seed}")
    with np.load(prediction_path) as archive:
        predictions = {
            key: np.asarray(archive[key], np.float32) for key in archive.files
        }
    with np.load(truth_path) as archive:
        truths = {
            key: np.asarray(archive[key], np.float32) for key in archive.files
        }
    if set(predictions) != set(record.predicted_effect_key.astype(str)) or set(
        truths
    ) != set(record.true_effect_key.astype(str)):
        raise RuntimeError(f"array key set mismatch {panel} seed={seed}")
    vector_lengths: set[int] = set()
    max_magnitude_diff = 0.0
    max_rmse_diff = 0.0
    for row in record.itertuples(index=False):
        predicted = predictions[str(row.predicted_effect_key)]
        truth = truths[str(row.true_effect_key)]
        if (
            predicted.ndim != 1
            or truth.ndim != 1
            or predicted.shape != truth.shape
            or not np.isfinite(predicted).all()
            or not np.isfinite(truth).all()
        ):
            raise RuntimeError(f"invalid effect vector {panel} seed={seed}/{row.task_id}")
        vector_lengths.add(int(predicted.size))
        magnitude = float(np.sqrt(np.mean(np.square(predicted, dtype=np.float64))))
        error = rmse(predicted, truth)
        max_magnitude_diff = max(
            max_magnitude_diff,
            abs(magnitude - float(row.predicted_effect_rms_magnitude)),
        )
        max_rmse_diff = max(max_rmse_diff, abs(error - float(row.true_error_rmse)))
    child_status = json.loads(
        child_status_path(panel, seed).read_text(encoding="utf-8")
    )
    if (
        vector_lengths != {int(child_status["n_genes"])}
        or max_magnitude_diff > 1e-7
        or max_rmse_diff > 1e-7
        or record.gene_order_hash.astype(str).nunique() != 1
        or str(record.gene_order_hash.iloc[0]) != str(lock["gene_order_hash"])
        or str(record.gene_order_hash.iloc[0])
        != str(child_status["gene_order_hash"])
        or not np.isfinite(record.true_error_rmse.to_numpy(float)).all()
    ):
        raise RuntimeError(f"effect/metric integrity failed {panel} seed={seed}")
    return (
        record,
        predictions,
        truths,
        {
            "panel": panel,
            "seed": seed,
            "score_lock_hash_chain": True,
            "truth_unlock_after_lock": True,
            "cell_task_order_match": True,
            "score_record_exact_match": True,
            "array_keys_match": True,
            "effect_vectors_finite": True,
            "vector_length": next(iter(vector_lengths)),
            "max_magnitude_recompute_diff": max_magnitude_diff,
            "max_rmse_recompute_diff": max_rmse_diff,
        },
    )


def load_native_outputs(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    lock_audit_rows: list[dict[str, Any]] = []
    for panel, config in PANELS.items():
        manifest = pd.read_csv(frozen_manifest_path(panel))
        expected = set(manifest["condition"].astype(str))
        panel_records: list[pd.DataFrame] = []
        predictions: dict[int, dict[str, np.ndarray]] = {}
        truths: dict[int, dict[str, np.ndarray]] = {}
        for seed in SEEDS:
            record, prediction_arrays, truth_arrays, lock_audit = (
                validate_locked_run(panel, seed)
            )
            lock_audit_rows.append(lock_audit)
            if (
                len(record) != 24
                or set(record["task_id"].astype(str)) != expected
                or record["task_id"].astype(str).nunique() != 24
            ):
                raise RuntimeError(f"strict task mismatch {panel} seed={seed}")
            predictions[seed] = prediction_arrays
            truths[seed] = truth_arrays
            record["panel"] = panel
            panel_records.append(record)
            for _, row in record.iterrows():
                single_rows.append(
                    {
                        "panel": panel,
                        "panel_label": config["label"],
                        "seed": seed,
                        "task_id": str(row.task_id),
                        "gene_order_hash": str(row.gene_order_hash),
                        "native_logvar": float(row.gears_uncertainty_logvar_mean),
                        "predicted_magnitude": float(
                            row.predicted_effect_rms_magnitude
                        ),
                        "own_rmse": float(row.true_error_rmse),
                        "predicted_effect_key": str(row.predicted_effect_key),
                        "true_effect_key": str(row.true_effect_key),
                        "pretruth_score_lock_sha256": str(
                            row.pretruth_score_lock_sha256
                        ),
                    }
                )
        combined = pd.concat(panel_records, ignore_index=True)
        for task_id, group in combined.groupby("task_id", sort=True):
            group = group.sort_values("fold_id")
            if (
                group["gene_order_hash"].astype(str).nunique() != 1
                or not np.isfinite(
                    group[
                        [
                            "predicted_effect_rms_magnitude",
                            "true_error_rmse",
                        ]
                    ].to_numpy(float)
                ).all()
            ):
                raise RuntimeError(f"seed-family scalar integrity failed {panel}/{task_id}")
            vectors = np.stack(
                [
                    predictions[int(row.fold_id)][str(row.predicted_effect_key)]
                    for _, row in group.iterrows()
                ]
            ).astype(float)
            true_vectors = [
                truths[int(row.fold_id)][str(row.true_effect_key)]
                for _, row in group.iterrows()
            ]
            reference_truth = np.asarray(true_vectors[0], float)
            true_max_diff = max(
                float(np.max(np.abs(np.asarray(value, float) - reference_truth)))
                for value in true_vectors
            )
            centroid = vectors.mean(axis=0)
            member_rmse = np.asarray(
                [rmse(vector, reference_truth) for vector in vectors], float
            )
            family_rms = float(np.sqrt(np.mean(member_rmse**2)))
            centroid_rmse = rmse(centroid, reference_truth)
            disagreement = float(np.sqrt(np.mean((vectors - centroid) ** 2)))
            identity_residual = abs(
                family_rms**2 - centroid_rmse**2 - disagreement**2
            )
            family_rows.append(
                {
                    "panel": panel,
                    "panel_label": config["label"],
                    "task_id": str(task_id),
                    "n_seeds": len(group),
                    "native_logvar_mean": float(
                        group["gears_uncertainty_logvar_mean"].mean()
                    ),
                    "seed_disagreement": disagreement,
                    "predicted_magnitude": float(
                        group["predicted_effect_rms_magnitude"].mean()
                    ),
                    "family_rms_error": family_rms,
                    "centroid_rmse": centroid_rmse,
                    "truth_max_abs_diff_across_seeds": true_max_diff,
                    "family_squared_identity_residual": identity_residual,
                    "gene_order_hash": str(group["gene_order_hash"].iloc[0]),
                }
            )
            invariant_rows.extend(
                [
                    {
                        "panel": panel,
                        "task_id": str(task_id),
                        "check": "truth_equal_across_seeds",
                        "value": true_max_diff,
                        "threshold": 1e-7,
                        "pass": true_max_diff <= 1e-7,
                    },
                    {
                        "panel": panel,
                        "task_id": str(task_id),
                        "check": "hilbert_family_identity",
                        "value": identity_residual,
                        "threshold": 1e-10,
                        "pass": identity_residual <= 1e-10,
                    },
                ]
            )
    return (
        pd.DataFrame(single_rows),
        pd.DataFrame(family_rows),
        pd.DataFrame(invariant_rows),
        pd.DataFrame(lock_audit_rows),
    )


def actual_support_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        expected_test = set(
            pd.read_csv(frozen_manifest_path(panel))["condition"].astype(str)
        )
        for seed in SEEDS:
            status = json.loads(
                child_status_path(panel, seed).read_text(encoding="utf-8")
            )
            condition_sets = status["actual_condition_sets"]
            train_val = set(map(str, condition_sets["train"])) | set(
                map(str, condition_sets["val"])
            )
            actual_test = set(map(str, condition_sets["test"]))
            if actual_test != expected_test:
                raise RuntimeError(f"actual test split changed {panel} seed={seed}")
            for condition in sorted(expected_test):
                gene = condition.replace("+ctrl", "")
                double_hits = sorted(
                    candidate
                    for candidate in train_val
                    if candidate != condition
                    and gene
                    in {token for token in candidate.split("+") if token != "ctrl"}
                )
                rows.append(
                    {
                        "panel": panel,
                        "seed": seed,
                        "condition": condition,
                        "task_gene": gene,
                        "audit_stage": "actual_child_set2conditions",
                        "actual_split_verified": True,
                        "exact_single_condition_in_train_or_val": (
                            condition in train_val
                        ),
                        "n_train_or_val_double_conditions_containing_gene": len(
                            double_hits
                        ),
                        "double_condition_examples": ";".join(double_hits[:10]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.exact_single_condition_in_train_or_val.any():
        raise RuntimeError("held-out exact single condition leaked into train/validation")
    return frame


def build_system_frame(family: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        panel: str,
        task_id: str,
        system: str,
        arm: str,
        score_name: str,
        risk: float,
        outcome_name: str,
        error: float,
    ) -> None:
        rows.append(
            {
                "panel": panel,
                "task_id": task_id,
                "system": system,
                "arm": arm,
                "score_name": score_name,
                "risk": float(risk),
                "outcome_name": outcome_name,
                "error": float(error),
            }
        )

    for _, row in family.iterrows():
        for score in (
            "native_logvar_mean",
            "seed_disagreement",
            "predicted_magnitude",
        ):
            add(
                row.panel,
                row.task_id,
                "GEARS-UQ",
                "same_prediction_family_rms",
                score,
                row[score],
                "family_rms_error",
                row.family_rms_error,
            )
    for panel, config in PANELS.items():
        pair = pd.read_csv(config["pair"])
        for _, row in pair.iterrows():
            for score in (
                "risk_gears_scgpt_disagreement",
                "risk_gears_predicted_magnitude",
                "risk_scgpt_predicted_magnitude",
            ):
                add(
                    panel,
                    str(row.perturbation),
                    "GEARS-scGPT pair",
                    "posthoc_pair",
                    score,
                    row[score],
                    "task_mean_rmse",
                    row.task_mean_rmse,
                )
    paper = pd.read_csv(PRESCRIBE_PAPER)
    panel_map = {"Norman_P1": "P1", "Norman_P2": "P2"}
    for _, row in paper.iterrows():
        panel = panel_map[str(row.panel)]
        risks = {
            "epistemic_confidence": -float(row.epistemic_confidence),
            "aleatoric_confidence": -float(row.aleatoric_confidence),
            "combined_confidence": -float(row.combined_confidence),
            "predicted_magnitude": -float(row.predicted_magnitude_rms),
        }
        for name, risk in risks.items():
            add(
                panel,
                str(row.task_id),
                "PRESCRIBE",
                "paper_effect_pearson_primary",
                name,
                risk,
                "one_minus_pearson_effect_accuracy",
                1.0 - float(row.pearson_effect_accuracy),
            )
    sensitivity = pd.read_csv(PRESCRIBE_RMSE)
    for _, row in sensitivity.iterrows():
        panel = panel_map[str(row.panel)]
        risks = {
            "risk_epistemic": float(row.risk_epistemic),
            "risk_aleatoric": float(row.risk_aleatoric),
            "risk_combined": float(row.risk_combined),
            "predicted_magnitude": float(row.magnitude_pred_rms),
        }
        for name, risk in risks.items():
            add(
                panel,
                str(row.task_id),
                "PRESCRIBE",
                "rmse_sensitivity",
                name,
                risk,
                "rmse_mean_profile",
                float(row.rmse_mean_profile),
            )
    frame = pd.DataFrame(rows)
    for (panel, system, arm, score), group in frame.groupby(
        ["panel", "system", "arm", "score_name"], sort=True
    ):
        expected = set(
            pd.read_csv(frozen_manifest_path(panel))["condition"].astype(str)
        )
        if len(group) != 24 or group.task_id.nunique() != 24:
            raise RuntimeError(
                f"Track B task join failed: {panel}/{system}/{arm}/{score}"
            )
        if set(group.task_id.astype(str)) != expected:
            raise RuntimeError(
                f"Track B task set differs from manifest: {panel}/{system}/{arm}/{score}"
            )
    return frame


def analyze_groups(
    systems: pd.DataFrame, single: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    association_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []

    # Single-seed Track A.
    for (panel, seed), group in single.groupby(["panel", "seed"], sort=True):
        for score in ("native_logvar", "predicted_magnitude"):
            x = group[score].to_numpy(float)
            y = group["own_rmse"].to_numpy(float)
            finite_ok = bool(np.isfinite(x).all() and np.isfinite(y).all())
            variable = bool(np.unique(x).size >= 2) if finite_ok else False
            if finite_ok and variable:
                low, high, valid = bootstrap_association(
                    x, y, stable_seed("single", panel, seed, score)
                )
                status = "estimable"
            else:
                low, high, valid = np.nan, np.nan, 0
                status = (
                    "non_estimable_nonfinite"
                    if not finite_ok
                    else "undefined_constant_score"
                )
            association_rows.append(
                {
                    "track": "A_single_seed",
                    "panel": panel,
                    "seed": seed,
                    "system": "GEARS-UQ",
                    "arm": "single_seed_own_error",
                    "score_name": score,
                    "outcome_name": "own_rmse",
                    "n_tasks": len(group),
                    "n_seeds": 1,
                    "n_tasks_per_seed": len(group),
                    "spearman": safe_spearman(x, y),
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "bootstrap_valid": valid,
                    "status": status,
                }
            )
    single_seed_rows = pd.DataFrame(association_rows)
    for (panel, score), group in single_seed_rows.groupby(
        ["panel", "score_name"], sort=True
    ):
        finite = group.spearman.to_numpy(float)
        finite = finite[np.isfinite(finite)]
        association_rows.append(
            {
                "track": "A_single_seed_median",
                "panel": panel,
                "seed": "median_of_11_22_33",
                "system": "GEARS-UQ",
                "arm": "single_seed_own_error",
                "score_name": score,
                "outcome_name": "own_rmse",
                "n_tasks": np.nan,
                "n_seeds": len(group),
                "n_tasks_per_seed": 24,
                "spearman": float(np.median(finite)) if len(finite) else np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "bootstrap_valid": 0,
                "status": "descriptive_seed_median",
            }
        )

    for keys, group in systems.groupby(
        ["panel", "system", "arm", "score_name", "outcome_name"], sort=True
    ):
        panel, system, arm, score, outcome = keys
        x = group.risk.to_numpy(float)
        y = group.error.to_numpy(float)
        ids = group.task_id.astype(str).to_numpy()
        finite_ok = bool(np.isfinite(x).all() and np.isfinite(y).all())
        variable = bool(np.unique(x).size >= 2) if finite_ok else False
        if finite_ok and variable:
            low, high, valid = bootstrap_association(
                x, y, stable_seed("association", *keys)
            )
            coverage, aurc = coverage_values(x, y, ids)
            utility_low, utility_high, utility_valid = bootstrap_utility(
                x, y, ids, stable_seed("utility", *keys)
            )
            status = "estimable"
        else:
            low, high, valid = np.nan, np.nan, 0
            coverage, aurc = [], np.nan
            utility_low, utility_high, utility_valid = np.nan, np.nan, 0
            status = (
                "non_estimable_nonfinite"
                if not finite_ok
                else "undefined_constant_score"
            )
        association_rows.append(
            {
                "track": "A_family" if system == "GEARS-UQ" else "B_system",
                "panel": panel,
                "seed": "",
                "system": system,
                "arm": arm,
                "score_name": score,
                "outcome_name": outcome,
                "n_tasks": len(group),
                "n_seeds": 3 if system == "GEARS-UQ" else np.nan,
                "n_tasks_per_seed": np.nan,
                "spearman": safe_spearman(x, y),
                "ci95_lower": low,
                "ci95_upper": high,
                "bootstrap_valid": valid,
                "status": status,
            }
        )
        if status != "estimable":
            continue
        for budget in (0.10, 0.20, 0.30):
            route = routing_values(x, y, ids, budget)
            routing_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "arm": arm,
                    "score_name": score,
                    "outcome_name": outcome,
                    **route,
                    "utility_ci95_lower": utility_low if budget == 0.20 else np.nan,
                    "utility_ci95_upper": utility_high if budget == 0.20 else np.nan,
                    "utility_bootstrap_valid": (
                        utility_valid if budget == 0.20 else 0
                    ),
                }
            )
        for row in coverage:
            coverage_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "arm": arm,
                    "score_name": score,
                    "outcome_name": outcome,
                    "normalized_aurc": aurc,
                    **row,
                }
            )

    association = pd.DataFrame(association_rows)
    routing = pd.DataFrame(routing_rows)
    coverage = pd.DataFrame(coverage_rows)
    for keys, group in association.loc[
        association.track.isin(["A_family", "B_system"])
    ].groupby(["system", "arm", "score_name", "outcome_name"], sort=True):
        system, arm, score, outcome = keys
        panel_rows = group[group.panel.isin(["P1", "P2"])]
        valid_panel_rows = panel_rows[
            (panel_rows.status == "estimable")
            & np.isfinite(panel_rows.spearman.astype(float))
        ]
        route = routing[
            (routing.system == system)
            & (routing.arm == arm)
            & (routing.score_name == score)
            & (routing.outcome_name == outcome)
            & (routing.budget == 0.20)
        ]
        curves = coverage[
            (coverage.system == system)
            & (coverage.arm == arm)
            & (coverage.score_name == score)
            & (coverage.outcome_name == outcome)
        ]
        panel_complete = (
            set(valid_panel_rows.panel.astype(str)) == {"P1", "P2"}
            and set(route.panel.astype(str)) == {"P1", "P2"}
            and set(curves.panel.astype(str)) == {"P1", "P2"}
            and np.isfinite(route.oracle_normalized_utility.to_numpy(float)).all()
            and np.isfinite(curves.normalized_aurc.to_numpy(float)).all()
        )
        system_rows.append(
            {
                "system": system,
                "arm": arm,
                "score_name": score,
                "outcome_name": outcome,
                "n_panels": len(panel_rows),
                "n_panels_valid": len(valid_panel_rows),
                "status": "estimable" if panel_complete else "NON_ESTIMABLE",
                "panel_macro_spearman": (
                    float(valid_panel_rows.spearman.mean())
                    if panel_complete
                    else np.nan
                ),
                "panel_macro_utility20": float(
                    route.oracle_normalized_utility.mean()
                )
                if panel_complete
                else np.nan,
                "panel_macro_normalized_aurc": (
                    float(curves.groupby("panel").normalized_aurc.first().mean())
                    if panel_complete
                    else np.nan
                ),
                "raw_error_values_directly_comparable_across_systems": False,
            }
        )

    paired_rows: list[dict[str, Any]] = []
    pair_specs = [
        (
            "GEARS-UQ",
            "same_prediction_family_rms",
            "native_logvar_mean",
            "seed_disagreement",
        ),
        (
            "GEARS-UQ",
            "same_prediction_family_rms",
            "native_logvar_mean",
            "predicted_magnitude",
        ),
        (
            "GEARS-UQ",
            "same_prediction_family_rms",
            "seed_disagreement",
            "predicted_magnitude",
        ),
        (
            "PRESCRIBE",
            "paper_effect_pearson_primary",
            "combined_confidence",
            "predicted_magnitude",
        ),
        (
            "PRESCRIBE",
            "rmse_sensitivity",
            "risk_combined",
            "predicted_magnitude",
        ),
    ]
    for system, arm, score_a, score_b in pair_specs:
        for panel in ("P1", "P2"):
            subset = systems[
                (systems.system == system)
                & (systems.arm == arm)
                & (systems.panel == panel)
                & (systems.score_name.isin([score_a, score_b]))
            ]
            pivot = subset.pivot(
                index="task_id", columns="score_name", values=["risk", "error"]
            )
            if len(pivot) != 24:
                raise RuntimeError(f"paired comparison join failed {pair_specs}/{panel}")
            risk_a = pivot[("risk", score_a)].to_numpy(float)
            risk_b = pivot[("risk", score_b)].to_numpy(float)
            error = pivot[("error", score_a)].to_numpy(float)
            if not (
                np.isfinite(risk_a).all()
                and np.isfinite(risk_b).all()
                and np.isfinite(error).all()
                and np.unique(risk_a).size >= 2
                and np.unique(risk_b).size >= 2
            ):
                paired_rows.append(
                    {
                        "panel": panel,
                        "system": system,
                        "arm": arm,
                        "score_a": score_a,
                        "score_b": score_b,
                        "spearman_a": np.nan,
                        "spearman_b": np.nan,
                        "paired_spearman_delta_a_minus_b": np.nan,
                        "delta_ci95_lower": np.nan,
                        "delta_ci95_upper": np.nan,
                        "bootstrap_valid": 0,
                        "score_score_spearman": np.nan,
                        "utility20_a": np.nan,
                        "utility20_b": np.nan,
                        "paired_utility20_delta_a_minus_b": np.nan,
                        "utility_delta_ci95_lower": np.nan,
                        "utility_delta_ci95_upper": np.nan,
                        "utility_delta_bootstrap_valid": 0,
                        "status": "NON_ESTIMABLE",
                    }
                )
                continue
            rng = np.random.default_rng(
                stable_seed("paired", system, arm, panel, score_a, score_b)
            )
            deltas: list[float] = []
            for _ in range(N_BOOT):
                indexes = rng.integers(0, len(error), len(error))
                value = safe_spearman(
                    risk_a[indexes], error[indexes]
                ) - safe_spearman(risk_b[indexes], error[indexes])
                if math.isfinite(value):
                    deltas.append(value)
            (
                utility_a,
                utility_b,
                utility_delta,
                utility_delta_low,
                utility_delta_high,
                utility_delta_valid,
            ) = bootstrap_paired_utility_delta(
                risk_a,
                risk_b,
                error,
                pivot.index.astype(str).to_numpy(),
                stable_seed(
                    "paired_utility",
                    system,
                    arm,
                    panel,
                    score_a,
                    score_b,
                ),
            )
            paired_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "arm": arm,
                    "score_a": score_a,
                    "score_b": score_b,
                    "spearman_a": safe_spearman(risk_a, error),
                    "spearman_b": safe_spearman(risk_b, error),
                    "paired_spearman_delta_a_minus_b": safe_spearman(
                        risk_a, error
                    )
                    - safe_spearman(risk_b, error),
                    "delta_ci95_lower": (
                        float(np.quantile(deltas, 0.025)) if deltas else np.nan
                    ),
                    "delta_ci95_upper": (
                        float(np.quantile(deltas, 0.975)) if deltas else np.nan
                    ),
                    "bootstrap_valid": len(deltas),
                    "score_score_spearman": safe_spearman(risk_a, risk_b),
                    "utility20_a": utility_a,
                    "utility20_b": utility_b,
                    "paired_utility20_delta_a_minus_b": utility_delta,
                    "utility_delta_ci95_lower": utility_delta_low,
                    "utility_delta_ci95_upper": utility_delta_high,
                    "utility_delta_bootstrap_valid": utility_delta_valid,
                    "status": "estimable",
                }
            )
    return (
        association,
        pd.DataFrame(paired_rows),
        routing,
        coverage,
        pd.DataFrame(system_rows),
    )


def dynamic_range(single: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (panel, seed), group in single.groupby(["panel", "seed"], sort=True):
        for score in ("native_logvar", "predicted_magnitude"):
            values = group[score].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "level": "single_seed",
                    "panel": panel,
                    "seed": seed,
                    "score_name": score,
                    "n": len(values),
                    "n_nan": int((~np.isfinite(values)).sum()),
                    "n_unique_finite": int(np.unique(finite).size),
                    "min": float(np.min(finite)) if len(finite) else np.nan,
                    "max": float(np.max(finite)) if len(finite) else np.nan,
                    "range": (
                        float(np.max(finite) - np.min(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "estimable": bool(len(finite) == len(values) and np.unique(finite).size >= 2),
                }
            )
    for panel, group in family.groupby("panel", sort=True):
        for score in (
            "native_logvar_mean",
            "seed_disagreement",
            "predicted_magnitude",
        ):
            values = group[score].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "level": "family",
                    "panel": panel,
                    "seed": "",
                    "score_name": score,
                    "n": len(values),
                    "n_nan": int((~np.isfinite(values)).sum()),
                    "n_unique_finite": int(np.unique(finite).size),
                    "min": float(np.min(finite)) if len(finite) else np.nan,
                    "max": float(np.max(finite)) if len(finite) else np.nan,
                    "range": (
                        float(np.max(finite) - np.min(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "estimable": bool(len(finite) == len(values) and np.unique(finite).size >= 2),
                }
            )
    return pd.DataFrame(rows)


def runtime_environment() -> pd.DataFrame:
    rows = [
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
        {"component": "numpy", "version": np.__version__},
        {"component": "pandas", "version": pd.__version__},
    ]
    for command, component in (
        ([str(PYTHON), "-c", "import torch; print(torch.__version__)"], "torch"),
        ([str(PYTHON), "-c", "import scanpy; print(scanpy.__version__)"], "scanpy"),
        (
            [
                str(PYTHON),
                "-c",
                "import importlib.metadata as m; print(m.version('cell-gears'))",
            ],
            "cell-gears",
        ),
        (
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            "gpu",
        ),
    ):
        try:
            value = subprocess.check_output(command, text=True).strip()
        except Exception as exc:
            value = f"unavailable: {exc!r}"
        rows.append({"component": component, "version": value})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return "NA"
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_figure(
    family: pd.DataFrame,
    association: pd.DataFrame,
    routing: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    colors = {"P1": "#3B6FB6", "P2": "#C96A3D"}
    for panel in ("P1", "P2"):
        take = family[family.panel == panel]
        axes[0, 0].scatter(
            take.native_logvar_mean,
            take.family_rms_error,
            label=panel,
            s=38,
            alpha=0.78,
            color=colors[panel],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0, 1].scatter(
            take.seed_disagreement,
            take.family_rms_error,
            label=panel,
            s=38,
            alpha=0.78,
            color=colors[panel],
            edgecolor="white",
            linewidth=0.4,
        )
    axes[0, 0].set(
        xlabel="Native mean log-variance (risk)",
        ylabel="GEARS-UQ family RMS error",
        title="a  Native GEARS score",
    )
    axes[0, 1].set(
        xlabel="Three-seed disagreement",
        ylabel="GEARS-UQ family RMS error",
        title="b  Same predictions, ensemble disagreement",
    )
    axes[0, 0].legend(frameon=False)

    score_order = [
        "native_logvar_mean",
        "seed_disagreement",
        "predicted_magnitude",
    ]
    short_labels = ["Native logvar", "Seed disagreement", "Magnitude"]
    take = routing[
        (routing.system == "GEARS-UQ")
        & (routing.arm == "same_prediction_family_rms")
        & np.isclose(routing.budget, 0.20)
    ]
    positions = np.arange(len(score_order))
    width = 0.36
    for panel, offset in (("P1", -width / 2), ("P2", width / 2)):
        values = [
            float(
                take[
                    (take.panel == panel) & (take.score_name == score)
                ].oracle_normalized_utility.iloc[0]
            )
            for score in score_order
        ]
        axes[1, 0].bar(
            positions + offset,
            values,
            width=width,
            color=colors[panel],
            label=panel,
            alpha=0.9,
        )
    axes[1, 0].set_xticks(positions, short_labels, rotation=12)
    axes[1, 0].axhline(0, color="#333333", lw=0.8)
    axes[1, 0].set(
        ylabel="20% oracle-normalized utility",
        title="c  Fixed review budget, same predictions",
    )
    axes[1, 0].legend(frameon=False)

    curve = coverage[
        (coverage.system == "GEARS-UQ")
        & (coverage.arm == "same_prediction_family_rms")
    ]
    styles = {
        "native_logvar_mean": "-",
        "seed_disagreement": "--",
        "predicted_magnitude": ":",
    }
    for (panel, score), group in curve.groupby(["panel", "score_name"], sort=True):
        axes[1, 1].plot(
            group.coverage,
            group.normalized_selective_error,
            linestyle=styles[score],
            color=colors[panel],
            label=f"{panel} {score}",
        )
    axes[1, 1].axhline(1, color="#777777", lw=0.8)
    axes[1, 1].set(
        xlabel="Coverage retained (low risk first)",
        ylabel="Selective error / full error",
        title="d  GEARS-UQ risk–coverage",
    )
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        "E195 | Native GEARS uncertainty on two frozen Norman panels",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "E195_native_uq_comparison.png", dpi=240)
    fig.savefig(FIGURES / "E195_native_uq_comparison.pdf")
    plt.close(fig)


def output_hash_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(OUT)
        if "raw_gears" in relative.parts:
            continue
        if relative == Path("reports/RUN_RECORD.md"):
            continue
        rows.append(
            {
                "path": str(relative),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def raw_artifact_hash_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            status = json.loads(
                child_status_path(panel, seed).read_text(encoding="utf-8")
            )
            groups = (
                (
                    "critical_output",
                    child_root(panel, seed),
                    status["critical_output_sha256"],
                ),
                (
                    "model",
                    child_root(panel, seed),
                    status["model_artifact_sha256"],
                ),
                (
                    "generated_cache",
                    panel_data_root(panel, seed),
                    status["cache_artifact_sha256"],
                ),
            )
            for artifact_type, root, artifacts in groups:
                for relative, expected_hash in sorted(artifacts.items()):
                    path = root / relative
                    observed = sha256_file(path)
                    rows.append(
                        {
                            "panel": panel,
                            "seed": seed,
                            "artifact_type": artifact_type,
                            "local_path": str(path),
                            "bytes": int(path.stat().st_size),
                            "sha256": observed,
                            "status_hash": expected_hash,
                            "hash_match": observed == expected_hash,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty or not frame.hash_match.all():
        raise RuntimeError("raw artifact hash manifest failed")
    return frame


def analyze() -> dict[str, Any]:
    ensure_dirs()
    verified_inputs = verify_recorded_inputs()
    provenance = git_provenance()
    if not provenance["e195_code_paths_clean"]:
        raise RuntimeError("E195 implementation changed after its input lock")
    single, family, invariants, lock_audit = load_native_outputs()
    support_audit = actual_support_audit()
    raw_artifacts = raw_artifact_hash_rows()
    systems = build_system_frame(family)
    association, deltas, routing, coverage, comparison = analyze_groups(
        systems, single
    )
    dynamic = dynamic_range(single, family)
    runtime = runtime_environment()

    single.to_csv(TABLES / "E195_SINGLE_SEED_TASKS.csv", index=False)
    family.to_csv(TABLES / "E195_FAMILY_TASKS.csv", index=False)
    association.to_csv(TABLES / "E195_ASSOCIATION.csv", index=False)
    deltas.to_csv(TABLES / "E195_PAIRED_SCORE_DELTAS.csv", index=False)
    routing.to_csv(TABLES / "E195_ROUTING_METRICS.csv", index=False)
    coverage.to_csv(TABLES / "E195_RISK_COVERAGE.csv", index=False)
    comparison.to_csv(TABLES / "E195_SYSTEM_COMPARISON.csv", index=False)
    dynamic.to_csv(TABLES / "E195_DYNAMIC_RANGE_AUDIT.csv", index=False)
    invariants.to_csv(TABLES / "E195_INVARIANT_AUDIT.csv", index=False)
    lock_audit.to_csv(TABLES / "E195_SCORE_LOCK_AUDIT.csv", index=False)
    support_audit.to_csv(TABLES / "E195_SUPPORT_EXPOSURE_AUDIT.csv", index=False)
    raw_artifacts.to_csv(TABLES / "E195_RAW_ARTIFACT_HASHES.csv", index=False)
    runtime.to_csv(TABLES / "E195_RUNTIME_ENVIRONMENT.csv", index=False)
    make_figure(family, association, routing, coverage)

    native_dynamic = dynamic[dynamic.score_name.str.contains("native_logvar")]
    all_runs_complete = all(
        completed(panel, seed) for panel in PANELS for seed in SEEDS
    )
    invariant_pass = bool(invariants["pass"].all())
    native_estimable = bool(native_dynamic.estimable.all())
    track_b_groups = systems.groupby(
        ["panel", "system", "arm", "score_name"]
    ).size()
    track_b_join_pass = bool((track_b_groups == 24).all())
    child_statuses = {
        (panel, seed): json.loads(
            child_status_path(panel, seed).read_text(encoding="utf-8")
        )
        for panel in PANELS
        for seed in SEEDS
    }
    initialization_distinct = all(
        len(
            {
                child_statuses[(panel, seed)]["initial_model_state_sha256"]
                for seed in SEEDS
            }
        )
        == len(SEEDS)
        for panel in PANELS
    )
    trained_changed = all(
        status["trained_model_state_sha256"]
        != status["initial_model_state_sha256"]
        for status in child_statuses.values()
    )
    score_lock_columns = [
        "score_lock_hash_chain",
        "truth_unlock_after_lock",
        "cell_task_order_match",
        "score_record_exact_match",
        "array_keys_match",
        "effect_vectors_finite",
    ]
    all_score_locks_verified = bool(
        lock_audit[score_lock_columns].astype(bool).all().all()
    )
    all_system_values_finite = bool(
        np.isfinite(systems[["risk", "error"]].to_numpy(float)).all()
    )
    system_macros_complete = bool(
        (comparison.status == "estimable").all()
        and (comparison.n_panels_valid == 2).all()
    )
    paired_utility_complete = bool(
        len(deltas) == 10
        and (deltas.utility_delta_bootstrap_valid == N_BOOT).all()
        and np.isfinite(
            deltas[
                [
                    "utility20_a",
                    "utility20_b",
                    "paired_utility20_delta_a_minus_b",
                    "utility_delta_ci95_lower",
                    "utility_delta_ci95_upper",
                ]
            ].to_numpy(float)
        ).all()
    )
    gates = {
        "recorded_inputs_reverified": len(verified_inputs) > 0,
        "e195_code_paths_clean": provenance["e195_code_paths_clean"],
        "six_runs_complete": all_runs_complete,
        "each_run_has_24_tasks": len(single) == 144,
        "two_panels_have_48_family_tasks": len(family) == 48,
        "pretruth_score_locks_verified": all_score_locks_verified,
        "actual_split_has_no_exact_single_leakage": bool(
            ~support_audit.exact_single_condition_in_train_or_val.any()
        ),
        "three_initializations_distinct_within_each_panel": initialization_distinct,
        "all_trained_states_differ_from_initial": trained_changed,
        "all_raw_artifact_hashes_match": bool(raw_artifacts.hash_match.all()),
        "native_logvar_estimable_all_runs": native_estimable,
        "truth_equal_across_seeds": bool(
            (
                family.truth_max_abs_diff_across_seeds.astype(float) <= 1e-7
            ).all()
        ),
        "family_identity_residual_le_1e-10": bool(
            (
                family.family_squared_identity_residual.astype(float) <= 1e-10
            ).all()
        ),
        "track_b_one_to_one": track_b_join_pass,
        "all_system_risk_and_error_values_finite": all_system_values_finite,
        "two_panel_macros_require_two_estimable_panels": system_macros_complete,
        "paired_utility_uses_shared_bootstrap_indices": paired_utility_complete,
        "all_invariants_pass": invariant_pass,
    }
    if not native_estimable:
        final_status = "NON_ESTIMABLE"
    elif all(gates.values()):
        final_status = "COMPLETE"
    else:
        final_status = "INTEGRITY_FAILURE"

    same_prediction_assoc = association[
        association.track == "A_family"
    ][
        [
            "panel",
            "score_name",
            "spearman",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    single_seed_assoc = association[
        (association.track == "A_single_seed")
        & association.score_name.isin(["native_logvar", "predicted_magnitude"])
    ][
        [
            "panel",
            "seed",
            "score_name",
            "spearman",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    same_prediction_utility = routing[
        (routing.system == "GEARS-UQ")
        & (routing.arm == "same_prediction_family_rms")
        & (routing.budget == 0.20)
    ][
        [
            "panel",
            "score_name",
            "oracle_normalized_utility",
            "utility_ci95_lower",
            "utility_ci95_upper",
        ]
    ]
    paired_same_prediction = deltas[
        (deltas.system == "GEARS-UQ")
        & (deltas.score_a == "native_logvar_mean")
    ][
        [
            "panel",
            "score_a",
            "score_b",
            "paired_spearman_delta_a_minus_b",
            "delta_ci95_lower",
            "delta_ci95_upper",
            "paired_utility20_delta_a_minus_b",
            "utility_delta_ci95_lower",
            "utility_delta_ci95_upper",
        ]
    ]
    support_rows: list[dict[str, Any]] = []
    for panel, group in support_audit.groupby("panel", sort=True):
        unique = group.drop_duplicates("condition")
        support_rows.append(
            {
                "panel": panel,
                "exact_single_leaks": int(
                    unique.exact_single_condition_in_train_or_val.sum()
                ),
                "tasks_with_double_history": int(
                    (
                        unique.n_train_or_val_double_conditions_containing_gene
                        > 0
                    ).sum()
                ),
                "n_tasks": len(unique),
                "double_history_conditions": int(
                    unique.n_train_or_val_double_conditions_containing_gene.sum()
                ),
            }
        )
    support_summary = pd.DataFrame(support_rows)
    prescribe_assoc = association[
        (association.track == "B_system")
        & (association.system == "PRESCRIBE")
        & association.score_name.isin(
            [
                "combined_confidence",
                "predicted_magnitude",
                "risk_combined",
            ]
        )
    ][
        [
            "panel",
            "arm",
            "score_name",
            "outcome_name",
            "spearman",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    prescribe_delta = deltas[deltas.system == "PRESCRIBE"][
        [
            "panel",
            "arm",
            "score_a",
            "score_b",
            "score_score_spearman",
            "paired_spearman_delta_a_minus_b",
            "delta_ci95_lower",
            "delta_ci95_upper",
        ]
    ]
    report = [
        "# E195｜GEARS 原生学习型误差代理直接复现",
        "",
        f"运行状态：**{final_status}**。",
        "",
        "E195 在两个事先固定的 Norman 面板上重新训练 3 个 GEARS-UQ 成员。"
        "每个成员先写入 prediction、native logvar 和 magnitude 的哈希锁，随后才读取"
        "测试真值。原生分数、seed 分歧和 magnitude 因而评价的是同一批 GEARS-UQ"
        "预测对自身误差的排序能力。",
        "",
        "## 同预测 family 结果",
        "",
        markdown_table(same_prediction_assoc),
        "",
        "Native logvar 在 P1/P2 都有中等正相关；magnitude 的点相关更高。"
        "这两个点估计本身不能证明差异，配对 bootstrap 结果如下。",
        "",
        markdown_table(paired_same_prediction),
        "",
        "## 初始化稳定性",
        "",
        markdown_table(single_seed_assoc),
        "",
        "Native logvar 的六个单 seed 相关波动更大，部分区间跨 0；magnitude 的"
        "六个点估计均为正且区间下界均高于 0。这里仍是每个面板 24 个任务的小样本"
        "结果，不能写成普遍优势。",
        "",
        "## 20% 复核预算",
        "",
        markdown_table(same_prediction_utility),
        "",
        "单分数区间用于描述各自效用；分数间结论只看上面的共享任务重采样配对差。"
        "Native、magnitude 与 seed disagreement 的相对次序在 P1/P2 发生变化，"
        "不能依据两面板 macro 点估计宣布稳定胜负。",
        "",
        "## 留出语义",
        "",
        markdown_table(support_summary),
        "",
        "Exact single condition 没有进入 train/validation，但部分目标基因曾以"
        "双扰动形式出现。因此 E195 是 condition holdout，不是 perturbation-gene "
        "cold start。",
        "",
        "## PRESCRIBE 终点敏感性",
        "",
        markdown_table(prescribe_assoc),
        "",
        markdown_table(prescribe_delta),
        "",
        "PRESCRIBE 的 combined confidence 与 magnitude 几乎同序，配对增量很小；"
        "RMSE 敏感性臂的方向又不同。该结果应写成分数高度冗余且依赖评价终点。",
        "",
        "## 解释边界",
        "",
        "- GEARS 的 uncertainty loss 不是完整高斯负对数似然；这里称为 native "
        "log-variance score，不称校准预测方差。",
        "- GEARS-UQ、GEARS-scGPT pair 和 PRESCRIBE 的预测器及误差终点不同；"
        "跨系统只比较排序、coverage 和复核效用，不比较原始误差大小。",
        "- P1/P2 已在旧实验中打开真值，E195 是 post-truth direct-competitor "
        "replication，不是新的盲测。",
        "- Seed disagreement 是 family RMS 平方恒等式的一部分；其经验相关较弱，"
        "不能包装成独立误差保证。",
        "- 原生分数若为常数或 NaN，按冻结合同保留并标为 NON_ESTIMABLE；"
        "相关为负不会被当作工程失败。",
        "",
        "![E195](../figures/E195_native_uq_comparison.png)",
    ]
    (REPORTS / "E195_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    interpretation = [
        "# E195 结果如何使用",
        "",
        "E195 支持“GEARS 原生 logvar 可作为候选风险信号”：两个面板的 family "
        "相关都为正。它不支持“已校准方差”或“稳定优于简单 magnitude”。Magnitude"
        "的相关点估计更高，但 paired 差值区间跨 0；固定 20% 预算的相对表现又随"
        "面板改变。",
        "",
        "Seed disagreement 的相关较弱，而且它参与 family RMS 恒等式，不能解释成"
        "独立保证。PRESCRIBE 的 combined confidence 与 magnitude 几乎同序，表现"
        "还依赖 effect-Pearson 或 RMSE 终点。",
        "",
        "GEARS-UQ、GEARS–scGPT pair 和 PRESCRIBE 不共享预测值与误差终点，跨系统"
        "并列表只能描述各自内部排序，不能宣称共同结果变量上的全面胜负。本实验是"
        "看过真值后的 condition-holdout 复现，不改变 E192 的 ABSTAIN。",
    ]
    (REPORTS / "E195_INTERPRETATION.md").write_text(
        "\n".join(interpretation) + "\n", encoding="utf-8"
    )

    status = {
        "experiment": "E195",
        "stage": "POSTTRUTH_DIRECT_COMPETITOR_REPLICATION",
        "status": final_status,
        "completed_at": now(),
        "n_single_seed_tasks": len(single),
        "n_family_tasks": len(family),
        "n_system_task_score_rows": len(systems),
        "n_association_rows": len(association),
        "n_bootstrap": N_BOOT,
        "max_truth_difference": float(
            family.truth_max_abs_diff_across_seeds.max()
        ),
        "max_family_identity_residual": float(
            family.family_squared_identity_residual.max()
        ),
        **provenance,
        "gates": gates,
        "claim_boundary": (
            "Post-truth same-prediction native-UQ replication; not a new blind "
            "validation and not a common-outcome dominance comparison."
        ),
    }
    (OUT / "E195_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hashes = output_hash_rows()
    run_lines = [
        "# E195 run record",
        "",
        f"- completed: {now()}",
        f"- Python: `{sys.executable}`",
        f"- final status: `{final_status}`",
        f"- bootstrap: {N_BOOT}",
        f"- git head: `{provenance['git_head']}`",
        f"- E195 code paths clean: `{provenance['e195_code_paths_clean']}`",
        f"- unrelated dirty entries recorded by hash: "
        f"`{provenance['working_tree_status_sha256']}`",
        f"- local raw-artifact entries: {len(raw_artifacts)}",
        "",
        "## Training commands",
        "",
    ]
    for panel in PANELS:
        for seed in SEEDS:
            wrapper = json.loads(
                wrapper_status_path(panel, seed).read_text(encoding="utf-8")
            )
            run_lines.extend(
                [
                    f"### {panel} seed {seed}",
                    "",
                    f"- elapsed seconds: {wrapper['elapsed_seconds']:.2f}",
                    f"- child status: `{wrapper['child_status'].get('status')}`",
                    f"- initial model hash: `{wrapper['child_status'].get('initial_model_state_sha256')}`",
                    f"- trained model hash: `{wrapper['child_status'].get('trained_model_state_sha256')}`",
                    "",
                    "```bash",
                    " ".join(map(str, wrapper["command"])),
                    "```",
                    "",
                ]
            )
    run_lines.extend(
        [
            "## Output hashes",
            "",
            "The run record excludes itself from the hash manifest to avoid a "
            "self-referential hash cycle.",
            "",
            markdown_table(hashes, digits=6),
        ]
    )
    (REPORTS / "RUN_RECORD.md").write_text(
        "\n".join(run_lines) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "analyze", "full"],
        default="full",
    )
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()
    if args.mode in {"prepare", "full"}:
        result = prepare()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"train", "full"}:
        train(rerun=args.rerun_complete)
    if args.mode in {"analyze", "full"}:
        result = analyze()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

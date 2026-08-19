#!/usr/bin/env python3
"""Develop a five-seed estimator on truth-blind E174 predictions only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = ROOT / "tools/scripts/run_e174_rotated_donor_panel_pretruth.py"
E174 = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
OUT = ROOT / "docs/实验结果/E175_e174_seed_extension_development_20260719"
PANELS = ("R01", "R02", "R03", "R04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
ORIGINAL_SEEDS = (3407, 3408, 3409)
EXTRA_SEEDS = (3410, 3411)
ALL_SEEDS = ORIGINAL_SEEDS + EXTRA_SEEDS
N_GENES = 512
BOOTSTRAPS = 2000
SCRIPT = Path(__file__).resolve()


class IntegrityFailure(RuntimeError):
    pass


def import_wrapper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e175_e174_wrapper", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import E174 wrapper: {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"uncommitted input: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def truth_blind_boundary() -> dict[str, Any]:
    status = json.loads((E174 / "PRETRUTH_ABORT_STATUS.json").read_text())
    required = {
        "status": "ABORTED_PRETRUTH_GATE",
        "heldout_donor_targeting_x_values_read": 0,
        "calibration_targeting_x_values_read": 0,
        "evaluation_targeting_x_values_read": 0,
        "calibration_runner_authorized": False,
        "final_evaluator_authorized": False,
    }
    mismatches = {
        key: {"expected": value, "observed": status.get(key)}
        for key, value in required.items()
        if status.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"E174 truth-blind abort boundary changed: {mismatches}")
    isolated = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E174")
    if any((isolated / panel / "F3A_calibration").exists() for panel in PANELS):
        raise IntegrityFailure("an E174 calibration truth directory exists")
    if any((isolated / panel / "F4_evaluation").exists() for panel in PANELS):
        raise IntegrityFailure("an E174 evaluation truth directory exists")
    return status


def run_panel(panel: str, device_name: str) -> dict[str, Any]:
    if panel not in PANELS:
        raise ValueError(panel)
    destination = OUT / "panel_runs" / panel
    if destination.exists():
        raise IntegrityFailure(f"append-only E175 panel output exists: {destination}")
    truth_blind_boundary()
    wrapper = import_wrapper()
    helper = wrapper.configure(wrapper.import_helper(), panel)
    asset_root = helper.DEFAULT_ASSETS
    head, branch, remote_heads, input_hashes = helper.formal_input_audit(asset_root)
    input_hashes.extend(
        [
            require_committed(SCRIPT, head),
            require_committed(E174 / "PRETRUTH_ABORT_STATUS.json", head),
            require_committed(
                E174 / f"pretruth_release/{panel}/PRETRUTH_GATE_SNAPSHOT.json", head
            ),
        ]
    )
    assets = helper.load_assets(asset_root)
    supervised, query, graph_audit = helper.build_graphs(assets)
    if int(
        graph_audit.loc[
            graph_audit.graph_role.eq("query")
            & graph_audit.task_id.isin(
                assets.tasks.loc[assets.tasks.donor_role.eq("test"), "task_id"]
            ),
            "contains_y",
        ].astype(bool).sum()
    ) != 0:
        raise IntegrityFailure("test query graph unexpectedly contains y")
    scoring = pd.read_csv(
        E174 / f"pretruth_release/{panel}/tables/PRETRUTH_SCORING_INTERFACE.csv",
        keep_default_na=False,
    )
    task_order = scoring.task_id.astype(str).tolist()
    genes = assets.panel.scgpt_token.astype(str).tolist()

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityFailure("registered CUDA device unavailable")
    arrays: dict[str, np.ndarray] = {}
    histories, audits = [], []
    for seed in EXTRA_SEEDS:
        sc, sc_history, sc_audit = helper.train_scgpt(
            seed, supervised, query, genes, device
        )
        if set(sc) != set(task_order):
            raise IntegrityFailure(f"scGPT seed {seed} task coverage changed")
        arrays[f"scGPT_seed{seed}"] = np.stack([sc[task] for task in task_order]).astype(
            np.float32
        )
        histories.append(sc_history)
        audits.append({"panel_id": panel, "seed": seed, "model": "scGPT", **sc_audit})
        del sc
        torch.cuda.empty_cache()

        ge, ge_history, ge_audit = helper.train_gears(
            seed,
            supervised,
            query,
            genes,
            assets.coexpression,
            device,
        )
        if set(ge) != set(task_order):
            raise IntegrityFailure(f"GEARS seed {seed} task coverage changed")
        arrays[f"GEARS_seed{seed}"] = np.stack([ge[task] for task in task_order]).astype(
            np.float32
        )
        histories.append(ge_history)
        audits.append({"panel_id": panel, "seed": seed, "model": "GEARS", **ge_audit})
        del ge
        torch.cuda.empty_cache()

    for name, matrix in arrays.items():
        if matrix.shape != (2160, N_GENES) or not np.isfinite(matrix).all():
            raise IntegrityFailure(f"invalid extra prediction array: {name}/{matrix.shape}")
    for sub in ("arrays", "tables", "reports"):
        (destination / sub).mkdir(parents=True, exist_ok=False)
    atomic_npz(destination / "arrays/EXTRA_SEED_PREDICTIONS.npz", arrays)
    atomic_csv(destination / "tables/TRAINING_HISTORY.csv", pd.concat(histories, ignore_index=True))
    atomic_csv(destination / "tables/MODEL_AUDIT.csv", pd.DataFrame(audits))
    atomic_csv(destination / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes + assets.input_hashes))
    primary = {
        path.relative_to(destination).as_posix(): sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    status = {
        "schema": "safeconf_e175_extra_seed_panel_status_v1",
        "experiment": "E175_e174_seed_extension_development",
        "panel_id": panel,
        "stage": "TRUTH_BLIND_EXTRA_SEED_TRAINING",
        "status": "COMPLETE",
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "extra_seeds": list(EXTRA_SEEDS),
        "prediction_tasks": len(task_order),
        "test_query_graphs_containing_y": 0,
        "e174_heldout_donor_targeting_x_values_read": 0,
        "e174_calibration_truth_used": False,
        "e174_evaluation_truth_used": False,
        "primary_output_sha256": primary,
        "deployment_authorized": False,
    }
    atomic_json(destination / "RUN_STATUS.json", status)
    atomic_bytes(
        destination / "reports/REPORT.md",
        (
            f"# E175 {panel} extra seeds\n\nSeeds 3410/3411 completed on immutable E174 F2 assets. "
            "Held-out donor targeting X and test error remain unread.\n"
        ).encode(),
    )
    return status


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1])


def kendall_w(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, float)
    ranks = np.stack([rankdata(row, method="average") for row in values])
    n, m = ranks.shape[1], ranks.shape[0]
    sums = ranks.sum(axis=0)
    numerator = 12.0 * float(np.sum((sums - m * (n + 1) / 2.0) ** 2))
    tie = 0.0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        tie += float(np.sum(counts**3 - counts))
    denominator = m * m * (n**3 - n) - m * tie
    return numerator / denominator if denominator > 0 else float("nan")


def stability(risks: np.ndarray, seed: int) -> dict[str, Any]:
    risks = np.asarray(risks, float)
    if risks.shape[0] != 5:
        raise IntegrityFailure("five-seed LOO gate requires five risk vectors")
    pairs = [(left, right) for left in range(5) for right in range(left + 1, 5)]
    correlations = np.asarray([spearman(risks[a], risks[b]) for a, b in pairs])
    median = float(np.median(correlations))
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        take = rng.integers(0, risks.shape[1], risks.shape[1])
        boot[draw] = float(np.median([spearman(risks[a, take], risks[b, take]) for a, b in pairs]))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_leave_one_seed_out_estimators": 5,
        "n_pairwise_correlations": len(pairs),
        "minimum_pairwise_spearman": float(correlations.min()),
        "median_pairwise_spearman": median,
        "maximum_pairwise_spearman": float(correlations.max()),
        "kendall_w": kendall_w(risks),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_ci95_lower": float(low),
        "bootstrap_ci95_upper": float(high),
        "passed": bool(median >= 0.5 and low > 0),
    }


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def verify_panel_output(panel: str, head: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = OUT / "panel_runs" / panel
    status_path = directory / "RUN_STATUS.json"
    status = json.loads(status_path.read_text())
    if (
        status.get("status") != "COMPLETE"
        or status.get("extra_seeds") != list(EXTRA_SEEDS)
        or status.get("e174_heldout_donor_targeting_x_values_read") != 0
    ):
        raise IntegrityFailure(f"{panel} extra-seed status changed")
    hashes = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        hashes.append(require_committed(path, head))
        relative = path.relative_to(directory).as_posix()
        if relative in status.get("primary_output_sha256", {}):
            if sha256_file(path) != status["primary_output_sha256"][relative]:
                raise IntegrityFailure(f"{panel} extra-seed output hash changed: {relative}")
    return status, hashes


def aggregate() -> dict[str, Any]:
    aggregate_out = OUT / "aggregate"
    if aggregate_out.exists():
        raise IntegrityFailure("append-only E175 aggregate exists")
    truth_blind_boundary()
    head = git_text("rev-parse", "HEAD")
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    remote_heads = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify aggregate inputs on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise IntegrityFailure(f"current HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    input_hashes = [require_committed(SCRIPT, head), require_committed(E174 / "PRETRUTH_ABORT_STATUS.json", head)]
    for panel in PANELS:
        _status, hashes = verify_panel_output(panel, head)
        input_hashes.extend(hashes)

    wrapper = import_wrapper()
    rows, comparisons = [], []
    for panel in PANELS:
        helper = wrapper.configure(wrapper.import_helper(), panel)
        release = E174 / "pretruth_release" / panel
        scoring = pd.read_csv(release / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
        with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as original:
            arrays = {
                f"{family}_seed{seed}": np.asarray(original[f"{family}_seed{seed}"], dtype=float)
                for family in ("scGPT", "GEARS")
                for seed in ORIGINAL_SEEDS
            }
        extra_path = OUT / f"panel_runs/{panel}/arrays/EXTRA_SEED_PREDICTIONS.npz"
        with np.load(extra_path, allow_pickle=False) as extra:
            arrays.update({name: np.asarray(extra[name], dtype=float) for name in extra.files})
        if any(matrix.shape != (2160, N_GENES) for matrix in arrays.values()):
            raise IntegrityFailure(f"{panel} five-seed prediction shape changed")
        train = scoring.donor_role.eq("train").to_numpy()
        z_context = scoring.z_context_train960.astype(float)
        z_support = scoring.z_log_support_train960.astype(float)
        loo_risk = []
        for omitted in ALL_SEEDS:
            retained = [seed for seed in ALL_SEEDS if seed != omitted]
            sc = np.mean(np.stack([arrays[f"scGPT_seed{seed}"] for seed in retained]), axis=0)
            ge = np.mean(np.stack([arrays[f"GEARS_seed{seed}"] for seed in retained]), axis=0)
            disagreement = np.sqrt(np.mean((sc - ge) ** 2, axis=1))
            z_disagreement = helper.zscore_exact(
                pd.Series(disagreement), pd.Series(disagreement[train])
            )
            loo_risk.append(-(z_context + z_support - z_disagreement).to_numpy(float))
        loo_risk = np.stack(loo_risk)
        test_indices = np.flatnonzero(scoring.donor_role.eq("test").to_numpy())
        test = scoring.iloc[test_indices].reset_index(drop=True)
        test_risk = loo_risk[:, test_indices]
        original_g4 = pd.read_csv(release / "tables/G4_SEED_STABILITY.csv")
        for state in STATES:
            for stratum, mask in {
                "all_200": test.culture_condition.eq(state).to_numpy(),
                "seen_160": (
                    test.culture_condition.eq(state)
                    & test.target_stratum.eq("DONOR_UNSEEN_ONLY")
                ).to_numpy(),
            }.items():
                result = stability(
                    test_risk[:, mask], stable_seed("E175", panel, state, stratum)
                )
                old = original_g4.loc[
                    original_g4.culture_condition.eq(state)
                    & original_g4.stratum.eq(stratum)
                ].iloc[0]
                rows.append(
                    {
                        "panel_id": panel,
                        "culture_condition": state,
                        "stratum": stratum,
                        "n_tasks": int(mask.sum()),
                        **result,
                    }
                )
                comparisons.append(
                    {
                        "panel_id": panel,
                        "culture_condition": state,
                        "stratum": stratum,
                        "three_seed_median_pairwise_spearman": float(old.median_pairwise_spearman),
                        "three_seed_bootstrap_ci95_lower": float(old.bootstrap_ci95_lower),
                        "three_seed_passed": bool(old.passed),
                        "five_seed_median_pairwise_spearman": result["median_pairwise_spearman"],
                        "five_seed_bootstrap_ci95_lower": result["bootstrap_ci95_lower"],
                        "five_seed_passed": result["passed"],
                        "median_spearman_gain": result["median_pairwise_spearman"]
                        - float(old.median_pairwise_spearman),
                    }
                )
    gate = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    all_passed = bool(len(gate) == 24 and gate.passed.astype(bool).all())
    decision = "FIVE_SEED_GATE_READY_FOR_NEW_TARGET_PROTOCOL" if all_passed else "FIVE_SEED_GATE_NOT_STABLE"
    for sub in ("tables", "reports"):
        (aggregate_out / sub).mkdir(parents=True, exist_ok=False)
    atomic_csv(aggregate_out / "tables/FIVE_SEED_LOO_G4.csv", gate)
    atomic_csv(aggregate_out / "tables/THREE_VS_FIVE_SEED_COMPARISON.csv", comparison)
    atomic_csv(aggregate_out / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
    status = {
        "schema": "safeconf_e175_seed_extension_aggregate_v1",
        "status": "COMPLETE",
        "decision": decision,
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "development_panels": list(PANELS),
        "seeds": list(ALL_SEEDS),
        "five_seed_g4_units": len(gate),
        "five_seed_g4_units_passed": int(gate.passed.astype(bool).sum()),
        "minimum_five_seed_median_pairwise_spearman": float(gate.median_pairwise_spearman.min()),
        "minimum_five_seed_ci95_lower": float(gate.bootstrap_ci95_lower.min()),
        "e174_heldout_donor_targeting_x_values_read": 0,
        "e174_calibration_truth_used": False,
        "e174_evaluation_truth_used": False,
        "new_target_truth_must_be_used_for_confirmation": True,
        "deployment_authorized": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(aggregate_out / "RUN_STATUS.json", status)
    report = f"""# E175 five-seed truth-blind development

E174 的 held-out donor targeting X、calibration truth 与 evaluation truth 仍全部未读。两个新增 seeds 3410/3411 与原 3407–3409 组成五 seed family；G4 改为五组 leave-one-seed-out four-seed family means，共 10 个 pairwise rank correlations。

24 个 panel×state×stratum 单元通过 {int(gate.passed.astype(bool).sum())}/24；最小 median pairwise Spearman 为 {gate.median_pairwise_spearman.min():.3f}，最小 bootstrap 95% CI 下界为 {gate.bootstrap_ci95_lower.min():.3f}。正式开发判定：`{decision}`。

即使全部通过，这也只说明五 seed 估计器可被冻结到另一批新目标；E174 本身已正式中止，不能重新命名为确认实验，也不能读取其真值来证明性能。
"""
    atomic_bytes(aggregate_out / "reports/E175_SEED_EXTENSION_REPORT.md", report.encode())
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--panel", choices=PANELS)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    result = aggregate() if args.aggregate else run_panel(args.panel, args.device)
    result = {**result, "wall_seconds_this_command": time.time() - started}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

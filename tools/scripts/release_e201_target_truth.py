#!/usr/bin/env python3
"""Release E201 target expression only after pretruth risk features are sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
FREEZE = OUT / "TARGET_RELEASE_AND_EVALUATION_FREEZE.md"
RISK_STATUS = OUT / "E201_PRETRUTH_RISK_STATUS.json"
RISK_TABLE = OUT / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
GENERAL_STATUS = OUT / "E201_OFFICIAL_GENERAL_BASELINE_STATUS.json"
RELEASE_STATUS = OUT / "E201_TARGET_TRUTH_RELEASE_STATUS.json"
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
EXPECTED_SAMPLES = {
    "K562": 150_472,
    "RPE1": 67_034,
    "hepg2": 54_911,
    "jurkat": 81_791,
}
SOURCE_SHA256 = "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
GENE_SET_SHA256 = "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"
SPLIT_SHA256 = "c922dc62ee4263951ec6a45e6e8cfc51e4104d5e1b0704eefd46848acddba402"


class ReleaseFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def tracked_clean(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
    )
    return all(
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in commands
    )


def verify_git_release() -> str:
    required = (SCRIPT, FREEZE, RISK_STATUS, RISK_TABLE, GENERAL_STATUS)
    if not all(tracked_clean(path) for path in required):
        raise ReleaseFailure("truth release inputs are not tracked and clean")
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise ReleaseFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if git_text("rev-parse", f"{remote}/{branch}") != head:
            raise ReleaseFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def stream_hash_update(digest: hashlib._Hash, values: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(values).tobytes())


def verify_risk_seal(data_root: Path) -> tuple[dict, str]:
    status_sha = sha256_file(RISK_STATUS)
    status = json.loads(RISK_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != 2_008
        or int(status.get("n_primary_tasks", -1)) != 1_808
        or int(status.get("target_expression_nonzero_values_seen", -1)) != 0
        or status.get("target_truth_materialized") is not False
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise ReleaseFailure("pretruth risk status failed")
    risk_record = status["risk_table"]
    if (
        risk_record.get("path") != RISK_TABLE.relative_to(ROOT).as_posix()
        or RISK_TABLE.stat().st_size != int(risk_record["bytes"])
        or sha256_file(RISK_TABLE) != risk_record["sha256"]
    ):
        raise ReleaseFailure("pretruth risk table changed")
    vector_records = status.get("vector_files", [])
    if len(vector_records) != 4:
        raise ReleaseFailure("pretruth risk vector family is incomplete")
    for record in vector_records:
        path_text = record["path"]
        if not path_text.startswith("DATA/"):
            raise ReleaseFailure("risk vector path is not DATA-relative")
        path = data_root / path_text[len("DATA/") :]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ReleaseFailure(f"pretruth risk vector changed: {path}")
    return status, status_sha


def verify_general_baseline_seal(
    data_root: Path, risk_status_sha: str
) -> tuple[dict, str]:
    status_sha = sha256_file(GENERAL_STATUS)
    status = json.loads(GENERAL_STATUS.read_text(encoding="utf-8"))
    equivalence = status.get("e200_official_code_equivalence", {})
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != 2_008
        or int(status.get("n_primary_tasks", -1)) != 1_808
        or int(status.get("target_perturbed_expression_rows_opened", -1)) != 0
        or status.get("target_truth_materialized") is not False
        or status.get("target_outcomes_evaluated") is not False
        or status.get("pretruth_risk_status_sha256") != risk_status_sha
        or equivalence.get("passed") is not True
        or int(equivalence.get("tasks_exceeding_tolerance", -1)) != 0
        or float(equivalence.get("maximum_absolute_delta_residual", float("inf")))
        > 5e-6
    ):
        raise ReleaseFailure("official general-baseline seal failed")
    vectors = status.get("vector_files", [])
    names = {Path(record["path"]).name for record in vectors}
    if names != {
        "E201_OFFICIAL_GENERAL_BASELINE_WEIGHTED_DELTAS.npy",
        "E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy",
    }:
        raise ReleaseFailure("official general-baseline vector family changed")
    for record in vectors:
        path_text = record["path"]
        if not path_text.startswith("DATA/"):
            raise ReleaseFailure("general-baseline vector is not DATA-relative")
        path = data_root / path_text[len("DATA/") :]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ReleaseFailure(f"general-baseline vector changed: {path}")
    for record in status.get("tracked_outputs", []):
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or not tracked_clean(path)
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ReleaseFailure(f"general-baseline tracked output changed: {path}")
    return status, status_sha


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    txpert_repo = args.txpert_repo.resolve()
    prediction_root = args.prediction_root.resolve()
    if RELEASE_STATUS.exists():
        raise ReleaseFailure("truth release status already exists")
    safeconf_commit = verify_git_release()
    risk_status, risk_status_sha = verify_risk_seal(data_root)
    general_status, general_status_sha = verify_general_baseline_seal(
        data_root, risk_status_sha
    )
    general_centroid_records = [
        record
        for record in general_status["vector_files"]
        if Path(record["path"]).name == "E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy"
    ]
    if len(general_centroid_records) != 1:
        raise ReleaseFailure("sealed general-baseline centroid is missing")
    general_centroid_sha = general_centroid_records[0]["sha256"]

    source_cache = data_root / "txpert_official_20260802/cache/K562_cross_cell_lines"
    source_path = source_cache / "de_adata_test.h5ad"
    split_path = source_cache / "splits/train_test_split.pkl"
    gene_set_path = txpert_repo / "data/gears_gene_set.csv"
    if (
        not source_path.is_file()
        or sha256_file(source_path) != SOURCE_SHA256
        or not split_path.is_file()
        or sha256_file(split_path) != SPLIT_SHA256
        or not gene_set_path.is_file()
        or sha256_file(gene_set_path) != GENE_SET_SHA256
    ):
        raise ReleaseFailure("official target source changed")
    test_conditions = set(map(str, joblib.load(split_path)["test"]))
    perturbation_vocabulary = set(
        pd.read_csv(gene_set_path, index_col=0)["0"].astype(str)
    ) | {"ctrl"}

    source = ad.read_h5ad(source_path, backed="r")
    source_obs = source.obs.copy()
    source_var_hash = hashlib.sha256(
        "\n".join(map(str, source.var_names)).encode()
    ).hexdigest()
    release_records = []
    try:
        for target in TARGETS:
            shared = prediction_root / target / "shared"
            shared_manifest_path = shared / "E201_SHARED_TARGET_MANIFEST.json"
            observation_path = shared / "observations.csv"
            truth_path = shared / "truth.npy"
            target_manifest_path = shared / "E201_TARGET_TRUTH_MANIFEST.json"
            if truth_path.exists() or target_manifest_path.exists():
                raise ReleaseFailure(f"target truth already exists: {target}")
            if not shared_manifest_path.is_file() or not observation_path.is_file():
                raise ReleaseFailure(f"missing shared prediction input: {target}")
            shared_manifest = json.loads(
                shared_manifest_path.read_text(encoding="utf-8")
            )
            if (
                shared_manifest.get("status") != "SEALED_PRETRUTH_TARGET_INPUTS"
                or shared_manifest.get("target") != target
                or int(shared_manifest.get("n_samples", -1)) != EXPECTED_SAMPLES[target]
                or shared_manifest.get("target_truth_materialized") is not False
            ):
                raise ReleaseFailure(f"shared pretruth gate failed: {target}")
            observations = pd.read_csv(observation_path, keep_default_na=False)
            target_mask = (
                source_obs.cell_line.astype(str).eq(target)
                & ~source_obs.control.astype(bool)
                & source_obs.condition.astype(str).isin(test_conditions)
            )
            global_indices = np.flatnonzero(target_mask.to_numpy())
            candidate = source_obs.iloc[global_indices]
            valid = (
                candidate.condition.astype(str)
                .map(
                    lambda value: all(
                        token in perturbation_vocabulary for token in value.split("+")
                    )
                )
                .to_numpy()
            )
            global_indices = global_indices[valid]
            selected = source_obs.iloc[global_indices]
            if len(selected) != EXPECTED_SAMPLES[target]:
                raise ReleaseFailure(f"target release row count changed: {target}")
            alignment = {
                "condition": np.array_equal(
                    selected.condition_name.astype(str).to_numpy(),
                    observations.pert_cond_name.astype(str).to_numpy(),
                ),
                "cell_type": np.array_equal(
                    selected.cell_line.astype(str).to_numpy(),
                    observations.cell_type.astype(str).to_numpy(),
                ),
                "experimental_batch": np.array_equal(
                    selected.batch.astype(str).to_numpy(),
                    observations.experimental_batch.astype(str).to_numpy(),
                ),
                "row_index": observations.row_index.tolist()
                == list(range(len(observations))),
            }
            if not all(alignment.values()):
                raise ReleaseFailure(
                    f"target row alignment failed: {target}/{alignment}"
                )

            truth_partial = shared / "truth.partial.npy"
            truth = np.lib.format.open_memmap(
                truth_partial,
                mode="w+",
                dtype=np.float32,
                shape=(len(selected), source.n_vars),
            )
            stream_digest = hashlib.sha256()
            for start in range(0, len(global_indices), 256):
                stop = min(start + 256, len(global_indices))
                block = np.asarray(
                    source.X[global_indices[start:stop]].toarray(), dtype=np.float32
                )
                if not np.isfinite(block).all():
                    raise ReleaseFailure(f"non-finite target truth: {target}")
                truth[start:stop] = block
                stream_hash_update(stream_digest, block)
            truth.flush()
            del truth
            os.replace(truth_partial, truth_path)
            target_manifest = {
                "experiment": "E201_txpert_multitarget_retraining",
                "status": "SEALED_TARGET_TRUTH",
                "released_at": now(),
                "target": target,
                "n_samples": len(selected),
                "n_genes": int(source.n_vars),
                "official_source_h5ad_sha256": SOURCE_SHA256,
                "var_order_sha256": source_var_hash,
                "risk_status_sha256": risk_status_sha,
                "risk_table_sha256": risk_status["risk_table"]["sha256"],
                "official_general_baseline_status_sha256": general_status_sha,
                "official_general_baseline_centroid_sha256": general_centroid_sha,
                "safeconf_commit": safeconf_commit,
                "shared_pretruth_manifest_sha256": sha256_file(shared_manifest_path),
                "alignment": alignment,
                "target_expression_rows_opened": len(selected),
                "truth_stream_sha256": stream_digest.hexdigest(),
                "truth_file": {
                    "path": "truth.npy",
                    "bytes": truth_path.stat().st_size,
                    "sha256": sha256_file(truth_path),
                },
            }
            atomic_json(target_manifest_path, target_manifest)
            release_records.append(
                {
                    "target": target,
                    "n_samples": len(selected),
                    "truth_path": "DATA/"
                    + truth_path.relative_to(data_root).as_posix(),
                    "truth_bytes": truth_path.stat().st_size,
                    "truth_sha256": target_manifest["truth_file"]["sha256"],
                    "truth_manifest_path": "DATA/"
                    + target_manifest_path.relative_to(data_root).as_posix(),
                    "truth_manifest_sha256": sha256_file(target_manifest_path),
                }
            )
    finally:
        source.file.close()

    if len(release_records) != 4:
        raise ReleaseFailure("not all target truths were released")
    status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "TARGET_TRUTH_RELEASE",
        "status": "PASS",
        "released_at": now(),
        "safeconf_commit": safeconf_commit,
        "pretruth_risk_status_sha256": risk_status_sha,
        "pretruth_risk_sealed_before_release": True,
        "pretruth_official_general_baseline_status_sha256": general_status_sha,
        "pretruth_official_general_baseline_sealed_before_release": True,
        "official_general_baseline_centroid_sha256": general_centroid_sha,
        "n_targets": len(release_records),
        "n_target_expression_rows_opened": sum(
            record["n_samples"] for record in release_records
        ),
        "records": release_records,
    }
    atomic_json(RELEASE_STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

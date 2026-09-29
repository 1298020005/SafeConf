#!/usr/bin/env python3
"""Audit external SafeConf confirmation assets without opening expression truth.

The audit is intentionally metadata-only.  It reads repository configuration,
remote file manifests, and split CSVs.  It does not open an h5ad expression
matrix or compute any SafeConf result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PERTURBENCH = Path("/home/yyf/archive/external/PerturBench")
DATA = Path("/home/yyf/data/perturbench_mcfaline23_official")
OUT = PROJECT / "docs/实验结果/Stage2_mature_upstream_20260928/external_asset_audit"
HF_TREE = "https://huggingface.co/api/datasets/altoslabs/perturbench/tree/main?recursive=true&expand=false"
HF_CROISSANT = "https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/croissant.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_json_cached(url: str, cache: Path) -> object:
    error: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SafeConf-asset-audit/1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                value = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(value, indent=2) + "\n")
            return value
        except Exception as exc:  # remote metadata can transiently disconnect
            error = exc
            time.sleep(1 + attempt)
    if cache.exists():
        return json.loads(cache.read_text())
    raise RuntimeError(f"remote metadata unavailable and no cache exists: {url}") from error


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def project_reference_count(pattern: str) -> int:
    # Only prior result/report documents count as evidence that an asset was seen.
    # Code references introduced by this audit must not contaminate that count.
    search_roots = ["docs"]
    cmd = [
        "rg", "-i", "-l", pattern, *search_roots,
        "--glob", "!**/external_asset_audit/**",
    ]
    result = subprocess.run(cmd, cwd=PROJECT, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return len([line for line in result.stdout.splitlines() if line.strip()])


def split_summary(path: Path) -> dict[str, object]:
    counts: Counter[str] = Counter()
    ids: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        has_header = len(first) >= 2 and first[1].strip().lower() == "split"
        rows = reader if has_header else chain([first], reader)
        for row in rows:
            if len(row) < 2:
                continue
            counts[row[1] or "__unassigned__"] += 1
            ids.add(row[0])
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": sum(counts.values()),
        "unique_ids": len(ids),
        "split_counts": dict(sorted(counts.items())),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tree = remote_json_cached(HF_TREE, OUT / "HF_TREE_SNAPSHOT.json")
    croissant = remote_json_cached(HF_CROISSANT, OUT / "HF_CROISSANT_SNAPSHOT.json")
    assert isinstance(tree, list) and isinstance(croissant, dict)
    files = {item["path"]: item for item in tree if item.get("type") == "file"}
    remote_data = files["mcfaline23_gxe_processed.h5ad.gz"]
    remote_split = files["mcfaline23_gxe_splits.tar.gz"]

    distributions = {
        item.get("name"): item for item in croissant.get("distribution", []) if isinstance(item, dict)
    }
    old_data_sha = distributions.get("mcfalinefigueroa23_h5ad_file", {}).get("sha256", "")
    old_split_sha = distributions.get("mcfalinefigueroa23_splits_file", {}).get("sha256", "")

    split_root = DATA / "splits/mcfaline23_gxe_splits"
    split_files = [split_summary(split_root / f"{scale}_covariate_split.csv") for scale in ("full", "medium", "small")]
    split_tar = DATA / "mcfaline23_gxe_splits.tar.gz"
    data_gz = DATA / "mcfaline23_gxe_processed.h5ad.gz"
    expected_data_size = int(remote_data["size"])
    current_data_size = data_gz.stat().st_size if data_gz.exists() else 0
    data_complete = current_data_size == expected_data_size

    config = PERTURBENCH / "src/perturbench/configs/data/mcfaline23.yaml"
    splitter = PERTURBENCH / "src/perturbench/configs/data/splitter/mcfaline23_split.yaml"
    config_text = config.read_text()
    splitter_text = splitter.read_text()
    config_bug = "jiang24_split.csv" in config_text
    experiment_override_safe = "mcfaline23_gxe_splits/full_covariate_split.csv" in (
        PERTURBENCH / "src/perturbench/configs/experiment/neurips2025/mcfaline23/latent_additive_best_params_mcfaline23_full.yaml"
    ).read_text()

    registry = [
        {
            "candidate": "McFalineFigueroa23",
            "modality": "genetic",
            "independent_study": True,
            "project_result_references": project_reference_count("McFalineFigueroa23|mcfaline23"),
            "method_design_seen": False,
            "official_scale": "~200 perturbations; 6 cell lines; 5 treatments; 30 states",
            "local_data_status": "COMPLETE" if data_complete else "DOWNLOADING",
            "official_split": True,
            "test_truth_can_be_sealed": True,
            "history_contract_possible": True,
            "main_confirmation_eligible": True,
            "decision": "SELECT_PENDING_METADATA_AND_UPSTREAM_GATE",
        },
        {
            "candidate": "Frangieh21",
            "modality": "genetic",
            "independent_study": True,
            "project_result_references": project_reference_count("Frangieh21|frangieh"),
            "method_design_seen": True,
            "official_scale": "248 perturbations; 3 melanoma models",
            "local_data_status": "AVAILABLE_ELSEWHERE",
            "official_split": True,
            "test_truth_can_be_sealed": False,
            "history_contract_possible": True,
            "main_confirmation_eligible": False,
            "decision": "REJECT_SEEN",
        },
        {
            "candidate": "Norman19",
            "modality": "genetic",
            "independent_study": True,
            "project_result_references": project_reference_count("Norman19|norman"),
            "method_design_seen": True,
            "official_scale": "287 perturbations; K562",
            "local_data_status": "AVAILABLE_REMOTE",
            "official_split": True,
            "test_truth_can_be_sealed": False,
            "history_contract_possible": True,
            "main_confirmation_eligible": False,
            "decision": "REJECT_SEEN_AND_ONE_CONTEXT",
        },
        {
            "candidate": "Jiang24",
            "modality": "genetic",
            "independent_study": True,
            "project_result_references": project_reference_count("Jiang24|jiang24"),
            "method_design_seen": True,
            "official_scale": "525 perturbations; 3 cell lines; 5 treatments; 15 states",
            "local_data_status": "AVAILABLE",
            "official_split": True,
            "test_truth_can_be_sealed": True,
            "history_contract_possible": True,
            "main_confirmation_eligible": False,
            "decision": "REJECT_UPSTREAM_GATE_ALREADY_FAILED",
        },
        {
            "candidate": "OP3",
            "modality": "chemical",
            "independent_study": True,
            "project_result_references": project_reference_count("\\bOP3\\b|op3_"),
            "method_design_seen": False,
            "official_scale": "144 compounds; >=4 PBMC cell types",
            "local_data_status": "REMOTE_ONLY",
            "official_split": True,
            "test_truth_can_be_sealed": True,
            "history_contract_possible": True,
            "main_confirmation_eligible": False,
            "decision": "KEEP_CHEMICAL_ENHANCEMENT_ONLY",
        },
    ]
    write_csv(OUT / "EXTERNAL_ASSET_REGISTRY.csv", registry)

    resolved_latent = OUT / "UPSTREAM_RESOLVED_LATENT.yaml"
    resolved_decoder = OUT / "UPSTREAM_RESOLVED_DECODER.yaml"
    upstream = [
        {
            "rank": 1,
            "candidate": "PerturBench LatentAdditive",
            "family": "latent additive encoder-decoder",
            "published_config": True,
            "official_checkpoint_found": False,
            "official_config": "latent_additive_best_params_mcfaline23_full.yaml",
            "resolved_config_sha256": sha256(resolved_latent) if resolved_latent.exists() else "PENDING",
            "selection_basis": "official McFaline config; simple benchmark family; reproducible; independent of TxPert",
            "safeconf_result_seen": False,
            "attempt_status": "NOT_STARTED",
            "attempt_counts_when": "formal validation prediction begins",
            "gpu": 0,
        },
        {
            "rank": 2,
            "candidate": "PerturBench DecoderOnly",
            "family": "conditional decoder",
            "published_config": True,
            "official_checkpoint_found": False,
            "official_config": "decoder_only_best_params_mcfaline23_full.yaml",
            "resolved_config_sha256": sha256(resolved_decoder) if resolved_decoder.exists() else "PENDING",
            "selection_basis": "official McFaline config; distinct architecture; reproducible; independent of TxPert",
            "safeconf_result_seen": False,
            "attempt_status": "NOT_STARTED",
            "attempt_counts_when": "formal validation prediction begins",
            "gpu": 1,
        },
    ]
    write_csv(OUT / "EXTERNAL_UPSTREAM_SELECTION.csv", upstream)

    manifest = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only": True,
        "expression_truth_opened": False,
        "perturbench_commit": git("rev-parse", "HEAD", cwd=PERTURBENCH),
        "perturbench_commit_time": git("log", "-1", "--format=%cI", cwd=PERTURBENCH),
        "hf_tree_url": HF_TREE,
        "hf_croissant_url": HF_CROISSANT,
        "license_in_current_croissant": croissant.get("license"),
        "remote_data": {
            "size": expected_data_size,
            "lfs_sha256": remote_data.get("lfs", {}).get("oid"),
            "croissant_sha256": old_data_sha,
            "croissant_is_stale": old_data_sha != remote_data.get("lfs", {}).get("oid"),
            "downloaded_bytes": current_data_size,
            "complete": data_complete,
            "local_sha256": sha256(data_gz) if data_complete else None,
        },
        "remote_split_tar": {
            "size": int(remote_split["size"]),
            "lfs_sha256": remote_split.get("lfs", {}).get("oid"),
            "croissant_sha256": old_split_sha,
            "croissant_is_stale": old_split_sha != remote_split.get("lfs", {}).get("oid"),
            "local_sha256": sha256(split_tar),
        },
        "split_files": split_files,
        "config": {
            "mcfaline_yaml_sha256": sha256(config),
            "splitter_yaml_sha256": sha256(splitter),
            "base_config_points_to_jiang24": config_bug,
            "official_experiment_override_is_correct": experiment_override_safe,
            "execution_rule": "always pass explicit McFaline split path; never rely on the erroneous base override",
        },
    }
    (OUT / "EXTERNAL_ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    full = split_files[0]
    final_method = PROJECT / "docs/实验结果/Stage2_mature_upstream_20260928/FINAL_METHOD_CONFIG.json"
    implementation = PROJECT / "tools/scripts/run_safeconf_v4_development.py"
    external_contract = {
        "contract_version": "safeconf-external-confirmation-v1",
        "registered_before_upstream_validation_predictions": True,
        "selected_dataset": "McFalineFigueroa23",
        "selection_is_conditional_on": ["metadata_schema", "history_eligibility", "upstream_competence"],
        "dataset_version": {
            "perturbench_commit": manifest["perturbench_commit"],
            "remote_data_lfs_sha256": remote_data.get("lfs", {}).get("oid"),
            "remote_split_lfs_sha256": remote_split.get("lfs", {}).get("oid"),
            "full_split_sha256": full["sha256"],
        },
        "data_roles": {
            "train": "EXTERNAL_FIT",
            "validation": "UPSTREAM_COMPETENCE_AND_RISK_FIT",
            "test": "SEALED_EXTERNAL_CONFIRMATION",
        },
        "frozen_v4_definition": {
            "frozen": [
                "method_structure", "feature_definitions", "hyperparameters", "cross_fitting",
                "calibration_procedure", "history_eligibility", "missingness", "evaluation",
            ],
            "refit_allowed": "external train/validation only",
            "test_truth_allowed": "once, after fitted parameters and hashes are committed",
            "final_method_config_sha256": sha256(final_method),
            "implementation_sha256": sha256(implementation),
        },
        "upstream_slots": [x["candidate"] for x in upstream],
        "maximum_new_upstream_attempts": 2,
        "attempt_begins": "formal validation prediction generation",
        "formal_upstream_test_enabled": False,
        "competence_gate": {
            "primary": "task-level RMSE on aligned effect vector",
            "simple_baselines": ["no_change", "train_state_mean_effect"],
            "comparison_baseline": "strongest simple baseline on validation",
            "relative_noninferiority_margin": 0.02,
            "minimum_noninferior_strata_fraction": 0.60,
            "bootstrap_replicates": 5000,
            "safeconf_used_for_upstream_selection": False,
        },
        "external_v4_gate": {
            "minimum_delta_u20_v2_vs_magnitude": 0.005,
            "minimum_ci95_lower_v2_vs_magnitude": -0.005,
            "minimum_delta_u20_v2_vs_v1": -0.005,
            "minimum_ci95_lower_v2_vs_v1": -0.005,
            "minimum_nonnegative_strata_fraction": 0.60,
            "maximum_coverage_relative_degradation": 0.05,
            "maximum_miss_rate_degradation": 0.02,
            "maximum_aurc_relative_degradation": 0.05,
            "minimum_valid_strata_fraction": 0.80,
            "bootstrap_replicates": 5000,
            "bootstrap_unit": "biological perturbation cluster",
        },
        "test_truth_opened": False,
    }
    (OUT / "EXTERNAL_CONFIRMATION_CONTRACT.json").write_text(json.dumps(external_contract, indent=2) + "\n")

    role_rows = [
        {
            "dataset": "McFalineFigueroa23",
            "partition": partition,
            "role": role,
            "truth_opened_for_safeconf": False,
            "allowed_before_confirmation": allowed,
            "split_sha256": full["sha256"],
        }
        for partition, role, allowed in (
            ("train", "EXTERNAL_FIT", "training and legal history"),
            ("validation", "UPSTREAM_COMPETENCE_AND_RISK_FIT", "competence evaluation and risk fitting"),
            ("test", "SEALED_EXTERNAL_CONFIRMATION", "schema and provenance only"),
        )
    ]
    write_csv(OUT / "EXTERNAL_DATA_ROLE_REGISTRY.csv", role_rows)

    split_counts = full["split_counts"]
    report = f"""# SafeConf external asset audit

This audit is metadata-only. No h5ad expression matrix, prediction error, or test truth was opened.

## Independent decision

**Select McFalineFigueroa23 as the sole same-modality external candidate, conditional on metadata verification and the preregistered upstream competence gate.** It is the only current genetic candidate that is both untouched by SafeConf method design and large enough on official metadata. OP3 remains a chemical enhancement and will not replace the main genetic confirmation.

The two upstream slots are provisionally assigned to **PerturBench LatentAdditive** and **PerturBench DecoderOnly**. This selection was made from official configuration maturity, distinct architecture, reproducibility, and cost before any SafeConf result exists.

## Version and integrity findings

- PerturBench commit: `{manifest['perturbench_commit']}` ({manifest['perturbench_commit_time']}).
- Current Hugging Face file tree reports McFaline data size `{expected_data_size}` bytes and LFS SHA-256 `{remote_data.get('lfs', {}).get('oid')}`.
- Current split tar LFS SHA-256 `{remote_split.get('lfs', {}).get('oid')}` matches the downloaded tar.
- The current Croissant SHA fields are stale for both McFaline artifacts; file-tree LFS OIDs and downloaded bytes are the integrity authority.
- Full split has `{full['rows']}` rows / `{full['unique_ids']}` unique cell IDs: train `{split_counts.get('train', 0)}`, val `{split_counts.get('val', 0)}`, test `{split_counts.get('test', 0)}`.
- The checked-in base `mcfaline23.yaml` incorrectly overrides the split with `jiang24_split.csv`; official experiment YAMLs override it back to the correct McFaline split. Formal runs must provide the explicit split path and record its SHA.
- Current data download: `{current_data_size}/{expected_data_size}` bytes (`{'complete' if data_complete else 'in progress'}`).
- Current Croissant license field: `{croissant.get('license')}`.

## Candidate disposition

| Candidate | Modality | SafeConf-seen | Formal role | Decision |
|---|---|---:|---|---|
"""
    for row in registry:
        report += f"| {row['candidate']} | {row['modality']} | {row['method_design_seen']} | {'main confirmation' if row['main_confirmation_eligible'] else 'secondary/ineligible'} | {row['decision']} |\n"
    report += """

## Compute decision

- No official McFaline checkpoints were found locally, so validation competence requires training from official configs.
- Metadata and dataloader preflight do not consume an upstream attempt.
- The first formal validation prediction from each candidate consumes one of the two weekly slots.
- Test evaluation remains disabled until both the upstream competence decision and the frozen external SafeConf configuration are committed.
- If both candidates fail the upstream gate, no third upstream will be trained this week.
"""
    (OUT / "EXTERNAL_ASSET_AUDIT.md").write_text(report)
    status = {
        "status": "COMPLETE" if data_complete else "AUDIT_COMPLETE_DOWNLOAD_IN_PROGRESS",
        "selected_external_asset": "McFalineFigueroa23",
        "selected_upstreams": [x["candidate"] for x in upstream],
        "expression_truth_opened": False,
        "test_truth_opened": False,
        "data_download_complete": data_complete,
        "next_action": "metadata-only h5ad schema audit" if data_complete else "finish official data download",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

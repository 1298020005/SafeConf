#!/usr/bin/env python3
"""Freeze E160 Wessels requirements without reading expression values.

Formal execution is permitted only after this runner, the analysis contract,
the plain-text scGPT perturbation vocabulary, and the four E158/E159 evidence
inputs are committed and byte-identical to their current Git HEAD blobs.

The Wessels AnnData is opened in backed read-only mode.  This runner reads only
``obs``, ``var_names`` and shape.  It never indexes or materializes ``X``.  The
raw file is streamed once to calculate MD5 and SHA256 simultaneously; that is
an opaque integrity read rather than semantic expression access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "WesselsSatija2023.h5ad"
)
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
ORIGINAL_PERT_EMBEDDING = PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl"

OUT = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
VOCABULARY = OUT / "SCGPT_PERTURBATION_VOCABULARY.txt"
RUNNER = Path(__file__).resolve()
RELEASE = OUT / "freeze"
STAGING = OUT / ".freeze.staging"
STAGING_SENTINEL = ".E160_STAGING.json"

E158_STATUS = (
    ROOT
    / "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714/"
    "attempt_001/RUN_STATUS.json"
)
E158_UNSEAL = E158_STATUS.parent / "UNSEAL_EVENT.json"
E159_ROOT = ROOT / "docs/实验结果/E159_prescribe_saturation_forensics_20260714"
E159_STATUS = E159_ROOT / "RUN_STATUS.json"
E159_POSTHOC = E159_ROOT / "tables/E159_POSTHOC_SPEARMAN.csv"

E10_SCRIPT = ROOT / "tools/scripts/run_e10_external_dataset_asset_audit.py"
E10_TABLE = (
    ROOT
    / "docs/实验结果/E10_external_task_validation_assets_20260707/"
    "tables/E10_CANDIDATE_RANKING.csv"
)
E10_COVERAGE = E10_TABLE.parent / "E10_OFFICIAL_FILE_COVERAGE.csv"
E10_STATUS = E10_TABLE.parents[1] / "RUN_STATUS.json"
E40_SCRIPT = ROOT / "tools/scripts/build_e40_multidim_data_inventory.py"
E40_TABLE = (
    ROOT
    / "docs/实验结果/E40_non_tahoe_multidim_data_acquisition_20260709/"
    "tables/non_tahoe_local_inventory.csv"
)
E40_STATUS = E40_TABLE.parents[1] / "RUN_STATUS.json"

SELECTION_SEED = "E160|Wessels|seen-component-unseen-pair|20260714|v1"
BOOTSTRAP_SEED = 3407
N_BOOTSTRAP = 10_000
MIN_VALID_BOOTSTRAP = 9_500
PERCENTILE_METHOD = "linear"
MIN_PAIR_CELLS = 75
N_TEST = 48
N_VAL = 24
N_PAIR_TRAIN = 44

EXPECTED_SHAPE = (30_707, 21_052)
EXPECTED_RAW_BYTES = 219_393_529
EXPECTED_RAW_MD5 = "6897bfdcda928a678208fecf4eeb282e"
EXPECTED_RAW_CONDITIONS = 187
EXPECTED_COMPATIBLE_SINGLES = 27
EXPECTED_COMPATIBLE_PAIRS = 142
EXPECTED_ELIGIBLE_PAIRS = 116
EXPECTED_INCOMPATIBLE_PAIR_VPRBP = 16
EXPECTED_VOCABULARY_ENTRIES = 60_697
EXPECTED_VOCABULARY_SHA256 = (
    "27b822c871a7905d419529bdc9e28d6608e4b56a155bd875908b02cc171fb084"
)
EXPECTED_ORIGINAL_EMBEDDING_SHA256 = (
    "9a5be69676bc09fbf996ae7be1d4faa09c9f32abbf733f33fc130153829ad8ce"
)

REQUIRED_RAW_LOG_UNIQUE = 24
REQUIRED_RAW_LOG_SAMPLE_STD = 1e-6
REQUIRED_PREDICTION_VECTOR_UNIQUE = 24
REQUIRED_PREDICTION_COORD_SAMPLE_STD = 1e-6
REQUIRED_BASELINE_UNIQUE = 2
REQUIRED_BASELINE_SAMPLE_STD = 1e-12

COMMITTED_INPUTS = {
    "runner": RUNNER,
    "contract": CONTRACT,
    "scgpt_perturbation_vocabulary": VOCABULARY,
    "e158_attempt1_run_status": E158_STATUS,
    "e158_attempt1_unseal_event": E158_UNSEAL,
    "e159_run_status": E159_STATUS,
    "e159_posthoc_spearman": E159_POSTHOC,
}

PAYLOAD_RELATIVE_PATHS = (
    "manifests/E160_CONDITION_AUDIT.csv",
    "manifests/E160_ELIGIBLE_PAIRS_SHA_ORDER.csv",
    "manifests/E160_SPLIT.csv",
    "manifests/E160_set2conditions.json",
    "manifests/E160_SOURCE_HASHES.csv",
    "manifests/E160_PRIOR_ACCESS_AUDIT.csv",
    "manifests/E160_PRIOR_EVIDENCE_AUDIT.json",
)
RELEASE_RELATIVE_PATHS = frozenset((*PAYLOAD_RELATIVE_PATHS, "RUN_STATUS.json"))
OUT_ROOT_ALLOWLIST = frozenset(
    {CONTRACT.name, VOCABULARY.name, RELEASE.name, STAGING.name}
)

PROHIBITED_EXPRESSION_READER_TOKENS = (
    "read_h5ad(",
    "scanpy",
    "anndata",
    "h5py",
    ".X",
    "['X']",
    '["X"]',
    "layers[",
)

EXPECTED_POSTHOC_RHOS = {
    ("Norman_P3", "pearson_effect_accuracy"): 0.17739130434782607,
    ("Norman_P4", "pearson_effect_accuracy"): 0.2904347826086956,
    ("Norman_P3", "raw_pearson_effect_accuracy_sensitivity"): -0.27304347826086955,
    ("Norman_P4", "raw_pearson_effect_accuracy_sensitivity"): -0.49999999999999994,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild every expected payload in memory and verify the frozen release.",
    )
    parser.add_argument(
        "--recover-staging",
        action="store_true",
        help="Remove only a validated E160 staging directory, then rebuild from scratch.",
    )
    args = parser.parse_args()
    if args.verify and args.recover_staging:
        parser.error("--verify and --recover-staging are mutually exclusive")
    return args


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_rank(condition: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\t{condition}".encode("utf-8")).hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git_output(*args: str, binary: bool = False) -> str | bytes:
    output = subprocess.check_output(["git", *args], cwd=ROOT)
    return output if binary else output.decode("utf-8").strip()


def committed_input_gate(head: str | None = None) -> dict[str, Any]:
    """Verify fixed inputs against ``head`` before raw hashing or AnnData open."""

    if head is None:
        head = str(git_output("rev-parse", "HEAD"))
    else:
        resolved = str(git_output("rev-parse", f"{head}^{{commit}}"))
        if resolved != head:
            raise RuntimeError(f"Frozen E160 git head is not a full commit id: {head}")
    files: dict[str, dict[str, Any]] = {}
    for role, path in COMMITTED_INPUTS.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Committed input must be a regular non-symlink file: {path}")
        relative = repo_relative(path)
        try:
            blob = git_output("show", f"{head}:{relative}", binary=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Missing committed HEAD blob for {role}: {relative}") from exc
        assert isinstance(blob, bytes)
        working = path.read_bytes()
        if working != blob:
            raise RuntimeError(f"{role} differs from its committed HEAD blob: {relative}")
        files[role] = {
            "path": relative,
            "sha256": sha256_bytes(working),
            "bytes": len(working),
            "matches_git_head_blob": True,
        }
    return {"git_head": head, "files": files}


def verification_gate_from_frozen_status(
    existing_status: dict[str, Any],
) -> dict[str, Any]:
    """Keep formal provenance stable while checking it also exists at current HEAD."""

    frozen = existing_status.get("committed_input_gate")
    if not isinstance(frozen, dict):
        raise RuntimeError("Frozen status has no committed_input_gate object")
    frozen_head = frozen.get("git_head")
    if not isinstance(frozen_head, str) or len(frozen_head) != 40:
        raise RuntimeError("Frozen status has no valid 40-character formal git head")

    formal_gate = committed_input_gate(frozen_head)
    if formal_gate != frozen:
        raise RuntimeError(
            "Frozen committed-input provenance does not reconstruct at the formal git head"
        )

    current_gate = committed_input_gate()
    if current_gate["files"] != formal_gate["files"]:
        raise RuntimeError(
            "A fixed E160 input differs between the formal and current git heads"
        )
    return formal_gate


def identity_tuple(value: os.stat_result) -> tuple[int, int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_size), int(value.st_mtime_ns))


def stream_raw_integrity() -> dict[str, Any]:
    """Calculate MD5 and SHA256 in one pass and capture the open-file identity."""

    link_stat = RAW.lstat()
    if stat.S_ISLNK(link_stat.st_mode) or not stat.S_ISREG(link_stat.st_mode):
        raise RuntimeError("Wessels raw must be a regular non-symlink file")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    total = 0
    with RAW.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
            total += len(block)
        after = os.fstat(handle.fileno())
    if identity_tuple(before) != identity_tuple(after):
        raise RuntimeError("Raw file identity changed during the one-pass hash")
    if total != int(before.st_size) or total != EXPECTED_RAW_BYTES:
        raise RuntimeError(f"Raw byte count changed: {total} != {EXPECTED_RAW_BYTES}")
    if md5.hexdigest() != EXPECTED_RAW_MD5:
        raise RuntimeError(
            f"Raw MD5 mismatch: {md5.hexdigest()} != {EXPECTED_RAW_MD5}"
        )
    result = {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "bytes": total,
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
    }
    assert_raw_identity(result, "immediately_after_hash")
    return result


def assert_raw_identity(expected: dict[str, Any], phase: str) -> None:
    current = RAW.lstat()
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise RuntimeError(f"Raw became non-regular or symlink at {phase}")
    observed = (int(current.st_dev), int(current.st_ino), int(current.st_size), int(current.st_mtime_ns))
    wanted = (
        int(expected["device"]),
        int(expected["inode"]),
        int(expected["bytes"]),
        int(expected["mtime_ns"]),
    )
    if observed != wanted:
        raise RuntimeError(f"Raw identity changed at {phase}: {observed} != {wanted}")


def load_frozen_vocabulary() -> tuple[set[str], str]:
    payload = VOCABULARY.read_bytes()
    if sha256_bytes(payload) != EXPECTED_VOCABULARY_SHA256:
        raise RuntimeError("Frozen plain-text perturbation vocabulary SHA256 changed")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Perturbation vocabulary is not UTF-8") from exc
    if len(lines) != EXPECTED_VOCABULARY_ENTRIES:
        raise RuntimeError("Perturbation vocabulary row count changed")
    if any(not value or value != value.strip() or value != value.upper() for value in lines):
        raise RuntimeError("Perturbation vocabulary must contain normalized uppercase tokens")
    if lines != sorted(lines) or len(lines) != len(set(lines)):
        raise RuntimeError("Perturbation vocabulary must be sorted and duplicate-free")

    original_hash = sha256_file(ORIGINAL_PERT_EMBEDDING)
    if original_hash != EXPECTED_ORIGINAL_EMBEDDING_SHA256:
        raise RuntimeError("Original scGPT perturbation embedding hash changed")
    return set(lines), original_hash


def validate_prior_evidence(gate: dict[str, Any]) -> dict[str, Any]:
    e158 = json.loads(E158_STATUS.read_text(encoding="utf-8"))
    unseal = json.loads(E158_UNSEAL.read_text(encoding="utf-8"))
    e159 = json.loads(E159_STATUS.read_text(encoding="utf-8"))
    posthoc = pd.read_csv(E159_POSTHOC)

    if e158.get("phase") != "failed_after_irreversible_test_unseal_preserve_attempt":
        raise RuntimeError("E158 attempt-1 is not the frozen failed-after-unseal attempt")
    if e158.get("test_data_unsealed") is not True:
        raise RuntimeError("E158 test_data_unsealed must be true")
    if e158.get("test_X_rows_materialized") is not True:
        raise RuntimeError("E158 test_X_rows_materialized must be true")
    if "Only 0 valid bootstrap replicates" not in str(e158.get("error", "")):
        raise RuntimeError("E158 frozen failure reason changed")
    if unseal.get("irreversible_record") is not True:
        raise RuntimeError("E158 unseal event is not marked irreversible")
    if unseal.get("event") != "E158_one_time_test_expression_unseal_after_both_E157_locks":
        raise RuntimeError("E158 unseal event name changed")
    unseal_sha = gate["files"]["e158_attempt1_unseal_event"]["sha256"]
    if e158.get("unseal_event_sha256") != unseal_sha:
        raise RuntimeError("E158 status no longer points to the committed unseal event")

    if e159.get("phase") != "complete_post_unseal_saturation_forensics":
        raise RuntimeError("E159 is not the frozen post-unseal forensic run")
    if e159.get("analysis_timing") != "post_unseal_forensic_not_preregistered":
        raise RuntimeError("E159 timing disclosure changed")
    findings = e159.get("forensic_findings", {})
    required_findings = {
        "all_official_and_magnitude_scores_constant_within_panel": True,
        "all_ten_predicted_pca_coordinates_constant_within_panel": True,
        "E158_primary_gate_estimable": False,
        "E158_primary_gate_result": "failed_non_estimable_constant_score",
        "undefined_rho_zero_filled": False,
        "P3_P4_can_be_rescued_posthoc": False,
        "raw_log_prob_analysis_role": "posthoc_hypothesis_generating_only",
    }
    for key, expected in required_findings.items():
        if findings.get(key) != expected:
            raise RuntimeError(f"E159 frozen forensic finding changed for {key}")

    observed_rhos: dict[str, float] = {}
    for (scope, endpoint), expected in EXPECTED_POSTHOC_RHOS.items():
        row = posthoc[
            posthoc["scope"].astype(str).eq(scope)
            & posthoc["score"].astype(str).eq("raw_log_prob")
            & posthoc["endpoint"].astype(str).eq(endpoint)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing unique E159 post-hoc row: {scope}/{endpoint}")
        item = row.iloc[0]
        if str(item["analysis_timing"]) != "post_unseal_forensic_not_preregistered":
            raise RuntimeError("E159 post-hoc row timing changed")
        if str(item["interpretation_constraint"]) != "hypothesis_generating_only":
            raise RuntimeError("E159 post-hoc interpretation constraint changed")
        rho = float(item["rho"])
        if not math.isclose(rho, expected, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"E159 post-hoc rho changed for {scope}/{endpoint}")
        observed_rhos[f"{scope}|{endpoint}"] = rho

    return {
        "E158_attempt1_phase": e158["phase"],
        "E158_test_data_unsealed": True,
        "E158_test_X_rows_materialized": True,
        "E158_failure": e158["error"],
        "E158_unseal_event_sha256": unseal_sha,
        "E159_phase": e159["phase"],
        "E159_analysis_timing": e159["analysis_timing"],
        "E159_official_combined_magnitude_and_PCA_predictions_saturated": True,
        "E159_raw_log_prob_role": "posthoc_hypothesis_generating_only",
        "E159_posthoc_rhos": observed_rhos,
        "E160_hypothesis_disclosure": (
            "new Wessels hypothesis jointly motivated after unseal by positive PCA10 "
            "rhos and negative raw-full-gene rhos; not retrospectively preregistered"
        ),
    }


def canonical_condition(raw_condition: str, nperts: int) -> tuple[str, tuple[str, ...]]:
    raw_value = str(raw_condition).strip()
    if nperts == 0:
        if raw_value.lower() != "control":
            raise RuntimeError(f"nperts=0 is not control: {raw_value!r}")
        return "ctrl", ()
    if raw_value != raw_value.upper():
        raise RuntimeError(f"Non-uppercase Wessels perturbation label: {raw_value!r}")
    if nperts == 1:
        if not raw_value or "_" in raw_value or "+" in raw_value:
            raise RuntimeError(f"Malformed Wessels single: {raw_value!r}")
        return f"{raw_value}+ctrl", (raw_value,)
    if nperts == 2:
        genes = tuple(sorted(part.upper() for part in raw_value.split("_") if part))
        if len(genes) != 2 or genes[0] == genes[1]:
            raise RuntimeError(f"Malformed Wessels pair: {raw_value!r}")
        return "+".join(genes), genes
    raise RuntimeError(f"Unsupported nperts={nperts} for {raw_value!r}")


def metadata_condition_audit(vocabulary: set[str]) -> tuple[pd.DataFrame, tuple[int, int]]:
    backed = ad.read_h5ad(RAW, backed="r")
    try:
        required = {"perturbation", "nperts", "Guide.Class"}
        if not required.issubset(backed.obs.columns):
            raise RuntimeError(f"Missing obs columns: {sorted(required - set(backed.obs.columns))}")
        obs = backed.obs[["perturbation", "nperts", "Guide.Class"]].copy()
        var_names = {str(gene).strip().upper() for gene in backed.var_names}
        shape = (int(backed.n_obs), int(backed.n_vars))
    finally:
        backed.file.close()

    if shape != EXPECTED_SHAPE:
        raise RuntimeError(f"Wessels shape changed: {shape} != {EXPECTED_SHAPE}")
    if obs.isna().any().any():
        raise RuntimeError("Wessels task metadata contains missing values")
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["Guide.Class"] = obs["Guide.Class"].astype(str)
    obs["nperts"] = pd.to_numeric(obs["nperts"], errors="raise").astype(int)

    if (obs.groupby("perturbation", observed=True)["nperts"].nunique() != 1).any():
        raise RuntimeError("A raw perturbation maps to multiple nperts values")
    counts = (
        obs.groupby(["perturbation", "nperts"], observed=True)
        .size()
        .rename("n_cells_obs_metadata")
        .reset_index()
    )
    guide_classes = (
        obs.groupby("perturbation", observed=True)["Guide.Class"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .to_dict()
    )
    expected_class = {0: "NT", 1: "Single", 2: "Dual"}

    rows: list[dict[str, Any]] = []
    for record in counts.itertuples(index=False):
        raw_condition = str(record.perturbation)
        nperts = int(record.nperts)
        guide_class = guide_classes[raw_condition]
        if guide_class != expected_class.get(nperts):
            raise RuntimeError(
                f"Guide.Class/nperts mismatch for {raw_condition}: {guide_class}/{nperts}"
            )
        canonical, genes = canonical_condition(raw_condition, nperts)
        in_vocabulary = all(gene in vocabulary for gene in genes)
        in_var = all(gene in var_names for gene in genes)
        compatible = in_vocabulary and in_var
        n_cells = int(record.n_cells_obs_metadata)
        eligible_pair = nperts == 2 and compatible and n_cells >= MIN_PAIR_CELLS
        if nperts == 0:
            exclusion = ""
        elif not in_vocabulary and not in_var:
            exclusion = "gene_missing_from_scgpt_vocabulary_and_expression_var"
        elif not in_vocabulary:
            exclusion = "gene_missing_from_scgpt_vocabulary"
        elif not in_var:
            exclusion = "gene_missing_from_expression_var"
        elif nperts == 2 and n_cells < MIN_PAIR_CELLS:
            exclusion = f"pair_below_{MIN_PAIR_CELLS}_cells"
        else:
            exclusion = ""
        rows.append(
            {
                "raw_condition": raw_condition,
                "canonical_condition": canonical,
                "perturbation_genes": ";".join(genes),
                "n_perturbation_genes": nperts,
                "guide_class": guide_class,
                "n_cells_obs_metadata": n_cells,
                "all_genes_in_scgpt_vocabulary": in_vocabulary,
                "all_genes_in_expression_var_metadata": in_var,
                "model_compatible": compatible,
                "eligible_pair_min75": eligible_pair,
                "selection_sha256": sha_rank(canonical) if eligible_pair else "",
                "exclusion_reason": exclusion,
            }
        )

    audit = pd.DataFrame(rows).sort_values("canonical_condition").reset_index(drop=True)
    if len(audit) != EXPECTED_RAW_CONDITIONS:
        raise RuntimeError(f"Raw condition count changed: {len(audit)}")
    if audit["canonical_condition"].duplicated().any():
        raise RuntimeError("Canonical condition collision")
    if int(audit["n_cells_obs_metadata"].sum()) != shape[0]:
        raise RuntimeError("Condition cell counts do not sum to n_obs")
    return audit, shape


def freeze_split(
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, Any]]:
    control = audit[audit["n_perturbation_genes"].eq(0)]
    singles = audit[audit["n_perturbation_genes"].eq(1) & audit["model_compatible"]]
    pairs = audit[audit["n_perturbation_genes"].eq(2) & audit["model_compatible"]]
    eligible = audit[audit["eligible_pair_min75"]].copy()

    if len(control) != 1 or control.iloc[0]["canonical_condition"] != "ctrl":
        raise RuntimeError("Expected exactly one ctrl")
    if len(singles) != EXPECTED_COMPATIBLE_SINGLES:
        raise RuntimeError(f"Compatible singles changed: {len(singles)}")
    if len(pairs) != EXPECTED_COMPATIBLE_PAIRS:
        raise RuntimeError(f"Compatible pairs changed: {len(pairs)}")
    if len(eligible) != EXPECTED_ELIGIBLE_PAIRS:
        raise RuntimeError(f"Eligible >=75-cell pairs changed: {len(eligible)}")

    vprbp = audit[
        audit["raw_condition"].eq("VPRBP") & audit["n_perturbation_genes"].eq(1)
    ]
    if len(vprbp) != 1:
        raise RuntimeError("Expected one VPRBP single")
    vprbp_row = vprbp.iloc[0]
    if bool(vprbp_row["all_genes_in_scgpt_vocabulary"]):
        raise RuntimeError("VPRBP unexpectedly exists in frozen scGPT vocabulary")
    if bool(vprbp_row["all_genes_in_expression_var_metadata"]):
        raise RuntimeError("VPRBP unexpectedly exists in expression var axis")
    incompatible_pairs = audit[
        audit["n_perturbation_genes"].eq(2) & ~audit["model_compatible"]
    ]
    if len(incompatible_pairs) != EXPECTED_INCOMPATIBLE_PAIR_VPRBP:
        raise RuntimeError("Unexpected incompatible pair count")
    for row in incompatible_pairs.itertuples(index=False):
        genes = str(row.perturbation_genes).split(";")
        if "VPRBP" not in genes:
            raise RuntimeError("An incompatible pair is not explained by VPRBP")
        if bool(row.all_genes_in_scgpt_vocabulary) or bool(
            row.all_genes_in_expression_var_metadata
        ):
            raise RuntimeError("A VPRBP pair is not jointly absent from vocabulary/var")

    eligible = eligible.sort_values(
        ["selection_sha256", "canonical_condition"], kind="stable"
    ).reset_index(drop=True)
    eligible["selection_rank"] = range(1, len(eligible) + 1)
    eligible["split"] = "train"
    eligible.loc[: N_TEST - 1, "split"] = "test"
    eligible.loc[N_TEST : N_TEST + N_VAL - 1, "split"] = "val"

    pair_test = sorted(eligible.loc[eligible["split"].eq("test"), "canonical_condition"])
    pair_val = sorted(eligible.loc[eligible["split"].eq("val"), "canonical_condition"])
    pair_train = sorted(eligible.loc[eligible["split"].eq("train"), "canonical_condition"])
    single_conditions = sorted(singles["canonical_condition"].astype(str))
    single_genes = set(singles["perturbation_genes"].astype(str))
    for row in eligible.itertuples(index=False):
        genes = {gene for gene in str(row.perturbation_genes).split(";") if gene}
        if not genes.issubset(single_genes):
            raise RuntimeError(f"Pair components not covered by train singles: {row.canonical_condition}")

    split = {
        "train": ["ctrl", *single_conditions, *pair_train],
        "val": pair_val,
        "test": pair_test,
    }
    sets = {role: set(values) for role, values in split.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("Condition split overlap")
    if (len(pair_test), len(pair_val), len(pair_train)) != (N_TEST, N_VAL, N_PAIR_TRAIN):
        raise RuntimeError("Pair split is not 48/24/44")

    assignment = {condition: role for role, values in split.items() for condition in values}
    audit = audit.copy()
    audit["split"] = audit["canonical_condition"].map(assignment).fillna("excluded")
    audit["selection_rank"] = audit["canonical_condition"].map(
        eligible.set_index("canonical_condition")["selection_rank"]
    )
    split_rows = []
    for role in ("train", "val", "test"):
        for condition in split[role]:
            condition_type = (
                "control" if condition == "ctrl" else "single" if condition.endswith("+ctrl") else "pair"
            )
            split_rows.append(
                {
                    "panel": "Wessels_E160",
                    "split": role,
                    "condition": condition,
                    "condition_type": condition_type,
                }
            )
    split_frame = pd.DataFrame(split_rows)
    metadata = {
        "n_raw_conditions": int(len(audit)),
        "n_control_conditions": 1,
        "n_compatible_singles_all_train": int(len(single_conditions)),
        "n_compatible_pairs": int(len(pairs)),
        "n_eligible_pairs_min75": int(len(eligible)),
        "n_pair_train": int(len(pair_train)),
        "n_pair_val": int(len(pair_val)),
        "n_pair_test": int(len(pair_test)),
        "n_total_train_conditions": int(len(split["train"])),
        "all_test_pair_components_seen_as_train_singles": True,
        "split_task_overlap": 0,
        "vprbp_absent_from_vocabulary_and_expression_var": True,
        "n_vprbp_pairs_excluded": int(len(incompatible_pairs)),
        "guide_class_nperts_mapping_strictly_verified": True,
        "canonical_gene_case": "uppercase",
    }
    return audit, eligible, split, split_frame, metadata


def validate_prior_asset_metadata() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    for stage, script, role in [
        ("E10", E10_SCRIPT, "path/size/official and legacy metadata inventory only"),
        ("E40", E40_SCRIPT, "precomputed h5ad_scan and download-status inventory only"),
    ]:
        source = script.read_text(encoding="utf-8")
        hits = sorted(token for token in PROHIBITED_EXPRESSION_READER_TOKENS if token in source)
        if hits:
            raise RuntimeError(f"{stage} script has expression-reader tokens: {hits}")
        rows.append(
            {
                "stage": stage,
                "script": repo_relative(script),
                "script_sha256": sha256_file(script),
                "expression_reader_token_hits": "",
                "documented_role": role,
                "wessels_prediction_score_or_error_generated": False,
            }
        )

    e10 = pd.read_csv(E10_TABLE)
    item = e10[e10["actual_file_name"].astype(str).eq(RAW.name)]
    if len(item) != 1:
        raise RuntimeError("E10 must contain one WesselsSatija2023 metadata row")
    item = item.iloc[0]
    if (int(item["n_obs"]), int(item["n_vars"])) != EXPECTED_SHAPE:
        raise RuntimeError("E10 Wessels shape changed")
    if str(item["recommended_role"]) != "background_asset":
        raise RuntimeError("E10 Wessels role changed")

    coverage = pd.read_csv(E10_COVERAGE)
    coverage = coverage[coverage["actual_file_name"].astype(str).eq(RAW.name)]
    if len(coverage) != 1:
        raise RuntimeError("E10 coverage must contain one WesselsSatija2023 row")
    coverage = coverage.iloc[0]
    if str(coverage["official_checksum"]) != f"md5:{EXPECTED_RAW_MD5}":
        raise RuntimeError("E10 official Wessels MD5 changed")
    if int(coverage["official_size_bytes"]) != EXPECTED_RAW_BYTES:
        raise RuntimeError("E10 official Wessels byte size changed")
    if str(coverage["size_check_status"]) != "size_match":
        raise RuntimeError("E10 Wessels size-match status changed")

    e40 = pd.read_csv(E40_TABLE)
    e40 = e40[e40["study_family"].astype(str).eq("WesselsSatija2023")]
    if len(e40) != 1 or int(e40.iloc[0]["total_cells_or_obs"]) != EXPECTED_SHAPE[0]:
        raise RuntimeError("E40 Wessels metadata row changed")

    sources = []
    for role, path, access in [
        ("E10_candidate_metadata", E10_TABLE, "metadata row only"),
        ("E10_official_coverage", E10_COVERAGE, "official checksum/size metadata only"),
        ("E10_status", E10_STATUS, "asset-audit status only"),
        ("E40_inventory_metadata", E40_TABLE, "study inventory row only"),
        ("E40_status", E40_STATUS, "inventory status only"),
    ]:
        sources.append(source_row(role, path, sha256_file(path), "", access, False))
    return pd.DataFrame(rows), sources


def source_row(
    role: str,
    path: Path | str,
    sha256: str,
    md5: str,
    access: str,
    git_verified: bool,
    size: int | None = None,
) -> dict[str, Any]:
    path_value = str(path)
    if isinstance(path, Path) and path.is_relative_to(ROOT):
        path_value = repo_relative(path)
    if size is None and isinstance(path, Path):
        size = path.stat().st_size
    return {
        "source_role": role,
        "path": path_value,
        "sha256": sha256,
        "md5": md5,
        "bytes": int(size or 0),
        "content_access": access,
        "git_head_blob_verified": git_verified,
    }


def source_manifest(
    gate: dict[str, Any],
    raw_integrity: dict[str, Any],
    original_embedding_sha: str,
    prior_sources: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for role, record in gate["files"].items():
        rows.append(
            source_row(
                role,
                ROOT / record["path"],
                record["sha256"],
                "",
                "committed fixed input",
                True,
                int(record["bytes"]),
            )
        )
    rows.append(
        source_row(
            "wessels_raw_container",
            RAW,
            raw_integrity["sha256"],
            raw_integrity["md5"],
            "one-pass opaque MD5+SHA256 plus obs/var/shape; X values not indexed",
            False,
            raw_integrity["bytes"],
        )
    )
    rows.append(
        source_row(
            "original_scgpt_perturbation_embedding_hash_only",
            ORIGINAL_PERT_EMBEDDING,
            original_embedding_sha,
            "",
            "byte hash only; no pickle deserialization",
            False,
        )
    )
    for role, path in [
        ("E10_inventory_script", E10_SCRIPT),
        ("E40_inventory_script", E40_SCRIPT),
    ]:
        rows.append(source_row(role, path, sha256_file(path), "", "static source audit", False))
    rows.extend(prior_sources)
    return pd.DataFrame(rows)


def coverage_requirements() -> list[dict[str, Any]]:
    values = [round(value / 100, 2) for value in range(50, 101, 5)]
    return [
        {"coverage": value, "required_retained_n": int(math.ceil(value * N_TEST))}
        for value in values
    ]


def artifact_payloads(
    audit: pd.DataFrame,
    eligible: pd.DataFrame,
    split: dict[str, list[str]],
    split_frame: pd.DataFrame,
    sources: pd.DataFrame,
    prior_access: pd.DataFrame,
    prior_evidence: dict[str, Any],
) -> dict[str, bytes]:
    candidate_columns = [
        "canonical_condition",
        "perturbation_genes",
        "n_cells_obs_metadata",
        "selection_sha256",
        "selection_rank",
        "split",
    ]
    return {
        "manifests/E160_CONDITION_AUDIT.csv": audit.to_csv(index=False).encode("utf-8"),
        "manifests/E160_ELIGIBLE_PAIRS_SHA_ORDER.csv": eligible[candidate_columns]
        .to_csv(index=False)
        .encode("utf-8"),
        "manifests/E160_SPLIT.csv": split_frame.to_csv(index=False).encode("utf-8"),
        "manifests/E160_set2conditions.json": json_bytes(split),
        "manifests/E160_SOURCE_HASHES.csv": sources.to_csv(index=False).encode("utf-8"),
        "manifests/E160_PRIOR_ACCESS_AUDIT.csv": prior_access.to_csv(index=False).encode(
            "utf-8"
        ),
        "manifests/E160_PRIOR_EVIDENCE_AUDIT.json": json_bytes(prior_evidence),
    }


def build_status(
    executed_at: str,
    gate: dict[str, Any],
    raw_integrity: dict[str, Any],
    shape: tuple[int, int],
    split_metadata: dict[str, Any],
    prior_evidence: dict[str, Any],
    payloads: dict[str, bytes],
) -> dict[str, Any]:
    return {
        "experiment": "E160_wessels_combination_contract",
        "phase": "requirements_frozen_test_expression_unopened",
        "frozen_at": "2026-07-14",
        "executed_at": executed_at,
        "requirements_only_not_gate_results": True,
        "dataset": "WesselsSatija2023",
        "raw_path": str(RAW),
        "raw_integrity": raw_integrity,
        "dataset_shape": list(shape),
        "metadata_only_selection": True,
        "ann_data_backed_read_only": True,
        "raw_file_streamed_once_for_simultaneous_hashes": True,
        "raw_X_values_indexed_or_materialized": False,
        "layers_opened_by_formal_runner": False,
        "prediction_score_or_error_used_for_selection": False,
        "pre_design_reconnaissance_20260714": {
            "obs_var_shape_inspected": True,
            "X_storage_type_and_dtype_metadata_inspected": "CSR int64 raw counts",
            "layers_presence_metadata_inspected": "no layers",
            "X_values_read": False,
            "hardcoded_counts_source": "pre-design metadata/storage-type reconnaissance",
        },
        "selection_seed": SELECTION_SEED,
        "minimum_pair_cells": MIN_PAIR_CELLS,
        "canonicalization": "uppercase; control->ctrl; single->GENE+ctrl; sorted pair joined by +",
        "required_main_truth": {
            "name": "train_only_PCA10_inverse_transform_effect",
            "task_mean": "mu_t = mean_i_in_task(log1p(10000*x_i/sum_g x_i)) on selected genes",
            "control_mean": "mu_ctrl = train-control cell mean after per-cell normalization/log1p",
            "pca_truth": "z_t=(mu_t-m_G)W^T; effect=(z_t W+m_G)-mu_ctrl",
            "prediction_effect": "effect_hat=(z_hat_t W+m_G)-mu_ctrl",
            "accuracy": "Pearson(effect_hat, PCA10-inverse-transform effect)",
        },
        "required_raw_full_gene_truth_sensitivity": {
            "truth_effect": "mu_t-mu_ctrl on the full selected-gene axis",
            "accuracy": "Pearson(effect_hat, raw full-selected-gene truth effect)",
            "mandatory_reporting": True,
        },
        "primary_score": "PRESCRIBE raw_log_prob; larger means higher confidence",
        "raw_log_prob_hypothesis_timing": prior_evidence["E160_hypothesis_disclosure"],
        "required_raw_log_prob_gate": {
            "required_rows": N_TEST,
            "required_one_row_per_test_task": True,
            "required_all_finite": True,
            "required_minimum_exact_unique_values": REQUIRED_RAW_LOG_UNIQUE,
            "required_sample_std_ddof": 1,
            "required_sample_std_strictly_greater_than": REQUIRED_RAW_LOG_SAMPLE_STD,
        },
        "required_prediction_non_degeneracy_gate": {
            "required_shape": [N_TEST, 10],
            "required_all_480_values_finite": True,
            "required_minimum_exact_unique_prediction_vectors": REQUIRED_PREDICTION_VECTOR_UNIQUE,
            "required_at_least_one_coordinate_sample_std_ddof": 1,
            "required_at_least_one_coordinate_sample_std_strictly_greater_than": (
                REQUIRED_PREDICTION_COORD_SAMPLE_STD
            ),
        },
        "required_baseline_estimability": {
            "baselines": [
                "official_combined_confidence=2*epistemic+aleatoric",
                "predicted_magnitude_rms",
            ],
            "required_rows": N_TEST,
            "required_all_finite": True,
            "required_minimum_exact_unique_values": REQUIRED_BASELINE_UNIQUE,
            "required_sample_std_ddof": 1,
            "required_sample_std_strictly_greater_than": REQUIRED_BASELINE_SAMPLE_STD,
            "failure_output": "NA with constant_or_nonfinite_baseline; never zero-fill",
        },
        "required_main_endpoint": {
            "statistic": "Spearman(raw_log_prob, PCA10 pearson_effect_accuracy)",
            "expected_direction": "positive",
            "constant_rule": "nonfinite or unique<2 or sample_std<=1e-12 => rho/CI NA and main gate fails",
            "confirmation": "point rho>0 and task-bootstrap 95% CI lower>0",
        },
        "required_bootstrap": {
            "rng": "numpy.random.default_rng",
            "seed": BOOTSTRAP_SEED,
            "replicates": N_BOOTSTRAP,
            "task_resample_size": N_TEST,
            "paired_indices_for_rawlog_and_baselines": True,
            "paired_delta": "rho_raw_log_prob - rho_baseline",
            "ci": "numpy.quantile([0.025,0.975], method='linear')",
            "percentile_method": PERCENTILE_METHOD,
            "minimum_valid_replicates": MIN_VALID_BOOTSTRAP,
        },
        "required_gene_cluster_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": N_BOOTSTRAP,
            "algorithm": (
                "sample K component genes with replacement; for every sampled gene append all "
                "test tasks containing it once; compute Spearman on the resulting task multiset"
            ),
            "role": "dependency sensitivity; does not replace primary task bootstrap",
        },
        "required_leave_one_gene_out": {
            "algorithm": "for every component gene remove all test pairs containing it and recompute rho",
            "required_outputs": "gene, removed_n, remaining_n, rho, estimability; min/median/max and positive fraction",
        },
        "required_secondary_endpoints": [
            "PCA10 frac_correct_direction_all and top20 true-effect direction",
            "PCA10 rmse_effect_error",
            "raw full-selected-gene Pearson/direction/RMSE sensitivity",
        ],
        "required_coverage": {
            "ranking": "raw_log_prob descending; canonical condition ascending for ties",
            "retained_n_rule": "ceil(coverage*48)",
            "grid": coverage_requirements(),
        },
        "prior_evidence_disclosure": prior_evidence,
        "committed_input_gate": gate,
        **split_metadata,
        "atomic_publication": {
            "staging": STAGING.name,
            "release": RELEASE.name,
            "explicit_allowlist": sorted(RELEASE_RELATIVE_PATHS),
            "symlink_or_unknown_path_policy": "reject",
            "recovery": "explicit --recover-staging with valid sentinel and allowlist only",
        },
        "artifact_sha256": {
            f"freeze/{relative}": sha256_bytes(payload) for relative, payload in payloads.items()
        },
    }


def assert_tree_allowlist(
    root: Path, allowed_files: set[str] | frozenset[str], require_exact: bool
) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Expected regular directory, not symlink: {root}")
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Symlink rejected inside {root}: {relative}")
        if path.is_file():
            if relative not in allowed_files:
                raise RuntimeError(f"Unknown file rejected inside {root}: {relative}")
            observed_files.add(relative)
        elif path.is_dir():
            prefix = f"{relative}/"
            if not any(item.startswith(prefix) for item in allowed_files):
                raise RuntimeError(f"Unknown directory rejected inside {root}: {relative}")
        else:
            raise RuntimeError(f"Non-regular path rejected inside {root}: {relative}")
    if require_exact and observed_files != set(allowed_files):
        raise RuntimeError(
            f"Release allowlist mismatch: missing={sorted(set(allowed_files)-observed_files)}, "
            f"extra={sorted(observed_files-set(allowed_files))}"
        )
    return observed_files


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree_directories(root: Path) -> None:
    """Persist every staging directory entry before the atomic rename."""

    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(root)


def recover_staging() -> None:
    allowed = set(RELEASE_RELATIVE_PATHS) | {STAGING_SENTINEL}
    observed = assert_tree_allowlist(STAGING, allowed, require_exact=False)
    if STAGING_SENTINEL not in observed:
        raise RuntimeError("Refusing staging recovery without the E160 sentinel")
    sentinel = json.loads((STAGING / STAGING_SENTINEL).read_text(encoding="utf-8"))
    expected = {
        "experiment": "E160_wessels_combination_contract",
        "purpose": "atomic_freeze_staging",
        "target": "freeze",
    }
    if sentinel != expected:
        raise RuntimeError("Refusing staging recovery: sentinel content mismatch")
    shutil.rmtree(STAGING)
    fsync_directory(OUT)


def inspect_output_root(verify: bool, allow_recovery: bool) -> None:
    if OUT.is_symlink() or not OUT.is_dir():
        raise RuntimeError(f"E160 output root must be a regular directory: {OUT}")
    entries = {path.name: path for path in OUT.iterdir()}
    unknown = set(entries) - set(OUT_ROOT_ALLOWLIST)
    if unknown:
        raise RuntimeError(f"Unknown E160 root paths rejected: {sorted(unknown)}")
    for required in (CONTRACT, VOCABULARY):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(f"Required E160 committed file is invalid: {required}")

    if STAGING.exists() or STAGING.is_symlink():
        if STAGING.is_symlink():
            raise RuntimeError("E160 staging symlink rejected")
        if not allow_recovery:
            raise RuntimeError(
                "E160 staging remains; inspect it and use --recover-staging explicitly"
            )
        recover_staging()

    if verify:
        if not RELEASE.exists():
            raise FileNotFoundError(RELEASE)
        assert_tree_allowlist(RELEASE, RELEASE_RELATIVE_PATHS, require_exact=True)
    elif RELEASE.exists() or RELEASE.is_symlink():
        raise FileExistsError("E160 freeze already exists; formal overwrite is forbidden")


def write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError(f"Symlink parent rejected: {path.parent}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_atomic(payloads: dict[str, bytes], status_bytes: bytes) -> None:
    STAGING.mkdir(mode=0o755, exist_ok=False)
    sentinel = {
        "experiment": "E160_wessels_combination_contract",
        "purpose": "atomic_freeze_staging",
        "target": "freeze",
    }
    write_new_file(STAGING / STAGING_SENTINEL, json_bytes(sentinel))
    for relative, payload in payloads.items():
        write_new_file(STAGING / relative, payload)
    write_new_file(STAGING / "RUN_STATUS.json", status_bytes)

    allowed_with_sentinel = set(RELEASE_RELATIVE_PATHS) | {STAGING_SENTINEL}
    assert_tree_allowlist(STAGING, allowed_with_sentinel, require_exact=True)
    for relative, expected in {**payloads, "RUN_STATUS.json": status_bytes}.items():
        if (STAGING / relative).read_bytes() != expected:
            raise RuntimeError(f"Staging payload reconstruction mismatch: {relative}")
    (STAGING / STAGING_SENTINEL).unlink()
    assert_tree_allowlist(STAGING, RELEASE_RELATIVE_PATHS, require_exact=True)
    fsync_tree_directories(STAGING)
    os.replace(STAGING, RELEASE)
    fsync_directory(OUT)


def verify_release(payloads: dict[str, bytes], expected_status_bytes: bytes) -> None:
    assert_tree_allowlist(RELEASE, RELEASE_RELATIVE_PATHS, require_exact=True)
    for relative, payload in payloads.items():
        if (RELEASE / relative).read_bytes() != payload:
            raise RuntimeError(f"Frozen payload differs byte-for-byte: {relative}")
    if (RELEASE / "RUN_STATUS.json").read_bytes() != expected_status_bytes:
        raise RuntimeError("Frozen RUN_STATUS differs from the complete reconstructed payload")


def main() -> None:
    args = parse_args()

    # These checks intentionally occur before raw hashing or AnnData open.
    existing_status_bytes: bytes | None = None
    existing_status: dict[str, Any] | None = None
    if args.verify:
        inspect_output_root(verify=True, allow_recovery=False)
        existing_status_bytes = (RELEASE / "RUN_STATUS.json").read_bytes()
        existing_status = json.loads(existing_status_bytes.decode("utf-8"))
        gate = verification_gate_from_frozen_status(existing_status)
    else:
        gate = committed_input_gate()
        inspect_output_root(
            verify=False, allow_recovery=args.recover_staging
        )

    prior_evidence = validate_prior_evidence(gate)
    vocabulary, original_embedding_sha = load_frozen_vocabulary()
    raw_integrity = stream_raw_integrity()
    assert_raw_identity(raw_integrity, "before_anndata_metadata_open")
    prior_access, prior_sources = validate_prior_asset_metadata()
    audit, shape = metadata_condition_audit(vocabulary)
    assert_raw_identity(raw_integrity, "after_anndata_metadata_close")
    audit, eligible, split, split_frame, split_metadata = freeze_split(audit)
    sources = source_manifest(gate, raw_integrity, original_embedding_sha, prior_sources)
    payloads = artifact_payloads(
        audit,
        eligible,
        split,
        split_frame,
        sources,
        prior_access,
        prior_evidence,
    )
    if set(payloads) != set(PAYLOAD_RELATIVE_PATHS):
        raise RuntimeError("Internal E160 payload allowlist mismatch")
    assert_raw_identity(raw_integrity, "before_release_or_verification")

    if args.verify:
        assert existing_status_bytes is not None
        assert existing_status is not None
        executed_at = existing_status.get("executed_at")
        if not isinstance(executed_at, str) or not executed_at:
            raise RuntimeError("Frozen status has no stable executed_at")
        expected_status = build_status(
            executed_at,
            gate,
            raw_integrity,
            shape,
            split_metadata,
            prior_evidence,
            payloads,
        )
        verify_release(payloads, json_bytes(expected_status))
        print(
            json.dumps(
                {
                    "phase": "full_payload_verification_passed",
                    "raw_X_values_indexed_or_materialized": False,
                    "n_pair_test": split_metadata["n_pair_test"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    executed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    status = build_status(
        executed_at,
        gate,
        raw_integrity,
        shape,
        split_metadata,
        prior_evidence,
        payloads,
    )
    publish_atomic(payloads, json_bytes(status))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build SafeConf governance artifacts without opening sealed confirmation values.

The script deliberately separates DEV/SEEN field audits from SEALED metadata.
For sealed candidates it records only path existence and cryptographic hashes of
the declared manifest files; it never reads task rows, feature values, vectors,
truth or result summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
PUB = Path("/home/yyf/proj/docs/实验结果")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing(path: Path | str) -> str:
    return str(Path(path).exists()).lower()


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_role_registry() -> None:
    # All published Stage2 assets have been inspected during prior development.
    # They are deliberately marked SEEN rather than retroactively promoted.
    rows: list[dict] = []
    assets = pd.read_csv(STAGE / "MATURE_ASSETS.csv")
    for _, row in assets.drop_duplicates(["asset", "predictor"]).iterrows():
        role = "SEEN"
        if str(row["asset"]).startswith("E84") or str(row["asset"]).startswith("E87") or str(row["asset"]).startswith("E89"):
            role = "SEEN"
        rows.append({
            "dataset": str(row["asset"]),
            "context": ";".join(sorted(set(assets.loc[assets.asset.eq(row.asset), "n_contexts"].astype(str)))),
            "task_range": str(int(row["n_unique_biological_tasks"])),
            "upstream_model": str(row["predictor"]),
            "past_experiment_ids": str(row["asset"]),
            "result_seen": "true",
            "method_design_influenced": "true",
            "role": role,
            "data_hash": sha256(Path(str(row["records_path"]))),
            "prediction_version": sha256(Path(str(row["prediction_archive"]))),
            "truth_version": sha256(Path(str(row["prediction_archive"])).with_name("true_effects.npz")),
            "metadata_only_until_freeze": "false",
            "notes": "Released retrospective development asset; not independent confirmation.",
        })

    # E201/E205 are the primary released development assets and were already
    # inspected in previous experiments. Register them explicitly rather than
    # letting their absence from MATURE_ASSETS imply an unknown role.
    for dataset, upstream, experiment, feature_path, vector_path, truth_path in [
        (
            "E201_TxPert_GAT", "TxPert_STRING_GAT", "E201;Stage2_txpert_risk_batch",
            ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_RISK_FEATURES.csv",
            Path("/home/yyf/data/txpert_official_20260802/e201/pretruth_vectors/E201_FAMILY_CENTROIDS.npy"),
            Path("/home/yyf/data/txpert_official_20260802/e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy"),
        ),
        (
            "E205_TxPert_Exphormer", "TxPert_Exphormer", "E205;Stage2_txpert_exphormer_risk_batch",
            ROOT / "docs/实验结果/E205_cross_family_disagreement_20260830/tables/E205_PRETRUTH_RISK_FEATURES.csv",
            Path("/home/yyf/data/txpert_official_20260802/e205/pretruth_vectors/E205_FAMILY_CENTROIDS.npy"),
            Path("/home/yyf/data/txpert_official_20260802/e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy"),
        ),
    ]:
        rows.append({
            "dataset": dataset,
            "context": "K562;RPE1;hepg2;jurkat",
            "task_range": "1808 primary_ge30",
            "upstream_model": upstream,
            "past_experiment_ids": experiment,
            "result_seen": "true",
            "method_design_influenced": "true",
            "role": "DEV",
            "data_hash": sha256(feature_path),
            "prediction_version": sha256(vector_path),
            "truth_version": sha256(truth_path),
            "metadata_only_until_freeze": "false",
            "notes": "Primary released development asset; never independent confirmation.",
        })

    # Existing 54-task GEARS formal output was explicitly inspected.
    gears_status = STAGE / "gears_risk_transfer/RUN_STATUS.json"
    rows.append({
        "dataset": "GEARS_formal_54",
        "context": "adamson;dixit;norman",
        "task_range": "54",
        "upstream_model": "GEARS",
        "past_experiment_ids": "Stage2_gears_risk_transfer",
        "result_seen": "true",
        "method_design_influenced": "true",
        "role": "SEEN",
        "data_hash": sha256(gears_status),
        "prediction_version": None,
        "truth_version": None,
        "metadata_only_until_freeze": "false",
        "notes": "Small retrospective development/transfer line; not confirmation.",
    })

    # E190 is an already-opened cross-study GEARS asset.  It is useful for
    # independent-family development evidence, but can never be relabelled as
    # confirmation because its target truth and earlier risk summaries were
    # inspected in July 2026.
    e190_public = PUB / "E190_adamson_to_replogle_direct_transfer_20260729"
    rows.append({
        "dataset": "E190_Adamson_to_Replogle_K562",
        "context": "Replogle K562;48 target batches",
        "task_range": "692 tasks;47 gene clusters",
        "upstream_model": "GEARS_3seed_centroid",
        "past_experiment_ids": "E190;E193;E197",
        "result_seen": "true",
        "method_design_influenced": "true",
        "role": "SEEN",
        "data_hash": sha256(e190_public / "E190_QUERY_MANIFEST.csv"),
        "prediction_version": sha256(e190_public / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz"),
        "truth_version": sha256(e190_public / "evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz"),
        "metadata_only_until_freeze": "false",
        "notes": "Already-opened cross-study asset. Eligible only for exploratory cross-family development evidence.",
    })

    # E216 was already formally evaluated on 2026-09-20.  Its result existed
    # before this v4 freeze and therefore must be SEEN, even though the sparse
    # mirror under /home/yyf/proj contained only the analysis contract.
    e216 = Path("/home/yyf/runtime_worktrees/publication_sprint_20260919/docs/实验结果/E216_jiang24_resource_bounded_confirmation_20260920")
    e216_status = e216 / "E216_FORMAL_EVALUATION_STATUS.json"
    rows.append({
        "dataset": "E216_Jiang24_resource",
        "context": "Jiang24 12 states",
        "task_range": "224",
        "upstream_model": "LatentAdditive+LinearAdditive_resource_bounded",
        "past_experiment_ids": "E216",
        "result_seen": "true",
        "method_design_influenced": "true",
        "role": "SEEN",
        "data_hash": sha256(e216_status),
        "prediction_version": sha256(e216 / "E216_PRETRUTH_TASK_SCORES.csv"),
        "truth_version": sha256(e216 / "E216_TASK_RESULTS.csv"),
        "metadata_only_until_freeze": "false",
        "notes": "Formal result was released on 2026-09-20; upstream competence and external confirmation were not supported. Never reuse as pristine confirmation.",
    })

    # Validation/development records are already seen and are registered as
    # separate rows from their still-sealed test endpoints.  This avoids the
    # misleading statement that an entire dataset is pristine merely because
    # its final test expression has not been opened.
    seen_validation = [
        (
            "E208_Jiang24_validation", "Jiang24 12 validation states", "216",
            "LatentAdditive+LinearAdditive", "E208;E233;E245",
            Path("/home/yyf/data/perturbench_e208/posttraining_20260921/validation_competence/E208_VALIDATION_COMPETENCE_STATUS.json"),
            "Validation competence failure and two registered repairs were seen before v4; no test perturbation truth was used.",
        ),
        (
            "E247_Kaden_CRISPRa_validation", "RPE1 validation TF perturbations", "376",
            "GEARS_two_seed_ensemble", "E247",
            ROOT / "docs/实验结果/E247_kaden_crispra_unseen_tf_20260924/E247_GEARS_ENSEMBLE_STATUS.json",
            "Validation result was seen and the ensemble failed the strongest-simple-baseline gate; test expression remains sealed.",
        ),
        (
            "E258_Feng2025_development", "4 train + 2 validation donors", "4292",
            "shrunk_mean+residual_MLP_panel", "E258",
            ROOT / "docs/实验结果/E258_feng2025_independent_confirmation_20260924/DEV_REPORT_20260925_历史何时有用.md",
            "Training/validation history analyses were seen and influenced evidence design; four-donor test perturbation truth remains sealed.",
        ),
    ]
    for dataset, context, task_range, upstream, exp, evidence, note in seen_validation:
        rows.append({
            "dataset": dataset,
            "context": context,
            "task_range": task_range,
            "upstream_model": upstream,
            "past_experiment_ids": exp,
            "result_seen": "true",
            "method_design_influenced": "true",
            "role": "SEEN",
            "data_hash": sha256(evidence),
            "prediction_version": None,
            "truth_version": None,
            "metadata_only_until_freeze": "false",
            "notes": note,
        })

    # These *test endpoints* remain sealed.  Only existence,
    # schema/provenance metadata and hashes of declared contract files are
    # recorded here; their development/validation records are the SEEN rows
    # above.
    sealed = [
        ("E208_Jiang24_test", "Jiang24 12 external states", "224", "E208", PUB / "E208_jiang24_external_confirmation_20260912", "Validation competence failed; test predictions never started and test perturbation truth remains sealed."),
        ("E247_Kaden_CRISPRa_test", "RPE1 unseen TF perturbations", "367", "E247", ROOT / "docs/实验结果/E247_kaden_crispra_unseen_tf_20260924", "Validation GEARS ensemble failed the strongest-simple-baseline competence gate; test expression remains sealed."),
        ("E258_Feng2025_test", "4 test donors; 7 cell lines", "2895", "E258", ROOT / "docs/实验结果/E258_feng2025_independent_confirmation_20260924", "Learning upstream failed the preregistered competence gate; four-donor test perturbation truth remains sealed."),
    ]
    for dataset, context, task_range, exp, path, note in sealed:
        contract_files = sorted(path.glob("*.md")) if path.exists() else []
        rows.append({
            "dataset": dataset,
            "context": context,
            "task_range": task_range,
            "upstream_model": "sealed_candidate_unknown",
            "past_experiment_ids": exp,
            "result_seen": "false",
            "method_design_influenced": "false",
            "role": "SEALED_CONFIRMATION",
            "data_hash": sha256(contract_files[0]) if contract_files else None,
            "prediction_version": None,
            "truth_version": None,
            "metadata_only_until_freeze": "true",
            "notes": note,
        })

    fields = [
        "dataset", "context", "task_range", "upstream_model", "past_experiment_ids",
        "result_seen", "method_design_influenced", "role", "data_hash",
        "prediction_version", "truth_version", "metadata_only_until_freeze", "notes",
    ]
    write_csv(STAGE / "DATA_ROLE_REGISTRY.csv", rows, fields)


def _coverage(series: pd.Series) -> tuple[int, int, float]:
    present = int(series.notna().sum())
    total = int(len(series))
    return present, total, (present / total if total else 0.0)


def build_history_audit() -> None:
    rows: list[dict] = []
    # TxPert source history is already target-background held out by E201's
    # released contract. No target truth is used to derive these support values.
    support = pd.read_csv(ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_SOURCE_CONTEXT_SUPPORT.csv")
    feature = pd.read_csv(ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_RISK_FEATURES.csv")
    feature = feature[feature.analysis_stratum.eq("primary_ge30")].copy()
    conflict_observed = int(feature.source_delta_dispersion_observed.notna().sum())
    total = int(len(feature))
    for source_type in ("H_internal", "H_external"):
        rows.append({
            "dataset_id": "E201_TxPert",
            "source_type": source_type,
            "study_id": "E201_source_contexts" if source_type == "H_internal" else "none_verified",
            "n_records": int(len(support)) if source_type == "H_internal" else 0,
            "n_tasks": total if source_type == "H_internal" else 0,
            "independent_contexts": int(support.source_context.nunique()) if source_type == "H_internal" else 0,
            "eligibility_status": "eligible_outer_train_only" if source_type == "H_internal" else "not_available",
            "same_dataset_as_target": "true" if source_type == "H_internal" else "false",
            "same_study_as_target": "true" if source_type == "H_internal" else "false",
            "independent_public_source": "false" if source_type == "H_internal" else "false",
            "quality_proxy_coverage": 0.0,
            "conflict_proxy_coverage": conflict_observed / total if source_type == "H_internal" else 0.0,
            "leakage_safe": "true" if source_type == "H_internal" else "not_applicable",
            "notes": "E201 contract permits three non-target source contexts; external history not verified in this asset.",
        })

    # E190 predicts Replogle K562 perturbations using an Adamson K562 source
    # study.  The Adamson effects were frozen before Replogle truth was built,
    # so they are genuine cross-study external history for this opened asset.
    rows.extend([
        {
            "dataset_id": "E190_Adamson_to_Replogle_K562",
            "source_type": "H_internal",
            "study_id": "none_verified",
            "n_records": 0,
            "n_tasks": 0,
            "independent_contexts": 0,
            "eligibility_status": "not_used",
            "same_dataset_as_target": "true",
            "same_study_as_target": "true",
            "independent_public_source": "false",
            "quality_proxy_coverage": 0.0,
            "conflict_proxy_coverage": 0.0,
            "leakage_safe": "not_applicable",
            "notes": "No Replogle target-study history is used in the E190 SafeConf adapter.",
        },
        {
            "dataset_id": "E190_Adamson_to_Replogle_K562",
            "source_type": "H_external",
            "study_id": "Adamson_source_K562",
            "n_records": 270,
            "n_tasks": 692,
            "independent_contexts": 1,
            "eligibility_status": "eligible_frozen_pretruth",
            "same_dataset_as_target": "false",
            "same_study_as_target": "false",
            "independent_public_source": "true",
            "quality_proxy_coverage": 0.0,
            "conflict_proxy_coverage": 1.0,
            "leakage_safe": "true",
            "notes": "Adamson train/validation fold effects and cell counts were frozen before Replogle target truth; conflict is fold dispersion, not multi-study quality.",
        },
    ])

    # Other published assets have prediction/truth but no contract-proven
    # public-history table; mark that absence instead of fabricating quality.
    for dataset in ["E112_Lara_exvivo", "E112_Santinha", "E84_E81", "E87", "E89", "GEARS_formal_54"]:
        rows.append({
            "dataset_id": dataset,
            "source_type": "H_internal",
            "study_id": "not_verified",
            "n_records": 0,
            "n_tasks": 0,
            "independent_contexts": 0,
            "eligibility_status": "not_available",
            "same_dataset_as_target": "unknown",
            "same_study_as_target": "unknown",
            "independent_public_source": "unknown",
            "quality_proxy_coverage": 0.0,
            "conflict_proxy_coverage": 0.0,
            "leakage_safe": "not_verified",
            "notes": "No legal history table registered; do not claim Public History quality/content.",
        })
    fields = [
        "dataset_id", "source_type", "study_id", "n_records", "n_tasks",
        "independent_contexts", "eligibility_status", "same_dataset_as_target",
        "same_study_as_target", "independent_public_source", "quality_proxy_coverage",
        "conflict_proxy_coverage", "leakage_safe", "notes",
    ]
    write_csv(STAGE / "HISTORY_SOURCE_AUDIT.csv", rows, fields)


def build_field_matrix() -> None:
    rows: list[dict] = []

    # Only DEV/SEEN assets are inspected here. SEALED rows are intentionally
    # absent until Final Candidate freeze.
    datasets = [
        ("E201_TxPert_GAT", "TxPert_STRING_GAT", "gene", 1808),
        ("E205_TxPert_Exphormer", "TxPert_Exphormer", "gene", 1808),
        ("E112_Lara_exvivo", "GEARS_context_mean_trainonly_graphs", "gene", 155),
        ("E112_Lara_exvivo", "scGPT_context_mean_finetuned", "gene", 155),
        ("E112_Santinha", "GEARS_context_mean_trainonly_graphs", "gene", 115),
        ("E112_Santinha", "scGPT_context_mean_finetuned", "gene", 115),
        ("E84_E81", "CPA_0.8.8_RDKIT_logdose", "chemical", 759),
        ("E87", "CPA_0.8.8_RDKIT_cross_dataset", "chemical", 553),
        ("E89", "CPA_0.8.8_RDKIT_sciPlex3", "chemical", 28),
        ("GEARS_formal_54", "GEARS", "gene", 54),
        ("E190_Adamson_to_Replogle_K562", "GEARS_3seed_centroid", "gene", 692),
    ]
    e201 = pd.read_csv(ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/tables/E201_PRETRUTH_RISK_FEATURES.csv")
    e201 = e201[e201.analysis_stratum.eq("primary_ge30")].copy()
    quality_present = int(e201.source_delta_dispersion_observed.notna().sum())
    quality_total = int(len(e201))
    feature_specs = [
        ("Support", "support", "task/source counts", "legal E201 source support", "true"),
        ("Relevance", "relevance", "prediction_source_cosine+model_source_gap", "current prediction versus legal source-history effect; prediction-time only", "true"),
        ("Quality", "quality", "not_available", "no replicate consistency or split-half quality field", "false"),
        ("Conflict", "conflict", "source_delta_dispersion_observed", "dispersion proxy; source-level conflict not separated", "true"),
        ("Content", "content", "source_mean_delta_row", "historical source effect content", "true"),
    ]
    for dataset, predictor, kind, n_tasks in datasets:
        for layer, field, derivation, formula, safe in feature_specs:
            if dataset.startswith("E201") or dataset.startswith("E205"):
                if field == "support":
                    raw = eligible = 1.0
                    missing = 0
                    studies = 1
                    contexts = 4
                elif field == "relevance":
                    raw = eligible = 1.0
                    missing = 0
                    studies = 1
                    contexts = 4
                elif field == "quality":
                    raw = eligible = 0.0
                    missing = n_tasks
                    studies = 0
                    contexts = 0
                elif field == "conflict":
                    raw = eligible = quality_present / quality_total
                    missing = quality_total - quality_present
                    studies = 1
                    contexts = 4
                elif field == "content":
                    raw = eligible = 1.0
                    missing = 0
                    studies = 1
                    contexts = 4
                else:
                    raw = eligible = 0.0
                    missing = n_tasks
                    studies = 0
                    contexts = 0
                leakage_safe = safe
            elif dataset.startswith("E190"):
                if field in {"support", "relevance", "conflict", "content"}:
                    raw = eligible = 1.0
                    missing = 0
                    studies = 1
                    contexts = 1
                    leakage_safe = "true"
                else:
                    raw = eligible = 0.0
                    missing = n_tasks
                    studies = 0
                    contexts = 0
                    leakage_safe = "false"
                if field == "support":
                    derivation = "Adamson pretruth source cell/guide counts"
                    formula = "count frozen Adamson source records by gene"
                elif field == "relevance":
                    derivation = "prediction/source effect geometry"
                    formula = "cosine and negative RMSE between GEARS prediction and frozen Adamson gene effect"
                elif field == "quality":
                    derivation = "not_available"
                    formula = "one source study/context cannot establish formal Quality"
                elif field == "conflict":
                    derivation = "Adamson fold dispersion proxy"
                    formula = "RMS dispersion of source pseudobulk fold effects"
                elif field == "content":
                    derivation = "Adamson frozen source gene effect"
                    formula = "magnitude of the cross-study source effect"
            else:
                raw = eligible = 0.0
                missing = n_tasks
                studies = 0
                contexts = 0
                leakage_safe = "false"
            rows.append({
                "dataset_id": dataset,
                "upstream_model_id": predictor,
                "field_name": layer,
                "raw_coverage": round(raw, 6),
                "eligible_coverage": round(eligible, 6),
                "missing_after_eligibility": int(missing),
                "n_independent_studies": int(studies),
                "n_independent_contexts": int(contexts),
                "derivation_type": derivation,
                "derivation_formula": formula,
                "leakage_safe": leakage_safe,
                "eligibility_rule_version": "HistoryEligibility-v1",
            })
    fields = [
        "dataset_id", "upstream_model_id", "field_name", "raw_coverage",
        "eligible_coverage", "missing_after_eligibility", "n_independent_studies",
        "n_independent_contexts", "derivation_type", "derivation_formula",
        "leakage_safe", "eligibility_rule_version",
    ]
    write_csv(STAGE / "FIELD_AVAILABILITY_MATRIX.csv", rows, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    build_role_registry()
    build_history_audit()
    build_field_matrix()
    print("wrote governance artifacts to", STAGE)


if __name__ == "__main__":
    main()

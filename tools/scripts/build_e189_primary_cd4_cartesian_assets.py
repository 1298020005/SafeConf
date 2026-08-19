#!/usr/bin/env python3
"""Build physically separated model and truth assets for E189.

Split identities are functions of metadata strings only.  The source E176 F2
bundle contains train/validation effects; F3/F4 contain held-out-donor truth.
This builder separates those vectors into support-specific model packages and
a fixed evaluation package.  The training runner never receives the latter.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E188 = ROOT / "docs/实验结果/E188_advisor_aligned_experiment_program_20260729"
OUT = ROOT / "docs/实验结果/E189_primary_cd4_formal_cartesian_20260729"
DATA = Path("/home/yyf/data/safeconf_e189_primary_cd4_cartesian")
E176 = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
E176_ASSETS = Path(
    "/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E176"
)
GO_SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
PANELS = ("H01", "H02", "H03", "H04")
SUPPORT_LEVELS = (1, 2, 3, 5)
N_SEEN = 120
N_UNSEEN = 40
N_GENES = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as store:
        return {key: np.asarray(store[key]) for key in store.files}


def task_map(tasks: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    result: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in tasks.iterrows():
        key = (
            str(row["donor_id"]),
            str(row["culture_condition"]),
            str(row["perturbed_gene_id"]),
        )
        if key in result:
            raise RuntimeError(f"duplicate task metadata: {key}")
        result[key] = row
    return result


def truth_for_test_donor(panel: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for stage, filename in (
        ("F3_calibration", "CALIBRATION_TARGET_EFFECTS.npz"),
        ("F4_evaluation", "EVALUATION_TARGET_EFFECTS.npz"),
    ):
        result.update(load_npz(E176_ASSETS / panel / stage / filename))
    if len(result) != 600:
        raise RuntimeError(f"{panel}: expected 600 held-out-donor truths, found {len(result)}")
    return result


def build_panel(
    panel: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    f2 = E176_ASSETS / panel / "F2_pretruth"
    tasks = pd.read_csv(f2 / "PRETRUTH_TASKS.csv", keep_default_na=False)
    metadata = task_map(tasks)
    effects = load_npz(f2 / "SEEN_TARGET_EFFECTS.npz")
    heldout_truth = truth_for_test_donor(panel)
    controls = load_npz(f2 / "CONTROL_PROFILES.npz")
    gene_panel = pd.read_csv(f2 / "GENE_PANEL.csv", keep_default_na=False)
    tokens = set(gene_panel.scgpt_token.astype(str))
    go_panel = pd.read_csv(GO_SOURCE)
    go_panel = go_panel.loc[
        go_panel.source.astype(str).isin(tokens)
        & go_panel.target.astype(str).isin(tokens)
    ].copy()
    go_panel = (
        go_panel.sort_values(["target", "importance"], ascending=[True, False])
        .groupby("target", as_index=False, group_keys=False)
        .head(21)
    )
    if go_panel.empty:
        raise RuntimeError(f"{panel}: frozen GO subgraph is empty")

    eligible = sorted(
        tasks.loc[
            tasks["target_stratum"].eq("DONOR_UNSEEN_ONLY"),
            "perturbed_gene_id",
        ].astype(str).unique()
    )
    if len(eligible) != 160:
        raise RuntimeError(f"{panel}: expected 160 eligible targets, found {len(eligible)}")
    ordered = sorted(eligible, key=lambda gene: stable_key("E189", panel, "column", gene))
    unseen = set(ordered[:N_UNSEEN])
    seen = set(ordered[N_UNSEEN:])
    if len(seen) != N_SEEN:
        raise RuntimeError(f"{panel}: seen/unseen target split failed")

    train_contexts = sorted(
        {
            (str(row.donor_id), str(row.culture_condition))
            for row in tasks[tasks["donor_role"].eq("train")].itertuples(index=False)
        }
    )
    validation_contexts = sorted(
        {
            (str(row.donor_id), str(row.culture_condition))
            for row in tasks[tasks["donor_role"].eq("validation")].itertuples(index=False)
        }
    )
    test_contexts = sorted(
        {
            (str(row.donor_id), str(row.culture_condition))
            for row in tasks[tasks["donor_role"].eq("test")].itertuples(index=False)
        }
    )
    if (len(train_contexts), len(validation_contexts), len(test_contexts)) != (6, 3, 3):
        raise RuntimeError(f"{panel}: unexpected context counts")

    selected_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    train_order: dict[str, list[tuple[str, str]]] = {}
    for gene in sorted(seen):
        order = sorted(
            train_contexts,
            key=lambda context: stable_key(
                "E189", panel, "pair_order", gene, context[0], context[1]
            ),
        )
        reserved, candidates = order[0], order[1:]
        train_order[gene] = candidates
        row = metadata[(reserved[0], reserved[1], gene)]
        query_rows.append({
            **row.to_dict(),
            "e189_setting": "random_missing_pair",
            "e189_seen_context": True,
            "e189_seen_perturbation": True,
        })
        selected_rows.append({
            "panel_id": panel,
            "perturbed_gene_id": gene,
            "e189_perturbation_role": "SEEN_PERTURBATION",
            "reserved_random_pair_donor": reserved[0],
            "reserved_random_pair_state": reserved[1],
            "candidate_train_context_order": "|".join(f"{a}::{b}" for a, b in candidates),
            "selection_sha256": stable_key("E189", panel, "target", gene),
        })

    for gene in sorted(seen):
        for donor, state in test_contexts:
            row = metadata[(donor, state, gene)]
            query_rows.append({
                **row.to_dict(),
                "e189_setting": "unseen_context_row",
                "e189_seen_context": False,
                "e189_seen_perturbation": True,
            })
    for gene in sorted(unseen):
        selected_rows.append({
            "panel_id": panel,
            "perturbed_gene_id": gene,
            "e189_perturbation_role": "UNSEEN_PERTURBATION",
            "reserved_random_pair_donor": "",
            "reserved_random_pair_state": "",
            "candidate_train_context_order": "",
            "selection_sha256": stable_key("E189", panel, "target", gene),
        })
        for donor, state in train_contexts:
            row = metadata[(donor, state, gene)]
            query_rows.append({
                **row.to_dict(),
                "e189_setting": "unseen_perturbation_column",
                "e189_seen_context": True,
                "e189_seen_perturbation": False,
            })
        for donor, state in test_contexts:
            row = metadata[(donor, state, gene)]
            query_rows.append({
                **row.to_dict(),
                "e189_setting": "double_unseen",
                "e189_seen_context": False,
                "e189_seen_perturbation": False,
            })

    query = pd.DataFrame(query_rows)
    expected = {
        "random_missing_pair": 120,
        "unseen_context_row": 360,
        "unseen_perturbation_column": 240,
        "double_unseen": 120,
    }
    if query["e189_setting"].value_counts().to_dict() != expected:
        raise RuntimeError(f"{panel}: query setting counts changed")
    if query["task_id"].duplicated().any():
        raise RuntimeError(f"{panel}: duplicate query tasks")

    fixed_truth: dict[str, np.ndarray] = {}
    for row in query.itertuples(index=False):
        task_id = str(row.task_id)
        if row.e189_setting in {"random_missing_pair", "unseen_perturbation_column"}:
            fixed_truth[task_id] = effects[task_id]
        else:
            fixed_truth[task_id] = heldout_truth[task_id]
    if len(fixed_truth) != 840 or any(value.shape != (N_GENES,) for value in fixed_truth.values()):
        raise RuntimeError(f"{panel}: evaluation truth construction failed")

    truth_root = DATA / "evaluation_truth" / panel
    if truth_root.exists():
        shutil.rmtree(truth_root)
    truth_root.mkdir(parents=True, exist_ok=True)
    query.to_csv(truth_root / "QUERY_TASKS.csv", index=False)
    save_npz(truth_root / "EVALUATION_TRUTH.npz", fixed_truth)

    locks: list[dict[str, object]] = []
    truth_locks: list[dict[str, object]] = []
    for path in sorted(truth_root.iterdir()):
        truth_locks.append({
            "panel_id": panel,
            "path": path.relative_to(DATA).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    support_train_ids: dict[int, set[str]] = {}
    for support in SUPPORT_LEVELS:
        package = DATA / "model_assets" / panel / f"support_{support}"
        if package.exists():
            shutil.rmtree(package)
        package.mkdir(parents=True, exist_ok=True)
        train_effects: dict[str, np.ndarray] = {}
        validation_effects: dict[str, np.ndarray] = {}
        train_rows: list[dict[str, object]] = []
        validation_rows: list[dict[str, object]] = []
        for gene in sorted(seen):
            for donor, state in train_order[gene][:support]:
                row = metadata[(donor, state, gene)]
                task_id = str(row["task_id"])
                train_effects[task_id] = effects[task_id]
                train_rows.append(row.to_dict())
            for donor, state in validation_contexts:
                row = metadata[(donor, state, gene)]
                task_id = str(row["task_id"])
                validation_effects[task_id] = effects[task_id]
                validation_rows.append(row.to_dict())

        train_ids = set(train_effects)
        validation_ids = set(validation_effects)
        query_ids = set(query["task_id"].astype(str))
        if train_ids & validation_ids or train_ids & query_ids or validation_ids & query_ids:
            raise RuntimeError(f"{panel}/support_{support}: exact task leakage")
        if len(train_ids) != N_SEEN * support:
            raise RuntimeError(f"{panel}/support_{support}: training task count failed")
        if len(validation_ids) != N_SEEN * len(validation_contexts):
            raise RuntimeError(f"{panel}/support_{support}: validation task count failed")
        supervised_genes = {
            str(metadata_key[2])
            for metadata_key, row in metadata.items()
            if str(row["task_id"]) in train_ids | validation_ids
        }
        if supervised_genes != seen or supervised_genes & unseen:
            raise RuntimeError(f"{panel}/support_{support}: unseen target entered supervision")
        support_train_ids[support] = train_ids
        if support > 1:
            previous = support_train_ids[SUPPORT_LEVELS[SUPPORT_LEVELS.index(support) - 1]]
            if not previous < train_ids:
                raise RuntimeError(f"{panel}: support packages are not strictly nested")

        pd.DataFrame(train_rows).to_csv(package / "TRAIN_TASKS.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(package / "VALIDATION_TASKS.csv", index=False)
        query.to_csv(package / "QUERY_TASKS.csv", index=False)
        save_npz(package / "TRAIN_EFFECTS.npz", train_effects)
        save_npz(package / "VALIDATION_EFFECTS.npz", validation_effects)
        shutil.copy2(f2 / "CONTROL_PROFILES.npz", package / "CONTROL_PROFILES.npz")
        shutil.copy2(f2 / "GENE_PANEL.csv", package / "GENE_PANEL.csv")
        shutil.copy2(
            f2 / "TRAIN_NTC_COEXPRESSION_EDGES.csv",
            package / "TRAIN_NTC_COEXPRESSION_EDGES.csv",
        )
        go_panel.to_csv(package / "GO_EDGES_PANEL.csv", index=False)
        manifest = {
            "experiment": "E189",
            "panel_id": panel,
            "support_contexts_per_seen_perturbation": support,
            "n_train_tasks": len(train_effects),
            "n_validation_tasks": len(validation_effects),
            "n_query_tasks": len(query),
            "n_seen_perturbations": N_SEEN,
            "n_unseen_perturbations": N_UNSEEN,
            "contains_evaluation_truth": False,
            "source_e176_bundle": str(f2),
            "split_selection_uses_effect_values": False,
        }
        (package / "ASSET_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in sorted(package.iterdir()):
            if path.is_file():
                locks.append({
                    "panel_id": panel,
                    "support": support,
                    "path": path.relative_to(DATA).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        observed = {path.name for path in package.iterdir()}
        expected_files = {
            "TRAIN_TASKS.csv",
            "VALIDATION_TASKS.csv",
            "QUERY_TASKS.csv",
            "TRAIN_EFFECTS.npz",
            "VALIDATION_EFFECTS.npz",
            "CONTROL_PROFILES.npz",
            "GENE_PANEL.csv",
            "TRAIN_NTC_COEXPRESSION_EDGES.csv",
            "GO_EDGES_PANEL.csv",
            "ASSET_MANIFEST.json",
        }
        if observed != expected_files:
            raise RuntimeError(f"{panel}/support_{support}: model asset allowlist failed")

    manifest_dir = OUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(manifest_dir / f"E189_{panel}_TARGET_SPLIT.csv", index=False)
    query.to_csv(manifest_dir / f"E189_{panel}_QUERY_TASKS.csv", index=False)
    return selected, locks, truth_locks


def main() -> None:
    if not (E188 / "PREREG_EXPERIMENT_PROGRAM.md").exists():
        raise RuntimeError("E188 preregistration is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    all_selected: list[pd.DataFrame] = []
    all_locks: list[dict[str, object]] = []
    all_truth_locks: list[dict[str, object]] = []
    for panel in PANELS:
        selected, locks, truth_locks = build_panel(panel)
        all_selected.append(selected)
        all_locks.extend(locks)
        all_truth_locks.extend(truth_locks)
        print(f"[E189 assets] {panel}: PASS", flush=True)
    pd.concat(all_selected, ignore_index=True).to_csv(
        OUT / "manifests/E189_ALL_TARGET_SPLITS.csv", index=False
    )
    pd.DataFrame(all_locks).to_csv(OUT / "MODEL_ASSET_LOCKS.csv", index=False)
    pd.DataFrame(all_truth_locks).to_csv(
        OUT / "EVALUATION_ASSET_LOCKS.csv", index=False
    )
    checkpoint_locks = []
    for name in ("args.json", "vocab.json", "best_model.pt"):
        path = SCGPT_CHECKPOINT / name
        checkpoint_locks.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(checkpoint_locks).to_csv(
        OUT / "SCGPT_CHECKPOINT_LOCKS.csv", index=False
    )
    status = {
        "experiment": "E189_asset_build",
        "status": "PASS",
        "panels": list(PANELS),
        "support_levels": list(SUPPORT_LEVELS),
        "tasks_per_panel_per_support": 840,
        "total_planned_task_instances": 13440,
        "model_assets_contain_evaluation_truth": False,
        "split_selection_uses_effect_values": False,
    }
    (OUT / "ASSET_BUILD_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "README_先看这个.md").write_text(
        "# E189 先看这个\n\n"
        "E189 使用同一 Primary CD4 数据和同一 scGPT–GEARS 合同，"
        "补小矩阵、随机 pair、整行、整列和双未见。"
        "先看 `ASSET_BUILD_STATUS.json` 和 `manifests/`；"
        "模型运行完成后再看 `reports/E189_REPORT.md`。\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

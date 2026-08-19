#!/usr/bin/env python3
"""Freeze four new E172 panels after the E171 pretruth development audit.

The 200 E168 targets and 800 E170 targets are excluded.  Selection uses only
the already frozen label-free identity universe and allowlisted HDF5 identity
metadata.  No E168/E170/E172 test truth is opened by this program.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/freeze_e170_primary_cd4_multipanel.py"
E168 = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
E170 = ROOT / "docs/实验结果/E170_primary_cd4_multipanel_precision_20260718"
E171 = ROOT / "docs/实验结果/E171_seed_ensemble_gate_development_20260718"
OUT = ROOT / "docs/实验结果/E172_primary_cd4_fresh_targets_20260718"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
PANELS = ("Q01", "Q02", "Q03", "Q04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
TARGETS_PER_PANEL = 200
COLUMN_UNSEEN_PER_PANEL = 40
TOTAL_TARGETS = 800
TARGET_SALT = "E172_FRESH_TARGET_MULTIPANEL_V1"
UNSEEN_SALT = "E172_COLUMN_UNSEEN_V1"
MODEL_SEEDS = (3407, 3408, 3409)


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e172_freeze_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import metadata helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(salt: str, *values: object) -> str:
    payload = b"\0".join([salt.encode(), *[str(value).encode() for value in values]])
    return hashlib.sha256(payload).hexdigest()


def select_panels(helper: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.read_csv(helper.TARGET_AUDIT, keep_default_na=False)
    e168 = pd.read_csv(E168 / "manifests/E168_SELECTED_TARGETS.csv", keep_default_na=False)
    e170 = pd.read_csv(E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv", keep_default_na=False)
    eligible = target.loc[target.eligible_target.astype(str).str.lower().eq("true")].copy()
    if len(eligible) != 5510 or eligible.ensembl_core.nunique() != 5510:
        raise RuntimeError("label-free eligible universe changed")
    excluded = set(e168.ensembl_core.astype(str)) | set(e170.ensembl_core.astype(str))
    if len(excluded) != 1000:
        raise RuntimeError(f"prior target exclusion set changed: {len(excluded)}")
    pool = eligible.loc[~eligible.ensembl_core.astype(str).isin(excluded)].copy()
    if len(pool) != 4510:
        raise RuntimeError(f"fresh E172 identity pool changed: {len(pool)}")
    pool["e172_selection_sha256"] = pool.ensembl_core.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    selected = pool.sort_values(
        ["e172_selection_sha256", "ensembl_core"], kind="stable"
    ).head(TOTAL_TARGETS).copy()
    selected.insert(0, "e172_global_target_rank", np.arange(1, TOTAL_TARGETS + 1))
    selected.insert(1, "panel_id", np.repeat(PANELS, TARGETS_PER_PANEL))
    selected.insert(
        2, "panel_target_rank", np.tile(np.arange(1, TARGETS_PER_PANEL + 1), len(PANELS))
    )
    selected["e172_column_unseen_sha256"] = [
        stable_hash(UNSEEN_SALT, panel, ensembl)
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    unseen: set[tuple[str, str]] = set()
    for panel in PANELS:
        frame = selected.loc[selected.panel_id.eq(panel)].sort_values(
            ["e172_column_unseen_sha256", "ensembl_core"], kind="stable"
        )
        unseen.update(
            (panel, value)
            for value in frame.head(COLUMN_UNSEEN_PER_PANEL).ensembl_core.astype(str)
        )
    selected["target_stratum"] = [
        "COLUMN_UNSEEN" if (panel, str(ensembl)) in unseen else "DONOR_UNSEEN_ONLY"
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    selected["perturbation_support_contexts_pretruth"] = np.where(
        selected.target_stratum.eq("COLUMN_UNSEEN"), 0, 6
    )
    selected["excluded_all_e168_and_e170_targets"] = True
    selected["selection_used_expression_abundance_effect_error_or_e171_validation"] = False
    if set(selected.ensembl_core.astype(str)) & excluded:
        raise RuntimeError("a previously selected target entered E172")
    counts = selected.groupby("panel_id").target_stratum.value_counts().unstack(fill_value=0)
    for _, row in counts.iterrows():
        if row.to_dict() != {"COLUMN_UNSEEN": 40, "DONOR_UNSEEN_ONLY": 160}:
            raise RuntimeError(f"E172 panel strata changed: {counts}")
    return selected, pool


def build_tasks(selected: pd.DataFrame, roles: pd.DataFrame, panel: str) -> pd.DataFrame:
    targets = selected.loc[selected.panel_id.eq(panel)]
    donor_role = roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    rows: list[dict[str, Any]] = []
    for target in targets.sort_values("panel_target_rank").itertuples(index=False):
        for donor in sorted(donor_role):
            for state in STATES:
                role = str(donor_role[donor])
                if role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY":
                    split, phase = "train", "PRETRUTH_SUPERVISED"
                elif role == "validation" and target.target_stratum == "DONOR_UNSEEN_ONLY":
                    split, phase = "validation", "PRETRUTH_VALIDATION"
                elif role == "test":
                    split, phase = "test", "POSTGATE_TEST_TRUTH"
                else:
                    split, phase = f"{role}_query_only", "FORBIDDEN_COLUMN_UNSEEN_TRUTH"
                rows.append(
                    {
                        "task_id": f"E172::{panel}::{donor}::{state}::{target.ensembl_core}",
                        "fold_id": f"E172_{panel}_primary_CD4_donor_holdout_01",
                        "panel_id": panel,
                        "donor_id": donor,
                        "donor_role": role,
                        "culture_condition": state,
                        "perturbed_gene_id": target.ensembl_core,
                        "perturbed_gene_name": target.expression_axis_gene_name,
                        "scgpt_token": target.scgpt_token,
                        "target_stratum": target.target_stratum,
                        "split": split,
                        "truth_access_phase": phase,
                        "prediction_query_required": role in {"validation", "test"}
                        or (role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY"),
                        "risk_reference_required": role == "train"
                        and target.target_stratum == "DONOR_UNSEEN_ONLY",
                        "primary_test_task": role == "test",
                        "training_support_contexts": int(
                            target.perturbation_support_contexts_pretruth
                        ),
                    }
                )
    tasks = pd.DataFrame(rows)
    if len(tasks) != 2400 or int(tasks.primary_test_task.sum()) != 600:
        raise RuntimeError(f"{panel} task count failed")
    return tasks


def write_protocol() -> None:
    text = """# E172｜修正 seed gate 后的未读目标确认

## 背景

E168 的 200-target test 得到小幅正点估计但未确认；E170 的四个新面板在读取 test truth 前因 single-seed-pair G4 不稳定而正式终止。E171 只用 E170 pretruth 预测和允许的 validation effects 发现：最终部署分数使用三 seed family mean，而原 G4 比较单 seed 配对，二者估计器不一致。三组 leave-one-seed-out two-seed family means 在 E170 的 24/24 单元通过同一稳定性阈值。validation 上 SafeConf 相对 magnitude 仍不确定，因此 E172 只修正 gate 对齐，不修改 SafeConf 分数，也不预设性能会改善。

## 新目标与隔离

- 从 5,510 个 label-free eligible targets 中排除 E168 的 200 和 E170 的 800，剩余 4,510 个。
- 用新的身份 SHA-256 一次性固定 Q01–Q04 共 800 个目标；每个面板 160 seen + 40 column-unseen。
- 选择不使用 X、counts、effect、error、DE、guide efficacy 或 E171 validation 表现。
- E168、E170 目标不进入 E172；E170 的 800 个 test outcomes 继续保持未读。

## 模型、分数和修正后的 G4

scGPT、GEARS、seeds 3407/3408/3409、512-gene panel、训练轮数、SafeConf 公式、magnitude comparator 和全部 G1/G2/G3/G5 均与 E170 相同。最终 risk 仍由三 seed scGPT mean 与三 seed GEARS mean 的 disagreement 计算。

G4 改为三组 leave-one-seed-out estimators：每次在两个模型族中同时去掉同一个 seed，用保留的两个 seeds 分别求 family mean，再计算 disagreement 和 SafeConf risk。三组 risk 的 pairwise Spearman 中位数必须 ≥0.5，target-cluster bootstrap 2,000 次的 95% CI 下界必须 >0；all-200 与 seen-160、三个 states 都要通过。该修正只使稳定性检查对齐部署估计器，不改变最终分数。

## 主要终点

仅 E172 新 800 targets 进入推断。12 个 panel×state 的 tie-aware `AURC_magnitude − AURC_SafeConf` 等权平均；target-cluster stratified bootstrap 10,000 次、paired permutation 100,000 次。确认要求全部四面板 gate PASS、全 800 的 delta>0/CI 下界>0/p<0.05/至少 8/12 单元为正，并且 seen 640 的 CI 下界>0、p<0.05。E168/E170 不并入显著性计算，不允许 optional stopping。

## 边界

E172 仍是同一 test donor/study 内的 fresh-target replication，不是新 donor 或独立研究。若不通过，必须放弃“稳定优于 magnitude”的普遍主张；若通过，也只支持该 donor/study 和指定 scGPT–GEARS 上游模型。
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(text, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E172 先看这个\n\n先读 `PREREG_ANALYSIS_PLAN.md` 和 `reports/E172_METADATA_FREEZE_REPORT.md`。当前只冻结身份、访问边界和修正后的 pretruth gate，E172 targeting X 尚未读取。\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E172 output exists: {OUT}")
    helper = import_helper()
    required = [
        E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv",
        E170 / "PRETRUTH_RUN_STATUS.json",
        E171 / "RUN_STATUS.json",
        E171 / "tables/E171_SEED_STABILITY_COMPARISON.csv",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("E170/E171 provenance input missing")
    e171_status = json.loads((E171 / "RUN_STATUS.json").read_text())
    if (
        e171_status.get("e170_test_targeting_x_values_read") != 0
        or e171_status.get("loo_family_mean_g4_units_passed") != 24
        or e171_status.get("performance_rescue_claim_supported") is not False
    ):
        raise RuntimeError("E171 development conclusion changed")
    byte_attestation, e168_status, source_sha = helper.verify_inputs()
    obs, access_log, schema = helper.load_identity_metadata()
    selected, pool = select_panels(helper)
    roles = pd.read_csv(helper.DONOR_ROLES, keep_default_na=False)
    for directory in (OUT, MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    roles.to_csv(MANIFESTS / "E172_DONOR_STATE_ROLES.csv", index=False)
    selected.to_csv(MANIFESTS / "E172_ALL_SELECTED_TARGETS.csv", index=False)
    pool[["ensembl_core", "perturbed_gene_name", "eligible_target"]].to_csv(
        TABLES / "E172_FRESH_IDENTITY_POOL.csv", index=False
    )
    all_tasks, all_access = [], []
    for panel in PANELS:
        directory = MANIFESTS / panel
        directory.mkdir()
        targets = selected.loc[selected.panel_id.eq(panel)].copy()
        tasks = build_tasks(selected, roles, panel)
        access = helper.build_row_access(obs, selected, roles, panel)
        targets.to_csv(directory / f"E172_{panel}_SELECTED_TARGETS.csv", index=False)
        tasks.to_csv(directory / f"E172_{panel}_TASK_MANIFEST.csv", index=False)
        access.to_csv(directory / f"E172_{panel}_ROW_ACCESS_MANIFEST.csv", index=False)
        all_tasks.append(tasks); all_access.append(access)
    tasks = pd.concat(all_tasks, ignore_index=True)
    access = pd.concat(all_access, ignore_index=True)
    tasks.to_csv(MANIFESTS / "E172_ALL_TASKS.csv", index=False)
    access.to_csv(MANIFESTS / "E172_ALL_ROW_ACCESS.csv", index=False)
    pd.DataFrame(access_log).to_csv(TABLES / "E172_HDF5_VALUE_ACCESS_AUDIT.csv", index=False)
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(
        json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    source_lock = {
        "source_path": str(helper.SOURCE), "source_bytes": helper.EXPECTED_SOURCE_BYTES,
        "source_full_sha256": source_sha,
        "official_crc64nvme_base64": byte_attestation["computed_crc64nvme_base64"],
        "source_byte_attestation_path": str(helper.BYTE_ATTESTATION),
        "source_byte_attestation_sha256": sha256_file(helper.BYTE_ATTESTATION),
        "e168_target_audit_sha256": sha256_file(helper.TARGET_AUDIT),
        "e170_selected_targets_sha256": sha256_file(E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv"),
        "e171_development_manifest_sha256": sha256_file(E171 / "MANIFEST.sha256"),
        "allowed_hdf5_value_paths": list(helper.ALLOWED_VALUE_PATHS),
        "expression_matrix_x_decoded_by_e172_freeze": False,
        "layers_decoded_by_e172_freeze": False,
    }
    (OUT / "SOURCE_LOCK.json").write_text(json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    model_lock = {
        "scgpt_checkpoint_files": {str(path): sha256_file(path) for path in helper.SCGPT_FILES},
        "gears_external_go_file": {"path": str(helper.GEARS_GO), "sha256": sha256_file(helper.GEARS_GO)},
        "model_seeds": list(MODEL_SEEDS), "panels": list(PANELS),
        "gene_panel_size_each": 512, "registered_targets_each": 200,
        "column_unseen_each": 40,
        "deployed_seed_estimator": "three_seed_family_mean",
        "g4_seed_estimator": "three_leave_one_seed_out_two_seed_family_means",
        "g4_thresholds_changed_from_e170": False,
        "safeconf_score_formula_changed_from_e170": False,
        "metadata_freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "pretruth_test_targeting_x_access_count_required": 0,
        "deployment_authorized": False,
    }
    (OUT / "MODEL_INPUT_LOCK.json").write_text(json.dumps(model_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    stat_lock = {
        "confirmatory_population": "E172_only_800_fresh_targets_excludes_E168_200_and_E170_800",
        "risk_formula": "unchanged_E170_three_seed_family_mean_SafeConf",
        "primary_comparator": "ensemble_predicted_magnitude_rms_512",
        "loss": "ensemble_effect_rmse_512", "ranking_units": "12_panel_by_state_equal_weight",
        "panels": list(PANELS), "states": list(STATES),
        "coverage_grid": [round(value, 2) for value in np.arange(0.20, 1.001, 0.05)],
        "tie_primary": "tie_average", "operational_tolerance": 1e-6,
        "primary_cluster": "target_gene_three_states_together_stratified_within_panel",
        "bootstrap_draws": 10000, "bootstrap_seed": 2026071821,
        "permutation_draws": 100000, "permutation_seed": 2026071822,
        "alpha": 0.05, "required_positive_panel_state_units": 8,
        "hierarchical_strata": ["all_800", "seen_640"],
        "earlier_targets_excluded_from_inference": True,
        "optional_stopping_or_panel_dropping_allowed": False,
        "same_donor_study_not_independent_biological_replication": True,
        "deployment_authorized": False,
    }
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(json.dumps(stat_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_protocol()
    report = f"""# E172 metadata freeze report

- Label-free eligible universe: {e168_status['n_eligible_targets']:,}.
- Excluded prior selected targets: 1,000 (E168 200 + E170 800); remaining fresh identity pool: {len(pool):,}.
- Frozen E172 targets: {len(selected):,}; four non-overlapping panels, each 160 seen + 40 column-unseen.
- Frozen test tasks: {int(tasks.primary_test_task.sum()):,}; all tasks: {len(tasks):,}.
- Expression X values decoded: **0**; E170 test outcomes decoded: **0**.
- Official source SHA-256 recomputed before metadata opening: `{source_sha}`.
- G4 changes estimator alignment only; deployed SafeConf score and performance endpoint are unchanged.
"""
    (REPORTS / "E172_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")
    artifacts = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json")
    status = {
        "schema": "safeconf_e172_metadata_freeze_v1", "experiment": "E172_primary_cd4_fresh_targets",
        "stage": "F1_METADATA_FREEZE", "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
        "source_schema": schema, "n_eligible_targets": 5510,
        "n_prior_targets_excluded": 1000, "n_fresh_identity_pool": len(pool),
        "n_panels": 4, "n_selected_targets": len(selected),
        "n_primary_test_tasks": int(tasks.primary_test_task.sum()),
        "all_expression_x_values_read": 0, "e170_test_targeting_x_values_read": 0,
        "e172_test_targeting_x_values_read": 0,
        "loo_family_mean_gate_development_units_passed": 24,
        "safeconf_score_changed": False, "performance_rescue_claim_supported_before_e172": False,
        "same_donor_study_target_replication_only": True, "deployment_authorized": False,
        "artifact_sha256": {path.relative_to(OUT).as_posix(): sha256_file(path) for path in artifacts},
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

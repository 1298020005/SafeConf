#!/usr/bin/env python3
"""Freeze E176 identities and four balanced donor rotations before any target truth is read."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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
E172 = ROOT / "docs/实验结果/E172_primary_cd4_fresh_targets_20260718"
E174 = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
E175 = ROOT / "docs/实验结果/E175_e174_seed_extension_development_20260719"
OUT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

PANELS = ("H01", "H02", "H03", "H04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
DONORS = ("CE0006864", "CE0008162", "CE0008678", "CE0010866")
TARGETS_PER_PANEL = 200
COLUMN_UNSEEN_PER_PANEL = 40
CALIBRATION_PER_PANEL = 40
TOTAL_TARGETS = 800
MODEL_SEEDS = (3407, 3408, 3409, 3410, 3411)
TARGET_SALT = "E176_FOUR_DONOR_FRESH_TARGET_V1"
UNSEEN_SALT = "E176_COLUMN_UNSEEN_V1"
CALIBRATION_SALT = "E176_DONOR_SPECIFIC_CALIBRATION_20PCT_V1"

# Each donor is test once, validation once, and train twice.
ROTATIONS = {
    "H01": {"test": "CE0006864", "validation": "CE0008162",
            "train": ("CE0008678", "CE0010866")},
    "H02": {"test": "CE0008162", "validation": "CE0008678",
            "train": ("CE0006864", "CE0010866")},
    "H03": {"test": "CE0008678", "validation": "CE0010866",
            "train": ("CE0006864", "CE0008162")},
    "H04": {"test": "CE0010866", "validation": "CE0006864",
            "train": ("CE0008162", "CE0008678")},
}


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e176_identity_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import identity helper: {HELPER_PATH}")
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


def require_prior_boundary() -> list[Path]:
    target_paths = [
        E168 / "manifests/E168_SELECTED_TARGETS.csv",
        E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv",
        E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv",
        E174 / "manifests/E174_ALL_SELECTED_TARGETS.csv",
    ]
    controls = [
        E174 / "PRETRUTH_ABORT_STATUS.json",
        E175 / "aggregate/RUN_STATUS.json",
        E175 / "aggregate/tables/FIVE_SEED_LOO_G4.csv",
    ]
    missing = [str(path) for path in [*target_paths, *controls] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prior evidence: {missing}")
    abort = json.loads(controls[0].read_text())
    seed_gate = json.loads(controls[1].read_text())
    if abort.get("decision") != "NO_CALIBRATION_OR_EVALUATION_TRUTH_ACCESS":
        raise RuntimeError("E174 sealed-truth boundary changed")
    required = {
        "status": "COMPLETE",
        "decision": "FIVE_SEED_GATE_READY_FOR_NEW_TARGET_PROTOCOL",
        "five_seed_g4_units": 24,
        "five_seed_g4_units_passed": 24,
        "e174_heldout_donor_targeting_x_values_read": 0,
        "new_target_truth_must_be_used_for_confirmation": True,
    }
    changed = {key: (value, seed_gate.get(key)) for key, value in required.items()
               if seed_gate.get(key) != value}
    if changed:
        raise RuntimeError(f"E175 development gate changed: {changed}")
    return target_paths


def select_targets(helper: Any, prior_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    audit = pd.read_csv(helper.TARGET_AUDIT, keep_default_na=False)
    eligible = audit.loc[audit.eligible_target.astype(str).str.lower().eq("true")].copy()
    if len(eligible) != 5510 or eligible.ensembl_core.nunique() != 5510:
        raise RuntimeError("label-free eligible target universe changed")
    if not eligible.n_identity_complete_guides.astype(int).eq(2).all():
        raise RuntimeError("E176 requires exactly two identity-complete guides per target")
    expected_sizes = [200, 800, 800, 800]
    prior_sets = [set(pd.read_csv(path).ensembl_core.astype(str)) for path in prior_paths]
    if [len(values) for values in prior_sets] != expected_sizes:
        raise RuntimeError("prior target manifest sizes changed")
    excluded = set().union(*prior_sets)
    if len(excluded) != 2600:
        raise RuntimeError(f"prior target sets overlap or changed: union={len(excluded)}")
    pool = eligible.loc[~eligible.ensembl_core.astype(str).isin(excluded)].copy()
    if len(pool) != 2910:
        raise RuntimeError(f"fresh identity pool changed: {len(pool)}")

    pool["e176_selection_sha256"] = pool.ensembl_core.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    selected = pool.sort_values(
        ["e176_selection_sha256", "ensembl_core"], kind="stable"
    ).head(TOTAL_TARGETS).copy()
    selected.insert(0, "e176_global_target_rank", np.arange(1, TOTAL_TARGETS + 1))
    selected.insert(1, "panel_id", np.repeat(PANELS, TARGETS_PER_PANEL))
    selected.insert(2, "panel_target_rank",
                    np.tile(np.arange(1, TARGETS_PER_PANEL + 1), len(PANELS)))

    selected["e176_column_unseen_sha256"] = [
        stable_hash(UNSEEN_SALT, panel, target)
        for panel, target in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    unseen: set[tuple[str, str]] = set()
    for panel in PANELS:
        block = selected.loc[selected.panel_id.eq(panel)].sort_values(
            ["e176_column_unseen_sha256", "ensembl_core"], kind="stable"
        )
        unseen.update((panel, target) for target in
                      block.head(COLUMN_UNSEEN_PER_PANEL).ensembl_core.astype(str))
    selected["target_stratum"] = [
        "COLUMN_UNSEEN" if (panel, str(target)) in unseen else "DONOR_UNSEEN_ONLY"
        for panel, target in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]

    selected["e176_calibration_sha256"] = [
        stable_hash(CALIBRATION_SALT, panel, stratum, target)
        for panel, stratum, target in zip(
            selected.panel_id, selected.target_stratum, selected.ensembl_core, strict=True
        )
    ]
    calibration: set[tuple[str, str]] = set()
    for panel in PANELS:
        for stratum, count in (("DONOR_UNSEEN_ONLY", 32), ("COLUMN_UNSEEN", 8)):
            block = selected.loc[
                selected.panel_id.eq(panel) & selected.target_stratum.eq(stratum)
            ].sort_values(["e176_calibration_sha256", "ensembl_core"], kind="stable")
            calibration.update((panel, target) for target in
                               block.head(count).ensembl_core.astype(str))
    selected["heldout_donor_partition"] = [
        "CALIBRATION_20PCT" if (panel, str(target)) in calibration
        else "EVALUATION_80PCT"
        for panel, target in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    selected["perturbation_support_contexts_pretruth"] = np.where(
        selected.target_stratum.eq("COLUMN_UNSEEN"), 0, 6
    )
    selected["excluded_all_e168_e170_e172_e174_targets"] = True
    selected["selection_used_expression_effect_error_or_prior_outcome"] = False

    if set(selected.ensembl_core.astype(str)) & excluded:
        raise RuntimeError("a prior selected target entered E176")
    for panel in PANELS:
        block = selected.loc[selected.panel_id.eq(panel)]
        if block.target_stratum.value_counts().to_dict() != {
            "DONOR_UNSEEN_ONLY": 160, "COLUMN_UNSEEN": 40
        }:
            raise RuntimeError(f"{panel} stratum counts changed")
        for stratum, expected in (
            ("DONOR_UNSEEN_ONLY", {"EVALUATION_80PCT": 128, "CALIBRATION_20PCT": 32}),
            ("COLUMN_UNSEEN", {"EVALUATION_80PCT": 32, "CALIBRATION_20PCT": 8}),
        ):
            observed = block.loc[block.target_stratum.eq(stratum)].heldout_donor_partition
            if observed.value_counts().to_dict() != expected:
                raise RuntimeError(f"{panel}/{stratum} partition counts changed")
    return selected, pool, len(excluded)


def panel_roles(helper: Any, panel: str) -> pd.DataFrame:
    roles = pd.read_csv(helper.DONOR_ROLES, keep_default_na=False).copy()
    if set(roles.donor_id.astype(str)) != set(DONORS):
        raise RuntimeError("source donor identities changed")
    rotation = ROTATIONS[panel]
    role_map = {donor: "train" for donor in rotation["train"]}
    role_map[str(rotation["validation"])] = "validation"
    role_map[str(rotation["test"])] = "test"
    roles.insert(0, "panel_id", panel)
    roles["prior_e168_role"] = roles.donor_role.astype(str)
    roles["donor_role"] = roles.donor_id.astype(str).map(role_map)
    roles["e176_role_frozen_before_targeting_x"] = True
    if roles.donor_role.value_counts().to_dict() != {"train": 6, "validation": 3, "test": 3}:
        raise RuntimeError(f"{panel} donor role counts changed")
    return roles


def build_tasks(selected: pd.DataFrame, roles: pd.DataFrame, panel: str) -> pd.DataFrame:
    donor_role = roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    rows: list[dict[str, Any]] = []
    targets = selected.loc[selected.panel_id.eq(panel)].sort_values("panel_target_rank")
    for target in targets.itertuples(index=False):
        for donor in sorted(donor_role):
            for state in STATES:
                role = str(donor_role[donor])
                if role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY":
                    split, phase = "train", "PRETRUTH_SUPERVISED"
                elif role == "validation" and target.target_stratum == "DONOR_UNSEEN_ONLY":
                    split, phase = "validation", "PRETRUTH_VALIDATION"
                elif role == "test" and target.heldout_donor_partition == "CALIBRATION_20PCT":
                    split, phase = "calibration", "POSTGATE_CALIBRATION_TRUTH"
                elif role == "test":
                    split, phase = "evaluation", "POSTCALIBRATION_EVALUATION_TRUTH"
                else:
                    split, phase = f"{role}_query_only", "FORBIDDEN_COLUMN_UNSEEN_TRUTH"
                is_test = role == "test"
                is_calibration = is_test and target.heldout_donor_partition == "CALIBRATION_20PCT"
                rows.append({
                    "task_id": f"E176::{panel}::{donor}::{state}::{target.ensembl_core}",
                    "fold_id": f"E176_{panel}_primary_CD4_four_donor_holdout_01",
                    "panel_id": panel,
                    "donor_id": donor,
                    "donor_role": role,
                    "culture_condition": state,
                    "perturbed_gene_id": target.ensembl_core,
                    "perturbed_gene_name": target.expression_axis_gene_name,
                    "scgpt_token": target.scgpt_token,
                    "target_stratum": target.target_stratum,
                    "heldout_donor_partition": target.heldout_donor_partition,
                    "split": split,
                    "truth_access_phase": phase,
                    "prediction_query_required": role in {"validation", "test"}
                    or (role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY"),
                    "risk_reference_required": role == "train"
                    and target.target_stratum == "DONOR_UNSEEN_ONLY",
                    "primary_test_task": is_test,
                    "calibration_test_task": is_calibration,
                    "evaluation_test_task": is_test and not is_calibration,
                    "training_support_contexts": int(target.perturbation_support_contexts_pretruth),
                })
    tasks = pd.DataFrame(rows)
    counts = {
        "all": len(tasks),
        "test": int(tasks.primary_test_task.sum()),
        "calibration": int(tasks.calibration_test_task.sum()),
        "evaluation": int(tasks.evaluation_test_task.sum()),
    }
    if counts != {"all": 2400, "test": 600, "calibration": 120, "evaluation": 480}:
        raise RuntimeError(f"{panel} task counts changed: {counts}")
    return tasks


def split_test_access(access: pd.DataFrame, targets: pd.DataFrame, panel: str) -> pd.DataFrame:
    partition = targets.set_index("ensembl_core").heldout_donor_partition.astype(str).to_dict()
    result = access.copy()
    test = result.x_access_phase.eq("POSTGATE_TEST_TRUTH_X")
    result.loc[test, "x_access_phase"] = result.loc[test, "ensembl_core"].map(
        lambda target: "POSTGATE_CALIBRATION_TRUTH_X"
        if partition[str(target)] == "CALIBRATION_20PCT"
        else "POSTCALIBRATION_EVALUATION_TRUTH_X"
    )
    expected = {
        "PRETRUTH_CONTROL_X": 11018,
        "PRETRUTH_TRAIN_X": 1920,
        "PRETRUTH_VALIDATION_X": 960,
        "POSTGATE_CALIBRATION_TRUTH_X": 240,
        "POSTCALIBRATION_EVALUATION_TRUTH_X": 960,
        "FORBIDDEN_COLUMN_UNSEEN_X": 720,
    }
    if result.x_access_phase.value_counts().to_dict() != expected:
        raise RuntimeError(f"{panel} access phases changed")
    return result


def write_protocol() -> None:
    protocol = """# E176｜四供体轮换、全新靶点的确认实验

## 问题

E168 和 E172 已否定旧版 SafeConf 排序稳定优于 predicted magnitude 的主张。E173 将方法收缩为可证明的模型对误差下界与少量真值校准的单侧风险上界。E174 因三种子排序不稳定，在读取测试真值前正式中止。E175 只使用 E174 的训练、验证和无标签查询预测，将估计器扩为五种子；24/24 个预注册稳定性单元通过。E176 用另一批从未进入开发的新靶点确认这个证书框架。

## 四个供体轮换

四个面板分别将 CE0006864、CE0008162、CE0008678、CE0010866 作为完整留出供体。每个供体恰好作为一次 test、一次 validation、两次 train。每个面板使用互不重叠的 200 个新靶点，其中 160 个可在训练供体观察，40 个为 column-unseen。

## 真值边界

从 5,510 个身份合格靶点中排除 E168、E170、E172、E174 的 2,600 个靶点，再按固定身份哈希选择 800 个。每个面板按身份预分配 40 个校准靶点和 160 个最终评价靶点。F2 只读控制、训练和验证表达；所有 test query 均不含 y。四面板五种子稳定性门全部通过并双远程提交后，才可开放校准真值。校准结果再次提交后，才可开放最终评价真值。

## 冻结输出

scGPT 与 GEARS 各使用 seeds 3407–3411，家族均值记作 p1 和 p2。d(p1,p2)/2 是两模型平均 RMSE 和最大 RMSE 的确定性下界。单侧 split conformal 以靶点为簇，一个靶点的三个刺激状态必须同时覆盖。每个供体用自己的 40 个校准靶点计算第 37 个次序统计量，对同供体的 160 个新靶点评价；目标覆盖率为 90%。基础误差模型固定为 E174 开真值前已选择的 magnitude 规格，不允许 E176 真值重新选特征。

## 主要报告

主要总体为 640 个最终评价靶点、1,920 个任务。报告整体与逐供体的靶点簇覆盖率、Clopper–Pearson 区间、上界宽度、精确下界违例数和平方误差分解残差。旧 SafeConf、magnitude 与 disagreement 的排序指标只作透明诊断，不再作为胜负主张。E176 仍来自同一项 Primary CD4 研究，属于多供体内部确认，不等于独立队列、湿实验或临床验证。
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(protocol, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E176 先看这个\n\nE176 冻结 800 个全新靶点和四个平衡供体轮换。当前阶段只允许身份元数据访问，targeting X 读取数为 0。\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E176 output exists: {OUT}")
    prior_paths = require_prior_boundary()
    helper = import_helper()
    byte_attestation, e168_status, source_sha = helper.verify_inputs()
    obs, access_log, schema = helper.load_identity_metadata()
    selected, pool, excluded_count = select_targets(helper, prior_paths)

    for directory in (OUT, MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    selected.to_csv(MANIFESTS / "E176_ALL_SELECTED_TARGETS.csv", index=False)
    pool[["ensembl_core", "perturbed_gene_name", "eligible_target"]].to_csv(
        TABLES / "E176_FRESH_IDENTITY_POOL.csv", index=False
    )

    all_roles: list[pd.DataFrame] = []
    all_tasks: list[pd.DataFrame] = []
    all_access: list[pd.DataFrame] = []
    for panel in PANELS:
        directory = MANIFESTS / panel
        directory.mkdir()
        targets = selected.loc[selected.panel_id.eq(panel)].copy()
        roles = panel_roles(helper, panel)
        tasks = build_tasks(selected, roles, panel)
        access = helper.build_row_access(obs, selected, roles, panel)
        access = split_test_access(access, targets, panel)
        targets.to_csv(directory / f"E176_{panel}_SELECTED_TARGETS.csv", index=False)
        roles.to_csv(directory / f"E176_{panel}_DONOR_STATE_ROLES.csv", index=False)
        tasks.to_csv(directory / f"E176_{panel}_TASK_MANIFEST.csv", index=False)
        access.to_csv(directory / f"E176_{panel}_ROW_ACCESS_MANIFEST.csv", index=False)
        all_roles.append(roles)
        all_tasks.append(tasks)
        all_access.append(access)
    roles = pd.concat(all_roles, ignore_index=True)
    tasks = pd.concat(all_tasks, ignore_index=True)
    access = pd.concat(all_access, ignore_index=True)
    roles.to_csv(MANIFESTS / "E176_ALL_DONOR_STATE_ROLES.csv", index=False)
    tasks.to_csv(MANIFESTS / "E176_ALL_TASKS.csv", index=False)
    access.to_csv(MANIFESTS / "E176_ALL_ROW_ACCESS.csv", index=False)
    pd.DataFrame(access_log).to_csv(TABLES / "E176_HDF5_VALUE_ACCESS_AUDIT.csv", index=False)
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(
        json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    source_lock = {
        "source_path": str(helper.SOURCE),
        "source_bytes": helper.EXPECTED_SOURCE_BYTES,
        "source_full_sha256": source_sha,
        "official_crc64nvme_base64": byte_attestation["computed_crc64nvme_base64"],
        "source_byte_attestation_path": str(helper.BYTE_ATTESTATION),
        "source_byte_attestation_sha256": sha256_file(helper.BYTE_ATTESTATION),
        "e168_target_audit_sha256": sha256_file(helper.TARGET_AUDIT),
        "prior_selected_target_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in prior_paths
        },
        "e175_gate_sha256": sha256_file(E175 / "aggregate/RUN_STATUS.json"),
        "allowed_hdf5_value_paths": list(helper.ALLOWED_VALUE_PATHS),
        "expression_matrix_x_decoded_by_e176_freeze": False,
        "layers_decoded_by_e176_freeze": False,
    }
    (OUT / "SOURCE_LOCK.json").write_text(
        json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    model_lock = {
        "scgpt_checkpoint_files": {str(path): sha256_file(path) for path in helper.SCGPT_FILES},
        "gears_external_go_file": {
            "path": str(helper.GEARS_GO), "sha256": sha256_file(helper.GEARS_GO)
        },
        "model_seeds": list(MODEL_SEEDS),
        "panels": list(PANELS),
        "donor_rotations": ROTATIONS,
        "gene_panel_size_each": 512,
        "registered_targets_each": 200,
        "column_unseen_each": 40,
        "deployed_seed_estimator": "five_seed_family_mean",
        "g4_seed_estimator": "five_leave_one_seed_out_four_seed_family_means",
        "primary_method": "pair_lower_certificate_plus_donor_specific_split_conformal_upper_bound",
        "base_error_model_spec": "magnitude_from_e174_method_freeze",
        "metadata_freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "pretruth_test_targeting_x_access_count_required": 0,
        "deployment_authorized": False,
    }
    (OUT / "MODEL_INPUT_LOCK.json").write_text(
        json.dumps(model_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    stat_lock = {
        "development_population": "E168_plus_E172_outcomes_and_E174_truth_blind_predictions",
        "calibration_population": "E176_40_targets_per_heldout_donor_160_total",
        "primary_evaluation_population": "E176_160_targets_per_heldout_donor_640_total",
        "pair_lower_certificate": "rmse(p1,p2)/2",
        "conformal_target": "pair_mean_rmse_and_ensemble_rmse",
        "target_cluster": "target_gene_three_states_simultaneously",
        "target_coverage": 0.90,
        "miscoverage_alpha": 0.10,
        "calibration_target_count_each_donor": 40,
        "finite_sample_order_rank_each_donor": math.ceil((40 + 1) * 0.90),
        "evaluation_target_count_each_donor": 160,
        "total_evaluation_target_count": 640,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026071906,
        "alpha": 0.05,
        "evaluation_truth_may_select_or_modify_method": False,
        "optional_stopping_or_panel_dropping_allowed": False,
        "all_four_donor_panels_required": True,
        "same_study_multi_donor_not_independent_study": True,
        "deployment_authorized": False,
    }
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(
        json.dumps(stat_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    write_protocol()

    report = f"""# E176 metadata freeze report

- Label-free eligible universe: {e168_status['n_eligible_targets']:,}.
- Excluded targets from E168/E170/E172/E174: {excluded_count:,}; fresh pool: {len(pool):,}.
- Frozen new targets: {len(selected):,}; each panel contains 160 seen + 40 column-unseen.
- Four donors are each used once as test, once as validation, and twice as train.
- Calibration targets: {int(selected.heldout_donor_partition.eq('CALIBRATION_20PCT').sum()):,}; sealed evaluation targets: {int(selected.heldout_donor_partition.eq('EVALUATION_80PCT').sum()):,}.
- Expression X values decoded: **0**; outcome values used for identity selection: **0**.
- Official source SHA-256: {source_sha}.
"""
    (REPORTS / "E176_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = sorted(path for path in OUT.rglob("*")
                       if path.is_file() and path.name != "RUN_STATUS.json")
    status = {
        "schema": "safeconf_e176_metadata_freeze_v1",
        "experiment": "E176_four_donor_fresh_confirmation",
        "stage": "F1_METADATA_FREEZE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "source_schema": schema,
        "n_eligible_targets": 5510,
        "n_prior_targets_excluded": excluded_count,
        "n_fresh_identity_pool": len(pool),
        "n_panels": len(PANELS),
        "n_selected_targets": len(selected),
        "n_calibration_targets": int(selected.heldout_donor_partition.eq("CALIBRATION_20PCT").sum()),
        "n_evaluation_targets": int(selected.heldout_donor_partition.eq("EVALUATION_80PCT").sum()),
        "n_calibration_tasks": int(tasks.calibration_test_task.sum()),
        "n_evaluation_tasks": int(tasks.evaluation_test_task.sum()),
        "all_expression_x_values_read": 0,
        "e176_calibration_targeting_x_values_read": 0,
        "e176_evaluation_targeting_x_values_read": 0,
        "five_seed_estimator_frozen": True,
        "all_four_donor_panels_required": True,
        "evaluation_truth_may_modify_method": False,
        "same_study_multi_donor_not_independent_study": True,
        "deployment_authorized": False,
        "artifact_sha256": {
            path.relative_to(OUT).as_posix(): sha256_file(path) for path in artifacts
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

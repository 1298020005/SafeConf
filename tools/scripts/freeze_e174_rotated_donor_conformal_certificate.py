#!/usr/bin/env python3
"""Freeze E174 before any expression value from its 800 targets is opened.

E174 rotates donor roles, excludes every target used by E168/E170/E172, and
pre-assigns 20% of the held-out-donor target columns to conformal calibration.
The remaining 80% stay sealed for the final evaluation.  This program opens
only the identity metadata allowlisted by the audited E170 freezer; ``X`` and
``layers`` are never indexed.
"""

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
E173 = ROOT / "docs/实验结果/E173_falsification_aware_pair_certificate_20260719"
OUT = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

PANELS = ("R01", "R02", "R03", "R04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
TARGETS_PER_PANEL = 200
COLUMN_UNSEEN_PER_PANEL = 40
CALIBRATION_PER_PANEL = 40
CALIBRATION_SEEN_PER_PANEL = 32
CALIBRATION_COLUMN_UNSEEN_PER_PANEL = 8
TOTAL_TARGETS = 800
TARGET_SALT = "E174_ROTATED_DONOR_FRESH_TARGET_V1"
UNSEEN_SALT = "E174_COLUMN_UNSEEN_V1"
CALIBRATION_SALT = "E174_HELDOUT_DONOR_CALIBRATION_20PCT_V1"
MODEL_SEEDS = (3407, 3408, 3409)
DONOR_ROLE_MAP = {
    "CE0008162": "train",
    "CE0010866": "train",
    "CE0006864": "validation",
    "CE0008678": "test",
}


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e174_freeze_helper", HELPER_PATH)
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


def verify_prior_boundary() -> None:
    required = [
        E168 / "manifests/E168_SELECTED_TARGETS.csv",
        E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv",
        E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv",
        E172 / "postgate_release/RUN_STATUS.json",
        E173 / "RUN_STATUS.json",
        E173 / "MANIFEST.sha256",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prior evidence: {missing}")
    e172 = json.loads((E172 / "postgate_release/RUN_STATUS.json").read_text())
    e173 = json.loads((E173 / "RUN_STATUS.json").read_text())
    if e172.get("decision") != "NO_TARGET_REPLICATION":
        raise RuntimeError("E172 decision changed")
    if (
        e173.get("status") != "COMPLETE"
        or e173.get("fixed_safeconf_stable_increment_vs_magnitude_supported") is not False
        or e173.get("incremental_router_state_on_e171_validation") != "ABSTAIN"
    ):
        raise RuntimeError("E173 falsification boundary changed")
    if sha256_file(E173 / "MANIFEST.sha256") != e173.get("manifest_sha256"):
        raise RuntimeError("E173 manifest hash changed")


def select_targets(helper: Any) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    audit = pd.read_csv(helper.TARGET_AUDIT, keep_default_na=False)
    eligible = audit.loc[audit.eligible_target.astype(str).str.lower().eq("true")].copy()
    if len(eligible) != 5510 or eligible.ensembl_core.nunique() != 5510:
        raise RuntimeError("label-free eligible target universe changed")
    if not eligible.n_identity_complete_guides.astype(int).eq(2).all():
        raise RuntimeError("E174 requires exactly two identity-complete guides per target")

    prior_paths = (
        E168 / "manifests/E168_SELECTED_TARGETS.csv",
        E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv",
        E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv",
    )
    prior_sets = [set(pd.read_csv(path).ensembl_core.astype(str)) for path in prior_paths]
    if [len(values) for values in prior_sets] != [200, 800, 800]:
        raise RuntimeError("prior target manifest sizes changed")
    excluded = set().union(*prior_sets)
    if len(excluded) != 1800:
        raise RuntimeError(f"prior target sets overlap or changed: union={len(excluded)}")
    pool = eligible.loc[~eligible.ensembl_core.astype(str).isin(excluded)].copy()
    if len(pool) != 3710:
        raise RuntimeError(f"fresh identity pool changed: {len(pool)}")

    pool["e174_selection_sha256"] = pool.ensembl_core.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    selected = pool.sort_values(
        ["e174_selection_sha256", "ensembl_core"], kind="stable"
    ).head(TOTAL_TARGETS).copy()
    selected.insert(0, "e174_global_target_rank", np.arange(1, TOTAL_TARGETS + 1))
    selected.insert(1, "panel_id", np.repeat(PANELS, TARGETS_PER_PANEL))
    selected.insert(
        2,
        "panel_target_rank",
        np.tile(np.arange(1, TARGETS_PER_PANEL + 1), len(PANELS)),
    )

    selected["e174_column_unseen_sha256"] = [
        stable_hash(UNSEEN_SALT, panel, ensembl)
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    unseen: set[tuple[str, str]] = set()
    for panel in PANELS:
        block = selected.loc[selected.panel_id.eq(panel)].sort_values(
            ["e174_column_unseen_sha256", "ensembl_core"], kind="stable"
        )
        unseen.update(
            (panel, value)
            for value in block.head(COLUMN_UNSEEN_PER_PANEL).ensembl_core.astype(str)
        )
    selected["target_stratum"] = [
        "COLUMN_UNSEEN" if (panel, str(ensembl)) in unseen else "DONOR_UNSEEN_ONLY"
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]

    selected["e174_calibration_sha256"] = [
        stable_hash(CALIBRATION_SALT, panel, stratum, ensembl)
        for panel, stratum, ensembl in zip(
            selected.panel_id,
            selected.target_stratum,
            selected.ensembl_core,
            strict=True,
        )
    ]
    calibration: set[tuple[str, str]] = set()
    for panel in PANELS:
        for stratum, count in (
            ("DONOR_UNSEEN_ONLY", CALIBRATION_SEEN_PER_PANEL),
            ("COLUMN_UNSEEN", CALIBRATION_COLUMN_UNSEEN_PER_PANEL),
        ):
            block = selected.loc[
                selected.panel_id.eq(panel) & selected.target_stratum.eq(stratum)
            ].sort_values(["e174_calibration_sha256", "ensembl_core"], kind="stable")
            calibration.update(
                (panel, value) for value in block.head(count).ensembl_core.astype(str)
            )
    selected["heldout_donor_partition"] = [
        "CALIBRATION_20PCT" if (panel, str(ensembl)) in calibration else "EVALUATION_80PCT"
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    selected["perturbation_support_contexts_pretruth"] = np.where(
        selected.target_stratum.eq("COLUMN_UNSEEN"), 0, 6
    )
    selected["excluded_all_e168_e170_e172_targets"] = True
    selected["selection_used_expression_effect_error_or_e173_outcome"] = False

    if set(selected.ensembl_core.astype(str)) & excluded:
        raise RuntimeError("a prior selected target entered E174")
    strata = selected.groupby("panel_id").target_stratum.value_counts().unstack(fill_value=0)
    partitions = (
        selected.groupby(["panel_id", "target_stratum"])
        .heldout_donor_partition.value_counts()
        .unstack(fill_value=0)
    )
    for panel in PANELS:
        if strata.loc[panel].to_dict() != {
            "COLUMN_UNSEEN": 40,
            "DONOR_UNSEEN_ONLY": 160,
        }:
            raise RuntimeError(f"{panel} target strata changed")
        if partitions.loc[(panel, "DONOR_UNSEEN_ONLY")].to_dict() != {
            "CALIBRATION_20PCT": 32,
            "EVALUATION_80PCT": 128,
        }:
            raise RuntimeError(f"{panel} seen calibration split changed")
        if partitions.loc[(panel, "COLUMN_UNSEEN")].to_dict() != {
            "CALIBRATION_20PCT": 8,
            "EVALUATION_80PCT": 32,
        }:
            raise RuntimeError(f"{panel} unseen calibration split changed")
    return selected, pool, len(excluded)


def rotated_roles(helper: Any) -> pd.DataFrame:
    roles = pd.read_csv(helper.DONOR_ROLES, keep_default_na=False).copy()
    if set(roles.donor_id.astype(str)) != set(DONOR_ROLE_MAP):
        raise RuntimeError("source donor identities changed")
    roles["prior_e168_role"] = roles.donor_role.astype(str)
    roles["donor_role"] = roles.donor_id.astype(str).map(DONOR_ROLE_MAP)
    roles["e174_role_frozen_before_targeting_x"] = True
    if roles.donor_role.value_counts().to_dict() != {
        "train": 6,
        "validation": 3,
        "test": 3,
    }:
        raise RuntimeError("rotated donor role counts changed")
    return roles


def build_tasks(selected: pd.DataFrame, roles: pd.DataFrame, panel: str) -> pd.DataFrame:
    donor_role = roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    rows: list[dict[str, Any]] = []
    for target in selected.loc[selected.panel_id.eq(panel)].sort_values("panel_target_rank").itertuples(index=False):
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
                rows.append(
                    {
                        "task_id": f"E174::{panel}::{donor}::{state}::{target.ensembl_core}",
                        "fold_id": f"E174_{panel}_primary_CD4_rotated_donor_holdout_01",
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
                    }
                )
    tasks = pd.DataFrame(rows)
    checks = {
        "all": len(tasks),
        "test": int(tasks.primary_test_task.sum()),
        "calibration": int(tasks.calibration_test_task.sum()),
        "evaluation": int(tasks.evaluation_test_task.sum()),
    }
    if checks != {"all": 2400, "test": 600, "calibration": 120, "evaluation": 480}:
        raise RuntimeError(f"{panel} task counts changed: {checks}")
    return tasks


def split_test_row_access(access: pd.DataFrame, selected: pd.DataFrame, panel: str) -> pd.DataFrame:
    partition = (
        selected.loc[selected.panel_id.eq(panel)]
        .set_index("ensembl_core")
        .heldout_donor_partition.astype(str)
        .to_dict()
    )
    access = access.copy()
    test = access.x_access_phase.eq("POSTGATE_TEST_TRUTH_X")
    access.loc[test, "x_access_phase"] = access.loc[test, "ensembl_core"].map(
        lambda value: (
            "POSTGATE_CALIBRATION_TRUTH_X"
            if partition[str(value)] == "CALIBRATION_20PCT"
            else "POSTCALIBRATION_EVALUATION_TRUTH_X"
        )
    )
    counts = access.x_access_phase.value_counts().to_dict()
    expected = {
        "PRETRUTH_CONTROL_X": 11018,
        "PRETRUTH_TRAIN_X": 1920,
        "PRETRUTH_VALIDATION_X": 960,
        "POSTGATE_CALIBRATION_TRUTH_X": 240,
        "POSTCALIBRATION_EVALUATION_TRUTH_X": 960,
        "FORBIDDEN_COLUMN_UNSEEN_X": 720,
    }
    if counts != expected:
        raise RuntimeError(f"{panel} access phases changed: {counts}")
    return access


def write_protocol() -> None:
    text = """# E174｜轮换供体上的小矩阵校准与隐藏靶点评价

## 已经排除的主张

E168 与 E172 两次冻结测试都没有确认固定 SafeConf 分数稳定优于 predicted magnitude。E174 不再把这项已被否定的增量当主要终点，也不允许用 E174 真值重新挑公式。系统收缩为两个可核查输出：由两模型预测分歧给出的精确模型对风险下界，以及在少量已观测靶点上校准的单侧风险上界。经验路由未在 validation 上证明增量时保持 `ABSTAIN`。

## 新供体角色与新靶点

- 训练供体：CE0008162、CE0010866；验证供体：CE0006864；最终留出供体：CE0008678。相较 E168/E172，测试供体由 CE0010866 轮换为 CE0008678，且一位原测试供体进入训练、一位原训练供体进入验证。
- 从 5,510 个身份合格靶点中排除 E168/E170/E172 已选择的 1,800 个，再按新的身份哈希一次性固定 R01–R04 共 800 个靶点。
- 每个面板 160 个 donor-unseen-only 与 40 个 column-unseen；选择过程不读取 X、counts、effect、error、DE、guide efficacy 或 E173 结果。

## 20% 校准小矩阵与 80% 最终评价

每个面板在读取表达前按靶点身份固定 40 个校准靶点（32 seen + 8 column-unseen），其余 160 个作为最终评价（128 + 32）。四面板合计 160 个校准靶点、640 个评价靶点；一个靶点的三个刺激状态始终进入同一分区。

执行顺序固定为：F2 仅训练并生成全部测试查询预测，测试 targeting X 读取数必须为 0；四面板 RIAG 通过并双远程提交后，F3A 只开放 160 个校准靶点；校准器、有限样本分位数和哈希再次提交后，F3B 才能开放剩余 640 个评价靶点。F3A 与 F3B 使用物理分离的资产目录和逐行访问清单。

## 冻结方法

对 scGPT 与 GEARS 三 seed family mean 预测记为 p1、p2，512 基因 RMSE 距离为 d。由三角不等式，`L_pair=d(p1,p2)/2` 在不读取真值时同时下界两模型 RMSE 的算术平均与两者最大 RMSE；低分歧不能解释为安全。

风险上界使用一侧 split conformal。基础误差估计器只在已解封的 E168+E172 3,000 个任务上拟合，输入为 predicted magnitude、d/2、状态与 seen/unseen 标记；具体特征、正则强度和回退规则在开放 E174 校准真值前写入代码并双远程冻结。F3A 对每个校准靶点取三个状态中最大的 `observed_error-base_estimate`，以 90% 目标覆盖率采用有限样本 order statistic `ceil((n+1)*0.90)`。最终上界不低于精确下界。magnitude-only 与 state/stratum constant conformal 是强基线；复合估计器只有在既有开发数据的预注册效率门通过时才进入主输出，否则自动回退，不使用 E174 评价真值决定。

## 最终评价

- 主要总体：E174 的 640 个隐藏评价靶点、1,920 个任务；E168/E170/E172 和 160 个校准靶点不进入主要统计量。
- 核查 1：pair-mean 与 pair-max 的精确下界违例必须为 0，平方误差分解残差不超过 1e-8。
- 核查 2：按靶点簇统计三个状态同时被 90% conformal 上界覆盖的比例，并给出 exact binomial 区间；覆盖是有限样本方法核查，不做事后阈值修正。
- 核查 3：报告上界宽度、相对 constant/magnitude 基线的效率、证书触发率与 target-cluster bootstrap 区间。
- 固定 SafeConf、magnitude、disagreement 的排序相关与 AURC 仅作透明的次要诊断；不得把未通过的比较改写成优势。

## 解释边界

E174 是同一 Primary CD4 研究中的供体角色轮换与全新靶点测试，强于重复使用同一测试供体，但仍不是独立研究、临床验证或湿实验。任何期刊分区和录用都不能由本实验保证。
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(text, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E174 先看这个\n\n先读 `PREREG_ANALYSIS_PLAN.md` 与 `reports/E174_METADATA_FREEZE_REPORT.md`。当前只冻结身份、供体角色和三阶段访问边界；800 个目标的 expression X 读取数为 0。\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"append-only E174 output exists: {OUT}")
    verify_prior_boundary()
    helper = import_helper()
    byte_attestation, e168_status, source_sha = helper.verify_inputs()
    obs, access_log, schema = helper.load_identity_metadata()
    selected, pool, excluded_count = select_targets(helper)
    roles = rotated_roles(helper)

    for directory in (OUT, MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    roles.to_csv(MANIFESTS / "E174_DONOR_STATE_ROLES.csv", index=False)
    selected.to_csv(MANIFESTS / "E174_ALL_SELECTED_TARGETS.csv", index=False)
    pool[["ensembl_core", "perturbed_gene_name", "eligible_target"]].to_csv(
        TABLES / "E174_FRESH_IDENTITY_POOL.csv", index=False
    )

    all_tasks: list[pd.DataFrame] = []
    all_access: list[pd.DataFrame] = []
    for panel in PANELS:
        directory = MANIFESTS / panel
        directory.mkdir()
        targets = selected.loc[selected.panel_id.eq(panel)].copy()
        tasks = build_tasks(selected, roles, panel)
        access = helper.build_row_access(obs, selected, roles, panel)
        access = split_test_row_access(access, selected, panel)
        targets.to_csv(directory / f"E174_{panel}_SELECTED_TARGETS.csv", index=False)
        tasks.to_csv(directory / f"E174_{panel}_TASK_MANIFEST.csv", index=False)
        access.to_csv(directory / f"E174_{panel}_ROW_ACCESS_MANIFEST.csv", index=False)
        all_tasks.append(tasks)
        all_access.append(access)
    tasks = pd.concat(all_tasks, ignore_index=True)
    access = pd.concat(all_access, ignore_index=True)
    tasks.to_csv(MANIFESTS / "E174_ALL_TASKS.csv", index=False)
    access.to_csv(MANIFESTS / "E174_ALL_ROW_ACCESS.csv", index=False)
    pd.DataFrame(access_log).to_csv(TABLES / "E174_HDF5_VALUE_ACCESS_AUDIT.csv", index=False)
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(
        json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prior_target_paths = (
        E168 / "manifests/E168_SELECTED_TARGETS.csv",
        E170 / "manifests/E170_ALL_SELECTED_TARGETS.csv",
        E172 / "manifests/E172_ALL_SELECTED_TARGETS.csv",
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
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in prior_target_paths
        },
        "e173_manifest_sha256": sha256_file(E173 / "MANIFEST.sha256"),
        "allowed_hdf5_value_paths": list(helper.ALLOWED_VALUE_PATHS),
        "expression_matrix_x_decoded_by_e174_freeze": False,
        "layers_decoded_by_e174_freeze": False,
    }
    (OUT / "SOURCE_LOCK.json").write_text(
        json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    model_lock = {
        "scgpt_checkpoint_files": {str(path): sha256_file(path) for path in helper.SCGPT_FILES},
        "gears_external_go_file": {
            "path": str(helper.GEARS_GO),
            "sha256": sha256_file(helper.GEARS_GO),
        },
        "model_seeds": list(MODEL_SEEDS),
        "panels": list(PANELS),
        "gene_panel_size_each": 512,
        "registered_targets_each": 200,
        "column_unseen_each": 40,
        "deployed_seed_estimator": "three_seed_family_mean",
        "g4_seed_estimator": "three_leave_one_seed_out_two_seed_family_means",
        "legacy_safeconf_formula_modified": False,
        "primary_method": "pair_lower_certificate_plus_split_conformal_upper_bound",
        "metadata_freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "pretruth_test_targeting_x_access_count_required": 0,
        "deployment_authorized": False,
    }
    (OUT / "MODEL_INPUT_LOCK.json").write_text(
        json.dumps(model_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    stat_lock = {
        "development_population": "E168_plus_E172_1000_prior_targets_3000_tasks",
        "calibration_population": "E174_rotated_test_donor_160_identity_frozen_targets_480_tasks",
        "primary_evaluation_population": "E174_rotated_test_donor_640_hidden_targets_1920_tasks",
        "pair_lower_certificate": "rmse(p1,p2)/2",
        "conformal_target": "pair_mean_rmse_and_ensemble_rmse",
        "target_cluster": "target_gene_three_states_simultaneously",
        "target_coverage": 0.90,
        "miscoverage_alpha": 0.10,
        "calibration_target_count": 160,
        "finite_sample_order_rank": math.ceil((160 + 1) * 0.90),
        "evaluation_target_count": 640,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026071903,
        "alpha": 0.05,
        "legacy_ranking_comparators": ["magnitude", "disagreement", "fixed_safeconf"],
        "evaluation_truth_may_select_or_modify_method": False,
        "optional_stopping_or_panel_dropping_allowed": False,
        "same_study_rotated_donor_not_independent_study": True,
        "deployment_authorized": False,
    }
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(
        json.dumps(stat_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    write_protocol()

    report = f"""# E174 metadata freeze report

- Label-free eligible universe: {e168_status['n_eligible_targets']:,}.
- Excluded prior selected targets: {excluded_count:,}; fresh identity pool: {len(pool):,}.
- Frozen new targets: {len(selected):,}; four panels each contain 160 seen + 40 column-unseen.
- Rotated test donor: CE0008678; validation donor: CE0006864; train donors: CE0008162 and CE0010866.
- Held-out donor calibration targets: {int(selected.heldout_donor_partition.eq('CALIBRATION_20PCT').sum()):,}; sealed evaluation targets: {int(selected.heldout_donor_partition.eq('EVALUATION_80PCT').sum()):,}.
- Calibration tasks: {int(tasks.calibration_test_task.sum()):,}; sealed evaluation tasks: {int(tasks.evaluation_test_task.sum()):,}.
- Expression X values decoded: **0**; prior outcomes used for identity selection: **0**.
- Official source SHA-256 recomputed before identity metadata opening: `{source_sha}`.
"""
    (REPORTS / "E174_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = sorted(
        path for path in OUT.rglob("*") if path.is_file() and path.name != "RUN_STATUS.json"
    )
    status = {
        "schema": "safeconf_e174_metadata_freeze_v1",
        "experiment": "E174_rotated_donor_conformal_certificate",
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
        "e174_calibration_targeting_x_values_read": 0,
        "e174_evaluation_targeting_x_values_read": 0,
        "test_donor_rotated_from_e168_e172": True,
        "fixed_safeconf_increment_claim_retired": True,
        "evaluation_truth_may_modify_method": False,
        "same_study_rotated_donor_not_independent_study": True,
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

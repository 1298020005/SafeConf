#!/usr/bin/env python3
"""Freeze four expression-blind E170 target-replication panels.

E168 has already been unsealed.  This program therefore excludes all 200
E168 targets and selects four non-overlapping panels from target identities
whose expression values have not been used by E168.  It opens only the
explicitly allowlisted AnnData identity metadata below; ``X`` and ``layers``
are never indexed.  All four panels are fixed together, before any E170
targeting expression is read.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E168 = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
OUT = ROOT / "docs/实验结果/E170_primary_cd4_multipanel_precision_20260718"
MANIFESTS = OUT / "manifests"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
SOURCE = DATA_ROOT / "source/GWCD4i.pseudobulk_merged.h5ad"
BYTE_ATTESTATION = DATA_ROOT / "E168_SOURCE_BYTE_ATTESTATION.json"
SCGPT_DIR = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
SCGPT_FILES = tuple(SCGPT_DIR / name for name in ("args.json", "vocab.json", "best_model.pt"))
GEARS_GO = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")

TARGET_AUDIT = E168 / "tables/E168_TARGET_ELIGIBILITY_AUDIT.csv"
PRIMARY_TARGETS = E168 / "manifests/E168_SELECTED_TARGETS.csv"
DONOR_ROLES = E168 / "manifests/E168_DONOR_STATE_ROLES.csv"
E168_STATUS = E168 / "RUN_STATUS.json"
E168_SOURCE_LOCK = E168 / "SOURCE_LOCK.json"

ALLOWED_VALUE_PATHS = (
    "obs/10xrun_id",
    "obs/donor_id",
    "obs/culture_condition",
    "obs/guide_id",
    "obs/perturbed_gene_name",
    "obs/perturbed_gene_id",
    "obs/guide_type",
)
FORBIDDEN_VALUE_ROOTS = ("X", "layers", "obsm", "obsp", "varm", "varp")
STATES = ("Rest", "Stim8hr", "Stim48hr")
PANELS = ("P01", "P02", "P03", "P04")
TARGETS_PER_PANEL = 200
COLUMN_UNSEEN_PER_PANEL = 40
TOTAL_TARGETS = len(PANELS) * TARGETS_PER_PANEL
TARGET_SALT = "E170_FRESH_TARGET_MULTIPANEL_V1"
UNSEEN_SALT = "E170_COLUMN_UNSEEN_V1"
MODEL_SEEDS = (3407, 3408, 3409)
EXPECTED_SOURCE_SHA256 = "fd2b8c21d357f8699ec34e2d5ebc1639612c27a0147a9ca94d4983822d93247e"
EXPECTED_SOURCE_BYTES = 44_566_657_140


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_hash(salt: str, *values: object) -> str:
    payload = b"\0".join([salt.encode(), *[str(value).encode() for value in values]])
    return hashlib.sha256(payload).hexdigest()


def decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "OSU":
        return values
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values],
        dtype=object,
    )


def read_metadata_value(handle: h5py.File, path: str, access: list[dict[str, Any]]) -> np.ndarray:
    if path not in ALLOWED_VALUE_PATHS:
        raise PermissionError(f"E170 metadata freeze denied HDF5 path: {path}")
    node = handle[path]
    if isinstance(node, h5py.Group):
        if set(node.keys()) != {"categories", "codes"}:
            raise RuntimeError(f"unexpected categorical encoding at {path}: {sorted(node.keys())}")
        categories = decode_array(node["categories"][...])
        codes = np.asarray(node["codes"][...], dtype=np.int64)
        values = np.asarray(
            [categories[code] if code >= 0 else "" for code in codes], dtype=object
        )
        encoding = "categorical/categories+codes"
    else:
        values = decode_array(node[...])
        encoding = str(node.attrs.get("encoding-type", "array"))
    access.append(
        {
            "hdf5_path": path,
            "value_access": True,
            "n_values": int(len(values)),
            "encoding": encoding,
            "purpose": "identity-only E170 panel and row-access freeze",
            "expression_or_outcome_value": False,
        }
    )
    return values


def canonical_type(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], str]:
    required = [
        SOURCE, BYTE_ATTESTATION, TARGET_AUDIT, PRIMARY_TARGETS, DONOR_ROLES,
        E168_STATUS, E168_SOURCE_LOCK, GEARS_GO, *SCGPT_FILES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen E170 input: {missing}")
    if SOURCE.is_symlink() or SOURCE.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise RuntimeError("official source is missing, symlinked, or has a changed byte length")
    attestation = json.loads(BYTE_ATTESTATION.read_text(encoding="utf-8"))
    attestation_required = {
        "assembled_path": str(SOURCE),
        "assembled_size": EXPECTED_SOURCE_BYTES,
        "sha256": EXPECTED_SOURCE_SHA256,
        "computed_crc64nvme_base64": "E2slkXBEb2c=",
        "crc64nvme_matches_official_full_object_checksum": True,
        "hdf5_opened": False,
        "expression_values_decoded": False,
    }
    mismatches = {
        key: {"expected": value, "observed": attestation.get(key)}
        for key, value in attestation_required.items()
        if attestation.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"source byte attestation mismatch: {mismatches}")

    e168_status = json.loads(E168_STATUS.read_text(encoding="utf-8"))
    for path in (TARGET_AUDIT, PRIMARY_TARGETS, DONOR_ROLES):
        relative = path.relative_to(E168).as_posix()
        expected = e168_status.get("artifact_sha256", {}).get(relative)
        observed = sha256_file(path)
        if not expected or expected != observed:
            raise RuntimeError(f"E168 label-free input hash changed: {relative}")

    # A complete pass is deliberately done before opening HDF5.  Reading bytes
    # for a checksum is not an expression-value access.
    observed_source_sha = sha256_file(SOURCE)
    if observed_source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"official source SHA-256 changed: {observed_source_sha}")
    return attestation, e168_status, observed_source_sha


def load_identity_metadata() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    access: list[dict[str, Any]] = []
    with h5py.File(SOURCE, "r") as handle:
        roots = sorted(map(str, handle.keys()))
        expected_roots = ["X", "layers", "obs", "obsm", "obsp", "uns", "var", "varm", "varp"]
        if roots != expected_roots:
            raise RuntimeError(f"AnnData roots changed: {roots}")
        obs = pd.DataFrame(
            {
                path.removeprefix("obs/"): read_metadata_value(handle, path, access)
                for path in ALLOWED_VALUE_PATHS
            }
        )
        x_node = handle["X"]
        if isinstance(x_node, h5py.Group):
            x_shape = tuple(map(int, x_node.attrs["shape"]))
        else:
            x_shape = tuple(map(int, x_node.shape))
    if x_shape != (278_684, 18_129) or len(obs) != x_shape[0]:
        raise RuntimeError(f"source schema changed: obs={len(obs)}, X={x_shape}")
    obs.insert(0, "metadata_row_index", np.arange(len(obs), dtype=np.int64))
    schema = {
        "anndata_root_keys_metadata_only": roots,
        "n_obs": int(x_shape[0]),
        "n_vars": int(x_shape[1]),
        "allowed_hdf5_values_decoded": list(ALLOWED_VALUE_PATHS),
        "forbidden_value_roots_decoded": [],
        "expression_matrix_X_decoded": False,
        "layers_decoded": False,
    }
    return obs, access, schema


def select_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.read_csv(TARGET_AUDIT, keep_default_na=False)
    primary = pd.read_csv(PRIMARY_TARGETS, keep_default_na=False)
    eligible = target.loc[target.eligible_target.astype(str).str.lower().eq("true")].copy()
    if len(eligible) != 5510 or eligible.ensembl_core.nunique() != 5510:
        raise RuntimeError(f"eligible target universe changed: {len(eligible)}")
    if not eligible.n_identity_complete_guides.astype(int).eq(2).all():
        raise RuntimeError("E170 requires exactly two identity-complete guides per target")
    primary_ids = set(primary.ensembl_core.astype(str))
    pool = eligible.loc[~eligible.ensembl_core.astype(str).isin(primary_ids)].copy()
    if len(pool) != 5310:
        raise RuntimeError(f"E170 untouched identity pool changed: {len(pool)}")
    pool["e170_selection_sha256"] = pool.ensembl_core.map(
        lambda value: stable_hash(TARGET_SALT, value)
    )
    selected = pool.sort_values(["e170_selection_sha256", "ensembl_core"], kind="stable").head(TOTAL_TARGETS).copy()
    if len(selected) != TOTAL_TARGETS:
        raise RuntimeError("insufficient E170 targets")
    selected.insert(0, "e170_global_target_rank", np.arange(1, TOTAL_TARGETS + 1))
    selected.insert(1, "panel_id", np.repeat(PANELS, TARGETS_PER_PANEL))
    selected.insert(2, "panel_target_rank", np.tile(np.arange(1, TARGETS_PER_PANEL + 1), len(PANELS)))
    selected["e170_column_unseen_sha256"] = [
        stable_hash(UNSEEN_SALT, panel, ensembl)
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    unseen: set[tuple[str, str]] = set()
    for panel in PANELS:
        frame = selected.loc[selected.panel_id.eq(panel)].sort_values(
            ["e170_column_unseen_sha256", "ensembl_core"], kind="stable"
        )
        unseen.update((panel, value) for value in frame.head(COLUMN_UNSEEN_PER_PANEL).ensembl_core.astype(str))
    selected["target_stratum"] = [
        "COLUMN_UNSEEN" if (panel, str(ensembl)) in unseen else "DONOR_UNSEEN_ONLY"
        for panel, ensembl in zip(selected.panel_id, selected.ensembl_core, strict=True)
    ]
    selected["perturbation_support_contexts_pretruth"] = np.where(
        selected.target_stratum.eq("COLUMN_UNSEEN"), 0, 6
    )
    selected["excluded_e168_primary_target"] = True
    selected["selection_used_expression_abundance_effect_or_error"] = False
    if set(selected.ensembl_core.astype(str)) & primary_ids:
        raise RuntimeError("an E168 primary target entered E170")
    counts = selected.groupby("panel_id").target_stratum.value_counts().unstack(fill_value=0)
    expected = {"COLUMN_UNSEEN": 40, "DONOR_UNSEEN_ONLY": 160}
    if any(row.to_dict() != expected for _, row in counts.iterrows()):
        raise RuntimeError(f"panel strata changed: {counts.to_dict()}")
    return selected, pool


def build_tasks(selected: pd.DataFrame, roles: pd.DataFrame, panel: str) -> pd.DataFrame:
    panel_targets = selected.loc[selected.panel_id.eq(panel)].copy()
    donor_role = roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    rows: list[dict[str, Any]] = []
    for target in panel_targets.sort_values("panel_target_rank").itertuples(index=False):
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
                        "task_id": f"E170::{panel}::{donor}::{state}::{target.ensembl_core}",
                        "fold_id": f"E170_{panel}_primary_CD4_donor_holdout_01",
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
                        "prediction_query_required": role in {"validation", "test"} or (
                            role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY"
                        ),
                        "risk_reference_required": role == "train" and target.target_stratum == "DONOR_UNSEEN_ONLY",
                        "primary_test_task": role == "test",
                        "training_support_contexts": int(target.perturbation_support_contexts_pretruth),
                    }
                )
    tasks = pd.DataFrame(rows)
    if len(tasks) != 2400 or int(tasks.primary_test_task.sum()) != 600:
        raise RuntimeError(f"{panel} task count failed")
    return tasks


def build_row_access(
    obs: pd.DataFrame,
    selected: pd.DataFrame,
    roles: pd.DataFrame,
    panel: str,
) -> pd.DataFrame:
    targets = selected.loc[selected.panel_id.eq(panel)].copy()
    donor_role = roles.drop_duplicates("donor_id").set_index("donor_id").donor_role.to_dict()
    expected_by_guide = {
        guide: str(row.ensembl_core)
        for row in targets.itertuples(index=False)
        for guide in str(row.eligible_guide_ids).split("+")
    }
    target_info = targets.set_index("ensembl_core").to_dict("index")
    frame = obs.copy()
    frame["guide_type_normalized"] = frame.guide_type.map(canonical_type)
    if not set(frame.guide_type_normalized.unique()).issubset({"targeting", "non-targeting"}):
        raise RuntimeError("unexpected guide type")
    relevant = frame.loc[
        frame.guide_type_normalized.eq("non-targeting")
        | frame.guide_id.astype(str).isin(expected_by_guide)
    ].copy()
    relevant["observed_ensembl_core"] = relevant.perturbed_gene_id.astype(str).str.split(".").str[0]
    relevant["expected_ensembl_core"] = relevant.guide_id.astype(str).map(expected_by_guide).fillna("")
    targeting = relevant.guide_type_normalized.eq("targeting")
    if not relevant.loc[targeting, "observed_ensembl_core"].eq(
        relevant.loc[targeting, "expected_ensembl_core"]
    ).all():
        raise RuntimeError(f"{panel} guide-to-target mapping changed")
    relevant["ensembl_core"] = relevant.expected_ensembl_core
    phases, strata, role_values = [], [], []
    for row in relevant.itertuples(index=False):
        role = str(donor_role[str(row.donor_id)])
        if row.guide_type_normalized == "non-targeting":
            phase, stratum = "PRETRUTH_CONTROL_X", "CONTROL"
        else:
            stratum = str(target_info[str(row.ensembl_core)]["target_stratum"])
            if role == "test":
                phase = "POSTGATE_TEST_TRUTH_X"
            elif stratum == "COLUMN_UNSEEN":
                phase = "FORBIDDEN_COLUMN_UNSEEN_X"
            elif role == "train":
                phase = "PRETRUTH_TRAIN_X"
            elif role == "validation":
                phase = "PRETRUTH_VALIDATION_X"
            else:
                raise RuntimeError(f"unexpected donor role: {role}")
        phases.append(phase); strata.append(stratum); role_values.append(role)
    relevant["x_access_phase"] = phases
    relevant["target_stratum"] = strata
    relevant["donor_role"] = role_values
    relevant.insert(1, "panel_id", panel)
    access = relevant[
        [
            "metadata_row_index", "panel_id", "10xrun_id", "donor_id", "donor_role",
            "culture_condition", "guide_id", "guide_type_normalized", "perturbed_gene_id",
            "perturbed_gene_name", "observed_ensembl_core", "expected_ensembl_core",
            "ensembl_core", "target_stratum", "x_access_phase",
        ]
    ].sort_values("metadata_row_index", kind="stable")
    expected_counts = {
        "PRETRUTH_CONTROL_X": 11018,
        "PRETRUTH_TRAIN_X": 1920,
        "PRETRUTH_VALIDATION_X": 960,
        "POSTGATE_TEST_TRUTH_X": 1200,
        "FORBIDDEN_COLUMN_UNSEEN_X": 720,
    }
    if access.x_access_phase.value_counts().to_dict() != expected_counts:
        raise RuntimeError(f"{panel} row phases changed: {access.x_access_phase.value_counts().to_dict()}")
    if access.metadata_row_index.duplicated().any():
        raise RuntimeError(f"{panel} maps one source row twice")
    return access


def write_protocol() -> None:
    protocol = """# E170｜Primary CD4 未读目标多面板精度复现

## 为什么做

E168 在一位完整留出供体的 200 个目标上得到小幅正点估计，但置信区间跨 0，正式判定为 `NO_CONFIRMATION`。E169 进一步发现，同一 state 和同一 seen/unseen 层内，context similarity 与 support 都是常数，SafeConf 实际排序只剩 model disagreement。E170 不修改 SafeConf、不换阈值，也不把 E168 的 200 个已解封目标并入新显著性检验；它用 800 个仍未读取 targeting X 的目标检验该小效应能否在更高目标样本量下复现。

## 一次性冻结的四个面板

- 从 E168 已冻结的 5,510 个 label-free eligible targets 中排除 E168 primary 200，剩余 5,310 个。
- 只按 `E170_FRESH_TARGET_MULTIPANEL_V1` 的 SHA-256 身份哈希选择前 800 个，分为 P01–P04，每个 200 个且互不重叠。
- 每个面板再按面板专属 SHA-256 固定 40 个 `COLUMN_UNSEEN`；其余 160 个为 `DONOR_UNSEEN_ONLY`。
- 选择不使用 targeting X、细胞数、总 counts、扰动效应、DE、guide efficacy、模型误差或 E168 目标表现。
- 四个面板的任务、可读行、模型、seed、gate、统计终点一次性冻结。禁止根据 P01 结果决定是否继续 P02–P04。

## 模型与风险分数

每个面板独立构造 512-gene panel：200 个注册 target genes 加 312 个只由两个 train donors 的 NTC 表达选出的背景基因。scGPT 与 GEARS 各运行 seeds 3407/3408/3409；test query graph 不含 `y`。风险分数、z-score reference、RIAG G1–G5 和 E168 完全相同，不加入 E169 看过 truth 后提出的新特征。

## 主要终点

- 新的 confirmatory population 仅为 E170 的 800 个目标、2,400 个 test tasks。
- 每个 panel×state 单独计算 tie-aware AURC；主要效应是 12 个 panel×state 的等权平均 `AURC_magnitude − AURC_SafeConf`。
- cluster bootstrap 以 target gene 为簇，在每个 panel 内分层重采样 10,000 次，三个 states 同进同出；paired permutation 以 target 为单位交换 SafeConf/magnitude，100,000 次。
- `TARGET_REPLICATION_PASS_NONTRIVIAL` 要求四个 pretruth gate 全部 PASS；全部 800 targets 的 delta>0、95% CI 下界>0、单侧 p<0.05、至少 8/12 panel×state delta>0；随后 seen 640 targets 也必须 CI 下界>0 且 p<0.05。其余结果原样记为 no confirmation 或 partial support，不换面板。

## 解释边界

E170 是同一公开数据、同一 test donor 上的未读目标复现，可以提高目标层面的估计精度，不能提供新的 donor/study 生物学重复。即使通过，也只能支持“在该 donor/study 和指定 scGPT–GEARS 预测器上，相对 magnitude 的目标排序增量”；独立队列和真实湿实验验证仍是更强投稿所需的另一层证据。
"""
    (OUT / "PREREG_ANALYSIS_PLAN.md").write_text(protocol, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E170 先看这个\n\n先读 `PREREG_ANALYSIS_PLAN.md` 与 `reports/E170_METADATA_FREEZE_REPORT.md`。四个面板目前只冻结身份和访问边界；E170 targeting X 尚未读取。\n",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"E170 metadata freeze output already exists and is append-only: {OUT}")
    for directory in (OUT, MANIFESTS, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    byte_attestation, e168_status, source_sha = verify_inputs()
    obs, access_log, schema = load_identity_metadata()
    selected, pool = select_panels()
    roles = pd.read_csv(DONOR_ROLES, keep_default_na=False)
    if roles.donor_role.value_counts().to_dict() != {"train": 6, "validation": 3, "test": 3}:
        raise RuntimeError("frozen donor roles changed")

    all_tasks, all_access = [], []
    roles.to_csv(MANIFESTS / "E170_DONOR_STATE_ROLES.csv", index=False)
    selected.to_csv(MANIFESTS / "E170_ALL_SELECTED_TARGETS.csv", index=False)
    pool[["ensembl_core", "perturbed_gene_name", "eligible_target"]].to_csv(
        TABLES / "E170_UNTOUCHED_IDENTITY_POOL.csv", index=False
    )
    for panel in PANELS:
        panel_dir = MANIFESTS / panel
        panel_dir.mkdir()
        targets = selected.loc[selected.panel_id.eq(panel)].copy()
        tasks = build_tasks(selected, roles, panel)
        row_access = build_row_access(obs, selected, roles, panel)
        targets.to_csv(panel_dir / f"E170_{panel}_SELECTED_TARGETS.csv", index=False)
        tasks.to_csv(panel_dir / f"E170_{panel}_TASK_MANIFEST.csv", index=False)
        row_access.to_csv(panel_dir / f"E170_{panel}_ROW_ACCESS_MANIFEST.csv", index=False)
        all_tasks.append(tasks); all_access.append(row_access)
    combined_tasks = pd.concat(all_tasks, ignore_index=True)
    combined_access = pd.concat(all_access, ignore_index=True)
    combined_tasks.to_csv(MANIFESTS / "E170_ALL_TASKS.csv", index=False)
    combined_access.to_csv(MANIFESTS / "E170_ALL_ROW_ACCESS.csv", index=False)
    pd.DataFrame(access_log).to_csv(TABLES / "E170_HDF5_VALUE_ACCESS_AUDIT.csv", index=False)
    (OUT / "HDF5_VALUE_ACCESS_LOG.json").write_text(
        json.dumps(access_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_lock = {
        "source_path": str(SOURCE),
        "source_bytes": EXPECTED_SOURCE_BYTES,
        "source_full_sha256": source_sha,
        "official_crc64nvme_base64": byte_attestation["computed_crc64nvme_base64"],
        "source_byte_attestation_path": str(BYTE_ATTESTATION),
        "source_byte_attestation_sha256": sha256_file(BYTE_ATTESTATION),
        "e168_source_lock_sha256": sha256_file(E168_SOURCE_LOCK),
        "e168_target_eligibility_audit_sha256": sha256_file(TARGET_AUDIT),
        "e168_primary_target_manifest_sha256": sha256_file(PRIMARY_TARGETS),
        "allowed_hdf5_value_paths": list(ALLOWED_VALUE_PATHS),
        "forbidden_value_roots": list(FORBIDDEN_VALUE_ROOTS),
        "expression_matrix_x_decoded_by_e170_freeze": False,
        "layers_decoded_by_e170_freeze": False,
    }
    (OUT / "SOURCE_LOCK.json").write_text(
        json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_lock = {
        "scgpt_checkpoint_files": {str(path): sha256_file(path) for path in SCGPT_FILES},
        "gears_external_go_file": {"path": str(GEARS_GO), "sha256": sha256_file(GEARS_GO)},
        "model_seeds": list(MODEL_SEEDS),
        "panels": list(PANELS),
        "gene_panel_size_each": 512,
        "registered_targets_each": TARGETS_PER_PANEL,
        "column_unseen_each": COLUMN_UNSEEN_PER_PANEL,
        "panel_runs_must_all_complete_without_optional_stopping": True,
        "pretruth_test_targeting_x_access_count_required": 0,
        "metadata_freeze_script_sha256": sha256_file(Path(__file__).resolve()),
        "deployment_authorized": False,
    }
    (OUT / "MODEL_INPUT_LOCK.json").write_text(
        json.dumps(model_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stat_lock = {
        "confirmatory_population": "E170_only_800_fresh_targets_excludes_E168_primary_200",
        "risk_formula": "negative_[z(context_similarity_max)+z(log1p(support_count))-z(disagreement_rmse)]",
        "primary_comparator": "ensemble_predicted_magnitude_rms_512",
        "loss": "ensemble_effect_rmse_512",
        "ranking_units": "12_panel_by_state_batches_equal_weight",
        "panels": list(PANELS),
        "states": list(STATES),
        "coverage_grid": [round(value, 2) for value in np.arange(0.20, 1.001, 0.05)],
        "tie_primary": "tie_average",
        "operational_tolerance": 1e-6,
        "primary_cluster": "target_gene_three_states_together_stratified_within_panel",
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026071801,
        "permutation_draws": 100000,
        "permutation_seed": 2026071802,
        "permutation_alternative": "SafeConf_delta_greater_than_zero",
        "alpha": 0.05,
        "required_positive_panel_state_units": 8,
        "hierarchical_strata": ["all_800", "seen_640"],
        "e168_primary_tasks_excluded_from_inference": True,
        "optional_stopping_or_panel_dropping_allowed": False,
        "same_donor_study_not_independent_biological_replication": True,
        "deployment_authorized": False,
    }
    (OUT / "STATISTICAL_ANALYSIS_LOCK.json").write_text(
        json.dumps(stat_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_protocol()

    report = f"""# E170 metadata freeze report

- E168 label-free eligible universe: {e168_status['n_eligible_targets']:,} targets.
- Excluded already unsealed E168 primary targets: 200; remaining identity pool: {len(pool):,}.
- Frozen E170 targets: {len(selected):,} in four non-overlapping panels; each panel 160 seen + 40 column-unseen.
- Frozen test tasks: {int(combined_tasks.primary_test_task.sum()):,}; all tasks: {len(combined_tasks):,}.
- HDF5 values decoded by this freeze: {sum(item['n_values'] for item in access_log):,} identity entries across {len(access_log)} allowlisted paths.
- Expression X values decoded: **0**; layers decoded: **0**; error/effect/DE/guide-efficacy values used for selection: **0**.
- Official source SHA-256 was recomputed before local HDF5 metadata was opened: `{source_sha}`.
- All four panels are frozen together. They are fresh target panels in the same donor/study, not four independent donor cohorts.
"""
    (REPORTS / "E170_METADATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path.name != "RUN_STATUS.json"
    )
    artifact_hashes = {
        path.relative_to(OUT).as_posix(): sha256_file(path) for path in artifacts
    }
    status = {
        "schema": "safeconf_e170_metadata_freeze_v1",
        "experiment": "E170_primary_cd4_multipanel_precision",
        "stage": "F1_METADATA_FREEZE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "source_schema": schema,
        "n_eligible_targets": 5510,
        "n_e168_primary_targets_excluded": 200,
        "n_untouched_identity_pool": len(pool),
        "n_panels": len(PANELS),
        "n_selected_targets": len(selected),
        "n_primary_test_tasks": int(combined_tasks.primary_test_task.sum()),
        "all_expression_x_values_read": 0,
        "test_targeting_x_values_read": 0,
        "forbidden_hdf5_values_read": 0,
        "all_panels_frozen_simultaneously": True,
        "e168_targets_excluded": True,
        "same_donor_study_target_replication_only": True,
        "deployment_authorized": False,
        "artifact_sha256": artifact_hashes,
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

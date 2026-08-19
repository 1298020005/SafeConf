#!/usr/bin/env python3
"""E155: freeze two new PRESCRIBE Norman panels without opening expression values.

This program is deliberately limited to AnnData observation/variable metadata,
the scGPT perturbation vocabulary, old split manifests, and source-code hashes.
It never accesses ``adata.X`` and never opens E95/E96 predictions, task metrics,
or error tables.  The new task panels are selected by a fixed SHA256 ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "NormanWeissman2019_filtered.h5ad"
)
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
OUT = ROOT / "docs/实验结果/E155_prescribe_norman_p3p4_contract_20260714"
MANIFESTS = OUT / "manifests"

PAPER_URL = (
    "https://papers.nips.cc/paper_files/paper/2025/file/"
    "d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf"
)
UPSTREAM_COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"
MODEL_SEED = 3407
SELECTION_SEED = "E155|Norman|P3P4|strict-gene-holdout|20260714|v1"
VALIDATION_SEED = "E155|Norman|P3P4|shared-validation|20260714|v1"
N_TEST_PER_PANEL = 24
N_VALIDATION = 20
MIN_TEST_CELLS = 200
N_TASK_BOOTSTRAP = 10_000

OLD_SPLITS = {
    "Norman_P1": ROOT
    / "docs/实验结果/E67_norman_scgpt_formal_fixed_panel_20260711/"
    "tables/E67_FIXED_SPLIT.csv",
    "Norman_P2": ROOT
    / "docs/实验结果/E76b_norman_scgpt_panel2_20260711/"
    "tables/E76b_FIXED_SPLIT.csv",
}

E91_MANIFESTS = {
    "Norman_P1": ROOT
    / "docs/实验结果/E91_prescribe_norman_contract_20260712/"
    "manifests/Norman_P1_set2conditions.pkl",
    "Norman_P2": ROOT
    / "docs/实验结果/E91_prescribe_norman_contract_20260712/"
    "manifests/Norman_P2_set2conditions.pkl",
}

PIPELINE_SCRIPTS = {
    "E91_contract": ROOT / "tools/scripts/run_e91_prescribe_norman_contract.py",
    "E92_assets": ROOT / "tools/scripts/prepare_e92_prescribe_assets.py",
    "E93_preprocess": ROOT / "tools/scripts/run_e93_prescribe_preprocess.py",
    "E94_CPA_reproduction": ROOT / "tools/scripts/run_e94_cpa_pinned_reproduction.py",
    "E95_native_runner": ROOT / "tools/scripts/run_e95_prescribe_norman.py",
    "E96_comparison": ROOT / "tools/scripts/run_e96_prescribe_comparison.py",
}

UPSTREAM_FILES = {
    "PRESCRIBE_Step1_preprocess": PRESCRIBE / "Step1_preprocess.py",
    "PRESCRIBE_Step2_train": PRESCRIBE / "Step2_train.py",
    "PRESCRIBE_Step3_test": PRESCRIBE / "Step3_test.py",
    "PRESCRIBE_data_loader_worktree": PRESCRIBE / "src/data/pertdata.py",
    "PRESCRIBE_GEARS_data_worktree": PRESCRIBE / "gears/pertdata.py",
    "scGPT_perturbation_embedding": PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl",
    "scGPT_gene_embedding": PRESCRIBE / "scLLM_weights/scGPT/gene_emb.pkl",
    "GEARS_gene2go": PRESCRIBE / "data/gene2go_all.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute the metadata-only selection and verify an existing freeze.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\t{value}".encode("utf-8")).hexdigest()


def normalize_condition(value: str) -> str:
    parts = str(value).replace("control", "ctrl").split("_")
    if len(parts) == 1 and parts[0] == "ctrl":
        return "ctrl"
    if len(parts) == 1:
        parts.append("ctrl")
    return "+".join(sorted(parts))


def perturbation_genes(condition: str) -> tuple[str, ...]:
    return tuple(gene for gene in str(condition).split("+") if gene != "ctrl")


def read_old_test_tasks() -> tuple[dict[str, set[str]], list[dict[str, object]]]:
    tests: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    for panel, path in OLD_SPLITS.items():
        frame = pd.read_csv(path)
        required = {"split", "condition"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{panel}: missing split columns")
        if frame["condition"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicate conditions")
        tests[panel] = set(frame.loc[frame["split"].eq("test"), "condition"].astype(str))
        rows.append(
            {
                "source_role": f"old_{panel}_split",
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "content_access": "split labels and task names only",
            }
        )
    if len(tests["Norman_P1"]) != 24 or len(tests["Norman_P2"]) != 24:
        raise RuntimeError("P1/P2 do not each contain 24 frozen test tasks")
    overlap = tests["Norman_P1"] & tests["Norman_P2"]
    if overlap:
        raise RuntimeError(f"P1/P2 test overlap changed: {sorted(overlap)}")
    for panel, path in E91_MANIFESTS.items():
        rows.append(
            {
                "source_role": f"old_E91_{panel}_set2conditions",
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "content_access": "hash only; no prediction or error",
            }
        )
    return tests, rows


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), *args], text=True
    ).strip()


def source_audit_rows(old_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(old_rows)
    selector = Path(__file__).resolve()
    rows.append(
        {
            "source_role": "E155_metadata_only_selector",
            "path": str(selector.relative_to(ROOT)),
            "sha256": sha256_file(selector),
            "bytes": selector.stat().st_size,
            "content_access": "executed contract source",
        }
    )
    rows.append(
        {
            "source_role": "Norman_raw_metadata_and_future_expression_source",
            "path": str(DATA),
            "sha256": sha256_file(DATA),
            "bytes": DATA.stat().st_size,
            "content_access": "freeze reads obs/var metadata only; X forbidden",
        }
    )
    for role, path in PIPELINE_SCRIPTS.items():
        rows.append(
            {
                "source_role": role,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "content_access": "source-code audit only",
            }
        )
    for role, path in UPSTREAM_FILES.items():
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "content_access": "source/vocabulary hash; no test output",
            }
        )
    upstream_bytes = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "show", f"{UPSTREAM_COMMIT}:src/data/pertdata.py"]
    )
    rows.append(
        {
            "source_role": "PRESCRIBE_data_loader_upstream_commit_blob",
            "path": f"git:{UPSTREAM_COMMIT}:src/data/pertdata.py",
            "sha256": sha256_bytes(upstream_bytes),
            "bytes": len(upstream_bytes),
            "content_access": "upstream source blob",
        }
    )
    local_patch = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "diff", "--", "src/data/pertdata.py"]
    )
    rows.append(
        {
            "source_role": "PRESCRIBE_data_loader_local_patch",
            "path": "git-diff:src/data/pertdata.py",
            "sha256": sha256_bytes(local_patch),
            "bytes": len(local_patch),
            "content_access": "local compatibility patch; not upstream code",
        }
    )
    return rows


def pipeline_audit() -> pd.DataFrame:
    notes = {
        "E91_contract": (
            "P1/P2 reused earlier frozen tasks and prohibited output-based selection; "
            "its 24+24 tasks are exclusions for E155."
        ),
        "E92_assets": (
            "Pins upstream commit/assets. E155 records both commit and dirty worktree; "
            "uncommitted loader patch must not be called official code."
        ),
        "E93_preprocess": (
            "Injects only the frozen split path into upstream Step1, but upstream Step1 "
            "selects HVGs/DE metadata before the split. P3/P4 primary analysis therefore "
            "requires a new train-only preprocessing adapter; current CLI supports p1/p2 only."
        ),
        "E94_CPA_reproduction": (
            "CPA reproduction is not used by E155 and contributes no task, prediction, "
            "score, or endpoint to the PRESCRIBE extension."
        ),
        "E95_native_runner": (
            "Native model/loss and combined confidence 2*epistemic+aleatoric are retained. "
            "Current task summary is RMSE-oriented and must be extended prospectively."
        ),
        "E96_comparison": (
            "Its RMSE-first evaluation is not the paper-default calibration endpoint. "
            "E155 freezes effect-vector Pearson accuracy before P3/P4 training."
        ),
    }
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "script": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "E155_decision": notes[stage],
            }
            for stage, path in PIPELINE_SCRIPTS.items()
        ]
    )


def select_panels() -> tuple[
    pd.DataFrame,
    dict[str, dict[str, list[str]]],
    dict[str, object],
    list[dict[str, object]],
    pd.DataFrame,
]:
    old_tests, old_rows = read_old_test_tasks()
    old_test_union = set().union(*old_tests.values())

    with (PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl").open("rb") as handle:
        embedding = pickle.load(handle)
    embedding_upper = {str(gene).upper() for gene in embedding}

    # backed='r' leaves X on disk.  Only obs and var_names are inspected below.
    adata = sc.read_h5ad(DATA, backed="r")
    try:
        raw_conditions = adata.obs["perturbation"].astype(str).map(normalize_condition)
        counts = raw_conditions.value_counts().sort_index()
        var_names = {str(gene) for gene in adata.var_names}
        shape = [int(adata.n_obs), int(adata.n_vars)]
    finally:
        adata.file.close()

    rows: list[dict[str, object]] = []
    for condition, n_cells in counts.items():
        genes = perturbation_genes(condition)
        in_embedding = all(gene.upper() in embedding_upper for gene in genes)
        in_expression_metadata = all(gene in var_names for gene in genes)
        is_single = len(genes) == 1 and condition.endswith("+ctrl")
        eligible = (
            is_single
            and int(n_cells) >= MIN_TEST_CELLS
            and in_embedding
            and in_expression_metadata
            and condition not in old_test_union
        )
        rows.append(
            {
                "condition": condition,
                "perturbation_genes": ";".join(genes),
                "n_perturbation_genes": len(genes),
                "n_cells_obs_metadata": int(n_cells),
                "is_single_gene_plus_ctrl": is_single,
                "all_perturbation_genes_in_scgpt_embedding": in_embedding,
                "all_perturbation_genes_in_expression_var_metadata": in_expression_metadata,
                "was_P1_test": condition in old_tests["Norman_P1"],
                "was_P2_test": condition in old_tests["Norman_P2"],
                "eligible_new_test_before_sha": eligible,
                "selection_sha256": sha_rank(SELECTION_SEED, condition) if eligible else "",
            }
        )
    audit = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    candidates = audit.loc[audit["eligible_new_test_before_sha"]].copy()
    candidates = candidates.sort_values(["selection_sha256", "condition"]).reset_index(drop=True)
    required = 2 * N_TEST_PER_PANEL
    if len(candidates) < required:
        raise RuntimeError(
            f"Only {len(candidates)} eligible unused tasks; need {required} for 24+24. "
            "No panel was frozen."
        )
    candidates["selection_rank"] = range(1, len(candidates) + 1)
    p3_test = candidates.iloc[:N_TEST_PER_PANEL]["condition"].tolist()
    p4_test = candidates.iloc[N_TEST_PER_PANEL:required]["condition"].tolist()
    reserve = candidates.iloc[required:]["condition"].tolist()
    if set(p3_test) & set(p4_test):
        raise RuntimeError("SHA-selected P3/P4 overlap")

    test_union = set(p3_test) | set(p4_test)
    heldout_genes = set().union(*(set(perturbation_genes(condition)) for condition in test_union))
    valid_conditions = {
        row.condition
        for row in audit.itertuples(index=False)
        if row.all_perturbation_genes_in_scgpt_embedding
        and row.all_perturbation_genes_in_expression_var_metadata
    }
    strict_development = sorted(
        condition
        for condition in valid_conditions
        if not (set(perturbation_genes(condition)) & heldout_genes)
    )
    if "ctrl" not in strict_development:
        raise RuntimeError("Control condition missing after strict gene holdout")
    val_candidates = [condition for condition in strict_development if condition != "ctrl"]
    val = sorted(val_candidates, key=lambda value: (sha_rank(VALIDATION_SEED, value), value))[
        :N_VALIDATION
    ]
    if len(val) != N_VALIDATION:
        raise RuntimeError("Insufficient strict-development conditions for validation")
    train = sorted(set(strict_development) - set(val))
    if "ctrl" not in train:
        raise RuntimeError("Control condition must be in training")

    panels = {
        "Norman_P3": {"train": train, "val": sorted(val), "test": sorted(p3_test)},
        "Norman_P4": {"train": train, "val": sorted(val), "test": sorted(p4_test)},
    }
    for panel, split in panels.items():
        split_sets = {key: set(value) for key, value in split.items()}
        if any(split_sets[a] & split_sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise RuntimeError(f"{panel}: split overlap")
        development_genes = set().union(
            *(set(perturbation_genes(c)) for c in split["train"] + split["val"])
        )
        if development_genes & heldout_genes:
            raise RuntimeError(f"{panel}: strict held-out gene leakage")
        if len(split["test"]) != N_TEST_PER_PANEL:
            raise RuntimeError(f"{panel}: wrong test count")

    assignments = {
        panel: {
            condition: role
            for role, values in split.items()
            for condition in values
        }
        for panel, split in panels.items()
    }
    audit["contains_any_P3P4_heldout_gene"] = audit["condition"].map(
        lambda value: bool(set(perturbation_genes(value)) & heldout_genes)
    )
    audit["P3_split"] = audit["condition"].map(assignments["Norman_P3"]).fillna("excluded")
    audit["P4_split"] = audit["condition"].map(assignments["Norman_P4"]).fillna("excluded")
    candidate_panel = {
        **{condition: "Norman_P3_test" for condition in p3_test},
        **{condition: "Norman_P4_test" for condition in p4_test},
        **{condition: "reserve_not_tested" for condition in reserve},
    }
    audit["new_test_SHA_assignment"] = audit["condition"].map(candidate_panel).fillna("")
    candidates["assignment"] = candidates["condition"].map(candidate_panel)

    metadata = {
        "dataset_shape": shape,
        "n_raw_conditions": int(len(audit)),
        "n_embedding_compatible_conditions": int(
            audit["all_perturbation_genes_in_scgpt_embedding"].sum()
        ),
        "n_unused_eligible_single_gene_tasks": int(len(candidates)),
        "n_P3_test": len(p3_test),
        "n_P4_test": len(p4_test),
        "n_reserve": len(reserve),
        "reserve_tasks": reserve,
        "n_union_heldout_genes": len(heldout_genes),
        "n_shared_train": len(train),
        "n_shared_val": len(val),
        "n_valid_conditions_excluded_by_strict_gene_holdout": int(
            len(valid_conditions - set(strict_development))
        ),
        "P3_P4_test_task_overlap": len(set(p3_test) & set(p4_test)),
        "P3_P4_test_gene_overlap": len(
            {gene for c in p3_test for gene in perturbation_genes(c)}
            & {gene for c in p4_test for gene in perturbation_genes(c)}
        ),
        "development_to_test_gene_overlap": 0,
    }
    return audit, panels, metadata, old_rows, candidates


def write_contract(metadata: dict[str, object], sources: pd.DataFrame) -> str:
    source_digest = sha256_bytes(sources.to_csv(index=False).encode("utf-8"))
    return f"""# E155｜PRESCRIBE Norman P3/P4 前瞻任务合同

冻结日期：2026-07-14。当前阶段只完成任务、切分和评价规则冻结；没有开始预处理、训练或测试。

## 1. 数据访问边界

冻结脚本只读取 Norman AnnData 的 `obs`、`var_names` 和文件形状，未访问 `X`。它读取 scGPT 扰动词表来排除官方模型不能编码的扰动，并读取 P1/P2 的旧切分任务名。以下内容在冻结阶段禁止打开：E95 的逐细胞预测、任务误差和 checkpoint；E96/E145 的逐任务结果表；P3/P4 的任何表达真值、预测或误差。

PRESCRIBE 论文：<{PAPER_URL}>。上游 Git commit 固定为 `{UPSTREAM_COMMIT}`。源码/数据哈希表为 `manifests/E155_SOURCE_HASHES.csv`，其 CSV 内容 SHA256 为 `{source_digest}`。工作树含为 Norman frozen split 添加的本地兼容补丁，合同将它与上游源码分别记录，不能把补丁称为官方实现。

## 2. 任务资格与固定选择

测试候选必须同时满足：Norman 单基因 `gene+ctrl`；`obs` 中至少 {MIN_TEST_CELLS} 个细胞；扰动基因存在于表达 `var_names` 和 PRESCRIBE 使用的 scGPT embedding；没有在 P1/P2 作为测试任务出现。合格候选共 {metadata['n_unused_eligible_single_gene_tasks']} 个。

候选按 `SHA256("{SELECTION_SEED}" + TAB + condition)` 升序排列。前 24 个固定为 P3，后 24 个固定为 P4，余下 {metadata['n_reserve']} 个只作预先登记的 reserve，不因后续结果替换任务。P3/P4 任务重叠为 {metadata['P3_P4_test_task_overlap']}，基因重叠为 {metadata['P3_P4_test_gene_overlap']}。

## 3. 严格基因留出切分

P3/P4 的 48 个测试基因组成共同 held-out gene set。任何包含其中一个基因的单扰动或组合扰动都不能进入任一面板的 train/val。两面板共享同一开发集：train {metadata['n_shared_train']} 个条件（含 control），val {metadata['n_shared_val']} 个条件；每个面板 test 24 个任务，另一面板的 24 个测试任务在本面板中保持 excluded。验证集按 `SHA256("{VALIDATION_SEED}" + TAB + condition)` 固定，control 强制留在 train。

该规则比 P1/P2 更严格：不仅测试任务本身不能进训练，含同一测试基因的组合扰动也被排除。这样 P3/P4 测试回答的是未见扰动基因的泛化，不是已见基因的新组合。

## 4. 固定模型与随机性

- PRESCRIBE native architecture/loss；上游 commit `{UPSTREAM_COMMIT}`，本地 loader patch 必须按哈希原样保留并单列披露。
- P3/P4 主分析不原样复用 E93：上游 Step1 在切分前用全数据选 HVG 和 DE metadata，会让测试表达参与特征预处理。E156 必须只用 train（含 control）拟合 HVG、PCA 和训练期 E-distance 排序标签；val/test 只做固定变换，测试表达只能在模型和分数全部锁定后用于评价。上游原样的 transductive preprocessing 若补跑，只能列为敏感性分析。
- 模型随机种子：{MODEL_SEED}；预处理随机种子同为 {MODEL_SEED}。
- formal：50 epochs，warmup 5 epochs，batch size 4096，deterministic training，与 E95 P1/P2 一致。
- P3/P4 不因 smoke 或 formal 中间结果修改任务、切分、种子、epoch、分数方向和评价终点。
- P1/P2 与 P3/P4 的预处理和基因留出强度不同，四个面板不能伪装成同一设计直接合并；P3/P4 先按本合同独立给出结果，再与 P1/P2 并列讨论。

## 5. 固定终点

对每个任务先求预测与真实平均表达，并减去同一个 control 平均表达，形成预测和真实 log-normalized 扰动效应向量。

1. **主要准确度终点**：`pearson_effect_accuracy`，即预测效应与真实效应向量的 Pearson 相关，越高越好；这是 PRESCRIBE 论文默认的 perturbation accuracy 口径。
2. **方向确认终点**：`frac_correct_direction_all`，逐基因预测效应与真实效应同号的比例，越高越好；`frac_correct_direction_top20_de` 为补充。
3. **误差补充终点**：`rmse_effect_error`，两个任务平均效应向量的 RMSE，越低越好。
4. **置信度**：任务内取 `epistemic_confidence` 和 `aleatoric_confidence` 均值；官方组合固定为 `combined_confidence = 2 * epistemic + aleatoric`。不拟合新权重。
5. **部署可见基线**：`predicted_magnitude_rms`。真实 magnitude 只可作诊断，不能参与风险排序。

主统计量是在 P3、P4 内分别计算 `combined_confidence` 与 Pearson accuracy 的 Spearman 相关，再取两面板等权宏平均；预期方向为正。外层 Pearson 关联作为敏感性分析。方向准确度预期正相关，RMSE 预期负相关。每个面板按任务做 {N_TASK_BOOTSTRAP:,} 次 bootstrap；宏平均在两个面板内分别重采样后等权合并。相同 bootstrap 索引用于 combined confidence 与 magnitude 的配对 Δρ。固定 coverage 为 90%、95% 和 50%–100%（每 5% 一档）。

## 6. 通过与边界

- **论文口径信号确认**：P3、P4 的 combined→Pearson Spearman 都大于 0，且双面板宏平均任务-bootstrap 95% CI 下界大于 0。
- **增量优势确认**：在上一条通过的基础上，combined 相对 predicted magnitude 的宏平均配对 Δρ 95% CI 下界大于 0。
- **方向生物学确认**：P3、P4 的 combined→direction accuracy 都大于 0，且宏平均 CI 下界大于 0。
- 第一条通过而增量优势不通过时，只能称为有可靠性排序信号，不能称为超越 magnitude 或具有独立增益。
- 第一条不通过时，P3/P4 复现失败；不得用 RMSE、某个单面板、reserve 替换或事后更改 coverage 来挽救主要结论。
- 任一测试任务缺失、测试基因进入 train/val、输出非有限或 checkpoint/任务不匹配时，先判合同失败；只能修复实现并保留审计记录，不能改任务。

## 7. 后续运行命令与资源预算

本次实际执行：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e155_prescribe_norman_p3p4_contract.py
```

下阶段需先新增并审查 p3/p4 专用 adapter；当前 E93/E95 CLI 只接受 p1/p2，下面是冻结的拟运行接口，不表示脚本已经存在：

```bash
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e156_prescribe_norman_p3p4_preprocess.py --panel p3
/home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e156_prescribe_norman_p3p4_preprocess.py --panel p4
CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e157_prescribe_norman_p3p4.py --panel p3 --mode formal --seed {MODEL_SEED}
CUDA_VISIBLE_DEVICES=1 /home/yyf/.conda/envs/prescribe_env/bin/python tools/scripts/run_e157_prescribe_norman_p3p4.py --panel p4 --mode formal --seed {MODEL_SEED}
```

环境固定为 `/home/yyf/.conda/envs/prescribe_env`（Python 3.9.25）。服务器有两块 Quadro RTX 6000 24GB；预处理主要使用 CPU，训练每面板使用一块 GPU。E95 P1/P2 的 50-epoch 训练实测约 0.55 和 1.72 小时；E155 开发集更小，训练预算按每面板 0.5–2 小时、双 GPU 并行 0.5–2 小时估计。上游 10,000-run E-test 预处理耗时未被 E93 单独计时，完整预处理和训练保守预留 6–12 小时。该时间是资源预算，不是完成记录。
"""


def write_or_verify(path: Path, payload: bytes, verify: bool) -> None:
    if verify:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact changed: {path}")
    else:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    current_commit = git_text("rev-parse", "HEAD")
    if current_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"PRESCRIBE commit changed: {current_commit}")

    audit, panels, metadata, old_rows, candidates = select_panels()
    source_frame = pd.DataFrame(source_audit_rows(old_rows))
    scripts = pipeline_audit()

    artifacts: dict[Path, bytes] = {}
    artifacts[MANIFESTS / "E155_CONDITION_AUDIT.csv"] = audit.to_csv(index=False).encode("utf-8")
    artifacts[MANIFESTS / "E155_NEW_TEST_CANDIDATES_SHA_ORDER.csv"] = candidates.to_csv(index=False).encode("utf-8")
    artifacts[MANIFESTS / "E155_SOURCE_HASHES.csv"] = source_frame.to_csv(index=False).encode("utf-8")
    artifacts[MANIFESTS / "E155_E91_E96_SCRIPT_AUDIT.csv"] = scripts.to_csv(index=False).encode("utf-8")
    for panel, split in panels.items():
        rows = [
            {"panel": panel, "split": role, "condition": condition}
            for role in ["train", "val", "test"]
            for condition in split[role]
        ]
        artifacts[MANIFESTS / f"{panel}_SPLIT.csv"] = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        artifacts[MANIFESTS / f"{panel}_set2conditions.pkl"] = pickle.dumps(split, protocol=4)

    contract = write_contract(metadata, source_frame)
    artifacts[OUT / "ANALYSIS_CONTRACT.md"] = contract.encode("utf-8")

    artifact_hashes = {
        str(path.relative_to(OUT)): sha256_bytes(payload)
        for path, payload in artifacts.items()
    }
    external_status = git_text("status", "--porcelain=v1")
    status = {
        "experiment": "E155_prescribe_norman_p3p4_contract",
        "phase": "contract_frozen_predictions_and_errors_unseen",
        "frozen_at": "2026-07-14",
        "executed_at": datetime.now().isoformat(timespec="seconds") if not args.verify else "VERIFICATION_ONLY",
        "metadata_only": True,
        "adata_X_accessed": False,
        "E95_E96_prediction_or_error_opened": False,
        "test_task_selection_method": "fixed SHA256 ordering over metadata-qualified unused tasks",
        "selection_seed": SELECTION_SEED,
        "validation_seed": VALIDATION_SEED,
        "model_and_preprocess_seed": MODEL_SEED,
        "minimum_test_cells": MIN_TEST_CELLS,
        "task_bootstrap_replicates": N_TASK_BOOTSTRAP,
        "primary_endpoint": "combined confidence vs task Pearson effect accuracy; panel Spearman and two-panel macro task-bootstrap CI",
        "secondary_direction_endpoint": "fraction correct effect direction across genes",
        "supplementary_error_endpoint": "task mean-effect RMSE",
        "baseline": "predicted effect RMS magnitude",
        "primary_preprocessing": "train-only HVG/PCA/training E-distance fit; val/test transform-only and truth sealed until evaluation",
        "upstream_transductive_preprocessing_role": "optional sensitivity analysis only",
        "prescribe_commit": current_commit,
        "prescribe_worktree_clean": external_status == "",
        "prescribe_worktree_status": external_status.splitlines(),
        **metadata,
        "artifact_sha256": artifact_hashes,
    }
    status_bytes = (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    if args.verify:
        for path, payload in artifacts.items():
            write_or_verify(path, payload, verify=True)
        existing = json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))
        for key in [
            "selection_seed", "validation_seed", "model_and_preprocess_seed",
            "n_unused_eligible_single_gene_tasks", "n_P3_test", "n_P4_test",
            "n_shared_train", "n_shared_val", "P3_P4_test_task_overlap",
            "P3_P4_test_gene_overlap", "development_to_test_gene_overlap",
            "artifact_sha256",
        ]:
            if existing.get(key) != status.get(key):
                raise RuntimeError(f"RUN_STATUS verification failed for {key}")
        print(json.dumps({"phase": "verification_passed", **metadata}, ensure_ascii=False, indent=2))
        return

    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {OUT}")
    for path, payload in artifacts.items():
        write_or_verify(path, payload, verify=False)
    (OUT / "RUN_STATUS.json").write_bytes(status_bytes)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

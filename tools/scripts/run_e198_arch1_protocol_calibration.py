#!/usr/bin/env python3
"""E198: preregistered scPertEval calibration on the external arch1 dataset.

``prepare`` treats the H5AD as opaque bytes: it records size/SHA-256 and
validates the frozen scPertEval implementation without loading AnnData.
``formal`` is allowed only after the runner and prepare artefacts are committed
and the current branch tip is identical on GitHub and Gitee.  It then opens the
dataset for the first time in this experiment and calibrates a fixed metric
panel against scPertEval's empirical positive and negative controls.

This is protocol calibration, not model validation and not a blind predictor
claim.  No model prediction is used anywhere in E198.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E198_arch1_protocol_calibration_20260801"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
FREEZE = OUT / "ANALYSIS_FREEZE.md"
STATUS = OUT / "E198_STATUS.json"
OUTPUT_HASH_INDEX = TABLES / "E198_OUTPUT_HASHES.csv"
AUDIT_STATUS = OUT / "E198_AUDIT_STATUS.json"

DATASET = Path(
    "/home/yyf/data/scperteval_official_20260730/"
    "arch1_processed_complete.h5ad"
)
EXPECTED_DATASET_BYTES = 4_377_281_643
EXPECTED_PERTURBATIONS = 150

SCPERTEVAL_REPO = Path("/home/yyf/archive/external/scPertEval")
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
SCPERTEVAL_INPUTS = (
    "src/scperteval/api.py",
    "src/scperteval/calibrators.py",
    "src/scperteval/context.py",
    "src/scperteval/dataset.py",
    "src/scperteval/io.py",
    "src/scperteval/protocols/metrics.py",
    "src/scperteval/protocols/table.py",
    "src/scperteval/runner.py",
)

SEED = 20_260_801
SUBSAMPLE = 2_048
MIN_CELLS = 30
WORKERS = 8
DE_METHOD = "t-test"
N_BOOTSTRAP = 5_000
WILSON_Z = 1.959963984540054
NUMERIC_TOL = 1e-12

# The panel spans absolute, shape, retrieval, population and DE endpoints.
# Sinkhorn is deliberately absent: it requires an optional dependency and is
# not needed to answer whether the five endpoint families are identifiable.
PROTOCOL_SPECS = (
    "pearson",
    "pearson_ctrl",
    "pearson_pert",
    "mse",
    "wmse_exp2",
    "rank",
    "transpose_rank",
    "energy_distance_pca_k=50",
    "unbiased_mmd_median_pca_k=50",
    "de_auprc",
    "de_auroc",
    "de_overlap_k=50",
)

AXIS_PRIORITY = {
    "absolute": ("mse", "wmse_exp2"),
    "direction": ("pearson_pert", "pearson_ctrl", "pearson"),
    "retrieval": ("rank", "transpose_rank"),
    "population": (
        "energy_distance_pca_k=50",
        "unbiased_mmd_median_pca_k=50",
    ),
    "de": ("de_auprc", "de_overlap_k=50", "de_auroc"),
}
PROTOCOL_AXIS = {
    protocol: axis
    for axis, protocols in AXIS_PRIORITY.items()
    for protocol in protocols
}

FORMAL_PAYLOADS = (
    TABLES / "E198_PROTOCOL_CALIBRATION.csv",
    TABLES / "E198_PROTOCOL_SUMMARY.csv",
    TABLES / "E198_ENDPOINT_SELECTION.csv",
    TABLES / "E198_FORMAL_GATES.csv",
    TABLES / "E198_RUNTIME.csv",
    FIGURES / "E198_protocol_calibration.png",
    FIGURES / "E198_protocol_calibration.pdf",
    REPORTS / "E198_REPORT.md",
    REPORTS / "E198_INTERPRETATION.md",
    OUT / "README_先看这个.md",
)


class ContractFailure(RuntimeError):
    """Fail-closed E198 integrity error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    for base, prefix in (
        (ROOT.resolve(), ""),
        (Path("/home/yyf/data").resolve(), "DATA/"),
        (Path("/home/yyf/archive/external").resolve(), "EXTERNAL/"),
    ):
        try:
            return prefix + resolved.relative_to(base).as_posix()
        except ValueError:
            pass
    return resolved.name


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def git_clean_for_path(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel],
            check=False,
        ).returncode
        == 0
        and subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--cached",
                "--quiet",
                "HEAD",
                "--",
                rel,
            ],
            check=False,
        ).returncode
        == 0
    )


def tracked_in_head(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{rel}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def remote_tip(remote: str, branch: str) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-remote", remote, f"refs/heads/{branch}"],
        text=True,
    ).strip()
    if not output:
        raise ContractFailure(f"missing remote branch: {remote}/{branch}")
    return output.split()[0]


def add_hash(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise ContractFailure(f"missing input: {path}")
    rows.append(
        {
            "role": role,
            "path": logical_path(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    )


def gate_row(
    rows: list[dict[str, Any]],
    phase: str,
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
    detail: str = "",
) -> None:
    rows.append(
        {
            "phase": phase,
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "detail": detail,
        }
    )
    if not passed:
        raise ContractFailure(f"{phase} gate failed: {check}")


def write_status(status: str, **fields: Any) -> None:
    write_json_atomic(
        STATUS,
        {
            "experiment": "E198_arch1_protocol_calibration",
            "analysis_class": "EXTERNAL_PROTOCOL_CALIBRATION",
            "generated_at": now(),
            "status": status,
            **fields,
        },
    )


def runtime_environment() -> pd.DataFrame:
    rows = [
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
        {"component": "cpu_count", "version": str(os.cpu_count() or "unknown")},
    ]
    for distribution in (
        "anndata",
        "h5py",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "scperteval",
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        rows.append({"component": distribution, "version": version})
    return pd.DataFrame(rows)


def import_frozen_scperteval():
    try:
        import scperteval as sp
    except ImportError as exc:
        raise ContractFailure(
            "run E198 in /home/yyf/.venvs/scperteval_env"
        ) from exc
    module_path = Path(sp.__file__).resolve()
    try:
        module_path.relative_to(SCPERTEVAL_REPO.resolve())
    except ValueError as exc:
        raise ContractFailure(
            f"scPertEval imported outside frozen checkout: {module_path}"
        ) from exc
    return sp


def resolve_protocol_metadata() -> pd.DataFrame:
    import_frozen_scperteval()
    from scperteval.protocols.resolve import resolve_protocols

    rows = []
    for spec in PROTOCOL_SPECS:
        resolved = resolve_protocols([spec])
        if len(resolved) != 1:
            raise ContractFailure(f"protocol does not resolve uniquely: {spec}")
        p = resolved[0]
        rows.append(
            {
                "protocol": p.name,
                "axis": PROTOCOL_AXIS[p.name],
                "representation": p.representation,
                "space": p.space,
                "scope": p.scope,
                "better": p.better,
                "perfect": float(p.perfect),
                "default_positive": p.default_positive or "generic",
                "default_negative": p.default_negative or "generic",
                "requires_extra": p.requires_extra or "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.protocol.tolist() != list(PROTOCOL_SPECS):
        raise ContractFailure("resolved protocol names changed")
    return frame


def independent_drf(
    positive: np.ndarray,
    negative: np.ndarray,
    better: str,
    perfect: float,
) -> np.ndarray:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    out = np.full(len(positive), np.nan, dtype=float)
    finite = np.isfinite(positive) & np.isfinite(negative)
    beyond = negative > perfect if better == "higher" else negative < perfect
    valid = finite & ~beyond
    if better == "higher":
        numerator = positive - negative
        denominator = perfect - negative
    else:
        numerator = negative - positive
        denominator = negative - perfect
    out[valid] = np.clip(
        numerator[valid] / (denominator[valid] + 1e-6), -1.0, 1.0
    )
    return out


def independent_bds(
    positive: np.ndarray, negative: np.ndarray, better: str
) -> np.ndarray:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if better == "higher":
        return (positive > negative).astype(float)
    return (positive < negative).astype(float)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    z2 = WILSON_Z**2
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = (
        WILSON_Z
        * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return center - half, center + half


def bootstrap_median(values: np.ndarray, protocol: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    salt = int.from_bytes(
        hashlib.sha256(protocol.encode()).digest()[:4], "little"
    )
    rng = np.random.default_rng((SEED + salt) % (2**32))
    draws = np.median(
        rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True), axis=1
    )
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def synthetic_smoke(return_payload: bool = False) -> dict[str, Any]:
    """Exercise every protocol family without touching the arch1 H5AD."""
    sp = import_frozen_scperteval()
    try:
        import anndata as ad
        from scipy import sparse
    except ImportError as exc:
        raise ContractFailure("synthetic smoke dependencies missing") from exc

    rng = np.random.default_rng(198)
    genes = [f"g{i}" for i in range(64)]
    labels: list[str] = ["control"] * 80
    chunks = [rng.normal(0.0, 0.3, size=(80, len(genes)))]
    for index in range(6):
        effect = np.zeros(len(genes))
        effect[index * 6 : index * 6 + 8] = 0.8 + index * 0.08
        chunks.append(rng.normal(effect, 0.3, size=(48, len(genes))))
        labels.extend([f"p{index}"] * 48)
    matrix = np.vstack(chunks).astype(np.float32)
    adata = ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=pd.DataFrame(
            {"perturbation": labels},
            index=[f"c{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    )
    prepared = sp.prepare(
        adata,
        list(PROTOCOL_SPECS),
        subsample=64,
        seed=SEED,
        min_cells=20,
        workers=1,
        name="E198_synthetic_no_arch1",
    )
    protocol_metadata = resolve_protocol_metadata().set_index("protocol")
    checked = []
    for protocol in PROTOCOL_SPECS:
        result = sp.calibrate(
            prepared, protocol, de_method=DE_METHOD, calibrator="drf"
        )
        frame = result.per_perturbation
        if len(frame) != 6 or frame.protocol.astype(str).nunique() != 1:
            raise ContractFailure(f"synthetic protocol malformed: {protocol}")
        metadata = protocol_metadata.loc[protocol]
        observed = frame.drf.to_numpy(float)
        expected = independent_drf(
            frame.raw_positive.to_numpy(float),
            frame.raw_negative.to_numpy(float),
            str(metadata.better),
            float(metadata.perfect),
        )
        if not np.array_equal(np.isfinite(observed), np.isfinite(expected)):
            raise ContractFailure(f"synthetic DRF NA mask mismatch: {protocol}")
        finite = np.isfinite(observed)
        delta = (
            float(np.max(np.abs(observed[finite] - expected[finite])))
            if finite.any()
            else 0.0
        )
        if delta > NUMERIC_TOL:
            raise ContractFailure(
                f"synthetic DRF formula mismatch: {protocol} {delta}"
            )
        checked.append(protocol)
    payload = {
        "status": "PASS",
        "uses_only_synthetic_in_memory_data": True,
        "arch1_loaded": False,
        "protocols_checked": checked,
    }
    if not return_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def freeze_markdown(data_hash: str, script_hash: str) -> str:
    protocols = "\n".join(f"- `{p}`" for p in PROTOCOL_SPECS)
    priorities = "\n".join(
        f"- `{axis}`: " + " → ".join(f"`{p}`" for p in items)
        for axis, items in AXIS_PRIORITY.items()
    )
    return f"""# E198｜arch1 外部评价协议校准冻结

冻结时间：{now()}

分析性质：`EXTERNAL_PROTOCOL_CALIBRATION`。本实验不训练模型、不读取预测文件、
不验证 SafeConf，也不把 `arch1` 写成盲法 leaderboard 测试。目标是在后续 E199
打开模型结果之前，先确定哪些评价协议能区分技术重复与无信息参考。

## 数据合同

- 数据集：scPertEval 官方 `arch1` processed training split；
- 公开元数据：H1 hESC、150 个 CRISPRi 扰动；
- 文件大小：`{EXPECTED_DATASET_BYTES}` bytes；
- 文件 SHA-256：`{data_hash}`；
- `prepare` 只做字节哈希，没有调用 AnnData/HDF5 解析；
- E198 formal 首次按数据对象打开文件；
- 该数据只有一个 context，不能作为 unseen-row 或 cross-context 证据。

## 实现与资源合同

- scPertEval commit：`{SCPERTEVAL_COMMIT}`；
- runner SHA-256：`{script_hash}`；
- seed：`{SEED}`；min cells：`{MIN_CELLS}`；
- all-perturbed/control subsample：`{SUBSAMPLE}`；workers：`{WORKERS}`；
- DE：`{DE_METHOD}`；PCA：50 components；bootstrap：`{N_BOOTSTRAP}`；
- 不运行 Sinkhorn，不安装或临时补入可选依赖；
- formal 前 runner、冻结文件、输入哈希和 prepare gates 必须已提交，且 GitHub、
  Gitee 与本地 HEAD 三者完全一致。

## 固定协议

{protocols}

## 评价门槛

每个协议保存逐扰动 `raw_positive`、`raw_negative` 和官方 DRF，并独立复算 DRF 与
BDS。完整性失败会中止实验；指标表现差只形成科学负结果，不中止或删表。

- `REJECT`：BDS < 0.5、DRF 中位数 ≤ 0，或有限 DRF 比例 < 90%；
- `SECONDARY_ONLY`：达到 BDS ≥ 0.5 且 DRF 中位数 > 0，但任一主要不确定性门槛
  未过；
- `PRIMARY_ELIGIBLE`：BDS 的双侧 95% Wilson 下界 > 0.5，且 5,000 次按扰动
  重抽样的 DRF 中位数 95% 下界 > 0，有限 DRF 比例 ≥ 90%。

官方文档只明确规定 BDS < 0.5 的协议不可信；Wilson 与 bootstrap 是本实验事前
增加的主要端点门槛，不冒充 scPertEval 官方阈值。

## 后续端点选择优先级

每个生物学轴最多选择一个 `PRIMARY_ELIGIBLE` 协议，严格按以下顺序取第一个；
没有通过者就留空，不能按 E199 模型表现改选：

{priorities}

## 失败与停止规则

1. 数据大小/SHA、scPertEval commit/source hash 或冻结 Git 状态不符，拒绝 formal；
2. 任一协议不是 150 个扰动、raw controls 出现正负无穷、官方与独立 NA mask
   不同，或 DRF 公式差值超过 `{NUMERIC_TOL}`，拒绝发布完整状态；官方定义允许的
   `NaN` 保留，并通过有限 DRF 比例进入科学裁决；
3. 不因某个协议 BDS/DRF 低而换参数、换 PCA、换 DE 或删除该协议；
4. 不把 E198 写成预测模型性能、SafeConf 外部确认或跨 context 结果；
5. 正式输出存在或已标记 COMPLETE 时拒绝覆盖。
"""


def prepare() -> None:
    if any(path.exists() for path in FORMAL_PAYLOADS) or OUTPUT_HASH_INDEX.exists():
        raise ContractFailure("formal E198 payload already exists; prepare overwrite refused")
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    gates: list[dict[str, Any]] = []
    gate_row(gates, "prepare", "dataset_exists", DATASET.is_file(), True, DATASET.is_file())
    observed_bytes = DATASET.stat().st_size if DATASET.is_file() else -1
    gate_row(
        gates,
        "prepare",
        "dataset_bytes",
        observed_bytes,
        EXPECTED_DATASET_BYTES,
        observed_bytes == EXPECTED_DATASET_BYTES,
    )
    gate_row(
        gates,
        "prepare",
        "scperteval_commit",
        git_output(SCPERTEVAL_REPO, "rev-parse", "HEAD"),
        SCPERTEVAL_COMMIT,
        git_output(SCPERTEVAL_REPO, "rev-parse", "HEAD") == SCPERTEVAL_COMMIT,
    )
    external_dirty = git_output(SCPERTEVAL_REPO, "status", "--porcelain")
    gate_row(
        gates,
        "prepare",
        "scperteval_worktree_clean",
        external_dirty or "clean",
        "clean",
        external_dirty == "",
    )
    metadata = resolve_protocol_metadata()
    gate_row(
        gates,
        "prepare",
        "protocol_count",
        len(metadata),
        len(PROTOCOL_SPECS),
        len(metadata) == len(PROTOCOL_SPECS),
    )
    gate_row(
        gates,
        "prepare",
        "optional_extras_absent",
        ";".join(metadata.requires_extra.astype(str).unique()),
        "empty",
        (metadata.requires_extra.astype(str) == "").all(),
    )
    smoke = synthetic_smoke(return_payload=True)
    gate_row(
        gates,
        "prepare",
        "synthetic_all_protocols",
        len(smoke["protocols_checked"]),
        len(PROTOCOL_SPECS),
        smoke["status"] == "PASS"
        and len(smoke["protocols_checked"]) == len(PROTOCOL_SPECS),
        "arch1 not loaded",
    )

    data_hash = sha256_file(DATASET)
    hashes: list[dict[str, Any]] = []
    add_hash(hashes, "E198:runner", SCRIPT)
    hashes.append(
        {
            "role": "E198:arch1_opaque_bytes",
            "path": logical_path(DATASET),
            "bytes": int(DATASET.stat().st_size),
            "sha256": data_hash,
        }
    )
    for relative in SCPERTEVAL_INPUTS:
        add_hash(
            hashes,
            f"scPertEval:{relative}",
            SCPERTEVAL_REPO / relative,
        )

    script_hash = sha256_file(SCRIPT)
    write_text_atomic(FREEZE, freeze_markdown(data_hash, script_hash))
    write_csv_atomic(TABLES / "E198_INPUT_HASHES.csv", pd.DataFrame(hashes))
    write_csv_atomic(TABLES / "E198_PREPARE_GATES.csv", pd.DataFrame(gates))
    write_csv_atomic(TABLES / "E198_PROTOCOL_METADATA.csv", metadata)
    write_csv_atomic(TABLES / "E198_RUNTIME_ENVIRONMENT.csv", runtime_environment())
    report = f"""# E198 prepare 报告

- 状态：`PREPARED_NOT_RUN`；
- arch1 只按不透明字节读取并计算 SHA-256，未调用 AnnData/HDF5；
- 数据字节：`{observed_bytes}`；
- SHA-256：`{data_hash}`；
- scPertEval commit：`{SCPERTEVAL_COMMIT}`；
- 固定协议：{len(PROTOCOL_SPECS)} 个；
- synthetic smoke：{len(smoke['protocols_checked'])}/{len(PROTOCOL_SPECS)} 通过；
- formal 必须等本次冻结提交到 GitHub 与 Gitee 后运行。
"""
    write_text_atomic(REPORTS / "E198_PREPARE_REPORT.md", report)
    write_status(
        "PREPARED_NOT_RUN",
        dataset_opened_as_anndata=False,
        dataset_bytes=observed_bytes,
        dataset_sha256=data_hash,
        scperteval_commit=SCPERTEVAL_COMMIT,
        protocols=list(PROTOCOL_SPECS),
        prepare_gates_passed=len(gates),
        prepare_gates_total=len(gates),
        prepared_parent_git_head=git_output(ROOT, "rev-parse", "HEAD"),
    )
    print(
        json.dumps(
            {
                "status": "PREPARED_NOT_RUN",
                "dataset_opened_as_anndata": False,
                "dataset_bytes": observed_bytes,
                "dataset_sha256": data_hash,
                "prepare_gates": len(gates),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def verify_prepare_inputs() -> tuple[list[dict[str, Any]], str, str]:
    required = (
        SCRIPT,
        FREEZE,
        TABLES / "E198_INPUT_HASHES.csv",
        TABLES / "E198_PREPARE_GATES.csv",
        TABLES / "E198_PROTOCOL_METADATA.csv",
        TABLES / "E198_RUNTIME_ENVIRONMENT.csv",
        REPORTS / "E198_PREPARE_REPORT.md",
        STATUS,
    )
    for path in required:
        if not path.is_file() or not tracked_in_head(path) or not git_clean_for_path(path):
            raise ContractFailure(
                f"formal requires committed unchanged prepare artefact: {logical_path(path)}"
            )
    previous = json.loads(STATUS.read_text(encoding="utf-8"))
    if previous.get("status") != "PREPARED_NOT_RUN":
        raise ContractFailure(f"unexpected prepare status: {previous.get('status')}")

    recorded = pd.read_csv(TABLES / "E198_INPUT_HASHES.csv", keep_default_na=False)
    if recorded.columns.tolist() != ["role", "path", "bytes", "sha256"]:
        raise ContractFailure("E198 input hash schema changed")
    path_map: dict[str, Path] = {
        logical_path(SCRIPT): SCRIPT,
        logical_path(DATASET): DATASET,
    }
    for relative in SCPERTEVAL_INPUTS:
        path = SCPERTEVAL_REPO / relative
        path_map[logical_path(path)] = path
    for item in recorded.itertuples(index=False):
        path = path_map.get(str(item.path))
        if path is None or not path.is_file():
            raise ContractFailure(f"cannot resolve frozen input: {item.path}")
        if path.stat().st_size != int(item.bytes):
            raise ContractFailure(f"frozen input size changed: {item.path}")
        if sha256_file(path) != str(item.sha256):
            raise ContractFailure(f"frozen input SHA changed: {item.path}")

    if git_output(SCPERTEVAL_REPO, "rev-parse", "HEAD") != SCPERTEVAL_COMMIT:
        raise ContractFailure("scPertEval commit changed after freeze")
    if git_output(SCPERTEVAL_REPO, "status", "--porcelain"):
        raise ContractFailure("scPertEval checkout is dirty after freeze")

    head = git_output(ROOT, "rev-parse", "HEAD")
    branch = git_output(ROOT, "branch", "--show-current")
    github = remote_tip("github", branch)
    gitee = remote_tip("origin", branch)
    if github != head or gitee != head:
        raise ContractFailure(
            f"formal freeze not aligned: local={head} github={github} gitee={gitee}"
        )
    return recorded.to_dict("records"), head, branch


def summarise_protocols(
    calibration: pd.DataFrame,
    metadata: pd.DataFrame,
    official_aggregates: dict[str, dict[str, float]],
) -> pd.DataFrame:
    meta = metadata.set_index("protocol")
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOL_SPECS:
        frame = calibration.loc[calibration.protocol == protocol].copy()
        if len(frame) != EXPECTED_PERTURBATIONS:
            raise ContractFailure(f"{protocol}: unexpected perturbation count")
        better = str(meta.loc[protocol, "better"])
        bds_values = independent_bds(
            frame.raw_positive.to_numpy(float),
            frame.raw_negative.to_numpy(float),
            better,
        )
        successes = int(bds_values.sum())
        bds = successes / len(bds_values)
        bds_low, bds_high = wilson_interval(successes, len(bds_values))
        drf = frame.drf.to_numpy(float)
        finite = drf[np.isfinite(drf)]
        finite_fraction = len(finite) / len(drf)
        drf_median = float(np.median(finite)) if len(finite) else float("nan")
        drf_mean = float(np.mean(finite)) if len(finite) else float("nan")
        drf_low, drf_high = bootstrap_median(drf, protocol)
        if (
            finite_fraction >= 0.9
            and bds_low > 0.5
            and drf_low > 0.0
        ):
            decision = "PRIMARY_ELIGIBLE"
        elif finite_fraction >= 0.9 and bds >= 0.5 and drf_median > 0.0:
            decision = "SECONDARY_ONLY"
        else:
            decision = "REJECT"
        aggregates = official_aggregates[protocol]
        rows.append(
            {
                "protocol": protocol,
                "axis": str(meta.loc[protocol, "axis"]),
                "representation": str(meta.loc[protocol, "representation"]),
                "space": str(meta.loc[protocol, "space"]),
                "better": better,
                "n_perturbations": len(frame),
                "n_finite_drf": len(finite),
                "finite_drf_fraction": finite_fraction,
                "bds": bds,
                "bds_successes": successes,
                "bds_wilson_low": bds_low,
                "bds_wilson_high": bds_high,
                "drf_mean": drf_mean,
                "drf_median": drf_median,
                "drf_median_boot_low": drf_low,
                "drf_median_boot_high": drf_high,
                "official_drf_mean": float(aggregates["mean"]),
                "official_drf_median": float(aggregates["median"]),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def choose_endpoints(summary: pd.DataFrame) -> pd.DataFrame:
    decision = summary.set_index("protocol").decision.to_dict()
    rows = []
    for axis, priority in AXIS_PRIORITY.items():
        eligible = [p for p in priority if decision[p] == "PRIMARY_ELIGIBLE"]
        selected = eligible[0] if eligible else ""
        rows.append(
            {
                "axis": axis,
                "priority": ">".join(priority),
                "selected_protocol": selected,
                "selection_status": "SELECTED" if selected else "NO_PRIMARY_ENDPOINT",
                "rule": "first PRIMARY_ELIGIBLE in preregistered priority",
            }
        )
    return pd.DataFrame(rows)


def formal_integrity_gates(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    metadata: pd.DataFrame,
    official_aggregates: dict[str, dict[str, float]],
    head: str,
    branch: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, observed: Any, expected: Any, passed: bool, detail: str = ""):
        rows.append(
            {
                "phase": "formal",
                "check": check,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    add("git_head", head, head, len(head) == 40)
    add("git_branch", branch, "non-empty", bool(branch))
    add(
        "protocol_order",
        ";".join(calibration.protocol.drop_duplicates().astype(str)),
        ";".join(PROTOCOL_SPECS),
        calibration.protocol.drop_duplicates().astype(str).tolist()
        == list(PROTOCOL_SPECS),
    )
    add(
        "total_rows",
        len(calibration),
        EXPECTED_PERTURBATIONS * len(PROTOCOL_SPECS),
        len(calibration) == EXPECTED_PERTURBATIONS * len(PROTOCOL_SPECS),
    )
    add(
        "no_model_predictions",
        "calibration controls only",
        "calibration controls only",
        "prediction" not in calibration.columns,
    )
    add(
        "no_sinkhorn",
        ";".join(PROTOCOL_SPECS),
        "no sinkhorn",
        not any("sinkhorn" in p for p in PROTOCOL_SPECS),
    )
    meta = metadata.set_index("protocol")
    max_delta = 0.0
    aggregate_delta = 0.0
    all_masks = True
    no_raw_infinity = True
    all_counts = True
    for protocol in PROTOCOL_SPECS:
        frame = calibration.loc[calibration.protocol == protocol]
        all_counts &= len(frame) == EXPECTED_PERTURBATIONS
        raw_values = frame[["raw_positive", "raw_negative"]].to_numpy(float)
        no_raw_infinity &= bool(not np.isinf(raw_values).any())
        expected = independent_drf(
            frame.raw_positive.to_numpy(float),
            frame.raw_negative.to_numpy(float),
            str(meta.loc[protocol, "better"]),
            float(meta.loc[protocol, "perfect"]),
        )
        observed = frame.drf.to_numpy(float)
        masks = np.array_equal(np.isfinite(observed), np.isfinite(expected))
        all_masks &= masks
        finite = np.isfinite(observed) & np.isfinite(expected)
        if finite.any():
            max_delta = max(
                max_delta,
                float(np.max(np.abs(observed[finite] - expected[finite]))),
            )
        finite_observed = observed[np.isfinite(observed)]
        if len(finite_observed):
            agg = official_aggregates[protocol]
            aggregate_delta = max(
                aggregate_delta,
                abs(float(np.mean(finite_observed)) - float(agg["mean"])),
                abs(float(np.median(finite_observed)) - float(agg["median"])),
            )
    add("all_protocol_counts", all_counts, True, all_counts)
    add("raw_controls_no_infinity", no_raw_infinity, True, no_raw_infinity)
    add("drf_na_masks", all_masks, True, all_masks)
    add("independent_drf_max_delta", max_delta, f"<={NUMERIC_TOL}", max_delta <= NUMERIC_TOL)
    add(
        "official_aggregate_max_delta",
        aggregate_delta,
        f"<={NUMERIC_TOL}",
        aggregate_delta <= NUMERIC_TOL,
    )
    add(
        "summary_protocol_count",
        len(summary),
        len(PROTOCOL_SPECS),
        len(summary) == len(PROTOCOL_SPECS),
    )
    add(
        "selection_axis_count",
        len(selection),
        len(AXIS_PRIORITY),
        len(selection) == len(AXIS_PRIORITY),
    )
    add(
        "selection_only_primary",
        ";".join(selection.selected_protocol.astype(str)),
        "selected entries are PRIMARY_ELIGIBLE or empty",
        all(
            not p
            or summary.set_index("protocol").loc[p, "decision"]
            == "PRIMARY_ELIGIBLE"
            for p in selection.selected_protocol.astype(str)
        ),
    )
    frame = pd.DataFrame(rows)
    failed = frame.loc[~frame.passed.astype(bool)]
    if len(failed):
        raise ContractFailure(
            "formal integrity gates failed: " + ", ".join(failed.check.astype(str))
        )
    return frame


def make_figure(summary: pd.DataFrame, output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = summary.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    colors = {
        "PRIMARY_ELIGIBLE": "#2C7FB8",
        "SECONDARY_ONLY": "#E39D35",
        "REJECT": "#8A8A8A",
    }
    point_colors = [colors[value] for value in frame.decision]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].errorbar(
        frame.bds,
        y,
        xerr=np.vstack(
            [frame.bds - frame.bds_wilson_low, frame.bds_wilson_high - frame.bds]
        ),
        fmt="none",
        ecolor="#B7B7B7",
        elinewidth=1.2,
        capsize=2,
        zorder=1,
    )
    axes[0].scatter(frame.bds, y, c=point_colors, s=42, zorder=2)
    axes[0].axvline(0.5, color="#B74343", linestyle="--", linewidth=1)
    axes[0].set_xlim(0, 1.02)
    axes[0].set_xlabel("BDS (95% Wilson interval)")
    axes[0].set_yticks(y, frame.protocol)
    axes[0].set_title("A  Control discrimination")

    axes[1].errorbar(
        frame.drf_median,
        y,
        xerr=np.vstack(
            [
                frame.drf_median - frame.drf_median_boot_low,
                frame.drf_median_boot_high - frame.drf_median,
            ]
        ),
        fmt="none",
        ecolor="#B7B7B7",
        elinewidth=1.2,
        capsize=2,
        zorder=1,
    )
    axes[1].scatter(frame.drf_median, y, c=point_colors, s=42, zorder=2)
    axes[1].axvline(0, color="#B74343", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Median DRF (95% perturbation bootstrap)")
    axes[1].set_title("B  Recovered dynamic range")
    fig.suptitle(
        "E198 · arch1 evaluation-protocol calibration",
        x=0.08,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.015,
        "Blue: primary eligible   Orange: secondary only   Grey: reject",
        color="#4A4A4A",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.05, 0.05, 1, 0.94))
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / "E198_protocol_calibration.png", dpi=300, facecolor="white")
    fig.savefig(output_root / "E198_protocol_calibration.pdf", facecolor="white")
    plt.close(fig)


def reports_text(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    gates: pd.DataFrame,
    runtime: pd.DataFrame,
    head: str,
) -> tuple[str, str, str]:
    decision_counts = summary.decision.value_counts().to_dict()
    selected = selection.loc[selection.selection_status == "SELECTED"]
    selected_lines = (
        "\n".join(
            f"- {row.axis}: `{row.selected_protocol}`"
            for row in selected.itertuples(index=False)
        )
        or "- 没有协议通过主要端点门槛。"
    )
    table_lines = [
        "| protocol | axis | BDS | BDS 95% lower | median DRF | DRF 95% lower | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        table_lines.append(
            f"| `{row.protocol}` | {row.axis} | {row.bds:.4f} | "
            f"{row.bds_wilson_low:.4f} | {row.drf_median:.4f} | "
            f"{row.drf_median_boot_low:.4f} | `{row.decision}` |"
        )
    table = "\n".join(table_lines)
    report = f"""# E198｜arch1 评价协议校准

分析性质：`EXTERNAL_PROTOCOL_CALIBRATION`。本实验只校准评价协议，没有模型预测，
不能写成 SafeConf 外部验证。

## 完整性

- Git freeze：`{head}`；
- 12 个固定协议 × {EXPECTED_PERTURBATIONS} 个扰动；
- integrity gates：{int(gates.passed.sum())}/{len(gates)}；
- official DRF 与独立公式最大差：`{gates.loc[gates.check == 'independent_drf_max_delta', 'observed'].iloc[0]}`；
- 总协议运行时间：{runtime.seconds.sum():.1f} 秒；
- PRIMARY / SECONDARY / REJECT：{decision_counts.get('PRIMARY_ELIGIBLE', 0)} /
  {decision_counts.get('SECONDARY_ONLY', 0)} / {decision_counts.get('REJECT', 0)}。

## 逐协议结果

{table}

## 事前优先级选出的后续端点

{selected_lines}

没有通过的 axis 留空，E199 不得依据模型表现临时更换协议。
"""
    interpretation = f"""# E198 结果解释

E198 回答的是“这个指标在 arch1 上能否把技术重复与无信息参考分开”，不是“哪个
预测模型最好”。BDS 检查方向，DRF 检查恢复了多少有效动态范围；两个条件都稳定
通过才进入 E199 的主要端点。

固定优先级实际选择如下：

{selected_lines}

即便多个协议通过，也不能把它们当作相互独立的生物证据；它们使用同一批 150 个
扰动。`arch1` 只有 H1 hESC 一个背景，下一轮只能做未见扰动，不能把结果改写成整行
或跨细胞背景泛化。群体和 DE 协议使用真实细胞，不是由 centroid 复制出来的伪细胞。
"""
    readme = f"""# E198 先看这个

1. [正式报告](reports/E198_REPORT.md)
2. [结果解释](reports/E198_INTERPRETATION.md)
3. [协议汇总](tables/E198_PROTOCOL_SUMMARY.csv)
4. [后续端点选择](tables/E198_ENDPOINT_SELECTION.csv)
5. [逐扰动原始正负对照](tables/E198_PROTOCOL_CALIBRATION.csv)
6. [白底汇总图](figures/E198_protocol_calibration.png)

本目录是评价协议校准，不是模型验证。所有结果以 `E198_STATUS.json` 和输出哈希为准。
"""
    return report, interpretation, readme


def stage_formal_outputs(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    gates: pd.DataFrame,
    runtime: pd.DataFrame,
    head: str,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e198_formal_", dir=str(OUT)) as tmp:
        stage = Path(tmp)
        stage_tables = stage / "tables"
        stage_figures = stage / "figures"
        stage_reports = stage / "reports"
        stage_tables.mkdir()
        stage_figures.mkdir()
        stage_reports.mkdir()
        calibration.to_csv(stage_tables / "E198_PROTOCOL_CALIBRATION.csv", index=False)
        summary.to_csv(stage_tables / "E198_PROTOCOL_SUMMARY.csv", index=False)
        selection.to_csv(stage_tables / "E198_ENDPOINT_SELECTION.csv", index=False)
        gates.to_csv(stage_tables / "E198_FORMAL_GATES.csv", index=False)
        runtime.to_csv(stage_tables / "E198_RUNTIME.csv", index=False)
        make_figure(summary, stage_figures)
        report, interpretation, readme = reports_text(
            summary, selection, gates, runtime, head
        )
        (stage_reports / "E198_REPORT.md").write_text(report, encoding="utf-8")
        (stage_reports / "E198_INTERPRETATION.md").write_text(
            interpretation, encoding="utf-8"
        )
        (stage / "README_先看这个.md").write_text(readme, encoding="utf-8")

        for source in sorted(stage.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(stage)
            destination = OUT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ContractFailure(f"formal destination already exists: {destination}")
            source.replace(destination)


def write_output_hashes(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=logical_path):
        add_hash(rows, "E198:formal_output", path)
    frame = pd.DataFrame(rows)
    write_csv_atomic(OUTPUT_HASH_INDEX, frame)
    return frame


def formal() -> None:
    if any(path.exists() for path in FORMAL_PAYLOADS) or OUTPUT_HASH_INDEX.exists():
        raise ContractFailure("formal outputs already exist; overwrite refused")
    _, head, branch = verify_prepare_inputs()
    metadata = pd.read_csv(TABLES / "E198_PROTOCOL_METADATA.csv", keep_default_na=False)
    write_status(
        "RUNNING_EXTERNAL_PROTOCOL_CALIBRATION",
        formal_git_head=head,
        formal_branch=branch,
        dataset_opened_as_anndata=False,
    )
    started = time.time()
    try:
        sp = import_frozen_scperteval()
        # This is the first AnnData/HDF5 read performed by E198.
        prepared = sp.prepare(
            str(DATASET),
            list(PROTOCOL_SPECS),
            subsample=SUBSAMPLE,
            seed=SEED,
            min_cells=MIN_CELLS,
            workers=WORKERS,
            name="arch1_E198_external_protocol_calibration",
        )
        frames: list[pd.DataFrame] = []
        runtimes: list[dict[str, Any]] = []
        aggregates: dict[str, dict[str, float]] = {}
        meta = metadata.set_index("protocol")
        for protocol in PROTOCOL_SPECS:
            protocol_started = time.perf_counter()
            result = sp.calibrate(
                prepared,
                protocol,
                de_method=DE_METHOD,
                calibrator="drf",
            )
            seconds = time.perf_counter() - protocol_started
            frame = result.per_perturbation.copy()
            frame["axis"] = str(meta.loc[protocol, "axis"])
            frame["representation"] = str(meta.loc[protocol, "representation"])
            frame["space"] = str(meta.loc[protocol, "space"])
            frame["better"] = str(meta.loc[protocol, "better"])
            frame["perfect"] = float(meta.loc[protocol, "perfect"])
            independent = independent_drf(
                frame.raw_positive.to_numpy(float),
                frame.raw_negative.to_numpy(float),
                str(meta.loc[protocol, "better"]),
                float(meta.loc[protocol, "perfect"]),
            )
            frame["drf_independent"] = independent
            frame["bds_independent"] = independent_bds(
                frame.raw_positive.to_numpy(float),
                frame.raw_negative.to_numpy(float),
                str(meta.loc[protocol, "better"]),
            )
            frames.append(frame)
            aggregates[protocol] = {
                "mean": float(result.aggregate["mean"]),
                "median": float(result.aggregate["median"]),
            }
            runtimes.append(
                {
                    "protocol": protocol,
                    "axis": str(meta.loc[protocol, "axis"]),
                    "seconds": seconds,
                    "n_perturbations": len(frame),
                }
            )
            print(
                json.dumps(
                    {
                        "protocol": protocol,
                        "seconds": round(seconds, 3),
                        "rows": len(frame),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        calibration = pd.concat(frames, ignore_index=True)
        runtime = pd.DataFrame(runtimes)
        summary = summarise_protocols(calibration, metadata, aggregates)
        selection = choose_endpoints(summary)
        gates = formal_integrity_gates(
            calibration, summary, selection, metadata, aggregates, head, branch
        )
        stage_formal_outputs(
            calibration, summary, selection, gates, runtime, head
        )
        elapsed = time.time() - started
        write_status(
            "COMPLETE_EXTERNAL_PROTOCOL_CALIBRATION",
            formal_git_head=head,
            formal_branch=branch,
            dataset_opened_as_anndata=True,
            dataset_sha256=pd.read_csv(
                TABLES / "E198_INPUT_HASHES.csv", keep_default_na=False
            ).loc[lambda x: x.role == "E198:arch1_opaque_bytes", "sha256"].iloc[0],
            scperteval_commit=SCPERTEVAL_COMMIT,
            protocols=list(PROTOCOL_SPECS),
            n_perturbations=EXPECTED_PERTURBATIONS,
            n_protocol_rows=len(calibration),
            primary_eligible=int((summary.decision == "PRIMARY_ELIGIBLE").sum()),
            secondary_only=int((summary.decision == "SECONDARY_ONLY").sum()),
            rejected=int((summary.decision == "REJECT").sum()),
            selected_endpoints=selection.loc[
                selection.selection_status == "SELECTED", "selected_protocol"
            ].astype(str).tolist(),
            formal_gates_passed=int(gates.passed.astype(bool).sum()),
            formal_gates_total=len(gates),
            elapsed_seconds=elapsed,
            model_predictions_used=False,
            safeconf_validated=False,
            cross_context_claim=False,
        )
        hashes = write_output_hashes(list(FORMAL_PAYLOADS) + [STATUS])
        print(
            json.dumps(
                {
                    "status": "COMPLETE_EXTERNAL_PROTOCOL_CALIBRATION",
                    "rows": len(calibration),
                    "primary_eligible": int(
                        (summary.decision == "PRIMARY_ELIGIBLE").sum()
                    ),
                    "selected_endpoints": selection.loc[
                        selection.selection_status == "SELECTED",
                        "selected_protocol",
                    ].astype(str).tolist(),
                    "elapsed_seconds": round(elapsed, 1),
                    "output_hashes": len(hashes),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        write_status(
            "FAILED_EXTERNAL_PROTOCOL_CALIBRATION",
            formal_git_head=head,
            formal_branch=branch,
            dataset_opened_as_anndata=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_seconds=time.time() - started,
        )
        attempt = OUT / "ATTEMPT_LOG.md"
        previous = attempt.read_text(encoding="utf-8") if attempt.exists() else "# E198 attempt log\n"
        write_text_atomic(
            attempt,
            previous
            + f"\n- {now()} `{type(exc).__name__}`: {str(exc)}\n",
        )
        raise


def audit() -> None:
    if not OUTPUT_HASH_INDEX.is_file() or not STATUS.is_file():
        raise ContractFailure("formal outputs are incomplete; audit refused")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE_EXTERNAL_PROTOCOL_CALIBRATION":
        raise ContractFailure(f"formal status is not complete: {status.get('status')}")
    hashes = pd.read_csv(OUTPUT_HASH_INDEX, keep_default_na=False)
    path_map = {logical_path(path): path for path in list(FORMAL_PAYLOADS) + [STATUS]}
    hash_failures = []
    for item in hashes.itertuples(index=False):
        path = path_map.get(str(item.path))
        if (
            path is None
            or not path.is_file()
            or path.stat().st_size != int(item.bytes)
            or sha256_file(path) != str(item.sha256)
        ):
            hash_failures.append(str(item.path))
    calibration = pd.read_csv(TABLES / "E198_PROTOCOL_CALIBRATION.csv")
    summary = pd.read_csv(TABLES / "E198_PROTOCOL_SUMMARY.csv")
    metadata = pd.read_csv(TABLES / "E198_PROTOCOL_METADATA.csv", keep_default_na=False)
    max_delta = 0.0
    max_bds_delta = 0.0
    for protocol in PROTOCOL_SPECS:
        frame = calibration.loc[calibration.protocol == protocol]
        meta = metadata.set_index("protocol").loc[protocol]
        drf = independent_drf(
            frame.raw_positive.to_numpy(float),
            frame.raw_negative.to_numpy(float),
            str(meta.better),
            float(meta.perfect),
        )
        finite = np.isfinite(drf) & np.isfinite(frame.drf.to_numpy(float))
        if finite.any():
            max_delta = max(
                max_delta,
                float(
                    np.max(
                        np.abs(drf[finite] - frame.drf.to_numpy(float)[finite])
                    )
                ),
            )
        bds = independent_bds(
            frame.raw_positive.to_numpy(float),
            frame.raw_negative.to_numpy(float),
            str(meta.better),
        ).mean()
        saved_bds = float(summary.loc[summary.protocol == protocol, "bds"].iloc[0])
        max_bds_delta = max(max_bds_delta, abs(float(bds) - saved_bds))
    passed = (
        not hash_failures
        and max_delta <= NUMERIC_TOL
        and max_bds_delta <= NUMERIC_TOL
        and len(calibration) == EXPECTED_PERTURBATIONS * len(PROTOCOL_SPECS)
    )
    payload = {
        "experiment": "E198_arch1_protocol_calibration",
        "generated_at": now(),
        "status": "PASS" if passed else "FAIL",
        "formal_output_hashes_checked": len(hashes),
        "hash_failures": hash_failures,
        "calibration_rows": len(calibration),
        "independent_drf_max_delta": max_delta,
        "independent_bds_max_delta": max_bds_delta,
        "dataset_reopened": False,
    }
    write_json_atomic(AUDIT_STATUS, payload)
    write_text_atomic(
        REPORTS / "E198_INDEPENDENT_AUDIT.md",
        f"""# E198 独立复核

- 状态：`{payload['status']}`；
- 正式输出哈希：{len(hashes)}/{len(hashes)} 检查，失败 {len(hash_failures)}；
- 逐扰动 DRF 独立公式最大差：`{max_delta}`；
- BDS 汇总独立复算最大差：`{max_bds_delta}`；
- 正式行数：{len(calibration)}；
- 本复核没有重新打开 arch1 数据。
""",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise ContractFailure("E198 independent audit failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "formal", "synthetic-smoke", "audit"),
        default="prepare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "formal":
        formal()
    elif args.mode == "audit":
        audit()
    else:
        synthetic_smoke()


if __name__ == "__main__":
    main()

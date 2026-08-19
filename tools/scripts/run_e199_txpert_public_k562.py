#!/usr/bin/env python3
"""E199 source, asset, environment and leakage-contract audit.

This first implementation intentionally stops before model execution.  It
accepts the two official archives only after size, MD5, ZIP integrity and
source-version checks pass, then writes the frozen prepare artefacts that must
be committed to both remotes before any E199 prediction is run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E199_txpert_public_k562_20260802"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FREEZE = OUT / "ANALYSIS_FREEZE.md"
STATUS = OUT / "E199_STATUS.json"

TXPERT = Path("/home/yyf/archive/external/TxPert")
TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
SCPERTEVAL = Path("/home/yyf/archive/external/scPertEval")
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
SCPERTURBENCH = Path("/home/yyf/archive/external/scPerturBench")
SCPERTURBENCH_COMMIT = "6e24e7a9827e55d4567d2139427be9af0d1e7a6c"
VENV_PYTHON = Path("/home/yyf/.venvs/txpert-08d82eea/bin/python")

ASSET_DIR = Path("/home/yyf/data/txpert_official_20260802/cache")
ASSETS = {
    "checkpoints.zip": {
        "bytes": 291_708_781,
        "md5": "4058d19e14882a3bd2545b1512f1acde",
        "required_members": (
            "checkpoints/K562_unseen_pert_gat.ckpt",
            "checkpoints/K562_unseen_pert_exphormer.ckpt",
            "checkpoints/K562_unseen_pert_exphormer_mg.ckpt",
        ),
    },
    "K562_single_cell_line.zip": {
        "bytes": 678_058_077,
        "md5": "6be4c8239fd7e6b70d2ffe7b80c3c7bc",
        "required_suffixes": (
            "K562_single_cell_line/de_adata_test.h5ad",
            "K562_single_cell_line/splits/train_test_split.pkl",
            "K562_single_cell_line/splits/subgroup.pkl",
        ),
    },
}

SOURCE_FILES = (
    "README.md",
    "license.pdf",
    "pyproject.toml",
    "uv.lock",
    "main.py",
    "gspp/predictor.py",
    "gspp/data/datamodule.py",
    "gspp/data/graphmodule.py",
    "gspp/models/baselines.py",
    "configs/config-gat.yaml",
    "configs/config-exphormer.yaml",
    "configs/config-exphormer-mg.yaml",
    "configs/config-baseline.yaml",
)


class ContractFailure(RuntimeError):
    """Fail-closed E199 prepare error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def add_gate(
    rows: list[dict[str, Any]],
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
    detail: str = "",
) -> None:
    rows.append(
        {
            "phase": "prepare",
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise ContractFailure(f"function {name!r} not found in {path}")


def static_leakage_audit() -> pd.DataFrame:
    predictor = TXPERT / "gspp/predictor.py"
    predict_step = function_source(predictor, "predict_step")
    sample_inference = function_source(predictor, "sample_inference")
    forward = function_source(predictor, "forward")
    main_infer = function_source(TXPERT / "main.py", "infer")

    checks = [
        (
            "predict_step_truth_only_saved",
            predict_step.count("batch.x") == 1
            and "'ground_truths': batch.x" in predict_step,
            f"batch.x occurrences={predict_step.count('batch.x')}",
        ),
        (
            "sample_inference_has_no_target",
            "batch.x" not in sample_inference
            and "ground_truth" not in sample_inference
            and "target" not in sample_inference,
            "sample_inference receives control, perturbation indices and embeddings",
        ),
        (
            "forward_has_no_target",
            "target" not in forward and "ground_truth" not in forward,
            "forward signature excludes target expression",
        ),
        (
            "predict_mode_uses_trainer_predict",
            'elif cfg.mode == "predict"' in main_infer
            and "trainer.predict(GSP_model, datamodule)" in main_infer,
            "official predict branch inspected",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "detail": detail}
            for check, passed, detail in checks
        ]
    )


def prepare() -> None:
    if (OUT / "formal_evaluation").exists():
        raise ContractFailure("formal outputs already exist; prepare overwrite refused")

    gates: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []

    add_gate(gates, "freeze_exists", FREEZE.is_file(), True, FREEZE.is_file())
    add_gate(gates, "adapter_exists", (ROOT / "tools/scripts/txpert_public_adapter.py").is_file(), True,
             (ROOT / "tools/scripts/txpert_public_adapter.py").is_file())

    for label, repo, expected in (
        ("txpert", TXPERT, TXPERT_COMMIT),
        ("scperteval", SCPERTEVAL, SCPERTEVAL_COMMIT),
        ("scperturbench", SCPERTURBENCH, SCPERTURBENCH_COMMIT),
    ):
        observed = git_output(repo, "rev-parse", "HEAD")
        add_gate(gates, f"{label}_commit", observed, expected, observed == expected)
        status = git_output(repo, "status", "--porcelain")
        add_gate(gates, f"{label}_clean", status or "CLEAN", "CLEAN", not status)

    for filename, spec in ASSETS.items():
        path = ASSET_DIR / filename
        exists = path.is_file()
        add_gate(gates, f"{filename}:exists", exists, True, exists)
        if not exists:
            continue
        observed_bytes = path.stat().st_size
        observed_md5 = digest_file(path, "md5")
        observed_sha256 = digest_file(path, "sha256")
        add_gate(
            gates,
            f"{filename}:bytes",
            observed_bytes,
            spec["bytes"],
            observed_bytes == spec["bytes"],
        )
        add_gate(
            gates,
            f"{filename}:md5",
            observed_md5,
            spec["md5"],
            observed_md5 == spec["md5"],
        )
        try:
            with zipfile.ZipFile(path) as archive:
                corrupt = archive.testzip()
                members = archive.namelist()
        except zipfile.BadZipFile as exc:
            corrupt = f"BAD_ZIP:{exc}"
            members = []
        add_gate(gates, f"{filename}:zip_integrity", corrupt or "PASS", "PASS", corrupt is None)
        for member in spec.get("required_members", ()):
            add_gate(
                gates,
                f"{filename}:member:{member}",
                member in members,
                True,
                member in members,
            )
        for suffix in spec.get("required_suffixes", ()):
            matched = [member for member in members if member.endswith(suffix)]
            add_gate(
                gates,
                f"{filename}:suffix:{suffix}",
                len(matched),
                1,
                len(matched) == 1,
                matched[0] if len(matched) == 1 else "",
            )
        inputs.append(
            {
                "role": "official_zenodo_asset",
                "path": f"DATA/txpert_official_20260802/cache/{filename}",
                "bytes": observed_bytes,
                "md5": observed_md5,
                "sha256": observed_sha256,
            }
        )

    for rel in SOURCE_FILES:
        path = TXPERT / rel
        exists = path.is_file()
        add_gate(gates, f"source_exists:{rel}", exists, True, exists)
        if exists:
            inputs.append(
                {
                    "role": "txpert_source",
                    "path": f"EXTERNAL/TxPert/{rel}",
                    "bytes": path.stat().st_size,
                    "md5": digest_file(path, "md5"),
                    "sha256": digest_file(path, "sha256"),
                }
            )

    for role, path in (
        ("e199_runner", SCRIPT),
        ("e199_adapter", ROOT / "tools/scripts/txpert_public_adapter.py"),
        ("e199_freeze", FREEZE),
    ):
        if path.is_file():
            inputs.append(
                {
                    "role": role,
                    "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                    "bytes": path.stat().st_size,
                    "md5": digest_file(path, "md5"),
                    "sha256": digest_file(path, "sha256"),
                }
            )

    env_probe = subprocess.run(
        [
            str(VENV_PYTHON),
            "-c",
            "import torch,torch_geometric,scanpy,lightning;"
            "print(torch.__version__,torch.version.cuda,torch_geometric.__version__,"
            "scanpy.__version__,lightning.__version__,torch.cuda.is_available())",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    add_gate(gates, "environment_probe_exit", env_probe.returncode, 0, env_probe.returncode == 0,
             env_probe.stderr.strip())
    env_line = env_probe.stdout.strip()
    add_gate(gates, "environment_versions", env_line,
             "2.6.0+cu124 12.4 2.6.1 1.11.1 2.5.1 True",
             env_line == "2.6.0+cu124 12.4 2.6.1 1.11.1 2.5.1 True")

    leakage = static_leakage_audit()
    for row in leakage.itertuples(index=False):
        add_gate(gates, f"static:{row.check}", row.passed, True, bool(row.passed), row.detail)

    gates_df = pd.DataFrame(gates)
    failed = gates_df.loc[~gates_df["passed"].astype(bool)]
    write_csv(TABLES / "E199_PREPARE_GATES.csv", gates_df)
    write_csv(TABLES / "E199_INPUT_HASHES.csv", pd.DataFrame(inputs))
    write_csv(TABLES / "E199_STATIC_LEAKAGE_AUDIT.csv", leakage)
    write_csv(
        TABLES / "E199_RUNTIME_ENVIRONMENT.csv",
        pd.DataFrame(
            [
                {"component": "platform", "version": platform.platform()},
                {"component": "python", "version": platform.python_version()},
                {"component": "txpert_venv_probe", "version": env_line},
            ]
        ),
    )

    report = f"""# E199 prepare 报告

- 生成时间：`{now()}`
- TxPert commit：`{TXPERT_COMMIT}`
- Zenodo record：`15420279`
- 输入资产：{len(ASSETS)} 个官方 ZIP；只检查字节、哈希和 ZIP 目录，未执行模型。
- prepare gates：{int(gates_df.passed.sum())}/{len(gates_df)}
- 静态防泄漏检查：{int(leakage.passed.sum())}/{len(leakage)}
- 运行环境：`{env_line}`

下一步必须先把 runner、adapter、冻结合同和本 prepare 产物提交到 GitHub 与 Gitee，
确认三端 commit 完全相同，再解压资产并执行运行时 `batch.x` 置零不变性测试。任何
失败项都会阻止 formal；不会删除失败行后继续。
"""
    write_text(REPORTS / "E199_PREPARE_REPORT.md", report)
    write_json(
        STATUS,
        {
            "experiment": "E199_txpert_public_k562",
            "analysis_class": "RETROSPECTIVE_PUBLIC_MODEL_REPRODUCTION_AND_RISK_AUDIT",
            "generated_at": now(),
            "status": "PREPARE_COMPLETE" if failed.empty else "PREPARE_FAILED",
            "prepare_gates_passed": int(gates_df.passed.sum()),
            "prepare_gates_total": len(gates_df),
            "failed_checks": failed["check"].astype(str).tolist(),
            "parent_git_head": git_output(ROOT, "rev-parse", "HEAD"),
        },
    )
    if not failed.empty:
        raise ContractFailure(
            "E199 prepare failed: " + ", ".join(failed["check"].astype(str))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare",), default="prepare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare()


if __name__ == "__main__":
    main()

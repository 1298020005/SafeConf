#!/usr/bin/env python3
"""Add the independent post-freeze E153 scale-mixing audit.

This script does not change E153 tasks, scores, endpoints, bootstrap draws or
meta-analysis results.  It verifies the frozen E153 script/input hashes, then
adds a reproducible report supplement explaining the pooled Replogle Simpson
reversal found during independent review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
E153 = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714"
TABLES = E153 / "tables"
REPORT = E153 / "reports/E153_REPORT.md"
FREEZE = E153 / "FREEZE_STATUS.json"
STATUS = E153 / "RUN_STATUS.json"
FROZEN_SCRIPT = ROOT / "tools/scripts/run_e153_eight_study_formal_meta.py"
TASK_INPUT = TABLES / "E153_ABSOLUTE_TASK_INPUT.csv"
FOLD_RESULTS = TABLES / "E153_ABSOLUTE_FOLD_CORRELATIONS.csv"
LODO = TABLES / "E153_ABSOLUTE_LODO.csv"
AUDIT_TABLE = TABLES / "E153_REPLOGLE_SCALE_MIXING_AUDIT.csv"
AUDIT_REPORT = E153 / "POSTFREEZE_INDEPENDENT_AUDIT.md"

START = "<!-- E153_POSTFREEZE_SCALE_AUDIT_START -->"
END = "<!-- E153_POSTFREEZE_SCALE_AUDIT_END -->"
TARGET = "error_two_predictor_mean_rmse"
SCORE = "safeconf_calibrated_pair_risk"
POOLED_ESTIMAND = "pooled_context_task_median_sensitivity"


def sha256(path: Path, prefix: bool = False) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}" if prefix else digest


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE.read_text())
    checks = {
        "frozen_analysis_script": (
            sha256(FROZEN_SCRIPT, prefix=True),
            freeze["analysis_script_sha256"],
        ),
        "absolute_task_snapshot": (
            sha256(TASK_INPUT, prefix=True),
            freeze["absolute_snapshot_sha256"],
        ),
    }
    failures = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if failures:
        raise RuntimeError(f"E153 frozen inputs changed: {failures}")
    return {
        name: {"observed": observed, "expected": expected, "passed": True}
        for name, (observed, expected) in checks.items()
    }


def main() -> None:
    checks = verify_freeze()
    tasks = pd.read_csv(TASK_INPUT)
    folds = pd.read_csv(FOLD_RESULTS)
    lodo = pd.read_csv(LODO)

    replogle = tasks[tasks["dataset"].eq("Replogle_two_cellline")].copy()
    if len(replogle) != 256:
        raise RuntimeError(f"expected 256 Replogle tasks, observed {len(replogle)}")
    if replogle[["context", "perturbation"]].duplicated().any():
        raise RuntimeError("Replogle context-task rows are not unique")

    scale = (
        replogle.groupby(["context", "fold_id"], as_index=False)
        .agg(
            n_tasks=("task_id", "size"),
            mean_error=(TARGET, "mean"),
            mean_calibrated_safeconf=(SCORE, "mean"),
        )
        .merge(
            folds[
                folds["dataset"].eq("Replogle_two_cellline")
                & folds["score"].eq(SCORE)
            ][["fold_id", "spearman"]].rename(
                columns={"spearman": "within_fold_spearman"}
            ),
            on="fold_id",
            how="left",
            validate="one_to_one",
        )
    )
    pooled = float(spearmanr(replogle[SCORE], replogle[TARGET]).statistic)
    scale["pooled_cross_context_spearman"] = pooled
    scale.to_csv(AUDIT_TABLE, index=False)

    sensitivity = lodo[
        lodo["estimand"].eq(POOLED_ESTIMAND)
        & lodo["analysis"].eq("safeconf_minus_comparator")
        & lodo["removed_dataset"].eq("Replogle_two_cellline")
    ][
        [
            "effect",
            "removed_dataset",
            "pooled_z",
            "ci95_low_z",
            "ci95_high_z",
            "prediction_low_z",
            "prediction_high_z",
        ]
    ]
    if len(sensitivity) != 2:
        raise RuntimeError("expected two pooled-sensitivity Replogle LODO rows")

    scale_display = scale.copy()
    for column in [
        "mean_error",
        "mean_calibrated_safeconf",
        "within_fold_spearman",
        "pooled_cross_context_spearman",
    ]:
        scale_display[column] = scale_display[column].map(lambda value: f"{value:.4f}")
    sensitivity_display = sensitivity.copy()
    for column in sensitivity_display.columns[2:]:
        sensitivity_display[column] = sensitivity_display[column].map(
            lambda value: f"{value:.4f}"
        )

    section = f"""{START}
### Pooled 敏感性的独立复核：Replogle 尺度混排

该敏感性中 SafeConf−disagreement（z=-0.0271）和 SafeConf−magnitude
（z=-0.0379）均为负，**不支持主分析结论**。删除 Replogle 后的结果为：

{markdown_table(sensitivity_display)}

两项删除 Replogle 后均由负转正。Replogle 的逐细胞系诊断为：

{markdown_table(scale_display)}

K562 与 RPE1 各自只出现在一个 fold。两折内部 SafeConf–误差相关均为正；但两个 fold 的
原始校准分数均值和误差均值方向相反，跨 context 直接排序后 rho={pooled:.4f}。这是
Simpson 反转，不是字段错配。校准 SafeConf 属于 fold 特异尺度，不能把不同 fold 的原始值
直接当作同一量尺混排。

因此 pooled-median 结果应视作尺度混排诊断：它不能支持主结论，也不能覆盖或否定合同预定的
fold-macro 主分析。数值复核见 `tables/E153_REPLOGLE_SCALE_MIXING_AUDIT.csv`。
{END}"""

    report = REPORT.read_text()
    if START in report:
        before, rest = report.split(START, 1)
        _, after = rest.split(END, 1)
        report = before + section + after
    else:
        anchor = "\n## Directional-SafeConf：Nadig与Replogle"
        if anchor not in report:
            raise RuntimeError("E153 report insertion anchor not found")
        report = report.replace(anchor, "\n" + section + "\n" + anchor, 1)
    REPORT.write_text(report)

    audit = f"""# E153 冻结后独立复核

本复核只检查既有结果，没有修改任务、风险分数、端点、bootstrap draw 或元分析。

- 3465 行输入等于 E146 七研究 3209 行加 E151 Replogle 256 行。
- Replogle 为 128 个 perturbation × 2 个 context，共 256 个唯一任务。
- 独立复算确认 fold 相关、perturbation-cluster bootstrap、REML/mKH/PI 和方向两研究描述性合并。
- 必须披露的问题是 pooled Replogle 的 Simpson 反转；正文已加入尺度混排说明与 LODO。
- 冻结分析脚本 SHA-256：`{sha256(FROZEN_SCRIPT)}`，验证通过。
- 冻结绝对任务快照 SHA-256：`{sha256(TASK_INPUT)}`，验证通过。

本复核不把 E153 升格为新的预注册确认实验，也不提供跨研究、湿实验或期刊录用保证。
"""
    AUDIT_REPORT.write_text(audit)

    status = json.loads(STATUS.read_text())
    status["postfreeze_scale_mixing_audit"] = {
        "status": "complete",
        "changed_frozen_analysis": False,
        "freeze_checks": checks,
        "replogle_pooled_cross_context_spearman": pooled,
        "sensitivity_lodo_without_replogle": {
            row.effect: float(row.pooled_z)
            for row in sensitivity.itertuples(index=False)
        },
        "audit_table_sha256": sha256(AUDIT_TABLE),
        "audit_report_sha256": sha256(AUDIT_REPORT),
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status["postfreeze_scale_mixing_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

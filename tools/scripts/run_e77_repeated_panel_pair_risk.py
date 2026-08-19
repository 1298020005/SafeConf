#!/usr/bin/env python3
"""E77: repeat E74 on two non-overlapping panels in three datasets."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_e74_pair_risk_certificate as core


OUT = ROOT / "docs" / "实验结果" / "E77_repeated_panel_pair_risk_20260711"
SOURCES = {
    "Adamson_P1": ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711",
    "Adamson_P2": ROOT / "docs" / "实验结果" / "E76a_adamson_scgpt_panel2_20260711",
    "Norman_P1": ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711",
    "Norman_P2": ROOT / "docs" / "实验结果" / "E76b_norman_scgpt_panel2_20260711",
    "Frangieh_P1": ROOT / "docs" / "实验结果" / "E72_frangieh_scgpt_formal_fixed_panel_20260711",
    "Frangieh_P2": ROOT / "docs" / "实验结果" / "E76c_frangieh_scgpt_panel2_20260711",
}


def main() -> None:
    core.OUT = OUT
    core.TABLES, core.REPORTS = OUT / "tables", OUT / "reports"
    core.SOURCES = SOURCES
    core.main()
    report = core.REPORTS / "E74_REPORT.md"
    text = report.read_text(encoding="utf-8")
    text = (
        text.replace("# E74｜", "# E77｜")
        .replace("三数据集分层关联", "三数据集×两套不重叠面板的分层关联")
        .replace("72 个真实任务", "144 个真实任务")
        .replace("72/72", "144/144")
    )
    target = core.REPORTS / "E77_REPORT.md"
    target.write_text(text, encoding="utf-8")
    for source_name, target_name in [
        ("E74_TASK_CERTIFICATES.csv", "E77_TASK_CERTIFICATES.csv"),
        ("E74_STRATIFIED_ASSOCIATION.csv", "E77_STRATIFIED_ASSOCIATION.csv"),
        ("E74_INCREMENTAL_DELTA.csv", "E77_INCREMENTAL_DELTA.csv"),
    ]:
        shutil.copy2(core.TABLES / source_name, core.TABLES / target_name)
    (OUT / "README_先看这个.md").write_text("# E77 repeated-panel pair risk\n\n先读 `reports/E77_REPORT.md`。\n", encoding="utf-8")
    status_path = OUT / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["experiment"] = "E77_repeated_panel_pair_risk"
    status["underlying_datasets"] = ["Adamson", "Norman", "Frangieh"]
    status["panels_per_dataset"] = 2
    status["panel_task_overlap_within_dataset"] = 0
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

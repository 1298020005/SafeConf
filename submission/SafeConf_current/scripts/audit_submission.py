#!/usr/bin/env python3
"""Audit the canonical SafeConf submission package against frozen releases."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
E182 = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
E183 = ROOT / "docs/实验结果/E183_all_study_family_synthesis_20260724"
E145 = ROOT / "docs/实验结果/E145_prescribe_paper_endpoint_20260714"
E178 = ROOT / "docs/实验结果/E178_crossstudy_bilateral_certificate_audit_20260722"
E179 = ROOT / "docs/实验结果/E179_nested_uq_baseline_benchmark_20260723"
E187 = ROOT / "docs/实验结果/E187_advisor_difficulty_certificate_20260726"

checks: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> None:
    checks.append((bool(condition), message))


main = (HERE / "SafeConf_manuscript.md").read_text(encoding="utf-8")
supp = (HERE / "SafeConf_supplement.md").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
summary = pd.read_csv(E183 / "tables/E183_STUDY_SUMMARY.csv")
tasks = pd.read_csv(E183 / "tables/E183_COMBINED_TASK_CERTIFICATES.csv")
targets = pd.read_csv(E183 / "tables/E183_TARGET_CERTIFICATES.csv")
e182_targets = pd.read_csv(
    E182 / "final_evaluation/tables/E182_EVALUATION_TARGETS.csv"
)
with (E182 / "final_evaluation/E182_FINAL_SUMMARY.json").open() as handle:
    e182_final = json.load(handle)
with (E187 / "RUN_STATUS.json").open() as handle:
    e187_status = json.load(handle)
e187_macro = pd.read_csv(E187 / "tables/E187_MACRO_BOOTSTRAP.csv")
e187_cross = pd.read_csv(E187 / "tables/E187_CROSS_DATASET_SUMMARY.csv")
e178_shared = pd.read_csv(E178 / "tables/E178_SHARED_DIFFICULTY.csv")
e178_specific = pd.read_csv(E178 / "tables/E178_MODEL_SPECIFICITY.csv")
e179_methods = pd.read_csv(E179 / "tables/E179_METHOD_SUMMARY.csv")
e145_associations = pd.read_csv(E145 / "tables/E145_ASSOCIATIONS.csv")
e145_incremental = pd.read_csv(E145 / "tables/E145_INCREMENTAL_VS_MAGNITUDE.csv")
e145_redundancy = pd.read_csv(E145 / "tables/E145_SCORE_REDUNDANCY.csv")

bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
cited_keys = set(re.findall(r"@([A-Za-z0-9_:-]+)", main + "\n" + supp))
check(cited_keys == bib_keys, "all and only bibliography entries are cited")

check(int(summary["n_tasks"].sum()) == 2433, "study task total is 2,433")
check(int(summary["n_target_clusters"].sum()) == 737, "target total is 737")
check(len(tasks) == 2433, "combined task table has 2,433 rows")
check(len(targets) == 737, "combined target table has 737 rows")
check(int(summary["family_lower_violations"].sum()) == 0, "family lower violations are zero")
check(int(summary["worst_lower_violations"].sum()) == 0, "worst-member lower violations are zero")
check(
    int(summary["family_upper_tasks_covered"].sum()) == 2331,
    "family upper task count is 2,331/2,433",
)
check(
    int(summary["family_upper_targets_covered"].sum()) == 666,
    "family upper target count is 666/737",
)
check(
    int(summary["worst_upper_targets_covered"].sum()) == 688,
    "worst-member upper target count is 688/737",
)
check(
    abs(float(summary["max_identity_abs_residual"].max()) - 9.6819254003e-17)
    < 1e-28,
    "maximum identity residual matches E183",
)

check(int(e182_final["covered_targets"]) == 16, "E182 family target result remains 16/20")
check(
    int(e182_targets["family_rms_simultaneous_covered"].sum()) == 16,
    "E182 target table reconstructs 16 family-covered targets",
)
check(
    int(e182_targets["worst_member_simultaneous_covered"].sum()) == 19,
    "E182 target table reconstructs 19 worst-covered targets",
)
failed = set(
    e182_targets.loc[
        ~e182_targets["family_rms_simultaneous_covered"].astype(bool),
        "perturbation",
    ]
)
check(failed == {"HNRNPC", "DDX6", "SLTM", "DDX42"}, "E182 failure identities are frozen")
check(
    all(token in main and token in supp for token in ("16/20", "HNRNPC", "DDX42")),
    "main text and supplement both retain the negative result",
)

for number in (
    "2,433",
    "737",
    "2,331",
    "666/737",
    "688/737",
    "9.68 × 10⁻¹⁷",
    "8,777",
    "8,196",
    "581",
):
    check(number in main, f"main manuscript reports {number}")

check(e187_status["status"] == "PASS", "E187 difficulty audit status is PASS")
check(int(e187_status["cartesian_task_instances"]) == 8196, "E187 has 8,196 Cartesian tasks")
check(int(e187_status["cross_dataset_tasks"]) == 581, "E187 has 581 direct-transfer tasks")
check(int(e187_status["total_task_instances"]) == 8777, "E187 has 8,777 total tasks")
check(int(e187_status["family_rms_violations"]) == 0, "E187 family RMS violations are zero")
check(int(e187_status["family_worst_violations"]) == 0, "E187 worst-member violations are zero")
check(e187_status["cross_dataset_upper_claim"] is False, "E187 makes no cross-dataset upper claim")

full = e187_macro[e187_macro["train_fraction"] == 1.0].set_index("setting_label")
expected_tightness = {
    "Random pair": 0.32824833929548713,
    "Unseen context": 0.259861067067906,
    "Unseen perturbation": 0.17486368563427893,
    "Double unseen": 0.1475512978145298,
}
for setting, expected in expected_tightness.items():
    check(
        abs(float(full.loc[setting, "macro_median_tightness"]) - expected) < 1e-12,
        f"E187 full-fraction tightness matches for {setting}",
    )
check(
    list(e187_cross["n_tasks"].astype(int)) == [553, 28],
    "E187 direct-transfer task counts are 553 and 28",
)
check(
    int(e187_cross["family_rms_violations"].sum()) == 0
    and int(e187_cross["family_worst_violations"].sum()) == 0,
    "E187 direct-transfer lower violations are zero",
)
check(
    not e187_cross["cross_dataset_upper_claim"].astype(bool).any(),
    "E187 transfer table contains no upper claim",
)

shared = e178_shared.set_index("study")
check(
    abs(float(shared.loc["E176 四供体", "task_error_spearman"]) - 0.9746570670998985)
    < 1e-12,
    "Primary scGPT-GEARS error concordance matches E178",
)
check(
    abs(float(shared.loc["E177 独立研究", "task_error_spearman"]) - 0.9924474527965798)
    < 1e-12,
    "Sunshine scGPT-GEARS error concordance matches E178",
)
for study, outcome, expected in (
    ("E176 四供体", "scgpt_rmse", -0.21121550292032953),
    ("E176 四供体", "gears_rmse", -0.17394751795722596),
    ("E177 独立研究", "scgpt_rmse", 0.058573866086663),
    ("E177 独立研究", "gears_rmse", 0.0593557459734123),
):
    row = e178_specific[
        (e178_specific["study"] == study)
        & (e178_specific["score"] == "model_disagreement_rmse")
        & (e178_specific["outcome"] == outcome)
    ]
    check(
        len(row) == 1 and abs(float(row.iloc[0]["spearman_task"]) - expected) < 1e-12,
        f"E178 disagreement association matches for {study} {outcome}",
    )

for study, expected_upper, expected_coverage in (
    ("E176_primary_CD4", 0.206167146968, 0.8941),
    ("E177_Sunshine", 0.538920179132, 0.905),
):
    row = e179_methods[
        (e179_methods["study"] == study)
        & (e179_methods["method"] == "extra_trees_vector")
    ]
    check(
        len(row) == 1
        and abs(float(row.iloc[0]["mean_upper"]) - expected_upper) < 1e-10
        and abs(float(row.iloc[0]["mean_target_coverage"]) - expected_coverage) < 1e-10,
        f"E179 ExtraTrees summary matches for {study}",
    )

prescribe_assoc = e145_associations[
    (e145_associations["scope"] == "two_panel_macro")
    & (e145_associations["score"] == "combined_confidence")
    & (e145_associations["target"] == "pearson_effect_accuracy")
].iloc[0]
check(
    abs(float(prescribe_assoc["spearman_rho"]) - 0.3169565217391304) < 1e-12,
    "PRESCRIBE native-endpoint association matches E145",
)
prescribe_delta = e145_incremental[
    (e145_incremental["scope"] == "two_panel_macro")
    & (e145_incremental["score"] == "combined_confidence")
    & (e145_incremental["target"] == "pearson_effect_accuracy")
].iloc[0]
check(
    abs(float(prescribe_delta["raw_delta_rho"]) - 0.0065217391304348005) < 1e-12,
    "PRESCRIBE increment over magnitude matches E145",
)
prescribe_redundancy = e145_redundancy[
    (e145_redundancy["scope"] == "two_panel_macro")
    & (e145_redundancy["score"] == "combined_confidence")
].iloc[0]
check(
    abs(float(prescribe_redundancy["spearman_rho"]) - 0.9952173913043476) < 1e-12,
    "PRESCRIBE score redundancy matches E145",
)

check("conditional upper events" not in main, "manuscript avoids a conditional-coverage overclaim")
check(
    "no upper-coverage claim was transported across datasets" in main
    and "no conformal upper statement was made" in main,
    "cross-dataset upper-coverage boundary is explicit",
)

for figure in range(1, 6):
    stem = {
        1: "Figure_1_method_and_protocol",
        2: "Figure_2_cross_study_results",
        3: "Figure_3_gse225807_confirmation",
        4: "Figure_4_difficulty_ladder",
        5: "Figure_5_reproducibility_chain",
    }[figure]
    for suffix in ("png", "pdf", "svg"):
        path = HERE / "figures" / f"{stem}.{suffix}"
        check(path.exists() and path.stat().st_size > 5000, f"{stem}.{suffix} exists")

for suffix in ("png", "pdf", "svg"):
    check(
        not (HERE / "figures" / f"Figure_4_reproducibility_chain.{suffix}").exists(),
        f"stale Figure_4_reproducibility_chain.{suffix} is absent",
    )

for document in (
    "SafeConf_manuscript.docx",
    "SafeConf_supplement.docx",
    "Cover_letter_BMC_Bioinformatics.docx",
):
    path = HERE / document
    check(path.exists() and zipfile.is_zipfile(path), f"{document} is a valid Office archive")

with zipfile.ZipFile(HERE / "SafeConf_manuscript.docx") as archive:
    names = archive.namelist()
    media = [name for name in names if name.startswith("word/media/")]
    xml = archive.read("word/document.xml").decode("utf-8")
    footer_xml = "".join(
        archive.read(name).decode("utf-8")
        for name in names
        if name.startswith("word/footer") and name.endswith(".xml")
    )
    check(len(media) == 5, "main Word file embeds all five figures")
    check(xml.count("<m:oMath") >= 20, "main Word file contains editable equations")
    check("<w:lnNumType" in xml, "main Word file enables continuous line numbering")
    check(" PAGE " in footer_xml, "main Word file contains page-number fields")
    check(
        "ref-dixit2016perturbseq" in xml
        and "ref-bai2026pertadapt" in xml
        and "ref-dunn2020random" in xml,
        "Word citations were resolved",
    )
    check("16/20" in xml and "17/20" in xml, "Word file retains the failed registered gate")

required = {
    "SafeConf_manuscript.pdf",
    "SafeConf_supplement.pdf",
    "Cover_letter_BMC_Bioinformatics.pdf",
    "SUBMISSION_CHECKLIST.md",
    "styles/biomed-central.csl",
    "SafeConf_source_data.zip",
}
for relative in sorted(required):
    path = HERE / relative
    check(path.exists() and path.stat().st_size > 0, f"{relative} exists")

expected_zip_tables = {
    "Table_1_study_design.csv",
    "Table_2_certificate_results.csv",
    *(f"Table_S{i}_{suffix}.csv" for i, suffix in (
        (1, "family_comparisons"),
        (2, "gse225807_targets"),
        (3, "gse225807_tasks"),
        (4, "difficulty_setting_summary"),
        (5, "difficulty_macro_bootstrap"),
        (6, "cross_dataset_summary"),
        (7, "difficulty_task_certificates"),
        (8, "cross_dataset_task_certificates"),
        (9, "model_error_concordance"),
        (10, "score_model_specificity"),
        (11, "nested_upper_baselines"),
        (12, "prescribe_native_endpoint"),
        (13, "prescribe_incremental"),
        (14, "prescribe_redundancy"),
    )),
}
with zipfile.ZipFile(HERE / "SafeConf_source_data.zip") as archive:
    check(
        set(archive.namelist()) == expected_zip_tables,
        "source-data archive contains exactly 2 main and 14 supplementary tables",
    )

failures = [message for passed, message in checks if not passed]
if failures:
    print(f"SafeConf submission audit: FAIL ({len(failures)}/{len(checks)} failed)")
    for message in failures:
        print(f"- {message}")
    sys.exit(1)

print(f"SafeConf submission audit: PASS ({len(checks)} checks, 0 failed)")

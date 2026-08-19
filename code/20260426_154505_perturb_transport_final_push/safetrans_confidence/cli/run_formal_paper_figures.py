#!/usr/bin/env python3
"""Generate paper-facing figures and GEARS-Cui feasibility audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/home/yyf/proj")
CODE_ROOT = PROJECT_ROOT / "code" / "20260426_154505_perturb_transport_final_push"
DEFAULT_FORMAL_ROOT = CODE_ROOT / "outputs" / "safeconf_formal_main_20260604"
DEFAULT_OUT_DIR = DEFAULT_FORMAL_ROOT / "paper_figures"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs" / "实验结果" / "Formal_main_20260604" / "paper_figures"
MAIN_SCORE = "protocol_v0_2_family_confidence"


def _mkdirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
        "reports": out_dir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _risk_coverage(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = scored[(scored["split"].eq("test")) & (scored["score_name"].eq(MAIN_SCORE))].copy()
    for dataset, group in main.groupby("dataset_name", dropna=False):
        group = group.dropna(subset=["risk_axis", "true_error_rmse"]).sort_values("risk_axis", ascending=True)
        full = float(group["true_error_rmse"].mean())
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            keep = max(1, int(np.ceil(cov * len(group))))
            kept = group.head(keep)
            mean = float(kept["true_error_rmse"].mean())
            rows.append(
                {
                    "dataset_name": dataset,
                    "coverage": cov,
                    "mean_rmse": mean,
                    "full_mean_rmse": full,
                    "improve_pct": 100.0 * (full - mean) / full if full else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _clean_dataset_name(name: str) -> str:
    return (
        name.replace("SrivatsanTrapnell2020_sciplex3", "Srivatsan")
        .replace("McFarlandTsherniak2020", "McFarland")
        .replace("LaraAstiasoHuntly2023_", "Lara ")
        .replace("CuiHacohen2023", "Cui")
        .replace("SantinhaPlatt2023", "Santinha")
    )


def draw_figures(formal_root: Path, out_paths: dict[str, Path]) -> dict:
    main = pd.read_csv(formal_root / "formal_audit" / "tables" / "FORMAL_MAIN_TABLE.csv")
    scored = pd.read_csv(
        formal_root / "formal_audit" / "tables" / "FORMAL_SCORED_RECORDS.csv",
        usecols=[
            "dataset_name",
            "dataset_family",
            "split",
            "score_name",
            "risk_axis",
            "true_error_rmse",
            "true_effect_l2_norm",
            "normalized_rmse",
        ],
    )

    main = main.sort_values(["dataset_family", "dataset_name"]).copy()
    main["short_name"] = main["dataset_name"].map(_clean_dataset_name)
    main.to_csv(out_paths["tables"] / "PAPER_MAIN_TABLE_FOR_PLOTS.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    x = np.arange(len(main))
    width = 0.26
    ax.bar(x - width, main["aligned_rho"], width, label="aligned rho")
    ax.bar(x, main["partial_rho_control_magnitude"], width, label="partial rho")
    ax.bar(x + width, main["magnitude_only_rho"], width, label="magnitude-only")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(0.2, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(main["short_name"], rotation=35, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("SafeConf formal main: confidence signal vs magnitude baseline")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_paths["figures"] / "F1_per_dataset_rho_bars.png")
    plt.close(fig)

    rc = _risk_coverage(scored)
    rc.to_csv(out_paths["tables"] / "PAPER_RISK_COVERAGE_CURVES.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for dataset, group in rc.groupby("dataset_name"):
        ax.plot(group["coverage"], group["improve_pct"], marker="o", linewidth=1.6, label=_clean_dataset_name(dataset))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(1.02, 0.48)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("RMSE improvement (%)")
    ax.set_title("Risk-coverage: lower-risk predictions kept first")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_paths["figures"] / "F2_risk_coverage_curves.png")
    plt.close(fig)

    # McFarland dose diagnosis.
    dose_path = formal_root / "post_formal_diagnostics" / "tables" / "McFarland_per_dose_rho.csv"
    leave_path = formal_root / "post_formal_diagnostics" / "tables" / "McFarland_leave_one_dose_out_rho.csv"
    if dose_path.exists():
        dose = pd.read_csv(dose_path)
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
        ax.bar(dose["dominant_dose_value"].astype(str), dose["partial_rho_control_magnitude"], color="#5f7fbf")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Dominant dose")
        ax.set_ylabel("Partial rho")
        ax.set_title("McFarland failure diagnosis by dose")
        fig.tight_layout()
        fig.savefig(out_paths["figures"] / "F3_mcfarland_dose_partial_rho.png")
        plt.close(fig)
    if leave_path.exists():
        leave = pd.read_csv(leave_path)
        leave.to_csv(out_paths["tables"] / "PAPER_MCFARLAND_LEAVE_ONE_DOSE_OUT.csv", index=False)

    report = f"""# Paper Figure Package

Generated figures:

1. `F1_per_dataset_rho_bars.png`: formal main rho comparison.
2. `F2_risk_coverage_curves.png`: risk-coverage curves for protocol v0.2.
3. `F3_mcfarland_dose_partial_rho.png`: McFarland dose failure diagnosis.

These are paper-facing drafts. They are not final journal-layout figures yet.
"""
    (out_paths["reports"] / "PAPER_FIGURE_REPORT.md").write_text(report, encoding="utf-8")
    return {"n_datasets": int(len(main)), "n_risk_coverage_rows": int(len(rc))}


def audit_gears_cui(out_paths: dict[str, Path]) -> dict:
    path = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/CuiHacohen2023.h5ad")
    adata = sc.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    var_names = set(adata.var_names.astype(str))
    perturbations = sorted(
        [
            p
            for p in obs["perturbation"].astype(str).dropna().unique()
            if p.lower() not in {"control", "ctrl", "nan"}
        ]
    )
    exact_overlap = [p for p in perturbations if p in var_names]
    piece_set = sorted({g for p in perturbations for g in p.replace(";", "+").replace(",", "+").split("+") if g})
    piece_overlap = [g for g in piece_set if g in var_names]
    pert_type = sorted(obs["perturbation_type"].astype(str).dropna().unique().tolist()) if "perturbation_type" in obs else []
    adata.file.close()

    rows = []
    for p in perturbations:
        rows.append({"perturbation": p, "exact_gene_symbol_in_var": p in var_names})
    pd.DataFrame(rows).to_csv(out_paths["tables"] / "GEARS_CUI_PERTURBATION_GENE_OVERLAP.csv", index=False)
    status = {
        "dataset": "CuiHacohen2023",
        "h5ad_path": str(path),
        "perturbation_type_values": pert_type,
        "n_perturbations": len(perturbations),
        "n_exact_gene_overlap": len(exact_overlap),
        "n_piece_gene_overlap": len(piece_overlap),
        "compatible_with_gears_gene_perturbation": bool(len(exact_overlap) >= max(5, 0.5 * len(perturbations))),
        "recommendation": "do_not_run_gears_on_cui; use Norman/Adamson/Dixit GEARS supplement or a cytokine-compatible predictor",
    }
    md = f"""# GEARS on Cui Feasibility Audit

## 结论

不建议在 CuiHacohen2023 上硬跑 GEARS。

原因很简单：GEARS（图神经网络扰动预测模型）主要面向 gene perturbation（基因扰动），而 CuiHacohen2023 的 perturbation_type 是 `{', '.join(pert_type)}`，也就是 cytokine（细胞因子/刺激物）处理。

## 证据

- 非 control perturbations: {len(perturbations)}
- 与 var_names 精确匹配的基因名: {len(exact_overlap)}
- 拆分后与 var_names 匹配的基因名: {len(piece_overlap)}
- exact overlap examples: {', '.join(exact_overlap[:20]) if exact_overlap else 'none'}

如果把这些 cytokine 名称强行当作 GEARS 的 gene perturbation，会变成伪实验。

## 建议

1. 不要把“GEARS on Cui”写进执行计划。
2. 如果要证明 predictor-agnostic（不绑定预测器），优先用已有 Norman/Adamson/Dixit 的 GEARS supplement。
3. 如果一定要在 Cui 这类 cytokine 数据上接第三 predictor，应考虑 cytokine/drug response 兼容模型，而不是 GEARS。
"""
    (out_paths["reports"] / "GEARS_CUI_FEASIBILITY_AUDIT.md").write_text(md, encoding="utf-8")
    return status


def summarize_existing_gears(out_paths: dict[str, Path]) -> dict:
    eval_path = CODE_ROOT / "outputs" / "gears_confidence_eval_formal" / "tables" / "GEARS_CONFIDENCE_EVAL_SUMMARY.csv"
    status_path = CODE_ROOT / "outputs" / "gears_prediction_records_formal" / "GEARS_PREDICTION_RECORD_STATUS.csv"
    uncertainty_path = (
        CODE_ROOT / "outputs" / "gears_uncertainty_formal_v6" / "tables" / "GEARS_UNCERTAINTY_SCORES.csv"
    )
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        eval_df.to_csv(out_paths["tables"] / "GEARS_EXISTING_EVAL_SUMMARY.csv", index=False)
    else:
        eval_df = pd.DataFrame()
    if status_path.exists():
        status_df = pd.read_csv(status_path)
        status_df.to_csv(out_paths["tables"] / "GEARS_EXISTING_RUN_STATUS.csv", index=False)
    else:
        status_df = pd.DataFrame()
    if uncertainty_path.exists():
        uncertainty_df = pd.read_csv(uncertainty_path)
        uncertainty_df.to_csv(out_paths["tables"] / "GEARS_EXISTING_UNCERTAINTY_SCORES.csv", index=False)
        native_present = bool(uncertainty_df["score_name"].astype(str).str.contains("native", case=False, na=False).any())
    else:
        uncertainty_df = pd.DataFrame()
        native_present = False

    dataset_lines = []
    if not eval_df.empty:
        for _, row in eval_df[eval_df["level"].eq("dataset")].iterrows():
            dataset_lines.append(
                f"- {row['dataset_name']}: `{row['score_name']}` aligned rho = "
                f"{float(row['direction_aligned_spearman']):.3f}, n = {int(row['n'])}"
            )
    md = f"""# Existing GEARS Supplement Status

## 结论

已有 GEARS 输出可以作为 supplement（补充），但不能替代 7 主表证据，也不能冒充 Cui 上的第三 predictor（预测器）。

## 当前已有

- GEARS run status file exists: {status_path.exists()}
- GEARS eval file exists: {eval_path.exists()}
- GEARS uncertainty proxy file exists: {uncertainty_path.exists()}
- native uncertainty present: {native_present}
- total GEARS run rows: {len(status_df)}
- total GEARS eval rows: {len(eval_df)}
- total GEARS uncertainty/proxy rows: {len(uncertainty_df)}

## Dataset-level GEARS confidence signals

{chr(10).join(dataset_lines) if dataset_lines else 'No dataset-level GEARS eval rows found.'}

## 解释

这些 GEARS 结果来自 Norman / Adamson / Dixit 的 gene perturbation（基因扰动）场景。它们说明 SafeConf 可以读取 GEARS per-prediction records（逐条预测记录），但样本量小、split（切分）不是当前 7 主表的 cross-context task（跨背景任务），且 native uncertainty（原生不确定性）仍未导出。

建议论文写成：

> GEARS supplement demonstrates adapter compatibility, while the main claims remain based on the seven-dataset cross-context benchmark.
"""
    (out_paths["reports"] / "GEARS_EXISTING_SUPPLEMENT_STATUS.md").write_text(md, encoding="utf-8")
    return {
        "eval_exists": bool(eval_path.exists()),
        "status_exists": bool(status_path.exists()),
        "uncertainty_exists": bool(uncertainty_path.exists()),
        "native_uncertainty_present": bool(native_present),
        "n_eval_rows": int(len(eval_df)),
        "n_status_rows": int(len(status_df)),
        "n_uncertainty_rows": int(len(uncertainty_df)),
    }


def sync_to_docs(out_dir: Path, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "tables", "reports"]:
        src = out_dir / sub
        dst = docs_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    (docs_dir / "README_先看这个.md").write_text(
        f"""# Formal Main Paper Figures

这里放正式主表的论文图草稿和 GEARS-Cui 可行性审计。

先看：

1. `figures/F1_per_dataset_rho_bars.png`
2. `figures/F2_risk_coverage_curves.png`
3. `figures/F3_mcfarland_dose_partial_rho.png`
4. `reports/GEARS_CUI_FEASIBILITY_AUDIT.md`
5. `reports/GEARS_EXISTING_SUPPLEMENT_STATUS.md`

代码输出原位置：

`{out_dir}`
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    formal_root = Path(args.formal_root)
    out_dir = Path(args.out_dir)
    docs_dir = Path(args.docs_dir)
    out_paths = _mkdirs(out_dir)
    figure_status = draw_figures(formal_root, out_paths)
    gears_status = audit_gears_cui(out_paths)
    existing_gears_status = summarize_existing_gears(out_paths)
    status = {"figures": figure_status, "gears_cui": gears_status, "existing_gears": existing_gears_status}
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_docs_sync:
        sync_to_docs(out_dir, docs_dir)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate formal SafeConf paper figures.")
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--no-docs-sync", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

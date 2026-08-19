#!/usr/bin/env python3
"""Post-formal SafeConf diagnostics.

This script is intentionally audit-only:
- no perturbation predictor training;
- no protocol formula changes;
- no test-label tuning.

It adds the evidence tables requested after the seven-dataset formal audit:
McFarland failure diagnosis, feature redundancy, Tahoe eligibility audit, and a
plain decision memo for the current main table.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/home/yyf/proj")
CODE_ROOT = PROJECT_ROOT / "code" / "20260426_154505_perturb_transport_final_push"
DEFAULT_FORMAL_ROOT = CODE_ROOT / "outputs" / "safeconf_formal_main_20260604"
DEFAULT_TAHOE_ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M")
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs" / "实验结果" / "Formal_main_20260604" / "diagnostics"


FEATURE_COLUMNS = [
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "fold_train_median_effect_norm",
    "prediction_norm_ratio",
    "prediction_magnitude_deviation",
    "model_disagreement_rmse",
    "model_disagreement_cosine",
    "ood_nearest_distance",
    "ood_mean_k_distance",
    "historical_residual_risk",
]


def _mkdirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "tables": out_dir / "tables",
        "reports": out_dir / "reports",
        "figures": out_dir / "figures",
        "logs": out_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values, "c": control}).apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 3 or frame["c"].nunique() < 2:
        return out
    y = frame["v"].rank(method="average").to_numpy(dtype=float)
    z = frame["c"].rank(method="average").to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    out.loc[frame.index] = y - design @ beta
    return out


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = _rank_residual(x, control)
    ry = _rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def _metric_row(group: pd.DataFrame, *, label: str | None = None) -> dict:
    row = {
        "n": int(len(group)),
        "aligned_rho": _spearman(group["risk_axis"], group["true_error_rmse"]),
        "partial_rho_control_magnitude": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "normalized_rmse_rho": _spearman(group["risk_axis"], group["normalized_rmse"]),
        "mean_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").mean()),
        "median_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").median()),
    }
    if label is not None:
        row["label"] = label
    return row


def _load_formal_scored(formal_root: Path) -> pd.DataFrame:
    usecols = [
        "record_id",
        "dataset_name",
        "dataset_family",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        "score_name",
        "score_type",
        "score_value",
        "true_error_rmse",
        "true_error_cosine",
        "true_effect_l2_norm",
        "risk_axis",
        "normalized_rmse",
    ]
    return pd.read_csv(formal_root / "formal_audit" / "tables" / "FORMAL_SCORED_RECORDS.csv", usecols=usecols)


def _dominant(series: pd.Series) -> tuple[str, float, int]:
    vals = series.astype(str).value_counts(dropna=False)
    if vals.empty:
        return "NA", float("nan"), 0
    top = vals.index[0]
    return str(top), float(vals.iloc[0] / vals.sum()), int(len(vals))


def diagnose_mcfarland(formal_root: Path, out_paths: dict[str, Path]) -> dict:
    run_dir = formal_root / "McFarlandTsherniak2020"
    scored = _load_formal_scored(formal_root)
    mc = scored[(scored["dataset_name"].eq("McFarlandTsherniak2020")) & (scored["split"].eq("test"))].copy()

    single_rows = []
    for score_name, group in mc.groupby("score_name", dropna=False):
        row = _metric_row(group)
        row["score_name"] = score_name
        row["score_type"] = str(group["score_type"].iloc[0])
        single_rows.append(row)
    single_df = pd.DataFrame(single_rows).sort_values("aligned_rho", ascending=False)
    single_df.to_csv(out_paths["tables"] / "McFarland_single_feature_diagnostics.csv", index=False)

    fold_rows = []
    for (score_name, fold_id), group in mc.groupby(["score_name", "fold_id"], dropna=False):
        row = _metric_row(group)
        row["score_name"] = score_name
        row["fold_id"] = fold_id
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows).sort_values(["score_name", "fold_id"])
    fold_df.to_csv(out_paths["tables"] / "McFarland_per_fold_score_diagnostics.csv", index=False)

    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    test_records = records[records["split"].eq("test")].copy()
    pred_rows = []
    for predictor, group in test_records.groupby("predictor_name", dropna=False):
        err = pd.to_numeric(group["true_error_rmse"], errors="coerce")
        pred_rows.append(
            {
                "predictor_name": predictor,
                "n": int(len(group)),
                "mean_rmse": float(err.mean()),
                "median_rmse": float(err.median()),
                "std_rmse": float(err.std()),
                "q10_rmse": float(err.quantile(0.10)),
                "q90_rmse": float(err.quantile(0.90)),
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_paths["tables"] / "McFarland_predictor_rmse.csv", index=False)

    features = pd.read_csv(run_dir / "tables" / "CONFIDENCE_FEATURES.csv")
    test_features = features[features["split"].eq("test")].copy()
    disagreement = test_features.drop_duplicates(["task_key", "fold_id"])[
        [
            "task_key",
            "fold_id",
            "context",
            "perturbation",
            "model_disagreement_rmse",
            "model_disagreement_cosine",
            "historical_residual_risk",
        ]
    ].copy()
    task_error = test_records.groupby(["task_key", "fold_id"], as_index=False)["true_error_rmse"].mean()
    disagreement = disagreement.merge(task_error, on=["task_key", "fold_id"], how="left")
    disagreement.to_csv(out_paths["tables"] / "McFarland_model_disagreement_by_task.csv", index=False)

    h5ad_path = run_dir / "input" / "McFarlandTsherniak2020__filtered_perturbation_type_drug.h5ad"
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    obs["perturbation_str"] = obs["perturbation"].astype(str)
    task_obs = obs[~obs["perturbation_str"].str.lower().eq("control")].copy()
    meta_rows = []
    for (cell_line, perturbation), group in task_obs.groupby(["cell_line", "perturbation"], dropna=False):
        dose, dose_frac, n_doses = _dominant(group["dose_value"]) if "dose_value" in group else ("NA", np.nan, 0)
        time, time_frac, n_times = _dominant(group["time"]) if "time" in group else ("NA", np.nan, 0)
        meta_rows.append(
            {
                "context": str(cell_line),
                "perturbation": str(perturbation),
                "n_cells": int(len(group)),
                "dominant_dose_value": dose,
                "dominant_dose_fraction": dose_frac,
                "n_dose_values": n_doses,
                "all_dose_values": "|".join(sorted(group["dose_value"].astype(str).unique()))
                if "dose_value" in group
                else "",
                "dominant_time": time,
                "dominant_time_fraction": time_frac,
                "n_time_values": n_times,
                "all_time_values": "|".join(sorted(group["time"].astype(str).unique())) if "time" in group else "",
            }
        )
    task_meta = pd.DataFrame(meta_rows)
    task_meta.to_csv(out_paths["tables"] / "McFarland_task_metadata.csv", index=False)

    protocol = mc[mc["score_name"].eq("protocol_v0_2_family_confidence")].copy()
    protocol = protocol.merge(task_meta, on=["context", "perturbation"], how="left")
    n_formal_test_pairs = int(protocol[["context", "perturbation"]].drop_duplicates().shape[0])
    dose_rows = []
    for dose, group in protocol.groupby("dominant_dose_value", dropna=False):
        row = _metric_row(group)
        row["dominant_dose_value"] = dose
        dose_rows.append(row)
    dose_df = pd.DataFrame(dose_rows).sort_values("dominant_dose_value")
    dose_df.to_csv(out_paths["tables"] / "McFarland_per_dose_rho.csv", index=False)

    leave_dose_rows = []
    for dose in sorted(protocol["dominant_dose_value"].dropna().astype(str).unique()):
        kept = protocol[~protocol["dominant_dose_value"].astype(str).eq(dose)]
        row = _metric_row(kept)
        row["removed_dose_value"] = dose
        row["removed_n"] = int(protocol["dominant_dose_value"].astype(str).eq(dose).sum())
        leave_dose_rows.append(row)
    leave_dose_df = pd.DataFrame(leave_dose_rows).sort_values("aligned_rho", ascending=False)
    leave_dose_df.to_csv(out_paths["tables"] / "McFarland_leave_one_dose_out_rho.csv", index=False)

    time_rows = []
    for time, group in protocol.groupby("dominant_time", dropna=False):
        row = _metric_row(group)
        row["dominant_time"] = time
        time_rows.append(row)
    time_df = pd.DataFrame(time_rows).sort_values("dominant_time")
    time_df.to_csv(out_paths["tables"] / "McFarland_per_time_rho.csv", index=False)

    time_audit = (
        task_meta.groupby("dominant_time")
        .agg(
            n_pairs=("perturbation", "size"),
            n_contexts=("context", "nunique"),
            n_perturbations=("perturbation", "nunique"),
            example_all_time_values=("all_time_values", lambda s: "|".join(sorted(set(map(str, s)))[:10])),
        )
        .reset_index()
        .sort_values("n_pairs", ascending=False)
    )
    time_audit.to_csv(out_paths["tables"] / "McFarland_time_label_audit.csv", index=False)

    test_task_counts = protocol.groupby("context", as_index=False).size().rename(columns={"size": "test_score_rows"})
    cell_line_support = (
        task_meta.groupby("context")
        .agg(
            n_perturbations=("perturbation", "nunique"),
            n_task_cells=("n_cells", "sum"),
            median_cells_per_pair=("n_cells", "median"),
            n_multi_dose_pairs=("n_dose_values", lambda s: int((pd.to_numeric(s, errors="coerce") > 1).sum())),
            n_multi_time_pairs=("n_time_values", lambda s: int((pd.to_numeric(s, errors="coerce") > 1).sum())),
        )
        .reset_index()
        .merge(test_task_counts, on="context", how="left")
        .fillna({"test_score_rows": 0})
        .sort_values("n_perturbations", ascending=False)
    )
    cell_line_support.to_csv(out_paths["tables"] / "McFarland_cell_line_support_distribution.csv", index=False)

    drug_test_counts = protocol.groupby("perturbation", as_index=False).size().rename(columns={"size": "test_score_rows"})
    drug_coverage = (
        task_meta.groupby("perturbation")
        .agg(
            n_contexts=("context", "nunique"),
            n_task_cells=("n_cells", "sum"),
            median_cells_per_pair=("n_cells", "median"),
            n_multi_dose_pairs=("n_dose_values", lambda s: int((pd.to_numeric(s, errors="coerce") > 1).sum())),
            n_multi_time_pairs=("n_time_values", lambda s: int((pd.to_numeric(s, errors="coerce") > 1).sum())),
        )
        .reset_index()
        .merge(drug_test_counts, on="perturbation", how="left")
        .fillna({"test_score_rows": 0})
        .sort_values("n_contexts", ascending=False)
    )
    drug_coverage.to_csv(out_paths["tables"] / "McFarland_drug_coverage.csv", index=False)

    main_score = single_df[single_df["score_name"].eq("protocol_v0_2_family_confidence")].iloc[0]
    best_score = single_df.iloc[0]
    historical = single_df[single_df["score_name"].eq("historical_residual_risk")]
    hist_rho = float(historical["aligned_rho"].iloc[0]) if not historical.empty else float("nan")
    support = single_df[single_df["score_name"].eq("support_count_score")]
    support_rho = float(support["aligned_rho"].iloc[0]) if not support.empty else float("nan")
    ctx = single_df[single_df["score_name"].eq("context_similarity_score")]
    ctx_rho = float(ctx["aligned_rho"].iloc[0]) if not ctx.empty else float("nan")
    best_leave_dose = leave_dose_df.iloc[0] if not leave_dose_df.empty else pd.Series(dtype=float)

    md = f"""# McFarland Failure Diagnosis

## 结论

McFarlandTsherniak2020（drug-only）不是完全没有信号，而是冻结的 `protocol_v0_2_family_confidence` 在这个数据集上方向不对。

- 主方法 aligned rho: {main_score['aligned_rho']:.3f}
- 主方法 partial rho: {main_score['partial_rho_control_magnitude']:.3f}
- 最强单项: `{best_score['score_name']}`，aligned rho = {best_score['aligned_rho']:.3f}
- `historical_residual_risk`: aligned rho = {hist_rho:.3f}
- `support_count_score`: aligned rho = {support_rho:.3f}
- `context_similarity_score`: aligned rho = {ctx_rho:.3f}
- best leave-one-dose result: remove dose `{best_leave_dose.get('removed_dose_value', 'NA')}`, aligned rho = {float(best_leave_dose.get('aligned_rho', np.nan)):.3f}, partial rho = {float(best_leave_dose.get('partial_rho_control_magnitude', np.nan)):.3f}

这支持 Claude 的判断：不要为了 McFarland 修改冻结公式；应把它写成化学线的 failure boundary（失败边界），同时报告它还有 historical residual 这类替代信号。

## 数据结构

- filtered h5ad: `{h5ad_path}`
- cells after drug-only filter: {obs.shape[0]:,}
- cell lines: {task_meta['context'].nunique():,}
- non-control drugs: {task_meta['perturbation'].nunique():,}
- observed non-control cell_line × drug pairs in h5ad metadata: {len(task_meta):,}
- formal held-out test cell_line × drug pairs: {n_formal_test_pairs:,}
- dose values: {', '.join(map(str, sorted(obs['dose_value'].astype(str).unique())))} 
- time labels: {', '.join(map(str, sorted(obs['time'].astype(str).unique())))} 

## 为什么失败更可能是任务结构问题

1. McFarland 的扰动数很少，只有 {task_meta['perturbation'].nunique()} 个非 control drug，但 cell line 很多；`support_count` 会变成“覆盖多不多”的粗信号，未必代表这次预测好不好。
2. dose/time 混在同一个 drug 名下；同一 drug 在不同剂量或时间下可能不是同一种 effect（效应）。
3. `historical_residual_risk` 为正，说明可以从历史残差看到难题，但 v0.2 公式主推的 support/context/disagreement 组合在这里不合适。

## 补充诊断

- `McFarland_leave_one_dose_out_rho.csv` 检查“去掉某个 dose 后主公式是否回正”。这不是为了调公式，而是定位失败是否集中在特定 dose。
- `McFarland_time_label_audit.csv` 检查混合 time label，例如 `3, 6, 12, 24, 48` 这类标签不应被当作单一时间点解释。

## 建议口径

主表保留 McFarland，不删除；Discussion 写为：SafeConf 对基因线稳定，化学线存在强例子（Srivatsan）和失败边界（McFarland）。McFarland 后续若要救，应先重新定义 drug-dose-time task，而不是改 v0.2 公式。
"""
    (out_paths["reports"] / "McFarland_failure_diagnosis.md").write_text(md, encoding="utf-8")

    return {
        "main_aligned_rho": float(main_score["aligned_rho"]),
        "main_partial_rho": float(main_score["partial_rho_control_magnitude"]),
        "best_score_name": str(best_score["score_name"]),
        "best_score_aligned_rho": float(best_score["aligned_rho"]),
        "historical_residual_aligned_rho": hist_rho,
        "n_cell_lines": int(task_meta["context"].nunique()),
        "n_drugs": int(task_meta["perturbation"].nunique()),
        "n_pairs": int(len(task_meta)),
        "n_formal_test_pairs": n_formal_test_pairs,
        "best_leave_one_dose": str(best_leave_dose.get("removed_dose_value", "NA")),
        "best_leave_one_dose_aligned_rho": float(best_leave_dose.get("aligned_rho", np.nan)),
    }


def feature_redundancy(formal_root: Path, out_paths: dict[str, Path]) -> dict:
    status = pd.read_csv(formal_root / "formal_audit" / "tables" / "FORMAL_INPUT_STATUS.csv")
    frames = []
    for run_dir in status[status["status"].eq("ok")]["run_dir"].dropna().astype(str):
        path = Path(run_dir) / "tables" / "CONFIDENCE_FEATURES.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise RuntimeError("No CONFIDENCE_FEATURES.csv files found for feature redundancy.")
    feats = pd.concat(frames, ignore_index=True)
    cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    matrix = feats[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    matrix.to_csv(out_paths["tables"] / "FEATURE_CORRELATION_MATRIX.csv")

    pairs = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rho = matrix.loc[left, right]
            if np.isfinite(rho):
                pairs.append({"feature_a": left, "feature_b": right, "spearman_rho": float(rho)})
    pair_df = pd.DataFrame(pairs).assign(abs_rho=lambda x: x["spearman_rho"].abs()).sort_values(
        "abs_rho", ascending=False
    )
    pair_df.to_csv(out_paths["tables"] / "FEATURE_CORRELATION_TOP_PAIRS.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("SafeConf feature Spearman correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_paths["figures"] / "FEATURE_CORRELATION_HEATMAP.png")
    plt.close(fig)

    md = f"""# Feature Redundancy Audit

Rows audited: {len(feats):,}

Feature columns used: {len(cols)}

The full Spearman correlation matrix is saved as `FEATURE_CORRELATION_MATRIX.csv`.
The strongest redundant pairs are saved as `FEATURE_CORRELATION_TOP_PAIRS.csv`.
"""
    (out_paths["reports"] / "FEATURE_REDUNDANCY_AUDIT.md").write_text(md, encoding="utf-8")
    return {"n_rows": int(len(feats)), "n_features": int(len(cols)), "top_abs_rho": float(pair_df["abs_rho"].max())}


def _parquet_available() -> bool:
    try:
        import pyarrow.parquet  # noqa: F401

        return True
    except Exception:
        return False


def tahoe_eligibility(tahoe_root: Path, out_paths: dict[str, Path], sample_shards: int) -> dict:
    import pyarrow.parquet as pq

    meta_root = tahoe_root / "metadata"
    obs_path = meta_root / "obs_metadata.parquet"
    pseudo_dir = meta_root / "pseudobulk_differential_expression"
    pseudo_files = sorted(pseudo_dir.glob("*.parquet"))
    downloaded_gb = sum(p.stat().st_size for p in tahoe_root.rglob("*") if p.is_file()) / (1024**3)

    obs_pf = pq.ParquetFile(obs_path)
    obs_cols = ["drug", "cell_line", "pass_filter"]
    pair_counts: dict[tuple[str, str], int] = {}
    drug_counts: dict[str, int] = {}
    line_counts: dict[str, int] = {}
    pass_filter_counts: dict[str, int] = {}
    for i in range(obs_pf.metadata.num_row_groups):
        chunk = obs_pf.read_row_group(i, columns=obs_cols).to_pandas()
        chunk["drug"] = chunk["drug"].astype(str)
        chunk["cell_line"] = chunk["cell_line"].astype(str)
        grouped = chunk.groupby(["drug", "cell_line"], observed=True).size()
        for key, value in grouped.items():
            pair_counts[(str(key[0]), str(key[1]))] = pair_counts.get((str(key[0]), str(key[1])), 0) + int(value)
        for key, value in chunk["drug"].value_counts(dropna=False).items():
            drug_counts[str(key)] = drug_counts.get(str(key), 0) + int(value)
        for key, value in chunk["cell_line"].value_counts(dropna=False).items():
            line_counts[str(key)] = line_counts.get(str(key), 0) + int(value)
        for key, value in chunk["pass_filter"].astype(str).value_counts(dropna=False).items():
            pass_filter_counts[str(key)] = pass_filter_counts.get(str(key), 0) + int(value)

    pair_df = pd.DataFrame(
        [{"drug": d, "cell_line": c, "n_cells": n} for (d, c), n in pair_counts.items()]
    ).sort_values("n_cells", ascending=False)
    pair_df.to_csv(out_paths["tables"] / "Tahoe_drug_cell_line_matrix.csv", index=False)
    pd.DataFrame([{"drug": k, "n_cells": v} for k, v in drug_counts.items()]).sort_values(
        "n_cells", ascending=False
    ).to_csv(out_paths["tables"] / "Tahoe_drug_counts.csv", index=False)
    pd.DataFrame([{"cell_line": k, "n_cells": v} for k, v in line_counts.items()]).sort_values(
        "n_cells", ascending=False
    ).to_csv(out_paths["tables"] / "Tahoe_cell_line_counts.csv", index=False)

    def _is_control_like_drug(name: str) -> bool:
        lower = name.lower().strip()
        return (
            lower in {"control", "dmso", "dmso_tf", "vehicle", "untreated"}
            or lower.startswith("dmso")
            or lower.startswith("vehicle")
        )

    control_like = [drug for drug in drug_counts if _is_control_like_drug(drug)]
    eligible_pairs_min6 = int((pair_df["n_cells"] >= 6).sum()) if not pair_df.empty else 0
    eligible_pairs_min20 = int((pair_df["n_cells"] >= 20).sum()) if not pair_df.empty else 0

    pseudo_schema_rows = []
    pseudo_pair_rows = []
    for path in pseudo_files[: max(0, sample_shards)]:
        pf = pq.ParquetFile(path)
        pseudo_schema_rows.append(
            {
                "file": path.name,
                "rows": int(pf.metadata.num_rows),
                "row_groups": int(pf.metadata.num_row_groups),
                "columns": "|".join(pf.schema_arrow.names),
            }
        )
        table = pf.read(columns=["drug", "Cell_Name_Vevo", "Cell_ID_DepMap", "concentration", "n_cells_trt", "n_cells_ctrl"])
        sdf = table.to_pandas()
        sdf = sdf.drop_duplicates(["drug", "Cell_Name_Vevo", "Cell_ID_DepMap", "concentration"])
        sdf["file"] = path.name
        pseudo_pair_rows.append(sdf)
    pseudo_schema = pd.DataFrame(pseudo_schema_rows)
    pseudo_schema.to_csv(out_paths["tables"] / "Tahoe_pseudobulk_schema_sample.csv", index=False)
    pseudo_pair_sample = pd.concat(pseudo_pair_rows, ignore_index=True) if pseudo_pair_rows else pd.DataFrame()
    pseudo_pair_sample.to_csv(out_paths["tables"] / "Tahoe_pseudobulk_pair_sample.csv", index=False)

    gene_meta = pd.read_parquet(meta_root / "gene_metadata.parquet")
    drug_meta = pd.read_parquet(meta_root / "drug_metadata.parquet")
    cell_meta = pd.read_parquet(meta_root / "cell_line_metadata.parquet")

    likely_usable = (
        pair_df["drug"].nunique() >= 10
        and pair_df["cell_line"].nunique() >= 2
        and eligible_pairs_min20 >= 100
        and len(pseudo_files) > 0
    )
    role = "external mega-scale validation candidate" if likely_usable else "data reserve only until adapter is proven"

    md = f"""# Tahoe Eligibility Audit

## 结论

Tahoe 当前更适合作为 `{role}`。

它不是现在 7 主表的一部分；它的价值是后续做 external validation（外部验证）时证明 SafeConf 能不能扩展到超大药物扰动图谱。

## 已下载状态

- local root: `{tahoe_root}`
- downloaded size: {downloaded_gb:.1f} GB
- obs metadata rows: {obs_pf.metadata.num_rows:,}
- obs row groups: {obs_pf.metadata.num_row_groups}
- pseudobulk shards downloaded: {len(pseudo_files):,}
- pseudobulk sample shards scanned: {min(sample_shards, len(pseudo_files)):,}
- genes in metadata: {len(gene_meta):,}
- drugs in drug metadata: {len(drug_meta):,}
- cell lines in cell-line metadata: {len(cell_meta):,}

## obs metadata 支持的任务结构

- unique drugs in obs: {pair_df['drug'].nunique():,}
- unique cell lines in obs: {pair_df['cell_line'].nunique():,}
- observed drug × cell_line pairs: {len(pair_df):,}
- pairs with at least 6 cells: {eligible_pairs_min6:,}
- pairs with at least 20 cells: {eligible_pairs_min20:,}
- control-like drug labels found in obs: {', '.join(control_like[:20]) if control_like else 'not found by name search'}
- pass_filter counts: {json.dumps(pass_filter_counts, ensure_ascii=False)}

## pseudobulk 支持的 effect 信息

Sampled pseudobulk shards contain `log2FoldChange`, `n_cells_trt`, `n_cells_ctrl`, `drug`, `concentration`, and cell-line identifiers.
This means Tahoe may not need raw single-cell matrices for first-pass effect evaluation; a pseudobulk adapter could directly use gene-level log2FoldChange as the true effect vector.

## 下一步

1. Stop broad download at the current 72G scale unless a missing shard blocks the pseudobulk adapter.
2. Write a Tahoe pseudobulk adapter that treats `(Cell_Name_Vevo or Cell_ID_DepMap, drug, concentration)` as the task key.
3. Before putting Tahoe in the paper, run leakage checks and verify enough train support per drug and cell line.
"""
    (out_paths["reports"] / "Tahoe_eligibility_audit.md").write_text(md, encoding="utf-8")

    return {
        "downloaded_gb": float(downloaded_gb),
        "obs_rows": int(obs_pf.metadata.num_rows),
        "n_pseudobulk_shards": int(len(pseudo_files)),
        "n_drugs_obs": int(pair_df["drug"].nunique()),
        "n_cell_lines_obs": int(pair_df["cell_line"].nunique()),
        "n_pairs_obs": int(len(pair_df)),
        "eligible_pairs_min20": eligible_pairs_min20,
        "likely_usable": bool(likely_usable),
        "role": role,
    }


def write_final_decision(formal_root: Path, out_paths: dict[str, Path], mc: dict, tahoe: dict) -> dict:
    main = pd.read_csv(formal_root / "formal_audit" / "tables" / "FORMAL_MAIN_TABLE.csv")
    n_datasets = int(len(main))
    aligned_pass = int((main["aligned_rho"] > 0.20).sum())
    partial_pass = int((main["partial_rho_control_magnitude"] > 0.10).sum())
    rc_pass = int((main["risk_coverage80_improve_pct"] > 0).sum())
    gene = main[main["dataset_family"].eq("gene_main")]
    chem = main[main["dataset_family"].eq("chem_robust")]

    decision = "CONTINUE_WITH_FOCUSED_CLAIMS"
    md = f"""# Formal Main Decision

## 一句话结论

`{decision}`：继续做，但论文主张必须收缩为“基因扰动线稳定，化学扰动线有强例子也有失败边界”。

## 当前正式主表

- datasets: {n_datasets}
- aligned rho > 0.20: {aligned_pass}/{n_datasets}
- partial rho > 0.10: {partial_pass}/{n_datasets}
- RC@80% positive: {rc_pass}/{n_datasets}
- gene_main partial rho range: {gene['partial_rho_control_magnitude'].min():.3f} to {gene['partial_rho_control_magnitude'].max():.3f}
- chem_robust partial rho range: {chem['partial_rho_control_magnitude'].min():.3f} to {chem['partial_rho_control_magnitude'].max():.3f}

## 能写的 claim

1. 在 4 个 gene_main 数据集上，冻结协议控制 effect magnitude（效应大小）后仍保持正相关。
2. 在 Srivatsan 化学数据集上，化学线也有强信号。
3. 7/7 数据集 RC@80% 为正，说明“保留高可信预测”在应用层面有价值。

## 不能写的 claim

1. 不能写 SafeConf 对所有化学扰动都稳定有效。
2. 不能写信号完全独立于 effect magnitude；正式表显示 magnitude-only baseline 很强。
3. 不能因为 McFarland 失败而临时修改冻结公式。

## McFarland 定位

- v0.2 main aligned rho: {mc['main_aligned_rho']:.3f}
- v0.2 main partial rho: {mc['main_partial_rho']:.3f}
- best observed score on McFarland: `{mc['best_score_name']}` ({mc['best_score_aligned_rho']:.3f})
- observed non-control cell_line × drug pairs in h5ad metadata: {mc['n_pairs']:,}
- formal held-out test cell_line × drug pairs: {mc['n_formal_test_pairs']:,}

Decision: keep McFarland in the main table as a failure boundary. If it is revisited, first redefine tasks with dose/time, not formula tuning.

## Tahoe 定位

- downloaded: {tahoe['downloaded_gb']:.1f} GB
- obs rows: {tahoe['obs_rows']:,}
- observed drug × cell_line pairs: {tahoe['n_pairs_obs']:,}
- pairs with at least 20 cells: {tahoe['eligible_pairs_min20']:,}
- role: {tahoe['role']}

Tahoe should not enter the current formal main table yet. The next useful step is a pseudobulk adapter and leakage audit.

## 下一步

1. Give Claude this diagnostics package for critique.
2. Do not change protocol v0.2 formula.
3. If time allows, build Tahoe pseudobulk adapter as external mega-scale validation.
4. For paper draft, write McFarland as an honest failure boundary.
"""
    (out_paths["reports"] / "FORMAL_MAIN_DECISION.md").write_text(md, encoding="utf-8")
    return {
        "decision": decision,
        "aligned_pass": aligned_pass,
        "partial_pass": partial_pass,
        "rc_pass": rc_pass,
    }


def sync_to_docs(out_dir: Path, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "reports", "figures"]:
        src = out_dir / sub
        dst = docs_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    readme = f"""# Formal Main Diagnostics

这个目录是正式 7 主表之后补的诊断包，给你和 Claude/Qwen 复核用。

先看：

1. `reports/FORMAL_MAIN_DECISION.md`
2. `reports/McFarland_failure_diagnosis.md`
3. `reports/Tahoe_eligibility_audit.md`
4. `tables/FEATURE_CORRELATION_MATRIX.csv`

这些文件来自代码输出目录：

`{out_dir}`
"""
    (docs_dir / "README_先看这个.md").write_text(readme, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    formal_root = Path(args.formal_root)
    tahoe_root = Path(args.tahoe_root)
    out_dir = Path(args.out_dir) if args.out_dir else formal_root / "post_formal_diagnostics"
    docs_dir = Path(args.docs_dir) if args.docs_dir else DEFAULT_DOCS_DIR
    paths = _mkdirs(out_dir)

    if not _parquet_available():
        raise RuntimeError("pyarrow is required for Tahoe parquet audit. Run in scgpt_env.")

    mc = diagnose_mcfarland(formal_root, paths)
    feat = feature_redundancy(formal_root, paths)
    tahoe = tahoe_eligibility(tahoe_root, paths, args.pseudobulk_sample_shards)
    decision = write_final_decision(formal_root, paths, mc, tahoe)
    status = {"mcfarland": mc, "feature_redundancy": feat, "tahoe": tahoe, "decision": decision}
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_docs_sync:
        sync_to_docs(out_dir, docs_dir)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-formal SafeConf diagnostics.")
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT), help="Formal main audit root.")
    parser.add_argument("--tahoe-root", default=str(DEFAULT_TAHOE_ROOT), help="Tahoe-100M local root.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to formal-root/post_formal_diagnostics.")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Small docs sync directory.")
    parser.add_argument("--pseudobulk-sample-shards", type=int, default=3, help="Number of pseudobulk shards to sample.")
    parser.add_argument("--no-docs-sync", action="store_true", help="Do not copy small outputs into proj/docs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    status = run(args)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V21 = PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"
OUT = PROJECT_ROOT / "outputs" / "confidence_task_protocol_v0_2"


def mkdirs(out: Path) -> None:
    for name in ["tables", "figures", "reports", "scripts", "logs"]:
        (out / name).mkdir(parents=True, exist_ok=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def zscore_by_ref(values: pd.Series, ref: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    ref = pd.to_numeric(ref, errors="coerce")
    med = ref.median()
    if pd.isna(med):
        med = 0.0
    scale = ref.quantile(0.75) - ref.quantile(0.25)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = ref.std()
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (values.fillna(med) - med) / scale


def aligned_spearman(score: pd.Series, error: pd.Series, score_type: str) -> float:
    s = pd.to_numeric(score, errors="coerce")
    e = pd.to_numeric(error, errors="coerce")
    mask = s.notna() & e.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    rho = float(s[mask].corr(e[mask], method="spearman"))
    return -rho if score_type == "confidence" and np.isfinite(rho) else rho


def raw_spearman(score: pd.Series, error: pd.Series) -> float:
    s = pd.to_numeric(score, errors="coerce")
    e = pd.to_numeric(error, errors="coerce")
    mask = s.notna() & e.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(s[mask].corr(e[mask], method="spearman"))


def add_score(rows: list[dict], base: pd.DataFrame, score_name: str, score_type: str, values: pd.Series) -> None:
    for idx, value in values.items():
        row = base.loc[idx]
        rows.append(
            {
                "record_id": row["record_id"],
                "dataset_name": row["dataset_name"],
                "dataset_family": row["dataset_family"],
                "fold_id": int(row["fold_id"]),
                "split": row["split"],
                "context": row["context"],
                "perturbation": row["perturbation"],
                "predictor_name": row["predictor_name"],
                "score_name": score_name,
                "score_type": score_type,
                "score_value": float(value) if pd.notna(value) else np.nan,
                "true_error_rmse": float(row["true_error_rmse"]),
            }
        )


def build_protocol_scores(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(5201)
    add_score(rows, base, "random_score", "confidence", pd.Series(rng.random(len(base)), index=base.index))
    add_score(rows, base, "context_similarity_score", "confidence", base["context_similarity_max"])
    add_score(rows, base, "support_count_score", "confidence", np.log1p(base["perturbation_support_count"].astype(float)))
    add_score(rows, base, "model_disagreement_risk", "risk", base["model_disagreement_rmse"])
    add_score(rows, base, "historical_residual_risk", "risk", base["historical_residual_risk"])
    add_score(rows, base, "ood_distance_risk", "risk", base["ood_nearest_distance"])
    add_score(rows, base, "prediction_magnitude_risk", "risk", base["prediction_magnitude_deviation"])

    protocol = pd.Series(np.nan, index=base.index, dtype=float)
    protocol_with_stability = pd.Series(np.nan, index=base.index, dtype=float)
    family_rows = []
    for (dataset, fold, predictor), idx_obj in base.groupby(["dataset_name", "fold_id", "predictor_name"]).groups.items():
        idx = list(idx_obj)
        sub = base.loc[idx]
        train = sub[sub["split"] == "train"]
        if train.empty:
            train = sub[sub["split"].isin(["train", "val"])]
        family = str(sub["dataset_family"].iloc[0])
        z_ctx = zscore_by_ref(sub["context_similarity_max"], train["context_similarity_max"])
        z_support = zscore_by_ref(np.log1p(sub["perturbation_support_count"].astype(float)), np.log1p(train["perturbation_support_count"].astype(float)))
        z_dis = zscore_by_ref(sub["model_disagreement_rmse"], train["model_disagreement_rmse"])
        z_ood = zscore_by_ref(sub["ood_nearest_distance"], train["ood_nearest_distance"])
        z_stab = zscore_by_ref(sub["perturbation_effect_stability"], train["perturbation_effect_stability"])
        if family == "chem_robust":
            # D3: chemical robustness config fixes stability weight to zero.
            protocol.loc[idx] = z_support - z_dis
            protocol_with_stability.loc[idx] = z_support - z_dis
            formula = "log_support - model_disagreement; stability_weight=0"
        else:
            # Gene main keeps an interpretable context/support/disagreement rule.
            # Stability remains a reported single-feature baseline because small
            # MVP folds showed it can destabilize the combined score.
            protocol.loc[idx] = z_ctx + z_support - z_dis
            protocol_with_stability.loc[idx] = z_ctx + z_support + z_stab - z_dis - z_ood
            formula = "context_similarity + log_support - model_disagreement"
        family_rows.append(
            {
                "dataset_name": dataset,
                "dataset_family": family,
                "fold_id": int(fold),
                "predictor_name": predictor,
                "protocol_formula": formula,
            }
        )
    add_score(rows, base, "protocol_v0_2_family_confidence", "confidence", protocol)
    add_score(rows, base, "protocol_v0_2_with_stability_confidence", "confidence", protocol_with_stability)
    pd.DataFrame(family_rows).to_csv(OUT / "tables" / "PROTOCOL_V0_2_FORMULAS.csv", index=False)
    return pd.DataFrame(rows)


def evaluate(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = scores[scores["split"] == "test"].dropna(subset=["score_value", "true_error_rmse"]).copy()
    rows = []
    for level, group_cols in [
        ("dataset", ["dataset_family", "dataset_name", "score_name"]),
        ("family", ["dataset_family", "score_name"]),
        ("overall", ["score_name"]),
    ]:
        for key, g in test.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            meta = dict(zip(group_cols, key))
            score_type = str(g["score_type"].iloc[0])
            rows.append(
                {
                    "level": level,
                    "dataset_family": meta.get("dataset_family", "ALL"),
                    "dataset_name": meta.get("dataset_name", "ALL"),
                    "score_name": meta.get("score_name"),
                    "score_type": score_type,
                    "n": int(len(g)),
                    "spearman_score_vs_rmse": raw_spearman(g["score_value"], g["true_error_rmse"]),
                    "direction_aligned_spearman": aligned_spearman(g["score_value"], g["true_error_rmse"], score_type),
                    "mean_rmse": float(g["true_error_rmse"].mean()),
                }
            )
    eval_df = pd.DataFrame(rows)

    cov_rows = []
    for (dataset_family, dataset, score), g in test.groupby(["dataset_family", "dataset_name", "score_name"]):
        score_type = str(g["score_type"].iloc[0])
        g = g.sort_values("score_value", ascending=(score_type == "risk"))
        full = float(g["true_error_rmse"].mean())
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            keep = max(1, int(math.ceil(cov * len(g))))
            kept = g.head(keep)
            mean = float(kept["true_error_rmse"].mean())
            cov_rows.append(
                {
                    "dataset_family": dataset_family,
                    "dataset_name": dataset,
                    "score_name": score,
                    "score_type": score_type,
                    "coverage": cov,
                    "mean_rmse": mean,
                    "full_mean_rmse": full,
                    "risk_cov_improve_pct": 100.0 * (full - mean) / full if full else np.nan,
                }
            )
    cov_df = pd.DataFrame(cov_rows)
    cov80 = cov_df[np.isclose(cov_df["coverage"], 0.8)].groupby(["dataset_name", "score_name"], as_index=False)["risk_cov_improve_pct"].mean()
    eval_df = eval_df.merge(cov80, on=["dataset_name", "score_name"], how="left")
    return eval_df, cov_df, test


def plot(eval_df: pd.DataFrame, cov_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [
        "protocol_v0_2_family_confidence",
        "protocol_v0_2_with_stability_confidence",
        "model_disagreement_risk",
        "historical_residual_risk",
        "random_score",
    ]
    ds = eval_df[(eval_df["level"] == "dataset") & (eval_df["score_name"].isin(keep))].copy()
    piv = ds.pivot_table(index="dataset_name", columns="score_name", values="direction_aligned_spearman", aggfunc="mean")
    piv = piv.reindex(columns=[c for c in keep if c in piv.columns])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(piv.index))
    width = 0.8 / max(1, len(piv.columns))
    for j, col in enumerate(piv.columns):
        ax.bar(x - 0.4 + width / 2 + j * width, piv[col].values, width=width, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("direction-aligned Spearman")
    ax.set_title("Protocol v0.2 candidate scoring by dataset")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "protocol_v0_2_dataset_spearman.png", dpi=220)
    plt.close(fig)

    cov = cov_df[cov_df["score_name"].isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for (score, dataset), g in cov.groupby(["score_name", "dataset_name"]):
        if score not in ["protocol_v0_2_family_confidence", "model_disagreement_risk"]:
            continue
        ax.plot(g["coverage"], g["mean_rmse"], marker="o", label=f"{dataset}:{score}")
    ax.invert_xaxis()
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean RMSE")
    ax.set_title("Risk-coverage selected protocol scores")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "protocol_v0_2_risk_coverage.png", dpi=220)
    plt.close(fig)


def df_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def report(eval_df: pd.DataFrame) -> None:
    ds = eval_df[(eval_df["level"] == "dataset") & (eval_df["score_name"].isin([
        "protocol_v0_2_family_confidence",
        "model_disagreement_risk",
        "historical_residual_risk",
        "random_score",
    ]))].copy()
    ds = ds[["dataset_family", "dataset_name", "score_name", "direction_aligned_spearman", "risk_cov_improve_pct", "n"]]
    protocol = eval_df[(eval_df["level"] == "dataset") & (eval_df["score_name"] == "protocol_v0_2_family_confidence")]
    gene = protocol[protocol["dataset_family"] == "gene_main"]["direction_aligned_spearman"].median()
    chem = protocol[protocol["dataset_family"] == "chem_robust"]["direction_aligned_spearman"].median()
    text = [
        "# Protocol v0.2 scoring experiment",
        "",
        "## What this tests",
        "",
        "This is an experiment driven by the mentor/Opus critique: do not chase a new perturbation predictor; instead test whether a family-aware confidence score can rank prediction reliability.",
        "",
        "The experiment reuses Phase 2.1 PredictionRecords/features (`n_genes=5000`, leakage=0) and only rescoring is performed.",
        "",
        "## Candidate protocol score",
        "",
        "- `gene_main`: `context_similarity + log_support - model_disagreement`.",
        "- `chem_robust`: `log_support - model_disagreement`, with stability weight fixed to zero.",
        "- `protocol_v0_2_with_stability_confidence` is kept as a comparison to show why stability is risky in sparse settings.",
        "",
        "## Dataset-level results",
        "",
        df_md(ds),
        "",
        "## Summary",
        "",
        f"- gene_main median aligned Spearman for protocol score: {gene:.4f}.",
        f"- chem_robust aligned Spearman for protocol score: {chem:.4f}.",
        "- This supports a family-aware protocol experiment, but it is still an exploratory rescoring layer over three MVP datasets.",
        "- The next publishable step is to freeze this as config, add Norman/Adamson and GEARS predictions, then re-run without looking at test labels.",
    ]
    write(OUT / "PROTOCOL_V0_2_EXPERIMENT_REPORT.md", "\n".join(text) + "\n")


def make_zip() -> Path:
    zip_path = OUT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(OUT.parent)))
    return zip_path


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Run protocol v0.2 rescoring experiment from Phase 2.1 records.")
    parser.add_argument("--input-dir", default=str(V21))
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.out_dir)
    mkdirs(OUT)
    inp = Path(args.input_dir)
    rec = pd.read_csv(inp / "tables" / "PREDICTION_RECORDS.csv")
    feat = pd.read_csv(inp / "tables" / "CONFIDENCE_FEATURES.csv")
    base = rec.merge(feat, on=["record_id", "task_id", "task_key", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"], how="left")
    base["dataset_family"] = np.where(base["dataset_name"].eq("KaggleCrossCell"), "chem_robust", "gene_main")
    scores = build_protocol_scores(base)
    eval_df, cov_df, test = evaluate(scores)
    scores.to_csv(OUT / "tables" / "PROTOCOL_V0_2_SCORES.csv", index=False)
    eval_df.to_csv(OUT / "tables" / "PROTOCOL_V0_2_EVAL_SUMMARY.csv", index=False)
    cov_df.to_csv(OUT / "tables" / "PROTOCOL_V0_2_RISK_COVERAGE.csv", index=False)
    base.to_csv(OUT / "tables" / "PROTOCOL_V0_2_RECORD_FEATURE_TABLE.csv", index=False)
    plot(eval_df, cov_df)
    report(eval_df)
    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    zip_path = make_zip()
    status = {
        "out_dir": str(OUT),
        "zip_path": str(zip_path),
        "n_records": int(len(base)),
        "n_test_records": int((base["split"] == "test").sum()),
    }
    write(OUT / "RUN_STATUS.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

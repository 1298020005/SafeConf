#!/usr/bin/env python3
"""Re-express frozen Cartesian and cross-dataset predictions as family certificates.

E187 is retrospective. It does not train a predictor, select a task, change a
split, or create a conformal guarantee. The deterministic lower certificate is
computed from predictions only; target truth is read solely to evaluate the
certificate's tightness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "实验结果"
OUT = RESULTS / "E187_advisor_difficulty_certificate_20260726"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
SEED = 20260726
TOL = 1e-10

CARTESIAN_INPUTS = (
    (
        "Frangieh",
        RESULTS
        / "E98_frangieh_gene_cartesian_predictions_20260713"
        / "tables"
        / "E98_TASK_RISK_TABLE.csv",
    ),
    (
        None,
        RESULTS
        / "E100_gene_external_cartesian_predictions_20260713"
        / "tables"
        / "E100_TASK_RISK_TABLE.csv",
    ),
    (
        None,
        RESULTS
        / "E103_cui_cartesian_predictions_20260713"
        / "tables"
        / "E103_TASK_RISK_TABLE.csv",
    ),
)

CROSS_INPUTS = (
    (
        "sciPlex3_to_OpenProblems",
        RESULTS
        / "E87_sciplex_to_openproblems_cpa_20260712"
        / "tables"
        / "E87_TASK_SCORES.csv",
        "error_cpa_rmse",
        "error_ridge_rmse",
    ),
    (
        "sciPlex3_to_sciPlex4",
        RESULTS
        / "E89_sciplex3_to_sciplex4_cpa_20260712"
        / "tables"
        / "E89_TASK_SCORES.csv",
        "error_cpa_rmse",
        "error_interpolation_rmse",
    ),
)

SETTING_LABELS = {
    "random_missing_pair": "Random pair",
    "context_unseen_row": "Unseen context",
    "perturbation_unseen_column": "Unseen perturbation",
    "context_and_perturbation_unseen": "Double unseen",
}

COLORS = {
    "Random pair": "#3C5488",
    "Unseen context": "#4DBBD5",
    "Unseen perturbation": "#00A087",
    "Double unseen": "#E64B35",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    svg_path = FIGURES / f"{stem}.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def add_family_metrics(frame: pd.DataFrame, error_a: str, error_b: str) -> pd.DataFrame:
    result = frame.copy()
    e1 = result[error_a].to_numpy(dtype=float)
    e2 = result[error_b].to_numpy(dtype=float)
    distance = result["model_disagreement_rmse"].to_numpy(dtype=float)
    lower = distance / 2.0
    family_rms = np.sqrt((e1**2 + e2**2) / 2.0)
    worst_error = np.maximum(e1, e2)
    centroid_sq = np.maximum(family_rms**2 - lower**2, 0.0)

    result["family_rms_lower"] = lower
    result["family_worst_lower"] = lower
    result["observed_family_rms"] = family_rms
    result["observed_family_worst"] = worst_error
    result["observed_centroid_error"] = np.sqrt(centroid_sq)
    result["family_rms_tightness"] = np.divide(
        lower,
        family_rms,
        out=np.zeros_like(lower),
        where=family_rms > 0,
    )
    result["family_worst_tightness"] = np.divide(
        lower,
        worst_error,
        out=np.zeros_like(lower),
        where=worst_error > 0,
    )
    result["family_rms_violation"] = lower > family_rms + TOL
    result["family_worst_violation"] = lower > worst_error + TOL
    result["certificate_uses_target_truth"] = False
    result["truth_used_for_evaluation_only"] = True
    return result


def load_cartesian() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fallback_dataset, path in CARTESIAN_INPUTS:
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            frame["dataset"] = fallback_dataset
        frame = frame.loc[frame["split"].eq("test")].copy()
        frame = frame.rename(
            columns={
                "risk_model_disagreement": "model_disagreement_rmse",
                "error_source_knn_rmse": "error_predictor_a_rmse",
                "error_context_ridge_rmse": "error_predictor_b_rmse",
            }
        )
        frame = add_family_metrics(
            frame, "error_predictor_a_rmse", "error_predictor_b_rmse"
        )
        frame["setting_label"] = frame["setting"].map(SETTING_LABELS)
        frame["analysis_role"] = "within_study_difficulty_ladder"
        frame["task_instance_id"] = (
            frame["dataset"].astype(str)
            + "::"
            + frame["fold_id"].astype(str)
            + "::train"
            + (100 * frame["train_fraction"]).round().astype(int).astype(str)
            + "::"
            + frame["setting"].astype(str)
            + "::"
            + frame["context"].astype(str)
            + "::"
            + frame["perturbation"].astype(str)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cartesian_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        tasks.groupby(
            ["dataset", "train_fraction", "setting", "setting_label"], dropna=False
        )
        .agg(
            n_task_instances=("task_instance_id", "size"),
            n_perturbations=("perturbation", "nunique"),
            n_contexts=("context", "nunique"),
            family_rms_violations=("family_rms_violation", "sum"),
            family_worst_violations=("family_worst_violation", "sum"),
            median_family_rms=("observed_family_rms", "median"),
            median_family_rms_lower=("family_rms_lower", "median"),
            median_family_rms_tightness=("family_rms_tightness", "median"),
            q25_family_rms_tightness=(
                "family_rms_tightness",
                lambda x: x.quantile(0.25),
            ),
            q75_family_rms_tightness=(
                "family_rms_tightness",
                lambda x: x.quantile(0.75),
            ),
        )
        .reset_index()
    )
    return grouped


def cluster_bootstrap_macro(tasks: pd.DataFrame, repeats: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str]] = []
    for fraction in sorted(tasks["train_fraction"].unique()):
        for setting, label in SETTING_LABELS.items():
            subset = tasks.loc[
                tasks["train_fraction"].eq(fraction) & tasks["setting"].eq(setting)
            ]
            point = (
                subset.groupby("dataset")["family_rms_tightness"].median().mean()
            )
            draws = np.empty(repeats, dtype=float)
            by_dataset = []
            for _, dataset_frame in subset.groupby("dataset"):
                clusters = {
                    key: cluster.copy()
                    for key, cluster in dataset_frame.groupby("perturbation")
                }
                by_dataset.append(clusters)
            for index in range(repeats):
                dataset_medians = []
                for clusters in by_dataset:
                    keys = np.array(list(clusters), dtype=object)
                    selected = rng.choice(keys, size=len(keys), replace=True)
                    values = np.concatenate(
                        [
                            clusters[key]["family_rms_tightness"].to_numpy()
                            for key in selected
                        ]
                    )
                    dataset_medians.append(float(np.median(values)))
                draws[index] = float(np.mean(dataset_medians))
            rows.append(
                {
                    "train_fraction": fraction,
                    "setting": setting,
                    "setting_label": label,
                    "macro_median_tightness": point,
                    "cluster_bootstrap_ci95_low": np.quantile(draws, 0.025),
                    "cluster_bootstrap_ci95_high": np.quantile(draws, 0.975),
                    "bootstrap_unit": "perturbation within dataset",
                    "bootstrap_repeats": repeats,
                }
            )
    return pd.DataFrame(rows)


def load_cross_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for transfer, path, error_a, error_b in CROSS_INPUTS:
        frame = pd.read_csv(path)
        frame = add_family_metrics(frame, error_a, error_b)
        frame["transfer"] = transfer
        frame["analysis_role"] = "direct_cross_dataset_transfer"
        frame["task_instance_id"] = transfer + "::" + frame["task_key"].astype(str)
        tasks.append(frame)

        better_a = int((frame[error_a] < frame["zero_effect_rmse"]).sum())
        better_b = int((frame[error_b] < frame["zero_effect_rmse"]).sum())
        summaries.append(
            {
                "transfer": transfer,
                "n_tasks": len(frame),
                "family_rms_violations": int(frame["family_rms_violation"].sum()),
                "family_worst_violations": int(
                    frame["family_worst_violation"].sum()
                ),
                "median_family_rms": frame["observed_family_rms"].median(),
                "median_family_rms_lower": frame["family_rms_lower"].median(),
                "median_family_rms_tightness": frame[
                    "family_rms_tightness"
                ].median(),
                "q25_family_rms_tightness": frame[
                    "family_rms_tightness"
                ].quantile(0.25),
                "q75_family_rms_tightness": frame[
                    "family_rms_tightness"
                ].quantile(0.75),
                "predictor_a_better_than_zero_n": better_a,
                "predictor_b_better_than_zero_n": better_b,
                "predictor_a_better_than_zero_fraction": better_a / len(frame),
                "predictor_b_better_than_zero_fraction": better_b / len(frame),
                "target_calibration_available": False,
                "cross_dataset_upper_claim": False,
            }
        )
    return pd.concat(tasks, ignore_index=True), pd.DataFrame(summaries)


def draw_matrix(ax: plt.Axes, kind: str, title: str) -> None:
    nrow, ncol = 5, 6
    grid = np.full((nrow, ncol), 0.88)
    if kind == "pair":
        grid[1, 4] = 0.15
        grid[3, 2] = 0.15
    elif kind == "row":
        grid[3, :] = 0.15
    elif kind == "column":
        grid[:, 4] = 0.15
    elif kind == "double":
        grid[3, :] = 0.15
        grid[:, 4] = 0.15
        grid[3, 4] = 0.0
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=4, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(
    cartesian_tasks: pd.DataFrame,
    macro: pd.DataFrame,
    cross_tasks: pd.DataFrame,
) -> None:
    configure_plotting()
    fig = plt.figure(figsize=(7.2, 6.2))
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.62, 1.45),
        width_ratios=(1.45, 1.0),
        hspace=0.40,
        wspace=0.42,
    )

    top = outer[0, :].subgridspec(1, 4, wspace=0.32)
    for index, (kind, title) in enumerate(
        (
            ("pair", "Random pair"),
            ("row", "Unseen context"),
            ("column", "Unseen perturbation"),
            ("double", "Double unseen"),
        )
    ):
        draw_matrix(fig.add_subplot(top[0, index]), kind, title)

    ax_b = fig.add_subplot(outer[1, 0])
    for label in SETTING_LABELS.values():
        part = macro.loc[macro["setting_label"].eq(label)].sort_values(
            "train_fraction"
        )
        x = 100 * part["train_fraction"].to_numpy()
        y = part["macro_median_tightness"].to_numpy()
        low = part["cluster_bootstrap_ci95_low"].to_numpy()
        high = part["cluster_bootstrap_ci95_high"].to_numpy()
        ax_b.plot(
            x,
            y,
            marker="o",
            linewidth=1.6,
            markersize=4,
            color=COLORS[label],
            label=label,
        )
        ax_b.fill_between(x, low, high, color=COLORS[label], alpha=0.12)
    ax_b.set_xlabel("Observed training submatrix (%)")
    ax_b.set_ylabel("Median certified fraction of family RMS error")
    ax_b.set_xticks([25, 50, 75, 100])
    ax_b.set_ylim(0, 0.48)
    ax_b.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax_b.legend(frameon=False, ncol=1, loc="upper left")
    ax_b.spines[["top", "right"]].set_visible(False)

    right = outer[1, 1].subgridspec(2, 1, hspace=0.56)
    ax_c = fig.add_subplot(right[0, 0])
    current = cartesian_tasks.loc[cartesian_tasks["train_fraction"].eq(1.0)]
    labels = list(SETTING_LABELS.values())
    data = [
        current.loc[current["setting_label"].eq(label), "family_rms_tightness"]
        for label in labels
    ]
    box = ax_c.boxplot(
        data,
        positions=np.arange(len(labels)),
        widths=0.58,
        vert=False,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.62)
    ax_c.set_yticks(
        np.arange(len(labels)),
        ["Random pair", "Unseen context", "Unseen perturbation", "Double unseen"],
    )
    ax_c.set_xlabel("Task-level lower-bound tightness")
    ax_c.set_xlim(0, 0.78)
    ax_c.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax_c.spines[["top", "right"]].set_visible(False)

    ax_d = fig.add_subplot(right[1, 0])
    transfer_order = ["sciPlex3_to_OpenProblems", "sciPlex3_to_sciPlex4"]
    transfer_labels = ["OpenProblems", "sciPlex4"]
    values = [
        cross_tasks.loc[
            cross_tasks["transfer"].eq(transfer), "family_rms_tightness"
        ]
        for transfer in transfer_order
    ]
    transfer_box = ax_d.boxplot(
        values,
        widths=0.55,
        vert=False,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#8491B4", "alpha": 0.65, "linewidth": 0.7},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    for patch, color in zip(transfer_box["boxes"], ("#8491B4", "#91D1C2")):
        patch.set_facecolor(color)
    ax_d.set_yticks([1, 2], transfer_labels)
    ax_d.set_xlim(0, 0.9)
    ax_d.set_xlabel("Lower-bound tightness")
    ax_d.set_title("Direct transfer from sciPlex3", loc="left")
    ax_d.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.text(0.01, 0.98, "a", fontsize=11, fontweight="bold", va="top")
    fig.text(0.01, 0.60, "b", fontsize=11, fontweight="bold", va="top")
    fig.text(0.60, 0.60, "c", fontsize=11, fontweight="bold", va="top")
    fig.text(0.60, 0.31, "d", fontsize=11, fontweight="bold", va="top")
    save_figure(fig, "Figure_E187_difficulty_ladder")


def write_report(
    cartesian_tasks: pd.DataFrame,
    cartesian_table: pd.DataFrame,
    macro: pd.DataFrame,
    cross_tasks: pd.DataFrame,
    cross_summary: pd.DataFrame,
) -> None:
    full = cartesian_table.loc[cartesian_table["train_fraction"].eq(1.0)]
    setting_lines = []
    for label in SETTING_LABELS.values():
        row = macro.loc[
            macro["train_fraction"].eq(1.0)
            & macro["setting_label"].eq(label)
        ].iloc[0]
        setting_lines.append(
            f"- {label}: macro median tightness "
            f"{row.macro_median_tightness:.3f} "
            f"(95% cluster bootstrap interval "
            f"{row.cluster_bootstrap_ci95_low:.3f}–"
            f"{row.cluster_bootstrap_ci95_high:.3f})."
        )
    cross_lines = []
    for row in cross_summary.itertuples(index=False):
        cross_lines.append(
            f"- {row.transfer}: {row.n_tasks} tasks, zero lower-bound "
            f"violations, median tightness {row.median_family_rms_tightness:.3f}; "
            f"no target calibration was available, so no cross-dataset upper "
            f"coverage claim was made."
        )

    report = f"""# E187 | Advisor-defined difficulty ladder certificate audit

## Status

This is a retrospective re-analysis of prediction records frozen in E98, E100,
E103, E87, and E89. It does not alter any split, predictor, target, or revealed
truth. The lower certificate uses prediction vectors only; truth is used solely
to evaluate tightness and violations.

## Within-study Cartesian settings

The audit contains {len(cartesian_tasks):,} task instances from four data sets,
four training-submatrix fractions, and four test settings. The two-member family
lower bound had {int(cartesian_tasks.family_rms_violation.sum())} family-RMS
violations and {int(cartesian_tasks.family_worst_violation.sum())} worst-member
violations.

At the 100% training-submatrix level:

{chr(10).join(setting_lines)}

The double-unseen setting was the least informative on average. This is a
tightness result, not a validity failure: the deterministic lower inequality
remained exact in every setting.

## Direct cross-dataset transfer

{chr(10).join(cross_lines)}

The high cross-dataset tightness arose because the two transferred predictors
often failed by different amounts. It certifies that at least one family member
has substantial error; it does not identify the failed member and does not make
small disagreement safe.

## Relation to the advisor's questions

1. The scored error object is family RMS and family worst-member RMSE, not an
   unspecified model error.
2. Prediction magnitude and model distance are calculated from prediction
   vectors. Target perturbed expression is absent from the score.
3. Random-pair, unseen-context, unseen-perturbation, double-unseen, and
   25%/50%/75%/100% training-submatrix settings are all represented.
4. Direct dataset transfer is retained as a boundary analysis. Without target
   calibration, only the deterministic lower certificate is claimed.

## Interpretation limits

- The Cartesian genetic and cytokine experiments use two inductive reference
  predictors, not the five-seed scGPT and GEARS family used in the main
  confirmation studies.
- The same biological task may recur across training fractions and folds.
  Counts are task instances, not independent experiments.
- Bootstrap intervals resample perturbation identities within each data set and
  average data-set medians; they do not treat folds as independent studies.
- No conformal upper bound is transported directly across data sets.

## Reproducible outputs

- `tables/E187_CARTESIAN_TASK_CERTIFICATES.csv`
- `tables/E187_CARTESIAN_SETTING_SUMMARY.csv`
- `tables/E187_MACRO_BOOTSTRAP.csv`
- `tables/E187_CROSS_DATASET_TASK_CERTIFICATES.csv`
- `tables/E187_CROSS_DATASET_SUMMARY.csv`
- `tables/INPUT_HASHES.csv`
- `figures/Figure_E187_difficulty_ladder.*`
"""
    (REPORTS / "E187_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E187\n\n先看 `reports/E187_REPORT.md`。本实验是冻结结果的回顾性证书审计。\n",
        encoding="utf-8",
    )


def main() -> None:
    for directory in (TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    cartesian_tasks = load_cartesian()
    cartesian_table = cartesian_summary(cartesian_tasks)
    macro = cluster_bootstrap_macro(cartesian_tasks)
    cross_tasks, cross_summary = load_cross_dataset()

    required = [
        int(cartesian_tasks["family_rms_violation"].sum()) == 0,
        int(cartesian_tasks["family_worst_violation"].sum()) == 0,
        int(cross_tasks["family_rms_violation"].sum()) == 0,
        int(cross_tasks["family_worst_violation"].sum()) == 0,
        cartesian_tasks["setting_label"].notna().all(),
        set(cartesian_tasks["train_fraction"].unique()) == {0.25, 0.5, 0.75, 1.0},
    ]
    if not all(required):
        raise RuntimeError("E187 integrity gate failed")

    cartesian_tasks.to_csv(
        TABLES / "E187_CARTESIAN_TASK_CERTIFICATES.csv", index=False
    )
    cartesian_table.to_csv(
        TABLES / "E187_CARTESIAN_SETTING_SUMMARY.csv", index=False
    )
    macro.to_csv(TABLES / "E187_MACRO_BOOTSTRAP.csv", index=False)
    cross_tasks.to_csv(
        TABLES / "E187_CROSS_DATASET_TASK_CERTIFICATES.csv", index=False
    )
    cross_summary.to_csv(TABLES / "E187_CROSS_DATASET_SUMMARY.csv", index=False)

    input_paths = [path for _, path in CARTESIAN_INPUTS] + [
        path for _, path, _, _ in CROSS_INPUTS
    ]
    pd.DataFrame(
        [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths
        ]
    ).to_csv(TABLES / "INPUT_HASHES.csv", index=False)

    make_figure(cartesian_tasks, macro, cross_tasks)
    write_report(
        cartesian_tasks,
        cartesian_table,
        macro,
        cross_tasks,
        cross_summary,
    )

    status = {
        "experiment": "E187",
        "status": "PASS",
        "retrospective": True,
        "cartesian_task_instances": int(len(cartesian_tasks)),
        "cross_dataset_tasks": int(len(cross_tasks)),
        "total_task_instances": int(len(cartesian_tasks) + len(cross_tasks)),
        "family_rms_violations": int(
            cartesian_tasks["family_rms_violation"].sum()
            + cross_tasks["family_rms_violation"].sum()
        ),
        "family_worst_violations": int(
            cartesian_tasks["family_worst_violation"].sum()
            + cross_tasks["family_worst_violation"].sum()
        ),
        "seed": SEED,
        "truth_use": "evaluation_only",
        "cross_dataset_upper_claim": False,
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

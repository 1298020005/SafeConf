from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_GEARS_SUMMARY = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_codex_cout_old/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/reports/FORMAL_GEARS_FINAL_SUMMARY.csv"
)
DEFAULT_GEARS_STATUS = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_codex_cout_old/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/reports/FORMAL_GEARS_FINAL_STATUS_ALL_SEEDS.csv"
)


def write_gears_supplement(
    out_dir: Path,
    summary_path: Path = DEFAULT_GEARS_SUMMARY,
    status_path: Path = DEFAULT_GEARS_STATUS,
) -> pd.DataFrame:
    """Write GEARS formal-baseline supplement files.

    These are dataset-level GEARS metrics, not per-prediction confidence records.
    The report says that explicitly so they cannot be mistaken for the Phase 3
    cross-context confidence benchmark.
    """
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        df = pd.DataFrame(
            [
                {
                    "status": "missing",
                    "summary_path": str(summary_path),
                    "note": "GEARS formal summary not found; no GEARS supplement generated.",
                }
            ]
        )
        df.to_csv(tables / "GEARS_SUPPLEMENT_TABLE.csv", index=False)
        (reports / "GEARS_SUPPLEMENT_REPORT.md").write_text(
            "# GEARS Supplement\n\nGEARS formal summary was not found. No per-prediction confidence claim is made.\n",
            encoding="utf-8",
        )
        return df

    df = pd.read_csv(summary_path)
    df["source_summary"] = str(summary_path)
    df["source_status"] = str(status_path) if status_path.exists() else ""
    df["confidence_record_available"] = False
    df["role"] = "formal_gears_baseline_supplement_not_cross_context_confidence"
    df.to_csv(tables / "GEARS_SUPPLEMENT_TABLE.csv", index=False)

    report = [
        "# GEARS Supplement",
        "",
        "This table reuses the existing formal GEARS baseline summary.",
        "It is **not** a cross-context confidence-scoring result because no per-prediction GEARS confidence records are available in this phase.",
        "",
        "## Source files",
        "",
        f"- Summary: `{summary_path}`",
        f"- Status: `{status_path}` exists = `{status_path.exists()}`",
        "",
        "## Dataset-level formal GEARS metrics",
        "",
        "```",
        df.to_string(index=False),
        "```",
        "",
        "Next step for a true GEARS confidence result: export per-perturbation or per-task GEARS predicted effects and convert them to PredictionRecord rows.",
        "",
    ]
    (reports / "GEARS_SUPPLEMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    plot_df = df[df.get("status", "ok").astype(str).eq("ok")].copy() if "status" in df.columns else df.copy()
    if {"dataset", "mean_test_mse_de", "mean_test_pearson_de"}.issubset(plot_df.columns) and not plot_df.empty:
        fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
        x = range(len(plot_df))
        ax1.bar(x, plot_df["mean_test_mse_de"], color="#4C78A8", alpha=0.82, label="mean test MSE-DE")
        ax1.set_ylabel("MSE-DE")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(plot_df["dataset"], rotation=20, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(list(x), plot_df["mean_test_pearson_de"], color="#F58518", marker="o", label="Pearson-DE")
        ax2.set_ylabel("Pearson-DE")
        ax1.set_title("Formal GEARS baseline supplement")
        fig.tight_layout()
        fig.savefig(figures / "gears_formal_baseline_summary.png", dpi=220)
        fig.savefig(figures / "gears_supplement_summary.png", dpi=220)
        plt.close(fig)
    return df
